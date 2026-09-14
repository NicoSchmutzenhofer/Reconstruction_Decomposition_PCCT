"""
recon_comparison.py -- image quality of our reconstruction vs Siemens WFBP and VMI.

Reconstructs the insert slab with each of our DEFAULT_VARIANTS to trace the noise/resolution
trade-off our pipeline can reach, then measures the vendor images of the same scan against
that curve.  own and WFBP are compared threshold by threshold; VMI is on a different HU
scale, so it is measured as its own family and kept out of the paired difference analysis.

Our volumes are reconstructed on the vendor's matrix and FOV, so NPS frequency axes align,
and their native slices are averaged to the vendor's SliceThickness.  Standalone: the
production driver and output/reconstruction/ are never touched.

Three resumable stages:
    0  probe    CPU  read Siemens geometry, auto-locate the insert slab   -> match_config.json
    1  sweep    GPU  reconstruct the variants on that slab and grid       -> sweep/<variant>/
    2  metrics  CPU  noise, NPS, TTF, NEQ, d', bias, PSNR/SSIM            -> metrics/ figures/ qc/

    python -m reconstruction.recon_comparison --stage 0 --wfbp-dir ... --vmi-dir ...
    python -m reconstruction.recon_comparison --stage 1
    python -m reconstruction.recon_comparison --stage 2
    python -m reconstruction.recon_comparison --stage all --wfbp-dir ... --vmi-dir ...
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconstruction import image_quality_metrics as iq
from reconstruction import figure_panels as fp

logger = logging.getLogger("recon_comparison")
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════
# Five reconstruction settings.  Three FBP filters define the analytic
# noise/resolution trade-off curve; the two SIRT points then test themselves
# AGAINST that curve.  If SIRT lands on it, iterative reconstruction is apodisation
# by another name and buys nothing -- a real result.  If it lands below, the gain is
# genuine and measurable.  Same question the vendor comparison asks, one figure.
#
# NOTE  filter_name is inert for SIRT: _astra_reconstruct only sets FilterType in the
# FBP branch, so SIRT's noise/resolution knob is n_iter.  SIRT converges smooth->sharp
# and decelerates, so 25 vs 100 gives a visible segment where 100 vs 200 would not.
DEFAULT_VARIANTS = [
    {"name": "fbp_ramlak",     "algorithm": "fbp",  "filter_name": "ram-lak"},
    {"name": "fbp_shepplogan", "algorithm": "fbp",  "filter_name": "shepp-logan"},
    {"name": "fbp_hann",       "algorithm": "fbp",  "filter_name": "hann"},
    {"name": "sirt_25",        "algorithm": "sirt", "n_iter": 25},
    {"name": "sirt_100",       "algorithm": "sirt", "n_iter": 100},
]

THRESHOLD_LABELS = ("A", "B", "C", "D")

# Greyscale window for every HU image panel in this module.  Fixed rather than per-panel:
# an autoscaled window rescales a noisy reconstruction into looking as clean as a smooth
# one, which is the difference the image figures exist to show.  -200..300 HU spans the
# phantom body, the inserts and the fat/soft-tissue layers without clipping the spine.
HU_WINDOW_RECON = (-200.0, 300.0)

# HU the phantom's body material is specified to read.  The QRM DEP-V5 body is CTWater,
# datasheet 0 HU over 80-140 kV, which is what makes an absolute check possible at all.
# A phantom with a different body material needs this changed, so it is named rather
# than written as a bare 0.0 at the point of use.
WATER_HU = 0.0

# Raw scan used by stage 1; placeholders, pass --data-path / --desc-path.
_DEFAULT_DATA = "/path/to/full_sinogram.raw.mat"
_DEFAULT_DESC = "/path/to/descriptor.raw.mat"


@dataclass
class CompareConfig:
    """Serialisable configuration -- a config object, never module-level constants."""
    out_root: str = str(_REPO_ROOT / "output" / "research" / "recon_comparison")
    wfbp_dir: Optional[str] = None
    vmi_dir: Optional[str] = None

    data_path: str = _DEFAULT_DATA
    desc_path: str = _DEFAULT_DESC
    geo_dir: str = str(_REPO_ROOT / "geometry")

    # Grid/slab: None = take it from the Siemens export (stage 0).
    n_pixels: Optional[int] = None
    fov_mm: Optional[float] = None
    slice_thickness_mm: Optional[float] = None
    slice_spacing_mm: Optional[float] = None
    slab_mm: Optional[tuple] = None
    slab_pad_mm: float = 2.0
    max_slab_mm: float = 40.0
    # Axial extent of the phantom itself, read off the vendor volume (e.g. in Slicer).
    # Strongly recommended for a clinical scan range: without it the search covers the
    # whole acquisition, where the table or a scan-end artefact can out-score the
    # phantom and place the slab hundreds of mm away from it.
    phantom_z_mm: Optional[tuple] = None
    use_phantom_z: bool = True      # False ignores a configured phantom_z_mm
    slab_select: str = "peak"       # "peak" (most insert-covered) | "longest"
    expect_inserts: Optional[int] = None   # warn if the detected count differs

    # Left-right mirror between our reconstruction and the vendor's, applied to the
    # OUTPUT volume only -- never by un-flipping detector channels, which would break the
    # helical handedness (invariant #2, README).  Measured on this scanner with
    # probe_dicom_origin: with the vendor frame converted to ours, the body centroid
    # matches only after x -> -x (0.30 mm residual) and the principal axis of the phantom
    # goes +3.04 deg -> 177.06 deg, i.e. negated (0.10 mm residual).  A genuine rotation
    # is excluded: it would flip the sign of the centroid's y as well, and does not.
    # Set False to reproduce the unmirrored behaviour for an A/B check.
    mirror_x: bool = True

    # Reconstruction settings held fixed across the sweep (only the swept knobs vary).
    geometry_model: str = "curved"
    z_weighting: str = "balanced"
    wavelet_ring_threshold: float = 2.0
    spike_mad_k: float = 5.0
    ipr_mad_k: float = 6.0
    patient_position: str = "HFS"

    variants: list = field(default_factory=lambda: [dict(v) for v in DEFAULT_VARIANTS])

    # Figures: share one scale across the four per-threshold panels so they can be read
    # against each other. True restores per-panel autoscaling for a single-panel close-up.
    free_axes: bool = False

    # Also write every panel of a multi-panel figure on its own, to figures/panels/, so
    # the thesis can place them at a readable size instead of a quarter of \textwidth
    # each. Cropped from the rendered figure, so the shared scale above is preserved.
    split_figures: bool = True

    # Metric parameters
    nps_patch_px: int = 64          # requested; reduced automatically if it does not fit
    insert_mad_k: float = 4.0       # lower = detect fainter inserts (more false positives)
    # Hand-drawn insert prior (Slicer .nrrd / .seg.nrrd).  Positions only need to be
    # roughly right: every edge is refined from the image, and a partial annotation is
    # fine.  Auto-discovered as <wfbp_dir>/Segmentation.nrrd when not given.
    segmentation: Optional[str] = None
    # Label value inside the segmentation that marks the UNIFORM region the noise is
    # measured in.  Strongly recommended on an anthropomorphic phantom, where the body
    # outline contains lung, bone and the table and is not a noise region at all.
    background_label: Optional[int] = None
    task_diameter_mm: float = 5.0
    task_contrast_hu: float = 50.0
    bias_smooth_mm: float = 5.0

    force: bool = False

    def to_dict(self):
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════
# Small helpers
# ═════════════════════════════════════════════════════════════════════
def _sitk():
    import SimpleITK as sitk
    return sitk


def load_segmentation(path, ref_img, low_priority_label=None):
    """
    Read a Slicer segmentation and resample it onto `ref_img`'s grid.

    Resampling is done in PHYSICAL space (nearest-neighbour), never by array index: a
    Slicer .seg.nrrd is normally cropped to the segment bounding box, so its extent and
    origin differ from the volume it was drawn on, and index-space alignment would be
    silently wrong.  This also maps the segmentation onto our own reconstruction, whose
    slab covers a different z range from the vendor series it was drawn on.

    A .seg.nrrd may store each segment as a separate binary layer in a 4-D array; those
    are merged into one label map (layer i -> label i+1).

    Returns an integer label array shaped like ref_img, or None if nothing overlaps.
    """
    sitk = _sitk()
    seg = sitk.ReadImage(str(path))
    layers = ([seg[:, :, :, i] for i in range(seg.GetSize()[3])]
              if seg.GetDimension() == 4 else [seg])

    merged, deferred = None, None
    for i, layer in enumerate(layers):
        r = sitk.Resample(layer, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor,
                          0, sitk.sitkUInt16)
        a = sitk.GetArrayFromImage(r).astype(np.int32)
        if merged is None:
            merged = np.zeros_like(a)
        if len(layers) == 1:
            merged = a                       # already a label map with distinct values
        elif low_priority_label is not None and i + 1 == int(low_priority_label):
            deferred = a > 0                 # applied last, and only where nothing else is
        else:
            merged[a > 0] = i + 1
    if deferred is not None:
        # The background segment is allowed to overlap the inserts -- painting the whole
        # module is the natural way to draw it -- but an insert must never be overwritten
        # by it, or it would go undetected and then be counted as noise.  Priority is
        # explicit rather than depending on the order the segments happen to be stored in.
        merged[(merged == 0) & deferred] = int(low_priority_label)
    if merged is None or not np.any(merged > 0):
        return None
    return merged


def _load_match(out_root):
    """Read stage 0's output, with an actionable message if it was never run."""
    p = Path(out_root) / "match_config.json"
    if not p.exists():
        raise SystemExit(
            f"{p} not found -- run stage 0 first:\n"
            f"  python -m reconstruction.recon_comparison --stage 0 "
            f"--wfbp-dir ... --vmi-dir ... --out-root {out_root}")
    return json.loads(p.read_text(encoding="utf-8"))


def _jsonable(o):
    """
    Recursively convert numpy types/arrays so json.dump accepts them.

    Keys are coerced to str: JSON has no other kind, and assuming they already were one
    made this raise on any dict keyed by int (decomp_validation's layer_of_label).  The
    "_map" filter -- which keeps whole density volumes out of the sidecars -- is applied
    only to string keys, since that is the only kind it can meaningfully match.
    """
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()
                if not (isinstance(k, str) and k.endswith("_map"))}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib unavailable -- skipping figures")
        return None


# ═════════════════════════════════════════════════════════════════════
# Stage 0 -- probe the Siemens export
# ═════════════════════════════════════════════════════════════════════
def _classify_text(text, loose=False):
    """
    Classify one string as a monoenergetic or threshold series.

    Real NAEOTOM SeriesDescriptions look like
        'MonoEnergeticPlus 70 keV'
        'ProtocolModel WFBP_T1 Qr40f(3) 0.4 (0.4) [A,1]_0'
    so the keV number is NOT adjacent to the word 'Mono' -- the product name sits in
    between.  Match 'mono' and the keV figure independently rather than assuming they
    are neighbours.  'VNC' and other spectral products match neither and are ignored.
    """
    if re.search(r"mono", text, re.IGNORECASE):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*ke?V", text, re.IGNORECASE)
        if m:
            return ("mono", float(m.group(1)))
    m = re.search(r"WFBP[_ ]*T([0-9]+)", text, re.IGNORECASE)
    if m:
        return ("threshold", int(m.group(1)))
    if loose:
        m = re.search(r"[_ /\\]T([0-9]+)[_ ]", text)
        if m:
            return ("threshold", int(m.group(1)))
    return None


def _classify_series(desc, path):
    """
    Series -> ('mono', keV) | ('threshold', n) | None.

    The SeriesDescription is tried ALONE first, and only then description+path.  A
    Siemens export puts several products in one folder -- the VMI export here also
    carries WFBP_T1/T2 series -- so a folder name containing 'Mono' must never be
    allowed to re-label a series whose own description says WFBP.

    Kept in step with decomposition/material_decomposition.py, but local, to avoid a
    reconstruction -> decomposition package dependency.
    """
    return (_classify_text(desc or "")
            or _classify_text(f"{desc or ''} {path or ''}", loose=True))


def _is_dicom(path):
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def index_dicom_series(folder, cache_dir=None):
    """
    Index DICOM series under `folder`, keeping the acquisition metadata the comparison
    needs.  SliceThickness and PixelSpacing are what make the comparison fair, and
    ConvolutionKernel records what we are comparing against.

    Reading 20-30 k headers takes minutes, and stages 0 and 2 both need the index, so
    the result is cached per folder under cache_dir and reused.  Delete the cache file
    (or pass --force) if the export itself changes.

    Returns {uid: {number, desc, kernel, thickness_mm, spacing_between_mm, pixel_mm,
                   rows, cols, recon_diameter_mm, files: [[z, path], ...]}}
    """
    cache = None
    if cache_dir:
        import hashlib
        h = hashlib.sha1(str(folder).encode("utf-8")).hexdigest()[:12]
        cache = Path(cache_dir) / f"dicom_index_{h}.json"
        if cache.exists():
            try:
                blob = json.loads(cache.read_text(encoding="utf-8"))
                if blob.get("folder") == str(folder):
                    logger.info("using cached DICOM index (%d series) %s",
                                len(blob["series"]), cache.name)
                    return blob["series"]
            except (OSError, ValueError, KeyError):
                pass

    sitk = _sitk()
    series = {}
    n_seen = 0
    for root, _dirs, names in os.walk(str(folder)):
        for name in names:
            fp = os.path.join(root, name)
            if not (name.lower().endswith((".dcm", ".ima")) or _is_dicom(fp)):
                continue
            r = sitk.ImageFileReader()
            r.SetFileName(fp)
            try:
                r.ReadImageInformation()
            except RuntimeError:
                continue

            def g(t):
                return r.GetMetaData(t).strip() if r.HasMetaDataKey(t) else ""

            uid = g("0020|000e")
            if not uid:
                continue
            try:
                z = float(g("0020|0032").split("\\")[2])
            except (ValueError, IndexError):
                z = 0.0
            if uid not in series:
                def fnum(tag, idx=None):
                    v = g(tag)
                    if not v:
                        return None
                    try:
                        return float(v.split("\\")[idx]) if idx is not None else float(v)
                    except (ValueError, IndexError):
                        return None
                series[uid] = {
                    "number": g("0020|0011"), "desc": g("0008|103e"),
                    "kernel": g("0018|1210"),
                    "thickness_mm": fnum("0018|0050"),
                    "spacing_between_mm": fnum("0018|0088"),
                    "pixel_mm": fnum("0028|0030", 0),
                    "recon_diameter_mm": fnum("0018|1100"),
                    "rows": int(float(g("0028|0010") or 0)) or None,
                    "cols": int(float(g("0028|0011") or 0)) or None,
                    "files": [],
                }
            series[uid]["files"].append([z, fp])
            n_seen += 1
            if n_seen % 2000 == 0:
                logger.info("  indexed %d DICOM headers ...", n_seen)
    for s in series.values():
        s["files"].sort(key=lambda zf: zf[0])
        if s["spacing_between_mm"] is None and len(s["files"]) > 1:
            dz = np.diff([z for z, _ in s["files"]])
            dz = dz[np.abs(dz) > 1e-6]
            s["spacing_between_mm"] = float(np.median(np.abs(dz))) if dz.size else None
    logger.info("indexed %d series / %d files under %s", len(series), n_seen, folder)
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"folder": str(folder), "series": series}),
                             encoding="utf-8")
            logger.info("cached index -> %s", cache.name)
        except OSError:
            pass
    return series


def select_family(index, kind):
    """Pick the channels of one product ('wfbp' or 'vmi') out of a series index."""
    want = "mono" if kind == "vmi" else "threshold"
    found = []
    for uid, s in index.items():
        spec = _classify_series(s.get("desc", ""), s["files"][0][1] if s["files"] else "")
        if not spec or spec[0] != want:
            continue
        label = f"{spec[1]:g}keV" if want == "mono" else f"T{spec[1]}"
        found.append({"uid": uid, "key": spec[1], "label": label, **s})
    found.sort(key=lambda d: d["key"])

    dup = [d["label"] for d in found]
    if len(set(dup)) != len(dup):
        raise ValueError(
            f"{kind}: duplicate channel labels {sorted(dup)} under one folder -- the "
            f"directory holds more than one {kind} reconstruction. Point --{kind}-dir at "
            f"the specific series folder, otherwise they are silently concatenated into "
            f"one oversized channel stack.")
    return found


def read_series_volume(entry, z_range=None):
    """Read one DICOM series as (array (Z,Y,X) float32, sitk image, z positions mm)."""
    sitk = _sitk()
    files = entry["files"]
    if z_range is not None:
        lo, hi = sorted(z_range)
        files = [f for f in files if lo <= f[0] <= hi] or files
    r = sitk.ImageSeriesReader()
    r.SetFileNames([p for _z, p in files])
    img = r.Execute()
    return (sitk.GetArrayFromImage(img).astype(np.float32), img,
            np.array([z for z, _ in files], dtype=float))


def _dicom_in_plane_origin(path):
    """
    (x, y) of ImagePositionPatient and the TableHeight, in mm, from one DICOM file.

    Siemens references the patient frame to the TABLE, not the isocentre, so for this
    scanner y is offset by exactly TableHeight from an isocentre-centred grid (measured:
    -338.010 vs -249.512, TableHeight 88.5).  Recording it here is what lets stage 1 write
    our volumes in the vendor's frame instead of assuming the two agree.
    """
    sitk = _sitk()
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()

    def tag(t):
        try:
            return r.GetMetaData(t).strip()
        except RuntimeError:
            return ""

    ipp = tag("0020|0032").split("\\")
    if len(ipp) < 2:
        return None, None, None
    th = tag("0018|1130")
    return (float(ipp[0]), float(ipp[1]),
            float(th) if th else None)


def stage0_probe(cfg: CompareConfig):
    """Read the vendor geometry and locate the insert slab -> match_config.json."""
    out = Path(cfg.out_root)
    (out / "qc").mkdir(parents=True, exist_ok=True)
    if not cfg.wfbp_dir and not cfg.vmi_dir:
        raise ValueError("stage 0 needs --wfbp-dir and/or --vmi-dir")

    families = {}
    for kind, folder in (("wfbp", cfg.wfbp_dir), ("vmi", cfg.vmi_dir)):
        if not folder:
            continue
        idx = index_dicom_series(folder, cache_dir=out)
        chans = select_family(idx, kind)
        if not chans:
            disc = [(s.get("number"), s.get("desc")) for s in idx.values()]
            raise ValueError(f"no '{kind}' series under {folder}. Discovered: {disc}")
        families[kind] = chans
        logger.info("%s: %d channels %s", kind, len(chans), [c["label"] for c in chans])

    # Geometry is taken from WFBP when available: it is the family our thresholds are
    # actually compared against 1:1, so matching its grid is what removes the confound.
    ref_kind = "wfbp" if "wfbp" in families else "vmi"
    ref = families[ref_kind][0]
    thickness = ref["thickness_mm"] or 1.0
    spacing = ref["spacing_between_mm"] or thickness
    pixel_mm = ref["pixel_mm"] or 1.0
    cols = ref["cols"] or 512
    fov = ref["recon_diameter_mm"] or (pixel_mm * cols)

    logger.info("reference geometry from %s/%s: %dx%d, pixel %.4f mm, FOV %.1f mm, "
                "thickness %.2f mm, spacing %.2f mm, kernel %r",
                ref_kind, ref["label"], ref["rows"], cols, pixel_mm, fov,
                thickness, spacing, ref["kernel"])

    origin_x, origin_y, table_height = (None, None, None)
    if ref["files"]:
        origin_x, origin_y, table_height = _dicom_in_plane_origin(ref["files"][0][1])
    if origin_x is None:
        logger.warning("no ImagePositionPatient on %s/%s -- stage 1 will assume an "
                       "isocentre-centred FOV for our volumes, which is only right if "
                       "the vendor grid is too", ref_kind, ref["label"])
    else:
        iso = -fov / 2 + pixel_mm / 2
        logger.info("vendor in-plane origin: x %+.3f  y %+.3f mm  "
                    "(isocentre-centred would be %+.3f; table height %s)",
                    origin_x, origin_y, iso,
                    f"{table_height:.1f} mm" if table_height else "not recorded")
        dx, dy = origin_x - iso, origin_y - iso
        if max(abs(dx), abs(dy)) > 0.5 * pixel_mm:
            logger.info("   -> offset from isocentre-centred: x %+.3f  y %+.3f mm; "
                        "our volumes are written on THIS origin so the two frames agree",
                        dx, dy)
    if cfg.mirror_x:
        logger.info("mirror_x on: our volumes are flipped left-right on write "
                    "(output-side only; detector channels are untouched)")
    if spacing < thickness - 1e-6:
        logger.info("  NOTE overlapping slices (spacing %.2f < thickness %.2f mm): the "
                    "sweep reproduces both, so the z-correlation matches too",
                    spacing, thickness)

    # Every family is compared against the reference geometry.  A segmentation transfers
    # across a spacing difference without trouble (it is resampled in physical space), but
    # the METRICS do not: slice thickness and pixel size both change the measured noise for
    # reasons that have nothing to do with the reconstruction, so a mismatch has to be
    # named rather than quietly folded into the result.
    logger.info("family geometry (reference = %s/%s):", ref_kind, ref["label"])
    for kind, chans in families.items():
        for c in chans:
            t_c = c["thickness_mm"] or float("nan")
            p_c = c["pixel_mm"] or float("nan")
            s_c = c["spacing_between_mm"] or float("nan")
            flags = []
            if np.isfinite(t_c) and abs(t_c - thickness) > 0.01:
                # noise ~ 1/sqrt(thickness): a thicker slice is quieter for free
                flags.append(f"thickness {t_c:.2f} vs {thickness:.2f} mm -> expect its "
                             f"noise to differ by ~{np.sqrt(thickness / t_c):.2f}x on that "
                             f"account ALONE")
            if np.isfinite(p_c) and abs(p_c - pixel_mm) > 1e-4:
                flags.append(f"pixel {p_c:.4f} vs {pixel_mm:.4f} mm -> its NPS frequency "
                             f"axis does not align with the reference; compare NPS curves "
                             f"between these two families with care")
            logger.info("   %-5s %-8s thickness %.2f  spacing %.2f  pixel %.4f  %s",
                        kind, c["label"], t_c, s_c, p_c,
                        "OK" if not flags else "MISMATCH")
            for f in flags:
                logger.warning("       %s: %s", c["label"], f)

    # Locate the insert layer on the reference volume (CPU, no GPU needed).
    # Reading is limited to the phantom when known: a Thx-Abdomen series is thousands of
    # slices, and the whole point of the restriction is that the rest of the scan range
    # (table, positioning aids, scan-end artefacts) can out-score the phantom entirely.
    phantom_z = cfg.phantom_z_mm if cfg.use_phantom_z else None
    if phantom_z:
        pad = max(cfg.slab_pad_mm, 5.0)
        read_range = (min(phantom_z) - pad, max(phantom_z) + pad)
        logger.info("restricting to the phantom: z = %.2f .. %.2f mm", *sorted(phantom_z))
    else:
        read_range = None
        logger.info("no --phantom-z given: searching the WHOLE scan range for the insert "
                    "layer. On a long clinical range the table or a scan-end artefact can "
                    "out-score the phantom -- check the QC figure carefully.")
    vol, img, zpos = read_series_volume(ref, z_range=read_range)
    logger.info("reference volume: %d slices, z = %.1f .. %.1f mm",
                vol.shape[0], zpos.min(), zpos.max())

    slab = iq.find_insert_slab(vol, pixel_mm, zpos, search_z_mm=phantom_z,
                               select=cfg.slab_select)
    logger.info("candidate insert layers (ranked by %s):", cfg.slab_select)
    for i, c in enumerate(slab["candidates"][:8]):
        mark = " <- selected" if (c["k_lo"], c["k_hi"]) == (slab["k_lo"], slab["k_hi"]) else ""
        logger.info("   %d. z = %+9.2f .. %+9.2f mm  %3d slices  peak %.4f  mean %.4f%s",
                    i + 1, c["z_lo_mm"], c["z_hi_mm"], c["n_slices"],
                    c["peak_score"], c["mean_score"], mark)
    if len(slab["candidates"]) > 1:
        logger.info("   (pass --slab z_lo,z_hi to force a different layer)")

    z_lo = slab["z_lo_mm"] - cfg.slab_pad_mm
    z_hi = slab["z_hi_mm"] + cfg.slab_pad_mm
    slab_source = f"structure score ({cfg.slab_select})"
    seg_insert_labels = None       # set when a segmentation defines the slab

    # A segmentation knows where the inserts are far better than any structure score can
    # infer it, so when one is supplied it defines the slab.  This matters when the score
    # is flat -- inserts that run as rods along z produce no distinguishable "layer", and
    # the score then spans the whole phantom and gets blindly trimmed to its middle.
    if cfg.segmentation and cfg.use_phantom_z is not False:
        labels = load_segmentation(cfg.segmentation, img,
                                   low_priority_label=cfg.background_label)
        if labels is None:
            logger.warning("segmentation %s does not overlap the reference volume -- "
                           "keeping the detected slab", cfg.segmentation)
        else:
            # The slab is defined by where the INSERTS are, so the background segment is
            # excluded here: it may be drawn over a different z range (or over the whole
            # module), and letting it stretch the slab would move the measurement away
            # from the inserts the TTF depends on.
            insert_labels = labels
            if cfg.background_label:
                insert_labels = np.where(labels == int(cfg.background_label), 0, labels)
            kz = np.where(insert_labels.reshape(insert_labels.shape[0], -1).any(axis=1))[0]
            if kz.size == 0:
                kz = np.where(labels.reshape(labels.shape[0], -1).any(axis=1))[0]
            z_lo = float(zpos[kz].min()) - cfg.slab_pad_mm
            z_hi = float(zpos[kz].max()) + cfg.slab_pad_mm
            slab_source = f"segmentation ({Path(cfg.segmentation).name})"
            seg_insert_labels = insert_labels
            logger.info("slab taken from the segmentation: z = %.2f .. %.2f mm "
                        "(%d annotated slices, labels %s)", z_lo, z_hi, kz.size,
                        sorted(int(v) for v in np.unique(labels) if v > 0))

    if cfg.slab_mm is not None:
        z_lo, z_hi = sorted(cfg.slab_mm)
        slab_source = "--slab"
        logger.info("slab overridden by --slab: %.2f .. %.2f mm", z_lo, z_hi)
    elif (z_hi - z_lo) > cfg.max_slab_mm:
        # The trim exists to keep the GPU sweep affordable when the slab came from a
        # structure score, where an over-long slab means detection failed.  A slab drawn
        # by hand is different: it is exactly the range the annotator cared about, and
        # trimming it silently drops whole insert layers -- which then surfaces at stage 2
        # as "found 12 inserts but 18 were expected", after the GPU work is already done.
        if seg_insert_labels is not None:
            logger.info("slab is %.1f mm, longer than --max-slab-mm %.1f, but it was DRAWN "
                        "-- keeping it whole so no annotated insert layer is lost "
                        "(costs ~%.0f%% more sweep time)", z_hi - z_lo, cfg.max_slab_mm,
                        100.0 * ((z_hi - z_lo) / cfg.max_slab_mm - 1.0))
        else:
            mid = 0.5 * (z_lo + z_hi)
            z_lo, z_hi = mid - cfg.max_slab_mm / 2, mid + cfg.max_slab_mm / 2
            logger.info("slab trimmed to %.1f mm about its centre", cfg.max_slab_mm)
    if phantom_z:                       # never let padding leave the phantom
        z_lo = max(z_lo, min(phantom_z))
        z_hi = min(z_hi, max(phantom_z))
    logger.info("insert slab: z = %.2f .. %.2f mm (%d slices in the vendor volume)",
                z_lo, z_hi, int(np.sum((zpos >= z_lo) & (zpos <= z_hi))))

    # How many annotated inserts does this slab actually keep?  Counted exactly the way
    # inserts_from_labelmap will count them later -- collapse along z, then label
    # connected components -- so this number IS what stage 2 will find.  Checking it here
    # is the whole point: at stage 0 a lost layer costs one flag to fix, at stage 2 it
    # costs the entire GPU sweep.
    n_annotated = n_in_slab = None
    if seg_insert_labels is not None:
        keep = (zpos >= z_lo) & (zpos <= z_hi)
        # Count PER LABEL VALUE, exactly as inserts_from_labelmap does.  Projecting every
        # label into one binary mask merges adjacent rows and undercounts: on this
        # segmentation it reported 6 where stage 2 then resolved 11.
        def _n_components(lab3d):
            return int(sum(ndimage.label(np.any(lab3d == v, axis=0))[1]
                           for v in np.unique(lab3d) if v > 0))

        n_annotated = _n_components(seg_insert_labels)
        n_in_slab = _n_components(seg_insert_labels[keep]) if keep.any() else 0
        logger.info("annotated inserts: %d drawn, %d inside the slab", n_annotated,
                    n_in_slab)
        if n_in_slab < n_annotated:
            logger.warning("the slab drops %d of %d annotated inserts. Widen it with "
                           "--max-slab-mm or --slab, or the missing inserts stay inside "
                           "the background region and inflate the noise SD.",
                           n_annotated - n_in_slab, n_annotated)
        if cfg.expect_inserts and n_annotated < int(cfg.expect_inserts):
            # Not an error: annotating every insert on a low-contrast phantom is hard,
            # and a partial annotation is supported by design.  Say so plainly here so
            # the stage-2 count is not read as a detection failure.
            logger.info("--expect-inserts is %d but only %d are drawn in the "
                        "segmentation. Stage 2 will report %d; that is the annotation, "
                        "not a detection failure.", int(cfg.expect_inserts), n_annotated,
                        n_in_slab)

    _qc_slab_figure(out / "qc" / "stage0_slab_detection.png", vol, zpos, slab, z_lo, z_hi,
                    pixel_mm, ref_kind, ref["label"], phantom_z)

    match = {
        "reference_family": ref_kind,
        "reference_label": ref["label"],
        "kernel": ref["kernel"],
        "n_pixels": int(cfg.n_pixels or cols),
        "fov_mm": float(cfg.fov_mm or fov),
        "pixel_mm": float((cfg.fov_mm or fov) / (cfg.n_pixels or cols)),
        "slice_thickness_mm": float(cfg.slice_thickness_mm or thickness),
        "slice_spacing_mm": float(cfg.slice_spacing_mm or spacing),
        "z_lo_mm": float(z_lo), "z_hi_mm": float(z_hi),
        "families": {k: [{"label": c["label"], "uid": c["uid"], "desc": c["desc"],
                          "kernel": c["kernel"], "thickness_mm": c["thickness_mm"],
                          "spacing_between_mm": c["spacing_between_mm"],
                          "pixel_mm": c["pixel_mm"], "rows": c["rows"], "cols": c["cols"],
                          "n_files": len(c["files"])}
                         for c in v] for k, v in families.items()},
        # In-plane origin of the vendor grid.  Stage 1 used to assume an isocentre-centred
        # FOV for our volumes; on this scanner the vendor frame is table-referenced, so
        # that assumption displaced every ROI by the table height.  None -> stage 1 falls
        # back to the old assumption and says so.
        "origin_x_mm": origin_x, "origin_y_mm": origin_y,
        "table_height_mm": table_height,
        # What the segmentation actually contains, so stage 2 can report "12 of the 13
        # drawn" instead of "12 of the 18 you hoped for".
        "n_annotated_inserts": n_annotated,
        "n_inserts_in_slab": n_in_slab,
        "mirror_x": bool(cfg.mirror_x),
        "wfbp_dir": cfg.wfbp_dir, "vmi_dir": cfg.vmi_dir,
        # Persisted because each stage is a SEPARATE process: stage 2 is invoked
        # without --wfbp-dir, so auto-discovery cannot run there and the segmentation
        # would silently be dropped between stages.
        "segmentation": cfg.segmentation,
        "background_label": cfg.background_label,
    }
    _write_json(out / "match_config.json", match)
    logger.info("wrote %s", out / "match_config.json")
    logger.info("CHECK output/research/recon_comparison/qc/stage0_slab_detection.png "
                "before running stage 1 -- the marked band must contain the inserts.")
    del vol
    gc.collect()
    return match


def _qc_slab_figure(path, vol, zpos, slab, z_lo, z_hi, pixel_mm, kind, label,
                    phantom_z=None):
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(zpos, slab["score"], lw=1.2)
    ax[0].axhline(slab["threshold"], color="grey", ls=":", label="threshold")
    # Every candidate layer is drawn, not just the winner: a phantom with several insert
    # layers should show several bands, and seeing the ones that were NOT chosen is how
    # you tell "picked the wrong layer" from "only found one".
    for i, c in enumerate(slab.get("candidates", [])):
        ax[0].axvspan(c["z_lo_mm"], c["z_hi_mm"], color="tab:blue", alpha=0.12,
                      label="candidate layers" if i == 0 else None)
    if phantom_z:
        for zv in sorted(phantom_z):
            ax[0].axvline(zv, color="tab:green", ls="--", lw=1.0)
        ax[0].axvline(np.nan, color="tab:green", ls="--", lw=1.0, label="--phantom-z")
    ax[0].axvspan(z_lo, z_hi, color="tab:orange", alpha=0.35, label="selected slab")
    ax[0].set_xlabel("z (mm)")
    ax[0].set_ylabel("fraction of body voxels far from median")
    ax[0].set_title(f"insert-layer score ({len(slab.get('candidates', []))} candidates)")
    ax[0].legend(fontsize=8)

    k_mid = int(np.clip((slab["k_lo"] + slab["k_hi"]) // 2, 0, vol.shape[0] - 1))
    ax[1].imshow(vol[k_mid], cmap="gray", vmin=-200, vmax=300)
    ax[1].set_title(f"selected slab, mid slice (k={k_mid}, z={zpos[k_mid]:+.1f} mm)")
    ax[1].axis("off")

    k_out = 0 if slab["k_lo"] > vol.shape[0] // 2 else vol.shape[0] - 1
    ax[2].imshow(vol[k_out], cmap="gray", vmin=-200, vmax=300)
    ax[2].set_title(f"outside the slab (k={k_out}, z={zpos[k_out]:+.1f} mm)")
    ax[2].axis("off")
    fig.suptitle(f"Stage 0 slab detection -- reference {kind} {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
# Stage 1 -- the reconstruction sweep (GPU)
# ═════════════════════════════════════════════════════════════════════
def _variant_signature(variant, match, cfg: CompareConfig):
    """Everything that would invalidate a cached variant if it changed."""
    return {
        "variant": variant,
        "n_pixels": match["n_pixels"], "fov_mm": match["fov_mm"],
        "slice_thickness_mm": match["slice_thickness_mm"],
        "slice_spacing_mm": match["slice_spacing_mm"],
        "z_lo_mm": match["z_lo_mm"], "z_hi_mm": match["z_hi_mm"],
        "geometry_model": cfg.geometry_model, "z_weighting": cfg.z_weighting,
        "wavelet_ring_threshold": cfg.wavelet_ring_threshold,
        "spike_mad_k": cfg.spike_mad_k, "ipr_mad_k": cfg.ipr_mad_k,
        "data_path": cfg.data_path,
        # Output-space geometry.  Part of the signature because a cached volume written
        # on the wrong origin is silently unusable, but listed in _GEOMETRY_ONLY_KEYS
        # because it can be corrected without touching the GPU.
        "origin_x_mm": match.get("origin_x_mm"),
        "origin_y_mm": match.get("origin_y_mm"),
        "mirror_x": bool(match.get("mirror_x", cfg.mirror_x)),
    }


# Signature keys that describe how the volume was WRITTEN rather than how it was
# reconstructed.  A variant differing only in these is repaired in place.
_GEOMETRY_ONLY_KEYS = ("origin_x_mm", "origin_y_mm", "mirror_x")


def _stored_signature(vdir: Path):
    try:
        return json.loads((vdir / "sweep_config.json").read_text(encoding="utf-8")
                          ).get("signature")
    except (OSError, ValueError):
        return None


def _differs_only_in_geometry(stored, signature):
    """True if the cached volume is sound and only its written geometry is out of date."""
    if not isinstance(stored, dict):
        return False
    strip = lambda d: _jsonable({k: v for k, v in d.items()
                                 if k not in _GEOMETRY_ONLY_KEYS})
    return (strip(stored) == strip(signature)
            and _jsonable(stored) != _jsonable(signature))


def _repair_variant_geometry(vdir: Path, stored, signature):
    """
    Rewrite a cached variant's origin and mirror state without reconstructing it.

    Both corrections are pure output-space transforms, so the reconstructed values are
    untouched and re-running the GPU sweep would buy nothing.  The mirror is applied
    RELATIVE to the state the volume was written in, so repairing twice is a no-op
    rather than flipping it back.
    """
    sitk = _sitk()
    flip = bool(stored.get("mirror_x", False)) != bool(signature.get("mirror_x", False))
    ox, oy = signature.get("origin_x_mm"), signature.get("origin_y_mm")
    for lbl in THRESHOLD_LABELS:
        p = vdir / f"reconstruction_thr_{lbl}_HU.nii.gz"
        img = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(img)
        if flip:
            arr = np.ascontiguousarray(arr[:, :, ::-1])
        out = sitk.GetImageFromArray(arr)
        out.SetSpacing(img.GetSpacing())
        out.SetDirection(img.GetDirection())
        o = img.GetOrigin()
        out.SetOrigin((float(ox) if ox is not None else o[0],
                       float(oy) if oy is not None else o[1], float(o[2])))
        sitk.WriteImage(out, str(p))
    blob = json.loads((vdir / "sweep_config.json").read_text(encoding="utf-8"))
    blob["signature"] = signature
    _write_json(vdir / "sweep_config.json", blob)
    return flip


def _variant_is_current(vdir: Path, signature):
    """
    A variant counts as done only if every volume exists AND its recorded signature
    matches.  Checking file existence alone would silently reuse volumes reconstructed
    from a different slab or grid -- the failure mode that quietly corrupts a comparison.
    """
    cfg_path = vdir / "sweep_config.json"
    if not cfg_path.exists():
        return False
    if not all((vdir / f"reconstruction_thr_{l}_HU.nii.gz").exists() for l in THRESHOLD_LABELS):
        return False
    try:
        stored = json.loads(cfg_path.read_text(encoding="utf-8")).get("signature")
    except (OSError, ValueError):
        return False
    return _jsonable(stored) == _jsonable(signature)


def stage1_sweep(cfg: CompareConfig):
    """Reconstruct every variant over the matched slab and grid."""
    import h5py
    import scipy.io as sio
    from reconstruction.helical_reconstruction import (
        build_geom, detect_defect_channels, reconstruct_helical_stack,
        z_targets_for_full_scan, auto_hu_calibrate, apply_hu_calibration,
        to_output_frame)

    out = Path(cfg.out_root)
    match = _load_match(out)

    todo, repaired = [], 0
    for v in cfg.variants:
        vdir = out / "sweep" / v["name"]
        sig = _variant_signature(v, match, cfg)
        if not cfg.force and _variant_is_current(vdir, sig):
            logger.info("variant %-16s up to date -- skipping", v["name"])
            continue
        stored = _stored_signature(vdir)
        if (not cfg.force and stored is not None
                and all((vdir / f"reconstruction_thr_{l}_HU.nii.gz").exists()
                        for l in THRESHOLD_LABELS)
                and _differs_only_in_geometry(stored, sig)):
            flipped = _repair_variant_geometry(vdir, stored, sig)
            logger.info("variant %-16s repaired in place (origin%s) -- no GPU needed",
                        v["name"], " + mirror" if flipped else "")
            repaired += 1
            continue
        todo.append((v, vdir, sig))
    if repaired:
        logger.info("%d variant(s) corrected without reconstructing: the change was in "
                    "how the volume is written, not in the reconstruction itself",
                    repaired)
    if not todo:
        logger.info("all %d variants up to date (use --force to redo)", len(cfg.variants))
        return
    logger.info("reconstructing %d/%d variants", len(todo), len(cfg.variants))

    descriptor = sio.loadmat(cfg.desc_path, struct_as_record=True,
                             squeeze_me=False)["descriptor"].flat[0]
    geom = build_geom(descriptor, geo_dir=cfg.geo_dir, channels_flipped=True)

    end_margin = 1.0 if cfg.z_weighting == "balanced" else 0.5
    z_all, z_spacing = z_targets_for_full_scan(geom, oversample=1,
                                               end_margin_rotations=end_margin)
    # Native slices spanning the slab, padded by half the target thickness so the
    # outermost matched slice still has a full averaging window.
    pad = 0.5 * match["slice_thickness_mm"] + abs(z_spacing)
    # Report both frames: the slab comes from the vendor DICOM z (ImagePositionPatient)
    # while we reconstruct in our own helical z.  They should share the scanner table
    # coordinate, but that is an assumption -- printing both makes a mismatch obvious
    # instead of it showing up later as a phantom that is missing from qc/roi_own.png.
    logger.info("z frames: ours %.1f .. %.1f mm | requested slab (vendor frame) "
                "%.1f .. %.1f mm", z_all.min(), z_all.max(),
                match["z_lo_mm"], match["z_hi_mm"])
    sel = (z_all >= match["z_lo_mm"] - pad) & (z_all <= match["z_hi_mm"] + pad)
    z_native = z_all[sel]
    if z_native.size == 0:
        raise ValueError(
            f"slab z={match['z_lo_mm']:.1f}..{match['z_hi_mm']:.1f} mm lies outside the "
            f"reconstructable range z={z_all.min():.1f}..{z_all.max():.1f} mm. The vendor "
            f"z-origin and ours may differ -- check qc/stage0_slab_detection.png and pass "
            f"--slab explicitly.")
    z_target = np.arange(match["z_lo_mm"], match["z_hi_mm"] + 1e-6,
                         match["slice_spacing_mm"])
    logger.info("slab: %d native slices (dz=%.4f mm) -> %d matched slices "
                "(thickness %.2f mm, spacing %.2f mm)",
                z_native.size, abs(z_spacing), z_target.size,
                match["slice_thickness_mm"], match["slice_spacing_mm"])

    with h5py.File(cfg.data_path, "r") as f:
        sino_A = f[f["data_full"]["A"][3, 0]][:][:, :, ::-1].astype(np.float32)
    geom["spike_mask"] = detect_defect_channels(sino_A, spike_mad_k=cfg.spike_mad_k,
                                                ipr_mad_k=cfg.ipr_mad_k)
    logger.info("defect mask: %d channels (from threshold A, reused for B/C/D)",
                int(geom["spike_mask"].sum()))
    del sino_A
    gc.collect()

    sitk = _sitk()
    n_pixels = match["n_pixels"]
    xy = match["fov_mm"] / n_pixels
    # Write on the VENDOR's in-plane origin when stage 0 recorded it.  Assuming an
    # isocentre-centred FOV is only right if the vendor grid is centred too; on this
    # scanner it is table-referenced, and the difference put every ROI ~88 mm out.
    iso = -match["fov_mm"] / 2 + xy / 2
    ox = match.get("origin_x_mm")
    oy = match.get("origin_y_mm")
    if ox is None or oy is None:
        ox, oy = iso, iso
        logger.warning("no vendor in-plane origin in match_config.json -- assuming an "
                       "isocentre-centred FOV. Re-run stage 0 to record it.")
    else:
        logger.info("in-plane origin from the vendor: x %+.3f  y %+.3f mm "
                    "(isocentre-centred would be %+.3f)", ox, oy, iso)
    origin = (float(ox), float(oy), float(z_target[0]))
    mirror_x = bool(match.get("mirror_x", cfg.mirror_x))
    if mirror_x:
        logger.info("applying the left-right mirror to the written volumes "
                    "(x -> -x; reconstruction and detector channels unchanged)")

    for li, label in enumerate(THRESHOLD_LABELS):
        with h5py.File(cfg.data_path, "r") as f:
            sino = f[f["data_full"]["A"][3 - li, 0]][:][:, :, ::-1].astype(np.float32)
        logger.info("threshold %s loaded %s", label, sino.shape)
        for v, vdir, sig in todo:
            vdir.mkdir(parents=True, exist_ok=True)
            logger.info("  variant %-16s threshold %s ...", v["name"], label)
            vol = reconstruct_helical_stack(
                sino, geom, z_native, method="astra", n_pixels=n_pixels,
                filter_name=v.get("filter_name", "shepp-logan"),
                geometry_model=cfg.geometry_model,
                z_weighting=cfg.z_weighting,
                wavelet_ring_threshold=cfg.wavelet_ring_threshold,
                algorithm=v["algorithm"], n_iter=v.get("n_iter", 100),
            )
            # Z_SMOOTH stays off; thickness is matched by averaging instead.
            matched, n_used = iq.match_slice_thickness(
                vol, z_native, match["slice_thickness_mm"], z_target)
            del vol
            gc.collect()

            calib = auto_hu_calibrate(matched, fov_mm=match["fov_mm"])
            _write_json(vdir / f"calibration_thr_{label}.json", calib)
            vol_hu = apply_hu_calibration(matched, calib)
            del matched

            # One shared conversion with the production driver (helical_reconstruction
            # .to_output_frame), so the sweep and output/reconstruction/ cannot drift into
            # different conventions.  z_target already ascends here, so only the mirror and
            # the origin are in play; the origin is passed through unchanged because it was
            # read from the vendor DICOM rather than assumed.
            vol_hu, _o, _dz = to_output_frame(
                vol_hu, z_target, match["fov_mm"], n_pixels,
                mirror_x=mirror_x, table_height_mm=0.0)
            img = sitk.GetImageFromArray(vol_hu.astype(np.float32))
            img.SetSpacing((xy, xy, abs(match["slice_spacing_mm"])))
            img.SetOrigin(origin)
            img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
            sitk.WriteImage(img, str(vdir / f"reconstruction_thr_{label}_HU.nii.gz"))
            logger.info("    -> %s  HU [%.0f, %.0f]  (%.1f native slices averaged)",
                        f"reconstruction_thr_{label}_HU.nii.gz",
                        float(vol_hu.min()), float(vol_hu.max()), float(np.mean(n_used)))
            del vol_hu, img
            gc.collect()
        del sino
        gc.collect()

    for v, vdir, sig in todo:
        _write_json(vdir / "sweep_config.json",
                    {"signature": sig, "z_target_mm": z_target.tolist(),
                     "n_native_slices": int(z_native.size)})
    logger.info("sweep complete: %d variants", len(todo))


# ═════════════════════════════════════════════════════════════════════
# Stage 2 -- metrics
# ═════════════════════════════════════════════════════════════════════
def _slice_at_z(img, z_mm):
    """Index of the slice nearest a PHYSICAL z in this image's own geometry."""
    o, sp, n = img.GetOrigin()[2], img.GetSpacing()[2], img.GetSize()[2]
    d = img.GetDirection()
    step = sp * (d[8] if len(d) == 9 else 1.0)
    if abs(step) < 1e-9:
        return 0
    return int(np.clip(round((z_mm - o) / step), 0, n - 1))


def _z_span(img):
    """(z_lo, z_hi) in mm covered by this image, whichever way its z axis runs."""
    o, sp, n = img.GetOrigin()[2], img.GetSpacing()[2], img.GetSize()[2]
    d = img.GetDirection()
    step = sp * (d[8] if len(d) == 9 else 1.0)
    a, b = o, o + step * (max(int(n), 1) - 1)
    return (min(a, b), max(a, b))


def _check_pose_against_reference(refs, tol_mm=3.0, tol_deg=1.5):
    """
    Does every source still sit in the same frame, and at the same ANGLE?

    Runs automatically because the frame corrections now live in the reconstruction: if a
    future scan needs a different view-angle offset, or the mirror stops applying, nothing
    else in this pipeline would notice -- every stage completes and only the numbers are
    wrong.  The orientation QC figure shows it to a human; this says it in the log.

    Rotation is checked separately from translation because a centroid cannot see it: an
    object turned about its own centre has the same centroid.

    Every volume is sampled at ONE COMMON PHYSICAL z.  Taking each volume's middle slice by
    index instead compares different anatomy whenever the volumes cover different ranges --
    a 120-slice imported slab against a 2232-slice vendor series reported a 141 mm
    "displacement" and a 3.4 deg "rotation" for a phantom that had not moved at all, purely
    because the two middle slices sat ~400 mm apart. False alarms of that size are worse
    than no check: they train the reader to ignore the one that is real.
    """
    ref_tag = "wfbp" if "wfbp" in refs else next(iter(refs), None)
    if ref_tag is None or len(refs) < 2:
        return {}
    spans = [_z_span(img) for _v, img, _p in refs.values()]
    lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
    if lo > hi:
        logger.warning("pose check skipped: the sources share no z range "
                       "(%s) -- they are not covering the same object.",
                       ", ".join(f"{t}=[{s[0]:.0f},{s[1]:.0f}]"
                                 for t, s in zip(refs, spans)))
        return {}
    z_common = 0.5 * (lo + hi)
    logger.info("pose check at z = %+.1f mm (shared by all sources)", z_common)
    poses, out = {}, {}
    for tag, (vol, img, px) in refs.items():
        k = min(_slice_at_z(img, z_common), vol.shape[0] - 1)
        o = img.GetOrigin()
        poses[tag] = iq.body_pose(vol[k], px, origin_xy=(o[0], o[1]))
    for tag, p in poses.items():
        if tag == ref_tag:
            continue
        d = iq.pose_difference(p, poses[ref_tag])
        out[tag] = d
        rot = d["axis_deg"]
        logger.info("pose vs %s -- %s: translation (%+.2f, %+.2f) mm, rotation %+.2f deg "
                    "(bearing %+.2f)%s", ref_tag, tag,
                    *(d["translation_mm"] or (float("nan"),) * 2), rot,
                    d["bearing_deg"], "" if d["reliable"] else "  [axis UNRELIABLE: "
                    "body too round]")
        if d["distance_mm"] > tol_mm:
            logger.warning("   %s is displaced %.2f mm from %s -- ROIs drawn on one will "
                           "not measure the same material in the other.",
                           tag, d["distance_mm"], ref_tag)
        if d["reliable"] and abs(rot) > tol_deg:
            logger.warning("   %s is ROTATED %+.2f deg vs %s. Not an origin problem: "
                           "suspect the view-angle convention (_VIEW_ANGLE_OFFSET_DEG, "
                           "invariant #8, README). Check qc/qc_orientation.png.",
                           tag, rot, ref_tag)
    return out


def _pick_ttf_insert(rois_by_family, min_radius_mm=2.5):
    """
    Choose ONE physical insert for the TTF, used by every family.

    Otherwise each family measures its TTF on whichever of its own inserts happens to
    have the most contrast there -- on this scan `own` picked a -68 HU insert while
    `wfbp` picked a +44 HU one, so the two TTF numbers described different cylinders and
    were never comparable.  The choice is made on the family with the strongest contrast
    (VMI when present), and needs a radius big enough to carry an edge at all.
    """
    best, best_score = None, -1.0
    for tag in ("vmi", "wfbp", "own"):
        rois = rois_by_family.get(tag)
        if not rois:
            continue
        for ins in rois:
            if "geom_id" not in ins or ins["radius_mm"] < min_radius_mm:
                continue
            score = abs(ins.get("contrast_hu", 0.0))
            if score > best_score:
                best, best_score = int(ins["geom_id"]), score
        if best is not None:
            return best, tag, best_score
    return None, None, 0.0


def _measure_volume(vol, pixel_mm, rois, cfg: CompareConfig, body, patches, patch_px,
                    region, ttf_geom_id=None):
    """
    Full metric set for one volume, given ROIs fixed by the family's reference.

    NPS is measured slice by slice -- never on a z-average, which would divide the
    noise by sqrt(Z) and report a quantity no single image has.  The TTF is measured on
    the z-average instead: averaging slices leaves in-plane resolution untouched (the
    slab lies inside one insert layer, so the cylinders are z-invariant across it) while
    cutting the noise that would otherwise corrupt the differentiated edge profile.
    """
    res = {}
    # Noise MAGNITUDE from the whole region: needs no square blocks, so it survives a
    # phantom whose uniform material is only a thin ring between inserts.
    sd, n_vox = iq.region_noise_sd(vol, region)
    res["noise_sd_hu"] = sd
    res["noise_region_px"] = n_vox
    # ABSOLUTE accuracy.  Everything else on this list is relative or paired, so without
    # this nothing states whether the HU scale is right -- only whether two of them agree.
    # The same region serves: it is the phantom's uniform body material, inserts already
    # removed.  See the note in water_reference() on why the 'own' number is a calibration
    # residual rather than an accuracy.
    res["water"] = iq.water_reference(vol, region, pixel_mm, expected_hu=WATER_HU)
    # SNR on the ATTENUATION, not on the Hounsfield scale.  The phantom body is
    # water-equivalent and reads about 0 HU, so a Hounsfield ratio would divide by a mean
    # that is zero by construction and say nothing.  Writing mu = mu_water(1 + HU/1000)
    # into mu/sigma_mu cancels mu_water, so the figure below is exact without any
    # calibration constant and is available for the vendor channels too, where we have no
    # mu_water at all.  It is also the quantity the decomposition sees: the weighted
    # estimator forms its channel weights from this ratio, not from the HU noise.
    mean_hu = float(res["water"].get("mean_hu", np.nan))
    if not np.isfinite(mean_hu):
        mean_hu = float(res["water"].get("offset_hu", np.nan)) + WATER_HU
    res["snr_mu"] = (1000.0 + mean_hu) / sd if (sd > 0 and np.isfinite(mean_hu)) else float("nan")
    # Noise TEXTURE needs a Fourier estimate, hence square patches; it degrades to
    # unavailable rather than dragging the magnitude down with it.
    if patches:
        nps = iq.noise_power_spectrum(vol, patches, pixel_mm, patch_px=patch_px)
        res["f_av"] = nps["f_av"]
        res["f_peak"] = nps["f_peak"]
        res["nps_f"] = nps["f"]
        res["nps"] = nps["nps"]
        res["n_patches"] = nps["n_patches"]
        res["patch_heterogeneity"] = nps["patch_heterogeneity"]
        res["nps_sd_hu"] = nps["noise_sd_hu"]
    else:
        res["f_av"] = float("nan")
        res["f_peak"] = float("nan")
        res["n_patches"] = 0
        nps = None

    # Each insert is measured over the slices where it actually EXISTS.  On a layered
    # phantom the slab z-average is the wrong image for every insert: one present in a
    # third of the slices contributes its contrast on a third of them and background on
    # the rest, so its contrast reads about a third of the truth and its edge is buried.
    flat_all = vol.mean(axis=0)

    def _flat_for(ins):
        k0, k1 = ins.get("k_lo"), ins.get("k_hi")
        if k0 is None or k1 is None or (k0 <= 0 and k1 >= vol.shape[0] - 1):
            return flat_all
        return vol[k0:k1 + 1].mean(axis=0)

    flat = flat_all
    # Plausibility bounds on the edge measurement.  NOT theorems: the circular-edge
    # method deliberately samples the ESF below the pixel pitch (radial_edge_profile),
    # and that sub-pixel phase diversity genuinely lets a slanted/circular edge measure
    # past the grid's Nyquist -- ISO 12233 relies on it.  What makes these bounds valid
    # HERE is the geometry: the reconstruction is band-limited by the FBP filter and the
    # detector pitch, so a TTF50 at 1.25x the image Nyquist, or an edge width of a third
    # of a pixel, is not a very sharp reconstruction -- it is a radial profile that
    # locked onto noise because the insert had no contrast to find.
    # Such a measurement is kept OUT of the headline figure but recorded in the metrics,
    # so a bound that is too strict for some future geometry can be seen and relaxed
    # rather than silently swallowing data.
    nyquist = 0.5 / pixel_mm
    valid, rejected = [], []
    for ins in rois:
        try:
            t = iq.task_transfer_function(_flat_for(ins), (ins["cy"], ins["cx"]),
                                          ins["radius_mm"], pixel_mm)
        except ValueError:
            continue
        why = None
        if np.isfinite(t["ttf50"]) and t["ttf50"] > nyquist:
            why = f"TTF50 {t['ttf50']:.3f} > Nyquist {nyquist:.3f} cyc/mm"
        elif (np.isfinite(t["edge_width_1090_mm"])
              and t["edge_width_1090_mm"] < pixel_mm):
            why = (f"edge width {t['edge_width_1090_mm']:.3f} mm < one pixel "
                   f"{pixel_mm:.3f} mm")
        if why:
            rejected.append({"reason": why, "ttf50": float(t["ttf50"]),
                             "edge_width_1090_mm": float(t["edge_width_1090_mm"]),
                             "contrast_hu": float(t["contrast_hu"])})
            continue
        valid.append((ins, t))

    # Prefer the insert agreed across families, so the TTF describes the same object
    # everywhere -- but only while it still carries contrast IN THIS image.  Holding to it
    # unconditionally made the VMI series erratic: the agreed insert is iodine-like and
    # chosen at 40 keV, so by 140 keV it has almost no contrast left and its edge is
    # measured on noise.  Falling back keeps the curve meaningful, and which insert was
    # used is recorded so a mixed series is never mistaken for a like-for-like one.
    best = None
    if valid:
        c_max = max(abs(t["contrast_hu"]) for _i, t in valid)
        # `is not None` guard matters: without it, ROIs that carry no geom_id at all match
        # a ttf_geom_id of None and are mistaken for the agreed insert.
        agreed = (next((t for i, t in valid if i.get("geom_id") == ttf_geom_id), None)
                  if ttf_geom_id is not None else None)
        if agreed is not None and abs(agreed["contrast_hu"]) >= 0.3 * c_max:
            best, res["ttf_insert_used"] = agreed, int(ttf_geom_id)
        else:
            ins_b, best = max(valid, key=lambda p: abs(p[1]["contrast_hu"]))
            res["ttf_insert_used"] = ins_b.get("geom_id")
            if agreed is not None:
                res["ttf_insert_fallback"] = True
                logger.info("    the agreed insert carries only %.0f HU here (best "
                            "available %.0f HU) -- TTF taken from insert %s instead",
                            abs(agreed["contrast_hu"]), c_max, res["ttf_insert_used"])
    if rejected:
        res["ttf_rejected"] = rejected
        logger.warning("    %d of %d insert edges gave an implausible TTF and were kept "
                       "out of the headline number (e.g. %s). They are recorded under "
                       "'ttf_rejected' in metrics.json.",
                       len(rejected), len(rois), rejected[0]["reason"])
    if best is not None:
        res.update({"ttf50": best["ttf50"], "ttf10": best["ttf10"],
                    "edge_width_1090_mm": best["edge_width_1090_mm"],
                    "ttf_contrast_hu": best["contrast_hu"],
                    "ttf_f": best["f"], "ttf": best["ttf"]})
        if nps is not None:
            f_neq, val = iq.neq(best["f"], best["ttf"], nps["f"], nps["nps"],
                                contrast_hu=abs(best["contrast_hu"]) or 1.0)
            res["neq_f"], res["neq"] = f_neq, val
            res.update(iq.detectability_index(best["f"], best["ttf"], nps["f"],
                                              nps["nps"], cfg.task_diameter_mm,
                                              cfg.task_contrast_hu))
    else:
        logger.warning("    no usable edge -- TTF/NEQ/d' unavailable for this volume")

    # ---- delivered contrast, at the SHARED geometry -----------------------------
    # The amplitude axis none of the other metrics has.  Noise SD measures spread (which
    # shrinks with amplitude), the TTF is normalised to 1 at DC by definition (so it
    # discards contrast), and d' assumes the task contrast is delivered.  A reconstruction
    # that has not converged violates exactly that assumption: it delivers less contrast
    # AND less noise, and every one of those three metrics rewards it.  Measuring the
    # contrast actually delivered is what catches it, and it stays a separate column so d'
    # keeps the standard definition and remains comparable with the literature.
    yy, xx = np.ogrid[:flat_all.shape[0], :flat_all.shape[1]]
    contrasts, by_id = [], {}
    for ins in rois:
        f_ins = _flat_for(ins)
        rr2 = (yy - ins["cy"]) ** 2 + (xx - ins["cx"]) ** 2
        r_px = max(float(ins["radius_px"]), 1.0)
        inner = rr2 <= (0.7 * r_px) ** 2
        outer = (rr2 >= (1.6 * r_px) ** 2) & (rr2 <= (2.6 * r_px) ** 2)
        c = (float(np.median(f_ins[inner]) - np.median(f_ins[outer]))
             if inner.sum() >= 4 and outer.sum() >= 8 else float("nan"))
        contrasts.append(c)
        if "geom_id" in ins:
            by_id[str(ins["geom_id"])] = c
    res["insert_contrast_hu"] = contrasts
    res["insert_contrast_by_id"] = by_id

    res["roi"] = iq.roi_statistics(vol, rois, body, pixel_mm)
    return res


def _contrast_fidelity(res):
    """
    Fraction of the vendor's insert contrast that each of our variants actually delivers.

    Paired 1:1 by energy channel (own A <-> WFBP T1, ...) because that is the only pairing
    where the two images are the same physical quantity.  Reported per insert as a median
    ratio, so one badly-measured insert cannot dominate.  A value near 1 means the
    reconstruction reproduces the vendor's contrast; well below 1 means it does not, and
    any noise or detectability number from it is flattering for the wrong reason.
    """
    for vname, per_thr in res.get("own", {}).items():
        for j, label in enumerate(THRESHOLD_LABELS):
            m = per_thr.get(label)
            wf = res.get("wfbp", {}).get(f"T{j + 1}")
            if not m or not wf:
                continue
            # Paired by geom_id -- the physical insert -- never by list position: the ROI
            # lists are sorted by refined centre and refinement shifts centres by
            # different amounts in each family, so index i is not the same cylinder.
            a, b = m.get("insert_contrast_by_id"), wf.get("insert_contrast_by_id")
            if not a or not b:
                continue
            ratios = [a[k] / b[k] for k in set(a) & set(b)
                      if np.isfinite(a[k]) and np.isfinite(b[k]) and abs(b[k]) > 5.0]
            if len(ratios) >= 3:
                m["contrast_fidelity"] = float(np.median(ratios))
                m["contrast_fidelity_n"] = len(ratios)


def family_insert_seeds(vol_ref, pixel_mm, cfg: CompareConfig, ref_img):
    """
    Just the insert ROIs for one family -- the cheap half of _family_rois.

    Exists so the shared geometry can be worked out from every family BEFORE any of them
    builds its noise region, without running (and logging) the full detection twice.

    Takes the VOLUME, not the slab average, and must stay that way: detecting here on a
    flattened image while _family_rois detects per layer gave 11 seeds against 17 ROIs, so
    several layered inserts collapsed onto one shared-geometry entry and shared a geom_id.
    """
    if not cfg.segmentation or ref_img is None:
        return []
    labels = load_segmentation(cfg.segmentation, ref_img,
                               low_priority_label=cfg.background_label)
    if labels is None:
        return []
    if cfg.background_label:
        labels = np.where(labels == int(cfg.background_label), 0, labels)
    return iq.inserts_from_labelmap(labels, vol_ref.mean(axis=0), pixel_mm,
                                    refine=True, vol_hu=vol_ref)


def _insert_xy_mm(ins, img):
    """Physical (x, y) of an insert centre, so families can be matched across grids."""
    p = img.TransformContinuousIndexToPhysicalPoint(
        (float(ins["cx"]), float(ins["cy"]), 0.0))
    return np.array([p[0], p[1]], float)


def consensus_insert_geometry(per_family, tol_mm=6.0, disagree_factor=1.5):
    """
    One radius per physical insert, shared by every family.

    WHY A SHARED RADIUS.  Each family refines its own edges, and in a low-CNR image that
    refinement collapses: the mid-level threshold lands in noise and the connected
    component it grabs is a noise blob whose size is set by the noise correlation length,
    not by the insert.  Measured on this scan, WFBP recovers the large inserts at
    4.9-5.1 mm while our reconstruction returns 1.2-2.9 mm for the same cylinders -- and
    tellingly the SMALL inserts agree, which is backwards for a resolution effect and is
    the signature of every insert converging on the same noise blob.

    Letting each family keep its own radius hides the very thing being measured: a worse
    image gets a smaller ROI, and a sharp edge is then measured on that small blob, which
    is how a TTF50 above Nyquist arises.  Fixing the geometry instead pushes the quality
    loss into the metric, where it belongs.

    WHY THE MAXIMUM.  The failure is one-directional -- refinement shrinks, it does not
    grow, because a noise blob is small and refine_insert_geometry clamps the accepted
    crossing to 2.5x the seed anyway.  The largest estimate is therefore the least
    corrupted.  The one way it can go wrong is two adjacent inserts bridging into a single
    component in a noisy image, so families that disagree by more than `disagree_factor`
    are logged rather than silently reconciled.

    per_family : {tag: (inserts, ref_img)}
    Returns [{"xy_mm", "radius_mm", "source", "per_family"}], one entry per insert.
    """
    # Anchor on the family that resolved the most inserts, so nothing it found is lost.
    anchor = max(per_family, key=lambda t: len(per_family[t][0]))
    entries = []
    for ins in per_family[anchor][0]:
        entries.append({"xy_mm": _insert_xy_mm(ins, per_family[anchor][1]),
                        "layer": int(ins.get("label", 0)),
                        "per_family": {}, "radius_mm": None, "source": None})
    for tag, (inserts, img) in per_family.items():
        for ins in inserts:
            xy = _insert_xy_mm(ins, img)
            layer = int(ins.get("label", 0))
            # Match within the LAYER.  On a layered phantom two inserts in different
            # layers can share an (x, y) exactly, so position alone would fuse them --
            # and every family uses the same segmentation, so the label is identical.
            cand = [(float(np.hypot(*(xy - e["xy_mm"]))), i)
                    for i, e in enumerate(entries) if e["layer"] == layer]
            if not cand:
                continue
            d, k = min(cand)
            if d > tol_mm:               # no counterpart: this family sees an extra ROI
                continue
            entries[k]["per_family"][tag] = {"radius_mm": float(ins["radius_mm"]),
                                             "refined": bool(ins["refined"]),
                                             "contrast_hu": float(ins["contrast_hu"])}
    live = [e for e in entries if e["per_family"]]
    # Distance to the nearest OTHER insert: a radius larger than that describes a disc
    # that swallows its neighbour's centre, which is a merged blob rather than a big
    # insert.  This is the one way "take the max" can fail, so it is checked rather than
    # trusted -- measured on this scan, a VMI blob at (+36.2, -177.7) refined to 9.81 mm
    # while its neighbour sat 7.85 mm away.
    for e in live:
        # Neighbours WITHIN THE LAYER only: an insert in another layer sits at a different
        # z and cannot be swallowed, however close it is in (x, y).
        d = [float(np.hypot(*(e["xy_mm"] - o["xy_mm"]))) for o in live
             if o is not e and o["layer"] == e["layer"]]
        e["_nn_mm"] = min(d) if d else float("inf")

    out = []
    for e in live:
        vals = e["per_family"]
        refined = {t: v["radius_mm"] for t, v in vals.items() if v["refined"]}
        pool = refined or {t: v["radius_mm"] for t, v in vals.items()}
        # Largest candidate that does not swallow a neighbour; if every candidate does,
        # keep the smallest and say so rather than silently propagating a merge.
        ok = {t: r for t, r in pool.items() if r <= e["_nn_mm"]}
        merged = sorted(set(pool) - set(ok))
        chosen = ok or pool
        src = max(chosen, key=chosen.get) if ok else min(chosen, key=chosen.get)
        e["radius_mm"], e["source"] = float(chosen[src]), src
        e["nearest_insert_mm"] = float(e["_nn_mm"])
        del e["_nn_mm"]
        if merged:
            logger.warning("    insert at (%+.1f, %+.1f) mm: %s gave a radius that "
                           "reaches past the neighbouring insert %.2f mm away (a merged "
                           "blob, not a big insert). Using %.2f mm from %s instead.",
                           e["xy_mm"][0], e["xy_mm"][1],
                           "/".join(f"{t} {pool[t]:.2f}" for t in merged),
                           e["nearest_insert_mm"], e["radius_mm"], src)
        lo, hi = min(pool.values()), max(pool.values())
        if lo > 0 and hi / lo > disagree_factor and not merged:
            logger.warning("    insert at (%+.1f, %+.1f) mm: families disagree on the "
                           "radius by %.1fx (%s). Taking %.2f mm from %s.",
                           e["xy_mm"][0], e["xy_mm"][1], hi / lo,
                           ", ".join(f"{t} {r:.2f}" for t, r in sorted(pool.items())),
                           e["radius_mm"], src)
        out.append(e)
    return out


def _family_rois(vol_ref, pixel_mm, cfg: CompareConfig, tag, qc_dir, force_patch_px=None,
                 ref_img=None, consensus=None):
    """
    Detect body/inserts/background patches ONCE per family, then reuse for every channel
    and variant of that family, so a metric difference can never come from the ROIs
    having moved.

    Detection runs on the z-AVERAGE of the slab, not on one slice.  The first volume the
    driver happens to load is whichever variant comes first (ram-lak -- deliberately the
    noisiest of the five), and on a single noisy slice a low-contrast insert can fail the
    contrast test, so it is never excluded from the background and the NPS patches then
    sit on top of it.  Averaging the slab lifts detection SNR by sqrt(Z) without moving
    anything, because the inserts are z-invariant within the layer stage 0 selected.
    """
    flat = vol_ref.mean(axis=0)
    body = iq.detect_body_mask(flat, pixel_mm)

    inserts, source, bg_region = [], "auto", None
    seg_path = cfg.segmentation
    if seg_path and ref_img is not None:
        labels = load_segmentation(seg_path, ref_img,
                                   low_priority_label=cfg.background_label)
        if labels is None:
            # Empty overlap is itself a finding: the segmentation was drawn on the vendor
            # volume, so if it maps nowhere onto ours the two z frames disagree.
            logger.warning("  %s: the segmentation does not overlap this volume at all. "
                           "For 'own' that means our helical z and the vendor's "
                           "ImagePositionPatient z are NOT the same frame. Falling back "
                           "to automatic detection for this family.", tag)
        else:
            present = sorted(int(v) for v in np.unique(labels) if v > 0)
            logger.info("  %s: segmentation labels present: %s", tag, present)
            if cfg.background_label and int(cfg.background_label) not in present:
                # Ignoring a wrong value silently would leave the background segment being
                # measured as an insert and the noise measured somewhere else entirely.
                logger.warning("  %s: --background-label %s is NOT in the segmentation "
                               "(labels present: %s). Falling back to automatic background "
                               "detection -- set the value that actually appears above.",
                               tag, cfg.background_label, present)
            elif cfg.background_label:
                # One drawn segment marks the uniform region the noise is measured in.
                # It must NOT also be treated as an insert.
                drawn = np.any(labels == int(cfg.background_label), axis=0)
                # Filling means an OUTLINE of the module works as well as a solid
                # paint -- which matters because Slicer's fill-between-slices needs
                # a clear path, and tracing the module boundary avoids the inserts
                # entirely.  A solid region is unchanged by this.
                bg_region = ndimage.binary_fill_holes(drawn)
                labels = np.where(labels == int(cfg.background_label), 0, labels)
                logger.info("  %s: background from label %s -- %d px drawn, %d px "
                            "after filling the interior", tag, cfg.background_label,
                            int(drawn.sum()), int(bg_region.sum()))
            # vol_ref is passed so each label is measured over ITS OWN z-slices: on a
            # multi-layer phantom, projecting all layers onto one plane merges inserts
            # that share an (x, y) between layers, and the slab z-average dilutes every
            # insert's contrast by its layer fraction.
            inserts = iq.inserts_from_labelmap(labels, flat, pixel_mm, refine=True,
                                               vol_hu=vol_ref)
            source = f"segmentation ({Path(seg_path).name})"
            n_ref = sum(1 for i in inserts if i["refined"])
            extents = iq.label_z_extents(labels)
            if len(extents) > 1 and any(e[2] < 0.9 * vol_ref.shape[0]
                                        for e in extents.values()):
                logger.info("  %s: the segmentation is LAYERED in z -- each label is "
                            "measured over its own slices:", tag)
                for v, (k0, k1, nk) in sorted(extents.items()):
                    n_here = sum(1 for i in inserts if i["label"] == v)
                    logger.info("      label %d: slices %d-%d (%d of %d, %.0f%% of the "
                                "slab) -> %d insert(s)", v, k0, k1, nk,
                                vol_ref.shape[0], 100.0 * nk / vol_ref.shape[0], n_here)
            logger.info("  %s: %d seeded ROIs from %d labels, %d edges refined from the "
                        "image", tag, len(inserts),
                        len({i["label"] for i in inserts}), n_ref)
            for i in inserts:
                if not i["refined"]:
                    logger.warning("      label %s ROI at (%.0f,%.0f): too little contrast "
                                   "to refine, kept the drawn outline (r=%.1f mm)",
                                   i["label"], i["cy"], i["cx"], i["radius_mm"])
    if not inserts:
        inserts = iq.detect_inserts(flat, body, pixel_mm,
                                    contrast_mad_k=cfg.insert_mad_k)

    # ---- one geometry for every family -----------------------------------------
    # Applied BEFORE the noise region and the NPS patches are built, because both exclude
    # the inserts by radius: a radius that collapsed in a noisy image would leave insert
    # edges inside the "background" and count their contrast as noise.
    if consensus and ref_img is not None:
        for ins in inserts:
            xy = _insert_xy_mm(ins, ref_img)
            d = [float(np.hypot(*(xy - e["xy_mm"]))) for e in consensus]
            if not d:
                continue
            k = int(np.argmin(d))
            if d[k] > 6.0:
                continue
            ins["own_radius_mm"] = ins["radius_mm"]      # kept: it IS a quality measure
            ins["radius_mm"] = float(consensus[k]["radius_mm"])
            ins["radius_px"] = ins["radius_mm"] / pixel_mm
            ins["radius_source"] = consensus[k]["source"]
            # Which PHYSICAL insert this is.  Required downstream: the ROI list is sorted
            # by refined centre, and refinement moves centres by different amounts in each
            # family, so list position is NOT the same insert across families -- pairing
            # by index compared different cylinders and made contrast fidelity nonsense.
            ins["geom_id"] = int(k)
        shrunk = [i for i in inserts
                  if i.get("own_radius_mm") and i["own_radius_mm"] < 0.7 * i["radius_mm"]]
        logger.info("  %s: using the shared insert geometry; %d of %d edges had "
                    "collapsed here by >30%% and were restored", tag, len(shrunk),
                    len(inserts))

    # ---- where the noise is measured -------------------------------------------
    # The body outline is NOT a noise region on an anthropomorphic phantom: lung, bone,
    # mediastinum and the table all lie inside it, and a patch on any of them measures
    # anatomy.  Preference order: a drawn background segment, else the automatically
    # detected homogeneous region, and only as a last resort the whole body.
    if bg_region is not None:
        region, region_src, edge_mm = bg_region, "drawn background label", 2.0
    else:
        region = iq.homogeneous_mask(flat, body, pixel_mm)
        lab, nlab = ndimage.label(region)
        if nlab > 1:                       # a fragmented mask cannot hold square patches
            sizes = ndimage.sum(region, lab, range(1, nlab + 1))
            region = lab == (1 + int(np.argmax(sizes)))
        region_src, edge_mm = "auto homogeneous region", 2.0
        if region.sum() < 0.02 * max(body.sum(), 1):
            logger.warning("  %s: the homogeneous region is only %d px; falling back to "
                           "the whole body, which on a non-uniform phantom measures "
                           "anatomy rather than noise", tag, int(region.sum()))
            region, region_src, edge_mm = body, "body outline (UNRELIABLE)", 10.0

    if force_patch_px:
        # Every family must share one patch size: the NPS frequency grid depends on it,
        # so curves measured at different sizes are not comparable.
        patches = iq.background_patches(region, inserts, pixel_mm, patch_px=force_patch_px,
                                        edge_margin_mm=edge_mm)
        patch_px, tried = force_patch_px, [(force_patch_px, len(patches))]
    else:
        patches, patch_px, tried = iq.auto_background_patches(
            region, inserts, pixel_mm, patch_px=cfg.nps_patch_px, edge_margin_mm=edge_mm)
        if patch_px != cfg.nps_patch_px:
            logger.info("  %s: %d px patches did not fit; using %d px (%.1f mm). Tried %s",
                        tag, cfg.nps_patch_px, patch_px, patch_px * pixel_mm,
                        ", ".join(f"{s}px->{n}" for s, n in tried))
    # Whatever the region came from, the inserts and the edge must come OUT of it before
    # anything is measured -- otherwise insert contrast is counted as noise.
    region = iq.noise_region(region, inserts, pixel_mm, edge_margin_mm=edge_mm)
    logger.info("  %s: noise region = %s (%d px, %.0f mm2 after removing inserts+edge)",
                tag, region_src, int(region.sum()), region.sum() * pixel_mm ** 2)
    if region.sum() < 200:
        logger.warning("  %s: the noise region is only %d px -- the noise SD will be very "
                       "uncertain. Draw a background segment and pass --background-label.",
                       tag, int(region.sum()))
    if len(patches) < 8 or patch_px < 24:
        logger.warning("  %s: only %d patches of %d px (%.1f mm) -- the noise SD is still "
                       "usable but the NPS CURVE is coarse (%d frequency bins to Nyquist) "
                       "and f_av is unreliable. Draw a background segment in a uniform "
                       "part of the phantom and pass --background-label to fix this.",
                       tag, len(patches), patch_px, patch_px * pixel_mm, patch_px // 2)

    body_px = int(body.sum())
    logger.info("  %s: body %d px (%.0f mm2), %d inserts [%s], %d NPS patches of "
                "%d px (%.1f mm)", tag, body_px, body_px * pixel_mm ** 2, len(inserts),
                source, len(patches), patch_px, patch_px * pixel_mm)
    for i, ins in enumerate(inserts):
        logger.info("      insert %d: r=%.1f mm  %+.0f HU vs background  circularity %.2f",
                    i, ins["radius_mm"], ins["contrast_hu"], ins["circularity"])
    if not inserts:
        logger.warning("  %s: no inserts detected -- TTF/NEQ/d' unavailable, and the "
                       "background is not insert-excluded. Check qc/roi_%s.nrrd", tag, tag)
    elif cfg.expect_inserts and len(inserts) != cfg.expect_inserts:
        logger.warning("  %s: found %d inserts but %d were expected. THE NOISE NUMBERS "
                       "FOR THIS FAMILY ARE NOT VALID: every missing insert stays inside "
                       "the background region, inflating the noise SD and dragging f_av "
                       "down. Supply --segmentation, or lower --insert-mad-k, then "
                       "confirm on qc/roi_%s.nrrd before using any of it.",
                       tag, len(inserts), cfg.expect_inserts, tag)

    # Written BEFORE any failure: when detection goes wrong the overlay is the only way
    # to see why, so it must exist even on the unhappy path.
    _qc_roi_volume(Path(qc_dir) / f"roi_{tag}.nrrd", vol_ref.shape[0], flat.shape, ref_img,
                   body, inserts, patches, patch_px, tag, source,
                   region=region, region_src=region_src)

    if not patches:
        logger.warning(
            "  %s: no square patch fits the noise region, so the NPS curve, f_av, NEQ and "
            "d' are unavailable for this family. The noise SD is still measured over the "
            "whole region and remains valid. Tried %s", tag,
            ", ".join(f"{s}px({s * pixel_mm:.0f}mm)->{n}" for s, n in tried))
    return body, inserts, patches, patch_px, region


def _qc_roi_volume(path, n_slices, shape_yx, ref_img, body, inserts, patches, patch_px,
                   tag, source, region=None, region_src=""):
    """
    Write the ROI placement as a 3-D LABEL VOLUME, loadable straight into Slicer on top
    of the data, instead of a single-slice picture.

    A mid-slice PNG can only ever show one plane, and the thing that has to be checked --
    do the circles sit on the inserts, does any NPS patch touch one -- is easier to judge
    while scrolling through the actual volume.  The labels are replicated through z
    because the ROIs really are applied to every slice of the slab (the NPS is measured
    on all of them), so the volume is a faithful picture of what was measured, not a
    stylised one.

    Label values (also written to the .json sidecar):
        1              NPS background patches -- the noise was measured HERE
        2 + i          insert i, at its measured radius
    The body is not painted; it is plainly visible in the underlying image.
    """
    sitk = _sitk()
    ny, nx = shape_yx
    plane = np.zeros((ny, nx), dtype=np.uint16)
    # 1 = the whole region accepted as uniform (what the noise SD is measured over),
    # 2 = the square blocks inside it that the NPS could actually use.  Showing only the
    # blocks hides the more important question -- was the right MATERIAL accepted at all.
    if region is not None:
        plane[np.asarray(region, dtype=bool)] = 1
    for (y0, x0) in patches:
        plane[y0:y0 + patch_px, x0:x0 + patch_px] = 2
    yy, xx = np.ogrid[:ny, :nx]
    legend = {"1": "noise region (SD measured here)",
              "2": "NPS patches (FFT blocks inside the region)"}
    nz = max(1, int(n_slices))
    vol = np.repeat(plane[None], nz, axis=0)
    # Inserts are painted ONLY on the slices they occupy.  A layered phantom has inserts
    # that exist over part of the slab, and replicating them through z would show ROIs on
    # slices where the insert is not -- which is exactly the mismatch that has to be
    # visible here.  Scrolling should make each layer appear and vanish.
    for i, ins in enumerate(inserts):
        disc = ((yy - ins["cy"]) ** 2 + (xx - ins["cx"]) ** 2) <= ins["radius_px"] ** 2
        k0 = int(ins.get("k_lo", 0) or 0)
        k1 = int(ins["k_hi"]) if ins.get("k_hi") is not None else nz - 1
        k0, k1 = max(0, min(k0, nz - 1)), max(0, min(k1, nz - 1))
        # inserts win over region/patches, so any overlap is immediately visible
        vol[k0:k1 + 1, disc] = 10 + i
        legend[str(10 + i)] = (f"insert {i}  r={ins['radius_mm']:.2f} mm  "
                               f"{ins.get('contrast_hu', float('nan')):+.0f} HU  "
                               f"slices {k0}-{k1}"
                               + ("  [refined]" if ins.get("refined") else ""))

    img = sitk.GetImageFromArray(vol)
    if ref_img is not None:          # carry the geometry so it overlays correctly
        img.SetSpacing(ref_img.GetSpacing())
        img.SetOrigin(ref_img.GetOrigin())
        img.SetDirection(ref_img.GetDirection())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))

    _write_json(Path(path).with_suffix(".json"), {
        "tag": tag, "roi_source": source, "patch_px": int(patch_px),
        "n_inserts": len(inserts), "n_patches": len(patches),
        "region_source": region_src,
        "region_px": int(np.asarray(region).sum()) if region is not None else 0,
        "labels": legend,
        "inserts": [{k: v for k, v in ins.items() if k != "area_px"} for ins in inserts],
    })


def stage2_metrics(cfg: CompareConfig):
    """Measure every family and write metrics + figures."""
    sitk = _sitk()
    out = Path(cfg.out_root)
    match = _load_match(out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    qc = out / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    if not cfg.segmentation and match.get("segmentation"):
        cfg.segmentation = match["segmentation"]
        logger.info("segmentation carried over from stage 0: %s", cfg.segmentation)
    if cfg.background_label is None and match.get("background_label"):
        cfg.background_label = match["background_label"]

    zr = (match["z_lo_mm"], match["z_hi_mm"])
    results = {"match_config": match, "own": {}, "wfbp": {}, "vmi": {}}

    # ═══ pass 1: reference volume + insert seeds for every family ═══════════════
    # The shared insert geometry has to be agreed before ANY family builds its noise
    # region, because the region excludes the inserts by radius.
    refs, chans_by_kind = {}, {}
    own_variants = [v["name"] for v in cfg.variants
                    if (out / "sweep" / v["name"] / "reconstruction_thr_A_HU.nii.gz").exists()]
    for v in cfg.variants:
        if v["name"] not in own_variants:
            logger.warning("variant %s missing -- run stage 1 first", v["name"])
    if own_variants:
        p = out / "sweep" / own_variants[0] / "reconstruction_thr_A_HU.nii.gz"
        img_own = sitk.ReadImage(str(p))
        refs["own"] = (sitk.GetArrayFromImage(img_own).astype(np.float32), img_own,
                       float(match["pixel_mm"]))
    for kind, folder in (("wfbp", match.get("wfbp_dir")), ("vmi", match.get("vmi_dir"))):
        if not folder or kind not in match["families"]:
            continue
        chans = select_family(index_dicom_series(folder, cache_dir=out), kind)
        chans_by_kind[kind] = chans
        vol, img, _z = read_series_volume(chans[0], z_range=zr)
        refs[kind] = (vol, img, float(img.GetSpacing()[0]))

    # Mid-slice only: the pose check needs one plane, and holding the full volumes past
    # this point would double stage 2's memory for nothing.
    _pose_refs = {t: (v[0][v[0].shape[0] // 2][None], v[1], v[2]) for t, v in refs.items()}

    seeds = {}
    for tag, (vol_ref, img_ref, px) in refs.items():
        seeds[tag] = (family_insert_seeds(vol_ref, px, cfg, img_ref), img_ref)
    consensus = consensus_insert_geometry({t: s for t, s in seeds.items() if s[0]})
    if consensus:
        logger.info("shared insert geometry over %d inserts (radius taken from the "
                    "family that resolved each edge best):", len(consensus))
        for e in consensus:
            per = "  ".join(f"{t}={v['radius_mm']:.2f}" for t, v in sorted(e["per_family"].items()))
            logger.info("    (%+7.1f,%+7.1f) mm  r=%.2f mm from %-5s |  %s",
                        e["xy_mm"][0], e["xy_mm"][1], e["radius_mm"], e["source"], per)
        results["insert_geometry"] = [
            {"x_mm": float(e["xy_mm"][0]), "y_mm": float(e["xy_mm"][1]),
             "radius_mm": e["radius_mm"], "source": e["source"],
             "per_family": e["per_family"]} for e in consensus]

    # ═══ pass 2: ROIs on the shared geometry, then measure ═════════════════════
    ctxs, shared_patch_px = {}, None
    for tag in ("own", "wfbp", "vmi"):
        if tag not in refs:
            continue
        vol_ref, img_ref, px = refs[tag]
        ctxs[tag] = _family_rois(vol_ref, px, cfg, tag, qc,
                                 force_patch_px=shared_patch_px, ref_img=img_ref,
                                 consensus=consensus)
        if shared_patch_px is None:
            shared_patch_px = ctxs[tag][3]
    refs.clear()
    gc.collect()

    results["pose"] = _check_pose_against_reference(
        {t: (v[0], v[1], v[2]) for t, v in _pose_refs.items()})

    ttf_geom_id, ttf_src, ttf_c = _pick_ttf_insert({t: c[1] for t, c in ctxs.items()})
    if ttf_geom_id is not None:
        logger.info("TTF measured on ONE agreed insert for every family: geom_id %d "
                    "(chosen on %s, |contrast| %.0f HU)", ttf_geom_id, ttf_src, ttf_c)
        results["ttf_insert"] = {"geom_id": ttf_geom_id, "chosen_on": ttf_src,
                                 "contrast_hu": float(ttf_c)}
    else:
        logger.warning("no insert is big enough to carry a TTF edge -- each family will "
                       "fall back to its own highest-contrast ROI, which makes the TTF "
                       "numbers NOT directly comparable between families")

    # ---- own: every swept variant ------------------------------------
    own_ctx = ctxs.get("own")
    for vname in own_variants:
        vdir = out / "sweep" / vname
        results["own"][vname] = {}
        for label in THRESHOLD_LABELS:
            p = vdir / f"reconstruction_thr_{label}_HU.nii.gz"
            if not p.exists():
                continue
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
            body, inserts, patches, patch_px, region = own_ctx
            logger.info("own/%s thr %s", vname, label)
            results["own"][vname][label] = _measure_volume(
                vol, float(match["pixel_mm"]), inserts, cfg, body, patches, patch_px,
                region, ttf_geom_id=ttf_geom_id)
            del vol
            gc.collect()

    # ---- Siemens families --------------------------------------------
    siemens_vols = {}
    for kind, chans in chans_by_kind.items():
        if kind not in ctxs:
            continue
        body, inserts, patches, patch_px, region = ctxs[kind]
        for c in chans:
            vol, img, _z = read_series_volume(c, z_range=zr)
            pixel_mm = float(img.GetSpacing()[0])
            logger.info("%s %s", kind, c["label"])
            results[kind][c["label"]] = _measure_volume(vol, pixel_mm, inserts, cfg,
                                                        body, patches, patch_px,
                                                        region, ttf_geom_id=ttf_geom_id)
            results[kind][c["label"]]["pixel_mm"] = pixel_mm
            results[kind][c["label"]]["kernel"] = c["kernel"]
            if kind == "wfbp":
                siemens_vols[c["label"]] = vol
            else:
                del vol
            gc.collect()

    _contrast_fidelity(results)

    # ---- paired difference: own vs WFBP, 1:1 by threshold ------------
    # VMI is excluded here because of the HU-scale mismatch, not its quality.
    if siemens_vols and own_ctx is not None:
        results["bias"] = _bias_analysis(cfg, match, siemens_vols, own_ctx)
        # Computed here rather than in the figure: metrics.json is written before
        # _figures runs, and these are numbers the text quotes, not decoration.
        results["bland_altman"] = _ba_summary(results["bias"])
    siemens_vols.clear()
    gc.collect()

    _write_json(out / "metrics" / "metrics.json", results)
    # ROIs for the orientation QC overlay; kept out of metrics.json (leading underscore).
    results["_rois"] = {t: [{"cy": i["cy"], "cx": i["cx"], "radius_px": i["radius_px"]}
                            for i in c[1]] for t, c in ctxs.items()}
    _figures(cfg, out, results, match)
    results.pop("_rois", None)
    _print_summary(results)
    logger.info("wrote %s", out / "metrics" / "metrics.json")
    return results


def _similarity(a, b, body):
    """
    Global similarity of one of our volumes against the vendor's, inside the phantom.

    Reported TWICE, raw and after the mean difference is removed, because the two answer
    different questions and only the second is about the images. The channel offset of
    -73.9 to -18.1 HU is large against the structure being compared, so a single PSNR
    would be a restatement of that offset: at threshold A it contributes ~5500 HU^2 to a
    mean squared error whose noise part is a few hundred. Subtracting the mean leaves what
    the offset does not explain.

    MAX is the Hounsfield RANGE of the reference inside the mask, not a display constant
    such as 4096. A PSNR is only comparable against another PSNR that used the same MAX,
    so the value it was computed with is returned alongside it rather than left implicit.

    SSIM is computed per slice over the mask's bounding box and averaged. A 3-D SSIM with
    an anisotropic voxel would weight the through-plane direction differently from the
    in-plane one, which is not a property of the images being compared.
    """
    m = np.asarray(body, bool)
    if m.ndim == 2:
        m = np.broadcast_to(m, a.shape)
    sel = m & np.isfinite(a) & np.isfinite(b)
    if int(sel.sum()) < 64:
        return None
    x, y = a[sel].astype(np.float64), b[sel].astype(np.float64)
    vmax = float(np.max(y) - np.min(y))
    if not np.isfinite(vmax) or vmax <= 0:
        return None
    d = x - y
    bias = float(np.mean(d))
    mse_raw = float(np.mean(d ** 2))
    mse_bc = float(np.mean((d - bias) ** 2))

    def psnr(mse):
        return 10.0 * np.log10(vmax ** 2 / mse) if mse > 0 else float("inf")

    out = {"max_hu": vmax, "bias_hu": bias, "rmse_hu": float(np.sqrt(mse_raw)),
           "rmse_debiased_hu": float(np.sqrt(mse_bc)),
           "psnr_db": psnr(mse_raw), "psnr_debiased_db": psnr(mse_bc)}
    try:
        from skimage.metrics import structural_similarity as _ssim
        zz, yy, xx = np.where(m)
        sl = (slice(zz.min(), zz.max() + 1), slice(yy.min(), yy.max() + 1),
              slice(xx.min(), xx.max() + 1))
        vals = [_ssim(y_[1], y_[0], data_range=vmax)
                for y_ in zip(b[sl].astype(np.float64), a[sl].astype(np.float64))]
        out["ssim"] = float(np.mean(vals)) if vals else float("nan")
    except Exception as exc:                                    # noqa: BLE001
        logger.info("SSIM unavailable (%s)", exc)
    return out


def _linfit(xs, ys):
    """
    Ordinary least-squares fit of ys on xs with the standard error of the slope.

    Written out rather than imported because the closed form is shorter than the guard an
    optional scipy import would need, and this path runs inside the figure code where a
    missing dependency must not be fatal.  Returns None for a degenerate panel -- fewer
    than three points or a constant x -- which a figure should skip rather than raise on.
    """
    x = np.asarray(list(xs), float)
    y = np.asarray(list(ys), float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 3:
        return None
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx <= 0.0:
        return None
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    dof = n - 2
    se = float(np.sqrt((sse / dof) / sxx)) if dof > 0 else float("nan")
    # t rather than 1.96: n is ~19 per variant, where the normal quantile understates the
    # interval by about 8 %.  Falls back to the normal value if scipy is unavailable, and
    # says which was used so the interval is never read as tighter than it is.
    try:
        from scipy import stats as _st
        tcrit, tdist = float(_st.t.ppf(0.975, dof)), "t"
    except Exception:
        tcrit, tdist = 1.96, "normal"
    return {"slope": slope, "intercept": intercept, "slope_se": se,
            "slope_ci95": [slope - tcrit * se, slope + tcrit * se],
            "t": slope / se if se > 0 else float("nan"),
            "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
            "n": n, "quantile": tdist}


def _ba_summary(bias):
    """
    Bland-Altman statistics per threshold: bias, limits of agreement, and the SLOPE of the
    difference against the mean of the two measurements.

    The slope is reported PER VARIANT as well as pooled, because only the per-variant
    figure is interpretable.  Each variant delivers its own fraction of the vendor's
    insert contrast (`contrast_fidelity`), so its difference against the vendor grows with
    contrast at its own rate: for a variant of fidelity c the difference is (c-1)T and the
    Bland-Altman abscissa is the MEAN of the two measurements, ((c+1)/2)T, so the expected
    slope is 2(c-1)/(c+1) -- not c-1, which is the slope against the vendor value alone.
    A variant at c = 0.20 therefore sits near -1.33 and one at c = 0.62 near -0.47, and a
    measured slope away from that is disagreement the contrast does not explain.
    Pooling five such populations averages five different slopes
    and describes none of them -- the same mistake that pooling the four channels made for
    the limits of agreement, which is why those are already computed per panel.  The
    pooled slope is kept only so that it sits beside the pooled bias and limits the text
    already quotes.
    """
    out = {}
    for label in THRESHOLD_LABELS:
        per_variant, px, py = {}, [], []
        for vname, per_thr in sorted(bias.items()):
            d = (per_thr or {}).get(label)
            if not d or not d.get("roi_pairs"):
                continue
            xs = [p["mean_hu"] for p in d["roi_pairs"]]
            ys = [p["diff_hu"] for p in d["roi_pairs"]]
            fit = _linfit(xs, ys)
            if fit:
                per_variant[vname] = fit
            px += list(xs)
            py += list(ys)
        if not py:
            continue
        mu, sg = float(np.mean(py)), float(np.std(py))
        out[label] = {"bias_hu": mu, "loa95_hu": 1.96 * sg, "n": len(py),
                      "pooled_fit": _linfit(px, py), "per_variant": per_variant}
    return out


def _bias_analysis(cfg, match, wfbp_vols, own_ctx):
    """Systematic (low-frequency) difference between our recon and WFBP, per threshold."""
    sitk = _sitk()
    out = Path(cfg.out_root)
    body, inserts, _patches, _patch_px, _region = own_ctx
    pixel_mm = match["pixel_mm"]
    bias = {}
    # T1..T4 are the vendor's names for the same cumulative thresholds as our A..D.
    pairs = list(zip(THRESHOLD_LABELS, [f"T{i}" for i in (1, 2, 3, 4)]))
    for v in cfg.variants:
        vdir = out / "sweep" / v["name"]
        if not (vdir / "reconstruction_thr_A_HU.nii.gz").exists():
            continue
        bias[v["name"]] = {}
        for own_lbl, wf_lbl in pairs:
            p = vdir / f"reconstruction_thr_{own_lbl}_HU.nii.gz"
            if not p.exists() or wf_lbl not in wfbp_vols:
                continue
            a = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
            b = wfbp_vols[wf_lbl]
            n = min(a.shape[0], b.shape[0])
            if a.shape[1:] != b.shape[1:]:
                logger.warning("bias %s %s: in-plane shape %s vs %s -- skipped "
                               "(stage 1 grid did not match the vendor's)",
                               v["name"], own_lbl, a.shape[1:], b.shape[1:])
                del a
                continue
            d = iq.difference_analysis(a[:n], b[:n], pixel_mm, body,
                                       smooth_mm=cfg.bias_smooth_mm)
            d.pop("systematic_map", None)
            # Computed here rather than in a pass of its own: both volumes are already
            # resident and reading them again is the only expensive part of this metric.
            sim = _similarity(a[:n], b[:n], body)
            if sim:
                d["similarity"] = sim
            # Bland-Altman inputs: paired ROI means
            sa = iq.roi_statistics(a[:n], inserts, body, pixel_mm)
            sb = iq.roi_statistics(b[:n], inserts, body, pixel_mm)
            d["roi_pairs"] = [
                {"roi": i, "own_hu": x["mean_hu"], "wfbp_hu": y["mean_hu"],
                 "diff_hu": x["mean_hu"] - y["mean_hu"],
                 "mean_hu": 0.5 * (x["mean_hu"] + y["mean_hu"])}
                for i, (x, y) in enumerate(zip(sa["inserts"], sb["inserts"]))]
            d["roi_pairs"].append(
                {"roi": "background", "own_hu": sa["background"]["mean_hu"],
                 "wfbp_hu": sb["background"]["mean_hu"],
                 "diff_hu": sa["background"]["mean_hu"] - sb["background"]["mean_hu"],
                 "mean_hu": 0.5 * (sa["background"]["mean_hu"] + sb["background"]["mean_hu"])})
            bias[v["name"]][own_lbl] = d
            del a
            gc.collect()
    return bias


# ═════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════
def _share_axes(axes, logy=False, free=False):
    """Put a row of panels on one scale so they can be compared without rescaling."""
    if free:
        return
    axes = [a for a in axes if a.has_data()]
    if len(axes) < 2:
        return
    xs = [a.get_xlim() for a in axes]
    ys = [a.get_ylim() for a in axes]
    lo_y = min(y[0] for y in ys)
    if logy:
        pos = [v for a in axes for line in a.get_lines() for v in line.get_ydata()
               if np.isfinite(v) and v > 0]
        pos += [v for a in axes for c in a.collections
                for v in np.asarray(c.get_offsets())[:, 1:].ravel()
                if np.isfinite(v) and v > 0]
        lo_y = min(pos) * 0.7 if pos else max(lo_y, 1e-3)
    for a in axes:
        if logy:
            a.set_yscale("log")
        a.set_xlim(min(x[0] for x in xs), max(x[1] for x in xs))
        a.set_ylim(lo_y, max(y[1] for y in ys))


_CAPTIONS = {
    "tradeoff_per_threshold.png": (
        "Noise-resolution trade-off per energy channel",
        "Background noise SD against TTF50 for each reconstruction variant (blue, joined "
        "in order of increasing TTF50) and the Siemens WFBP reference (red star), one "
        "panel per threshold. Down and to the right is better; a variant that moves left "
        "and down along the blue curve has only been smoothed, which is a move ALONG the "
        "trade-off rather than an improvement of it. The result is the vendor's position "
        "relative to our curve. All panels share one scale and the noise axis is "
        "logarithmic, so a factor of two is the same distance in every panel and the gap "
        "can be followed from threshold A to D. Thresholds are never combined on one "
        "axes: A collects all the photons and D very few, so a pooled plot would largely "
        "display the bins' photon statistics rather than the reconstruction differences."),
    "nps_curves.png": (
        "Noise power spectra",
        "Radially averaged NPS per threshold, normalised so the integral equals the "
        "variance. Two images with the same noise SD can look entirely different, and "
        "the difference lives here: the height of the curve is the noise magnitude, its "
        "position along the frequency axis is the noise texture. A leftward shift means "
        "coarser, blotchier noise, the signature of iterative reconstruction or "
        "denoising; fine-grained analytic noise sits further right."),
    "vmi_vs_kev.png": (
        "Siemens VMI family",
        "Noise magnitude, noise texture (f_av) and detectability index d' across the "
        "monoenergetic series. Reported separately from the threshold comparison because "
        "no VMI channel corresponds to a cumulative threshold image, so the two cannot be "
        "placed on a common axis. d' is task-dependent and the task is stated in the "
        "axis label."),
    "bland_altman_own_vs_wfbp.png": (
        "Bland-Altman: ROI-mean HU, ours vs WFBP",
        "Paired ROI-mean agreement between our reconstruction and the vendor's, one "
        "panel per threshold, with bias and 95 % limits of agreement computed within "
        "each panel. The channels differ in noise by an order of magnitude, so a single "
        "pooled set of limits would describe none of them. A vertical offset of the "
        "cloud is an HU-scale difference; a slope is a contrast-dependent one; the "
        "scatter is dominated by the two independent noise realisations and is "
        "characterised properly by the NPS, not here. The line through each variant's "
        "points is its own regression of difference on mean, and the corner box gives "
        "the range of those slopes: a variant delivering the fraction c of the vendor's "
        "insert contrast must produce a slope near 2(c-1)/(c+1), since the abscissa is "
        "the mean of the two measurements, so the spread across variants is the contrast "
        "column of the metrics table seen from the other side."),
    "noise_ordering.png": (
        "Background noise by energy channel",
        "Noise SD against energy channel for every swept variant and for the vendor "
        "reconstruction, on one logarithmic axis. The four channels are deliberately "
        "combined here, unlike in the trade-off figure: the finding is the SHAPE of each "
        "curve across A to D. Ours fall from the lowest threshold to the highest and the "
        "vendor's rises, so the two families order their channels in opposite directions "
        "on data from the same acquisition. Photon statistics alone cannot produce that, "
        "since both families see the same photons; the ordering is a property of the "
        "processing. Absolute heights are not comparable between the two families in the "
        "way the shapes are, because kernel and QIR cannot be matched."),
    "modality_comparison.png": (
        "Threshold A: our reconstruction and the Siemens products",
        "One slice and one window for all three, with the insert module zoomed beneath and "
        "the zoom box drawn above it. The variant shown is the one whose background noise "
        "matches the vendor's, because a visual comparison has to fix something and noise "
        "is the honest thing to fix: our smoothest variant beside the vendor would be a "
        "picture of the smoothing, and our sharpest a picture of its noise. At matched "
        "noise the panels differ in the quantity the trade-off curve reports numerically -- "
        "how sharp the image is for that noise. The window is stated rather than "
        "autoscaled per panel, which would rescale a noisy reconstruction into looking as "
        "clean as a smooth one."),
    "insert_localisation.png": (
        "Insert localisation",
        "Left: the refined radius each modality recovers for every insert, against the "
        "shared reference geometry. Right: the absolute radius error against the "
        "contrast-to-noise ratio the modality actually achieved on that insert. The "
        "recovered radius is a detectability statement in its own right, because in a "
        "low-CNR image the edge cannot be located and the fitted radius collapses toward "
        "the size of a noise blob."),
}


def _write_captions(fig_dir):
    """
    Draft captions, kept out of the figures themselves.

    Titles name what is plotted and nothing else -- a title that explains how to read the
    plot, or states its conclusion, is a caption that has wandered into the image, and it
    cannot be edited when the figure is placed in the thesis. The explanatory text lives
    here instead, one entry per figure, ready to be adapted into a LaTeX caption.
    """
    lines = ["# Figure captions", "",
             "Draft text for the thesis. The figures carry short titles naming the "
             "quantity; everything about how to read them belongs in the caption.", ""]
    panel_dir = fig_dir / "panels"
    panels = sorted(p.name for p in panel_dir.glob("*.png")) if panel_dir.is_dir() else []
    if panels:
        lines += [
            "## Single panels (`panels/`)", "",
            "Each panel of a multi-panel figure is also written on its own, cropped from "
            "the same rendered figure so the shared axis limits are identical -- panels "
            "stitched back together stay comparable. Place them at a readable width and "
            "include `<stem>_legend.png` once beneath the group:", "",
            "```latex",
            "\\begin{figure}[t]\\centering",
            "  \\includegraphics[width=.49\\textwidth]{panels/tradeoff_A}\\hfill",
            "  \\includegraphics[width=.49\\textwidth]{panels/tradeoff_B}\\\\[2pt]",
            "  \\includegraphics[width=.49\\textwidth]{panels/tradeoff_C}\\hfill",
            "  \\includegraphics[width=.49\\textwidth]{panels/tradeoff_D}\\\\[4pt]",
            "  \\includegraphics[width=.6\\textwidth]{panels/tradeoff_legend}",
            "  \\caption{...}",
            "\\end{figure}",
            "```", "",
            "Available: " + ", ".join(f"`{n}`" for n in panels), "",
        ]
    for name, (title, body) in _CAPTIONS.items():
        if (fig_dir / name).exists():
            lines += [f"## `{name}`", "", f"**{title}.** {body}", ""]
    (fig_dir / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def _save_panels(fig, axes, fig_dir, stem, labels, ylabel=None, dpi=200, pad=0.06):
    """
    Also write each panel of a multi-panel figure as its own file, under figures/panels/.

    A four-panel figure placed at \\textwidth gives each panel a quarter of the page, and
    its fonts shrink with it -- legible on screen, not on paper.  The same panel placed at
    0.5\\textwidth is twice the size for the same font, so the fix is to let the thesis
    stitch the panels itself.

    The panels are CROPPED OUT OF THE RENDERED FIGURE rather than redrawn, which is what
    keeps the shared axis limits _share_axes() established: four separately-drawn panels
    would each autoscale, and stitching them back together would silently produce a figure
    whose panels are no longer comparable -- the exact failure the shared scale exists to
    prevent.

    The legend lives under the whole figure, so it is written once as `<stem>_legend.png`
    for the thesis to place beneath the assembled panels.  `ylabel` is applied to every
    panel first: only the leftmost carries one in the combined figure, and a standalone
    panel C with an unlabelled y axis is not a figure.

    MUTATES the figure (labels, tick visibility), so call it after the combined figure has
    been saved and before the figure is closed.
    """
    panel_dir = Path(fig_dir) / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    axl = [a for a in np.atleast_1d(axes).ravel()]
    for a in axl:
        if ylabel is not None:
            a.set_ylabel(ylabel)
        # sharex/sharey suppress the tick LABELS on every panel but the first, and a
        # standalone panel with unnumbered axes is not readable. Restored here rather
        # than in the combined figure, where the suppression is right.
        a.tick_params(labelleft=True, labelbottom=True)
    fig.canvas.draw()                       # tight bboxes need a rendered figure
    try:
        rend = fig.canvas.get_renderer()
    except AttributeError:                  # non-Agg backend
        rend = None

    def _crop(keep, path):
        """Save the region of `keep`, with every other artist hidden."""
        # bbox_inches saves a RECTANGLE of the figure, so a neighbouring panel that
        # happens to lie inside it is drawn too: the first attempt produced a threshold-C
        # panel carrying threshold D's y-label and a stray vendor star from B.  Hiding
        # everything else is what makes the crop actually contain one panel.
        others = [a for a in axl if a is not keep]
        others += [l for l in fig.legends if l is not keep]
        # The suptitle is a FIGURE-level artist, not an axes and not a legend, so it was
        # not being hidden: a crop box that reaches into the title band carried a slice of
        # the title into the panel. Same for any other figure-level text.
        others += [t for t in fig.texts if t is not keep]
        if getattr(fig, "_suptitle", None) is not None and fig._suptitle is not keep:
            others.append(fig._suptitle)
        state = [o.get_visible() for o in others]
        for o in others:
            o.set_visible(False)
        try:
            bb = keep.get_tightbbox(rend)
            if bb is None:
                return False
            # ABSOLUTE margin, not a multiplicative one. expanded() scales about the
            # centre, so a wide, short artist -- a one-row legend is 1389 x 103 px --
            # gained about two pixels at the top and had its first row of glyphs sliced
            # off. A fixed border is the same on every crop whatever its shape.
            bb = bb.transformed(fig.dpi_scale_trans.inverted())
            bb = bb.expanded(1.0, 1.0).padded(pad) if hasattr(bb, "padded") else \
                bb.expanded(1.0 + 2 * pad / max(bb.width, 1e-6),
                            1.0 + 2 * pad / max(bb.height, 1e-6))
            fig.savefig(path, dpi=dpi, bbox_inches=bb)
            return True
        except (AttributeError, TypeError, ValueError):
            return False
        finally:
            for o, s in zip(others, state):
                o.set_visible(s)

    written = []
    for a, lab in zip(axl, list(labels) + [None] * len(axl)):
        if lab is None or not a.has_data():
            continue
        p = panel_dir / f"{stem}_{lab}.png"
        if _crop(a, p):
            written.append(p.name)
    for leg in fig.legends:
        if _crop(leg, panel_dir / f"{stem}_legend.png"):
            written.append(f"{stem}_legend.png")
            break
    if written:
        logger.info("  panels: %s", ", ".join(written))
    return written


def _legend_below(fig, source=None, ncol=4, fontsize=9, bottom=0.13, top=0.93):
    """
    One de-duplicated legend under the panels instead of inside the first one.

    An in-panel legend sits on top of the data it labels -- on the NPS panels it covered
    the low-frequency end of several curves, which is exactly where a denoised spectrum
    differs from an analytic one.  Placing it below costs a strip of white space and
    nothing else, and one legend for a row of panels is read once rather than four times.
    """
    axes = fig.get_axes() if source is None else np.atleast_1d(source).ravel()
    seen, handles, labels = set(), [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.tight_layout(rect=(0, bottom if handles else 0.0, 1, top))
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(ncol, len(labels)),
                   fontsize=fontsize, frameon=False)


def _insert_localisation_figure(path, res, cfg):
    """
    How well each modality recovers a known insert size -- the failure turned into a
    measurement.

    The refined radius is a detectability statement in its own right: in a low-CNR image
    the edge cannot be located, and the fitted radius collapses toward the noise blob
    size.  Plotting it against the shared geometry shows which modality tracks the truth,
    and plotting the error against insert CNR shows WHY -- the error follows
    contrast-to-noise, not the reconstruction's name.
    """
    plt = _mpl()
    geom = res.get("insert_geometry")
    if plt is None or not geom:
        return
    tags = sorted({t for e in geom for t in e["per_family"]})
    if not tags:
        return
    order = np.argsort([e["radius_mm"] for e in geom])
    ref_r = np.array([geom[i]["radius_mm"] for i in order])
    idx = np.arange(len(order))
    colours = {"own": "tab:blue", "wfbp": "tab:red", "vmi": "tab:green"}

    fig, ax = plt.subplots(1, 2, figsize=fp.panel_figsize(2, frac=0.8, height_in=3.6))
    ax[0].step(np.concatenate([idx, idx[-1:] + 1]) - 0.5,
               np.concatenate([ref_r, ref_r[-1:]]), where="post",
               color="0.4", lw=1.2, label="shared geometry (reference)")
    for t in tags:
        r = [geom[i]["per_family"].get(t, {}).get("radius_mm", np.nan) for i in order]
        ax[0].plot(idx, r, "o", ms=7, color=colours.get(t), label=fp.pretty(t, short=True), alpha=0.85)
    ax[0].set_xlabel("insert (sorted by reference radius)")
    ax[0].set_ylabel("measured radius (mm)")
    ax[0].set_title("Measured insert radius", fontsize=fp.PRINT_FONT_PT)
    ax[0].set_xticks(idx)
    ax[0].grid(alpha=0.3)

    # Localisation error against the contrast-to-noise the modality actually had.
    for t in tags:
        noise = None
        if t == "own":
            first = next(iter(res.get("own", {}).values()), {})
            noise = (first.get("A") or {}).get("noise_sd_hu")
        elif t == "wfbp":
            noise = (res.get("wfbp", {}).get("T1") or {}).get("noise_sd_hu")
        elif t == "vmi":
            vm = res.get("vmi", {})
            noise = (vm.get("70keV") or next(iter(vm.values()), {})).get("noise_sd_hu")
        if not noise:
            continue
        cnr, err = [], []
        for i in order:
            v = geom[i]["per_family"].get(t)
            if not v or not np.isfinite(v["radius_mm"]):
                continue
            cnr.append(abs(v["contrast_hu"]) / noise)
            err.append(abs(v["radius_mm"] - geom[i]["radius_mm"]))
        if cnr:
            ax[1].plot(cnr, err, "o", ms=7, color=colours.get(t), label=fp.pretty(t, short=True), alpha=0.85)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("insert CNR  (|contrast| / background noise SD)")
    # Two lines: at 0.8 textwidth over two panels this label is wider than the panel is
    # tall, and a rotated label longer than its axis is cropped by the panel crop.
    ax[1].set_ylabel("|radius error| (mm)\nagainst the shared geometry")
    ax[1].set_title("Radius error against contrast-to-noise ratio", fontsize=fp.PRINT_FONT_PT)
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Insert localisation", fontsize=fp.PRINT_FONT_PT + 1)
    _legend_below(fig, ax[0], ncol=4)
    fig.savefig(path, dpi=130)
    if cfg.split_figures:
        _save_panels(fig, ax, Path(path).parent, "insert_localisation",
                     ("radius", "error_vs_cnr"))
    plt.close(fig)


def _qc_orientation_figure(path, cfg, match, out, res_rois=None):
    """
    Is our reconstruction only MIRRORED relative to the vendor, or also flipped elsewhere?

    The frame work established a left-right mirror (§9) by matching the body centroid and
    the phantom's principal axis.  Those two quantities cannot distinguish a mirror from a
    mirror-plus-vertical-flip on a body that is roughly symmetric top to bottom, so this
    lays the candidates out side by side and lets the anatomy answer it: the couch, the
    spine and the off-centre module are all strongly asymmetric in y.

    Top row is the reference. Bottom row is our volume under each candidate transform --
    exactly one should match the vendor's anatomy.
    """
    plt = _mpl()
    sitk = _sitk()
    if plt is None:
        return
    z_mid = 0.5 * (match["z_lo_mm"] + match["z_hi_mm"])
    panels = []
    for kind, folder in (("wfbp", match.get("wfbp_dir")), ("vmi", match.get("vmi_dir"))):
        if not folder:
            continue
        try:
            ch = select_family(index_dicom_series(folder, cache_dir=out), kind)
            if not ch:
                continue
            vol, img, zpos = read_series_volume(ch[0], z_range=(z_mid - 3, z_mid + 3))
            k = int(np.argmin(np.abs(np.asarray(zpos) - z_mid)))
            panels.append((f"{kind} (reference)", vol[k]))
        except Exception as exc:                                # noqa: BLE001
            logger.warning("orientation QC: %s unavailable (%s)", kind, exc)

    own_slice = None
    for vname in [v["name"] for v in cfg.variants]:
        p = Path(out) / "sweep" / vname / "reconstruction_thr_A_HU.nii.gz"
        if p.exists():
            im = sitk.ReadImage(str(p))
            arr = sitk.GetArrayFromImage(im).astype(np.float32)
            # _slice_at_z, not local index arithmetic: it carries the z direction cosine,
            # which the inline version dropped. Correct only while every volume is written
            # ascending -- and silently indexing from the wrong end the moment one is not.
            own_slice = arr[min(_slice_at_z(im, z_mid), arr.shape[0] - 1)]
            break
    if own_slice is None or not panels:
        logger.info("orientation QC skipped (need the sweep and at least one vendor family)")
        return

    # As written (the mirror is already applied on write), then each further candidate.
    cands = [("own  as written", own_slice),
             ("own  un-mirrored (x)", own_slice[:, ::-1]),
             ("own  flipped in y", own_slice[::-1, :]),
             ("own  rotated 180 deg", own_slice[::-1, ::-1])]
    n = max(len(panels), len(cands))
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 8.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    # The insert ROIs are drawn on every panel, as in the decomposition QC: an orientation
    # that looks plausible on anatomy alone is settled the moment the circles either do or
    # do not land on the inserts.
    rois = []
    for tag in ("wfbp", "own", "vmi"):
        ctx = (res_rois or {}).get(tag)
        if ctx:
            rois = ctx
            break
    def _draw(ax, sl, lab, with_rois):
        ax.imshow(sl, cmap="gray", vmin=-200, vmax=300)
        ax.set_title(lab, fontsize=fp.PRINT_FONT_PT)
        if with_rois:
            for ins in rois:
                ax.add_patch(plt.Circle((ins["cx"], ins["cy"]), ins["radius_px"],
                                        fill=False, color="tab:red", lw=1.0))
    for i, (lab, sl) in enumerate(panels):
        _draw(axes[0][i], sl, lab, True)
    for i, (lab, sl) in enumerate(cands):
        # ROIs are in the REFERENCE frame, so they only mean anything on the candidate that
        # is meant to be in that frame -- drawing them on a flipped panel would be circular.
        _draw(axes[1][i], sl, lab, lab.endswith("as written"))
    fig.suptitle(f"Orientation QC at z = {z_mid:.1f} mm -- compare the couch, the spine and "
                 f"the off-centre module.\nExactly one bottom panel should match the top "
                 f"row; if it is not 'as written', the reconstruction needs that transform.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _noise_matched_variant(res, label="A"):
    """
    The sweep variant whose background noise is closest to the vendor's, on one threshold.

    A visual comparison has to fix something, and noise is the honest thing to fix: the
    trade-off is between noise and resolution, so showing our SMOOTHEST variant next to the
    vendor would be an image of the smoothing rather than of the reconstruction, and
    showing our sharpest would be an image of its noise.  At matched noise the panels
    differ in the quantity actually being compared -- how sharp the image is for that noise
    -- which is what the trade-off curve says numerically.
    """
    wf = (res.get("wfbp") or {}).get("T1") or {}
    target = wf.get("noise_sd_hu")
    best, best_d = None, np.inf
    for vname, per in (res.get("own") or {}).items():
        sd = (per.get(label) or {}).get("noise_sd_hu")
        if sd is None or not np.isfinite(sd):
            continue
        d = abs(np.log(max(sd, 1e-6) / max(target, 1e-6))) if target else 0.0
        if d < best_d:
            best, best_d = vname, d
    return best, target


def _modality_triptych_figure(path, cfg, out, res, match, split=False):
    """
    (R2) Our reconstruction, Siemens WFBP and Siemens VMI on one slice and one window.

    The reconstruction comparison had no image in it at all: five variants and two vendor
    products were compared entirely through scalars, so a reader could not see what any of
    the numbers were describing.

    Top row is the phantom, bottom row the same field zoomed to the insert module, with the
    zoom box drawn above it.  One window for every panel -- an autoscaled panel would
    rescale a noisy reconstruction into looking like a clean one, which is precisely the
    difference the figure exists to show.  The variant shown is the one whose background
    noise matches the vendor's, so the panels differ in sharpness rather than in smoothing.
    """
    plt = _mpl()
    sitk = _sitk()
    if plt is None or sitk is None:
        return
    z_mid = 0.5 * (match["z_lo_mm"] + match["z_hi_mm"])
    vname, target = _noise_matched_variant(res)
    panels = []
    if vname:
        p = Path(out) / "sweep" / vname / "reconstruction_thr_A_HU.nii.gz"
        if p.exists():
            im = sitk.ReadImage(str(p))
            arr = sitk.GetArrayFromImage(im).astype(np.float32)
            k = min(_slice_at_z(im, z_mid), arr.shape[0] - 1)
            panels.append((fp.pretty(vname), arr[k], im, vname))
            del arr
            gc.collect()
    for kind, folder in (("wfbp", match.get("wfbp_dir")), ("vmi", match.get("vmi_dir"))):
        if not folder:
            continue
        try:
            ch = select_family(index_dicom_series(folder, cache_dir=Path(out)), kind)
            if not ch:
                continue
            vol, img, zpos = read_series_volume(ch[0], z_range=(z_mid - 3, z_mid + 3))
            k = int(np.argmin(np.abs(np.asarray(zpos) - z_mid)))
            panels.append((fp.pretty(kind), vol[k], img, kind))
        except Exception as exc:                                # noqa: BLE001
            logger.warning("triptych: %s unavailable (%s)", kind, exc)
    if len(panels) < 2:
        logger.info("modality triptych skipped: need our sweep and at least one vendor set")
        return

    geom = res.get("insert_geometry") or []
    vmin, vmax = HU_WINDOW_RECON
    fig, axes = plt.subplots(2, len(panels), squeeze=False,
                             figsize=(fp.panel_figsize(1, frac=0.49)[0] * len(panels),
                                      7.1))
    for ci, (title, sl, img, _key) in enumerate(panels):
        sx = float(img.GetSpacing()[0])
        if geom:
            cx_mm = float(np.mean([e["x_mm"] for e in geom]))
            cy_mm = float(np.mean([e["y_mm"] for e in geom]))
            half = max(35.0, 1.6 * max(
                max(abs(e["x_mm"] - cx_mm), abs(e["y_mm"] - cy_mm)) for e in geom))
        else:
            cx_mm = cy_mm = None
            half = 60.0
        # Top: the whole phantom, so the module's place in the body is visible.
        fp.show_slice(axes[0][ci], sl, vmin=vmin, vmax=vmax, pixel_mm=sx,
                      scale_bar_mm=50.0, title=title,
                      title_size=fp.PRINT_FONT_PT)
        if cx_mm is not None:
            from matplotlib.patches import Rectangle
            zx, zy = fp.index_of_point(img, cx_mm, cy_mm, z_mid)
            y0, y1, x0, x1 = fp.crop_box_px(sl.shape, zx, zy, half / sx)
            axes[0][ci].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                            color="tab:orange", lw=1.0, zorder=5))
            fp.show_slice(axes[1][ci], sl[y0:y1, x0:x1], vmin=vmin, vmax=vmax,
                          pixel_mm=sx, scale_bar_mm=20.0,
                          title=f"{title}   (zoom)",
                          title_size=fp.PRINT_FONT_PT)
            for e in geom:
                px, py = fp.index_of_point(img, e["x_mm"], e["y_mm"], z_mid)
                fp.outline_circle(axes[1][ci], (px - x0, py - y0), e["radius_mm"] / sx,
                                  colour="tab:red", lw=0.8)
        else:
            axes[1][ci].axis("off")
    sub = f"   at matched background noise (~{target:.0f} HU)" if target else ""
    fig.suptitle(f"Threshold A: our reconstruction and the vendor products{sub}   "
                 f"({fp.window_label(vmin, vmax)})", fontsize=fp.PRINT_FONT_PT + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=200)
    if split:
        # Named from the source KEY, never from the panel title: the title is passed
        # through fp.pretty(), so "sirt_100" became "SIRT ×100" and the last token of it
        # became a file called modality_comparison_×100_full.png -- a non-ASCII name the
        # thesis cannot reference, sitting beside the stale sirt_100_* files of the run
        # before the rename. The key is stable, ASCII, and the same one every other
        # filename in the project uses.
        _save_panels(fig, axes, Path(path).parent, Path(path).stem,
                     [f"{k}_{r}" for r in ("full", "zoom")
                      for _t, _s, _i, k in panels], dpi=200)
    plt.close(fig)


def _noise_ordering_figure(path, res, cfg):
    """
    Background noise against energy channel, our variants and the vendor's on one axes.

    The one place in this comparison where the four channels belong together.  The
    trade-off figure keeps them apart because a pooled noise-against-resolution plot would
    mostly display the bins' photon statistics; here those statistics ARE the subject.
    The two families order their channels in opposite directions -- ours quietest at the
    highest threshold, where the fewest photons are counted, the vendor's noisiest there --
    and that ordering is a difference in what the two pipelines do to the data, not in how
    many photons reached them.  A single line per variant, because the claim is about the
    shape across A to D and not about any one channel.

    Log noise axis: the values span roughly 5 to 270 HU, so on a linear scale the two
    iterative variants and every vendor point would collapse onto the bottom and the
    ordering, which is the whole content, would be invisible.
    """
    plt = _mpl()
    if plt is None:
        return
    xs = np.arange(len(THRESHOLD_LABELS), dtype=float)
    own_names = [v["name"] for v in cfg.variants if v["name"] in (res.get("own") or {})]
    fig, ax = plt.subplots(figsize=fp.panel_figsize(1, frac=0.72, height_in=3.4))
    cmap = plt.get_cmap("Blues")
    drew = False
    for i, vname in enumerate(own_names):
        per = res["own"].get(vname, {})
        ys = [per.get(l, {}).get("noise_sd_hu", np.nan) for l in THRESHOLD_LABELS]
        if not np.any(np.isfinite(ys)):
            continue
        shade = cmap(0.55 + 0.42 * (i / max(len(own_names) - 1, 1)))
        ax.plot(xs, ys, "-o", ms=4, lw=1.3, color=shade, label=fp.pretty(vname))
        drew = True
    wf = [(res.get("wfbp") or {}).get(f"T{j + 1}", {}).get("noise_sd_hu", np.nan)
          for j in range(len(THRESHOLD_LABELS))]
    if np.any(np.isfinite(wf)):
        ax.plot(xs, wf, "-*", ms=12, lw=1.8, color="tab:red", label=fp.pretty("wfbp"))
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l} / T{j + 1}" for j, l in enumerate(THRESHOLD_LABELS)],
                       fontsize=fp.PRINT_FONT_PT - 1)
    ax.set_xlabel("energy channel   $\\rightarrow$ higher threshold",
                  fontsize=fp.PRINT_FONT_PT)
    ax.set_ylabel("noise SD in background (HU)\n$\\uparrow$ noisier",
                  fontsize=fp.PRINT_FONT_PT)
    ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Noise by energy channel", fontsize=fp.PRINT_FONT_PT + 1)
    _legend_below(fig, [ax], ncol=3, fontsize=fp.PRINT_FONT_PT - 1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _figures(cfg, out, res, match):
    plt = _mpl()
    if plt is None:
        return
    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _insert_localisation_figure(fig_dir / "insert_localisation.png", res, cfg)
    _noise_ordering_figure(fig_dir / "noise_ordering.png", res, cfg)
    _modality_triptych_figure(fig_dir / "modality_comparison.png", cfg, out, res, match,
                              split=cfg.split_figures)
    _qc_orientation_figure(Path(out) / "qc" / "qc_orientation.png", cfg, match, Path(out),
                           res_rois=res.get("_rois"))

    # --- the headline: noise vs resolution, ONE PANEL PER THRESHOLD ---
    # Thresholds are never mixed on one axes: A has all the photons and D very few, so a
    # combined plot would mostly display the bins' photon statistics and hide the
    # reconstruction differences it is meant to show.
    # Up, not right: this is a y axis, and a right arrow on a vertical axis is worse than
    # no arrow because the reader trusts it. Two lines so it fits rotated beside the axes.
    _TRADEOFF_YLABEL = "noise SD in background (HU)\n$\\uparrow$ noisier"
    notes = []          # (axes, annotation, x, up) -- re-sided once the limits are final
    fig, axes = plt.subplots(1, len(THRESHOLD_LABELS),
                             figsize=fp.panel_figsize(len(THRESHOLD_LABELS), frac=0.49,
                                                      height_in=3.8), squeeze=False)
    for j, label in enumerate(THRESHOLD_LABELS):
        ax = axes[0][j]
        xs, ys, names = [], [], []
        for vname in [v["name"] for v in cfg.variants]:
            m = res["own"].get(vname, {}).get(label)
            if m and np.isfinite(m.get("ttf50", np.nan)):
                xs.append(m["ttf50"])
                ys.append(m["noise_sd_hu"])
                names.append(vname)
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "-o", color="tab:blue",
                    label="our reconstruction (sweep)")
            # Alternating side, not a fixed offset: the five points lie along one curve,
            # so labels placed the same way all lean into the next point. The "FBP " is
            # dropped because the legend already says these are the swept variants and it
            # is the filter that distinguishes them.
            for i, (x, y, nm) in enumerate(zip(xs, ys, names)):
                up = (i % 2 == 0)
                notes.append((ax, ax.annotate(
                    fp.pretty(nm).replace("FBP ", ""), (x, y),
                    fontsize=fp.PRINT_FONT_PT - 1,
                    xytext=(4, 4) if up else (-4, -9),
                    ha="left" if up else "right",
                    textcoords="offset points"), x, up))
        wf = res["wfbp"].get(f"T{j + 1}")
        if wf and np.isfinite(wf.get("ttf50", np.nan)):
            ax.plot(wf["ttf50"], wf["noise_sd_hu"], "*", ms=16, color="tab:red",
                    label=fp.pretty("wfbp"))
        ax.set_xlabel("TTF$_{50}$ (cyc/mm)   $\\rightarrow$ sharper",
                      fontsize=fp.PRINT_FONT_PT)
        # Only the leftmost panel carries it in the combined figure; _save_panels puts it
        # back on every panel for the crops, which is where the thesis takes them from.
        if j == 0:
            ax.set_ylabel(_TRADEOFF_YLABEL, fontsize=fp.PRINT_FONT_PT)
        ax.set_title(f"threshold {label}  /  vendor T{j + 1}",
                     fontsize=fp.PRINT_FONT_PT)
        ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
        ax.grid(alpha=0.3)
        if j == 0 and not ax.get_legend_handles_labels()[0]:
            ax.text(0.5, 0.5, "no TTF available\n(no insert edge measured)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9,
                    color="tab:red")
    # One scale for all four panels, and log noise.  Autoscaled panels have to be
    # mentally rescaled before they can be compared, which defeats the point of showing
    # the channels side by side; and the noise axis spans 5-260 HU, so on a shared LINEAR
    # scale every vendor star and every SIRT point would flatten against the bottom.  On
    # log, a factor of two is the same distance in every panel, so how the gap evolves
    # from A to D can be read directly.
    _share_axes(axes[0], logy=True, free=cfg.free_axes)
    # Room for the per-point variant labels, which are annotations in offset points and so
    # are invisible to the autoscale: without this the topmost label at threshold A runs
    # into the panel title and the lowest at D into the tick labels. Every panel is padded
    # by the same factors, so the shared scale _share_axes just established survives.
    if not cfg.free_axes:
        for a in axes[0]:
            lo, hi = a.get_ylim()
            if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
                a.set_ylim(lo / 1.7, hi * 1.9)
            x0, x1 = a.get_xlim()
            dx = (x1 - x0) * 0.09
            a.set_xlim(x0 - dx, x1 + dx)
    # Which side a label sits on can only be decided once the limits are final: a label
    # placed to the left of a point that turns out to be the leftmost in the panel hangs
    # off the axes. Points in the left half get their label to the right and vice versa,
    # which the alternating up/down keeps from stacking on the neighbouring point.
    for a, note, x, up in notes:
        x0, x1 = a.get_xlim()
        right = (x - x0) / (x1 - x0) < 0.5 if x1 > x0 else True
        note.set_ha("left" if right else "right")
        note.xyann = (4, 4) if up else (-4, -9)
        if right != up:
            note.xyann = (4, -9) if right else (-4, 4)
    fig.suptitle("Noise–resolution trade-off per energy channel",
                 fontsize=fp.PRINT_FONT_PT + 1)
    _legend_below(fig, axes[0], ncol=4)
    fig.savefig(fig_dir / "tradeoff_per_threshold.png", dpi=200)
    if cfg.split_figures:
        _save_panels(fig, axes[0], fig_dir, "tradeoff", THRESHOLD_LABELS,
                     ylabel=_TRADEOFF_YLABEL, dpi=200)
    plt.close(fig)

    # --- NPS curves per threshold -------------------------------------
    fig, axes = plt.subplots(1, len(THRESHOLD_LABELS),
                             figsize=fp.panel_figsize(len(THRESHOLD_LABELS),
                                                      frac=0.49, height_in=3.5),
                             squeeze=False)
    for j, label in enumerate(THRESHOLD_LABELS):
        ax = axes[0][j]
        for vname in [v["name"] for v in cfg.variants]:
            m = res["own"].get(vname, {}).get(label)
            if m and "nps_f" in m:
                ax.plot(m["nps_f"], m["nps"], lw=1.1, label=fp.pretty(vname))
        wf = res["wfbp"].get(f"T{j + 1}")
        if wf and "nps_f" in wf:
            ax.plot(wf["nps_f"], wf["nps"], "k--", lw=1.8, label=fp.pretty("wfbp"))
        ax.set_xlabel("spatial frequency (cyc/mm)")
        if j == 0:
            ax.set_ylabel("NPS (HU$^2$ mm$^2$)")
        ax.set_title(f"threshold {label}", fontsize=fp.PRINT_FONT_PT)
        ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    _share_axes(axes[0], logy=True, free=cfg.free_axes)
    fig.suptitle("Noise power spectra", fontsize=fp.PRINT_FONT_PT + 1)
    # Below the panels, not inside the first one: it was covering the low-frequency end
    # of the curves, which is precisely where a denoised spectrum parts from an analytic
    # one and therefore the part of the figure that carries the result.
    _legend_below(fig, axes[0], ncol=3, fontsize=fp.PRINT_FONT_PT - 1)
    fig.savefig(fig_dir / "nps_curves.png", dpi=200)
    if cfg.split_figures:
        _save_panels(fig, axes[0], fig_dir, "nps", THRESHOLD_LABELS,
                     ylabel="NPS (HU$^2$ mm$^2$)")
    plt.close(fig)

    # --- VMI as its own family: noise and texture vs keV --------------
    if res["vmi"]:
        kev, sd, fav, dp = [], [], [], []
        for lbl, m in res["vmi"].items():
            try:
                kev.append(float(lbl.replace("keV", "")))
            except ValueError:
                continue
            sd.append(m["noise_sd_hu"])
            fav.append(m["f_av"])
            dp.append(m.get("d_prime", np.nan))
        o = np.argsort(kev)
        kev = np.array(kev)[o]
        fig, ax = plt.subplots(1, 3, figsize=fp.panel_figsize(3, frac=0.49, height_in=3.4))
        ax[0].plot(kev, np.array(sd)[o], "-o")
        ax[0].set_xlabel("keV")
        ax[0].set_ylabel("noise SD (HU)")
        ax[0].set_title("Noise magnitude", fontsize=fp.PRINT_FONT_PT)
        ax[1].plot(kev, np.array(fav)[o], "-o", color="tab:green")
        ax[1].set_xlabel("keV")
        ax[1].set_ylabel("$f_{av}$ (cyc/mm)")
        ax[1].set_title("Noise texture", fontsize=fp.PRINT_FONT_PT)
        ax[2].plot(kev, np.array(dp)[o], "-o", color="tab:purple")
        ax[2].set_xlabel("keV")
        # The task belongs in the axis label: d' is meaningless without it, so it is a
        # unit of the quantity rather than a description of the plot.
        ax[2].set_ylabel(f"d'  ({cfg.task_diameter_mm:g} mm, {cfg.task_contrast_hu:g} HU task)")
        ax[2].set_title("Detectability", fontsize=fp.PRINT_FONT_PT)
        for a in ax:
            a.grid(alpha=0.3)
        fig.suptitle("Vendor monoenergetic series", fontsize=fp.PRINT_FONT_PT + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(fig_dir / "vmi_vs_kev.png", dpi=200)
        if cfg.split_figures:
            # No shared ylabel: the three panels plot different quantities.
            _save_panels(fig, ax, fig_dir, "vmi", ("noise", "texture", "dprime"))
        plt.close(fig)

    # --- Bland-Altman, own vs WFBP: ONE PANEL PER THRESHOLD -----------
    # Pooling all four channels onto one axes put ~5 variants x 4 thresholds x 18 ROIs on
    # top of each other, which is unreadable -- but the pooling was also wrong.  The
    # channels have very different noise, so a single set of limits of agreement mixes
    # populations whose spreads differ by an order of magnitude and describes none of
    # them.  Faceting fixes the crowding and the statistic at the same time: the bias and
    # the 95 % LoA below are computed per channel, which is what they mean.
    bias = res.get("bias") or {}
    if bias:
        thr_labels = [l for l in THRESHOLD_LABELS
                      if any(l in per for per in bias.values())]
        if thr_labels:
            # Shared scales unless --free-axes, for the same reason as the trade-off
            # figure: autoscaled panels have to be mentally rescaled before they can be
            # compared, which is the whole point of showing the channels side by side.
            fig, axes = plt.subplots(1, len(thr_labels),
                                     figsize=fp.panel_figsize(len(thr_labels), frac=0.49,
                                                      height_in=3.5),
                                     squeeze=False, sharex=not cfg.free_axes,
                                     sharey=not cfg.free_axes)
            ba = res.get("bland_altman") or {}
            for j, label in enumerate(thr_labels):
                ax = axes[0][j]
                wf_lbl = f"T{THRESHOLD_LABELS.index(label) + 1}"
                stats = ba.get(label) or {}
                fits = stats.get("per_variant") or {}
                pooled, slopes = [], []
                for vname, per_thr in bias.items():
                    d = per_thr.get(label)
                    if not d:
                        continue
                    xs = [p["mean_hu"] for p in d["roi_pairs"]]
                    ys = [p["diff_hu"] for p in d["roi_pairs"]]
                    sc = ax.scatter(xs, ys, s=16, alpha=0.6, linewidths=0,
                                    label=fp.pretty(vname))
                    # One regression line per variant, in that variant's own colour. A
                    # single pooled line would average five populations whose slopes
                    # differ by construction, since each variant delivers a different
                    # fraction of the vendor's contrast -- see _ba_summary.
                    fit = fits.get(vname)
                    if fit and xs:
                        gx = np.array([min(xs), max(xs)], float)
                        ax.plot(gx, fit["intercept"] + fit["slope"] * gx, "-",
                                color=sc.get_facecolor()[0], lw=1.1, alpha=0.9, zorder=3)
                        slopes.append(fit["slope"])
                    pooled += ys
                if pooled:
                    mu, sg = float(np.mean(pooled)), float(np.std(pooled))
                    ax.axhline(mu, color="k", lw=1.2)
                    ax.axhline(mu + 1.96 * sg, color="k", ls="--", lw=1.0)
                    ax.axhline(mu - 1.96 * sg, color="k", ls="--", lw=1.0)
                    # Above the axes, not in a corner of them, and not in the shared
                    # legend. A corner box covers data as soon as the box or the cloud
                    # changes -- adding the slope made it three lines tall and it buried
                    # the lowest-contrast variant's regression line together with the
                    # points it was fitted to. The strip below the title cannot cover
                    # anything, and it stays attached to its own panel, which a shared
                    # legend would not: these numbers differ per channel.
                    txt = f"bias {mu:+.1f} HU   95 % LoA ±{1.96 * sg:.1f} HU"
                    if slopes:
                        txt += (f"   slope {min(slopes):+.2f} to {max(slopes):+.2f}"
                                if len(slopes) > 1 else f"   slope {slopes[0]:+.2f}")
                    ax.text(0.5, 1.008, txt, transform=ax.transAxes, ha="center",
                            va="bottom", fontsize=fp.PRINT_FONT_PT - 2, color="0.25")
                ax.axhline(0, color="0.6", lw=0.8, zorder=0)
                ax.set_xlabel("mean of the two measurements (HU)")
                if j == 0:
                    ax.set_ylabel("ours − WFBP (HU)")
                # pad clears the number strip the title now sits above.
                ax.set_title(f"threshold {label}  /  vendor {wf_lbl}",
                             fontsize=fp.PRINT_FONT_PT, pad=13.0)
                ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
                ax.grid(alpha=0.3)
            fig.suptitle("Bland–Altman: region-of-interest mean HU, "
                         "ours against the vendor", fontsize=fp.PRINT_FONT_PT + 1)
            _legend_below(fig, axes[0], ncol=3, fontsize=fp.PRINT_FONT_PT - 1)
            fig.savefig(fig_dir / "bland_altman_own_vs_wfbp.png", dpi=200)
            if cfg.split_figures:
                _save_panels(fig, axes[0], fig_dir, "bland_altman", thr_labels,
                             ylabel="ours − WFBP (HU)")
            plt.close(fig)

    _write_captions(fig_dir)


def _print_summary(res):
    def water(m):
        w = m.get("water") or {}
        v = w.get("offset_hu", float("nan"))
        # A wall-contaminated region gives a number that looks fine and is not, so it is
        # marked here rather than left to be read off the JSON.
        flag = "!" if abs(w.get("wall_bias_hu") or 0.0) > 5.0 else " "
        return f"{v:9.1f}{flag}"

    print("\n" + "=" * 106)
    print("RECONSTRUCTION COMPARISON -- summary")
    print("=" * 106)
    hdr = f"{'family / variant':24s} {'ch':>6s} {'noise SD':>10s} {'SNR mu':>8s} " \
          f"{'f_av':>8s} {'TTF50':>8s} {'edge90':>8s} {'d prime':>9s} {'contrast':>9s} " \
          f"{'water dHU':>10s}"
    print(hdr)
    print("-" * 106)
    for vname, per in res.get("own", {}).items():
        for label, m in per.items():
            print(f"{'own/' + vname:24s} {label:>6s} {m['noise_sd_hu']:10.2f} "
                  f"{m.get('snr_mu', float('nan')):8.1f} "
                  f"{m['f_av']:8.3f} {m.get('ttf50', float('nan')):8.3f} "
                  f"{m.get('edge_width_1090_mm', float('nan')):8.3f} "
                  f"{m.get('d_prime', float('nan')):9.2f} "
                  f"{m.get('contrast_fidelity', float('nan')):9.2f} {water(m)}")
    for kind in ("wfbp", "vmi"):
        for label, m in res.get(kind, {}).items():
            print(f"{'siemens/' + kind:24s} {label:>6s} {m['noise_sd_hu']:10.2f} "
                  f"{m.get('snr_mu', float('nan')):8.1f} "
                  f"{m['f_av']:8.3f} {m.get('ttf50', float('nan')):8.3f} "
                  f"{m.get('edge_width_1090_mm', float('nan')):8.3f} "
                  f"{m.get('d_prime', float('nan')):9.2f}"
                  f"{'          ' if kind != 'wfbp' else '      1.00'} {water(m)}")
    bias = res.get("bias") or {}
    rows = [(v, l, d["similarity"]) for v, per in sorted(bias.items())
            for l, d in sorted((per or {}).items()) if (d or {}).get("similarity")]
    if rows:
        print("\n" + "-" * 106)
        print("SIMILARITY against the vendor reconstruction, inside the phantom mask")
        print("-" * 106)
        print(f"  {'ch':>3s} {'variant':22s} {'MAX HU':>8s} {'RMSE':>8s} {'PSNR':>8s} "
              f"{'RMSE-bc':>9s} {'PSNR-bc':>9s} {'SSIM':>7s}")
        for v, l, s in sorted(rows, key=lambda r: (r[1], r[0])):
            print(f"  {l:>3s} {v:22s} {s['max_hu']:8.0f} {s['rmse_hu']:8.1f} "
                  f"{s['psnr_db']:8.2f} {s['rmse_debiased_hu']:9.1f} "
                  f"{s['psnr_debiased_db']:9.2f} {s.get('ssim', float('nan')):7.3f}")
        print("-" * 106)
        print("PSNR uses MAX = the Hounsfield RANGE of the reference inside the mask, not a\n"
              "display constant, so it is comparable only against a PSNR computed the same way.\n"
              "-bc is after the mean difference is removed. Read that column, not the raw one:\n"
              "the channel offset of -73.9 to -18.1 HU dominates the raw mean squared error, so\n"
              "the raw PSNR largely restates the offset the Bland-Altman block already reports.")

    ba = res.get("bland_altman") or {}
    if ba:
        print("\n" + "-" * 106)
        print("BLAND-ALTMAN, ours vs WFBP -- difference against the mean of the two")
        print("-" * 106)
        print(f"{'ch':>4s} {'variant':22s} {'bias HU':>9s} {'LoA HU':>9s} "
              f"{'slope':>8s} {'95 % CI':>18s} {'r2':>6s} {'n':>4s}")
        for label in THRESHOLD_LABELS:
            s = ba.get(label)
            if not s:
                continue
            pf = s.get("pooled_fit") or {}
            lo, hi = (pf.get("slope_ci95") or [float("nan")] * 2)
            print(f"{label:>4s} {'pooled':22s} {s['bias_hu']:9.1f} {s['loa95_hu']:9.1f} "
                  f"{pf.get('slope', float('nan')):8.3f} "
                  f"{lo:8.3f} {hi:8.3f} {pf.get('r2', float('nan')):6.2f} {s['n']:4d}")
            for vname, f in sorted((s.get("per_variant") or {}).items()):
                lo, hi = f["slope_ci95"]
                print(f"{'':>4s} {vname[:22]:22s} {'':>9s} {'':>9s} "
                      f"{f['slope']:8.3f} {lo:8.3f} {hi:8.3f} {f['r2']:6.2f} {f['n']:4d}")
        print("-" * 106)
        print("SLOPE is the contrast-dependent part of the disagreement. Read it per "
              "variant, not\npooled: a variant delivering the fraction c of the vendor's "
              "insert contrast must give a\nslope near 2(c-1)/(c+1) -- the abscissa is the "
              "MEAN of the two measurements, not the\nvendor value -- so five variants of "
              "different contrast have five different slopes and\nthe pooled figure "
              "describes none of them. Compare each slope against 2(c-1)/(c+1) for that "
              "\nvariant's CONTRAST column above; a slope away from it is disagreement the "
              "contrast does\nnot explain.")
    print("=" * 106)
    print("Read the trade-off figure, not this table alone: lower noise at lower TTF50 "
          "is\njust a smoother kernel, which is a move ALONG the curve, not a better "
          "reconstruction.")
    print("CONTRAST tells you whether a low noise SD is real: it is the fraction of the "
          "vendor's\ninsert contrast the variant actually delivers. Well below 1 means "
          "the reconstruction has\nnot converged, and its noise, TTF and d' are all "
          "flattering for that reason.")
    print(f"WATER dHU is the mean of the phantom's body material minus the {WATER_HU:.0f} HU "
          "it is\nspecified to read -- the only ABSOLUTE number here; every other column "
          "compares one\nreconstruction with another. For siemens/* it is accuracy. For "
          "own/* it is a CALIBRATION\nRESIDUAL, not accuracy: auto_hu_calibrate sets "
          "mu_water from the body histogram mode, so\nthe scale is anchored to this same "
          "material -- a large value means the mode and the drawn\nregion disagree, which "
          "is still worth knowing. A trailing '!' means the region is\nwall-contaminated "
          "and the value should not be used (see water.wall_bias_hu).")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m reconstruction.recon_comparison",
        description="Compare our reconstruction against Siemens WFBP and VMI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", default="all", choices=["0", "1", "2", "all", "probe",
                                                      "sweep", "metrics"],
                   help="0/probe = read Siemens geometry + find slab (CPU); "
                        "1/sweep = reconstruct the variants (GPU); "
                        "2/metrics = measure everything (CPU)")
    p.add_argument("--wfbp-dir", help="folder of ONE Siemens WFBP threshold series set")
    p.add_argument("--vmi-dir", help="folder of ONE Siemens monoenergetic (VMI) series set")
    p.add_argument("--out-root", default=str(_REPO_ROOT / "output" / "research"
                                             / "recon_comparison"))
    p.add_argument("--data-path", default=_DEFAULT_DATA, help="raw sinogram .mat (HDF5)")
    p.add_argument("--desc-path", default=_DEFAULT_DESC, help="descriptor .mat (v5 struct)")
    p.add_argument("--geo-dir", default=str(_REPO_ROOT / "geometry"))
    p.add_argument("--no-mirror-x", dest="mirror_x", action="store_false",
                   help="do not flip our volumes left-right on write. The mirror is on "
                        "by default because it was measured on this scanner (see "
                        "probe_dicom_origin): the vendor and our reconstruction agree on "
                        "the phantom's position and tilt only after x -> -x. Use this to "
                        "reproduce the unmirrored behaviour for an A/B check")
    p.set_defaults(mirror_x=True)
    p.add_argument("--free-axes", action="store_true",
                   help="let each per-threshold panel autoscale. Default is one shared "
                        "scale across the four panels (log noise axis) so they can be "
                        "compared without rescaling each one by eye")
    p.add_argument("--no-split-figures", dest="split_figures", action="store_false",
                   help="do not write the per-panel copies under figures/panels/. They "
                        "exist so the thesis can place one panel at 0.5\\textwidth "
                        "instead of four at a quarter each")
    p.set_defaults(split_figures=True)
    p.add_argument("--slab",
                   help="override the auto-detected slab, 'z_lo,z_hi' in mm. "
                        "Write it as --slab=-1500,-1470 -- a value starting with "
                        "'-' is otherwise parsed as a flag")
    p.add_argument("--phantom-z", metavar="LO,HI",
                   help="axial extent of the phantom in mm (read off the vendor volume, "
                        "e.g. in Slicer). Restricts the insert-layer search to the "
                        "phantom. Strongly recommended for a clinical scan range. "
                        "Write it as --phantom-z=-1515.5,-1408.3 -- a value "
                        "starting with '-' is otherwise parsed as a flag")
    p.add_argument("--no-phantom-z", action="store_true",
                   help="ignore --phantom-z and search the whole scan range")
    p.add_argument("--slab-select", choices=["peak", "longest"], default="peak",
                   help="which candidate layer to take: peak = most insert-covered "
                        "slice (best when layers carry different insert counts)")
    p.add_argument("--expect-inserts", type=int,
                   help="expected number of inserts; warns when detection disagrees")
    p.add_argument("--max-slab-mm", type=float, default=40.0,
                   help="cap on the swept slab so the GPU stage stays affordable")
    p.add_argument("--n-pixels", type=int, help="override the matched matrix size")
    p.add_argument("--fov-mm", type=float, help="override the matched FOV")
    p.add_argument("--nps-patch-px", type=int, default=64,
                   help="requested NPS patch size; halved automatically until it fits "
                        "the phantom, and the chosen size is shared by all families")
    p.add_argument("--segmentation", metavar="FILE",
                   help="hand-drawn insert prior (Slicer .nrrd/.seg.nrrd). Positions "
                        "only need to be roughly right - every edge is refined from "
                        "the image, and a partial annotation is fine. Defaults to "
                        "<wfbp-dir>/Segmentation.nrrd if that exists")
    p.add_argument("--background-label", type=int, metavar="N",
                   help="label value in the segmentation marking a UNIFORM region for "
                        "the noise measurement. On an anthropomorphic phantom this is "
                        "close to required: the body outline contains lung, bone and "
                        "the table, none of which are noise")
    p.add_argument("--insert-mad-k", type=float, default=4.0,
                   help="insert-detection sensitivity in robust SDs; lower finds "
                        "fainter inserts. Check qc/roi_*.png after changing it")
    p.add_argument("--task-diameter-mm", type=float, default=5.0)
    p.add_argument("--task-contrast-hu", type=float, default=50.0)
    p.add_argument("--variants", help="comma-separated subset of variant names to run")
    p.add_argument("--list-series", metavar="DIR",
                   help="print every DICOM series under DIR with its geometry and exit "
                        "(use this to find the right --wfbp-dir/--vmi-dir)")
    p.add_argument("--force", action="store_true",
                   help="re-run sweep variants even if they are up to date")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _list_series(folder, cache_dir=None):
    idx = index_dicom_series(folder, cache_dir=cache_dir)
    print(f"\n{len(idx)} series under {folder}\n")
    print(f"{'#':>5s} {'files':>6s} {'thick':>7s} {'space':>7s} {'pixel':>7s} "
          f"{'matrix':>10s} {'kernel':<12s} {'class':<12s} description")
    print("-" * 118)
    for uid, s in sorted(idx.items(), key=lambda kv: str(kv[1].get("number"))):
        spec = _classify_series(s.get("desc", ""), s["files"][0][1] if s["files"] else "")
        cls = f"{spec[0]}:{spec[1]:g}" if spec else "-"
        print(f"{str(s['number']):>5s} {len(s['files']):>6d} "
              f"{(s['thickness_mm'] or 0):7.2f} {(s['spacing_between_mm'] or 0):7.2f} "
              f"{(s['pixel_mm'] or 0):7.4f} "
              f"{str(s['rows']) + 'x' + str(s['cols']):>10s} "
              f"{(s['kernel'] or '-'):<12s} {cls:<12s} {s['desc']}")
    print("\nPoint --wfbp-dir / --vmi-dir at a folder holding exactly ONE set: duplicate "
          "\nchannel labels across sets would otherwise be concatenated into one stack.")


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if a.list_series:
        Path(a.out_root).mkdir(parents=True, exist_ok=True)
        _list_series(a.list_series, cache_dir=a.out_root)
        return 0

    cfg = CompareConfig(
        out_root=a.out_root, wfbp_dir=a.wfbp_dir, vmi_dir=a.vmi_dir,
        data_path=a.data_path, desc_path=a.desc_path, geo_dir=a.geo_dir,
        n_pixels=a.n_pixels, fov_mm=a.fov_mm, max_slab_mm=a.max_slab_mm,
        nps_patch_px=a.nps_patch_px, insert_mad_k=a.insert_mad_k,
        segmentation=a.segmentation, background_label=a.background_label,
        task_diameter_mm=a.task_diameter_mm,
        task_contrast_hu=a.task_contrast_hu, force=a.force,
        slab_mm=tuple(float(x) for x in a.slab.split(",")) if a.slab else None,
        phantom_z_mm=(tuple(float(x) for x in a.phantom_z.split(","))
                      if a.phantom_z else None),
        use_phantom_z=not a.no_phantom_z,
        slab_select=a.slab_select, expect_inserts=a.expect_inserts,
        mirror_x=a.mirror_x, free_axes=a.free_axes, split_figures=a.split_figures,
    )
    if a.variants:
        want = {s.strip() for s in a.variants.split(",")}
        cfg.variants = [v for v in cfg.variants if v["name"] in want]
        if not cfg.variants:
            raise SystemExit(f"no variants matched {sorted(want)}; available: "
                             f"{[v['name'] for v in DEFAULT_VARIANTS]}")

    if not cfg.segmentation and cfg.wfbp_dir:
        for name in ("Segmentation.nrrd", "Segmentation.seg.nrrd"):
            cand = Path(cfg.wfbp_dir) / name
            if cand.exists():
                cfg.segmentation = str(cand)
                logger.info("using the segmentation found next to the WFBP data: %s",
                            cand)
                break
    if cfg.segmentation and not Path(cfg.segmentation).exists():
        raise SystemExit(f"--segmentation {cfg.segmentation} does not exist")

    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)
    _write_json(Path(cfg.out_root) / "compare_config.json", cfg.to_dict())

    stage = {"probe": "0", "sweep": "1", "metrics": "2"}.get(a.stage, a.stage)
    if stage in ("0", "all"):
        stage0_probe(cfg)
    if stage in ("1", "all"):
        stage1_sweep(cfg)
    if stage in ("2", "all"):
        stage2_metrics(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
