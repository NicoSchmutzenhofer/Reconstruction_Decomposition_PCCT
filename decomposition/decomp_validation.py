"""
decomp_validation.py -- validate the material decomposition against the QRM DEP-V5 phantom.

Scores the iodine and calcium maps of every run on the phantom's 18 inserts: in each of
three layers, an agent-free base pair and two HU-matched Ca/I pairs (low, high).  Metrics are
the separation within each pair, the reading on the base inserts, linearity and response
sign against CONC_RELATIVE, and consistency across layers.  Layer and role are assigned from
the measured HU alone, never from the order the inserts were drawn in.

Two comparisons use these metrics:
    RQ1  4 bins vs the scanner's 2-bin product (`--s2bin-dir`).  Its maps are imported into
         runs/<key>/, not solved, and both sides are scored on iodine only (`rq1_iodine`),
         since 2 bins give no calcium map.  The 2-bin value is shown against the range of
         all 4-bin runs.
    RQ2  input modality: own reconstruction vs vendor WFBP vs vendor VMI.

Segmentation contract (`Segmentation_decomposition.nrrd`, drawn on the WFBP volume):
    labels 1-3   layer 1, one label per HU-matched PAIR, outermost pair first
    labels 4-6   layer 2, likewise
    labels 7-9   layer 3, likewise
    label 10     the module boundary, drawn as a filled ring -> filled to its OUTER edge
    Each pair label holds TWO centroid markers (not outlines), grown to the 5 mm insert
    radius; their z-extent comes from fill-between-slices.

Three resumable stages:
    0  probe       CPU   ROIs, layer/role assignment      -> decomp_match.json
    1  decompose   CPU   source x mode x estimator        -> runs/<key>/
    2  metrics     CPU   metrics + figures                -> metrics/ figures/ qc/

    python -m decomposition.decomp_validation --probe-siemens <2-bin export>   # run FIRST
    python -m decomposition.decomp_validation --stage 0 --wfbp-dir ... --vmi-dir ... --own-dir ...
    python -m decomposition.decomp_validation --stage all ...
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconstruction import image_quality_metrics as iq
from reconstruction import figure_panels as fp
from reconstruction.recon_comparison import (
    load_segmentation, index_dicom_series, select_family, read_series_volume,
    _share_axes, _legend_below, _mpl, _sitk, _write_json, _jsonable,
    _check_pose_against_reference, _slice_at_z, _save_panels)
from decomposition import material_decomposition as md
from decomposition import decomposition_modes as modes

logger = logging.getLogger("decomp_validation")
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════
# Constants taken from the QRM DEP-V5 datasheet (V20200827OL)
# ═════════════════════════════════════════════════════════════════════
INSERT_DIAMETER_MM = 10.0            # "18 lesions ... D 10 mm / H 10 mm"
INSERT_RADIUS_MM = INSERT_DIAMETER_MM / 2
BODY_HU = 0.0                        # "Phantom body 0 HU at 80 - 140 kV"

# Base-material HU at 140 kV -- the value valid at OUR tube voltage, which is what makes
# the layers identifiable without any assumption about drawing order.  The three values are
# far apart, so the assignment is unambiguous even with our noise.
LAYER_BASE_HU_140 = {"A": -100.0, "B": 55.0, "C": -24.0}
LAYER_BASE_MATERIAL = {"A": "fat", "B": "soft tissue", "C": "fat + soft tissue"}

# Nominal HU at 120 kV, for reference only -- never used for assignment, because our scan
# is at 140 kV and both agents lose contrast as kV rises.
LAYER_DOPED_HU_120 = {"A": {"low": -50.0, "high": 60.0},
                      "B": {"low": 100.0, "high": 200.0},
                      "C": {"low": 30.0, "high": 140.0}}

# INFERRED, not stated by QRM: relative agent concentration of the three roles.  Derived by
# interpolating the base HU to 120 kV and taking the agent's HU increment; the three layers
# independently give 3.06 / 3.31 / 2.99, and that agreement is the argument.
CONC_RELATIVE = {"base": 0.0, "low": 1.0, "high": 3.1}
CONC_RELATIVE_PER_LAYER = {"A": 3.06, "B": 3.31, "C": 2.99}
CONC_IS_INFERRED = True

ROLES = ("base", "low", "high")
AGENTS = ("Iodine", "HA")
# "HA" is the internal key and the column name in the signature table; on a figure the
# reader has not met it, so the panels spell it out.
MATERIAL_LABELS = {"Iodine": "iodine", "HA": "calcium hydroxyapatite",
                   "Water": "water", "SoftTissue": "soft tissue"}
WATER_LABEL = 999                    # the module body inside roi_labels.nrrd

# ── RQ1: 4 bins vs 2 bins ────────────────────────────────────────────────────
# The vendor's 2-bin mode can estimate at most 2 materials, so it has an iodine map and a
# lumped base and NO calcium map.  Scoring our 4-bin runs on an Iodine+HA average against a
# 2-bin run that can only offer Iodine would credit us with information the other mode is
# mathematically barred from having.  So RQ1 is computed on this restricted agent set, on
# BOTH sides, and stored in its own block (`rq1_iodine`) next to the full-agent block that
# RQ2 uses.  Two blocks rather than one switched global: the restriction is then visible in
# the output instead of implicit in whichever code path ran.
RQ1_AGENTS = ("Iodine",)
SOURCE_2BIN = "s2bin"                # the vendor 2-bin export
VENDOR_MODE = "siemens_2bin"         # imported, never solved
VENDOR_ESTIMATOR = "vendor"

# ── Which measurement stack each source is solved from ───────────────────────
# The three sources do NOT carry the same quantity, so one bin_domain cannot serve them:
#
#   own   'cumulative'  our volumes reconstruct the raw threshold sinograms, which count
#                       every photon ABOVE the threshold -- nested A>=B>=C>=D. Fed as-is.
#   wfbp  'exclusive'   the separate energy bins, formed by image-domain differencing of
#                       the vendor's thresholds ([A-B, B-C, C-D, D]). Image domain only:
#                       Invariant #3 (README) forbids the sinogram-domain version.
#   vmi   'direct'      monoenergetic keV images are not thresholds at all -- there is
#                       nothing nested to subtract, and each channel's mu/rho comes from
#                       K-edge-aware interpolation at its own keV. material_decomposition
#                       coerces any mono stack to 'direct' itself, so this only makes the
#                       intent explicit rather than relying on that safety net.
#
# s2bin is absent on purpose: the vendor 2-bin maps are imported, never solved.
BIN_DOMAIN_BY_SOURCE = {"own": "cumulative", "wfbp": "exclusive", "vmi": "direct"}


@dataclass
class ValidationConfig:
    """Serialisable configuration -- a config object, never module-level constants."""
    out_root: str = str(_REPO_ROOT / "output" / "research" / "decomp_validation")
    own_dir: str = str(_REPO_ROOT / "output" / "reconstruction")
    wfbp_dir: Optional[str] = None
    vmi_dir: Optional[str] = None
    # The Siemens 2-bin export (PBV + VNC folders).  Its maps are IMPORTED, not solved.
    s2bin_dir: Optional[str] = None
    # Series descriptions to recognise inside that export.  Substring, case-insensitive.
    s2bin_iodine_match: str = "pbv"
    s2bin_vnc_match: str = "vnc"

    # Auto-discovered as <wfbp_dir>/Segmentation_decomposition.nrrd when not given.
    segmentation: Optional[str] = None
    boundary_label: int = 10             # the module boundary ring
    pair_labels: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    layer_of_label: dict = field(default_factory=lambda: {
        1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2})

    insert_radius_mm: float = INSERT_RADIUS_MM
    roi_inner_frac: float = 0.7          # measure inside 70 % of the radius (partial volume)
    edge_margin_mm: float = 3.0          # keep the water ROI off the module boundary
    insert_margin_mm: float = 3.0        # and off the inserts

    sources: tuple = ("own", "wfbp", "vmi")
    mode: str = "phantom_water_ca_i"     # basis whose base column is the phantom body material
    sensitivity_mode: Optional[str] = "phantom_ca_i"   # second basis; None to skip
    estimators: tuple = ("ols", "wls", "wls_denoise", "wls_joint")
    # Per source, because the sources hold different quantities -- see BIN_DOMAIN_BY_SOURCE.
    bin_domain_by_source: dict = field(
        default_factory=lambda: dict(BIN_DOMAIN_BY_SOURCE))
    # Force one domain on every source. None = use the per-source table above. Only for
    # deliberately re-running a source in a domain that is not its natural one.
    bin_domain: Optional[str] = None
    threshold_option: int = 1

    # wls_denoise / wls_joint can fail to converge.  A run past this is stopped and REPORTED
    # as truncated -- a half-converged map returned silently is worse than no map.
    estimator_timeout_s: float = 1800.0
    joint_iters_cap: int = 10

    # DEFAULT OFF since python_reconstruction.py adopted the corrections itself
    # (OUTPUT_MIRROR_X / TABLE_HEIGHT_MM -> helical_reconstruction.to_output_frame).
    # Applying them again here would flip the volume back.  Set True only for volumes
    # reconstructed before that change.
    own_frame_fix: bool = False

    # Per-channel water gain, fitted so a water ROI reads the tabulated water value.  Stated
    # here rather than inherited: `_one_run` used to leave it out of DecompConfig entirely,
    # so every validation run silently took the dataclass default (True) while decompose.py's
    # own module default is False -- two entry points to one solver disagreeing about whether
    # a calibration was applied, with nothing in run.json to say which.  It is now passed
    # explicitly and therefore recorded.  It matters for more than bookkeeping: the
    # reconstruction carries a channel-dependent HU offset of tens of HU, which enters the
    # linear system as an additive per-channel error no basis column can absorb, and this
    # gain is the pipeline's only lever on it.  `--no-water-calibration` runs the comparison.
    water_calibration: bool = True

    # How the CUMULATIVE window averages are weighted over energy. 'uniform' is the
    # published table's own weighting and the default; 'spectrum' weights by the photon
    # fluence a counting detector sees, which is the sensitivity run the inversion question
    # asks for. It reaches the `own` source only, since that is the one solved on windows
    # this code integrates -- wfbp reads the published disjoint rows and vmi is point
    # values, and neither may be moved by a weighting choice. See material_library.Weighting.
    signature_weighting: str = "uniform"     # 'uniform' | 'spectrum'
    spectrum_kvp: float = 140.0
    spectrum_water_cm: float = 16.0

    # One separation + linearity figure per run is 24 of each. Detail goes to the best
    # run; True restores one per run when a specific combination needs inspecting.
    all_figures: bool = False

    # Also write every multi-panel figure one panel per file, under figures/panels/, the
    # way recon_comparison already does -- four panels at \textwidth give each a quarter of
    # the page, and the fonts shrink with it.
    split_figures: bool = True
    # Physical half-width of the crop used by the image figures. The QC grid keeps its own
    # generous pad (see _roi_placement_figure): there the surrounding body IS the check.
    image_half_mm: float = 62.0
    # Slice thickness averaged into each image panel. A single 0.4 mm slice of a material
    # map is mostly noise; the inserts are 10 mm tall, so averaging a few mm is honest and
    # shows what the ROI statistics were actually measured on.
    image_slab_mm: float = 4.0

    free_axes: bool = False
    force: bool = False

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


# ═════════════════════════════════════════════════════════════════════
# Stage 0 -- ROIs, layer and role assignment
# ═════════════════════════════════════════════════════════════════════
def _components_of_label(labels3d, value):
    """Every connected component of one label -> (cy, cx, k_lo, k_hi, n_px)."""
    mask = labels3d == int(value)
    if not mask.any():
        return []
    proj = np.any(mask, axis=0)
    comps, n = ndimage.label(proj)
    out = []
    for i in range(1, n + 1):
        c2 = comps == i
        kz = np.nonzero(np.any(mask & c2[None], axis=(1, 2)))[0]
        if kz.size == 0:
            continue
        cy, cx = ndimage.center_of_mass(c2)
        out.append({"cy": float(cy), "cx": float(cx),
                    "k_lo": int(kz.min()), "k_hi": int(kz.max()),
                    "n_seed_px": int(c2.sum())})
    out.sort(key=lambda d: (d["cx"], d["cy"]))
    return out


def _disc(shape_yx, cy, cx, r_px):
    yy, xx = np.ogrid[:shape_yx[0], :shape_yx[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r_px ** 2


def measure_rois(vol, inserts, pixel_mm, cfg: ValidationConfig, reduce=np.mean):
    """Mean and SD inside each insert, over the slices that insert actually occupies."""
    r_px = cfg.insert_radius_mm / pixel_mm * cfg.roi_inner_frac
    out = []
    for ins in inserts:
        d = _disc(vol.shape[1:], ins["cy"], ins["cx"], r_px)
        sub = vol[ins["k_lo"]:ins["k_hi"] + 1][:, d]
        sub = sub[np.isfinite(sub)]
        out.append({"mean": float(reduce(sub)) if sub.size else float("nan"),
                    "sd": float(np.std(sub)) if sub.size else float("nan"),
                    "n": int(sub.size)})
    return out


def assign_layers_and_roles(inserts, hu_by_insert):
    """
    Turn measured HU into layer identity and insert role, using only the datasheet.

    Two facts do all the work.  Within a layer the manual gives base < low < high, so
    ranking the three PAIRS by mean HU names the roles.  And the base-material HU at 140 kV
    is -100 / +55 / -24 for layers A / B / C -- three values far enough apart that the
    layer follows from the base pair's HU alone.  Nothing here uses the order the inserts
    were drawn in, or which side of the phantom they sit on.
    """
    by_layer = {}
    for ins, hu in zip(inserts, hu_by_insert):
        by_layer.setdefault(ins["layer_index"], []).append((ins, hu))

    assignment = {}
    for li, items in sorted(by_layer.items()):
        pairs = {}
        for ins, hu in items:
            pairs.setdefault(ins["pair_label"], []).append(hu)
        pair_mean = {p: float(np.mean(v)) for p, v in pairs.items()}
        ranked = sorted(pair_mean, key=pair_mean.get)          # base < low < high
        role_of_pair = {p: ROLES[i] for i, p in enumerate(ranked)}
        base_hu = pair_mean[ranked[0]]
        layer = min(LAYER_BASE_HU_140, key=lambda k: abs(LAYER_BASE_HU_140[k] - base_hu))
        assignment[li] = {"layer": layer, "base_hu_measured": base_hu,
                          "base_hu_nominal_140kV": LAYER_BASE_HU_140[layer],
                          "role_of_pair": role_of_pair,
                          "pair_mean_hu": pair_mean}
    used = [a["layer"] for a in assignment.values()]
    if len(set(used)) != len(used):
        logger.warning("two layers were assigned the same identity %s -- the base HU did "
                       "not separate them. Check qc/identification.png before using any "
                       "layer-resolved number.", used)
    return assignment


def stage0_probe(cfg: ValidationConfig):
    """Build the insert ROIs and assign layer/role -> decomp_match.json."""
    sitk = _sitk()
    out = Path(cfg.out_root)
    (out / "qc").mkdir(parents=True, exist_ok=True)

    if not cfg.segmentation and cfg.wfbp_dir:
        guess = Path(cfg.wfbp_dir) / "Segmentation_decomposition.nrrd"
        if guess.exists():
            cfg.segmentation = str(guess)
            logger.info("using the segmentation found next to the WFBP data: %s", guess)
    if not cfg.segmentation:
        raise ValueError("no segmentation: pass --segmentation")
    if not cfg.wfbp_dir:
        raise ValueError("stage 0 needs --wfbp-dir (the segmentation was drawn on WFBP)")

    chans = select_family(index_dicom_series(cfg.wfbp_dir, cache_dir=out), "wfbp")
    if not chans:
        raise ValueError(f"no WFBP series under {cfg.wfbp_dir}")
    vol, ref_img, zpos = read_series_volume(chans[0])
    pixel_mm = float(ref_img.GetSpacing()[0])
    logger.info("reference %s: %s, pixel %.4f mm", chans[0]["label"], vol.shape, pixel_mm)

    labels = load_segmentation(cfg.segmentation, ref_img)
    if labels is None:
        raise ValueError("the segmentation does not overlap the WFBP volume")
    present = sorted(int(v) for v in np.unique(labels) if v > 0)
    logger.info("segmentation labels present: %s", present)

    # ---- the module boundary --------------------------------------------------
    # Drawn as a filled ring.  Filling the holes returns the disc bounded by its OUTER
    # contour, which is what was asked for: the inner contour could clip an insert, and a
    # little of the surrounding phantom inside the ROI is the safer error.
    boundary = None
    if cfg.boundary_label in present:
        drawn = np.any(labels == int(cfg.boundary_label), axis=0)
        boundary = ndimage.binary_fill_holes(drawn)
        logger.info("module boundary from label %d: %d px drawn -> %d px filled "
                    "(%.0f mm2, outer contour)", cfg.boundary_label, int(drawn.sum()),
                    int(boundary.sum()), boundary.sum() * pixel_mm ** 2)
    else:
        logger.warning("boundary label %d not in the segmentation -- the water ROI and the "
                       "module outline will be unavailable", cfg.boundary_label)

    # ---- inserts ---------------------------------------------------------------
    inserts, gid = [], 0
    for lab in cfg.pair_labels:
        if lab not in present:
            logger.warning("pair label %d missing from the segmentation", lab)
            continue
        comps = _components_of_label(labels, lab)
        if len(comps) != 2:
            logger.warning("pair label %d split into %d component(s), expected 2 (one "
                           "HU-matched pair). Check the drawing.", lab, len(comps))
        for m, c in enumerate(comps):
            # Physical z as well as the index: the sources do not share a z axis, so
            # anything that has to find "the same slice" in another volume must go
            # through mm.  Indexing every modality with the reference's k put the QC
            # figure's own panels ~200 slices away from the phantom.
            inserts.append({**c, "geom_id": gid, "pair_label": int(lab), "member": m,
                            "layer_index": int(cfg.layer_of_label[lab]),
                            "radius_mm": float(cfg.insert_radius_mm),
                            "z_lo_mm": float(zpos[c["k_lo"]]),
                            "z_hi_mm": float(zpos[c["k_hi"]]),
                            "x_mm": float(ref_img.TransformContinuousIndexToPhysicalPoint(
                                (c["cx"], c["cy"], 0.0))[0]),
                            "y_mm": float(ref_img.TransformContinuousIndexToPhysicalPoint(
                                (c["cx"], c["cy"], 0.0))[1])})
            gid += 1
    if not inserts:
        raise ValueError("no insert ROIs found -- check the label values")
    logger.info("%d insert ROIs from %d pair labels (grown to the manual's %.1f mm radius)",
                len(inserts), len({i["pair_label"] for i in inserts}), cfg.insert_radius_mm)

    # ---- layer / role assignment, from HU alone --------------------------------
    hu = [r["mean"] for r in measure_rois(vol, inserts, pixel_mm, cfg)]
    assignment = assign_layers_and_roles(inserts, hu)
    for ins, h in zip(inserts, hu):
        a = assignment[ins["layer_index"]]
        ins["layer"] = a["layer"]
        ins["role"] = a["role_of_pair"][ins["pair_label"]]
        ins["hu_wfbp_ref"] = float(h)
    for li, a in sorted(assignment.items()):
        logger.info("segmentation layer %d -> layer %s (%s): base pair measured %+.0f HU, "
                    "datasheet %+.0f HU at 140 kV", li + 1, a["layer"],
                    LAYER_BASE_MATERIAL[a["layer"]], a["base_hu_measured"],
                    a["base_hu_nominal_140kV"])
        for p, role in sorted(a["role_of_pair"].items(), key=lambda kv: ROLES.index(kv[1])):
            logger.info("      pair label %d -> %-4s  (%.0f HU)", p, role,
                        a["pair_mean_hu"][p])

    # ---- QC only: does the drawing order agree with the HU-based roles? ---------
    _qc_drawing_order(inserts, assignment)

    # ---- the water region ------------------------------------------------------
    # Restricted to the annotated slices, so the reference is measured where the inserts
    # are rather than over parts of the module the segmentation never covered.
    match_k_lo = min(i["k_lo"] for i in inserts)
    match_k_hi = max(i["k_hi"] for i in inserts)
    water = None
    if boundary is not None:
        water = iq.noise_region(boundary, [{"cy": i["cy"], "cx": i["cx"],
                                            "radius_px": cfg.insert_radius_mm / pixel_mm}
                                           for i in inserts], pixel_mm,
                                insert_margin_mm=cfg.insert_margin_mm,
                                edge_margin_mm=cfg.edge_margin_mm)
        logger.info("water ROI (module body, datasheet 0 HU at 80-140 kV): %d px, %.0f mm2",
                    int(water.sum()), water.sum() * pixel_mm ** 2)
        sel = vol[match_k_lo:match_k_hi + 1][:, water] if water.any() else np.array([])
        if sel.size:
            logger.info("   measured there on WFBP %s: %+.1f HU (datasheet %+.0f) -- this "
                        "is the water reference the HU scale can be checked against",
                        chans[0]["label"], float(np.mean(sel)), BODY_HU)

    k_lo, k_hi = match_k_lo, match_k_hi
    z_lo, z_hi = float(min(zpos[k_lo], zpos[k_hi])), float(max(zpos[k_lo], zpos[k_hi]))

    # ---- ROIs as a LABEL VOLUME in physical space ------------------------------
    # Not as array indices.  The three sources do not share a grid: our reconstruction has
    # a slightly different slice spacing from the vendor's, and (until the production driver
    # adopts the recon_comparison fix) a different in-plane origin as well.  Index-based ROIs
    # would silently measure the wrong voxels there -- the same failure that made the first
    # reconstruction comparison unusable.  A label volume carries its own geometry and is
    # resampled onto whatever grid each source turns out to have.
    roi_lab = np.zeros(vol.shape, np.uint16)
    r_in = cfg.insert_radius_mm / pixel_mm * cfg.roi_inner_frac
    for ins in inserts:
        d = _disc(vol.shape[1:], ins["cy"], ins["cx"], r_in)
        roi_lab[ins["k_lo"]:ins["k_hi"] + 1][:, d] = ins["geom_id"] + 1
    if water is not None and water.any():
        blk = roi_lab[k_lo:k_hi + 1]
        blk[:, water & (blk[0] == 0)] = WATER_LABEL
        roi_lab[k_lo:k_hi + 1] = blk
    roi_img = sitk.GetImageFromArray(roi_lab)
    roi_img.CopyInformation(ref_img)
    sitk.WriteImage(roi_img, str(out / "roi_labels.nrrd"))
    logger.info("wrote %s (inserts 1-%d, water %d) -- resampled per source in stage 2",
                out / "roi_labels.nrrd", len(inserts), WATER_LABEL)
    logger.info("annotated slab: slices %d-%d, z = %.2f .. %.2f mm", k_lo, k_hi, z_lo, z_hi)

    # Module centre and radius in PHYSICAL mm, so the QC figure can crop to the phantom on
    # any grid instead of showing a 500 mm FOV that is mostly black.
    module = None
    if boundary is not None and boundary.any():
        ys, xs = np.nonzero(boundary)
        p0 = ref_img.TransformContinuousIndexToPhysicalPoint(
            (float(xs.mean()), float(ys.mean()), 0.0))
        module = {"cx_mm": float(p0[0]), "cy_mm": float(p0[1]),
                  "radius_mm": float(np.sqrt(boundary.sum() / np.pi) * pixel_mm)}
        logger.info("module: centre (%.1f, %.1f) mm, radius %.1f mm",
                    module["cx_mm"], module["cy_mm"], module["radius_mm"])

    match = {
        "segmentation": cfg.segmentation,
        "module": module,
        "reference": {"family": "wfbp", "label": chans[0]["label"],
                      "pixel_mm": pixel_mm, "shape": list(vol.shape),
                      # Needed to put our own reconstruction into this frame; Siemens
                      # references the patient frame to the table, not the isocentre.
                      "origin_x_mm": float(ref_img.GetOrigin()[0]),
                      "origin_y_mm": float(ref_img.GetOrigin()[1]),
                      "origin_z_mm": float(ref_img.GetOrigin()[2])},
        "inserts": inserts,
        "assignment": {str(k): _jsonable(v) for k, v in assignment.items()},
        "z_lo_mm": z_lo, "z_hi_mm": z_hi, "k_lo": k_lo, "k_hi": k_hi,
        "boundary_px": int(boundary.sum()) if boundary is not None else 0,
        "water_px": int(water.sum()) if water is not None else 0,
        "own_dir": cfg.own_dir, "wfbp_dir": cfg.wfbp_dir, "vmi_dir": cfg.vmi_dir,
        "insert_radius_mm": cfg.insert_radius_mm,
        "datasheet": {"insert_diameter_mm": INSERT_DIAMETER_MM, "body_hu": BODY_HU,
                      "base_hu_140kV": LAYER_BASE_HU_140,
                      "conc_relative": CONC_RELATIVE,
                      "conc_relative_is_inferred": CONC_IS_INFERRED,
                      "conc_relative_per_layer": CONC_RELATIVE_PER_LAYER},
    }
    _write_json(out / "decomp_match.json", match)
    np.save(out / "_masks.npy", np.stack([boundary if boundary is not None
                                          else np.zeros(vol.shape[1:], bool),
                                          water if water is not None
                                          else np.zeros(vol.shape[1:], bool)]))
    logger.info("wrote %s", out / "decomp_match.json")
    return match


def _qc_drawing_order(inserts, assignment):
    """
    QC ONLY: does the radial drawing order agree with the HU-derived roles?

    The inserts were drawn outermost pair first.  If that order lines up with base/low/high
    it is an independent confirmation of the assignment; if it does not, one of the two is
    wrong and the log says so.  Neither this nor the Ca/I side is allowed to influence the
    assignment itself -- an assumption that feeds the analysis it is meant to support stops
    being a check.
    """
    for li, a in sorted(assignment.items()):
        cent_y = np.mean([i["cy"] for i in inserts if i["layer_index"] == li])
        cent_x = np.mean([i["cx"] for i in inserts if i["layer_index"] == li])
        radius = {}
        for p in a["role_of_pair"]:
            members = [i for i in inserts if i["pair_label"] == p]
            radius[p] = float(np.mean([np.hypot(i["cy"] - cent_y, i["cx"] - cent_x)
                                       for i in members]))
        drawn_order = sorted(a["role_of_pair"])                 # label 1,2,3 = outer->inner
        by_radius = sorted(radius, key=radius.get, reverse=True)
        agree = drawn_order == by_radius
        logger.info("   QC layer %s: drawing order %s vs radial order %s -> %s",
                    a["layer"], drawn_order, by_radius,
                    "consistent" if agree else "DISAGREE (check the drawing)")


# ═════════════════════════════════════════════════════════════════════
# The Siemens 2-bin export -- probed, then imported (never solved)
# ═════════════════════════════════════════════════════════════════════
_PIXEL_TAGS = {
    "photometric": "0028|0004",
    "samples_per_px": "0028|0002",
    "rescale_intercept": "0028|1052",
    "rescale_slope": "0028|1053",
    "rescale_type": "0028|1054",
    "modality": "0008|0060",
    "image_type": "0008|0008",
}


def _series_pixel_metadata(path):
    """The tags that decide whether a vendor series is quantitative at all."""
    sitk = _sitk()
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    try:
        r.ReadImageInformation()
    except RuntimeError as exc:                                    # noqa: BLE001
        return {"error": str(exc)}
    out = {}
    for name, tag in _PIXEL_TAGS.items():
        out[name] = r.GetMetaData(tag).strip() if r.HasMetaDataKey(tag) else ""
    return out


def probe_siemens_export(folder, out_root=None):
    """
    List every series in a Siemens export with the metadata that decides usability.

    The one question that has to be answered before anything is built: was the iodine map
    (PBV) exported as a quantitative MONOCHROME2 image, or as an RGB colour blend?  An RGB
    series carries no densities at all -- it is a picture of a map, not a map -- and would
    read as three near-identical channels that average into something plausible-looking.
    Rescale slope/intercept and RescaleType say what the stored values MEAN, which is the
    other half of the same question.

    Purely diagnostic: reads one header per series plus one middle slice.
    """
    if out_root:
        Path(out_root).mkdir(parents=True, exist_ok=True)
    index = index_dicom_series(folder, cache_dir=out_root)
    if not index:
        logger.warning("no DICOM series found under %s", folder)
        return {}
    sitk = _sitk()
    report = {}
    print(f"\n{'series':46s} {'photom':>12s} {'spp':>4s} {'slope':>8s} "
          f"{'incept':>8s} {'type':>10s} {'n':>5s} {'z range (mm)':>22s}")
    print("-" * 122)
    for uid, s in sorted(index.items(), key=lambda kv: kv[1].get("number") or 0):
        files = s.get("files") or []
        if not files:
            continue
        meta = _series_pixel_metadata(files[len(files) // 2][1])
        zs = [z for z, _p in files]
        rec = {"desc": s.get("desc", ""), "number": s.get("number"),
               "n_slices": len(files), "z_lo_mm": min(zs), "z_hi_mm": max(zs),
               "pixel_mm": s.get("pixel_mm"), "thickness_mm": s.get("thickness_mm"),
               **meta}
        # Value range from ONE middle slice -- enough to tell a density map from an HU
        # image (an HU volume runs to -1000 in air; a density map does not).
        try:
            img = sitk.ReadImage(files[len(files) // 2][1])
            arr = sitk.GetArrayFromImage(img).astype(np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                rec["p01"], rec["p50"], rec["p99"] = (
                    float(np.percentile(arr, 1)), float(np.percentile(arr, 50)),
                    float(np.percentile(arr, 99)))
                rec["min"], rec["max"] = float(arr.min()), float(arr.max())
        except RuntimeError as exc:                                # noqa: BLE001
            rec["read_error"] = str(exc)
        report[uid] = rec
        print(f"{rec['desc'][:46]:46s} {rec.get('photometric', ''):>12s} "
              f"{rec.get('samples_per_px', ''):>4s} {rec.get('rescale_slope', ''):>8s} "
              f"{rec.get('rescale_intercept', ''):>8s} {rec.get('rescale_type', ''):>10s} "
              f"{rec['n_slices']:5d} {rec['z_lo_mm']:10.1f} .. {rec['z_hi_mm']:8.1f}")
        if "p01" in rec:
            print(f"{'':46s} values  p1={rec['p01']:.3g}  median={rec['p50']:.3g}  "
                  f"p99={rec['p99']:.3g}  [{rec['min']:.3g}, {rec['max']:.3g}]")
    print("-" * 122)
    rgb = [r["desc"] for r in report.values()
           if r.get("photometric", "").upper().startswith("RGB")
           or r.get("samples_per_px") not in ("", "1")]
    if rgb:
        print("WARNING: these series are COLOUR, not quantitative -- a picture of a map "
              "rather than a map:")
        for d in rgb:
            print(f"    {d}")
        print("  They cannot be scored. Ask for a re-export of the numeric series.")
    return report


def _pick_series(index, match, what):
    """The one series whose description contains `match` (case-insensitive)."""
    hits = [dict(s, uid=uid) for uid, s in index.items()
            if match.lower() in (s.get("desc") or "").lower()]
    if not hits:
        have = sorted({(s.get("desc") or "?") for s in index.values()})
        raise ValueError(
            f"no series matching {match!r} for the {what} map. Descriptions present: "
            f"{have}. Run --probe-siemens on this folder and set the match string.")
    if len(hits) > 1:
        # Concatenating two products into one stack is the failure this guards; the same
        # trap select_family() closes for the threshold and mono families.
        raise ValueError(
            f"{len(hits)} series match {match!r} for the {what} map: "
            f"{[h.get('desc') for h in hits]}. Point --s2bin-dir at the folder holding "
            f"exactly one, or narrow --s2bin-{what}-match.")
    return hits[0]


def _read_vendor_map(entry, z_range, what):
    """One vendor series as (array, sitk image), rejecting anything non-quantitative."""
    meta = _series_pixel_metadata(entry["files"][0][1])
    if meta.get("samples_per_px") not in ("", "1") or \
            meta.get("photometric", "").upper().startswith(("RGB", "PALETTE", "YBR")):
        raise ValueError(
            f"the {what} series '{entry.get('desc')}' is {meta.get('photometric')} with "
            f"{meta.get('samples_per_px')} samples/pixel -- a colour rendering, not a "
            f"quantitative map. Averaging its channels would produce a plausible-looking "
            f"number with no physical meaning, so it is refused. Re-export the numeric "
            f"series.")
    vol, img, _z = read_series_volume(entry, z_range=z_range)
    if vol.ndim != 3:
        raise ValueError(f"the {what} series read back with shape {vol.shape}, expected "
                         f"(Z, Y, X)")
    return vol, img, meta


# ═════════════════════════════════════════════════════════════════════
# Stage 1 -- decompose every source x mode x estimator
# ═════════════════════════════════════════════════════════════════════
def _run_key(source, mode, estimator):
    return f"{source}__{mode}__{estimator}"


def bin_domain_for(cfg: ValidationConfig, source: str) -> str:
    """
    The measurement stack this source is solved from -- see BIN_DOMAIN_BY_SOURCE.

    An unknown source falls back to 'cumulative' (feed as-is), which is the only choice that
    cannot silently invent a quantity: 'exclusive' would subtract channels that may not be
    nested at all.
    """
    if cfg.bin_domain:
        return cfg.bin_domain
    return (cfg.bin_domain_by_source or {}).get(source, "cumulative")


def _signature(cfg: ValidationConfig, source, mode, estimator, match):
    # The domain ACTUALLY used, not the config field: a run cached under one domain must be
    # redone when the source's domain changes, and a global override must invalidate too.
    domain = bin_domain_for(cfg, source)
    return {"source": source, "mode": mode, "estimator": estimator,
            "bin_domain": domain,
            # Which windows M was built on. Derivable from bin_domain today, but recorded
            # explicitly because it changed WITHOUT the domain changing: runs cached before
            # cumulative-M existed carry the same bin_domain and a different matrix, and
            # would otherwise be silently reused.
            "m_windows": "cumulative" if domain == "cumulative" else "disjoint",
            "threshold_option": cfg.threshold_option,
            # Present only when it is OFF, which keeps every run cached before this field
            # existed valid: those were all solved with the calibration on, since _one_run
            # omitted it from DecompConfig and inherited the dataclass default of True.
            # Adding the key unconditionally would invalidate all 24 for a change that moves
            # no number, while turning the calibration off still re-solves, which is the
            # case the signature is actually for.
            **({} if cfg.water_calibration else {"water_calibration": False}),
            # Same "present only when non-default" rule, and for the same reason: the
            # uniform weighting is what every cached run already used, and it changes no
            # number for the disjoint and monoenergetic sources in any case, since it
            # reaches only the windows this code integrates itself.
            **({} if cfg.signature_weighting == "uniform"
               else {"signature_weighting": cfg.signature_weighting,
                     "spectrum_kvp": float(cfg.spectrum_kvp),
                     "spectrum_water_cm": float(cfg.spectrum_water_cm)}),
            "z_lo_mm": match["z_lo_mm"], "z_hi_mm": match["z_hi_mm"]}


def stage1_decompose(cfg: ValidationConfig):
    """Run the solver for each source x mode x estimator over the annotated slab."""
    out = Path(cfg.out_root)
    match = json.loads((out / "decomp_match.json").read_text(encoding="utf-8"))
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    dirs = {"own": cfg.own_dir, "wfbp": cfg.wfbp_dir, "vmi": cfg.vmi_dir,
            SOURCE_2BIN: cfg.s2bin_dir}
    wanted_modes = [cfg.mode] + ([cfg.sensitivity_mode] if cfg.sensitivity_mode else [])

    for source in cfg.sources:
        folder = dirs.get(source)
        if not folder:
            logger.warning("no folder for source %r -- skipping", source)
            continue
        # The vendor export is IMPORTED, not solved: it has no basis to choose and no
        # estimator to run, so it is one run rather than modes x estimators.
        if source == SOURCE_2BIN:
            key = _run_key(source, VENDOR_MODE, VENDOR_ESTIMATOR)
            vdir = runs / key
            sig = _signature(cfg, source, VENDOR_MODE, VENDOR_ESTIMATOR, match)
            sig["iodine_match"] = cfg.s2bin_iodine_match
            sig["vnc_match"] = cfg.s2bin_vnc_match
            done = vdir / "run.json"
            if not cfg.force and done.exists():
                try:
                    if json.loads(done.read_text(encoding="utf-8")).get("signature") == sig:
                        logger.info("%-46s up to date -- skipping", key)
                        continue
                except (OSError, ValueError):
                    pass
            _import_vendor_run(cfg, match, folder, vdir, sig)
            continue
        for mode in wanted_modes:
            for est in cfg.estimators:
                key = _run_key(source, mode, est)
                vdir = runs / key
                sig = _signature(cfg, source, mode, est, match)
                done = vdir / "run.json"
                if not cfg.force and done.exists():
                    try:
                        if json.loads(done.read_text(encoding="utf-8")).get("signature") == sig:
                            logger.info("%-46s up to date -- skipping", key)
                            continue
                    except (OSError, ValueError):
                        pass
                _one_run(cfg, match, source, folder, mode, est, vdir, sig)
    logger.info("stage 1 complete")


def _fix_own_frame(volumes, ref, match, cfg: ValidationConfig):
    """
    Put our own reconstruction into the vendor's frame before decomposing it.

    output/reconstruction/ is written by python_reconstruction.py, which has NOT adopted the
    two corrections recon_comparison established for the frame: its in-plane
    origin is isocentre-centred where Siemens references the patient frame to the table, and
    its content is left-right mirrored relative to the vendor.  The segmentation is drawn on
    WFBP, so without this the ROIs would measure the wrong voxels -- silently, because every
    stage still runs.

    Both are output-space transforms, so nothing about the reconstruction is re-done here.
    Pass --no-own-frame-fix once the production driver applies them itself, or when
    --own-dir already points at corrected volumes (a recon_comparison sweep variant).
    """
    sitk = _sitk()
    ox = match["reference"].get("origin_x_mm")
    oy = match["reference"].get("origin_y_mm")
    vol = np.ascontiguousarray(np.asarray(volumes)[..., ::-1])       # x -> -x
    o = list(ref.GetOrigin())
    if ox is not None and oy is not None:
        o[0], o[1] = float(ox), float(oy)
    new_ref = sitk.Image(ref.GetSize(), ref.GetPixelID())
    new_ref.SetSpacing(ref.GetSpacing())
    new_ref.SetDirection(ref.GetDirection())
    new_ref.SetOrigin(tuple(o))
    logger.info("   own: applied the vendor frame -- mirrored x, origin (%.2f, %.2f) mm "
                "(was (%.2f, %.2f))", o[0], o[1], ref.GetOrigin()[0], ref.GetOrigin()[1])
    return vol, new_ref


def _import_vendor_run(cfg, match, folder, vdir, sig):
    """
    Write the Siemens 2-bin export into the layout stage 2 already reads.

    Stage 2 walks runs/*/ and loads decomp_<mode>_<material>.nii.gz for each material in the
    mode spec -- it never touches the solver.  So the vendor product becomes a first-class
    run simply by landing in that layout, and not one line of the metric code has to learn
    that this run was imported rather than computed.

    Only the iodine map is written as a material.  VNC is an HU image, not a density map;
    saving it as one would put an HU-scaled volume through metrics that assume densities.
    It goes beside the run as vnc_hu.nii.gz, where the QC figure and the pose check can use
    it and nothing scores it.
    """
    sitk = _sitk()
    vdir.mkdir(parents=True, exist_ok=True)
    key = _run_key(SOURCE_2BIN, VENDOR_MODE, VENDOR_ESTIMATOR)
    logger.info("%-46s importing (no solve)", key)
    t0 = time.time()
    err, channels, iodine_meta = None, [], {}
    z_range = (match["z_lo_mm"], match["z_hi_mm"])
    try:
        index = index_dicom_series(folder, cache_dir=Path(cfg.out_root))
        iod_entry = _pick_series(index, cfg.s2bin_iodine_match, "iodine")
        vol, img, meta = _read_vendor_map(iod_entry, z_range, "iodine")
        iodine_meta = dict(meta, desc=str(iod_entry.get("desc")))
        logger.info("   iodine map: '%s'  %s  slope=%s intercept=%s type=%r",
                    iod_entry.get("desc"), vol.shape, meta.get("rescale_slope"),
                    meta.get("rescale_intercept"), meta.get("rescale_type"))
        out_img = sitk.GetImageFromArray(vol.astype(np.float32))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(vdir / f"decomp_{VENDOR_MODE}_Iodine.nii.gz"))
        channels.append(str(iod_entry.get("desc")))

        # VNC is optional: the comparison stands on the iodine map alone, so a missing VNC
        # costs the QC panel and nothing else.
        try:
            vnc_entry = _pick_series(index, cfg.s2bin_vnc_match, "vnc")
            vnc, vimg, vmeta = _read_vendor_map(vnc_entry, z_range, "vnc")
            vout = sitk.GetImageFromArray(vnc.astype(np.float32))
            vout.CopyInformation(vimg)
            sitk.WriteImage(vout, str(vdir / "vnc_hu.nii.gz"))
            channels.append(str(vnc_entry.get("desc")))
            logger.info("   VNC (QC only): '%s'  %s", vnc_entry.get("desc"), vnc.shape)
            del vnc, vimg
        except ValueError as exc:                                  # noqa: BLE001
            logger.warning("   no VNC series (%s) -- QC panel unavailable, metrics "
                           "unaffected", exc)
        del vol, img
        gc.collect()
    except Exception as exc:                                       # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        logger.error("   FAILED %s", err)
    dt = time.time() - t0
    logger.info("   %.1f s", dt)
    _write_json(vdir / "run.json", {
        "signature": sig, "runtime_s": dt, "truncated": False, "error": err,
        "channels": channels, "mode": VENDOR_MODE, "estimator": VENDOR_ESTIMATOR,
        "source": SOURCE_2BIN, "imported": True, "n_energy_channels": 2,
        # Kept so the units question stays answerable from the results alone: the
        # datasheet gives no mg/ml, so nothing downstream can check the scale for us.
        "vendor_meta": iodine_meta,
    })


def _one_run(cfg, match, source, folder, mode, est, vdir, sig):
    vdir.mkdir(parents=True, exist_ok=True)
    domain = bin_domain_for(cfg, source)
    logger.info("%-46s ... (%s bins)", _run_key(source, mode, est), domain)
    dcfg = md.DecompConfig(
        mode=mode, estimator=est, bin_domain=domain,
        threshold_option=cfg.threshold_option,
        input_source=source,
        input_format="nifti" if source == "own" else "dicom",
        input_dir=folder, dicom_dir=None if source == "own" else folder,
        output_dir=str(vdir),
        z_slab_mm=(match["z_lo_mm"], match["z_hi_mm"]),
        joint_iters=min(cfg.joint_iters_cap, 10),
        water_calibration=cfg.water_calibration,
        signature_weighting=cfg.signature_weighting,
        spectrum_kvp=cfg.spectrum_kvp, spectrum_water_cm=cfg.spectrum_water_cm,
    )
    t0 = time.time()
    truncated, err = False, None
    try:
        volumes, channels, ref = md.load_energy_stack(dcfg)
        if source == "own" and cfg.own_frame_fix:
            volumes, ref = _fix_own_frame(volumes, ref, match, cfg)
        deadline = t0 + float(cfg.estimator_timeout_s)

        def _cancel():
            return time.time() > deadline

        result = md.decompose(volumes, dcfg, channels=channels, cancel=_cancel)
        md.save_decomp_result(result, ref, vdir)
        # Record the energy-channel HU too: the identification figure needs it, and
        # reloading the stack in stage 2 for three sources would double the I/O.
        np.save(vdir / "channels_hu.npy", volumes.astype(np.float32))
        chan_labels = [c.label for c in channels]
        del volumes
        gc.collect()
    except Exception as exc:                                   # noqa: BLE001
        err, chan_labels = f"{type(exc).__name__}: {exc}", []
        logger.error("   FAILED %s", err)
    dt = time.time() - t0
    if err is None and dt > cfg.estimator_timeout_s:
        truncated = True
        logger.warning("   ran past the %.0f s cap -- reported as TRUNCATED, not as a "
                       "converged result", cfg.estimator_timeout_s)
    logger.info("   %.1f s%s", dt, "  (truncated)" if truncated else "")
    _write_json(vdir / "run.json", {"signature": sig, "runtime_s": dt,
                                    "truncated": truncated, "error": err,
                                    "channels": chan_labels, "mode": mode,
                                    "estimator": est, "source": source,
                                    # Recorded per run: the sources are deliberately
                                    # solved in different domains, so a results file
                                    # that did not say which would be unreadable.
                                    "bin_domain": domain,
                                    "n_energy_channels": len(chan_labels) or None})


# ═════════════════════════════════════════════════════════════════════
# Stage 2 -- metrics
# ═════════════════════════════════════════════════════════════════════
def _measure_by_label(vol, lab_arr, inserts):
    """Mean/SD per insert, read from the resampled ROI label volume."""
    out = []
    for ins in inserts:
        sel = vol[lab_arr == ins["geom_id"] + 1]
        sel = sel[np.isfinite(sel)]
        out.append({"mean": float(sel.mean()) if sel.size else float("nan"),
                    "sd": float(sel.std()) if sel.size else float("nan"),
                    "n": int(sel.size)})
    return out


def _pairs_of(inserts):
    """[(pair_label, [insert, insert]), ...] preserving member order."""
    by = {}
    for i in inserts:
        by.setdefault(i["pair_label"], []).append(i)
    return sorted(by.items())


def carrier_value(values):
    """
    Of several readings of one agent, the one from the insert that CARRIES it.

    The carrier is the largest reading by MAGNITUDE, not the largest signed one, and the
    difference is not academic: several solves put a whole material map below zero (all
    three CaHA panels of the proportionality figure sit in the negative half, and some
    iodine runs read negative on the very lesions that carry the most iodine).  A signed
    `max` over such a map returns the reading CLOSEST TO ZERO -- the insert with the least
    agent in it -- so every metric built on "the member that carries the agent" silently
    described the wrong insert, and the ones that divide by it divided by something near
    zero.  That produced a leak ratio of -5066 on a sign-inverted run and leak ratios above
    1 on an axis where 1 is defined as total failure.

    The SIGN IS KEPT on the returned value.  Selection and orientation are separate
    questions: the caller still needs to see that the map is upside down, and
    `specificity_metrics` reports exactly that as `reference_negative`.  Absorbing it here
    would hide a failed solve behind a healthy-looking number.
    """
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    return float(v[int(np.argmax(np.abs(v)))]) if v.size else float("nan")


def carrier_insert(members, values):
    """
    Which insert of a group carries the agent -- the one `carrier_value` would select.

    Separate from `carrier_value` because the figures need the insert (to outline it, to
    crop it), not its reading.  Same rule, so a figure that claims to show the member the
    metric used actually shows it; a signed max here would outline one insert while the fit
    described another.  NaN readings sort last rather than winning by accident.
    """
    def _key(m):
        g = m["geom_id"]
        v = values[g] if 0 <= g < len(values) else float("nan")
        return abs(v) if np.isfinite(v) else -np.inf
    return max(members, key=_key)


def _mean_of_finite(values):
    """
    Mean over the finite entries; NaN when there are none -- and no warning either way.

    `np.nanmean` of an all-NaN list is exactly right here (no number to report) but says so
    with a RuntimeWarning on every call, which in a run over ~24 runs x 9 pairs buries the
    log and trains the reader to scroll past warnings.  The condition it flags is reported
    once, deliberately, in the summary's `pairs` column instead.
    """
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def doped_pairs(sep):
    """
    The low/high pairs this run actually measured -- the only ones that may be averaged.

    A pair whose ROI produced no voxels on this source's grid reads NaN in every map, and
    NaN is not a result: `sign_ok` on two NaN deltas evaluates False, so a pair that was
    never measured would be reported as the solver tracking HU rather than material -- the
    one failure this phantom exists to expose, credited to a missing measurement.  They are
    dropped here and counted in the summary, so the loss is visible rather than scored.
    """
    return [s for s in sep if s.get("role") in ("low", "high") and s.get("measured", True)]


def separation_metrics(inserts, dens, noise, agents=AGENTS):
    """
    Per HU-matched pair: how far apart the two members land in material space.

    The pair members are the same HU by construction (datasheet), so any difference in the
    material maps IS material information.  `sign_ok` demands that the two agents move in
    OPPOSITE directions: both moving together means the decomposition is tracking HU rather
    than material -- the one failure the phantom exists to expose.  It needs both agents, so
    it is absent from a single-agent (rank-limited) run rather than reported as False.

    `leak_ratio_<agent>` is the same question asked without any units: the agent's reading on
    the pair member that does NOT carry it, over its reading on the member that does.  0 is
    perfect separation, 1 is pure HU tracking.  Being a ratio within one map it survives a
    change of unit, which is what lets a vendor map in mg/ml be compared with ours in g/cm3.
    """
    out = []
    for lab, mem in _pairs_of(inserts):
        if len(mem) != 2:
            continue
        a, b = mem[0]["geom_id"], mem[1]["geom_id"]
        rec = {"pair_label": lab, "layer": mem[0].get("layer"),
               "role": mem[0].get("role"), "geom_ids": [a, b]}
        for mat in agents:
            if mat not in dens:
                continue
            d = dens[mat][a] - dens[mat][b]
            s = float(np.hypot(noise[mat][a], noise[mat][b]) / np.sqrt(2))
            rec[f"delta_{mat}"] = float(d)
            rec[f"sep_index_{mat}"] = float(abs(d) / s) if s > 0 else float("nan")
            # The carrier is the reading with the larger MAGNITUDE (see carrier_value); the
            # other member is the one this agent should be blind to.  Both sides of the
            # ratio are magnitudes, so it lands in [0, 1] -- 0 separated, 1 HU tracking --
            # whichever side of zero the map happens to sit on.  Signed, a map offset below
            # zero picked the wrong member and returned ratios above 1, i.e. past the value
            # the axis defines as total failure, and -5066 on an inverted one.  Base pairs
            # carry neither agent, so the ratio is meaningless there and is left out rather
            # than reported as ~1.
            if mem[0].get("role") in ("low", "high"):
                pair = [dens[mat][a], dens[mat][b]]
                hi = carrier_value(pair)
                lo = min(pair, key=lambda x: abs(x) if np.isfinite(x) else np.inf)
                rec[f"leak_ratio_{mat}"] = (float(abs(lo) / abs(hi)) if np.isfinite(hi)
                                            and abs(hi) > 1e-12 else float("nan"))
        # Measured = every agent this run HAS read a number on both members.  Judged on the
        # agents present in `dens`, not on `agents`, so a rank-limited run is not marked
        # unmeasured for lacking the map it cannot produce -- that is a separate statement,
        # made by `is_rankable`.  An ROI that fell outside the volume gives mean = sd = NaN
        # (`_measure_by_label`), which is what this catches.
        present = [m for m in agents if m in dens]
        rec["measured"] = bool(present) and all(
            np.isfinite(rec.get(f"delta_{m}", np.nan)) for m in present)
        if rec["measured"] and all(f"delta_{m}" in rec for m in AGENTS):
            rec["sign_ok"] = bool(rec["delta_Iodine"] * rec["delta_HA"] < 0)
        out.append(rec)
    return out


def specificity_metrics(inserts, dens, agents=AGENTS):
    """
    Base pairs contain neither agent, so both maps should read zero there.

    `leak_frac` normalises that by the agent's own reading on the `high` lesions, which is
    what makes the number comparable between maps in different units -- the absolute mean is
    in whatever unit the map came in, and a vendor mg/ml map cannot be held against our
    g/cm3 one without it.

    BOTH SIDES OF THE RATIO ARE MAGNITUDES.  The numerator always was; the denominator was
    signed, so a run whose iodine map reads NEGATIVE on the very lesions that carry the most
    iodine produced a negative leak fraction -- and since the axis and the console both say
    lower is better, the most inverted map in the set was reported as the best of the 24.  A
    base lesion holds no iodine, so only how far from zero it reads can mean anything; which
    side of zero cannot.  The signed mean is kept as `mean`, so nothing is lost by taking
    the magnitude here.

    A negative `high_reference` is not a detail to absorb into an absolute value: it says
    the map is sign-inverted, i.e. not a concentration map at all.  It is recorded as
    `reference_negative` and reported, because a small leak fraction computed against an
    inverted reference is not a good result.
    """
    ids = [i["geom_id"] for i in inserts if i.get("role") == "base"]
    hi_ids = [i["geom_id"] for i in inserts if i.get("role") == "high"]
    out = {}
    for mat in agents:
        if mat not in dens or not ids:
            continue
        v = np.array([dens[mat][i] for i in ids], float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        rec = {"mean": float(v.mean()), "sd": float(v.std()),
               "max_abs": float(np.max(np.abs(v))), "n": int(v.size)}
        ref = np.array([dens[mat][i] for i in hi_ids], float)
        ref = ref[np.isfinite(ref)]
        # The reference is the LARGEST-MAGNITUDE reading among the high lesions, not the
        # largest signed one.  `hi_ids` holds both members of each high pair and only one of
        # them carries this agent, so a plain max picks the carrier on a healthy map -- but
        # on a map offset or inverted below zero it picks the NON-carrier, whose reading is
        # near zero, and a near-zero denominator turns a broken solve into a spectacular
        # leak fraction.  Taking the extreme by magnitude selects the same insert either
        # way; its sign is then reported separately rather than silently scaling the ratio.
        # The mean over all six is not the alternative: it would halve the reference and
        # double every leak fraction, since half those inserts carry the other agent.
        scale = carrier_value(ref)
        rec["high_reference"] = scale
        rec["reference_negative"] = bool(np.isfinite(scale) and scale < 0)
        rec["leak_frac"] = (float(np.mean(np.abs(v)) / abs(scale))
                            if np.isfinite(scale) and abs(scale) > 1e-12 else float("nan"))
        out[mat] = rec
    return out


def linearity_metrics(inserts, dens, agents=AGENTS):
    """
    Measured density against the INFERRED relative concentration 0 : 1 : 3.1.

    Reported per layer AND per agent, with the layer-specific ratio, because the 3.1 is our
    interpolation rather than a QRM figure -- the spread between layers is part of the
    result, not noise to be averaged away.
    """
    out = {}
    for mat in agents:
        if mat not in dens:
            continue
        for layer in ("A", "B", "C"):
            sel = [i for i in inserts if i.get("layer") == layer]
            if not sel:
                continue
            # In each doped pair the member with the larger density of THIS agent is the
            # one carrying it; the other carries the opposite agent.
            xs, ys = [], []
            for _lab, mem in _pairs_of(sel):
                if len(mem) != 2:
                    continue
                role = mem[0].get("role")
                vals = [dens[mat][m["geom_id"]] for m in mem]
                # carrier_value, not max: on a map that sits below zero a signed max picks
                # the member with the LEAST agent, and the fitted line then describes the
                # wrong three inserts while R2 -- which is blind to that -- still reads ~1.
                take = float(np.mean(vals)) if role == "base" else carrier_value(vals)
                xs.append(CONC_RELATIVE.get(role, np.nan))
                ys.append(take)
            xs, ys = np.array(xs, float), np.array(ys, float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() < 3:
                continue
            slope, icept = np.polyfit(xs[ok], ys[ok], 1)
            pred = slope * xs[ok] + icept
            ss_res = float(np.sum((ys[ok] - pred) ** 2))
            ss_tot = float(np.sum((ys[ok] - ys[ok].mean()) ** 2))
            by_role = {r: float(y) for r, y in zip(
                [i.get("role") for _l, m in _pairs_of(sel) for i in m[:1]], ys)}
            lo, hi = by_role.get("low"), by_role.get("high")
            out[f"{mat}/{layer}"] = {
                "slope": float(slope), "intercept": float(icept),
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "measured_high_over_low": (float(hi / lo) if lo and hi and abs(lo) > 1e-9
                                           else float("nan")),
                "inferred_ratio": CONC_RELATIVE_PER_LAYER.get(layer),
                "x": xs[ok].tolist(), "y": ys[ok].tolist()}
    return out


def response_metrics(linearity, agents=AGENTS):
    """
    Does each agent's map go UP when THAT agent's concentration goes up?

    This is the question `sign_ok` cannot answer, and the gap is not hypothetical.  The sign
    test asks whether the two agent maps move in OPPOSITE directions across an HU-matched
    pair.  It rules out a solver that splits one attenuation between two columns with the
    SAME sign.  It does not rule out a solver that splits it with OPPOSITE signs -- which is
    exactly what an ill-conditioned basis does, since the near-null direction of `M` has
    mixed signs by construction.  Such a run passes every sign test while neither map is
    assigned to the material it is named after.

    `linearity_metrics` already fits a line per agent per layer over base -> low -> high of
    that agent's own concentration.  The SIGN of that slope is the missing verdict, and it
    was being discarded: `measured_high_over_low` is a ratio of two readings that may both
    be negative, so a map anti-correlated with its own agent still reports a healthy-looking
    positive ratio.  A map whose slope is negative in every layer is tracking something, but
    it is not the agent on the label.

    Reported per agent as a fraction of layers, not as a single boolean, because one layer
    out of three is a different statement from three out of three, and averaging it away
    would hide which.  `anti_correlated` is the strong case: no layer responds upward.
    """
    out = {}
    for mat in agents:
        slopes = [e["slope"] for k, e in linearity.items()
                  if k.split("/")[0] == mat and np.isfinite(e.get("slope", np.nan))]
        if not slopes:
            continue
        n_pos = sum(1 for s in slopes if s > 0)
        out[mat] = {"slopes": [float(s) for s in slopes],
                    "n_layers": len(slopes), "n_positive": int(n_pos),
                    "response_ok": bool(n_pos == len(slopes)),
                    "anti_correlated": bool(n_pos == 0)}
    return out


def inverted_agents(run, agents=AGENTS):
    """
    Which agent maps of this run read negative on the lesions carrying the most agent.

    `specificity_metrics` records this per agent as `reference_negative`; until now only the
    iodine flag reached a printed table, through the RQ1 block.  A run can be inverted in one
    map and upright in the other, so the answer is a list rather than a flag.
    """
    spec = run.get("specificity") or {}
    return [m for m in agents if (spec.get(m) or {}).get("reference_negative")]


def anti_correlated_agents(run, agents=AGENTS):
    """Agent maps that fall as their own agent rises in every layer measured."""
    resp = run.get("response") or {}
    return [m for m in agents if (resp.get(m) or {}).get("anti_correlated")]


def response_fraction(run, agents=AGENTS):
    """Fraction of agent x layer fits that respond upward; NaN when none were fitted."""
    resp = run.get("response") or {}
    tot = sum((resp.get(m) or {}).get("n_layers", 0) for m in agents)
    pos = sum((resp.get(m) or {}).get("n_positive", 0) for m in agents)
    return float(pos) / tot if tot else float("nan")


def mean_leak_ratio(run, agents=AGENTS):
    """
    Mean over doped pairs and agents of the agent's reading on the member that does NOT
    carry it, over its reading on the member that does.  0 is separated, 1 is HU tracking.

    Computed per pair by `separation_metrics` and until now aggregated only inside the RQ1
    block, for iodine alone.  It belongs beside `sep idx` in the main table: a large
    separation at a leak ratio near 1 is a map that follows the attenuation, and the two
    numbers say that together in a way neither says alone.
    """
    sep = doped_pairs(run.get("separation", []))
    return _mean_of_finite([_mean_of_finite([s.get(f"leak_ratio_{m}") for m in agents])
                            for s in sep])


def cross_layer_metrics(inserts, dens, agents=AGENTS):
    """
    Same agent, same level, three different base materials -> should give one density.

    Needs no calibration at all, which is what makes it worth having: any spread across the
    layers is base-material crosstalk and nothing else.  `cv` is a ratio, so it survives a
    change of unit -- which is what makes it usable for the 4-bin vs 2-bin question, where
    the two sides do not share a scale.
    """
    out = {}
    for mat in agents:
        if mat not in dens:
            continue
        for role in ("low", "high"):
            vals = []
            for layer in ("A", "B", "C"):
                sel = [i for i in inserts
                       if i.get("layer") == layer and i.get("role") == role]
                if len(sel) == 2:
                    # By magnitude (carrier_value), for the same reason as everywhere else:
                    # a signed max on a map offset below zero returns the insert with the
                    # LEAST of this agent.
                    vals.append(carrier_value([dens[mat][i["geom_id"]] for i in sel]))
            v = np.array(vals, float)
            v = v[np.isfinite(v)]
            if v.size >= 2:
                out[f"{mat}/{role}"] = {
                    "values": v.tolist(), "mean": float(v.mean()), "sd": float(v.std()),
                    "cv": float(v.std() / abs(v.mean())) if abs(v.mean()) > 1e-12
                    else float("nan")}
    return out


def stage2_metrics(cfg: ValidationConfig):
    sitk = _sitk()
    out = Path(cfg.out_root)
    match = json.loads((out / "decomp_match.json").read_text(encoding="utf-8"))
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "qc").mkdir(parents=True, exist_ok=True)

    inserts = match["inserts"]
    pixel_mm = float(match["reference"]["pixel_mm"])
    masks = np.load(out / "_masks.npy")
    boundary, water = masks[0].astype(bool), masks[1].astype(bool)
    roi_img = sitk.ReadImage(str(out / "roi_labels.nrrd"))

    results = {"match": match, "runs": {}}
    for vdir in sorted((out / "runs").glob("*")):
        rj = vdir / "run.json"
        if not rj.exists():
            continue
        info = json.loads(rj.read_text(encoding="utf-8"))
        if info.get("error"):
            logger.warning("%s failed at stage 1 (%s) -- skipped", vdir.name, info["error"])
            continue
        mode, source = info["mode"], info["source"]
        spec = modes.get_mode(mode)
        dens, noise, lab_arr = {}, {}, None
        for mat in spec.materials:
            p = vdir / f"decomp_{mode}_{mat}.nii.gz"
            if not p.exists():
                continue
            img = sitk.ReadImage(str(p))
            vol = sitk.GetArrayFromImage(img).astype(np.float32)
            if lab_arr is None:
                # Resample the ROIs onto THIS source's grid, in physical space. The sources
                # differ slightly in slice spacing and, for 'own', in origin, so
                # reusing array indices across them would measure the wrong voxels.
                lab_arr = sitk.GetArrayFromImage(sitk.Resample(
                    roi_img, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                    sitk.sitkUInt16))
                n_hit = sum(1 for i in inserts if np.any(lab_arr == i["geom_id"] + 1))
                logger.info("  %s: %d/%d insert ROIs land inside this volume",
                            vdir.name, n_hit, len(inserts))
                if n_hit < len(inserts):
                    logger.warning("  %s: %d ROI(s) fell outside -- the grids do not "
                                   "overlap as expected. Check qc/qc_roi_placement.png "
                                   "before using these numbers.",
                                   vdir.name, len(inserts) - n_hit)
            stats = _measure_by_label(vol, lab_arr, inserts)
            dens[mat] = [s["mean"] for s in stats]
            noise[mat] = [s["sd"] for s in stats]
            wsel = vol[lab_arr == WATER_LABEL]
            if wsel.size:
                info.setdefault("water_density", {})[mat] = float(np.mean(wsel))
            del vol, img
            gc.collect()

        rec = dict(info)
        rec["materials"] = list(dens)
        rec["density"] = dens
        rec["density_sd"] = noise
        rec["separation"] = separation_metrics(inserts, dens, noise)
        rec["specificity"] = specificity_metrics(inserts, dens)
        rec["linearity"] = linearity_metrics(inserts, dens)
        rec["response"] = response_metrics(rec["linearity"])
        rec["cross_layer"] = cross_layer_metrics(inserts, dens)
        # RQ1 (4 bins vs 2) is scored on iodine alone, on BOTH sides -- see RQ1_AGENTS.
        # Computed for every run, not only the vendor one, because the restriction has to
        # apply to us as well or the comparison credits us with the HA map the 2-bin mode
        # cannot produce.  Stored beside the full block so the two questions stay legible
        # and neither can quietly borrow the other's numbers.
        rec["rq1_iodine"] = {
            "agents": list(RQ1_AGENTS),
            "separation": separation_metrics(inserts, dens, noise, agents=RQ1_AGENTS),
            "specificity": specificity_metrics(inserts, dens, agents=RQ1_AGENTS),
            "linearity": linearity_metrics(inserts, dens, agents=RQ1_AGENTS),
            "response": response_metrics(linearity_metrics(inserts, dens,
                                                           agents=RQ1_AGENTS),
                                         agents=RQ1_AGENTS),
            "cross_layer": cross_layer_metrics(inserts, dens, agents=RQ1_AGENTS),
        }
        rec["n_energy_channels"] = info.get("n_energy_channels", len(info.get("channels")
                                                                    or []) or None)
        results["runs"][vdir.name] = rec
        logger.info("measured %s", vdir.name)

    _write_json(out / "metrics" / "decomp_validation.json", results)
    _figures(cfg, out, results, boundary, water)
    _print_summary(results)
    logger.info("wrote %s", out / "metrics" / "decomp_validation.json")
    return results


# ═════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════
_LAYER_COL = {"A": "tab:blue", "B": "tab:red", "C": "tab:green"}
_ROLE_MARK = {"base": "o", "low": "s", "high": "^"}


def _separation_figure(path, run, inserts, cfg):
    """Small multiples: one panel per layer, three HU-matched pairs each."""
    plt = _mpl()
    if plt is None or not run.get("separation"):
        return
    dens = run["density"]
    if not all(m in dens for m in AGENTS):
        return
    layers = [l for l in ("A", "B", "C")
              if any(i.get("layer") == l for i in inserts)]
    if not layers:
        return
    # Written once and used twice: on the combined figure's leftmost panel and again by
    # _save_panels for every crop. Two copies of the same string is how the wrong arrow
    # survived on the trade-off figure -- the copy the thesis uses was not the one edited.
    # Two lines. Rotated, it is the longest LINE that has to fit the panel height, and the
    # spelled-out name on one line is wider than the axes is tall, so the crop cut it at
    # both ends.
    _CAHA_YLABEL = "calcium hydroxyapatite\ndensity  (g/cm$^3$)"
    fig, axes = plt.subplots(1, len(layers),
                             figsize=fp.panel_figsize(len(layers), frac=0.49,
                                                      height_in=3.6),
                             sharex=True, sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], layers):
        for _lab, mem in _pairs_of([i for i in inserts if i.get("layer") == layer]):
            if len(mem) != 2:
                continue
            x = [dens["Iodine"][m["geom_id"]] for m in mem]
            y = [dens["HA"][m["geom_id"]] for m in mem]
            ax.plot(x, y, "-", color="0.55", lw=1.3, zorder=1)
            ax.scatter(x, y, marker=_ROLE_MARK.get(mem[0].get("role"), "o"), s=110,
                       facecolor=_LAYER_COL.get(layer, "0.5"), edgecolor="k", lw=0.7,
                       zorder=3)
        ax.axhline(0, color="0.75", lw=0.8)
        ax.axvline(0, color="0.75", lw=0.8)
        ax.set_title(f"layer {layer}  ({LAYER_BASE_MATERIAL.get(layer, '')})",
                     fontsize=fp.PRINT_FONT_PT)
        ax.set_xlabel("iodine density  (g/cm$^3$)", fontsize=fp.PRINT_FONT_PT)
        ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
        ax.grid(alpha=0.22)
    axes[0][0].set_ylabel(_CAHA_YLABEL, fontsize=fp.PRINT_FONT_PT)
    from matplotlib.lines import Line2D
    # Proxy handles on the figure, not inside panel 0: loc="best" put the key on top of the
    # very pair lines it labels whenever the cloud filled the axes.
    for k, m in _ROLE_MARK.items():
        axes[0][0].add_line(Line2D([], [], marker=m, ls="", color="0.45", mec="k",
                                   label=f"{k} level"))
    _share_axes(axes[0], logy=False, free=cfg.free_axes)
    # The run is named in the caption; three internal keys in a title are three names the
    # reader has no way to expand.
    fig.suptitle("Calcium against iodine in the material plane",
                 fontsize=fp.PRINT_FONT_PT + 1)
    _legend_below(fig, axes[0][0], ncol=3, fontsize=fp.PRINT_FONT_PT - 1)
    fig.savefig(path, dpi=200)
    if cfg.split_figures:
        _save_panels(fig, axes[0], Path(path).parent, Path(path).stem, layers,
                     ylabel=_CAHA_YLABEL, dpi=200)
    plt.close(fig)


def _linearity_figure(path, run, cfg):
    plt = _mpl()
    lin = run.get("linearity") or {}
    if plt is None or not lin:
        return
    fig, axes = plt.subplots(1, len(AGENTS),
                             figsize=fp.panel_figsize(len(AGENTS), frac=0.49,
                                                      height_in=3.6), squeeze=False)
    for ax, mat in zip(axes[0], AGENTS):
        fits = []
        for layer in ("A", "B", "C"):
            e = lin.get(f"{mat}/{layer}")
            if not e:
                continue
            ax.plot(e["x"], e["y"], "o-", color=_LAYER_COL[layer], label=f"layer {layer}")
            # Two significant figures, one short line per layer. The four-decimal R2 and
            # the "=" labels were what forced this box down to an unreadable size, and
            # neither extra digit was doing any work.
            fits.append(f"{layer}   R$^2$ {e['r2']:.2f}   "
                        f"hi/lo {e['measured_high_over_low']:.1f}")
        # Per-panel fit numbers in the corner, not in the legend: they differ between the
        # two agents, so one shared legend would attach iodine's R2 to the CaHA panel.
        if fits:
            ax.text(0.03, 0.97, "\n".join(fits), transform=ax.transAxes,
                    fontsize=fp.PRINT_FONT_PT - 1, va="top",
                    bbox={"fc": "white", "ec": "0.7", "alpha": 0.85, "pad": 3.0})
        ax.set_xlabel("relative agent concentration\n(0 : 1 : 3.1, inferred)",
                      fontsize=fp.PRINT_FONT_PT)
        ax.set_ylabel(f"{MATERIAL_LABELS.get(mat, mat)}\ndensity  (g/cm$^3$)",
                      fontsize=fp.PRINT_FONT_PT)
        ax.set_title(MATERIAL_LABELS.get(mat, mat), fontsize=fp.PRINT_FONT_PT)
        ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
        ax.grid(alpha=0.25)
    # Which run produced it goes in the caption: three internal keys in a title are three
    # names the reader cannot expand.
    fig.suptitle("Proportionality", fontsize=fp.PRINT_FONT_PT + 1)
    _legend_below(fig, axes[0][0], ncol=3, fontsize=fp.PRINT_FONT_PT - 1)
    fig.savefig(path, dpi=200)
    if cfg.split_figures:
        _save_panels(fig, axes[0], Path(path).parent, Path(path).stem, AGENTS, dpi=200)
    plt.close(fig)


def _estimator_tradeoff_figure(path, results, cfg, split=False):
    """
    (F) Does a more expensive estimator buy anything?  One line per source, per metric.

    Replaces a bias-against-noise scatter that could not be read: 24 runs carried three
    encodings at once (colour = estimator, shape = source, area = runtime), most points
    landed exactly on top of one another -- `wls` was invisible under another estimator in
    every panel -- and two oversized markers were clipped by the axes.  Nothing about that
    plot showed the one thing being asked, which is whether moving along the estimator list
    changes anything.

    So the estimator becomes the x AXIS, in order of cost, and the answer is the slope of a
    line: flat means the extra work bought nothing for that source.  Runtime leaves the
    marker area and becomes its own panel -- it is a different quantity in different units,
    and an area encoding cannot be read off to better than a factor of two anyway.
    """
    plt = _mpl()
    if plt is None:
        return
    order = [e for e in cfg.estimators]
    series = {}                                   # (source, mode) -> {estimator: (b, n, t)}
    for key, r in results["runs"].items():
        # The imported vendor run is not an estimator choice, and its map is in the
        # vendor's own unit -- putting it on a g/cm3 bias/noise axis would rescale it
        # around a point that cannot be compared there anyway.
        if r.get("source") == SOURCE_2BIN:
            continue
        spec = r.get("specificity", {})
        bias = _mean_of_finite([abs(spec[m]["mean"]) for m in spec])
        sd = _mean_of_finite([_mean_of_finite(v)
                              for v in r.get("density_sd", {}).values()])
        series.setdefault((r["source"], r["mode"]), {})[r["estimator"]] = (
            bias, sd, float(r.get("runtime_s", np.nan)), bool(r.get("truncated")))
    if not series:
        return
    est_seen = [e for e in order if any(e in v for v in series.values())]
    est_seen += sorted({e for v in series.values() for e in v} - set(est_seen))
    if not est_seen:
        return
    x = np.arange(len(est_seen), dtype=float)

    src_col = {"own": "tab:blue", "wfbp": "tab:red", "vmi": "tab:green"}
    modes_seen = sorted({m for _s, m in series})
    # Basis as line style, source as colour: the figure has two factors and only one of
    # them is the question, so the second must not compete for the same channel.
    mode_ls = {m: ls for m, ls in zip(modes_seen, ["-", "--", ":", "-."])}

    # Short enough to fit the panel height. Rotated 90 degrees, a label longer than the
    # panel is tall runs off the canvas and is cut at BOTH ends by the panel crop -- which
    # is how "bias |apparent agent where there is none| ... down better" reached the page
    # as "as |apparent agent" and "down bet". What the quantity means belongs in the
    # caption, which has room for it; the axis carries the name and the unit.
    # Balanced across the two lines: rotated, it is the LONGEST line that has to fit the
    # panel height, so putting the unit on the second line costs nothing and stops
    # "within-insert noise SD (g/cm3)" from losing its "better" off the top.
    panels = (("bias\n(g/cm$^3$)   $\\downarrow$ better", 0, False),
              ("within-insert noise SD\n(g/cm$^3$)   $\\downarrow$ better", 1, False),
              ("runtime  (s)", 2, True))
    fig, axes = plt.subplots(1, 3, figsize=fp.panel_figsize(3, frac=0.49, height_in=3.5))
    for ax, (ylab, idx, logy) in zip(axes, panels):
        for si, ((src, mode), by_est) in enumerate(sorted(series.items())):
            y = np.array([by_est.get(e, (np.nan,) * 4)[idx] for e in est_seen], float)
            if not np.any(np.isfinite(y)):
                continue
            # A small deterministic x offset per series. Two series that land on exactly
            # the same values -- which is the ANSWER whenever the basis does not matter --
            # would otherwise draw one line and hide the other, the failure that made the
            # scatter this replaces unreadable.
            dx = 0.055 * (si - 0.5 * (len(series) - 1)) / max(1, 0.5 * (len(series) - 1))
            ax.plot(x + dx, y, marker="o", ms=4.5, lw=1.3, color=src_col.get(src, "0.4"),
                    ls=mode_ls.get(mode, "-"), alpha=0.9,
                    label=f"{fp.pretty(src, short=True)}  /  {fp.pretty(mode)}")
            trunc = [i for i, e in enumerate(est_seen)
                     if by_est.get(e, (0, 0, 0, False))[3]]
            if trunc:
                ax.plot(x[trunc] + dx, y[trunc], "x", ms=9, mew=1.8, color="crimson",
                        zorder=5, label="stopped at the time cap")
        ax.set_xticks(x)
        # Horizontal, not rotated: four short names fit, and the rotated two-line labels on
        # the other bar figures were landing under the neighbouring group.
        ax.set_xticklabels([fp.pretty(e).replace(" + ", "\n+ ") for e in est_seen],
                           fontsize=fp.PRINT_FONT_PT - 1)
        # Down, not left: this is a y axis, and an arrow pointing the wrong way is worse
        # than none -- the reader trusts it.
        ax.set_ylabel(ylab, fontsize=fp.PRINT_FONT_PT)
        ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
        ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale("log")
    for ax in axes:
        ax.set_xlabel("estimator, in order of cost", fontsize=fp.PRINT_FONT_PT)
    fig.suptitle("Estimator and basis", fontsize=fp.PRINT_FONT_PT + 1)
    _legend_below(fig, axes, ncol=3, fontsize=fp.PRINT_FONT_PT - 1)
    fig.savefig(path, dpi=200)
    if split:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem,
                     ["bias", "noise", "runtime"], dpi=200)
    plt.close(fig)


def _short_mode(name):
    """Basis name for an axis tick: the shared 'phantom_' prefix is not distinguishing."""
    return str(name)[len("phantom_"):] if str(name).startswith("phantom_") else str(name)


# Eight groups across one panel leave about 0.4 in each, which the full basis name does not
# fit into at any legible size. Only the base material distinguishes the two bases -- both
# carry Ca and I -- so that word alone says which is which.
_BASIS_TICK = {"phantom_ca_i": "Soft tissue", "phantom_water_ca_i": "Water",
               "siemens_2bin": "Vendor 2-bin"}


def _basis_tick(name):
    return _BASIS_TICK.get(str(name), _short_mode(name))


def _source_comparison_figure(path, results, cfg):
    plt = _mpl()
    if plt is None:
        return
    rows, ref_2bin = {}, float("nan")
    for key, r in results["runs"].items():
        sep = doped_pairs(r.get("separation", []))
        if not sep:
            continue
        # The 2-bin import belongs on this figure as a REFERENCE LINE, not as a fourth
        # source bar: RQ2 compares reconstructions of the same 4-bin data, and a run with
        # a different channel count is not one of those.
        if r.get("source") == SOURCE_2BIN:
            ref_2bin = _mean_of_finite([s.get("sep_index_Iodine") for s in sep])
            continue
        idx = _mean_of_finite([_mean_of_finite([s.get(f"sep_index_{m}") for m in AGENTS])
                               for s in sep])
        ok = float(np.mean([1.0 if s.get("sign_ok") else 0.0 for s in sep]))
        # The third panel is the one the first two cannot supply. A bar that is tall on the
        # left and full on the right still says nothing about whether each map is assigned
        # to the material it is named after, and an ill-conditioned solve satisfies both by
        # splitting one attenuation across two columns of opposite sign.
        resp = response_fraction(r)
        rows.setdefault((r["mode"], r["estimator"]), {})[r["source"]] = (idx, ok, resp)
    if not rows:
        return
    keys = sorted(rows)
    srcs = sorted({s for v in rows.values() for s in v})
    # Taller than wide: the 45-degree tick labels take roughly a third of the panel, and
    # the axes has to keep enough height for the y-label to fit rotated beside it.
    fig, axes = plt.subplots(1, 3, figsize=fp.panel_figsize(3, frac=0.49, height_in=4.6))
    w = 0.8 / max(len(srcs), 1)
    nan3 = (np.nan,) * 3
    for si, src in enumerate(srcs):
        xs = np.arange(len(keys)) + si * w
        lab = fp.pretty(src, short=True)
        axes[0].bar(xs, [rows[k].get(src, nan3)[0] for k in keys], w, label=lab)
        axes[1].bar(xs, [100 * rows[k].get(src, nan3)[1] for k in keys], w, label=lab)
        axes[2].bar(xs, [100 * rows[k].get(src, nan3)[2] for k in keys], w, label=lab)
    # Up, not right: these are y-axis labels, and an arrow pointing along the wrong axis
    # is worse than none because the reader trusts it.
    # Short enough to fit the panel height once rotated; what each quantity means is in the
    # caption, which has the room for it.
    # Two lines, not one: rotated 90 degrees a single-line label is as long as the panel is
    # tall and gets cut at both ends by the panel crop. Split across two parallel lines it
    # is half as long in the direction that matters and keeps the wording.
    for ax, lab in ((axes[0], "separation index\n$\\uparrow$ better"),
                    (axes[1], "opposite-sign pairs  (%)\n$\\uparrow$ better"),
                    (axes[2], "maps rising with own agent  (%)\n$\\uparrow$ better")):
        ax.set_xticks(np.arange(len(keys)) + 0.4 - w / 2)
        # 45 degrees, anchored. Adjacent rotated labels clear each other when the tick
        # spacing exceeds the label block height divided by sin(angle): at 30 degrees that
        # is twice the height and they collide, at 45 it is 1.41 times and they do not.
        # rotation_mode="anchor" rotates about the anchor rather than the text centre,
        # without which ha="right" no longer lands the label under its own tick.
        # One line, not two: at 8 pt a two-line block is 0.22 in tall and needs 0.31 in of
        # tick spacing at 45 degrees, against the 0.36 in eight groups leave in a panel this
        # wide -- too close, and it collided. A single line halves the block and needs only
        # 0.16 in.
        ax.set_xticklabels([f"{_basis_tick(m)} · {fp.pretty(e)}" for m, e in keys],
                           fontsize=fp.PRINT_FONT_PT - 1, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel(lab, fontsize=fp.PRINT_FONT_PT)
        ax.tick_params(axis="y", labelsize=fp.PRINT_FONT_PT - 1)
        ax.grid(alpha=0.25, axis="y")
    if np.isfinite(ref_2bin):
        # Iodine-only on the 2-bin side by necessity, so the line is not strictly the same
        # quantity as the bars (which average both agents). Labelled as a reference, not a
        # competitor -- RQ1's own figure is where the like-for-like comparison lives.
        axes[0].axhline(ref_2bin, color="k", ls=":", lw=1.6,
                        label="Vendor 2-bin (iodine only)")
    axes[1].axhline(100, color="0.4", ls="--", lw=1.0)
    axes[2].axhline(100, color="0.4", ls="--", lw=1.0)
    fig.suptitle("Source comparison", fontsize=fp.PRINT_FONT_PT + 1)
    # One legend for all panels: they carry the same series, so repeated keys were
    # the same information printed three times.
    _legend_below(fig, axes, ncol=5, bottom=0.22)
    fig.savefig(path, dpi=200)
    if cfg.split_figures:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem,
                     ["separation", "sign", "response"], dpi=200)
    plt.close(fig)


def _load_qc_images(cfg, match):
    """
    The one HU volume per source that the placement figures are drawn from.

    Separated from the drawing because the same volumes serve two figures -- a wide QC grid
    and a tight results grid -- and re-reading a full WFBP and VMI DICOM series to draw the
    same slice twice is minutes of I/O for nothing.
    """
    sitk = _sitk()
    if sitk is None:
        return {}
    imgs = {}
    for kind, folder in (("wfbp", match.get("wfbp_dir")), ("vmi", match.get("vmi_dir"))):
        if not folder:
            continue
        try:
            ch = select_family(index_dicom_series(folder, cache_dir=Path(cfg.out_root)), kind)
            if ch:
                v, im, _z = read_series_volume(ch[0])
                imgs[kind] = (v, im)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("QC: could not read %s (%s)", kind, exc)
    own = Path(match.get("own_dir", "")) / "reconstruction_thr_A_HU.nii.gz"
    if own.exists():
        oi = sitk.ReadImage(str(own))
        ov = sitk.GetArrayFromImage(oi).astype(np.float32)
        if cfg.own_frame_fix:
            # Shown the way it is DECOMPOSED, or the QC would vouch for a different volume.
            ov = np.ascontiguousarray(ov[..., ::-1])
            o = list(oi.GetOrigin())
            ox, oy = match["reference"].get("origin_x_mm"), match["reference"].get("origin_y_mm")
            if ox is not None:
                o[0], o[1] = float(ox), float(oy)
            oi2 = sitk.Image(oi.GetSize(), oi.GetPixelID())
            oi2.SetSpacing(oi.GetSpacing()); oi2.SetDirection(oi.GetDirection())
            oi2.SetOrigin(tuple(o))
            oi = oi2
        imgs["own"] = (ov, oi)
    # The 2-bin VNC, shown from what stage 1 actually imported rather than re-read from
    # DICOM -- so the QC vouches for the file the metrics were measured on.  VNC and not
    # the iodine map because an HU image is what the eye can judge insert placement in.
    vnc = Path(cfg.out_root) / "runs" / _run_key(
        SOURCE_2BIN, VENDOR_MODE, VENDOR_ESTIMATOR) / "vnc_hu.nii.gz"
    if vnc.exists():
        vi = sitk.ReadImage(str(vnc))
        imgs[SOURCE_2BIN] = (sitk.GetArrayFromImage(vi).astype(np.float32), vi)
    return imgs


# Greyscale window for every HU panel in this module.  Fixed, not per-panel: an autoscaled
# window makes a noisy source and a clean one look alike, which is the difference the
# figures exist to show.  -200..300 HU spans fat through the doped inserts.
HU_WINDOW = (-200.0, 300.0)


#: role -> the first of the two ID numbers that role occupies within its layer.
_ROLE_ID = {"base": 1, "low": 3, "high": 5}


def insert_id(ins) -> str:
    """
    A stable name for one insert, as layer letter plus number: A1 to A6, B1 to B6, C1 to C6.

    Built from the LAYER and the ROLE, never from the drawing order or the geometry id, so
    the same insert carries the same name in every figure and in the table of Section 3.1
    even if the segmentation is redrawn. Odd numbers are the first member of a pair and
    even numbers the second: 1/2 are the agent-free base lesions, 3/4 the low-concentration
    pair and 5/6 the high one. Which member of a pair holds calcium and which holds iodine
    is datasheet knowledge and is deliberately NOT encoded here, because no metric in this
    module is allowed to use it -- see the note on the segmentation contract.
    """
    layer = str(ins.get("layer", "?"))
    base = _ROLE_ID.get(str(ins.get("role", "")), 0)
    if not base:
        return f"{layer}?"
    return f"{layer}{base + int(ins.get('member', 0) or 0)}"


def _roi_placement_figure(path, cfg, match, imgs, pad_mm=70.0, wide=True,
                          split=False, dpi=200):
    """
    One row per source, one column per insert layer, ROIs drawn as outlines.

    Written TWICE from the same volumes, at two crops, because the two jobs conflict:

      * `wide` (pad 70 mm, under qc/) is the check.  Cropping tight to the 100 mm module
        proves the ROIs sit on inserts but hides whether the module itself is in the right
        place inside the body -- the failure that actually happened.  The surrounding
        phantom, spine and couch have to stay in frame.
      * the tight version (under figures/) is the thesis figure.  There the body is context
        the reader does not need, and at 70 mm pad the inserts are a few pixels across on
        the printed page.

    Every panel is found by PHYSICAL z: indexing each source with the reference volume's
    slice number put the own panels ~200 slices off the phantom, because our reconstruction
    has its own z origin and a slightly different slice spacing.

    The 2-bin row matters more than the others despite the phantom not having been moved:
    it is a SEPARATE ACQUISITION, so it is the one source whose agreement with a
    segmentation drawn on the 4-bin WFBP is an assumption rather than a construction.
    """
    plt = _mpl()
    if plt is None or not imgs:
        return
    inserts = match["inserts"]
    module = match.get("module")
    layers = sorted({i["layer_index"] for i in inserts})
    order = [k for k in ("own", "wfbp", "vmi", SOURCE_2BIN) if k in imgs]
    vmin, vmax = HU_WINDOW

    # Each panel is placed at .32\textwidth in the thesis, so it is authored at that width:
    # at 3.1 in it was being scaled to 0.64 and its 9 pt titles reached the page at 5.8 pt.
    pw = fp.panel_figsize(1, frac=0.32)[0]
    fig, axes = plt.subplots(len(order), len(layers), squeeze=False,
                             figsize=(pw * len(layers), (pw + 0.25) * len(order)))
    for ri, kind in enumerate(order):
        vol, img = imgs[kind]
        sx = img.GetSpacing()[0]
        for ci, li in enumerate(layers):
            ax = axes[ri][ci]
            sel = [i for i in inserts if i["layer_index"] == li]
            z_mid = float(np.mean([0.5 * (i["z_lo_mm"] + i["z_hi_mm"]) for i in sel]))
            sl = vol[_slice_at_z(img, z_mid)]
            if module:
                cx, cy = fp.index_of_point(img, module["cx_mm"], module["cy_mm"], z_mid)
                half = (module["radius_mm"] + pad_mm) / sx
                y0, y1, x0, x1 = fp.crop_box_px(sl.shape, cx, cy, half)
            else:
                y0, y1, x0, x1 = 0, sl.shape[0], 0, sl.shape[1]
            fp.show_slice(ax, sl[y0:y1, x0:x1], vmin=vmin, vmax=vmax, pixel_mm=sx,
                          scale_bar_mm=20.0, title_size=fp.PRINT_FONT_PT,
                          title=f"{fp.pretty(kind, short=True)}   "
                                f"layer {sel[0].get('layer', '?')}"
                                + (f"   z = {z_mid:.1f} mm" if wide else ""))
            for i in sel:
                p = fp.index_of_point(img, i["x_mm"], i["y_mm"], z_mid)
                r_px = cfg.insert_radius_mm / sx
                fp.outline_circle(ax, (p[0] - x0, p[1] - y0), r_px,
                                  colour="tab:red", lw=1.1)
                # ABOVE the circle, never inside it. The reader is being asked to judge
                # the pixels the ROI encloses, so a label sitting on them defeats the
                # figure; the row above is empty by construction, since the ROI is drawn
                # at the datasheet radius inside a larger insert.
                ax.annotate(insert_id(i), (p[0] - x0, p[1] - y0 - r_px),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", va="bottom", color="tab:red",
                            fontsize=fp.PRINT_FONT_PT - 3)
            if module:
                fp.outline_circle(ax, (cx - x0, cy - y0), module["radius_mm"] / sx,
                                  colour="tab:orange", lw=1.0)
    # A real legend, not a sentence in the title: the colour key is what the reader needs
    # while looking at the panels, and a title that carries it cannot be edited later.
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color="tab:red", lw=1.4,
                   label=f"insert ROI, datasheet radius {cfg.insert_radius_mm:.1f} mm"),
            Line2D([], [], color="tab:orange", lw=1.4, label="module boundary")]
    fig.legend(handles=keys, loc="lower center", ncol=2, frameon=False, fontsize=9)
    # rect leaves room for the legend AND the suptitle -- tight_layout reserves space for
    # neither, which is what let the title land on top of the first row's panel titles.
    fig.suptitle("ROI placement" + (f"   ({fp.window_label(vmin, vmax)})" if wide else ""),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(path, dpi=dpi)
    if split:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem,
                     [f"{k}_{l}" for k in order for l in layers], dpi=dpi)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
# Image figures -- what the numbers were measured on
# ═════════════════════════════════════════════════════════════════════
# The validation had no picture of a material map anywhere: every figure was a scalar
# summary of maps the reader never saw.  These are drawn in stage 2 from the volumes each
# run already wrote (runs/<key>/decomp_<mode>_<material>.nii.gz), so they cost no solve.
#
# Colour, not grey, for a material map: the quantity IS the value, and a density map is
# conventionally shown on a sequential colour scale with the scale stated.  ROI geometry is
# drawn as an outline in a colour the map does not contain, never as a fill -- a tint over
# a material map changes the very brightness the reader is being asked to judge.
MAP_CMAP = "inferno"
INSERT_OUTLINE = "cyan"
MODULE_OUTLINE = "w"
_ROLE_COL = {"base": "0.55", "low": "tab:orange", "high": "tab:red"}


def _run_dir(cfg, key):
    return Path(cfg.out_root) / "runs" / key


def _slab_at_z(vol, img, z_mm, slab_mm):
    """
    Mean over a slab centred on a physical z, as one 2-D panel.

    A single 0.4 mm slice of a material map is mostly noise -- the maps are solved
    voxelwise, so nothing smooths them along z.  The inserts are 10 mm tall, so averaging a
    few mm shows what the ROI statistics were actually computed over rather than one noisy
    plane of it.
    """
    k0 = _slice_at_z(img, z_mm - 0.5 * float(slab_mm))
    k1 = _slice_at_z(img, z_mm + 0.5 * float(slab_mm))
    lo, hi = (k0, k1) if k0 <= k1 else (k1, k0)
    sl = vol[lo:hi + 1]
    return np.asarray(sl[0] if sl.shape[0] == 1 else np.nanmean(sl, axis=0), float)


def _map_panel(cfg, key, mode, mat, z_mm):
    """
    (2-D slab, image, pixel_mm) of one material map, with the volume freed immediately.

    One map at a time and dropped as soon as the slab is taken: a full-FOV vendor volume is
    gigabytes, and stage 2 already walks up to 24 runs.
    """
    sitk = _sitk()
    p = _run_dir(cfg, key) / f"decomp_{mode}_{mat}.nii.gz"
    if sitk is None or not p.exists():
        return None
    img = sitk.ReadImage(str(p))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = _slab_at_z(vol, img, z_mm, cfg.image_slab_mm)
    del vol
    gc.collect()
    return sl, img, float(img.GetSpacing()[0])


def _panel_geom(img, module, z_mm, half_mm, shape_yx):
    """(crop box, module centre in crop coords, pixel_mm) for one image panel."""
    sx = float(img.GetSpacing()[0])
    if not module:
        return (0, shape_yx[0], 0, shape_yx[1]), None, sx
    cx, cy = fp.index_of_point(img, module["cx_mm"], module["cy_mm"], z_mm)
    box = fp.crop_box_px(shape_yx, cx, cy, half_mm / sx)
    return box, (cx - box[2], cy - box[0]), sx


def _draw_rois(ax, img, inserts, z_mm, box, radius_mm, colour=INSERT_OUTLINE, lw=0.9):
    for i in inserts:
        p = fp.index_of_point(img, i["x_mm"], i["y_mm"], z_mm)
        fp.outline_circle(ax, (p[0] - box[2], p[1] - box[0]),
                          radius_mm / float(img.GetSpacing()[0]), colour=colour, lw=lw)


def _layer_z(inserts, layer_index):
    sel = [i for i in inserts if i["layer_index"] == layer_index]
    return float(np.mean([0.5 * (i["z_lo_mm"] + i["z_hi_mm"]) for i in sel])) if sel else None


def _reference_layer(run, inserts, agent="Iodine"):
    """
    The layer whose `high` lesion carries the most of `agent` -- the clearest one to show.

    Chosen from the data rather than fixed, but chosen ONCE and reused for every panel of a
    figure: a per-panel choice would put different sources on different slices and the
    comparison would silently stop being one.
    """
    dens = (run.get("density") or {}).get(agent)
    layers = sorted({i["layer_index"] for i in inserts})
    if not dens:
        return layers[len(layers) // 2] if layers else None
    best, best_v = None, -np.inf
    for li in layers:
        sel = [i for i in inserts if i["layer_index"] == li and i.get("role") == "high"]
        v = max((dens[i["geom_id"]] for i in sel
                 if np.isfinite(dens[i["geom_id"]])), default=-np.inf)
        if v > best_v:
            best, best_v = li, v
    return best if best is not None else (layers[0] if layers else None)


def _high_reference(run, agent="Iodine"):
    """The agent's own reading on the high lesions -- the scale each map is normalised by."""
    e = (run.get("cross_layer") or {}).get(f"{agent}/high") or {}
    v = e.get("mean")
    return float(v) if v is not None and np.isfinite(v) and abs(v) > 1e-12 else None


def _best_run_per_source(runs):
    """One representative run per source: the highest-scoring, or the only one."""
    out = {}
    for key, r in runs.items():
        src = r.get("source")
        if src is None:
            continue
        prev = out.get(src)
        if prev is None or run_score(r) > run_score(prev[1]):
            out[src] = (key, r)
    return out


def _material_gallery_figure(path, cfg, results, split=False):
    """
    (A) The material maps themselves: one row per source, one column per material.

    Restricted to the sources WE solved.  The vendor's 2-bin map is in mg/ml where ours is
    g/cm3, so putting it in a shared-window column would be a units error dressed up as a
    comparison -- it gets its own normalised figure (rq1_iodine_maps) instead.

    One window per material column, computed over every row at once, so a noisy source
    cannot be rescaled into looking like a clean one.
    """
    plt = _mpl()
    runs = results.get("runs", {})
    if plt is None or not runs:
        return
    inserts = results["match"]["inserts"]
    module = results["match"].get("module")
    per_src = {s: v for s, v in _best_run_per_source(runs).items() if s != SOURCE_2BIN}
    order = [s for s in ("own", "wfbp", "vmi") if s in per_src]
    if not order:
        return
    ref_key, ref_run = per_src[order[0]]
    li = _reference_layer(ref_run, inserts)
    z_mm = _layer_z(inserts, li)
    if z_mm is None:
        return
    layer = next((i.get("layer") for i in inserts if i["layer_index"] == li), "?")

    panels = {}                                        # (source, material) -> (arr, img, sx)
    for src in order:
        key, run = per_src[src]
        for mat in AGENTS:
            got = _map_panel(cfg, key, run["mode"], mat, z_mm)
            if got:
                panels[(src, mat)] = got
    if not panels:
        logger.info("material gallery skipped: no decomp_*.nii.gz found under runs/")
        return

    # constrained layout with the colourbars BELOW their column.  An imshow axes holds a
    # square aspect, so a colourbar down the right takes its width out of the panel height
    # too and leaves a band of white between every row; underneath it competes with
    # nothing, and the panels tile.
    fig, axes = plt.subplots(len(order), len(AGENTS), squeeze=False,
                             figsize=(fp.panel_figsize(1, frac=0.32)[0] * len(AGENTS),
                                      fp.panel_figsize(1, frac=0.32)[0] * len(order) + 0.55),
                             layout="constrained")
    for ci, mat in enumerate(AGENTS):
        col = [panels[(s, mat)][0] for s in order if (s, mat) in panels]
        vmin, vmax = fp.shared_window(col, 2.0, 99.5) if col else (0.0, 1.0)
        im = None
        for ri, src in enumerate(order):
            ax = axes[ri][ci]
            got = panels.get((src, mat))
            if not got:
                fp.greyed_panel(ax, f"{mat} map\nnot written")
                continue
            arr, img, sx = got
            box, _c, _sx = _panel_geom(img, module, z_mm, cfg.image_half_mm, arr.shape)
            im = fp.show_slice(ax, arr[box[0]:box[1], box[2]:box[3]], vmin=vmin, vmax=vmax,
                               cmap=MAP_CMAP, pixel_mm=sx, scale_bar_mm=20.0,
                               title=f"{fp.pretty(src, short=True)}\n"
                                     f"{MATERIAL_LABELS.get(mat, mat)}",
                               title_size=fp.PRINT_FONT_PT)
            _draw_rois(ax, img, [i for i in inserts if i["layer_index"] == li], z_mm, box,
                       cfg.insert_radius_mm)
            if module:
                cx, cy = fp.index_of_point(img, module["cx_mm"], module["cy_mm"], z_mm)
                fp.outline_circle(ax, (cx - box[2], cy - box[0]),
                                  module["radius_mm"] / sx, colour=MODULE_OUTLINE, lw=0.8)
        if im is not None:
            cb = fig.colorbar(im, ax=[axes[r][ci] for r in range(len(order))],
                              location="bottom", fraction=0.035, pad=0.015)
            # The column heading already names the material and a colourbar label cannot
            # wrap, so the bar carries only the quantity and its unit.
            cb.set_label("density  (g/cm$^3$)", fontsize=fp.PRINT_FONT_PT - 1)
            cb.ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
    # Which basis and which estimator produced these maps belongs in the caption, where it
    # can be written out; a configuration key in the title is a name the reader cannot
    # expand and it competes with the one thing the title has to say.
    fig.suptitle(f"Material maps   layer {layer}", fontsize=fp.PRINT_FONT_PT + 1)
    fig.savefig(path, dpi=200)
    if split:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem,
                     [f"{s}_{m}" for s in order for m in AGENTS], dpi=200)
    plt.close(fig)


def _rq1_maps_figure(path, cfg, results, split=False):
    """
    (B) RQ1 made verifiable by eye: our iodine map beside the vendor's 2-bin iodine map.

    Both are normalised by their OWN high-lesion reading, which is the same device that
    makes the RQ1 metrics comparable -- every one of them is a ratio within a single map,
    because the vendor works in mg/ml and we work in g/cm3 and the datasheet states no
    concentrations at all.  After that normalisation a value of 1 means "as bright as a
    high lesion of this map" on both sides, and the leak that leak_ratio reports is
    something the reader can see sitting on a calcium insert rather than a number to trust.

    The two members of each HU-matched pair are outlined separately, because the claim
    being checked is that one of them should be dark.
    """
    plt = _mpl()
    runs = results.get("runs", {})
    if plt is None or not runs:
        return
    inserts = results["match"]["inserts"]
    module = results["match"].get("module")
    per_src = _best_run_per_source(runs)
    vendor = per_src.get(SOURCE_2BIN)
    ours = next((per_src[s] for s in ("wfbp", "own", "vmi") if s in per_src), None)
    if not vendor or not ours:
        logger.info("rq1 map figure skipped: need both a 4-bin run and the 2-bin import")
        return
    li = _reference_layer(ours[1], inserts)
    z_mm = _layer_z(inserts, li)
    if z_mm is None:
        return
    sel = [i for i in inserts if i["layer_index"] == li]
    layer = next((i.get("layer") for i in sel), "?")

    cols = []
    for label, (key, run) in (("4-bin  (ours)", ours), ("2-bin  (Siemens)", vendor)):
        got = _map_panel(cfg, key, run["mode"], "Iodine", z_mm)
        ref = _high_reference(run, "Iodine")
        if got and ref:
            cols.append((f"{label}   {run['source']}", got[0] / ref, got[1], got[2]))
    if len(cols) < 2:
        logger.info("rq1 map figure skipped: a high-lesion reference is missing")
        return

    # vmax at least 1: the colour scale's whole meaning is that 1 = a high lesion of this
    # map, and a bar topping out at 0.35 hides the value the label is defined by.  The
    # inserts are a small part of the crop, so a plain percentile never reaches it.
    vmin, vmax = fp.shared_window([c[1] for c in cols], 2.0, 99.5)
    vmax = max(vmax, 1.05)
    fig, axes = plt.subplots(1, len(cols), squeeze=False, layout="constrained",
                             figsize=(3.5 * len(cols), 4.55))
    dens = (ours[1].get("density") or {}).get("Iodine") or []
    im = None
    for ax, (title, arr, img, sx) in zip(axes[0], cols):
        box, _c, _s = _panel_geom(img, module, z_mm, cfg.image_half_mm, arr.shape)
        im = fp.show_slice(ax, arr[box[0]:box[1], box[2]:box[3]], vmin=vmin, vmax=vmax,
                           cmap=MAP_CMAP, pixel_mm=sx, scale_bar_mm=20.0, title=title)
        # The two members of a DOPED pair get different outlines: the whole question is
        # whether the map can tell them apart, and an identical marker on both would ask
        # the reader to guess which one should be dark.  The base pair carries neither
        # agent, so naming a carrier there would invent a distinction that does not exist.
        for i in sel:
            mem = [m for m in sel if m["pair_label"] == i["pair_label"]]
            doped = i.get("role") in ("low", "high") and len(mem) == 2 \
                and len(dens) > max(m["geom_id"] for m in mem)
            carrier = doped and i["geom_id"] == carrier_insert(mem, dens)["geom_id"]
            fp.outline_circle(ax, np.subtract(fp.index_of_point(img, i["x_mm"], i["y_mm"],
                                                                z_mm), (box[2], box[0])),
                              cfg.insert_radius_mm / sx,
                              colour=INSERT_OUTLINE if carrier else "w",
                              lw=1.1,
                              ls="-" if carrier else ((0, (2, 2)) if doped else (0, (1, 3))))
    from matplotlib.lines import Line2D
    # "outside lower center", not "lower center": a constrained layout does not reserve
    # room for a plain figure legend, and the keys landed on top of the colourbar label.
    # Short keys: the full sentences ran off the right edge at three columns. What each
    # outline MEANS -- that the dashed one should read dark -- belongs in the caption.
    fig.legend(handles=[Line2D([], [], color=INSERT_OUTLINE, lw=1.4,
                               label="iodine carrier"),
                        Line2D([], [], color="0.5", lw=1.4, ls=(0, (2, 2)),
                               label="its HU-matched partner"),
                        Line2D([], [], color="0.5", lw=1.4, ls=(0, (1, 3)),
                               label="base pair (no agent)")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes[0]), location="bottom", fraction=0.05,
                          pad=0.015)
        cb.set_label("iodine signal  (high lesion = 1)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Iodine map, 4 bins versus 2 bins   layer {layer}", fontsize=11)
    fig.savefig(path, dpi=200)
    if split:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem,
                     ["4bin", "2bin"], dpi=200)
    plt.close(fig)


def _capability_figure(path, cfg, results, split=False):
    """
    (G) The other half of RQ1: 4 bins yield a calcium map, 2 bins cannot yield one at all.

    A capability statement, deliberately kept out of the performance numbers -- N energy
    channels determine at most N materials, so this is a property of the acquisition and
    not a result the vendor's solver could improve on.  The empty panel is labelled rather
    than omitted: a missing panel reads as an oversight, a labelled one states the
    restriction, which is the entire point.
    """
    plt = _mpl()
    runs = results.get("runs", {})
    if plt is None or not runs:
        return
    inserts = results["match"]["inserts"]
    module = results["match"].get("module")
    per_src = _best_run_per_source(runs)
    ours = next((per_src[s] for s in ("wfbp", "own", "vmi") if s in per_src), None)
    if not ours:
        return
    key, run = ours
    li = _reference_layer(run, inserts)
    z_mm = _layer_z(inserts, li)
    if z_mm is None:
        return
    got = _map_panel(cfg, key, run["mode"], "HA", z_mm)
    if not got:
        return
    arr, img, sx = got
    box, _c, _s = _panel_geom(img, module, z_mm, cfg.image_half_mm, arr.shape)
    crop = arr[box[0]:box[1], box[2]:box[3]]
    vmin, vmax = fp.shared_window([crop], 2.0, 99.5)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.35), layout="constrained")
    im = fp.show_slice(axes[0], crop, vmin=vmin, vmax=vmax, cmap=MAP_CMAP, pixel_mm=sx,
                       scale_bar_mm=20.0, title=f"4 bins   CaHA map   ({run['source']})")
    _draw_rois(axes[0], img, [i for i in inserts if i["layer_index"] == li], z_mm, box,
               cfg.insert_radius_mm)
    fp.greyed_panel(axes[1],
                    "2 bins\n\nno CaHA map exists\n\ntwo energy channels determine\n"
                    "at most two materials")
    axes[1].set_title("2 bins   CaHA map", fontsize=9)
    cb = fig.colorbar(im, ax=axes[0], location="bottom", fraction=0.05, pad=0.015)
    cb.set_label("CaHA density  (g/cm$^3$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Calcium map: a capability of the 4-bin acquisition", fontsize=11)
    fig.savefig(path, dpi=200)
    if split:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem, ["4bin", "2bin"], dpi=200)
    plt.close(fig)


def _contact_sheet_figure(path, cfg, results, run, mat, split=False):
    """
    (C) The three concentration levels as IMAGES -- the linearity plot's three dots, shown.

    The proportionality plot asks the reader to accept that three anonymous dots are a
    base, a low and a high lesion.  Here each dot is the insert it was measured on, cropped
    from the material map at one shared window, with its measured density underneath: the
    thumbnails should brighten 0 : 1 : 3.1 across a row.

    The insert shown for each level is the pair member with the LARGER reading of this
    agent, which is exactly the member linearity_metrics takes.  So the sheet also audits
    that choice: if the "high" thumbnail does not look like the brightest lesion in its
    row, the selection picked the wrong member and the fitted line is describing the wrong
    inserts.
    """
    plt = _mpl()
    dens = (run.get("density") or {}).get(mat)
    if plt is None or not dens:
        return
    inserts = results["match"]["inserts"]
    li_all = sorted({i["layer_index"] for i in inserts})
    got_any = False
    picks = {}                              # (layer_index, role) -> insert
    for li in li_all:
        for role in ROLES:
            mem = [i for i in inserts if i["layer_index"] == li and i.get("role") == role]
            mem = [i for i in mem if np.isfinite(dens[i["geom_id"]])]
            if mem:
                # carrier_insert, the same rule linearity_metrics uses -- this sheet exists
                # to audit that selection, so it has to show the insert the fit took.
                picks[(li, role)] = carrier_insert(mem, dens)
    if not picks:
        return

    # One crop per layer, so all three levels of a row come from the same slab.
    crops, sx_of = {}, {}
    half_mm = 3.0 * cfg.insert_radius_mm
    for li in li_all:
        z_mm = _layer_z(inserts, li)
        got = _map_panel(cfg, _run_key(run["source"], run["mode"], run["estimator"]),
                         run["mode"], mat, z_mm)
        if not got:
            continue
        arr, img, sx = got
        for role in ROLES:
            ins = picks.get((li, role))
            if ins is None:
                continue
            cx, cy = fp.index_of_point(img, ins["x_mm"], ins["y_mm"], z_mm)
            y0, y1, x0, x1 = fp.crop_box_px(arr.shape, cx, cy, half_mm / sx)
            crops[(li, role)] = arr[y0:y1, x0:x1]
            sx_of[(li, role)] = sx
            got_any = True
    if not got_any:
        return

    vmin, vmax = fp.shared_window(list(crops.values()), 1.0, 99.5)
    fig, axes = plt.subplots(len(li_all), len(ROLES), squeeze=False,
                             layout="constrained",
                             figsize=(2.3 * len(ROLES), 2.55 * len(li_all) + 0.7))
    im = None
    for ri, li in enumerate(li_all):
        layer = next((i.get("layer") for i in inserts if i["layer_index"] == li), "?")
        low = dens[picks[(li, "low")]["geom_id"]] if (li, "low") in picks else None
        for ci, role in enumerate(ROLES):
            ax = axes[ri][ci]
            if (li, role) not in crops:
                fp.greyed_panel(ax, "not measured")
                continue
            v = dens[picks[(li, role)]["geom_id"]]
            rel = f"   ({v / low:.2f}x low)" if low and abs(low) > 1e-12 and role != "low" \
                else ""
            im = fp.show_slice(ax, crops[(li, role)], vmin=vmin, vmax=vmax, cmap=MAP_CMAP,
                               pixel_mm=sx_of[(li, role)], scale_bar_mm=5.0,
                               title=f"layer {layer}   {role}")
            ax.set_xlabel(f"{v:+.4f} g/cm$^3${rel}", fontsize=8)
            ax.xaxis.set_label_position("bottom")
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), location="bottom",
                          fraction=0.03, pad=0.015)
        cb.set_label(f"{mat} density  (g/cm$^3$)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"{mat} at the three concentration levels   "
                 f"(nominal 0 : 1 : 3.1, inferred)", fontsize=11)
    fig.savefig(path, dpi=200)
    if split:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem,
                     [f"{next((i.get('layer') for i in inserts if i['layer_index'] == li), li)}"
                      f"_{r}" for li in li_all for r in ROLES], dpi=200)
    plt.close(fig)


def _material_space_figure(path, cfg, results, split=False):
    """
    (D) Every insert voxel in the (iodine, CaHA) plane -- the sign test at voxel level.

    The separation figure plots ROI MEANS, one point per insert; this plots the voxels
    behind them.  A working decomposition puts the calcium-carrying and iodine-carrying
    members of a pair in different directions from the base cloud.  A solver tracking HU
    puts them on one line through the origin, because it is reading a single contrast and
    splitting it between two columns in a fixed proportion -- and that line is visible here
    while two overlapping ROI means are not.

    Voxels are subsampled per insert so the file stays a figure rather than a point dump;
    the clouds are shapes, and their shape survives subsampling.
    """
    plt = _mpl()
    sitk = _sitk()
    runs = results.get("runs", {})
    if plt is None or sitk is None or not runs:
        return
    roi_path = Path(cfg.out_root) / "roi_labels.nrrd"
    if not roi_path.exists():
        return
    roi_img = sitk.ReadImage(str(roi_path))
    inserts = results["match"]["inserts"]
    per_src = {s: v for s, v in _best_run_per_source(runs).items() if s != SOURCE_2BIN}
    order = [s for s in ("own", "wfbp", "vmi") if s in per_src]
    if not order:
        return
    rng = np.random.default_rng(0)

    clouds = {}                                   # source -> {geom_id: (iod[], ha[])}
    for src in order:
        key, run = per_src[src]
        if not all(m in (run.get("density") or {}) for m in AGENTS):
            continue
        vals, lab_arr = {}, None
        for mat in AGENTS:
            p = _run_dir(cfg, key) / f"decomp_{run['mode']}_{mat}.nii.gz"
            if not p.exists():
                vals = {}
                break
            img = sitk.ReadImage(str(p))
            vol = sitk.GetArrayFromImage(img).astype(np.float32)
            if lab_arr is None:
                lab_arr = sitk.GetArrayFromImage(sitk.Resample(
                    roi_img, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
                    sitk.sitkUInt16))
            per_insert = {}
            for i in inserts:
                v = vol[lab_arr == i["geom_id"] + 1]
                v = v[np.isfinite(v)]
                if v.size > 400:
                    v = rng.choice(v, 400, replace=False)
                per_insert[i["geom_id"]] = v
            vals[mat] = per_insert
            del vol, img
            gc.collect()
        if vals:
            clouds[src] = vals

    if not clouds:
        logger.info("material-space figure skipped: no run carries both agents on disk")
        return
    shown = [s for s in order if s in clouds]
    fig, axes = plt.subplots(1, len(shown), squeeze=False,
                             figsize=(4.3 * len(shown), 4.5), sharex=True, sharey=True)
    for ax, src in zip(axes[0], shown):
        iod, ha = clouds[src]["Iodine"], clouds[src]["HA"]
        for i in inserts:
            g = i["geom_id"]
            x, y = iod.get(g), ha.get(g)
            if x is None or y is None or not x.size or x.size != y.size:
                continue
            ax.plot(x, y, ".", ms=1.6, alpha=0.30,
                    color=_ROLE_COL.get(i.get("role"), "0.5"), rasterized=True, zorder=1)
        # One connector per doped pair, joining the two clouds' centres. Along a common
        # direction from the origin = HU tracking; across it = material information.
        for _lab, mem in _pairs_of(inserts):
            if len(mem) != 2 or mem[0].get("role") == "base":
                continue
            pts = [(np.mean(iod[m["geom_id"]]), np.mean(ha[m["geom_id"]]))
                   for m in mem if iod.get(m["geom_id"], np.array([])).size]
            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], "-",
                        color="k", lw=0.9, alpha=0.8, zorder=3)
        ax.axhline(0, color="0.75", lw=0.7)
        ax.axvline(0, color="0.75", lw=0.7)
        ax.set_title(src, fontsize=11)
        ax.set_xlabel("iodine density  (g/cm$^3$)")
        ax.grid(alpha=0.2)
    axes[0][0].set_ylabel("CaHA density  (g/cm$^3$)")
    from matplotlib.lines import Line2D
    for role, col in _ROLE_COL.items():
        axes[0][0].add_line(Line2D([], [], marker="o", ls="", color=col,
                                   label=f"{role} lesion"))
    axes[0][0].add_line(Line2D([], [], color="k", lw=1.0, label="HU-matched pair"))
    _share_axes(axes[0], logy=False, free=cfg.free_axes)
    _legend_below(fig, axes[0][0], ncol=4)
    fig.suptitle("Insert voxels in the material plane", fontsize=11)
    fig.savefig(path, dpi=200)
    if split:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem, shown,
                     ylabel="CaHA density  (g/cm$^3$)", dpi=200)
    plt.close(fig)


class RunScore(NamedTuple):
    """Ordered so a plain tuple comparison ranks runs; named so callers can read a field."""
    upright: float
    response: float
    sign: float
    index: float


def run_score(run, agents=AGENTS):
    """
    Rank runs so 'the best one' means something defensible.

    Four keys, most disqualifying first, because each earlier one can make the next
    meaningless rather than merely worse:

    1. `upright` -- no agent map reads negative on the lesions carrying the most of it.  An
       inverted map is not a concentration map, and every metric below is a magnitude, so an
       inverted run produces small, healthy-looking numbers from a failed solve.
    2. `response` -- each agent map rises with its OWN concentration.  This is the key that
       was missing.  The sign test below only demands that the two maps move in opposite
       directions across a matched pair, which an ill-conditioned solve satisfies for free by
       splitting one attenuation across two columns of opposite sign.  A run can therefore
       score 100 % on the sign test while its calcium map falls as calcium rises, and the
       old two-key score put exactly such a run first.
    3. `sign` -- the two agents move in opposite directions across an HU-matched pair.
    4. `index` -- and only then does the magnitude of the separation decide.

    Keys 1 and 2 are graded, not gates: a set in which every run fails them still produces a
    ranking, and the summary names the failures rather than leaving the winner to look clean.
    """
    sep = doped_pairs(run.get("separation", []))
    if not sep:
        return RunScore(-1.0, -1.0, -1.0, -1.0)
    upright = 0.0 if inverted_agents(run, agents) else 1.0
    resp = response_fraction(run, agents)
    sign = float(np.mean([1.0 if s.get("sign_ok") else 0.0 for s in sep]))
    idx = _mean_of_finite([_mean_of_finite([s.get(f"sep_index_{m}") for m in agents])
                           for s in sep])
    return RunScore(upright, resp if np.isfinite(resp) else -1.0, sign,
                    idx if np.isfinite(idx) else -1.0)


def is_rankable(run):
    """
    Can this run be placed in the estimator x basis ranking at all?

    The sign test needs both agents, and a rank-limited run (2 energy channels -> at most 2
    materials, hence no HA map) cannot supply it.  Scoring one anyway hands it sign = 0 and
    parks it permanently last, which reads as a damning result but is an artefact of the
    ranking rather than a measurement.  run_score exists to choose among OUR estimator and
    basis choices; the vendor import is a reference point, not a candidate.
    """
    return all(m in (run.get("density") or {}) for m in AGENTS)


def _mode_comparison_figure(path, results, cfg):
    """
    One figure for the basis question: does naming the base column differently move the
    result?  Same data, same estimator, two bases -- so the gap IS the base-material
    sensitivity, which is the reason both are run rather than one being argued for.
    """
    plt = _mpl()
    if plt is None:
        return
    by = {}
    for r in results["runs"].values():
        # The vendor import has no basis -- it was never solved on one. Its key is dropped
        # by the >=2 filter below, but its mode would still enter `all_modes` and draw an
        # all-NaN third series with a legend entry claiming a basis that does not exist.
        if r.get("source") == SOURCE_2BIN:
            continue
        by.setdefault((r["source"], r["estimator"]), {})[r["mode"]] = r
    keys = sorted(k for k, v in by.items() if len(v) >= 2)
    if not keys:
        return
    all_modes = sorted({m for v in by.values() for m in v})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    w = 0.8 / max(len(all_modes), 1)
    for mi, mode in enumerate(all_modes):
        xs = np.arange(len(keys)) + mi * w
        sep, spec = [], []
        for k in keys:
            r = by[k].get(mode)
            sep.append(run_score(r).index if r else np.nan)
            sp = (r or {}).get("specificity", {})
            spec.append(_mean_of_finite([abs(sp[m]["mean"]) for m in sp]))
        axes[0].bar(xs, sep, w, label=_short_mode(mode))
        axes[1].bar(xs, spec, w, label=_short_mode(mode))
    for ax, lab in ((axes[0], "mean separation index   $\\uparrow$ better"),
                    (axes[1], "|bias| in the agent-free base lesions   "
                              "$\\downarrow$ better")):
        ax.set_xticks(np.arange(len(keys)) + 0.4 - w / 2)
        ax.set_xticklabels([f"{s}\n{e}" for s, e in keys], fontsize=8,
                           rotation=20, ha="right")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Basis sensitivity", fontsize=12)
    _legend_below(fig, axes, ncol=4, bottom=0.22)
    fig.savefig(path, dpi=200)
    if cfg.split_figures:
        _save_panels(fig, axes, Path(path).parent, Path(path).stem,
                     ["separation", "bias"], dpi=200)
    plt.close(fig)


# ── RQ1: 4 bins vs 2 bins ────────────────────────────────────────────────────
# Every entry is DIMENSIONLESS and LOWER IS BETTER, which is what allows a vendor map in
# mg/ml and ours in g/cm3 on one axis, and what makes the four panels read the same way.
RQ1_METRICS = (
    ("leak_ratio", "Iodine leak onto Ca",
     "leak ratio   0 = separated, 1 = HU tracking"),
    # Both sides of this ratio are magnitudes -- see specificity_metrics. A base lesion
    # holds no iodine, so only distance from zero can mean anything there.
    ("base_leak_frac", "Base-lesion leak",
     "|iodine read| / |high-lesion signal|"),
    ("cross_layer_cv", "Base-material crosstalk",
     "CV of one iodine level over 3 layers"),
    # Plain hyphen, not mathtext: this string is also printed to the console by
    # _print_rq1, where "$-$" would show as literal LaTeX.
    ("ratio_error", "High/low ratio error",
     "|measured / inferred 3.1  -  1|"),
)
# Axis labels are SHORT. The third field above is the console column explanation, and set
# on a rotated axis it is longer than the panel is tall, so the panel crop cut it at both
# ends -- "leak ratio 0 = separated, 1 = HU tracking" reached the page as "ak ratio". What
# each measure means is defined in the methods and repeated in the caption, both of which
# have room for a sentence; the axis needs only the name.
RQ1_AXIS_LABELS = {
    "leak_ratio":     "leak ratio",
    "base_leak_frac": "base-lesion leak fraction",
    "cross_layer_cv": "cross-layer CV",
    "ratio_error":    "high/low ratio error",
}


def rq1_summary(run):
    """
    The four iodine-only, unit-free numbers RQ1 is decided on.

    All four are ratios within a single map, so none of them can be moved by the vendor's
    unit being mg/ml where ours is g/cm3 -- the one property that makes the comparison
    possible at all, given the datasheet states no concentrations.
    """
    blk = run.get("rq1_iodine") or {}
    out = {k: float("nan") for k, *_rest in RQ1_METRICS}

    doped = doped_pairs(blk.get("separation", []))
    out["leak_ratio"] = _mean_of_finite([s.get("leak_ratio_Iodine") for s in doped])

    spec = (blk.get("specificity") or {}).get("Iodine") or {}
    out["base_leak_frac"] = float(spec.get("leak_frac", np.nan))
    # Carried alongside the number, not folded into it: a leak fraction measured against a
    # sign-inverted map is small for the wrong reason, and the reader has to be told which
    # runs those are rather than seeing them sit low on the axis.
    out["reference_negative"] = bool(spec.get("reference_negative", False))

    out["cross_layer_cv"] = _mean_of_finite(
        [v.get("cv") for k, v in (blk.get("cross_layer") or {}).items()
         if k.startswith("Iodine/")])

    errs = []
    for key, v in (blk.get("linearity") or {}).items():
        if not key.startswith("Iodine/"):
            continue
        meas, inf = v.get("measured_high_over_low"), v.get("inferred_ratio")
        if meas and inf and np.isfinite(meas) and abs(inf) > 1e-9:
            errs.append(abs(meas / inf - 1.0))
    out["ratio_error"] = _mean_of_finite(errs)
    # Same reasoning as `reference_negative`, one step further: an iodine map that falls as
    # iodine rises is not a concentration map either, and every number above is a magnitude,
    # so it lands on the axis looking respectable. Carried, never folded in.
    resp = (blk.get("response") or {}).get("Iodine") or {}
    out["response_ok"] = bool(resp.get("response_ok", True))
    out["usable"] = bool(not out["reference_negative"] and out["response_ok"])
    return out


def _bins_figure(path, results, cfg):
    """
    The RQ1 figure: the 4-bin ENVELOPE against the 2-bin point.

    Not a two-bar chart, deliberately.  The vendor's 2-bin product differs from our 4-bin
    result in four ways at once -- channel count, solver, reconstruction and proprietary
    denoising -- so a single paired comparison could not attribute a difference to the bin
    count.  It does not have to: the 4-bin side is a whole distribution over modalities,
    estimators and bases, and if the 2-bin point falls outside all of it, the only property
    every one of those runs shares is the channel count.

    Both sides are scored on IODINE ALONE.  Two bins can estimate at most two materials, so
    no calcium map exists on that side; averaging our Iodine and HA would score us on
    information the other mode is mathematically barred from having.
    """
    plt = _mpl()
    if plt is None:
        return
    four, two = [], []
    for key, r in results.get("runs", {}).items():
        (two if r.get("source") == SOURCE_2BIN else four).append((key, rq1_summary(r)))
    if not four or not two:
        logger.info("bins figure skipped: need both 4-bin runs (%d) and the 2-bin import "
                    "(%d). Pass --s2bin-dir to add the vendor export.", len(four), len(two))
        return

    fig, axes = plt.subplots(1, len(RQ1_METRICS),
                             figsize=fp.panel_figsize(len(RQ1_METRICS), frac=0.49,
                                                      height_in=3.9))
    rng = np.random.default_rng(0)
    # A run whose iodine map reads negative on the high lesions, or falls as iodine rises,
    # is not a concentration map, and every ratio computed from it is small for the wrong
    # reason.  Marked, not dropped -- it belongs to the envelope, but a reader must not take
    # a low point from such a map as a good result.  The two defects share one marker
    # because they carry the same warning; the console table names which is which.
    inv = np.array([not bool(s.get("usable", True)) for _k, s in four])
    for ax, (mkey, title, ylab) in zip(np.atleast_1d(axes), RQ1_METRICS):
        raw = np.array([s[mkey] for _k, s in four], float)
        ok = np.isfinite(raw)
        vals, flag = raw[ok], inv[ok]
        if vals.size:
            ax.boxplot(vals, positions=[0], widths=0.55, showfliers=False,
                       medianprops={"color": "tab:blue"})
            jitter = rng.normal(0, 0.055, vals.size)
            ax.plot(jitter[~flag], vals[~flag], "o", ms=5, alpha=0.65,
                    color="tab:blue", label=f"4-bin runs (n={vals.size})")
            if flag.any():
                ax.plot(jitter[flag], vals[flag], "o", ms=6, mfc="none", mew=1.3,
                        color="tab:blue",
                        label=f"iodine map not usable (n={int(flag.sum())})")
        for k, s in two:
            v = s[mkey]
            if np.isfinite(v):
                ax.axhline(v, color="tab:red", lw=2.0, ls="--",
                           label=fp.pretty("s2bin"))
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.set_title(title, fontsize=fp.PRINT_FONT_PT)
        # Direction in the axis label, the way the trade-off figures carry it -- so the
        # panel title can stay a name and the reading rule still travels with the figure.
        ax.set_ylabel(f"{RQ1_AXIS_LABELS.get(mkey, ylab)}\n$\\downarrow$ better",
                      fontsize=fp.PRINT_FONT_PT)
        ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
        ax.grid(alpha=0.25, axis="y")
    # "iodine only" stays in the title rather than moving to the caption: it is not a
    # description of the figure, it is the definition of what the numbers are, and a
    # reader who misses it would take the panels for a full material comparison.
    fig.suptitle("Four bins against two  (iodine map only)",
                 fontsize=fp.PRINT_FONT_PT + 1)
    _legend_below(fig, np.atleast_1d(axes)[0], ncol=2, fontsize=fp.PRINT_FONT_PT - 1)
    fig.savefig(path, dpi=200)
    if cfg.split_figures:
        _save_panels(fig, np.atleast_1d(axes), Path(path).parent, Path(path).stem,
                     [k for k, *_r in RQ1_METRICS], dpi=200)
    plt.close(fig)


_CAPTIONS = {
    "bins_4_vs_2.png": (
        "4-bin versus 2-bin decomposition",
        "Every 4-bin run (modality x estimator x basis) as a distribution, with the "
        "scanner's own 2-bin product as a single line across it. Presented as an envelope "
        "against a point rather than a two-bar chart because the vendor product differs "
        "from ours in channel count, solver, reconstruction and proprietary denoising at "
        "once: only where the 2-bin value falls outside the entire envelope is the channel "
        "count the one property every run beating it shares. Both sides are scored on the "
        "IODINE map alone -- with two energy channels a linear decomposition estimates at "
        "most two materials, so no calcium map exists on that side, and averaging our "
        "iodine and calcium would credit us with information the other mode is barred from "
        "having. All four quantities are dimensionless ratios of MAGNITUDES within a single "
        "map, which is what allows a vendor map in mg/ml to be compared with ours in g/cm3; "
        "lower is better on every axis. Magnitudes on both sides because several solves put "
        "a whole material map below zero: a signed ratio then reported the most inverted "
        "run as the best of the set, and picked the insert with the LEAST agent as the one "
        "carrying it. An open marker is a run whose iodine map did not come out usable -- it "
        "either reads negative where iodine actually is, or falls as iodine rises. Such a "
        "run belongs to the envelope, but a low value from it is small for the wrong "
        "reason. Noise and absolute concentration are not "
        "claimable here: the acquisitions differ, and the datasheet states no "
        "concentrations."),
    "source_comparison.png": (
        "Source comparison",
        "Separation index, sign-test pass rate and upward-response rate for each input "
        "modality, grouped by basis and estimator -- does reconstruction quality propagate "
        "into the material maps? The three panels answer three different questions and have "
        "to be read together. The first is how far apart the two members of an HU-matched "
        "pair land, which says nothing about direction. The second is whether the two agent "
        "maps move in OPPOSITE directions across that pair, which rules out a solver that "
        "splits one attenuation across two columns of the same sign. The third is whether "
        "each map RISES with its own agent's concentration, which is the only one of the "
        "three that says the map is assigned to the material it is named after: an "
        "ill-conditioned solve passes the second for free by splitting that attenuation "
        "across two columns of opposite sign instead, and can read 100 % on it with a "
        "calcium map that falls as calcium rises. The dotted line is the 2-bin reference "
        "and is iodine-only by necessity, so it is not strictly the same quantity as the "
        "bars, which average both agents."),
    "estimator_tradeoff.png": (
        "Estimator and basis",
        "Bias, noise and runtime against the estimator list in order of cost, one line per "
        "source and basis (colour is the source, line style the basis). The question is "
        "whether a more expensive estimator buys anything, so the estimator is the x axis "
        "and the answer is the slope: a flat line means the extra work bought nothing for "
        "that source. Runtime has its own panel rather than being a marker area -- it is a "
        "different quantity in different units, and an area cannot be read off to better "
        "than a factor of two. Series are offset by a fraction of a tick so that two which "
        "land on identical values -- the answer whenever the basis does not matter -- draw "
        "two visible lines rather than one hiding the other. A cross marks a run stopped "
        "at the time cap: reported as truncated, not returned as if converged."),
    "mode_comparison.png": (
        "Basis sensitivity",
        "The same data and estimator solved on two bases whose base column differs "
        "(CTWater versus soft tissue). The gap between them measures how much the "
        "base-material choice moves the result, which is why both are run rather than one "
        "being argued for."),
    "separation_best.png": (
        "Calcium/iodine separation",
        "Each grey line joins the two inserts of an HU-matched pair -- identical in a "
        "conventional image by construction, so any separation in the material plane is "
        "material information and not contrast. A pair lying along the diagonal means the "
        "solver is tracking HU rather than material, the one failure the phantom exists to "
        "expose. Marker shape is the agent level, colour the layer."),
    "linearity_best.png": (
        "Proportionality",
        "Measured density against relative agent concentration, per layer and agent. The "
        "0 : 1 : 3.1 spacing is INFERRED by us from the datasheet HU and is not a QRM "
        "figure; the per-layer values are 3.06 / 3.31 / 2.99 and the spread between layers "
        "is part of the result rather than noise to be averaged away."),
    "roi_placement.png": (
        "ROI placement",
        "The HU image of every source at each insert layer, with the measurement ROIs "
        "drawn as outlines rather than a filled overlay -- a tint would cover the pixels "
        "the reader is being asked to judge. Every panel is located by PHYSICAL z, not by "
        "slice index: our reconstruction has its own origin and a slightly different slice "
        "spacing from the vendor's, so a shared slice number lands ~200 slices off the phantom. "
        "The 2-bin row is the one that has to be checked: it is a separate acquisition, so "
        "its agreement with a segmentation drawn on the 4-bin WFBP is an assumption rather "
        "than a construction. The wider version under qc/ keeps the surrounding body in "
        "frame, which is what shows whether the module itself sits in the right place."),
    "material_maps.png": (
        "Material maps",
        "The iodine and CaHA maps each source yields, at the insert layer where iodine "
        "contrast is largest, on one window per material computed over all three sources "
        "together -- a per-panel window would rescale a noisy map into looking like a clean "
        "one, which is the difference being shown. Only the sources we solved appear: the "
        "vendor's 2-bin map is in mg/ml where ours is g/cm3, so it cannot share this "
        "colour scale and has its own normalised figure."),
    "rq1_iodine_maps.png": (
        "Iodine map, 4 bins versus 2 bins",
        "The same physical slice from our 4-bin iodine map and the scanner's 2-bin one, "
        "each divided by its OWN high-lesion reading. That normalisation is the same "
        "device the RQ1 metrics rest on -- every one of them is a ratio within a single "
        "map, because the vendor works in mg/ml, we work in g/cm3, and the datasheet "
        "states no concentrations at all. After it, 1 means 'as bright as a high lesion of "
        "this map' on both sides. The two members of each HU-matched pair are outlined "
        "differently because the claim being checked is that one of them should be dark: "
        "brightness on the dashed outline IS the leak that leak_ratio reports."),
    "rq1_capability.png": (
        "Calcium map: a capability of the 4-bin acquisition",
        "N energy channels determine at most N materials, so a 2-bin acquisition yields no "
        "calcium map at all -- not a worse one. The empty panel is labelled rather than "
        "omitted, because an omitted panel reads as an oversight where a labelled one "
        "states the restriction. This is deliberately kept apart from the performance "
        "numbers: it is a property of the acquisition, and no solver on the vendor's side "
        "could close it."),
    "contact_sheet_Iodine.png": (
        "Iodine at the three concentration levels",
        "The proportionality plot's three points, shown as the inserts they were measured "
        "on: one row per layer, the base, low and high lesion across it, cropped from the "
        "material map at one shared window with the measured density underneath. The "
        "thumbnails should brighten 0 : 1 : 3.1 along a row. The insert shown at each "
        "level is the pair member with the larger reading of this agent, which is the "
        "member the linearity fit uses -- so the sheet also audits that choice: if the "
        "'high' thumbnail is not the brightest lesion in its row, the fit is describing "
        "the wrong inserts."),
    "contact_sheet_HA.png": (
        "CaHA at the three concentration levels",
        "As for iodine: one row per layer, base / low / high across it, one shared window, "
        "the measured density underneath. Read together with the iodine sheet the two "
        "state the phantom's design -- the lesion that is bright here should be the one "
        "that is dark there, because the pair members are the same HU by construction."),
    "material_space.png": (
        "Insert voxels in the material plane",
        "Every insert voxel plotted as (iodine, CaHA), one panel per source, with a black "
        "line joining the two clouds of each HU-matched pair. This is the sign test at "
        "voxel level rather than at ROI-mean level: a working decomposition sends the two "
        "members of a pair in different directions from the base cloud, whereas a solver "
        "tracking HU puts them on one line through the origin, because it is reading a "
        "single contrast and splitting it between two columns in a fixed proportion. Two "
        "overlapping ROI means cannot show that line; the clouds can."),
}


def _write_captions(fig_dir):
    """
    Draft captions, kept out of the figures themselves.

    Titles name what is plotted; a title that explains how to read the plot, or states its
    conclusion, is a caption that has wandered into the image, where it cannot be edited
    when the figure is placed in the thesis.
    """
    lines = ["# Figure captions -- decomposition validation", "",
             "Draft text for the thesis. The figures carry short titles naming the "
             "quantity; everything about how to read them belongs in the caption.", ""]
    panel_dir = Path(fig_dir) / "panels"
    panels = sorted(p.name for p in panel_dir.glob("*.png")) if panel_dir.is_dir() else []
    if panels:
        lines += [
            "## Single panels (`panels/`)", "",
            "Each panel of a multi-panel figure is also written on its own, cropped from "
            "the same rendered figure so the shared axis limits and the shared image "
            "window are identical -- panels stitched back together stay comparable. Place "
            "them at a readable width and include `<stem>_legend.png` once beneath the "
            "group:", "",
            "```latex",
            "\\begin{figure}[t]\\centering",
            "  \\includegraphics[width=.49\\textwidth]{panels/bins_4_vs_2_leak_ratio}\\hfill",
            "  \\includegraphics[width=.49\\textwidth]{panels/bins_4_vs_2_base_leak_frac}"
            "\\\\[2pt]",
            "  \\includegraphics[width=.49\\textwidth]{panels/bins_4_vs_2_cross_layer_cv}"
            "\\hfill",
            "  \\includegraphics[width=.49\\textwidth]{panels/bins_4_vs_2_ratio_error}"
            "\\\\[4pt]",
            "  \\includegraphics[width=.6\\textwidth]{panels/bins_4_vs_2_legend}",
            "  \\caption{...}",
            "\\end{figure}",
            "```", "",
            "Available: " + ", ".join(f"`{n}`" for n in panels), "",
        ]
    for name, (title, body) in _CAPTIONS.items():
        if (fig_dir / name).exists():
            lines += [f"## `{name}`", "", f"**{title}.** {body}", ""]
    (fig_dir / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def _figures(cfg, out, results, boundary, water):
    """
    Per-run detail for the BEST run only, plus the summary figures.

    24 separation plots and 24 linearity plots is not a result set, it is a directory.  The
    detail figures go to the run that wins run_score(); everything else is compared in the
    three summary figures, which is where a reader would look anyway.  --all-figures
    restores one per run when a specific combination needs inspecting.
    """
    fig_dir = Path(out) / "figures"
    # Loaded once, drawn twice: the wide grid is the check (is the module in the right
    # place inside the body?) and the tight one is the thesis figure (are the ROIs on the
    # inserts?).  Re-reading two full DICOM series to answer the second would be minutes.
    imgs = _load_qc_images(cfg, results["match"])
    if imgs:
        # Automatic, not just visual: the grids below show a human whether the ROIs land
        # right, but a few degrees of rotation is invisible by eye and still moves an ROI
        # 8-9 mm at the module's radius.  Done here because the volumes are already in
        # memory, and once -- it is a property of the data, not of a figure.
        _check_pose_against_reference({t: (v[0], v[1], float(v[1].GetSpacing()[0]))
                                       for t, v in imgs.items()})
        _roi_placement_figure(Path(out) / "qc" / "qc_roi_placement.png", cfg,
                              results["match"], imgs, pad_mm=70.0, wide=True)
        _roi_placement_figure(fig_dir / "roi_placement.png", cfg, results["match"], imgs,
                              pad_mm=12.0, wide=False, split=cfg.split_figures)
        del imgs
        gc.collect()
    runs = results.get("runs", {})
    if not runs:
        return
    inserts = results["match"]["inserts"]
    if cfg.all_figures:
        for key, run in runs.items():
            _separation_figure(fig_dir / f"separation_{key}.png", run, inserts, cfg)
            _linearity_figure(fig_dir / f"linearity_{key}.png", run, cfg)
    else:
        rankable = {k: r for k, r in runs.items() if is_rankable(r)}
        if not rankable:
            logger.warning("no run carries both agents -- no best run to draw detail for")
            _bins_figure(fig_dir / "bins_4_vs_2.png", results, cfg)
            _mode_comparison_figure(fig_dir / "mode_comparison.png", results, cfg)
            _estimator_tradeoff_figure(fig_dir / "estimator_tradeoff.png", results, cfg,
                                       split=cfg.split_figures)
            _source_comparison_figure(fig_dir / "source_comparison.png", results, cfg)
            _material_gallery_figure(fig_dir / "material_maps.png", cfg, results,
                                     split=cfg.split_figures)
            _rq1_maps_figure(fig_dir / "rq1_iodine_maps.png", cfg, results,
                             split=cfg.split_figures)
            _write_captions(fig_dir)
            return
        best_key = max(rankable, key=lambda k: run_score(rankable[k]))
        best = rankable[best_key]
        sc = run_score(best)
        logger.info("detail figures for the best run: %s (%.0f%% of pairs sign-correct, "
                    "%.0f%% of agent-layer fits responding upward, mean separation index "
                    "%.2f). Pass --all-figures for one per run.",
                    best_key, 100 * sc.sign, 100 * max(sc.response, 0.0), sc.index)
        # "Best" is a ranking within this set, not a certificate. If the winner is itself
        # inverted or anti-correlated, the figures drawn from it are figures of a failed
        # solve and have to say so here, not only in the summary table further down.
        bad = inverted_agents(best) + anti_correlated_agents(best)
        if bad:
            logger.warning("   the best-ranked run is itself defective: %s. The detail "
                           "figures below show a failed solve -- read them as such.",
                           ", ".join(sorted(set(bad))))
        results["best_run"] = best_key
        _separation_figure(fig_dir / "separation_best.png", best, inserts, cfg)
        _linearity_figure(fig_dir / "linearity_best.png", best, cfg)
        # The contact sheet is the linearity plot's three dots shown as the inserts they
        # were measured on, so it is drawn for the same run and only that run.
        for mat in AGENTS:
            _contact_sheet_figure(fig_dir / f"contact_sheet_{mat}.png", cfg, results, best,
                                  mat, split=cfg.split_figures)
    _bins_figure(fig_dir / "bins_4_vs_2.png", results, cfg)
    _mode_comparison_figure(fig_dir / "mode_comparison.png", results, cfg)
    _estimator_tradeoff_figure(fig_dir / "estimator_tradeoff.png", results, cfg,
                               split=cfg.split_figures)
    _source_comparison_figure(fig_dir / "source_comparison.png", results, cfg)
    # Image figures last: they re-read the material maps from disk, so a failure here
    # cannot cost the metric figures that are already written.
    _material_gallery_figure(fig_dir / "material_maps.png", cfg, results,
                             split=cfg.split_figures)
    _rq1_maps_figure(fig_dir / "rq1_iodine_maps.png", cfg, results,
                     split=cfg.split_figures)
    _capability_figure(fig_dir / "rq1_capability.png", cfg, results,
                       split=cfg.split_figures)
    _material_space_figure(fig_dir / "material_space.png", cfg, results,
                           split=cfg.split_figures)
    _write_captions(fig_dir)


def _print_summary(results):
    print("\n" + "=" * 140)
    print("DECOMPOSITION VALIDATION -- QRM DEP-V5")
    print("=" * 140)
    print(f"{'run':44s} {'bins':>11s} {'pairs':>7s} {'sep idx':>9s} {'sign ok':>9s} "
          f"{'resp up':>9s} {'leak':>7s} {'orient':>9s} "
          f"{'|bias|':>11s} {'hi/lo':>8s} {'runtime':>9s}")
    print("-" * 140)
    defective = {}
    for key, r in sorted(results["runs"].items()):
        all_doped = [s for s in r.get("separation", []) if s.get("role") in ("low", "high")]
        sep = doped_pairs(all_doped)
        idx = _mean_of_finite([_mean_of_finite([s.get(f"sep_index_{m}") for m in AGENTS])
                               for s in sep])
        # Only pairs that carry a verdict. A rank-limited run has none -- averaging its
        # absent verdicts as zeros printed a flat 0 %, which reads as every pair failing
        # the sign test rather than as the test not existing for that run.
        verdicts = [1.0 if s["sign_ok"] else 0.0 for s in sep if "sign_ok" in s]
        ok = 100 * float(np.mean(verdicts)) if verdicts else float("nan")
        ok_s = f"{ok:8.0f}%" if np.isfinite(ok) else f"{'n/a':>8s} "
        spec = r.get("specificity", {})
        bias = _mean_of_finite([abs(spec[m]["mean"]) for m in spec])
        lin = r.get("linearity", {})
        hl = _mean_of_finite([e["measured_high_over_low"] for e in lin.values()])
        # The vendor run was imported, not solved, so it has no domain -- shown as
        # 'imported' rather than a bare None, which would read as a missing value.
        dom = r.get("bin_domain") or ("imported" if r.get("imported") else "?")
        # Orientation and response belong in THIS table, not only in the RQ1 block below.
        # This is the table the numbers get quoted from, and a run whose map is upside down
        # or falls as its own agent rises was printing here as an ordinary result.
        resp = r.get("response") or {}
        n_pos = sum((resp.get(m) or {}).get("n_positive", 0) for m in AGENTS)
        n_tot = sum((resp.get(m) or {}).get("n_layers", 0) for m in AGENTS)
        resp_s = f"{f'{n_pos}/{n_tot}':>9s}" if n_tot else f"{'n/a':>9s}"
        leak = mean_leak_ratio(r)
        inv, anti = inverted_agents(r), anti_correlated_agents(r)
        short = {"Iodine": "I", "HA": "Ca"}
        marks = ([f"{short.get(m, m)}-" for m in inv]
                 + [f"{short.get(m, m)}~" for m in anti if m not in inv])
        if marks:
            defective[key] = (inv, anti)
        print(f"{key:44s} {dom:>11s} {f'{len(sep)}/{len(all_doped)}':>7s} "
              f"{idx:9.2f} {ok_s} {resp_s} {leak:7.2f} {','.join(marks) or 'ok':>9s} "
              f"{bias:11.2e} {hl:8.2f} "
              f"{r.get('runtime_s', float('nan')):8.0f}s"
              + ("  TRUNCATED" if r.get("truncated") else "")
              + ("" if is_rankable(r) else "  [iodine only]"))
    print("=" * 140)
    # ASCII on purpose: a Windows console defaults to cp1252 and a stray non-ASCII
    # character here aborts the whole summary, taking the RQ1 table below with it.
    print("bins     = the stack the source was solved from. own=cumulative (thresholds fed")
    print("           as-is), wfbp=exclusive (separate bins, image-domain differencing),")
    print("           vmi=direct (monoenergetic keV -- nothing nested to subtract).")
    print("pairs    = doped pairs MEASURED / drawn. A pair whose ROI produced no voxels on")
    print("           this source's grid reads NaN everywhere and is excluded from every")
    print("           column right of here -- scoring it would count a missing measurement")
    print("           as a failed sign test. Six doped pairs are drawn (3 layers x low+high);")
    print("           anything under 6/6 means check qc/qc_roi_placement.png for that source.")
    print("sep idx  = |delta density| / pooled SD between the two members of a matched pair")
    print("sign ok  = fraction of pairs where iodine and CaHA move in OPPOSITE directions;")
    print("           anything below 100 % means the solver is partly tracking HU, not material")
    print("resp up  = agent x layer fits whose map RISES with its own agent's concentration.")
    print("           sign ok cannot see this: it only demands the two maps move in opposite")
    print("           directions, which an ill-conditioned solve satisfies for free by")
    print("           splitting one attenuation across two columns of OPPOSITE sign. A run")
    print("           can read 100 % sign ok with a calcium map that falls as calcium rises.")
    print("leak     = agent's reading on the pair member that does NOT carry it, over its")
    print("           reading on the one that does. 0 = separated, 1 = pure HU tracking. Read")
    print("           it WITH sep idx: a large separation at leak ~1 is not material contrast.")
    print("orient   = map orientation. 'ok', or the agents that are defective: I-/Ca- reads")
    print("           NEGATIVE on the lesions carrying the most agent (not a concentration")
    print("           map at all), I~/Ca~ falls as its own agent rises in every layer.")
    print("|bias|   = apparent agent density in the base lesions, which contain none")
    print("hi/lo    = measured high/low ratio; the INFERRED nominal is ~3.1 (not a QRM figure)")
    print("[iodine only] = rank-limited: 2 energy channels cannot yield a calcium map, so")
    print("           sign ok and the both-agent averages do not exist for that run. This")
    print("           table is NOT the 4-vs-2 comparison -- it mixes agent counts. Read")
    print("           figures/bins_4_vs_2.png, which scores iodine alone on both sides.")

    # Named once, in full, so a reader quoting this table into a document cannot copy a
    # defective run as an ordinary result. Every metric here is a magnitude, so a failed
    # solve produces small, plausible numbers rather than an obviously broken one.
    if defective:
        print()
        print(f"WARNING  {len(defective)} of {len(results['runs'])} run(s) did not produce a "
              "usable material map.")
        print("         Their sep idx, sign ok, |bias| and hi/lo are computed from")
        print("         magnitudes and look ordinary. They are not results:")
        for k, (inv, anti) in sorted(defective.items()):
            why = []
            if inv:
                why.append("sign-inverted: " + ", ".join(inv))
            if anti:
                why.append("anti-correlated with its own agent: " + ", ".join(anti))
            print(f"           {k:44s} {'; '.join(why)}")

    _print_rq1(results)


def _print_rq1(results):
    """
    The 4-bin envelope against the 2-bin point, in numbers.

    Every column is dimensionless and lower is better, which is what makes a vendor map in
    mg/ml comparable with ours in g/cm3 -- the datasheet gives no concentrations, so an
    absolute scale is not available to either side.
    """
    four, two = [], []
    for key, r in sorted(results.get("runs", {}).items()):
        (two if r.get("source") == SOURCE_2BIN else four).append((key, rq1_summary(r)))
    if not two:
        return
    cols = [k for k, *_rest in RQ1_METRICS]
    print("\n" + "=" * 104)
    print("RQ1  --  4-bin decomposition vs the scanner's 2-bin mode   (IODINE ONLY, both sides)")
    print("=" * 104)
    hdr = f"{'':30s}" + "".join(f"{c:>18s}" for c in cols)
    print(hdr)
    print("-" * 104)

    def _envelope(rows, label):
        if not rows:
            return
        arr = {c: np.array([s[c] for _k, s in rows], float) for c in cols}
        for name, fn in ((f"{label}  best", np.nanmin), (f"{label}  median", np.nanmedian),
                         (f"{label}  worst", np.nanmax)):
            vals = []
            for c in cols:
                v = arr[c][np.isfinite(arr[c])]
                vals.append(fn(v) if v.size else float("nan"))
            print(f"{name:30s}" + "".join(f"{v:18.4f}" for v in vals))

    _envelope(four, "4-bin")
    # The envelope argument -- "a difference belongs to the bin count only where the 2-bin
    # point falls outside the whole 4-bin distribution" -- is only as good as the runs in
    # that distribution. A run whose iodine map is inverted or anti-correlated widens the
    # envelope in whichever direction its magnitudes happen to land, so the same claim is
    # printed twice: over every run, and over the runs that produced a usable iodine map.
    # If the two disagree, the disagreement is the result.
    usable = [(k, s) for k, s in four if s.get("usable")]
    if usable and len(usable) != len(four):
        print("-" * 104)
        _envelope(usable, "  usable only")
        print(f"{'':30s}" + f"  ({len(usable)} of {len(four)} four-bin runs produced a "
              "usable iodine map)")
    print("-" * 104)
    for key, s in two:
        print(f"{'2-bin  ' + key.split('__')[0]:30s}"
              + "".join(f"{s[c]:18.4f}" for c in cols))
    print("-" * 104)
    for c, title, ylab in RQ1_METRICS:
        print(f"{c:18s} {title} -- {ylab}")
    print("LOWER IS BETTER in every column. The claim holds only where the 2-bin value")
    print("falls outside the whole 4-bin range -- then the one property shared by every")
    print("run that beats it is the channel count, not the solver or the reconstruction.")
    print("NOT claimable from this: noise (different acquisition) or absolute concentration.")
    # A sign-inverted or anti-correlated map makes every ratio above small for the wrong
    # reason, so the runs it affects are named rather than left to sit low on the axis
    # looking like winners.
    bad = [(k, s) for k, s in four + two if not s.get("usable", True)]
    if bad:
        print()
        print(f"WARNING  {len(bad)} run(s) did not produce a usable iodine map. Every ratio")
        print("         above is a magnitude, so these still produce small, good-looking")
        print("         numbers. Treat them as failed solves, not as results:")
        for k, s in bad:
            why = []
            if s.get("reference_negative"):
                why.append("reads NEGATIVE on the high-iodine lesions (sign-inverted)")
            if not s.get("response_ok", True):
                why.append("falls as iodine rises (anti-correlated)")
            print(f"           {k:44s} {'; '.join(why)}")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m decomposition.decomp_validation",
        description="Validate the material decomposition against the QRM DEP-V5 phantom.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", default="all", choices=["0", "1", "2", "all"])
    p.add_argument("--out-root", default=ValidationConfig.out_root)
    p.add_argument("--own-dir", default=ValidationConfig.own_dir)
    p.add_argument("--wfbp-dir")
    p.add_argument("--vmi-dir")
    p.add_argument("--s2bin-dir",
                   help="the Siemens 2-BIN export (PBV + VNC folders). Its maps are "
                        "IMPORTED, not solved -- this is the RQ1 reference point")
    p.add_argument("--s2bin-iodine-match", default=ValidationConfig.s2bin_iodine_match,
                   help="series-description substring for the vendor IODINE map. PBV "
                        "(pulmonary blood volume) is the iodine map; VNC is not")
    p.add_argument("--s2bin-vnc-match", default=ValidationConfig.s2bin_vnc_match,
                   help="series-description substring for the VNC image (QC only)")
    p.add_argument("--probe-siemens", metavar="DIR",
                   help="list every series in a Siemens export with the metadata that "
                        "decides usability (photometric interpretation, rescale, value "
                        "range), then exit. Run this before --s2bin-dir")
    p.add_argument("--segmentation",
                   help="default: <wfbp-dir>/Segmentation_decomposition.nrrd")
    p.add_argument("--boundary-label", type=int, default=10)
    p.add_argument("--sources", default="own,wfbp,vmi",
                   help=f"comma-separated; add '{SOURCE_2BIN}' for the vendor 2-bin import")
    p.add_argument("--mode", default=ValidationConfig.mode)
    p.add_argument("--sensitivity-mode", default=ValidationConfig.sensitivity_mode,
                   help="second basis, run for comparison; '' to skip")
    p.add_argument("--estimators", default="ols,wls,wls_denoise,wls_joint")
    p.add_argument("--bin-domain", default=None,
                   choices=["cumulative", "exclusive", "direct"],
                   help="force ONE domain on every source, overriding the per-source "
                        "defaults (own=cumulative, wfbp=exclusive, vmi=direct). Only for "
                        "deliberately re-running a source outside its natural domain")
    p.add_argument("--bin-domain-map", default=None, metavar="SRC=DOMAIN,...",
                   help="override the per-source domain for named sources only, "
                        "e.g. 'wfbp=cumulative'. Sources not named keep their default")
    p.add_argument("--timeout", type=float, default=ValidationConfig.estimator_timeout_s,
                   help="per-run wall-clock cap in seconds; a run past it is reported as "
                        "truncated rather than as a converged result")
    p.add_argument("--own-frame-fix", dest="own_frame_fix", action="store_true",
                   help="mirror and re-origin output/reconstruction/ before decomposing. "
                        "OFF by default: python_reconstruction.py now writes in the vendor "
                        "frame itself, so applying it again would flip the volume back. "
                        "Use only for volumes reconstructed before that change")
    p.set_defaults(own_frame_fix=False)
    p.add_argument("--no-water-calibration", dest="water_calibration",
                   action="store_false",
                   help="solve on the physical per-channel water attenuation instead of "
                        "fitting a per-channel gain so a water ROI reads the tabulated "
                        "value. ON by default. The reconstruction carries a "
                        "channel-dependent HU offset that enters the solve as an additive "
                        "error no basis column can absorb, and this gain is the only lever "
                        "the pipeline has on it, so the two settings are worth comparing")
    p.set_defaults(water_calibration=True)
    p.add_argument("--signature-weighting", default=ValidationConfig.signature_weighting,
                   choices=["uniform", "spectrum"],
                   help="how the CUMULATIVE window averages of M are weighted over "
                        "energy. 'uniform' (default) is what the published bin-averaged "
                        "table itself is and must stay the default. 'spectrum' weights by "
                        "the photon fluence a counting detector sees, which is defensible "
                        "for the nested windows [T_i, 140] that this code integrates "
                        "itself: a flat mean over 120 keV weights 20 keV, where iodine "
                        "reads 25 cm^2/g and hardly a photon survives the patient, the "
                        "same as 100 keV where the counts are. Reaches the 'own' source "
                        "only -- wfbp reads the published disjoint rows and vmi is point "
                        "values. Recorded in the run signature, so it re-solves rather "
                        "than reusing a cached run")
    p.add_argument("--spectrum-kvp", type=float, default=ValidationConfig.spectrum_kvp,
                   help="bremsstrahlung endpoint for --signature-weighting spectrum")
    p.add_argument("--spectrum-water-cm", type=float,
                   default=ValidationConfig.spectrum_water_cm,
                   help="water path used as the phantom filtration for the same. The "
                        "tube's own filtration is not modelled, which makes the beam "
                        "softer than the real one, so the test errs towards understating "
                        "the difference rather than overstating it")
    p.add_argument("--all-figures", action="store_true",
                   help="one separation and linearity figure per run (24 of each). Default "
                        "is detail for the best-scoring run only, plus the summaries")
    p.add_argument("--no-split-figures", dest="split_figures", action="store_false",
                   help="do not also write each panel of a multi-panel figure on its own "
                        "under figures/panels/ (default: write them, for LaTeX)")
    p.set_defaults(split_figures=True)
    p.add_argument("--free-axes", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    a = build_parser().parse_args(argv)
    if a.probe_siemens:
        probe_siemens_export(a.probe_siemens, out_root=a.out_root)
        return 0
    cfg = ValidationConfig(
        out_root=a.out_root, own_dir=a.own_dir, wfbp_dir=a.wfbp_dir, vmi_dir=a.vmi_dir,
        s2bin_dir=a.s2bin_dir, s2bin_iodine_match=a.s2bin_iodine_match,
        s2bin_vnc_match=a.s2bin_vnc_match,
        segmentation=a.segmentation, boundary_label=a.boundary_label,
        sources=tuple(s.strip() for s in a.sources.split(",") if s.strip()),
        mode=a.mode, sensitivity_mode=(a.sensitivity_mode or None),
        estimators=tuple(s.strip() for s in a.estimators.split(",") if s.strip()),
        bin_domain=a.bin_domain, estimator_timeout_s=a.timeout,
        own_frame_fix=a.own_frame_fix, water_calibration=a.water_calibration,
        signature_weighting=a.signature_weighting, spectrum_kvp=a.spectrum_kvp,
        spectrum_water_cm=a.spectrum_water_cm,
        all_figures=a.all_figures,
        split_figures=a.split_figures, free_axes=a.free_axes, force=a.force)
    if a.bin_domain_map:
        allowed = ("cumulative", "exclusive", "direct")
        for item in a.bin_domain_map.split(","):
            if not item.strip():
                continue
            src, _, dom = item.partition("=")
            src, dom = src.strip(), dom.strip()
            if dom not in allowed:
                raise SystemExit(f"--bin-domain-map: '{item.strip()}' -- domain must be "
                                 f"one of {allowed}")
            cfg.bin_domain_by_source[src] = dom
    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)
    for s in cfg.sources:
        if s != SOURCE_2BIN:
            logger.info("source %-5s -> %s bins", s, bin_domain_for(cfg, s))
    _write_json(Path(cfg.out_root) / "validation_config.json", cfg.to_dict())

    if a.stage in ("0", "all"):
        stage0_probe(cfg)
    if a.stage in ("1", "all"):
        stage1_decompose(cfg)
    if a.stage in ("2", "all"):
        stage2_metrics(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
