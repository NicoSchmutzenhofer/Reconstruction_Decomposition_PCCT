"""
probe_dicom_origin.py -- do our volumes and the vendor export share an in-plane frame?

A segmentation is resampled onto both by physical coordinate, so an x/y offset or a rotation
between the two frames would misplace it on one of them without any error.  Three checks,
cheapest first:

  1. HEADERS  vendor ImagePositionPatient / ImageOrientationPatient vs an isocentre-centred
              FOV of the same size with identity direction.
  2. POSE     body centroid and orientation of both volumes, in physical mm.  A shift points
              at the origin header; a rotation points at the view-angle convention
              (invariant #8, README) and needs a re-reconstruction.
  3. SEGMENT  physical bounding box and direction matrix of the segmentation.

CPU only, seconds, writes nothing.

    python -m reconstruction.probe_dicom_origin --wfbp-dir /path/to/wfbp
    python -m reconstruction.probe_dicom_origin            # paths from match_config.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconstruction import image_quality_metrics as iq
from reconstruction.recon_comparison import (
    _REPO_ROOT, index_dicom_series, select_family, read_series_volume, load_segmentation)


def _sitk():
    import SimpleITK as sitk
    return sitk


def _header(path):
    """ImagePositionPatient / ImageOrientationPatient / grid of one DICOM file."""
    sitk = _sitk()
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()

    def tag(t, default=""):
        try:
            return r.GetMetaData(t).strip()
        except RuntimeError:
            return default

    return {
        "ipp": [float(v) for v in tag("0020|0032", "nan\\nan\\nan").split("\\")],
        "iop": [float(v) for v in tag("0020|0037", "\\".join(["nan"] * 6)).split("\\")],
        "spacing": [float(v) for v in tag("0028|0030", "nan\\nan").split("\\")],
        "rows": int(tag("0028|0010", "0") or 0),
        "cols": int(tag("0028|0011", "0") or 0),
        "recon_diameter": tag("0018|1100", ""),
        "table_height": tag("0018|1130", ""),
        "patient_position": tag("0018|5100", ""),
    }


def _physical_xy(idx_xyz, img):
    """Vectorised index -> physical (x, y) mm, honouring the direction matrix."""
    d = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return (np.asarray(img.GetOrigin(), float)
            + (idx_xyz * np.asarray(img.GetSpacing(), float)) @ d.T)[:, :2]


def _shape_frame(vol, img, label, dense_hu=300.0):
    """
    Position AND orientation of the body in the middle slice, in physical mm.

    A centroid is a first moment and is blind to rotation about itself, so two further
    quantities are measured:

      * PRINCIPAL AXIS of the body mask, from the second central moments.  Precise for an
        elongated phantom, but an axis is a line rather than an arrow, so it is only
        defined modulo 180 deg.  `elongation` says whether to trust it at all -- for a
        near-circular body the axis is arbitrary and the angle is meaningless.
      * BEARING of the dense structure (the spine insert) from the body centroid.  Much
        coarser, but a full 360 deg and unambiguous, so it resolves the modulo-180
        question the principal axis leaves open.

    Both are needed: the principal axis alone cannot tell a 3 deg rotation from a 183 deg
    one, and the bearing alone is too coarse to see a few degrees.
    """
    arr = np.asarray(vol)
    k = arr.shape[0] // 2
    sl = arr[k]
    sx, sy = img.GetSpacing()[0], img.GetSpacing()[1]
    mask = iq.detect_body_mask(sl, 0.5 * (sx + sy))
    if not mask.any():
        print(f"    {label:<28s} no body found in the middle slice")
        return None

    yy, xx = np.nonzero(mask)
    idx = np.stack([xx, yy, np.full(xx.size, k)], axis=1).astype(float)
    pts = _physical_xy(idx, img)
    centre = pts.mean(axis=0)

    # Principal axis from the 2x2 covariance of the mask coordinates.
    evals, evecs = np.linalg.eigh(np.cov(pts.T))
    major = evecs[:, int(np.argmax(evals))]
    axis_deg = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    lo, hi = float(np.min(evals)), float(np.max(evals))
    elong = float(np.sqrt(hi / lo)) if lo > 0 else float("inf")

    # Bearing of the dense structure (spine) from the body centroid.
    dense = mask & (sl > dense_hu)
    bearing_deg, n_dense = float("nan"), int(dense.sum())
    if n_dense >= 20:
        dy, dx = np.nonzero(dense)
        didx = np.stack([dx, dy, np.full(dx.size, k)], axis=1).astype(float)
        v = _physical_xy(didx, img).mean(axis=0) - centre
        if np.hypot(*v) > 2.0:                     # ignore a spine sitting on the centroid
            bearing_deg = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)

    o = img.GetOrigin()
    print(f"    {label:<28s} origin ({o[0]:+8.2f}, {o[1]:+8.2f})   "
          f"centroid ({centre[0]:+8.2f}, {centre[1]:+8.2f}) mm   {mask.sum()} px")
    print(f"    {'':<28s} principal axis {axis_deg:6.2f} deg "
          f"(elongation {elong:.2f})   dense bearing "
          f"{bearing_deg:6.2f} deg ({n_dense} px)")
    return {"centre": centre, "axis_deg": axis_deg, "elong": elong,
            "bearing_deg": bearing_deg}


def main(argv=None):
    # Plain-ASCII description on purpose: argparse echoes it to stdout, and a Windows
    # console in cp1252 cannot encode the box-drawing characters used in the docstring.
    p = argparse.ArgumentParser(
        description="Check whether our reconstructed volumes and the Siemens export "
                    "share an in-plane coordinate frame (see the module docstring).")
    p.add_argument("--out-root",
                   default=str(_REPO_ROOT / "output" / "research" / "recon_comparison"))
    p.add_argument("--wfbp-dir", default=None, help="default: from match_config.json")
    p.add_argument("--vmi-dir", default=None)
    p.add_argument("--segmentation", default=None, help="default: from match_config.json")
    p.add_argument("--variant", default=None,
                   help="sweep variant to compare against (default: the first one found)")
    a = p.parse_args(argv)

    sitk = _sitk()
    out = Path(a.out_root)
    match = {}
    mpath = out / "match_config.json"
    if mpath.exists():
        import json
        match = json.loads(mpath.read_text(encoding="utf-8"))
    else:
        print(f"note: {mpath} not found -- run stage 0 first, or pass --wfbp-dir")

    wfbp_dir = a.wfbp_dir or match.get("wfbp_dir")
    vmi_dir = a.vmi_dir or match.get("vmi_dir")
    seg = a.segmentation or match.get("segmentation")
    if not wfbp_dir:
        p.error("no WFBP folder: pass --wfbp-dir or run stage 0 first")

    # ── 1. headers ───────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("1. DICOM HEADERS -- vendor in-plane origin vs an isocentre-centred FOV")
    print("=" * 88)
    families = {}
    for kind, folder in (("wfbp", wfbp_dir), ("vmi", vmi_dir)):
        if not folder:
            continue
        chans = select_family(index_dicom_series(folder, cache_dir=out), kind)
        if not chans:
            print(f"  {kind}: no channels found under {folder}")
            continue
        families[kind] = chans
        c = chans[0]
        h = _header(c["files"][0][1])
        fov = h["cols"] * h["spacing"][0]
        expect = -fov / 2 + h["spacing"][0] / 2
        dx = h["ipp"][0] - expect
        dy = h["ipp"][1] - expect
        print(f"\n  {kind}/{c['label']}   {h['rows']}x{h['cols']}  "
              f"pixel {h['spacing'][0]:.4f} mm  FOV {fov:.1f} mm")
        # IOP is (row_x, row_y, row_z, col_x, col_y, col_z).  Our volumes are written
        # with an identity direction matrix, i.e. row = +x, col = +y, so any departure
        # here is a rotation or flip between the two frames that no origin fix can undo.
        iop = np.asarray(h["iop"], float)
        print(f"    ImageOrientationPatient  {h['iop']}")
        if np.all(np.isfinite(iop)):
            row_deg = float(np.degrees(np.arctan2(iop[1], iop[0])))
            col_deg = float(np.degrees(np.arctan2(iop[4], iop[3])))
            oop = float(max(abs(iop[2]), abs(iop[5])))
            print(f"    -> row axis {row_deg:+7.2f} deg, col axis {col_deg:+7.2f} deg "
                  f"from ours (identity); out-of-plane component {oop:.4f}")
            if abs(row_deg) < 0.5 and abs(col_deg - 90.0) < 0.5 and oop < 1e-3:
                print("       => axial and identity-oriented; same convention as ours.")
            else:
                print("       => NOT identity. The vendor grid is rotated or flipped "
                      "relative to ours; this is a rotation, not an origin offset.")
        print(f"    ImagePositionPatient     x={h['ipp'][0]:+9.3f}  y={h['ipp'][1]:+9.3f}")
        print(f"    isocentre-centred would  x={expect:+9.3f}  y={expect:+9.3f}   "
              f"<- what stage 1 assumes for OUR volumes")
        print(f"    DIFFERENCE               x={dx:+9.3f}  y={dy:+9.3f}  mm")
        if h["recon_diameter"]:
            print(f"    ReconstructionDiameter   {h['recon_diameter']}")
        if h["table_height"]:
            print(f"    TableHeight              {h['table_height']}")
        if h["patient_position"]:
            print(f"    PatientPosition          {h['patient_position']}")
        if abs(dx) < 0.51 and abs(dy) < 0.51:
            print("    => vendor grid IS isocentre-centred; the frames agree on paper.")
        else:
            print("    => vendor grid is NOT isocentre-centred. This offset is exactly "
                  "what displaces the ROIs in our volumes.")

    # ── 2. measured body centroids ───────────────────────────────────────────
    print("\n" + "=" * 88)
    print("2. MEASURED POSE -- translation AND rotation, independent of frame labelling")
    print("=" * 88)
    z_range = ((match["z_lo_mm"], match["z_hi_mm"])
               if "z_lo_mm" in match else None)
    vendor = None
    for kind, chans in families.items():
        vol, img, _z = read_series_volume(chans[0], z_range=z_range)
        f = _shape_frame(vol, img, f"{kind}/{chans[0]['label']}")
        if kind == "wfbp":
            vendor = f
        del vol, img

    ours, vdir = None, None
    sweep = out / "sweep"
    cands = ([sweep / a.variant] if a.variant
             else sorted(d for d in sweep.glob("*") if d.is_dir()))
    for d in cands:
        f = d / "reconstruction_thr_A_HU.nii.gz"
        if f.exists():
            img = sitk.ReadImage(str(f))
            ours = _shape_frame(sitk.GetArrayFromImage(img), img, f"ours/{d.name}")
            vdir = d
            break
    if vdir is None:
        print("    no sweep volume found -- run stage 1 first")

    if vendor is not None and ours is not None:
        d = ours["centre"] - vendor["centre"]
        shifted = np.hypot(*d) >= 3.0
        print(f"\n    TRANSLATION (ours - wfbp)  x={d[0]:+8.2f}  y={d[1]:+8.2f}  mm"
              f"   ({np.hypot(*d):.2f} mm total)")

        # Signed difference of two axes, wrapped into (-90, +90]: an axis is a line, so
        # 179 deg apart is a 1 deg rotation, not a 179 deg one.
        da = (ours["axis_deg"] - vendor["axis_deg"] + 90.0) % 180.0 - 90.0
        weak = min(ours["elong"], vendor["elong"]) < 1.15
        print(f"    ROTATION, principal axis   {da:+8.2f} deg  (mod 180)"
              + ("   UNRELIABLE: body is near-circular, the axis is arbitrary"
                 if weak else ""))

        db = float("nan")
        if np.isfinite(ours["bearing_deg"]) and np.isfinite(vendor["bearing_deg"]):
            db = (ours["bearing_deg"] - vendor["bearing_deg"] + 180.0) % 360.0 - 180.0
            print(f"    ROTATION, dense bearing    {db:+8.2f} deg  (full 360, coarse)")
        else:
            print("    ROTATION, dense bearing         n/a  (no dense structure found; "
                  "the principal axis is then only known modulo 180 deg)")

        rotated = (not weak and abs(da) >= 1.0) or (np.isfinite(db) and abs(db) >= 5.0)
        print()
        if not shifted and not rotated:
            print("    => the two volumes are registered in-plane. The ROI displacement "
                  "is NOT a frame mismatch; look elsewhere.")
        elif shifted and not rotated:
            print("    => PURE TRANSLATION. Matches the overlay. Fixable by correcting "
                  "the written origin -- no re-reconstruction needed.")
        elif rotated and not shifted:
            print("    => PURE ROTATION. Not an origin problem: suspect the view-angle "
                  "convention (_VIEW_ANGLE_OFFSET_DEG, invariant #8 in the README). "
                  "Re-running stage 1 IS required after any change.")
        else:
            print("    => TRANSLATION AND ROTATION together. Fix the rotation first "
                  "(view-angle convention); the residual shift is only meaningful once "
                  "the volumes share an orientation.")
        if np.isfinite(db) and not weak and abs(abs(db) - abs(da)) > 20.0:
            print("    NOTE: the two rotation estimates disagree. The bearing is coarse "
                  "but unambiguous -- if it says ~180 deg, the volumes are flipped and "
                  "the principal axis cannot see it.")

    # ── 3. where the segmentation actually sits ──────────────────────────────
    if seg and Path(seg).exists():
        print("\n" + "=" * 88)
        print("3. SEGMENTATION EXTENT -- which volume does it physically fall inside?")
        print("=" * 88)
        s = sitk.ReadImage(str(seg))
        lo = s.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, 0.0))
        hi = s.TransformContinuousIndexToPhysicalPoint(
            tuple(float(n - 1) for n in s.GetSize()[:3]))
        print(f"    file      {seg}")
        sd = np.asarray(s.GetDirection(), float)
        sd = sd.reshape(3, 3) if sd.size == 9 else sd.reshape(4, 4)[:3, :3]
        print(f"    direction {np.array2string(sd, precision=3, suppress_small=True)}"
              f"   {'identity' if np.allclose(sd, np.eye(3), atol=1e-3) else 'ROTATED'}")
        print(f"    x {min(lo[0], hi[0]):+9.2f} .. {max(lo[0], hi[0]):+9.2f} mm")
        print(f"    y {min(lo[1], hi[1]):+9.2f} .. {max(lo[1], hi[1]):+9.2f} mm")
        print(f"    z {min(lo[2], hi[2]):+9.2f} .. {max(lo[2], hi[2]):+9.2f} mm")
        if vdir is not None:
            img = sitk.ReadImage(str(vdir / "reconstruction_thr_A_HU.nii.gz"))
            lab = load_segmentation(seg, img,
                                    low_priority_label=match.get("background_label"))
            n = 0 if lab is None else int((lab > 0).sum())
            print(f"    voxels landing inside ours/{vdir.name}: {n}")
            o, sz, sp = img.GetOrigin(), img.GetSize(), img.GetSpacing()
            print(f"    ours spans x {o[0]:+9.2f} .. {o[0] + sz[0] * sp[0]:+9.2f} mm, "
                  f"y {o[1]:+9.2f} .. {o[1] + sz[1] * sp[1]:+9.2f} mm")

    print("\n" + "=" * 88)
    print("Read section 2 first: it is a measurement. Section 1 explains WHY.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
