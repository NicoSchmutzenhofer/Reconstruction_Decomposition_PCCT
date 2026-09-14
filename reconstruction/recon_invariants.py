"""
recon_invariants.py -- run-time regression checks for the reconstruction pipeline.

The driver calls these at each key stage, and flush_invariant_log writes the collected
warnings to invariant_log.json, so a change that degrades the output shows up as a diff of
that log.  Existing checks must keep their meaning; new checks are appended.

Geometry checks hard-fail with InvariantError, since a wrong geometry makes the output
wrong.  HU, ring-index, mask-size and slice-continuity checks only warn, so a run of several
hours still finishes and returns its diagnostics.

Vendor reference values (source-to-axis and source-to-detector distance, channel count) are
read from the gitignored geometry/detector_spec.json; without it those comparisons are
reported as unverified.

Usage in python_reconstruction.py:
    check_geometry(geom)                          # after build_geom
    check_defect_mask(geom['spike_mask'], n_ch)   # after detect_defect_channels
    check_threshold_ordering(sino_A, sino_D)      # after loading thr A and D
    check_sinogram_preprocessed(sino_proc, label) # after each preprocess
    check_reconstruction(img, geom, label)        # after each FBP slice
    check_cross_threshold(water_hus)              # after all 4 thresholds
    check_output_format(vol, geom, z_spacing)     # before each NIfTI write
    flush_invariant_log(out_dir)                  # at end of run
"""

import json
import time
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
class InvariantError(RuntimeError):
    """Raised when a hard-fail geometry invariant is violated."""


_LOG: list[dict] = []     # accumulates soft-warn entries for this run


def _warn(check: str, msg: str, data: dict | None = None) -> None:
    entry = {
        "time":  time.strftime("%H:%M:%S"),
        "check": check,
        "msg":   msg,
    }
    if data:
        entry["data"] = data
    _LOG.append(entry)
    print(f"\n[INVARIANT WARN]  {check}: {msg}")
    if data:
        for k, v in data.items():
            print(f"    {k}: {v}")
    print()


def flush_invariant_log(out_dir: Path | str) -> None:
    """Write all accumulated soft-warn entries to <out_dir>/invariant_log.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "invariant_log.json"
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    run_entry = {
        "run_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warnings":  _LOG,
        "n_warnings": len(_LOG),
    }
    existing.append(run_entry)
    path.write_text(json.dumps(existing, indent=2))
    if _LOG:
        print(f"[invariants] {len(_LOG)} warning(s) written to {path}")
    else:
        print(f"[invariants] All checks passed — log at {path}")


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Geometry invariants (hard-fail)
# ─────────────────────────────────────────────────────────────────────────────
# The reference distances and channel count come from geometry/detector_spec.json,
# never from this file; without it those comparisons are soft-warned as unverified.
_SAD_TOL_MM = 0.5
_SDD_TOL_MM = 0.5
_SPEC_FILENAME = "detector_spec.json"
_SPEC_KEYS = ("sad_mm", "sdd_mm", "n_channels")


def load_detector_spec(geo_dir=None) -> dict:
    """
    The vendor detector specification, or {} when the datasheet is not present.

    Looked up as <geo_dir>/detector_spec.json, defaulting to the repository's
    geometry/ folder, and expected to hold `sad_mm`, `sdd_mm` and `n_channels`.
    Returning {} rather than raising is deliberate: the file is gitignored along
    with the rest of geometry/, so a fresh clone legitimately lacks it, and the
    absence of a reference value is a reason to report an unverified check, never
    a reason to stop a reconstruction that is otherwise sound.
    """
    p = Path(geo_dir) if geo_dir else Path(__file__).resolve().parent.parent / "geometry"
    p = p / _SPEC_FILENAME
    try:
        spec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: spec[k] for k in _SPEC_KEYS if k in spec}


def check_geometry(geom: dict, geo_dir=None) -> None:
    """
    Hard-fail checks on the geometry dict produced by build_geom().

    Raises InvariantError if any critical geometry parameter deviates from the
    vendor specification, or from what a helical scan can physically be.  The
    comparisons against the specification need geometry/detector_spec.json; when
    it is missing they are soft-warned as unverified and the self-consistency
    checks below — projections per rotation, rotation count, pitch against the
    measured collimation width — still run, since those follow from the scan
    itself and need no datasheet.
    """
    sad = geom['SAD']
    sdd = geom['SDD']
    n_ch = len(geom['channel_betas'])
    P    = geom['proj_per_rotation']
    n_total = geom['n_total_proj']
    pitch = geom['pitch_mm']
    row_z = geom['row_z_mm']

    errors = []
    spec = load_detector_spec(geo_dir)
    if not spec:
        # ASCII only: this reaches a console that may be cp1252, where one stray
        # non-ASCII character aborts the whole message.
        _warn("geometry_spec",
              f"geometry/{_SPEC_FILENAME} not found - SAD, SDD and channel count "
              f"were NOT checked against the vendor specification "
              f"(measured: SAD={sad:.2f} mm, SDD={sdd:.2f} mm, {n_ch} channels)")

    if "sad_mm" in spec and abs(sad - spec["sad_mm"]) > _SAD_TOL_MM:
        errors.append(
            f"SAD={sad:.2f} mm differs from expected {spec['sad_mm']} mm "
            f"by {abs(sad - spec['sad_mm']):.2f} mm (tol={_SAD_TOL_MM} mm)"
        )

    if "sdd_mm" in spec and abs(sdd - spec["sdd_mm"]) > _SDD_TOL_MM:
        errors.append(
            f"SDD={sdd:.2f} mm differs from expected {spec['sdd_mm']} mm "
            f"by {abs(sdd - spec['sdd_mm']):.2f} mm (tol={_SDD_TOL_MM} mm)"
        )

    if "n_channels" in spec and n_ch != spec["n_channels"]:
        errors.append(
            f"channel count={n_ch}, expected {spec['n_channels']} for this geometry"
        )

    if P < 800:
        errors.append(f"proj_per_rotation={P} seems too low (expect >800)")

    n_rot = n_total / P if P > 0 else 0
    if n_rot < 5:
        errors.append(
            f"n_total_proj={n_total} / P={P} = {n_rot:.1f} rotations; "
            f"expect at least 5 for a helical scan"
        )

    # Pitch plausibility: |pitch| < 2 × total collimation width
    n_rows = len(row_z)
    z_span = abs(row_z.max() - row_z.min())
    max_pitch = 2.0 * z_span
    if abs(pitch) > max_pitch:
        errors.append(
            f"pitch={pitch:+.3f} mm exceeds 2× z-span ({max_pitch:.2f} mm); "
            f"likely a table-position unit error"
        )

    if abs(pitch) < 1e-4:
        errors.append(
            f"pitch={pitch:+.6f} mm is near zero — axial scan or unit issue"
        )

    # row_z_mm must straddle z=0 (already checked in build_geom, but belt+braces)
    lo, hi = float(row_z.min()), float(row_z.max())
    if not (lo <= 0.0 <= hi):
        errors.append(
            f"row_z_mm [{lo:+.3f}, {hi:+.3f}] mm does not straddle z=0 — "
            f"SSR interpolation would clip"
        )

    if errors:
        msg = "Geometry invariant(s) failed:\n" + "\n".join(f"  • {e}" for e in errors)
        raise InvariantError(msg)

    print(f"[invariants] Geometry OK  "
          f"(SAD={sad:.0f} SDD={sdd:.0f} ch={n_ch} P={P} pitch={pitch:+.3f}mm)")


# ─────────────────────────────────────────────────────────────────────────────
# C2 — Threshold ordering (soft-warn)
# ─────────────────────────────────────────────────────────────────────────────

def check_threshold_ordering(sino_A: np.ndarray, sino_D: np.ndarray) -> None:
    """
    Verify that threshold A sinogram has higher peak signal than D.

    Through-air rays carry the maximum signal (least attenuation), and threshold
    A counts all photons above the lowest threshold while D counts only the
    hardest photons.  Therefore max(A) should always exceed max(D).

    NOTE: medians are NOT used here.  Per-threshold gain calibration and
    energy-dependent dynamic-range geometry can make median(A) < median(D)
    in legitimately correct data, because the bulk of values lives in the
    threshold-dependent attenuated range, not at the through-air peak.
    Compare with 99th percentile as a noise-robust signal proxy.
    """
    max_A = float(np.max(sino_A))
    max_D = float(np.max(sino_D))
    p99_A = float(np.percentile(sino_A, 99))
    p99_D = float(np.percentile(sino_D, 99))
    print(f"[invariants] Threshold ordering: max(A)={max_A:.1f}  max(D)={max_D:.1f}  "
          f"p99(A)={p99_A:.1f}  p99(D)={p99_D:.1f}")
    if max_A <= max_D:
        _warn(
            "check_threshold_ordering",
            "max(A) <= max(D) — thresholds may be swapped at load time",
            {"max_A": round(max_A, 2), "max_D": round(max_D, 2)},
        )
    elif p99_A < p99_D * 0.95:
        _warn(
            "check_threshold_ordering",
            "p99(A) significantly < p99(D) — unusual signal distribution, "
            "check for partial threshold swap or load error",
            {"p99_A": round(p99_A, 2), "p99_D": round(p99_D, 2)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# C3 — Defect mask (soft-warn)
# ─────────────────────────────────────────────────────────────────────────────

def check_defect_mask(mask: np.ndarray, n_channels: int) -> None:
    """
    Verify that the defect mask flags a plausible number of channels.

    An empty mask (0 flags) means the spike detector found nothing — likely
    a threshold-parameter problem and spikes are still in the data.
    A mask flagging >5% of channels is likely over-aggressive.
    """
    n_flagged = int(mask.sum())
    frac = n_flagged / max(n_channels, 1)
    print(f"[invariants] Defect mask: {n_flagged}/{n_channels} channels ({100*frac:.1f}%)")

    if n_flagged == 0:
        _warn(
            "check_defect_mask",
            "No defective channels found — spike detector may have missed them. "
            "Review spike_detection diagnostics.",
            {"n_flagged": 0, "n_channels": n_channels},
        )
    elif frac > 0.15:
        # Threshold is 15% (not 5%) because PCCT scans with metal objects
        # legitimately have many inter-module gap spike channels. ±1 dilation
        # keeps total masking reasonable; above 15% is still a sign that the
        # dilation is bridging or that a whole detector module is missing.
        _warn(
            "check_defect_mask",
            f"{100*frac:.1f}% of channels flagged — detection may be too aggressive "
            f"or a whole detector module may be absent",
            {"n_flagged": n_flagged, "n_channels": n_channels, "frac": round(frac, 4)},
        )

    # Largest contiguous run (after dilation channels tend to cluster)
    changes = np.diff(mask.astype(int))
    starts  = np.where(changes == 1)[0] + 1
    ends    = np.where(changes == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [n_channels]])
    if len(starts) > 0 and len(ends) > 0:
        longest_run = int(np.max(ends - starts))
        if longest_run > 30:
            _warn(
                "check_defect_mask",
                f"Largest contiguous defect cluster = {longest_run} channels; "
                f"> 30 suggests a whole detector module is missing",
                {"longest_run": longest_run},
            )


# ─────────────────────────────────────────────────────────────────────────────
# C4 — Sinogram after preprocessing (soft-warn)
# ─────────────────────────────────────────────────────────────────────────────

def check_sinogram_preprocessed(sino: np.ndarray, label: str = "") -> None:
    """
    Verify the preprocessed 2-D axial sinogram [n_proj, n_channels].

    Checks:
    • No negative values survived the non-negative clip.
    • At least 90% of values are finite.
    • Dynamic range is plausible (not all-zero, not wildly clipped).
    """
    tag = f"check_sinogram_preprocessed({label})"
    if sino.min() < -1e-3:
        _warn(tag, "Negative values after preprocessing — non-negative clip may not have run",
              {"min": float(sino.min())})

    nan_frac = float(np.isnan(sino).mean())
    if nan_frac > 0.001:
        _warn(tag, f"{100*nan_frac:.2f}% NaN values in sinogram",
              {"nan_frac": round(nan_frac, 5)})

    smax = float(sino.max())
    if smax < 1e-6:
        _warn(tag, "Sinogram is all-zero after preprocessing — likely a load or sign error",
              {"max": smax})

    p99 = float(np.percentile(sino, 99))
    p01 = float(np.percentile(sino, 1))
    if p01 > 0.5 * p99:
        _warn(tag,
              "Air regions may not be reaching 0 — baseline subtraction may have failed",
              {"p01": round(p01, 3), "p99": round(p99, 3)})

    print(f"[invariants] Sinogram({label}) OK  "
          f"min={sino.min():.3f}  p99={p99:.3f}  max={smax:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# C5 — Reconstruction slice (soft-warn, hard-fail on NaN/infinite)
# ─────────────────────────────────────────────────────────────────────────────

def check_reconstruction(
    img: np.ndarray,
    geom: dict,
    label: str = "",
    fov_mm: float = 500.0,
    water_hu_tol: float = 10.0,
    water_hu_mean: float | None = None,
) -> None:
    """
    Validate a single reconstructed slice [n_pixels, n_pixels].

    Hard checks (raise InvariantError):
    • No NaN or infinite values.

    Soft checks (log warning):
    • Centre-of-mass within 5% of FOV from image centre (catches COR / geometry
      sign errors; applies only when a non-trivial fraction of pixels is active).
    • If water_hu_mean is provided (from HU calibration): mean in central ROI
      within ±water_hu_tol HU of 0.
    """
    tag = f"check_reconstruction({label})"
    n = img.shape[0]

    # ── Hard: NaN / inf ──────────────────────────────────────────────
    if not np.isfinite(img).all():
        n_bad = int(~np.isfinite(img).sum())
        raise InvariantError(
            f"{tag}: {n_bad} non-finite pixel(s) in reconstruction — "
            f"geometry error or ASTRA failure"
        )

    # ── Soft: centre-of-mass ─────────────────────────────────────────
    # Tolerance is 10% of FOV (50 mm at FOV=500 mm).  Phantoms placed
    # casually on the patient table typically sit 20-40 mm from isocentre,
    # which is fine.  >50 mm offset suggests a COR error or geometry sign bug.
    threshold = float(np.percentile(img[img > 0], 50)) if (img > 0).any() else None
    if threshold is not None:
        active = img > threshold
        if active.sum() > 0.01 * img.size:
            ys, xs = np.where(active)
            cx = float(xs.mean()) - n / 2
            cy = float(ys.mean()) - n / 2
            px_per_mm = n / fov_mm
            cx_mm = cx / px_per_mm
            cy_mm = cy / px_per_mm
            tol_mm = 0.10 * fov_mm
            if abs(cx_mm) > tol_mm or abs(cy_mm) > tol_mm:
                _warn(tag,
                      f"Centre-of-mass offset ({cx_mm:+.1f}, {cy_mm:+.1f}) mm "
                      f"exceeds 10% of FOV ({tol_mm:.0f} mm) — possible COR or geometry error",
                      {"cx_mm": round(cx_mm, 2), "cy_mm": round(cy_mm, 2),
                       "tol_mm": round(tol_mm, 1)})
            else:
                print(f"[invariants] Reconstruction({label}) COM at ({cx_mm:+.1f}, {cy_mm:+.1f}) mm "
                      f"(within tolerance)")

    # ── Soft: water HU in central ROI ───────────────────────────────
    if water_hu_mean is not None:
        roi_r = max(1, int(0.05 * n))   # central 10% of FOV square
        centre = n // 2
        roi = img[centre - roi_r: centre + roi_r, centre - roi_r: centre + roi_r]
        mu = float(roi.mean())
        if abs(mu - water_hu_mean) > water_hu_tol:
            _warn(tag,
                  f"Central ROI mean HU={mu:.1f} differs from water ({water_hu_mean:.1f} HU) "
                  f"by {abs(mu - water_hu_mean):.1f} HU (tol={water_hu_tol:.0f} HU)",
                  {"roi_mean_hu": round(mu, 2), "water_hu_mean": round(water_hu_mean, 2)})

    v_lo = float(np.percentile(img, 1))
    v_hi = float(np.percentile(img, 99))
    print(f"[invariants] Reconstruction({label}) OK  "
          f"p1={v_lo:.2f}  p99={v_hi:.2f}  shape={img.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# C6 — Slice-to-slice continuity (soft-warn, full-mode runs)
# ─────────────────────────────────────────────────────────────────────────────

def check_slice_continuity(
    slices: list[np.ndarray],
    label: str = "",
    tol: float = 5.0,
) -> None:
    """
    Check that adjacent reconstructed slices have consistent body-region means.

    Call with a list of 3 consecutive slices.  Uses an Otsu-segmented body
    mask intersected across all slices to define a region known to contain
    phantom/body voxels (not air).  This is robust to off-centre phantoms
    where the geometric image centre falls in air.

    If the within-body mean varies by more than `tol` (in whatever units the
    slices are in — HU or raw attenuation) between adjacent slices, the SSR
    row interpolation or z-window may have a bug.
    """
    if len(slices) < 2:
        return
    tag = f"check_slice_continuity({label})"

    # Compute Otsu threshold once on stacked data
    stack = np.stack(slices, axis=0)
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return
    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    bins = 256
    counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    total = counts.sum()
    cum   = np.cumsum(counts)
    cumw  = np.cumsum(counts * (edges[:-1] + edges[1:]) * 0.5)
    mu_t  = cumw[-1] / (total + 1e-9)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma2_b = np.where(
            (cum > 0) & (cum < total),
            (mu_t * cum - cumw) ** 2 / (cum * (total - cum) + 1e-9),
            0.0,
        )
    otsu = float(edges[np.argmax(sigma2_b)])

    # Intersect body masks across slices — region present in all three
    body_each = [(s > otsu) for s in slices]
    body_common = body_each[0]
    for b in body_each[1:]:
        body_common = body_common & b

    if body_common.sum() < 100:
        # Off-centre or non-overlapping body — fall back to per-slice mask mean
        means = [float(s[b].mean()) if b.any() else float('nan')
                 for s, b in zip(slices, body_each)]
    else:
        means = [float(s[body_common].mean()) for s in slices]

    if any(not np.isfinite(m) for m in means):
        return

    diffs = [abs(means[i+1] - means[i]) for i in range(len(means) - 1)]
    if max(diffs) > tol:
        _warn(tag,
              f"Slice-to-slice in-body mean jump of {max(diffs):.2f} exceeds tol={tol} — "
              f"check SSR row mapping / z-window",
              {"in_body_means": [round(m, 2) for m in means],
               "max_diff": round(max(diffs), 2)})
    else:
        print(f"[invariants] Slice continuity({label}) OK  "
              f"in-body means {[round(m,2) for m in means]}")


# ─────────────────────────────────────────────────────────────────────────────
# C7 — Per-threshold cross-check (soft-warn)
# ─────────────────────────────────────────────────────────────────────────────

def check_cross_threshold(
    mu_water_raw: dict[str, float],
    tol_frac: float = 0.10,
) -> None:
    """
    Check that the auto-detected mu_water values (in raw scanner units) are
    consistent across the four thresholds.

    mu_water_raw: {'A': mu_water_A, 'B': ..., 'C': ..., 'D': ...}
                  Raw-unit water reference values from auto_hu_calibrate().

    Why mu_water (raw) and not central-ROI HU
    ─────────────────────────────────────────
    After HU calibration, water voxels have HU≈0 by construction (we calibrated
    against them).  So checking HU values of a central ROI tells us nothing
    about the *calibration*; it tells us whether the central ROI is actually
    water.  For off-centre phantoms, the central ROI is air, giving HU≈-1000
    even though the calibration was correct.

    The right cross-threshold check is on the raw mu_water values themselves:
    these come from real water voxels (Otsu+erode segmentation), and should
    vary smoothly across thresholds (within ~10% of the mean) because the
    physical water attenuation differs only slightly across the bins.  Large
    spread indicates a per-threshold calibration drift.
    """
    tag = "check_cross_threshold"
    labels = ['A', 'B', 'C', 'D']
    present = [l for l in labels if l in mu_water_raw]
    if len(present) < 2:
        return

    values = np.array([mu_water_raw[l] for l in present], dtype=float)
    mean = float(values.mean())
    spread = float(values.max() - values.min())
    rel_spread = spread / max(abs(mean), 1e-6)

    print(f"[invariants] Cross-threshold mu_water (raw): "
          + "  ".join(f"{l}={v:.2f}" for l, v in zip(present, values))
          + f"  spread={spread:.2f} ({100*rel_spread:.1f}%)")

    if rel_spread > tol_frac:
        _warn(tag,
              f"mu_water spread {100*rel_spread:.1f}% across thresholds exceeds "
              f"{100*tol_frac:.0f}% — per-threshold calibration drift?",
              {"mu_water_raw": {l: round(v, 3) for l, v in mu_water_raw.items()},
               "spread": round(spread, 3),
               "rel_spread": round(rel_spread, 4)})


# ─────────────────────────────────────────────────────────────────────────────
# C8 — Output format (hard-fail)
# ─────────────────────────────────────────────────────────────────────────────

def check_output_format(
    vol: np.ndarray,
    geom: dict,
    z_targets: np.ndarray,
    n_pixels: int,
) -> None:
    """
    Hard-fail checks before a NIfTI write.

    • Volume dtype must be float32.
    • Volume shape must be (n_slices, n_pixels, n_pixels).
    • No NaN or infinite values.
    • Slice count must match len(z_targets).
    """
    tag = "check_output_format"
    errors = []

    if vol.dtype != np.float32:
        errors.append(f"dtype={vol.dtype}, expected float32")

    expected_shape = (len(z_targets), n_pixels, n_pixels)
    if vol.shape != expected_shape:
        errors.append(f"shape={vol.shape}, expected {expected_shape}")

    if not np.isfinite(vol).all():
        n_bad = int(~np.isfinite(vol).sum())
        errors.append(f"{n_bad} non-finite voxel(s) in volume")

    if errors:
        raise InvariantError(
            tag + " — output format check(s) failed:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    print(f"[invariants] Output format OK  shape={vol.shape}  dtype={vol.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# Ring-index helper (used by adaptive wavelet gating in helical_reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ring_index(img: np.ndarray) -> float:
    """
    Compute a ring-artifact index for a 2-D reconstruction.

    Converts to polar coordinates and measures the ratio of the along-angle
    standard deviation (ring-like variation) to the overall image noise.

    Returns a dimensionless index:
      ~1.0  →  no ring structure (angular variation ≈ radial variation)
      >2.0  →  noticeable rings
      >5.0  →  strong rings
    """
    from scipy.ndimage import map_coordinates

    n = img.shape[0]
    centre = n / 2
    r_max  = min(centre, n / 2) * 0.9     # 90% of inscribed circle

    n_r     = int(r_max)
    n_theta = 360
    r_vals  = np.linspace(1, r_max, n_r)
    t_vals  = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    rr, tt = np.meshgrid(r_vals, t_vals, indexing='ij')
    xs = centre + rr * np.cos(tt)
    ys = centre + rr * np.sin(tt)
    coords = np.array([ys.ravel(), xs.ravel()])
    polar  = map_coordinates(img.astype(np.float64), coords, order=1,
                              mode='nearest').reshape(n_r, n_theta)

    # Ring index: std along theta at each r, then median over r / global std
    ring_std_per_r = polar.std(axis=1)     # (n_r,) — angular std at each radius
    global_std     = float(img.std())
    if global_std < 1e-9:
        return 1.0
    ring_index = float(np.median(ring_std_per_r)) / global_std
    return ring_index


# ─────────────────────────────────────────────────────────────────────────────
# C9 — Reconstruction orientation (soft-warn).  Purely diagnostic — NEVER
#       modifies the image.  Verifies that the fixed dense structure (patient
#       table / phantom holder) lands at the BOTTOM of the displayed image, i.e.
#       that the descriptor-derived view angle + Siemens→ASTRA convention
#       constants produced the correct rotation.  If it is off, the exact
#       residual is logged so _VIEW_ANGLE_OFFSET_DEG can be set once.
# ─────────────────────────────────────────────────────────────────────────────

def check_orientation(
    img: np.ndarray,
    geom: dict,
    label: str = "",
    tol_deg: float = 35.0,
) -> None:
    """
    Soft check on absolute orientation.

    The patient table / phantom holder is a dense structure physically fixed at
    the BOTTOM of the gantry.  We locate the brightest peripheral structure and
    report the angle at which it actually lands.  Display convention (matplotlib
    imshow origin='upper'): angle from +x (right) with +y pointing DOWN, so
        bottom = +90°,  right = 0°,  top = −90°,  left = 180°.

    This check is metadata-independent verification: orientation correctness
    comes from the descriptor-derived view angle in helical_reconstruction.py
    (build_geom + _VIEW_ANGLE_SIGN/_VIEW_ANGLE_OFFSET_DEG).  This routine only
    measures the result and logs the residual; it never rotates the image.
    """
    tag = f"check_orientation({label})"
    if img.ndim != 2:
        return
    n = img.shape[0]
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return

    # Brightest structures (the table/holder is among the most attenuating).
    hi = float(np.percentile(finite, 99.0))
    ys, xs = np.where(img >= hi)
    if ys.size < 20:
        print(f"[invariants] Orientation({label}) skipped — no distinct dense structure")
        return

    cy = cx = n / 2.0
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    # Keep only peripheral bright pixels so central inserts don't dominate.
    peripheral = r > 0.45 * (n / 2.0)
    if int(peripheral.sum()) < 20:
        print(f"[invariants] Orientation({label}) skipped — "
              f"no peripheral dense structure (table not clearly imaged)")
        return

    # Angular position of each peripheral bright pixel (display frame: +y down,
    # so bottom = +90°).  Use the circular mean, and require the pixels to be
    # angularly CONCENTRATED (resultant length R̄) — otherwise the "structure"
    # is just scattered noise and we should not judge orientation from it.
    ang  = np.arctan2(ys[peripheral] - cy, xs[peripheral] - cx)
    c, s = float(np.cos(ang).mean()), float(np.sin(ang).mean())
    rbar = float(np.hypot(c, s))                                 # 0=dispersed, 1=tight
    if rbar < 0.45:
        print(f"[invariants] Orientation({label}) skipped — peripheral bright "
              f"pixels not angularly coherent (R={rbar:.2f}); no clear table")
        return

    disp_angle = float(np.degrees(np.arctan2(s, c)))             # bottom = +90°
    dev = ((disp_angle - 90.0 + 180.0) % 360.0) - 180.0          # signed dev. from bottom

    if abs(dev) > tol_deg:
        _warn(tag,
              f"Dense peripheral structure at {disp_angle:+.1f}° (expected +90°, "
              f"bottom) — reconstruction appears rotated by ~{-dev:+.1f}°. "
              f"Set _VIEW_ANGLE_OFFSET_DEG in helical_reconstruction.py by "
              f"±{abs(dev):.1f}° (pick the sign that brings this toward +90°).",
              {"structure_angle_deg": round(disp_angle, 1),
               "expected_deg": 90.0,
               "deviation_deg": round(dev, 1)})
    else:
        print(f"[invariants] Orientation({label}) OK  "
              f"dense structure at {disp_angle:+.1f} deg (~bottom)")


# ─────────────────────────────────────────────────────────────────────────────
# C10 — Angular balance of the 'balanced' helical weighting (soft-warn).
#        The fix for the rotating low-frequency lobe relies on every view angle
#        receiving the SAME total weight (sum == 1).  A non-uniform sum means the
#        rebinning reintroduced an angular bias (which would rotate with z), so we
#        log the residual.  Only meaningful for z_weighting='balanced'.
# ─────────────────────────────────────────────────────────────────────────────

def check_angular_balance(
    weight_sums: np.ndarray,
    label: str = "",
    tol: float = 1e-3,
) -> None:
    """
    Soft check that the 'balanced' helical weighting is angularly uniform.

    `weight_sums` is the per-view-angle realised weight sum returned by
    rebin_helical_to_axial(..., z_weighting='balanced', return_weights=True).
    By construction it should be 1.0 at every angle (uniform angular weighting →
    no rotating shading).  Warns if any angle deviates by more than `tol`.
    """
    tag = f"check_angular_balance({label})"
    w = np.asarray(weight_sums, dtype=float).ravel()
    if w.size == 0:
        return
    max_dev = float(np.max(np.abs(w - 1.0)))
    n_off   = int(np.sum(np.abs(w - 1.0) > tol))
    if max_dev > tol:
        _warn(tag,
              f"[{label}] per-view-angle weight not uniform: max|w-1|={max_dev:.2e} "
              f"over {n_off}/{w.size} angles (expected ~1.0). The balanced "
              f"weighting did not fully cancel the angular bias — the rotating "
              f"low-frequency lobe may persist.",
              {"max_dev": round(max_dev, 6),
               "n_off_angles": n_off,
               "n_angles": int(w.size)})
    else:
        print(f"[invariants] AngularBalance({label}) OK  "
              f"max|w-1|={max_dev:.1e} over {w.size} angles")
