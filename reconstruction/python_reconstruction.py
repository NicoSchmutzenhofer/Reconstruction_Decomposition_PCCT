"""
python_reconstruction.py -- reconstruction driver: 4 threshold sinograms in, 4 volumes out.

Loads the HDF5 sinogram and the MATLAB descriptor, reconstructs each energy threshold on its
own with helical_reconstruction (vol_A = lowest threshold ... vol_D = highest) and writes the
NIfTIs.  Threshold sinograms are never subtracted (invariant #3, README); energy-bin images,
where needed, are formed from the reconstructed volumes.

All settings are the constants at the top of this file.  The committed value is often tuned
away from the default its comment describes, so read the value, not the comment.

The HDF5 stores the thresholds in reverse (invariant #1, README): _load_threshold maps
logical index i to physical index 3 - i, and any new access to the file must do the same.

FAST_MODE = True reconstructs a slab preview of all four thresholds, seconds each, for
tuning.  FAST_MODE = False reconstructs the full helical volumes (hours) and writes a
raw-attenuation and an HU-calibrated NIfTI per threshold plus a 4-D stack.  One threshold
sinogram (~16 GB) is held at a time; peak memory is about 19 GB.
"""

import gc
import numpy as np
import h5py
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
import sys

import json

from helical_reconstruction import (
    build_geom,
    detect_defect_channels,
    rebin_helical_to_axial,
    preprocess_sinogram,
    apply_cor_shift,
    reconstruct_helical_stack,
    reconstruct_slab,
    z_targets_for_full_scan,
    _astra_fbp,
    apply_mar,
    auto_hu_calibrate,
    apply_hu_calibration,
    to_output_frame,
    reset_wavelet_stats,
    report_wavelet_stats,
    z_average,
)
from recon_invariants import (
    check_geometry,
    check_defect_mask,
    check_threshold_ordering,
    check_sinogram_preprocessed,
    check_reconstruction,
    check_orientation,
    check_angular_balance,
    check_slice_continuity,
    check_cross_threshold,
    check_output_format,
    flush_invariant_log,
)

sys.stdout.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
FAST_MODE = False           # True = single-slice preview only, False = full volumes
N_PIXELS  = 512           # reconstruction grid size
FOV_MM    = 500.0         # field of view in mm
slice_idx = 250           # example index (phantom with the inserts)

# Metal Artifact Reduction
# ────────────────────────
# MAR_STRENGTH controls the blend between the original and the interpolated
# (streak-free) sinogram in metal-contaminated projection bins:
#
#   0.0        — off (no MAR), recommended first pass and for initial tuning
#   0.3 – 0.5  — soft: streaks noticeably reduced, metal attenuation values
#                preserved at ~60-70%; good for material decomposition
#   1.0        — full replacement: maximum streak suppression but metal signal
#                in the sinogram is fully interpolated away; metal voxel values
#                in the reconstruction will be underestimated
#
# MAR_METAL_THRESHOLD: image-domain value above which voxels are classified
# as metal in the first-pass FBP.  None = auto-detect from histogram valley.
# Set manually if auto-detection misses thin metal objects or flags too much.
#
# In FAST_MODE, a before/after single-slice comparison is shown when MAR is on.
# In FULL mode, MAR roughly doubles reconstruction time (one extra FBP/slice).
MAR_STRENGTH        = 0.0   # 0.0 = off
MAR_METAL_THRESHOLD = None  # None = auto

# Hampel filter
# ─────────────
# Per-projection spike suppression.  DEFAULT IS OFF (None).
#
# WHY off: inside a water phantom the sinogram profile is smooth so the
# local MAD is small (~50–100 scanner units).  A bone-equivalent insert
# creates a narrow peak of ~300–500 above background.  At threshold=7.0,
# 7 × 80 = 560 ≥ 400 → the insert gets replaced with the water median,
# compressing bone from ~200 HU to ~40–50 HU.
#
# The hard-defect channel mask handles actual detector spikes.
# Only enable if residual rings are still visible after reviewing the mask:
#   None   — disabled (preserves correct HU for all tissue/insert types)
#   20–30  — very light backup, only catches extreme point spikes
HAMPEL_THRESHOLD = None  # None = off (recommended)

# FBP filter
# ──────────
# 'shepp-logan' (default) — smooth quantitative kernel, ~Siemens Qr40.
#   Good for low-contrast insert visibility and material decomposition inputs.
# 'ram-lak'               — full-bandwidth ramp; use only for MTF / geometry
#   verification (30–60% more noise than shepp-logan).
# Other ASTRA options: 'cosine', 'hamming', 'hann', etc.
FILTER_NAME = 'hann'

# Detector geometry model
# ───────────────────────
# 'curved' (default) — cosine pre-weight + cubic remap from equiangular
#   (the physical curved detector) to equispaced flat-detector sampling.
#   Fixes radial position errors and peripheral HU cupping.
# 'flat'   — legacy approximation; use only for A/B comparison.
GEOMETRY_MODEL = 'curved'

# Wavelet stripe removal
# ──────────────────────
# Applied adaptively: only when the sinogram's stripe proxy exceeds a gate.
# WAVELET_RING_THRESHOLD=2.0 means stripe SNR must indicate clear ring structure.
# Set to 0.0 to always apply; 999 to never apply.
WAVELET_RING_THRESHOLD = 2.0

# HU calibration
# ──────────────
# ENABLE_HU_CALIBRATION=True: auto-detect mu_water and mu_air from the
#   reconstructed volume (per threshold), convert output to Hounsfield Units,
#   and write both raw-attenuation and HU-scaled NIfTIs.
# Calibration parameters are cached in output/reconstruction/calibration_thr_<label>.json.
ENABLE_HU_CALIBRATION = True

# Z-smoothing (post-recon noise reduction)
# ────────────────────────────────────────
# Gaussian smoothing along the slice axis to boost SNR for low-contrast
# insert visibility.  Value is the FWHM in mm of the z-Gaussian.
#   0.0 (default) — off, native 0.4 mm slice thickness preserved.
#   1.5 – 3.0     — moderate smoothing, ~2× SNR boost at 1.5–3 mm effective
#                    slice thickness.  Recommended for QRM DE phantom inserts.
#   > 5.0         — aggressive; only use for very low-contrast features.
# Spatial in-slice resolution is unaffected.
Z_SMOOTH_MM = 3.0

# Helical projection-axis weighting  (rotating "light cone" fix)
# ─────────────────────────────────────────────────────────────
# How per-projection rays are weighted when one helical rotation is rebinned
# into an axial slice.  The old raised-cosine z-window ('hann') is, over a
# one-rotation window, an angular apodization whose peak rides the helix — it
# produces a low-frequency brightness lobe that ROTATES as you scroll through z.
#   'balanced' (default) — angularly-balanced helical weighting: suppresses the
#       oblique helix-end rays (no z-shading bands) AND normalises every view
#       angle to equal total weight (no angular bias → nothing rotates).  Purely
#       a sinogram-formation change, so FBP/SIRT/CGLS all benefit identically.
#   'hann'     — previous behaviour (rotating lobe); for A-B comparison only.
#   'none'     — uniform weighting (no rotation, but the z-shading bands return);
#       the diagnostic midpoint between 'hann' and 'balanced'.
Z_WEIGHTING = 'balanced'

# Reconstruction algorithm
# ────────────────────────
# 'fbp'  (default) — filtered back-projection, the established method.  Fast.
#                    Uses FILTER_NAME / GEOMETRY_MODEL above.  Nothing about the
#                    existing pipeline changes while this is selected.
# 'sirt'           — SIRT_CUDA with a non-negativity constraint.  Well-understood,
#                    defensible noise/streak suppression for the photon-starved
#                    high thresholds.  N_ITER trades resolution (more = sharper +
#                    noisier).  ~2·N_ITER × slower than FBP per slice.
# 'cgls'           — CGLS_CUDA.  Faster convergence but semi-convergent.
RECON_METHOD = 'sirt'
N_ITER       = 100        # iterations for 'sirt'/'cgls' (ignored for 'fbp')

# Preview slab thickness
# ──────────────────────
# In FAST mode the preview reconstructs an image-domain average of the slices
# spanning PREVIEW_SLAB_MM (centred on the preview slice), instead of a single
# noisy 0.4 mm slice.  This makes the preview SNR match what the full volume
# will look like at the same effective slice thickness — so tuning decisions
# transfer.  Slice count is derived from geometry (slab / z_spacing).
#   0.0      — single native slice (legacy behaviour)
#   1.5–3.0  — recommended for low-contrast insert visibility (2.0 ≈ 5 slices)
PREVIEW_SLAB_MM = 3.0

# Defect-channel detection sensitivity
# ────────────────────────────────────
# MAD multipliers for detect_defect_channels.  Higher = mask FEWER channels.
# Defaults (5.0 / 6.0) reproduce the previous behaviour.  Raise SPIKE_MAD_K
# (e.g. 7–9) if the cross-threshold diagnostic shows the flagged channels are
# object-induced rather than true (threshold-independent) detector defects.
SPIKE_MAD_K = 5.0
IPR_MAD_K   = 6.0

# Preview HU display window  (level, width) in HU — used when HU calibration is
# on.  All four thresholds are shown on this SAME window so HU similarity /
# separation is judged fairly.  (40, 400) is a standard soft-tissue window.
PREVIEW_HU_WL = (40.0, 400.0)

# Patient position  (orientation labelling)
# ─────────────────────────────────────────
# DICOM PatientPosition — how the patient/phantom sits in the bore.  The raw
# .mat descriptor does NOT carry this field, so it is supplied here.  It does
# NOT affect the gantry-frame reconstruction (the table is always at the bottom
# of the bore); it only fixes the array→patient (LPS) axis labelling — which
# side is the patient's Left/Right, Anterior/Posterior, Head/Foot.
#
# Accepted DICOM defined terms:
#   'HFS' Head First-Supine     'FFS' Feet First-Supine
#   'HFP' Head First-Prone      'FFP' Feet First-Prone
#   'HFDR'/'HFDL'/'FFDR'/'FFDL' decubitus (on-side) variants
#
# Only 'HFS' is implemented AND validated against the Siemens reference.  The
# other axial positions follow from HFS by a fixed flip table (relative to HFS):
#   FFS : flip Left-Right,         flip Head-Foot        (A-P unchanged)
#   HFP : flip Left-Right,         flip Anterior-Posterior (Head-Foot unchanged)
#   FFP : flip Anterior-Posterior, flip Head-Foot        (Left-Right unchanged)
#   (decubitus = 90° in-plane rotation, direction set by Right vs Left)
# When a non-HFS scan first appears, derive its direction matrix from this table,
# validate it once against the reference, then wire it in below.  Until then the
# pipeline ERRORS OUT rather than silently mislabelling orientation.
# Output frame  (match the vendor, so nothing downstream has to patch it)
# ─────────────────────────────────────────────────────────────────────
# Applied on WRITE only -- the reconstruction and the detector channels are untouched
# (Invariant #2, README: a residual mirror is fixed here, never by
# un-flipping channels, which would break the helical handedness).
#
# OUTPUT_MIRROR_X: our volumes come out mirrored left-right relative to Siemens.
#   Measured on this scanner: the body centroid and the phantom's principal axis agree
#   with the vendor only after x -> -x (0.30 mm and 0.10 deg residual).  See
#   the frame work of the reconstruction comparison.
#
# TABLE_HEIGHT_MM: Siemens references the patient frame to the TABLE, not the isocentre.
#   Measured 88.5 mm here, exactly the DICOM TableHeight, and it is a property of the
#   scan setup rather than of the scanner -- read it off the vendor export
#   (`python -m reconstruction.probe_dicom_origin`) for a new acquisition.
#   0.0 keeps our own isocentre-centred frame.
#
# z ORDER is corrected unconditionally inside to_output_frame: on a scan whose table
# travels toward decreasing z, slice 0 is the HIGHEST z, and writing that with a positive
# spacing claimed the opposite -- an mm->slice lookup then landed ~240 slices out.
OUTPUT_MIRROR_X = True
TABLE_HEIGHT_MM = 88.5

PATIENT_POSITION = 'HFS'

_KNOWN_PATIENT_POSITIONS = {'HFS', 'FFS', 'HFP', 'FFP',
                            'HFDR', 'HFDL', 'FFDR', 'FFDL'}
if PATIENT_POSITION not in _KNOWN_PATIENT_POSITIONS:
    raise ValueError(
        f"PATIENT_POSITION={PATIENT_POSITION!r} is not a recognised DICOM "
        f"PatientPosition. Expected one of {sorted(_KNOWN_PATIENT_POSITIONS)}.")
if PATIENT_POSITION != 'HFS':
    raise NotImplementedError(
        f"PATIENT_POSITION={PATIENT_POSITION!r}: only 'HFS' is implemented and "
        f"validated. Derive the array→LPS direction matrix from HFS via the flip "
        f"table documented above, validate it once against the reference, then "
        f"wire it in here.")

# Raw scan: the HDF5 threshold sinogram and its MATLAB descriptor.  Fill in your own files.
data_path = Path(r"/path/to/full_sinogram.raw.mat")
desc_path = Path(r"/path/to/descriptor.raw.mat")
# Repo root = parent of this reconstruction/ folder; reconstruction outputs go to
# repo-root/output/reconstruction regardless of the working directory the job is launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
out_dir   = _REPO_ROOT / "output" / "reconstruction"
out_dir.mkdir(parents=True, exist_ok=True)

THR_LABELS = {0: 'A  (T1, all-E)', 1: 'B  (T2)',
              2: 'C  (T3)',        3: 'D  (T4, hard)'}

if FAST_MODE:
    mode_str = (f"FAST (slab {PREVIEW_SLAB_MM:.1f} mm)"
                if PREVIEW_SLAB_MM > 0 else "FAST (single-slice)")
else:
    mode_str = "FULL (complete volumes)"
mar_str    = f"MAR={MAR_STRENGTH}" if MAR_STRENGTH > 0 else "no MAR"
recon_str  = (f"{RECON_METHOD.upper()} x{N_ITER}"
              if RECON_METHOD != 'fbp' else "FBP")
print(f"=== 4-bin PCCT Reconstruction — NAEOTOM Alpha  "
      f"[{mode_str}  {recon_str}  {mar_str}] ===\n")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _load_threshold(f, logical_idx):
    """
    Load threshold logical_idx (0=A … 3=D) from an open HDF5 file.
    Physical storage is reversed: physical_idx = 3 - logical_idx.
    Returns float32 [N_proj, n_rows, n_channels] with channels flipped.

    Channels ARE flipped (data[:, :, ::-1]); build_geom is called with
    channels_flipped=True.  This flip is REQUIRED: in native channel order the
    full volume develops a z-direction helix (the object spirals along the table
    axis), confirmed by a full-volume A/B test.  In fan-beam a detector-channel
    flip is equivalent to the gantry rotation sense, so the flip sets the correct
    helical handedness.  Do not remove it -- see invariant #2 in the README.
    """
    ref  = f['data_full']['A'][3 - logical_idx, 0]
    data = f[ref][:]
    return data[:, :, ::-1].astype(np.float32)


def _sitk_vol(arr, xy_spacing, z_spacing, origin):
    """Wrap a numpy volume in a SimpleITK image with correct metadata."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((xy_spacing, xy_spacing, abs(z_spacing)))
    img.SetOrigin(origin)
    img.SetDirection((1, 0, 0,  0, 1, 0,  0, 0, 1))
    return img



# ─────────────────────────────────────────────────────────────────────
# 1. Geometry
# ─────────────────────────────────────────────────────────────────────
print("Building geometry from descriptor...")
desc_data  = sio.loadmat(str(desc_path), struct_as_record=True, squeeze_me=False)
descriptor = desc_data['descriptor'].flat[0]
geom = build_geom(descriptor, geo_dir=_REPO_ROOT / "geometry", channels_flipped=True)
del desc_data

print(f"  SAD={geom['SAD']:.1f} mm  SDD={geom['SDD']:.1f} mm")
print(f"  Channels={len(geom['channel_betas'])}  n_air={geom['n_air_channels']}")
print(f"  Rows={len(geom['row_z_mm'])}  "
      f"z=[{geom['row_z_mm'].min():+.2f}, {geom['row_z_mm'].max():+.2f}] mm")
print(f"  slice_width={geom['slice_width_mm']:.4f} mm  "
      f"pitch={geom['pitch_mm']:+.3f} mm/rot")

# ── C1: Geometry invariants (hard-fail) ──────────────────────────────
check_geometry(geom)
print(f"  det_alignment={geom['det_alignment']:+.4f} ch")

# 'balanced' rebinning spans 2 rotations, so trim a full rotation at each end
# (n_rot_balanced/2 = 1.0); the one-rotation modes only need half a rotation.
_end_margin_rot = 1.0 if Z_WEIGHTING == 'balanced' else 0.5
z_targets, z_spacing = z_targets_for_full_scan(
    geom, oversample=1, end_margin_rotations=_end_margin_rot)
z_centre = float(z_targets[slice_idx])
xy_spacing = FOV_MM / N_PIXELS
origin     = (-FOV_MM/2 + xy_spacing/2, -FOV_MM/2 + xy_spacing/2, float(z_targets[0]))

print(f"\nReconstructable range: z={z_targets[0]:+.2f} -> {z_targets[-1]:+.2f} mm  "
      f"({len(z_targets)} slices  dz={z_spacing:.4f} mm)")
print(f"Preview slice: z={z_centre:+.2f} mm")

# Verify HDF5 structure
with h5py.File(str(data_path), 'r') as f:
    n_thr  = f['data_full']['A'].shape[0]
    ref0   = f['data_full']['A'][0, 0]
    shape0 = f[ref0].shape
    gb_sino = np.prod(shape0) * 4 / 1e9
    gb_vol  = N_PIXELS * N_PIXELS * len(z_targets) * 4 / 1e9
    print(f"\ndata_full.A: {n_thr} thresholds  |  "
          f"sinogram {shape0} ({gb_sino:.1f} GB each)  |  "
          f"volume ({len(z_targets)},{N_PIXELS},{N_PIXELS}) ({gb_vol:.1f} GB each)")
    if n_thr != 4:
        raise ValueError(f"Expected 4 thresholds, got {n_thr}")

# ─────────────────────────────────────────────────────────────────────
# 2. Defect detection — load threshold A, detect once, reuse for all
# ─────────────────────────────────────────────────────────────────────
print("\n--- Defect channel detection from threshold A (highest SNR) ---")
with h5py.File(str(data_path), 'r') as f:
    sino_A = _load_threshold(f, 0)
print(f"  A: {sino_A.shape}  [{sino_A.min():.1f}, {sino_A.max():.1f}]")

geom['spike_mask'] = detect_defect_channels(
    sino_A, spike_mad_k=SPIKE_MAD_K, ipr_mad_k=IPR_MAD_K)
print(f"  {geom['spike_mask'].sum()} channels masked — applied to all thresholds\n")

# ── C3: Defect mask invariant ────────────────────────────────────────
check_defect_mask(geom['spike_mask'], len(geom['channel_betas']))

# ── Cross-threshold mask diagnostic setup ────────────────────────────
# True detector defects are threshold-independent; object-induced false
# positives are not.  Store each threshold's independently-detected mask and
# compare overlap after the loop.  Capture A's mid-row mean profile now while
# sino_A is in memory (for the diagnostic figure).
diag_masks   = {'A': geom['spike_mask']}
_didx        = np.linspace(0, sino_A.shape[0] - 1, 500, dtype=int)
diag_profile = sino_A[_didx, sino_A.shape[1] // 2, :].mean(axis=0).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────
# 3. Main loop: reconstruct each threshold
# ─────────────────────────────────────────────────────────────────────
# Preview figure: 3 rows normally, 4 rows when MAR is enabled in FAST mode
n_preview_rows = 4 if (FAST_MODE and MAR_STRENGTH > 0) else 3
fig, axes = plt.subplots(n_preview_rows, 4, figsize=(22, 5 * n_preview_rows))
fig.suptitle(
    f'4-threshold preview  z={z_centre:+.2f} mm   [{mode_str}  {mar_str}]',
    fontsize=12
)
axes[0, 0].set_ylabel('mid-projection profile')
axes[1, 0].set_ylabel('sinogram (preprocessed)')
axes[2, 0].set_ylabel('FBP reconstruction')
if n_preview_rows == 4:
    axes[3, 0].set_ylabel(f'FBP + MAR (strength={MAR_STRENGTH})')

# For full mode: collect volumes for the 4D NIfTI (reload from disk if needed)
saved_paths = {}

thresholds_to_process = [(0, sino_A)]   # A already loaded; loop adds B, C, D
# We'll handle the loop manually to control memory precisely

# C2: Threshold ordering check — sample threshold D (first 500 projections only)
# to avoid a peak-RAM spike from holding both sino_A and sino_D fully in memory.
print("\n--- Threshold ordering check (A vs D, sampled) ---")
with h5py.File(str(data_path), 'r') as f:
    ref_D = f['data_full']['A'][0, 0]           # physical 0 = logical D
    sino_D_sample = f[ref_D][:500, :, :].astype(np.float32)[:, :, ::-1]  # flip in numpy, not h5py
check_threshold_ordering(sino_A[:500], sino_D_sample)
del sino_D_sample;  gc.collect()

# Per-threshold mu_water in raw units (for C7 cross-threshold check, full mode)
mu_water_by_label: dict[str, float] = {}

for logical_idx, label in [(0, 'A'), (1, 'B'), (2, 'C'), (3, 'D')]:
    print(f"{'='*60}")
    print(f"Threshold {logical_idx}  —  {THR_LABELS[logical_idx]}")
    print(f"{'='*60}")
    reset_wavelet_stats()

    # Load sinogram (A was already loaded; reload others)
    if logical_idx == 0:
        sino = sino_A
    else:
        with h5py.File(str(data_path), 'r') as f:
            sino = _load_threshold(f, logical_idx)
        print(f"  Loaded: {sino.shape}  [{sino.min():.1f}, {sino.max():.1f}]")

    # Cross-threshold mask diagnostic: independently detect on B/C/D
    # so the overlap with A reveals true (threshold-independent) defects.
    if label != 'A':
        diag_masks[label] = detect_defect_channels(
            sino, spike_mad_k=SPIKE_MAD_K, ipr_mad_k=IPR_MAD_K, verbose=False)

    # ── Preview: rebin + preprocess one slice ─────────────────────────
    if Z_WEIGHTING == 'balanced':
        sino_raw_prev, angles_prev, wsum_prev = rebin_helical_to_axial(
            sino, geom, z_centre, z_weighting=Z_WEIGHTING, return_weights=True)
        if label == 'A':                       # C10: weighting must be uniform
            check_angular_balance(wsum_prev, label=label)
    else:
        sino_raw_prev, angles_prev = rebin_helical_to_axial(
            sino, geom, z_centre, z_weighting=Z_WEIGHTING)
    sino_proc_prev = preprocess_sinogram(
        sino_raw_prev, geom,
        hampel_threshold=HAMPEL_THRESHOLD,
        wavelet_ring_threshold=WAVELET_RING_THRESHOLD)
    sino_shift_prev = apply_cor_shift(sino_proc_prev, geom['det_alignment'])

    # ── C4: Sinogram invariant ────────────────────────────────────────
    check_sinogram_preprocessed(sino_proc_prev, label=label)

    k = sino_raw_prev.shape[0] // 2
    axes[0, logical_idx].plot(sino_raw_prev[k],  lw=0.8,
                               color='steelblue',  label='raw')
    axes[0, logical_idx].plot(sino_proc_prev[k], lw=0.8,
                               color='darkorange', label='preprocessed', alpha=0.9)
    axes[0, logical_idx].axhline(0, color='k', lw=0.5)
    axes[0, logical_idx].set_title(f"Threshold {label}\n{THR_LABELS[logical_idx]}")
    axes[0, logical_idx].legend(fontsize=7)

    vmax = np.percentile(sino_proc_prev, 99)
    axes[1, logical_idx].imshow(sino_proc_prev, aspect='auto',
                                 cmap='gray', vmin=0, vmax=vmax)
    axes[1, logical_idx].set_title(f"Threshold {label}\nsinogram")
    axes[1, logical_idx].set_xlabel('channel')

    # ── Slab reconstruction: image-domain average over PREVIEW_SLAB_MM
    # so the preview SNR matches the full volume, using RECON_METHOD.
    img_recon, n_slab = reconstruct_slab(
        sino, geom, z_centre, PREVIEW_SLAB_MM, z_spacing,
        method='astra', n_pixels=N_PIXELS, window="360",
        z_weighting=Z_WEIGHTING,
        hampel_threshold=HAMPEL_THRESHOLD,
        filter_name=FILTER_NAME, geometry_model=GEOMETRY_MODEL,
        wavelet_ring_threshold=WAVELET_RING_THRESHOLD,
        algorithm=RECON_METHOD, n_iter=N_ITER,
    )
    eff_mm = n_slab * abs(z_spacing)
    if n_slab > 1:
        print(f"  preview slab: {n_slab} slices → {eff_mm:.2f} mm effective "
              f"thickness  [{recon_str}]")

    # ── C5: Reconstruction invariant ─────────────────────────────────
    check_reconstruction(img_recon, geom, label=label, fov_mm=FOV_MM)

    # ── C9: Orientation invariant (logs table angle; never rotates) ───
    check_orientation(img_recon, geom, label=label)

    # raw-domain display window (also used by the MAR row below)
    pos = img_recon[img_recon > 0]
    v_lo, v_hi = (np.percentile(pos, [1, 99]) if pos.size else (0.0, 1.0))

    # ── HU calibration in FAST mode — all thresholds shown on the
    # SAME HU window so HU similarity/separation can be judged fairly.
    if ENABLE_HU_CALIBRATION:
        calib = auto_hu_calibrate(img_recon[np.newaxis, :, :], fov_mm=FOV_MM)
        mu_water_by_label[label] = float(calib['mu_water'])
        img_hu = apply_hu_calibration(img_recon[np.newaxis, :, :], calib)[0]
        lvl, wid = PREVIEW_HU_WL
        hu_lo, hu_hi = lvl - wid / 2.0, lvl + wid / 2.0
        axes[2, logical_idx].imshow(img_hu, cmap='gray', vmin=hu_lo, vmax=hu_hi)
        axes[2, logical_idx].set_title(
            f"Threshold {label}  {recon_str} → HU\n"
            f"slab {eff_mm:.1f} mm  µ_w={calib['mu_water']:.4g}  "
            f"W/L=[{hu_lo:.0f},{hu_hi:.0f}]")
    else:
        axes[2, logical_idx].imshow(img_recon, cmap='gray', vmin=v_lo, vmax=v_hi)
        axes[2, logical_idx].set_title(
            f"Threshold {label}  {recon_str} ({FILTER_NAME}/{GEOMETRY_MODEL})\n"
            f"slab {eff_mm:.1f} mm  z={z_centre:+.1f} mm")
    axes[2, logical_idx].axis('off')

    # ── MAR row (FAST mode only, when MAR is enabled) ─────────────────
    if FAST_MODE and MAR_STRENGTH > 0:
        sino_mar, metal_mask = apply_mar(
            sino_shift_prev, angles_prev, geom, N_PIXELS,
            strength=MAR_STRENGTH,
            metal_threshold=MAR_METAL_THRESHOLD,
            geometry_model=GEOMETRY_MODEL,
        )
        img_mar = _astra_fbp(sino_mar, angles_prev, geom, n_pixels=N_PIXELS,
                              filter_name=FILTER_NAME,
                              geometry_model=GEOMETRY_MODEL)
        axes[3, logical_idx].imshow(img_mar, cmap='gray', vmin=v_lo, vmax=v_hi)
        n_metal = metal_mask.sum() if metal_mask is not None else 0
        axes[3, logical_idx].set_title(
            f"Threshold {label}  MAR (s={MAR_STRENGTH})\n"
            f"{n_metal} metal vx  z={z_centre:+.1f} mm"
        )
        axes[3, logical_idx].axis('off')

    # ── Full mode: reconstruct complete volume ────────────────────────
    if not FAST_MODE:
        print(f"\n  Full reconstruction ({len(z_targets)} slices) ...")
        vol = reconstruct_helical_stack(
            sino, geom, z_targets, method='astra', n_pixels=N_PIXELS,
            mar_strength=MAR_STRENGTH,
            mar_metal_threshold=MAR_METAL_THRESHOLD,
            hampel_threshold=HAMPEL_THRESHOLD,
            filter_name=FILTER_NAME,
            geometry_model=GEOMETRY_MODEL,
            z_weighting=Z_WEIGHTING,
            wavelet_ring_threshold=WAVELET_RING_THRESHOLD,
            algorithm=RECON_METHOD, n_iter=N_ITER,
        )
        print(f"  vol_{label}: {vol.shape}  "
              f"range=[{vol.min():.3f}, {vol.max():.3f}]")

        # ── Optional z-smoothing for SNR boost on low-contrast inserts
        if Z_SMOOTH_MM > 0:
            vol = z_average(vol, Z_SMOOTH_MM, z_spacing)

        # ── C8: Output format invariant ───────────────────────────────
        check_output_format(vol, geom, z_targets, N_PIXELS)

        # ── B2: Auto HU calibration ───────────────────────────────────
        calib_path = out_dir / f'calibration_thr_{label}.json'
        if ENABLE_HU_CALIBRATION:
            if calib_path.exists():
                calib = json.loads(calib_path.read_text())
                print(f"  [HU calibration] Loaded cached calibration from {calib_path.name}")
            else:
                print(f"  [HU calibration] Running auto-calibration for threshold {label} ...")
                calib = auto_hu_calibrate(vol, fov_mm=FOV_MM)
                calib_path.write_text(json.dumps(calib, indent=2))
                print(f"  [HU calibration] Saved to {calib_path.name}")

            vol_hu = apply_hu_calibration(vol, calib)
            print(f"  vol_{label} HU: range=[{vol_hu.min():.1f}, {vol_hu.max():.1f}]")

            # Track raw mu_water for cross-threshold consistency check (C7)
            mu_water_by_label[label] = float(calib['mu_water'])

            out_path_hu = out_dir / f'reconstruction_thr_{label}_HU.nii.gz'
            arr_o, origin_o, dz_o = to_output_frame(
                vol_hu, z_targets, FOV_MM, N_PIXELS,
                mirror_x=OUTPUT_MIRROR_X, table_height_mm=TABLE_HEIGHT_MM)
            sitk.WriteImage(_sitk_vol(arr_o, xy_spacing, dz_o, origin_o),
                            str(out_path_hu))
            print(f"  -> Saved HU volume {out_path_hu.name}")

        # Save raw-attenuation NIfTI regardless
        out_path = out_dir / f'reconstruction_thr_{label}.nii.gz'
        arr_r, origin_r, dz_r = to_output_frame(
            vol, z_targets, FOV_MM, N_PIXELS,
            mirror_x=OUTPUT_MIRROR_X, table_height_mm=TABLE_HEIGHT_MM)
        sitk.WriteImage(_sitk_vol(arr_r, xy_spacing, dz_r, origin_r),
                        str(out_path))
        saved_paths[label] = out_path
        print(f"  -> Saved {out_path.name}")

        # ── C6: Slice continuity — use HU vol when available, raw otherwise.
        # Body-aware ROI inside check_slice_continuity handles off-centre phantoms.
        mid = len(z_targets) // 2
        if ENABLE_HU_CALIBRATION:
            check_slice_continuity(
                [vol_hu[mid-1], vol_hu[mid], vol_hu[mid+1]],
                label=f"{label} (HU)", tol=5.0,
            )
        else:
            check_slice_continuity(
                [vol[mid-1], vol[mid], vol[mid+1]],
                label=f"{label} (raw)", tol=0.5,
            )

        del vol
        if ENABLE_HU_CALIBRATION:
            del vol_hu

    # One-line summary of wavelet-gating decisions for this threshold
    report_wavelet_stats(label=label)

    # Free sinogram before loading next
    del sino;  gc.collect()
    print()

# Free sino_A reference (used as first iteration's sino)
del sino_A;  gc.collect()

# ── C7: Cross-threshold mu_water consistency (FAST slab or full volume) ──
if len(mu_water_by_label) == 4:
    check_cross_threshold(mu_water_by_label)

# ─────────────────────────────────────────────────────────────────────
# 4. Save preview figure
# ─────────────────────────────────────────────────────────────────────
plt.tight_layout()
preview_name = 'preview_4thresholds_fast.png' if FAST_MODE else 'preview_4thresholds.png'
plt.savefig(out_dir / preview_name, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {out_dir / preview_name}")

# ─────────────────────────────────────────────────────────────────────
# 4b. Channel-masking diagnostic — purely diagnostic, changes nothing
# ─────────────────────────────────────────────────────────────────────
# True detector defects are threshold-independent.  Comparing each threshold's
# independently-detected mask to A's tells us whether the ~14% flagged channels
# are genuine defects (high overlap) or object/threshold-induced false positives
# (low overlap → raise SPIKE_MAD_K to mask fewer channels).
if len(diag_masks) >= 2:
    n_ch       = len(geom['channel_betas'])
    mask_A     = diag_masks['A']
    all_labels = [l for l in ['A', 'B', 'C', 'D'] if l in diag_masks]

    def _jaccard(a, b):
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / union) if union else 1.0

    masked_in_all = np.ones(n_ch, dtype=bool)
    for l in all_labels:
        masked_in_all &= diag_masks[l]

    fig_m, (axp, axt) = plt.subplots(
        2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 2]})

    axp.plot(diag_profile, color='steelblue', lw=0.8)
    in_run, run_start = False, 0
    for c in range(n_ch):                       # shade A-masked channel runs
        if mask_A[c] and not in_run:
            run_start, in_run = c, True
        elif not mask_A[c] and in_run:
            axp.axvspan(run_start - 0.5, c - 0.5, color='red', alpha=0.25)
            in_run = False
    if in_run:
        axp.axvspan(run_start - 0.5, n_ch - 0.5, color='red', alpha=0.25)
    axp.set_xlim(0, n_ch)
    axp.set_xlabel('channel');  axp.set_ylabel('mean value (threshold A)')
    axp.set_title(
        f"Threshold A mid-row mean profile — {int(mask_A.sum())}/{n_ch} channels "
        f"masked ({100*mask_A.sum()/n_ch:.1f}%, shaded)   "
        f"[SPIKE_MAD_K={SPIKE_MAD_K}, IPR_MAD_K={IPR_MAD_K}]")

    lines = ["Cross-threshold defect-mask comparison",
             "(genuine detector defects are threshold-independent → high overlap with A)",
             ""]
    for l in all_labels:
        m = diag_masks[l]
        extra = "" if l == 'A' else f"   Jaccard(A,{l}) = {_jaccard(mask_A, m):.2f}"
        lines.append(f"  {l}: {int(m.sum()):4d} masked ({100*m.sum()/n_ch:4.1f}%){extra}")
    frac_core = masked_in_all.sum() / max(int(mask_A.sum()), 1)
    lines += ["",
              f"  masked in ALL {len(all_labels)} thresholds (core defects): "
              f"{int(masked_in_all.sum())} channels",
              f"  -> {100*frac_core:.0f}% of A's mask is threshold-independent"]
    if frac_core < 0.7:
        lines += ["", "  NOTE: low threshold-independent fraction — many flagged",
                  "        channels look object-induced; consider raising SPIKE_MAD_K."]
    axt.axis('off')
    axt.text(0.01, 0.98, "\n".join(lines), va='top', ha='left',
             family='monospace', fontsize=10, transform=axt.transAxes)

    fig_m.tight_layout()
    fig_m.savefig(out_dir / 'defect_mask_diagnostic.png',
                  dpi=120, bbox_inches='tight')
    plt.close(fig_m)
    print(f"Saved: {out_dir / 'defect_mask_diagnostic.png'}  "
          f"(core/all={int(masked_in_all.sum())}, "
          f"{100*frac_core:.0f}% of A threshold-independent)")

# ─────────────────────────────────────────────────────────────────────
# 5. Full mode only: central-slice summary + 4D NIfTI
# ─────────────────────────────────────────────────────────────────────
if not FAST_MODE:
    print("\nLoading volumes for summary figure...")
    vols = {
        lbl: sitk.GetArrayFromImage(
            sitk.ReadImage(str(saved_paths[lbl]))
        ).astype(np.float32)
        for lbl in ['A', 'B', 'C', 'D']
    }

    fig_sum, axes_sum = plt.subplots(2, 2, figsize=(12, 12))
    fig_sum.suptitle('4 threshold reconstructions — central slice', fontsize=13)
    for ax, (lbl, vol) in zip(axes_sum.flat, vols.items()):
        img  = vol[vol.shape[0] // 2]
        pos  = img[img > 0]
        v_lo, v_hi = (np.percentile(pos, [1, 99]) if pos.size else (0.0, 1.0))
        ax.imshow(img, cmap='gray', vmin=v_lo, vmax=v_hi)
        ax.set_title(f"Threshold {lbl}  ({THR_LABELS[['A','B','C','D'].index(lbl)]})\n"
                     f"W/L=[{v_lo:.1f}, {v_hi:.1f}]")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'reconstruction_4thr_summary.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir}/reconstruction_4thr_summary.png")

    # 4D NIfTI [z, y, x, 4]  — thresholds A, B, C, D as 4th dimension
    print("\nSaving 4D multi-threshold volume for material decomposition...")
    vol_4d = np.stack([vols[lbl] for lbl in ['A', 'B', 'C', 'D']],
                       axis=-1).astype(np.float32)
    sitk_4d = sitk.GetImageFromArray(vol_4d)
    sitk_4d.SetSpacing((xy_spacing, xy_spacing, abs(z_spacing), 1.0))
    sitk_4d.SetOrigin(origin + (0.0,))
    sitk.WriteImage(sitk_4d,
                    str(out_dir / 'reconstruction_4thr_multienergy.nii.gz'))
    print(f"  -> Saved {vol_4d.shape}  "
          f"{out_dir}/reconstruction_4thr_multienergy.nii.gz")

flush_invariant_log(out_dir)
print("\n=== Done ===")