"""
helical_reconstruction.py -- Single-Slice Rebinning (SSR) helical reconstruction library.

Pure library for the Siemens NAEOTOM Alpha photon-counting CT: geometry build, defect
detection, rebinning, preprocessing, metal-artefact reduction, reconstruction and HU
calibration.  No raw-data I/O and no __main__; python_reconstruction.py is the driver.

Scan-dependent values are derived from the descriptor and the detector tables, never written
here (invariant #7, README).  The tables are not part of the repository: build_geom reads them
from geo_dir under the file names given in its detector_spec.json, and fails loudly without them.

The sinograms are vendor-preprocessed, not line integrals, so every threshold is reconstructed
on its own and thresholds are never subtracted here (invariant #3, README).  The outermost
channels read near zero and serve as the air reference for baseline subtraction.

Defective channels are handled in two steps.  detect_defect_channels builds one channel mask
per scan (geom['spike_mask']) from max_excess = col_max - p99, which catches spikes that fire
at under 1 % of angles and that a percentile alone misses (invariant #4, README).
suppress_projection_spikes is an optional per-projection Hampel filter in preprocess_sinogram,
off by default (invariant #5, README).
"""

import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


# Per-threshold wavelet-gating statistics, reset by reset_wavelet_stats().
_wavelet_stats = {'applied': 0, 'skipped': 0, 'snr_sum': 0.0, 'last_threshold': 0.0}


# ─────────────────────────────────────────────────────────────────────
# View-angle convention: Siemens gantry tube-angle frame → ASTRA 'fanflat'
# ─────────────────────────────────────────────────────────────────────
# The reconstruction angle fed to ASTRA is derived per-scan from the descriptor
# (ScanDescr.Det.FirstTubeAngle + a uniform 360/FramesPerRotation increment), so
# the absolute orientation is handled automatically for ANY scan:
#
#     view_angle(k) = _VIEW_ANGLE_SIGN · tube_angle(k) + deg2rad(_VIEW_ANGLE_OFFSET_DEG)
#
# The two constants below describe the fixed relationship between the two
# COORDINATE SYSTEMS (Siemens gantry vs ASTRA fanflat, whose angle-0 places the
# source at (0,-SAD)).  They are NOT scan data and are never tuned per scan —
# they are the same for every acquisition of this scanner.
#
#   _VIEW_ANGLE_SIGN       preserves the (already correct, non-mirrored) rotation
#                          sense of the previous pipeline; the observed problem
#                          was a pure rotation, not a mirror, so +1.
#   _VIEW_ANGLE_OFFSET_DEG the fixed angular gap between Siemens' tube-angle zero
#                          and ASTRA's angle-0.  check_orientation() (in
#                          recon_invariants.py) measures and logs the patient-
#                          table angle every run; if the table is not at the
#                          bottom, its reported offset is set here ONCE and is
#                          then permanent — the production pipeline never
#                          determines the angle per run.
_VIEW_ANGLE_SIGN       = +1.0
_VIEW_ANGLE_OFFSET_DEG = -90.0


def reset_wavelet_stats():
    """Reset the wavelet-gating counters (call once per threshold)."""
    _wavelet_stats.update(applied=0, skipped=0, snr_sum=0.0, last_threshold=0.0)


def report_wavelet_stats(label=""):
    """Print a one-line summary of how often wavelet stripe removal was triggered."""
    a, s = _wavelet_stats['applied'], _wavelet_stats['skipped']
    total = a + s
    if total == 0:
        return
    avg_snr = _wavelet_stats['snr_sum'] / total
    print(f"[preprocess summary  {label}] wavelet stripe removal: "
          f"{a}/{total} slices applied  ({100*a/total:.1f}%)  "
          f"avg stripe_snr={avg_snr:.4f}  "
          f"threshold={_wavelet_stats['last_threshold']:.4f}")


# ─────────────────────────────────────────────────────────────────────
def scalar(val):
    """Unwrap nested numpy array wrapping from scipy.io struct fields."""
    while hasattr(val, '__len__') or (isinstance(val, np.ndarray) and val.ndim > 0):
        val = val.flat[0]
    return val.item() if isinstance(val, np.generic) else val


def _parse_modepar_xml(descriptor):
    """Parse ModeParXML, return dict of tag→text. Empty dict on failure."""
    try:
        raw = descriptor['ModeParXML'].flat[0]
        while isinstance(raw, (tuple, list, np.ndarray)):
            raw = raw.flat[0] if isinstance(raw, np.ndarray) else raw[0]
        if hasattr(raw, 'item'):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        root = ET.fromstring(raw)
        return {e.tag: e.text.strip() for e in root.iter()
                if e.text and e.text.strip()}
    except Exception as ex:
        print(f"[build_geom] Warning: could not parse ModeParXML: {ex}")
        return {}


def _select_active_rows(row_z_full, n_active, row_mapping):
    """Select n_active rows from the full row-position table according to row_mapping."""
    n_full = len(row_z_full)
    lo = (n_full - n_active) // 2

    if not isinstance(row_mapping, str):
        return np.asarray(row_z_full[row_mapping], dtype=np.float64)

    options = {
        "central":  row_z_full[lo : lo + n_active],
        "first":    row_z_full[:n_active],
        "last":     row_z_full[-n_active:],
        "reversed": row_z_full[lo : lo + n_active][::-1],
    }
    if row_mapping not in options:
        raise ValueError(f"row_mapping must be one of {list(options)}. Got: {row_mapping!r}")
    return np.asarray(options[row_mapping], dtype=np.float64)


def _check_row_mapping(row_z_mm, label):
    """Raise if row_z_mm does not straddle z=0 (SSR would be invalid)."""
    lo, hi = float(row_z_mm.min()), float(row_z_mm.max())
    if not (lo <= 0.0 <= hi):
        raise ValueError(
            f"Row mapping '{label}': row_z_mm [{lo:+.3f}, {hi:+.3f}] mm "
            f"does not straddle z=0. SSR interpolation will clip to boundary rows.\n"
            f"Try row_mapping='first', 'last', or 'reversed'."
        )
    print(f"[build_geom] row_z_mm OK — [{lo:+.3f}, {hi:+.3f}] mm  (mapping='{label}')")


_DETECTOR_SPEC = "detector_spec.json"
_TABLE_KEYS = ("channel_angles_file", "row_positions_file")


def _load_detector_tables(geo_dir):
    """
    Channel fan angles (rad) and detector row z-positions at isocentre (mm).

    The table file names are not written in this file: <geo_dir>/detector_spec.json names
    them under `channel_angles_file` and `row_positions_file`, so the repository carries no
    vendor file name and a detector with differently named tables needs no code change.
    """
    geo_dir = Path(geo_dir)
    spec_path = geo_dir / _DETECTOR_SPEC
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise FileNotFoundError(
            f"{spec_path} is missing or not valid JSON. It must name the detector tables "
            f"under {list(_TABLE_KEYS)} (see the README, invariant #7).") from ex
    missing = [k for k in _TABLE_KEYS if k not in spec]
    if missing:
        raise KeyError(f"{spec_path} does not name {missing} (see the README, invariant #7).")
    channel_betas = np.loadtxt(geo_dir / spec["channel_angles_file"]).astype(np.float64)
    row_z_full = np.loadtxt(geo_dir / spec["row_positions_file"]).astype(np.float64)
    return channel_betas, row_z_full


# ─────────────────────────────────────────────────────────────────────
def detect_defect_channels(sino_full, n_sample=500,
                           spike_mad_k=5.0, ipr_mad_k=6.0, verbose=True):
    """
    Robustly detect defective detector channels from a sinogram.

    IMPORTANT — why we use max_excess, not IPR
    ──────────────────────────────────────────
    Inter-module gap spikes in PCCT data are angle-dependent: they fire at
    perhaps 1–5 projections out of thousands.  With n_sample=500, the spike
    may appear at 0–2 sampled projections.  np.percentile(sample, 99) at
    500 rows is the 5th-highest value — if the spike appears at only 1-2
    rows it will be the 1st or 2nd highest, so p99 is still the *normal*
    background level.  Therefore ipr = p99 - p01 looks completely normal
    for these channels and threshold-based IPR detection misses them.

    The fix: use   max_excess = col_max − p99
    col_max is ALWAYS the spike value (regardless of how rarely it fires).
    Subtracting p99 removes normal signal variation, leaving only the excess
    above what a spike-free channel could produce.  A genuine spike channel
    will have a large, localised max_excess; its neighbours will not.

    Two detection passes are combined:
      (a) max_excess-based  — catches rare/acute spikes (primary fix)
      (b) IPR-based         — catches persistent hot channels and partial
                              defects that appear at many angles
      (c) dead-channel      — near-zero IPR (stuck/open channels)

    Parameters
    ----------
    sino_full : ndarray [N_proj, n_rows, n_channels]
    n_sample  : projections to sample (500 is robust; increase to 1000 if
                spikes are extremely rare, e.g. only 1-2 per rotation)
    spike_mad_k : float  MAD multiplier for the max-excess spike test (default
                5.0).  RAISE (e.g. 7–9) to mask fewer channels if the diagnostic
                shows the flagged channels are object-induced, not true defects.
    ipr_mad_k : float  MAD multiplier for the persistent-hot-channel test
                (default 6.0).
    verbose   : bool  print the per-channel diagnostic lists (default True).
                Set False for the cross-threshold comparison runs so the log
                is not spammed four times.

    Returns
    -------
    defect_mask : bool ndarray [n_channels]  True = defective
    """
    from scipy.ndimage import median_filter

    N, R, C = sino_full.shape
    idx     = np.linspace(0, N - 1, n_sample, dtype=int)
    mid_row = R // 2
    sample  = sino_full[idx, mid_row, :].astype(np.float64)   # (n_sample, C)

    p01     = np.percentile(sample,  1, axis=0)                # (C,)
    p99     = np.percentile(sample, 99, axis=0)                # (C,)
    col_max = sample.max(axis=0)                               # (C,)
    ipr     = p99 - p01                                        # (C,)

    # ── (a) Max-excess spike detection ───────────────────────────────
    # max_excess isolates the rare-but-extreme spike above the 99th-percentile
    # background.  Local median removes the smooth channel-to-channel gain
    # variation so only localised spikes pass the threshold.
    max_excess           = col_max - p99                       # ≥0 always
    me_local_median      = median_filter(max_excess, size=51)
    me_residual          = max_excess - me_local_median
    me_mad               = np.median(np.abs(me_residual - np.median(me_residual)))
    me_threshold         = spike_mad_k * max(me_mad, 1e-3 * col_max.mean())
    spike_from_max       = me_residual > me_threshold

    # ── (b) IPR-based spike detection (persistent hot channels) ──────
    ipr_local_median     = median_filter(ipr, size=51)
    ipr_residual         = ipr - ipr_local_median
    ipr_mad              = np.median(np.abs(ipr_residual - np.median(ipr_residual)))
    ipr_threshold        = ipr_mad_k * max(ipr_mad, 1e-3 * ipr.mean())
    spike_from_ipr       = ipr_residual > ipr_threshold

    # ── (c) Dead channel detection (near-zero IPR) ────────────────────
    dead_raw = ipr < 0.10 * np.maximum(ipr_local_median, 1e-6)

    # ── combine and dilate by ±1 channel ─────────────────────────────
    # Dilation catches the nearest partial-volume neighbours of each spike.
    # Kept at ±1 (not ±3) because this detector has many inter-module gap
    # spikes spaced 4–6 channels apart (especially in scans with metal).
    # A ±3 dilation bridges adjacent spikes and masks entire central regions
    # (~21% of channels observed in practice), degrading the reconstruction
    # far more than the spikes themselves would.  ±1 is sufficient to
    # interpolate the immediate neighbours of a flagged channel.
    raw_defect  = spike_from_max | spike_from_ipr | dead_raw
    defect_mask = raw_defect.copy()
    for shift in [-1, 1]:
        defect_mask |= np.roll(raw_defect, shift)

    # Diagnostics
    spike_ch_max = np.where(spike_from_max)[0].tolist()
    spike_ch_ipr = np.where(spike_from_ipr)[0].tolist()
    dead_ch      = np.where(dead_raw)[0].tolist()
    total_ch     = np.where(defect_mask)[0].tolist()

    if verbose:
        print(f"[detect_defect_channels] max_excess MAD={me_mad:.2f},  "
              f"spike(max) threshold={me_threshold:.2f}")
        print(f"  spike channels (max_excess):  {spike_ch_max}")
        print(f"  spike channels (IPR):         {spike_ch_ipr}")
        print(f"  dead  channels (low IPR):     {dead_ch}")
        print(f"  total masked after dilation ({len(total_ch)} ch): {total_ch}")

        if len(total_ch) == 0:
            print("  WARNING: no defective channels found — check spike_detection.png")
        if len(total_ch) > 0.05 * C:
            print(f"  WARNING: {len(total_ch)/C*100:.1f}% of channels flagged — "
                  f"threshold may be too aggressive")

    return defect_mask


# ─────────────────────────────────────────────────────────────────────
def suppress_projection_spikes(sino, window=31, threshold=5.0):
    """
    Per-projection Hampel-filter spike suppression.

    Operates on each projection (row) of the 2-D axial sinogram independently.
    For every projection, computes a running local median and local MAD along
    the channel axis; any channel whose value deviates from the local median by
    more than `threshold` × local MAD is replaced by the local median.

    This is the CT-standard approach for angle-dependent detector artifacts
    (inter-module gap spikes, occasional hot pixels):
      • No global channel list needed — fully adaptive per-projection
      • Catches spikes that appear at exactly one angle (e.g. direct-beam
        alignment), which a pre-computed global mask may miss if that angle
        was not in the detection sample
      • `window=31` is wide enough to define a robust local baseline across
        the smooth sinogram profile while narrow enough to resolve 2-channel
        spike clusters
      • `threshold=5.0` (5× MAD ≈ 7σ for Gaussian data) is very conservative:
        only true outliers are suppressed, not genuine signal variation

    Parameters
    ----------
    sino      : ndarray [n_proj, n_channels]  float32 or float64
    window    : int   local neighbourhood width in channels (must be odd;
                      31 is appropriate for PCCT with ~1400 channels)
    threshold : float MAD multiplier; 5.0 = conservative, 3.5 = aggressive
                      Increase threshold if valid high-attenuation channels
                      are being over-suppressed.

    Returns
    -------
    sino_out  : ndarray [n_proj, n_channels], same dtype — spikes replaced
    spike_map : bool ndarray [n_proj, n_channels] — True where suppressed
                (useful for diagnostics; not used further in the pipeline)
    """
    from scipy.ndimage import median_filter

    in_dtype = sino.dtype
    sino_f   = sino.astype(np.float64)

    # Running local median along channels: kernel (1, window)
    local_med = median_filter(sino_f, size=(1, window), mode='nearest')
    residual  = sino_f - local_med

    # Running local MAD: median of |residual| in same kernel
    local_mad = median_filter(np.abs(residual), size=(1, window), mode='nearest')
    local_mad = np.maximum(local_mad, 1e-6)     # prevent div-by-zero in air regions

    spike_map          = np.abs(residual) > threshold * local_mad
    sino_out           = sino_f.copy()
    sino_out[spike_map] = local_med[spike_map]

    n_flagged = int(spike_map.sum())
    if n_flagged > 0:
        n_proj_affected = int(spike_map.any(axis=1).sum())
        print(f"[suppress_projection_spikes] suppressed {n_flagged} samples "
              f"in {n_proj_affected}/{sino.shape[0]} projections "
              f"({100.*n_flagged/sino.size:.3f}% of sinogram)")
    else:
        print("[suppress_projection_spikes] no per-projection spikes found")

    return sino_out.astype(in_dtype), spike_map


# ─────────────────────────────────────────────────────────────────────
def build_geom(descriptor, geo_dir=".", channels_flipped=True, row_mapping="central"):
    """
    Build the geometry dict from the scan descriptor.

    All scan-dependent values (slice width, active rows, air channels) are
    derived from the descriptor/geometry files — nothing hardcoded.

    Call detect_defect_channels(sino_full) separately after loading the sinogram
    and store the result: geom['spike_mask'] = detect_defect_channels(sino_full)

    Parameters
    ----------
    descriptor      : scipy.io struct from the .mat descriptor file
    geo_dir         : directory holding detector_spec.json and the two detector
                      tables it names (see _load_detector_tables)
    channels_flipped: True if sino[:, :, ::-1] was applied at load time (the
                      pipeline flips channels, so this is True; it negates the
                      det_alignment sign). Required -- see invariant #2 in the README.
    row_mapping     : "central" | "first" | "last" | "reversed" | index array
    """
    config     = descriptor['Config'].flat[0]
    scan_descr = descriptor['ScanDescr'].flat[0]
    cnt_lookup = descriptor['CntLookup'].flat[0]
    geo_dir    = Path(geo_dir)

    # ── scanner geometry ─────────────────────────────────────────────
    SAD               = float(scalar(config['RadiusFocusPath']))
    SDD               = float(scalar(config['DistanceFocusDetector']))
    proj_per_rotation = int(scalar(scan_descr['FramesPerRotation']))
    n_total_proj      = int(scalar(scan_descr['NoOfReadings']))

    # ── focal-spot z per projection ───────────────────────────────────
    table_um         = np.asarray(cnt_lookup['TablePosition']).ravel().astype(np.float64)
    z_focus_per_proj = table_um * 1e-3                          # µm → mm
    pitch_mm         = ((z_focus_per_proj[-1] - z_focus_per_proj[0])
                        / (n_total_proj / proj_per_rotation))

    # ── per-projection view (tube) angle ──────────────────────────────
    # The physical gantry angle of the X-ray tube at reading 0 is in
    # ScanDescr.Det.FirstTubeAngle (detector A = first element), in millidegrees;
    # the increment is uniform (FlyingFocalSpot=None) at 360/FramesPerRotation.
    # Using this REPLACES the old assumption that reading 0 = ASTRA angle 0, which
    # ignored the start angle and rotated the whole reconstruction (e.g. the
    # patient table appeared on the side instead of at the bottom).  Everything
    # here is read from the descriptor, so orientation is correct for any scan.
    # SystemAngle / TubeBOffsetAngle are the dual-source A↔B mounting offset and
    # are deliberately NOT used (we reconstruct system A only).
    try:
        first_tube_angle_deg = float(scalar(scan_descr['Det']['FirstTubeAngle'])) / 1000.0
        k_idx          = np.arange(len(z_focus_per_proj), dtype=np.float64)
        tube_angle_rad = (np.deg2rad(first_tube_angle_deg)
                          + k_idx * (2.0 * np.pi / proj_per_rotation))
        view_angle_per_proj = (_VIEW_ANGLE_SIGN * tube_angle_rad
                               + np.deg2rad(_VIEW_ANGLE_OFFSET_DEG))
        print(f"[build_geom] FirstTubeAngle(A) = {first_tube_angle_deg:.3f} deg  "
              f"-> view-angle convention sign={_VIEW_ANGLE_SIGN:+.0f}, "
              f"offset={_VIEW_ANGLE_OFFSET_DEG:+.1f} deg")
    except Exception as ex:
        raise ValueError(
            "Could not read ScanDescr.Det.FirstTubeAngle, which sets the absolute "
            "scan start angle required for correct orientation. Refusing to fall "
            "back to reading-0 = angle 0 — that silently rotates the whole volume "
            "by the start angle (~151-350 deg). Check the descriptor field path."
        ) from ex

    # ── channel geometry ──────────────────────────────────────────────
    channel_betas, row_z_full = _load_detector_tables(geo_dir)
    beta_abs       = np.abs(channel_betas)
    n_air_channels = int(np.sum(beta_abs >= 0.90 * beta_abs.max()))
    n_air_channels = max(n_air_channels, 8)

    # ── collimated slice width ────────────────────────────────────────
    modepar = _parse_modepar_xml(descriptor)
    if 'SliceWidthCollimated' in modepar:
        slice_width_mm = float(modepar['SliceWidthCollimated']) * 1e-3
        print(f"[build_geom] SliceWidthCollimated from XML = "
              f"{float(modepar['SliceWidthCollimated']):.0f} µm → {slice_width_mm:.4f} mm")
    else:
        n_tmp  = int(scalar(scan_descr['NoOfSlices']))
        lo_tmp = (len(row_z_full) - n_tmp) // 2
        rows_tmp = row_z_full[lo_tmp : lo_tmp + n_tmp]
        slice_width_mm = float(abs(rows_tmp[0] - rows_tmp[-1]) / (n_tmp - 1))
        print(f"[build_geom] SliceWidthCollimated derived from the row table = {slice_width_mm:.4f} mm")

    # ── active row count ──────────────────────────────────────────────
    if 'NoOfSlicesCollimated' in modepar:
        n_active = int(modepar['NoOfSlicesCollimated'])
        print(f"[build_geom] NoOfSlicesCollimated from XML = {n_active}")
    else:
        n_active = int(scalar(scan_descr['NoOfSlices']))
        print(f"[build_geom] NoOfSlices from ScanDescr = {n_active}")

    # ── row z-positions ───────────────────────────────────────────────
    row_z_mm = _select_active_rows(row_z_full, n_active, row_mapping)
    _check_row_mapping(row_z_mm, row_mapping)

    # ── detector alignment ────────────────────────────────────────────
    det_alignment = float(scalar(config['DetectorAlignment']))
    if channels_flipped:
        det_alignment = -det_alignment

    return {
        'SAD':                  SAD,
        'SDD':                  SDD,
        'proj_per_rotation':    proj_per_rotation,
        'n_total_proj':         n_total_proj,
        'channel_betas':        channel_betas,
        'detector_spacing_deg': float(scalar(config['FanBeamGrid'])) * 2,
        'n_air_channels':       n_air_channels,
        'row_z_mm':             row_z_mm,
        'slice_width_mm':       slice_width_mm,
        'z_spacing_mm':         slice_width_mm,
        'z_focus_per_proj':     z_focus_per_proj,
        'view_angle_per_proj':  view_angle_per_proj,
        'pitch_mm':             pitch_mm,
        'det_alignment':        det_alignment,
        'spike_mask':           None,
    }


# ─────────────────────────────────────────────────────────────────────
def remove_stripes_wavelet_fft(sino, level=None, sigma=1.5, wavelet='db5'):
    """
    Wavelet-FFT stripe removal (Münch et al., Opt. Express 2009).

    Handles residual ring-producing gain non-uniformity that was not caught by
    hard-defect detection or per-projection spike suppression (e.g. broad
    partial stripes, smooth gain drifts across channel groups).

    Parameters
    ----------
    sino    : ndarray [n_proj, n_channels]
    level   : int  wavelet levels (None = auto, capped at 6)
    sigma   : float  Gaussian damping width; 1.5 = conservative (preserves
              signal), 3–5 = aggressive (removes more rings)
    wavelet : PyWavelets wavelet name; 'db5' is standard for CT

    Returns
    -------
    ndarray [n_proj, n_channels]  stripe-suppressed, same dtype as input
    """
    import pywt

    in_dtype = sino.dtype
    sino     = sino.astype(np.float64)
    n_proj, n_ch = sino.shape

    if level is None:
        level = min(pywt.dwt_max_level(n_ch, wavelet), 6)

    coeffs = pywt.wavedec(sino, wavelet, level=level, axis=1)

    freqs = np.fft.fftfreq(n_proj)
    damp  = 1.0 - np.exp(-freqs**2 / (2.0 * sigma**2))

    for i in range(1, len(coeffs)):
        fft_d      = np.fft.fft(coeffs[i], axis=0)
        fft_d     *= damp[:, np.newaxis]
        coeffs[i]  = np.real(np.fft.ifft(fft_d, axis=0))

    result = pywt.waverec(coeffs, wavelet, axis=1)
    return result[:, :n_ch].astype(in_dtype)


def preprocess_sinogram(sino_axial, geom, hampel_threshold=None,
                         wavelet_ring_threshold=2.0):
    """
    Full preprocessing pipeline for one rebinned 2-D sinogram.

    Pipeline
    ────────
    0. Per-projection Hampel spike suppression (OPTIONAL, default OFF).

       WHY IT IS OFF BY DEFAULT
       ───────────────────────
       The Hampel filter computes a running local median and MAD along the
       channel axis and replaces any channel deviating more than
       hampel_threshold × MAD from the local median.  Inside a large water
       phantom, the profile is smooth and the MAD is small (~50–100 scanner
       units).  A bone-equivalent insert creates a narrow local peak of
       perhaps 300–500 units above background.  At threshold=7.0:
           7 × 80 = 560 ≥ 400  →  insert peak replaced with water median
       This suppresses high-attenuation inserts (spine, metal) and causes
       their reconstructed HU values to be severely compressed toward water.

       The hard-defect mask (Step 1) handles the actual detector spikes.
       Only enable the Hampel (threshold ≥ 20) if residual spike rings are
       still visible AFTER reviewing the hard-defect mask diagnostics.

    1. Hard defect channel interpolation — removes channels flagged globally
       by detect_defect_channels(), stored in geom['spike_mask'].

    2. Adaptive wavelet-FFT stripe removal — ONLY applied when the
       reconstructed slice (without wavelet) has a ring-index above
       wavelet_ring_threshold (default 2.0).  This prevents the suppressor
       from washing out small, high-contrast inserts (e.g. low-density
       calcium equivalents in the QRM DE phantom) when no rings are present.
       Requires PyWavelets; skipped entirely if not installed.

    3. Per-projection air baseline subtraction — outer n_air_channels define
       the zero-attenuation reference; subtracting makes air → 0.

    4. Non-negative clip.

    Parameters
    ----------
    sino_axial              : ndarray [n_proj, n_channels]
    geom                    : dict from build_geom(); geom['spike_mask'] should be set
    hampel_threshold        : float | None
        MAD multiplier for the per-projection Hampel filter.
        None  = disabled (default — preserves insert and metal HU values).
        ≥ 20  = light backup spike suppression, still very conservative.
        7–10  = original setting, NOT recommended — suppresses bone/metal.
    wavelet_ring_threshold  : float (default 2.0)
        Ring-index value above which wavelet stripe removal is applied.
        A ring index of 2.0 means the angular std at a given radius is 2×
        the global image std — clearly ring-dominated.  Set to 0.0 to always
        apply, or to a very large value (e.g. 999) to never apply.

    Returns
    -------
    ndarray [n_proj, n_channels]  clean, non-negative line integrals
    """
    sino = sino_axial.copy().astype(np.float32)

    # ── Step 0: Hampel filter (disabled by default) ───────────────────
    if hampel_threshold is not None:
        sino, _ = suppress_projection_spikes(sino, window=31,
                                              threshold=float(hampel_threshold))

    # ── Step 1: hard defect channel interpolation ─────────────────────
    defect_mask = geom.get('spike_mask')
    if defect_mask is not None and defect_mask.any():
        good = np.where(~defect_mask)[0]
        for c in np.where(defect_mask)[0]:
            left  = good[good < c][-1] if (good < c).any() else good[0]
            right = good[good > c][0]  if (good > c).any() else good[-1]
            t = (c - left) / max(right - left, 1)
            sino[:, c] = (1.0 - t) * sino[:, left] + t * sino[:, right]

    # ── Step 2: adaptive wavelet-FFT stripe removal ───────────────────
    # Gate on ring index: only apply when rings are actually present.
    # We do a quick preview FBP of the un-wavelet sinogram to measure rings,
    # but to keep preprocess_sinogram free of FBP imports we use a cheaper
    # proxy: the along-projection standard deviation of the column means.
    # A strong stripe signal makes the column-mean profile rough (high std
    # relative to its own median), which is a fast, FBP-free ring proxy.
    try:
        col_means  = sino.mean(axis=0)                  # (n_channels,)
        col_med    = float(np.median(col_means))
        col_std    = float(col_means.std())
        stripe_snr = col_std / max(abs(col_med), 1e-6)  # rough stripe proxy

        # Empirical mapping: stripe_snr > 0.04 ≈ ring_index > 2.0 in practice.
        # This avoids needing a full FBP pass inside preprocessing.
        stripe_threshold = 0.04 * wavelet_ring_threshold / 2.0
        if stripe_snr > stripe_threshold:
            sino = remove_stripes_wavelet_fft(sino, sigma=1.5)
            _wavelet_stats['applied'] += 1
        else:
            _wavelet_stats['skipped'] += 1
        _wavelet_stats['snr_sum'] += float(stripe_snr)
        _wavelet_stats['last_threshold'] = float(stripe_threshold)
    except ImportError:
        pass

    # ── Step 3: per-projection air baseline subtraction ──────────────
    n = geom['n_air_channels']
    air_mean = 0.5 * (sino[:, :n].mean(axis=1, keepdims=True)
                    + sino[:, -n:].mean(axis=1, keepdims=True))
    sino -= air_mean

    # ── Step 4: non-negative clip ─────────────────────────────────────
    return np.clip(sino, 0.0, None)


def apply_cor_shift(sino_axial, det_alignment):
    """Sub-pixel centre-of-rotation correction in the channel direction."""
    from scipy.ndimage import shift as ndshift
    return ndshift(sino_axial, [0, -det_alignment], order=3, mode='nearest')


# ─────────────────────────────────────────────────────────────────────
def rebin_helical_to_axial(sino_full, geom, z_target_mm, window="360",
                            z_window=True, z_weighting=None,
                            n_rot_balanced=2, z_taper_fwhm_mm=None,
                            return_weights=False):
    """
    Build a 2-D axial sinogram at z_target_mm.

    For every projection k in the window, linearly interpolate between the
    two detector rows whose isocentre z-positions bracket
        z_offset(k) = z_target - z_focus(k).

    Parameters
    ----------
    sino_full   : float32 [N_proj, n_rows, n_channels]
    geom        : dict from build_geom()
    z_target_mm : float  target slice z position in mm
    window      : "360" (default) | "180+fan"  — window for the 'hann'/'none'
        modes.  Ignored when z_weighting='balanced' (it builds its own
        multi-rotation window).
    z_window    : bool (default True)  — LEGACY switch, kept for back-compat.
        Maps to z_weighting='hann' (True) / 'none' (False) when z_weighting is
        not given explicitly.
    z_weighting : None | 'hann' | 'none' | 'balanced'
        Projection-axis weighting applied during rebinning.  Default None →
        derived from z_window, so existing callers are byte-for-byte unchanged.
          'hann'     — raised-cosine (Hann) taper over a one-rotation window.
            Suppresses the partial-row rays at the helix entry/exit that cause
            10–25 HU z-shading bands, BUT because the window is exactly one
            rotation the taper is also an angular apodization whose peak sits at
            the window-centre view angle; that angle advances with z, so the
            low-frequency shading ROTATES as you scroll (the "rotating light
            cone").  This is the historical behaviour.
          'none'     — no projection weighting (uniform).  No rotating bias, but
            the oblique entry/exit rays reintroduce the z-shading bands.
          'balanced' — angularly-balanced helical weighting (the driver default).
            Spans n_rot_balanced rotations so every view angle has several
            candidate rays at different z-offsets; weights each ray by a
            symmetric taper in |z_offset| (suppresses oblique rays → no
            z-shading) and NORMALISES per view angle so every angle's total
            weight is 1 (uniform angular weighting → nothing rotates).  Standard
            helical 360°LI / complementary-rebinning redundancy weighting
            (Crawford & King, Med. Phys. 1990).  Purely a sinogram-formation
            change, so FBP/SIRT/CGLS all benefit identically.
    n_rot_balanced : int (default 2)  — rotations spanned by the 'balanced'
        window.  ≥2 so each view angle has multiple z-offset candidates.
    z_taper_fwhm_mm : float | None  — |z_offset| taper width for 'balanced'.
        None → 1.5 × geom['slice_width_mm'] (derived, never hardcoded per scan).
    return_weights : bool (default False)  — if True, also return the realised
        per-view-angle weight sum (diagnostic; ≈1 everywhere for 'balanced').

    Returns
    -------
    (sino_axial [n_proj, n_ch], angles_rad [n_proj])                if not return_weights
    (sino_axial [n_proj, n_ch], angles_rad [n_proj], wsum [n_proj])  if return_weights
    """
    N_total, n_rows, n_channels = sino_full.shape
    if N_total != len(geom['z_focus_per_proj']):
        raise ValueError(f"sinogram has {N_total} projections but geom has "
                         f"{len(geom['z_focus_per_proj'])}")
    if n_rows != len(geom['row_z_mm']):
        raise ValueError(f"sinogram has {n_rows} rows but geom has "
                         f"{len(geom['row_z_mm'])}. Check row_mapping.")

    P       = geom['proj_per_rotation']
    z_focus = geom['z_focus_per_proj']
    row_z   = geom['row_z_mm']

    k_centre = int(np.argmin(np.abs(z_focus - z_target_mm)))

    # ── resolve weighting mode (explicit z_weighting wins; else legacy z_window)
    if z_weighting is None:
        z_weighting = 'hann' if z_window else 'none'
    z_weighting = str(z_weighting).lower()
    if z_weighting not in ('hann', 'none', 'balanced'):
        raise ValueError(f"Unknown z_weighting: {z_weighting!r}. "
                         f"Use 'balanced', 'hann', or 'none'.")

    if z_weighting == 'balanced':
        # Multi-rotation window so every view angle has several z-offset
        # candidates (required for the per-angle normalisation further down).
        n_rot  = max(2, int(round(n_rot_balanced)))
        n_span = n_rot * P
        k_lo   = k_centre - n_span // 2
        k_hi   = k_lo + n_span
    elif window == "360":
        k_lo = k_centre - P // 2
        k_hi = k_lo + P
    elif window == "180+fan":
        beta_max_deg = np.degrees(np.max(np.abs(geom['channel_betas'])))
        n_win = int(np.ceil((180.0 + 2.0 * beta_max_deg) / 360.0 * P))
        k_lo  = k_centre - n_win // 2
        k_hi  = k_lo + n_win
    else:
        raise ValueError(f"Unknown window: {window!r}")

    if k_lo < 0 or k_hi > N_total:
        raise ValueError(
            f"Window [{k_lo},{k_hi}) out of range for {N_total} projections "
            f"at z={z_target_mm:+.2f} mm.\n"
            f"Valid range: z_focus[{P//2}]={z_focus[P//2]:+.2f} to "
            f"z_focus[{-(P//2)}]={z_focus[-(P//2)]:+.2f} mm"
        )

    k_range  = np.arange(k_lo, k_hi)
    z_offset = z_target_mm - z_focus[k_range]

    if row_z[0] > row_z[-1]:
        asc, target = -row_z, -z_offset
    else:
        asc, target = row_z, z_offset

    r1 = np.searchsorted(asc, target).clip(1, n_rows - 1)
    r0 = r1 - 1
    z0, z1 = row_z[r0], row_z[r1]
    denom   = np.where(z1 != z0, z1 - z0, 1.0)
    w1      = np.clip((z_offset - z0) / denom, 0.0, 1.0)
    w0      = 1.0 - w1

    rowint = (w0[:, None] * sino_full[k_range, r0, :]
            + w1[:, None] * sino_full[k_range, r1, :])

    # Absolute view angle per projection, descriptor-derived (correct orientation
    # for any scan).  build_geom raises if FirstTubeAngle is unreadable, so this is
    # always populated; the None guard is defence in depth — we refuse to silently
    # revert to the index-based (rotated) assumption.
    view = geom.get('view_angle_per_proj')
    if view is None:
        raise ValueError(
            "geom['view_angle_per_proj'] is None — the descriptor-derived view "
            "angle is missing. build_geom should have raised on an unreadable "
            "FirstTubeAngle; rebuild geom from a descriptor that contains it.")
    view = np.asarray(view)

    # ── Projection-axis weighting ─────────────────────────────────────
    if z_weighting == 'balanced':
        # Angularly-balanced helical weighting: taper each ray by |z_offset| to
        # suppress the oblique helix-end rays that cause z-shading, then
        # NORMALISE per view angle so every angle's total weight is 1.  Uniform
        # angular weighting → no directional bias → the low-frequency lobe no
        # longer rotates with z.  Collapses the n_rot-rotation window to one ray
        # per canonical view angle (P angles), preserving the return shape.
        rowint_r = rowint.reshape(n_rot, P, n_channels)
        zoff_r   = z_offset.reshape(n_rot, P)
        T = (z_taper_fwhm_mm if z_taper_fwhm_mm is not None
             else 1.5 * geom['slice_width_mm'])
        w_raw = np.clip(1.0 - np.abs(zoff_r) / T, 0.0, None)        # (n_rot, P)
        S     = w_raw.sum(axis=0)                                   # (P,)
        w     = w_raw / np.where(S > 0.0, S, 1.0)[None, :]          # (n_rot, P)
        # Edge fallback: if the taper clipped every ray of an angle (S == 0,
        # only near the scan ends), keep the single nearest-|z_offset| ray so
        # ASTRA never sees a missing view angle.
        zero = np.where(S <= 0.0)[0]
        if zero.size:
            rstar = np.argmin(np.abs(zoff_r[:, zero]), axis=0)
            w[:, zero] = 0.0
            w[rstar, zero] = 1.0
        sino_axial  = (w[:, :, None] * rowint_r).sum(axis=0)        # (P, n_ch)
        weight_sums = w.sum(axis=0)                                 # (P,) ≈ 1.0
        angles_rad  = view[k_lo:k_lo + P] % (2.0 * np.pi)
    elif z_weighting == 'hann':
        # Legacy raised-cosine z-window: a one-rotation taper that doubles as an
        # angular apodization whose peak rotates with z (the "rotating light
        # cone").  Kept for back-compat / A-B comparison.
        n_win = len(k_range)
        hann  = np.hanning(n_win).astype(np.float32)   # 0→1→0 over window
        hann  = hann / (hann.mean() + 1e-12)           # mean 1 (preserve HU level)
        sino_axial  = rowint * hann[:, np.newaxis]
        weight_sums = hann
        angles_rad  = view[k_range] % (2.0 * np.pi)
    else:  # 'none'
        sino_axial  = rowint
        weight_sums = np.ones(len(k_range), dtype=np.float32)
        angles_rad  = view[k_range] % (2.0 * np.pi)

    if return_weights:
        return sino_axial, angles_rad, weight_sums
    return sino_axial, angles_rad


# ─────────────────────────────────────────────────────────────────────
def _remap_curved_to_flat(sino_axial, geom, oversample=1.5):
    """
    Remap a sinogram sampled on a curved (equiangular) detector to equispaced
    positions on a virtual flat detector at distance SDD, then return the
    remapped sinogram and the corresponding uniform detector pixel width.

    The detector is curved: each channel k is defined by its fan angle β_k
    (radians, from the channel-angle table).  ASTRA's 'fanflat' projector
    assumes a flat detector with uniform pixel spacing.  Using angularly-sampled
    data directly on a flat-detector FBP introduces radial position errors of
    several mm and HU cupping of 30–80 HU at the FOV periphery.

    Steps
    ─────
    1. Cosine pre-weight: multiply each channel by cos(β_k).  This accounts for
       the extra arc-length contribution of oblique rays on the curved array; the
       equiangular FBP formula requires this weight (Kak & Slaney 1988, ch. 3).
    2. Map to flat: flat positions s_k = SDD · tan(β_k) (mm from detector centre).
    3. Define a uniform flat grid at ~1.5× oversampling to suppress aliasing near
       the periphery (Nyquist density of the angular sampling grows as sec²(β)).
    4. Cubic interpolation from s_k (non-uniform) to s_uniform (uniform).

    Parameters
    ----------
    sino_axial : ndarray [n_proj, n_channels]
    geom       : dict from build_geom() — must contain 'channel_betas' and 'SDD'
    oversample : float  oversampling factor for the uniform flat grid (≥ 1.0)

    Returns
    -------
    sino_flat  : ndarray [n_proj, n_det_uniform]  remapped sinogram
    det_width  : float  uniform pixel width on the flat detector (mm)
    n_uniform  : int    number of flat-detector pixels
    """
    from scipy.interpolate import interp1d

    betas   = geom['channel_betas']          # (n_channels,) radians, non-uniform
    SDD     = geom['SDD']

    # 1. Cosine pre-weight
    cos_w      = np.cos(betas)                # (n_channels,)
    sino_w     = sino_axial * cos_w[np.newaxis, :]

    # 2. Source flat-detector positions for each channel
    s_curved   = SDD * np.tan(betas)          # (n_channels,) mm, non-uniform

    # 3. Uniform flat grid — span the same range as s_curved
    s_lo, s_hi = s_curved.min(), s_curved.max()
    n_uniform  = int(np.ceil(len(betas) * oversample))
    s_uniform  = np.linspace(s_lo, s_hi, n_uniform)
    det_width  = float((s_hi - s_lo) / (n_uniform - 1)) if n_uniform > 1 else 1.0

    # 4. Cubic spline interpolation (extrapolate='extrapolate' handles
    #    the small overshoot at the edges from the tan() stretch)
    interp = interp1d(s_curved, sino_w, kind='cubic', axis=1,
                      bounds_error=False, fill_value=0.0)
    sino_flat = interp(s_uniform).astype(sino_axial.dtype)

    return sino_flat, det_width, n_uniform


def _astra_reconstruct(sino_axial, angles_rad, geom, n_pixels,
                       algorithm='fbp', filter_name='shepp-logan',
                       geometry_model='curved', n_iter=150,
                       min_constraint=0.0):
    """
    Fan-beam reconstruction via ASTRA Toolbox — analytic (FBP) or iterative
    (SIRT, CGLS).  All algorithms share the same geometry build and curved→flat
    remap, so they are directly comparable.

    Parameters
    ----------
    sino_axial     : ndarray [n_proj, n_channels]  preprocessed, COR-corrected
    angles_rad     : ndarray [n_proj]
    geom           : dict from build_geom()
    n_pixels       : int  reconstruction grid size
    algorithm      : 'fbp' (default) | 'sirt' | 'cgls'
                     'fbp'  — filtered back-projection (FBP_CUDA).  Fast, the
                       established default; uses filter_name.
                     'sirt' — Simultaneous Iterative Reconstruction Technique
                       (SIRT_CUDA) with a non-negativity constraint.  Strong,
                       well-understood noise/streak suppression for the
                       photon-starved high thresholds; n_iter controls the
                       noise/resolution trade-off (more iterations → sharper +
                       noisier).  No regularisation weight to hand-tune.
                     'cgls' — Conjugate-Gradient Least Squares (CGLS_CUDA).
                       Faster convergence than SIRT but semi-convergent (noise
                       rises if over-iterated), so n_iter is touchier.
    filter_name    : str  ASTRA FilterType string (FBP only).
                     'shepp-logan' (default) — smooth quantitative kernel,
                       ~Siemens Qr40, good for low-contrast inserts and
                       material decomposition inputs.
                     'ram-lak' — full-bandwidth ramp; use for MTF / geometry
                       verification only (30–60% more noise than shepp-logan).
                     Other ASTRA options: 'cosine', 'hamming', 'hann', etc.
    geometry_model : 'curved' (default) | 'flat'
                     'curved' — remap from equiangular (physical) to equispaced
                       flat-detector sampling before reconstruction (correct for
                       the curved detector).  Fixes radial position errors
                       and peripheral HU cupping.
                     'flat'   — legacy behaviour: treat angularly-sampled
                       channels as equispaced flat-detector pixels.  Use only
                       for A/B comparison with prior results.
    n_iter         : int  number of iterations for 'sirt'/'cgls' (ignored for
                     'fbp').
    min_constraint : float  lower clamp applied each SIRT iteration (default 0.0;
                     valid because the data are non-negative line integrals).
    """
    import astra

    if geometry_model == 'curved':
        sino_fbp, det_width, n_det = _remap_curved_to_flat(sino_axial, geom)
    else:
        # Legacy flat-detector approximation
        n_det     = sino_axial.shape[1]
        det_width = geom['detector_spacing_deg'] * (np.pi / 180.0) * geom['SDD']
        sino_fbp  = sino_axial

    proj_geom = astra.create_proj_geom('fanflat', det_width, n_det, angles_rad,
                                       geom['SAD'], geom['SDD'] - geom['SAD'])
    vol_geom = astra.create_vol_geom(n_pixels, n_pixels)
    sino_id  = astra.data2d.create('-sino', proj_geom, sino_fbp)
    rec_id   = astra.data2d.create('-vol',  vol_geom)

    algo = str(algorithm).lower()
    if algo == 'fbp':
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId']     = sino_id
        cfg['FilterType']           = filter_name
    elif algo == 'sirt':
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId']     = sino_id
        cfg['option']               = {'MinConstraint': float(min_constraint)}
    elif algo == 'cgls':
        cfg = astra.astra_dict('CGLS_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId']     = sino_id
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. "
                         f"Use 'fbp', 'sirt', or 'cgls'.")

    alg_id = astra.algorithm.create(cfg)
    if algo == 'fbp':
        astra.algorithm.run(alg_id)
    else:
        astra.algorithm.run(alg_id, int(n_iter))
    img = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    return img


def _astra_fbp(sino_axial, angles_rad, geom, n_pixels,
               filter_name='shepp-logan', geometry_model='curved'):
    """Thin FBP wrapper around _astra_reconstruct (kept for back-compat)."""
    return _astra_reconstruct(sino_axial, angles_rad, geom, n_pixels,
                              algorithm='fbp', filter_name=filter_name,
                              geometry_model=geometry_model)


def _astra_forward_project(mask_2d, angles_rad, geom, n_pixels,
                            geometry_model='curved'):
    """
    Forward-project a 2-D image (e.g. binary metal mask) to sinogram space.

    geometry_model must match the value used in _astra_fbp so that the metal
    sinogram trace M is in the same coordinate system as the preprocessed data.
    When 'curved', the forward-projected sinogram is in equispaced flat-detector
    space (after the same remap applied in FBP).
    """
    import astra

    if geometry_model == 'curved':
        # Build flat-detector geometry consistent with _remap_curved_to_flat
        _, det_width_flat, n_det_flat = _remap_curved_to_flat(
            np.zeros((1, len(geom['channel_betas'])), dtype=np.float32), geom
        )
        det_width = det_width_flat
        n_det     = n_det_flat
    else:
        n_det     = len(geom['channel_betas'])
        det_width = geom['detector_spacing_deg'] * (np.pi / 180.0) * geom['SDD']

    proj_geom = astra.create_proj_geom('fanflat', det_width, n_det, angles_rad,
                                       geom['SAD'], geom['SDD'] - geom['SAD'])
    vol_geom  = astra.create_vol_geom(n_pixels, n_pixels)
    vol_id    = astra.data2d.create('-vol',  vol_geom, mask_2d.astype(float))
    sino_id   = astra.data2d.create('-sino', proj_geom)
    cfg = astra.astra_dict('FP_CUDA')
    cfg['VolumeDataId']     = vol_id
    cfg['ProjectionDataId'] = sino_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    fp = astra.data2d.get(sino_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(vol_id)
    astra.data2d.delete(sino_id)
    return fp


# ─────────────────────────────────────────────────────────────────────
def segment_metal_from_fbp(img_fbp, metal_threshold=None, dilate_px=2):
    """
    Segment metal voxels from a first-pass FBP reconstruction.

    Metal objects (rings, screws, phone components) appear at the extreme
    high-attenuation tail of the FBP value distribution.  The threshold is
    found at the valley between the main object peak and the metal tail by
    searching the smoothed histogram between the 90th and 99.9th percentile
    of positive values.

    Parameters
    ----------
    img_fbp         : ndarray [ny, nx]  first-pass FBP (non-negative clipped)
    metal_threshold : float | None  manual threshold; None = auto-detect
    dilate_px       : int  dilation radius in pixels (catches partial-volume
                      edge voxels at metal boundaries)

    Returns
    -------
    metal_mask : bool ndarray [ny, nx]
    threshold  : float  threshold that was used
    """
    from scipy.ndimage import uniform_filter1d, binary_dilation, label as ndlabel

    pos = img_fbp[img_fbp > 0]
    if pos.size == 0:
        return np.zeros_like(img_fbp, dtype=bool), 0.0

    if metal_threshold is None:
        p90, p999 = np.percentile(pos, [90, 99.9])
        if p999 <= p90 * 1.05:
            # No clear high-attenuation tail — no metal
            return np.zeros_like(img_fbp, dtype=bool), float(p999)
        bins    = np.linspace(p90, p999, 512)
        counts, edges = np.histogram(pos, bins=bins)
        smooth  = uniform_filter1d(counts.astype(float), size=15)
        valley  = int(np.argmin(smooth))
        metal_threshold = float(edges[valley])

    mask_raw = img_fbp > metal_threshold

    # Remove isolated noise pixels (components < 4 px are noise, not metal)
    labeled, n_comp = ndlabel(mask_raw)
    for cid in range(1, n_comp + 1):
        if (labeled == cid).sum() < 4:
            mask_raw[labeled == cid] = False

    if dilate_px > 0 and mask_raw.any():
        struct   = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
        mask_raw = binary_dilation(mask_raw, structure=struct)

    return mask_raw, metal_threshold


# ─────────────────────────────────────────────────────────────────────
def apply_mar(sino_preprocessed, angles_rad, geom, n_pixels,
              strength=0.4, metal_threshold=None, geometry_model='curved'):
    """
    Soft Metal Artifact Reduction (MAR) for fan-beam CT.

    Designed specifically for scans where the metal objects must be preserved
    for subsequent material decomposition.  The key parameter is `strength`:

        strength = 0.0  →  pass-through, no change
        strength = 0.3  →  soft:  streaks visibly reduced, ~70% of metal
                            signal perturbation kept in the sinogram — good
                            starting point for material decomposition
        strength = 0.5  →  moderate:  clear streak suppression, ~50% metal
                            signal kept — recommended default
        strength = 1.0  →  full projection completion:  maximum streak
                            suppression, metal sinogram bins fully replaced
                            by interpolation — metal attenuation values in
                            the reconstruction will be underestimated

    Algorithm
    ─────────
    1. First-pass FBP of the preprocessed sinogram.
    2. Threshold-based metal segmentation (auto or manual) with morphological
       dilation to capture partial-volume edge voxels.
    3. Forward-project the binary metal mask → metal sinogram trace M.
    4. For each projection, linearly interpolate the sinogram over the bins
       where M > 1% of peak → produces a metal-free reference sinogram.
    5. Blend in the correction:
           sino_out[M] = (1 - strength) × sino_orig[M]
                       +      strength  × sino_interpolated[M]
       Non-metal bins (M = 0) are NEVER modified.

    The blend preserves (1 − strength) of the original metal signal, so the
    reconstructed metal region is not zeroed out — it just has reduced
    streak-inducing inconsistency injected.

    Parameters
    ----------
    sino_preprocessed : ndarray [n_proj, n_ch]  preprocessed, COR-corrected
    angles_rad        : ndarray [n_proj]
    geom              : dict from build_geom()
    n_pixels          : int  reconstruction grid size
    strength          : float  blend factor 0.0–1.0  (default 0.4)
    metal_threshold   : float | None  image-domain threshold; None = auto

    Returns
    -------
    sino_corrected : ndarray [n_proj, n_ch]  MAR-corrected sinogram
    metal_mask     : bool ndarray [n_pixels, n_pixels]  metal segmentation
                     (None if strength == 0 or no metal found)
    """
    if strength <= 0.0:
        return sino_preprocessed, None

    # 1. First-pass FBP
    img_fbp     = np.clip(_astra_fbp(sino_preprocessed, angles_rad, geom, n_pixels,
                                     geometry_model=geometry_model),
                          0.0, None)

    # 2. Metal segmentation
    metal_mask, thr_used = segment_metal_from_fbp(img_fbp, metal_threshold,
                                                   dilate_px=2)
    if not metal_mask.any():
        return sino_preprocessed, metal_mask

    # 3. Forward-project metal mask → binary sinogram trace
    metal_fp        = _astra_forward_project(metal_mask.astype(float),
                                              angles_rad, geom, n_pixels,
                                              geometry_model=geometry_model)
    metal_sino_mask = metal_fp > 0.01 * metal_fp.max()

    if not metal_sino_mask.any():
        return sino_preprocessed, metal_mask

    # 4. Inpaint metal bins by linear interpolation along channels per projection
    ch_idx        = np.arange(sino_preprocessed.shape[1], dtype=float)
    sino_inpainted = sino_preprocessed.copy()
    for k in range(sino_preprocessed.shape[0]):
        bad  = metal_sino_mask[k]
        good = ~bad
        if bad.any() and good.sum() >= 2:
            sino_inpainted[k, bad] = np.interp(
                ch_idx[bad], ch_idx[good], sino_preprocessed[k, good]
            )

    # 5. Soft blend — only in metal-contaminated bins
    sino_corrected = sino_preprocessed.copy()
    sino_corrected[metal_sino_mask] = (
        (1.0 - strength) * sino_preprocessed[metal_sino_mask]
        + strength       * sino_inpainted[metal_sino_mask]
    )

    return sino_corrected, metal_mask


# ─────────────────────────────────────────────────────────────────────
def reconstruct_helical_slice(sino_full, geom, z_target_mm,
                              method="astra", n_pixels=512, window="360",
                              mar_strength=0.0, mar_metal_threshold=None,
                              hampel_threshold=None,
                              filter_name='shepp-logan',
                              geometry_model='curved',
                              z_window=True, z_weighting=None,
                              wavelet_ring_threshold=2.0,
                              algorithm='fbp', n_iter=150):
    """
    Reconstruct one axial slice at z_target_mm from raw sinogram data.

    Parameters
    ----------
    mar_strength           : float 0–1  MAR blend factor (0 = off, default)
    mar_metal_threshold    : float | None  image-domain metal threshold (None=auto)
    hampel_threshold       : float | None  Hampel MAD multiplier (None=off, default)
                             See preprocess_sinogram() for guidance on values.
    filter_name            : str  ASTRA filter type (default 'shepp-logan').
                             Use 'ram-lak' for MTF/geometry verification.
    geometry_model         : 'curved' (default) | 'flat'
                             'curved' applies cosine pre-weight + cubic remap
                             from equiangular to equispaced flat-detector
                             sampling before FBP.
    z_window               : bool (default True)  LEGACY raised-cosine weight
                             switch; superseded by z_weighting.
    z_weighting            : None | 'balanced' | 'hann' | 'none'  projection-axis
                             weighting (default None → derived from z_window).
                             'balanced' removes the rotating low-frequency lobe;
                             see rebin_helical_to_axial for details.
    wavelet_ring_threshold : float (default 2.0)  ring-index gate for wavelet.
    algorithm              : 'fbp' (default) | 'sirt' | 'cgls'  reconstruction
                             algorithm (method must be 'astra').  See
                             _astra_reconstruct() for the trade-offs.
    n_iter                 : int (default 150)  iterations for sirt/cgls.
    """
    sino_axial, angles_rad = rebin_helical_to_axial(
        sino_full, geom, z_target_mm, window,
        z_window=z_window, z_weighting=z_weighting)
    sino_axial = preprocess_sinogram(sino_axial, geom,
                                      hampel_threshold=hampel_threshold,
                                      wavelet_ring_threshold=wavelet_ring_threshold)
    sino_axial = apply_cor_shift(sino_axial, geom['det_alignment'])

    if mar_strength > 0.0:
        sino_axial, _ = apply_mar(sino_axial, angles_rad, geom, n_pixels,
                                   strength=mar_strength,
                                   metal_threshold=mar_metal_threshold,
                                   geometry_model=geometry_model)

    if method == "iradon":
        from skimage.transform import iradon
        return iradon(sino_axial.T, theta=np.degrees(angles_rad),
                      circle=True, filter_name='ramp')
    elif method == "astra":
        return _astra_reconstruct(sino_axial, angles_rad, geom, n_pixels,
                                  algorithm=algorithm, filter_name=filter_name,
                                  geometry_model=geometry_model, n_iter=n_iter)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def reconstruct_helical_stack(sino_full, geom, z_targets_mm,
                              method="astra", n_pixels=512, window="360",
                              mar_strength=0.0, mar_metal_threshold=None,
                              hampel_threshold=None,
                              filter_name='shepp-logan',
                              geometry_model='curved',
                              z_window=True, z_weighting=None,
                              wavelet_ring_threshold=2.0,
                              algorithm='fbp', n_iter=150):
    """
    Reconstruct a stack of axial slices. Returns float32 [n_slices, ny, nx].

    When mar_strength > 0, each slice incurs one additional FBP pass for metal
    detection (roughly 2× reconstruction time).

    New parameters (all forwarded to reconstruct_helical_slice):
    filter_name            : 'shepp-logan' default (smooth, low-noise)
    geometry_model         : 'curved' default (correct NAEOTOM detector model)
    z_window               : True default (LEGACY raised-cosine weight switch)
    z_weighting            : None default → derived from z_window. 'balanced'
                             removes the rotating low-frequency lobe (see
                             rebin_helical_to_axial).
    wavelet_ring_threshold : 2.0 default (adaptive stripe removal gate)
    algorithm              : 'fbp' default | 'sirt' | 'cgls'.  NOTE: 'sirt'/'cgls'
                             multiply runtime by ~2·n_iter per slice — for a full
                             helical volume this is hours→days; use deliberately.
    n_iter                 : 150 default (iterations for sirt/cgls)
    """
    z_targets_mm = np.atleast_1d(np.asarray(z_targets_mm, dtype=float))
    vol = np.zeros((len(z_targets_mm), n_pixels, n_pixels), dtype=np.float32)
    for i, z in enumerate(z_targets_mm):
        if i % 100 == 0 or i == len(z_targets_mm) - 1:
            print(f"  [{i+1:>5d}/{len(z_targets_mm)}]  z = {z:+.2f} mm")
        vol[i] = reconstruct_helical_slice(
            sino_full, geom, z,
            method=method, n_pixels=n_pixels, window=window,
            mar_strength=mar_strength, mar_metal_threshold=mar_metal_threshold,
            hampel_threshold=hampel_threshold,
            filter_name=filter_name, geometry_model=geometry_model,
            z_window=z_window, z_weighting=z_weighting,
            wavelet_ring_threshold=wavelet_ring_threshold,
            algorithm=algorithm, n_iter=n_iter,
        )
    return vol


def reconstruct_slab(sino_full, geom, z_center_mm, slab_mm, z_spacing_mm,
                     **slice_kwargs):
    """
    Reconstruct an image-domain average of the slices spanning `slab_mm`,
    centred on `z_center_mm`.

    This mirrors exactly what z_average() does to a full volume, so a FAST-mode
    single-slice preview can be made to show the SNR (and effective slice
    thickness) the full volume will have — instead of the noisiest possible
    0.4 mm single slice.  The number of slices is derived from the geometry
    (slab_mm / z_spacing), never hardcoded; the count is forced odd so the slab
    is centred on z_center_mm.

    Parameters
    ----------
    sino_full    : float32 [N_proj, n_rows, n_channels]
    geom         : dict from build_geom()
    z_center_mm  : float  centre of the slab (mm)
    slab_mm      : float  effective slab thickness (mm).  <= one slice → single
                   slice (identical to reconstruct_helical_slice).
    z_spacing_mm : float  slice spacing (mm), from z_targets_for_full_scan().
    slice_kwargs : forwarded to reconstruct_helical_slice (method, n_pixels,
                   filter_name, geometry_model, algorithm, n_iter, ...).

    Returns
    -------
    (img_mean [n_pixels, n_pixels], n_used int)  averaged slice + slice count.
    """
    dz = abs(z_spacing_mm)
    n_slices = 1 if (slab_mm <= dz or dz < 1e-9) else int(round(slab_mm / dz))
    if n_slices < 1:
        n_slices = 1
    if n_slices % 2 == 0:               # force odd → centred slab
        n_slices += 1
    n_half = n_slices // 2

    acc, n_used = None, 0
    for off in range(-n_half, n_half + 1):
        z = z_center_mm + off * dz
        try:
            img = reconstruct_helical_slice(sino_full, geom, z, **slice_kwargs)
        except ValueError:
            continue                    # z outside reconstructable range (edge)
        acc = img if acc is None else acc + img
        n_used += 1

    if n_used == 0:
        raise ValueError(
            f"reconstruct_slab: no valid slices in slab at z={z_center_mm:+.2f} mm "
            f"(slab_mm={slab_mm})"
        )
    return (acc / n_used).astype(np.float32), n_used


def z_targets_for_full_scan(geom, oversample=1, end_margin_rotations=0.5):
    """
    Return z_targets_mm covering the full reconstructable scan range,
    with spacing derived entirely from geom — nothing hardcoded.

    oversample=1: one slice per collimated row width (Nyquist, default)
    oversample=2: half-step (2× finer, 2× slower)

    end_margin_rotations: rotations of focal-spot travel trimmed at each end so
      every slice has a full rebinning window.  0.5 (default) reproduces the
      historical P//2 trim — correct for the one-rotation 'hann'/'none' window.
      Pass n_rot_balanced/2 (e.g. 1.0) for the wider 'balanced' window, otherwise
      the end slices raise out-of-range in rebin_helical_to_axial.
    """
    P       = geom['proj_per_rotation']
    z_focus = geom['z_focus_per_proj']
    m       = max(1, int(round(end_margin_rotations * P)))
    z_start = z_focus[m]
    z_end   = z_focus[-m]
    z_step  = np.sign(z_end - z_start) * geom['z_spacing_mm'] / oversample
    return np.arange(z_start, z_end, z_step), geom['z_spacing_mm'] / oversample


# ─────────────────────────────────────────────────────────────────────
def auto_hu_calibrate(vol_native, fov_mm=500.0, erode_mm=10.0,
                      erode_fallback_mm=5.0, max_slices=50):
    """
    Automatically estimate (mu_water, mu_air) for HU calibration without
    any operator-specified ROI.

    Suitable for any scan whose dominant interior material is water-equivalent
    (both QRM phantoms and real patients).  Works per-threshold by calling
    this function on each threshold's reconstructed volume.

    Performance note
    ────────────────
    All heavy processing (Otsu, connected components, erosion) runs on a
    subsample of max_slices (default 50) evenly spaced slices rather than
    the full volume.  mu_water and mu_air are statistics that converge on
    far less data — 50 × 512 × 512 ≈ 13M voxels gives the same result as
    the full 2262-slice volume and runs in seconds rather than minutes.

    Algorithm
    ─────────
    1. Subsample to max_slices evenly spaced slices.
    2. Otsu threshold → binarise → per-slice 2D largest connected component
       → body mask.  (2D labelling per slice is far faster than 3D on a
       large volume and sufficient because the phantom cross-section is
       the same object in every slice.)
    3. Morphological erosion (10 mm default, 5 mm fallback) to stay clear
       of the boundary, inserts, and high-attenuation rings.
    4. Within the eroded body, discard the top and bottom 5th percentile
       (removes metal, air pockets, high-contrast inserts).
    5. mu_water = mode of remaining voxels (robust to inserts — the water-
       equivalent material is the dominant population).
    6. mu_air = median of outside-body voxels, excluding a gantry-shadow
       ring near the FOV edge (|r| > 0.95 × FOV/2).
    7. Returns a dict; apply HU = 1000*(vol - mu_water)/(mu_water - mu_air).

    Parameters
    ----------
    vol_native       : ndarray [n_slices, n_y, n_x]  scanner-native units
    fov_mm           : float  field of view diameter used in reconstruction
    erode_mm         : float  erosion radius in mm (default 10)
    erode_fallback_mm: float  erosion fallback radius if body becomes empty
    max_slices       : int    number of evenly spaced slices to subsample
                       (default 50; increase only if the phantom occupies
                       very few slices of the full volume)

    Returns
    -------
    calib : dict with keys
        mu_water     : float  estimated water attenuation (scanner units)
        mu_air       : float  estimated air attenuation (scanner units)
        n_water_vox  : int    voxels used for mu_water estimate (subsampled)
        n_air_vox    : int    voxels used for mu_air estimate (subsampled)
        warnings     : list[str]  any fallback conditions encountered
    """
    from scipy.ndimage import binary_erosion, label as ndlabel

    warnings_out = []

    # ── 0. Subsample to max_slices evenly spaced slices ───────────────
    n_sl, n_y, n_x = vol_native.shape
    step = max(1, n_sl // max_slices)
    vol_f = vol_native[::step].astype(np.float32)   # (≤max_slices, n_y, n_x)
    n_sub = vol_f.shape[0]
    print(f"[auto_hu_calibrate] subsampled {n_sl} → {n_sub} slices (step={step})")

    px_mm = fov_mm / n_x

    # ── 1. Otsu threshold (on subsampled histogram) ───────────────────
    finite_flat = vol_f.ravel()
    finite_flat = finite_flat[np.isfinite(finite_flat)]
    lo = float(np.percentile(finite_flat, 1))
    hi = float(np.percentile(finite_flat, 99))
    counts, edges = np.histogram(finite_flat, bins=256,
                                  range=(lo, hi))
    total  = counts.sum()
    cum    = np.cumsum(counts)
    cumw   = np.cumsum(counts * (edges[:-1] + edges[1:]) * 0.5)
    mu_t   = cumw[-1] / (total + 1e-9)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma2_b = np.where(
            (cum > 0) & (cum < total),
            (mu_t * cum - cumw) ** 2 / (cum * (total - cum) + 1e-9),
            0.0,
        )
    otsu_thr = float(edges[np.argmax(sigma2_b)])

    # ── 2. Per-slice 2D body mask (largest component each slice) ──────
    # 2D labelling per slice is O(n_y × n_x) × n_sub — vastly faster than
    # one 3D label call on the full volume.
    body_mask = np.zeros(vol_f.shape, dtype=bool)
    disk_r    = max(1, int(round(erode_mm / px_mm)))
    yy, xx    = np.ogrid[-disk_r:disk_r+1, -disk_r:disk_r+1]
    disk      = (yy**2 + xx**2) <= disk_r**2

    disk_fb_r = max(1, int(round(erode_fallback_mm / px_mm)))
    yy_fb, xx_fb = np.ogrid[-disk_fb_r:disk_fb_r+1, -disk_fb_r:disk_fb_r+1]
    disk_fb   = (yy_fb**2 + xx_fb**2) <= disk_fb_r**2

    eroded    = np.zeros(vol_f.shape, dtype=bool)
    fallback_triggered = False

    for s in range(n_sub):
        sl      = vol_f[s]
        bin_sl  = sl > otsu_thr
        labeled, n_comp = ndlabel(bin_sl)
        if n_comp == 0:
            continue
        # Largest component via bincount (O(n_pixels), no per-component loop)
        sizes     = np.bincount(labeled.ravel())
        sizes[0]  = 0          # exclude background label 0
        body_mask[s] = (labeled == sizes.argmax())

        # Erosion for this slice
        er = binary_erosion(body_mask[s], structure=disk)
        if er.sum() < 20 and not fallback_triggered:
            er = binary_erosion(body_mask[s], structure=disk_fb)
            fallback_triggered = True
        eroded[s] = er

    if fallback_triggered:
        warnings_out.append(
            f"Erosion at {erode_mm} mm left <20 px/slice; "
            f"fell back to {erode_fallback_mm} mm for affected slices"
        )

    if body_mask.sum() == 0:
        warnings_out.append("No object found above Otsu threshold — using global median")
        mu_water = float(np.median(finite_flat))
        mu_air   = float(lo)
        return dict(mu_water=mu_water, mu_air=mu_air,
                    n_water_vox=0, n_air_vox=0, warnings=warnings_out)

    # ── 3. mu_water from eroded interior (trimmed 5–95%) ─────────────
    water_vals = vol_f[eroded]
    if len(water_vals) < 10:
        warnings_out.append("Too few voxels in eroded body — mu_water estimate unreliable")
        mu_water = float(np.median(finite_flat[finite_flat > otsu_thr]))
    else:
        p05, p95 = np.percentile(water_vals, [5, 95])
        water_inner = water_vals[(water_vals >= p05) & (water_vals <= p95)]
        if len(water_inner) < 10:
            water_inner = water_vals
        counts_w, edges_w = np.histogram(water_inner, bins=64)
        mu_water = float((edges_w[np.argmax(counts_w)] +
                          edges_w[np.argmax(counts_w) + 1]) * 0.5)

    # ── 4. mu_air from outside-body inside-FOV region ────────────────
    ys, xs = np.mgrid[0:n_y, 0:n_x]
    r_frac  = np.sqrt((ys - n_y/2)**2 + (xs - n_x/2)**2) / (n_x / 2)
    inside_fov = (r_frac <= 0.95)[np.newaxis, :, :]   # broadcast over slices
    outside_body = (~body_mask) & inside_fov
    air_vals = vol_f[outside_body]
    if len(air_vals) < 10:
        mu_air = float(lo)
        warnings_out.append("No air pixels found outside body; used p1 of volume as mu_air")
    else:
        mu_air = float(np.median(air_vals))

    print(f"[auto_hu_calibrate] mu_water={mu_water:.2f}  mu_air={mu_air:.2f}  "
          f"n_water={eroded.sum()}  n_air={outside_body.sum()}  "
          f"(from {n_sub} sampled slices)")
    for w in warnings_out:
        print(f"  [calibration warn] {w}")

    return dict(
        mu_water    = float(mu_water),
        mu_air      = float(mu_air),
        n_water_vox = int(eroded.sum()),
        n_air_vox   = int(outside_body.sum()),
        warnings    = warnings_out,
    )


def to_output_frame(vol, z_targets_mm, fov_mm, n_pixels,
                    mirror_x=True, table_height_mm=0.0):
    """
    Put a reconstructed stack into the vendor's frame before it is written.

    ONE place, used by both the production driver and the comparison sweep, so the two
    cannot drift apart and nothing downstream has to patch the output a second time.
    Three corrections, all output-space -- the reconstruction itself is untouched:

    1. z ORDER.  z_targets_for_full_scan walks the helix in the direction the table
       actually moved, so on a scan whose z decreases, slice 0 is the HIGHEST z.  Writing
       that with a positive spacing and an identity direction claims the opposite, and
       every consumer that converts mm -> slice then lands hundreds of slices away.  The
       array is flipped so the written volume always ascends in z.

    2. LEFT-RIGHT MIRROR.  Our reconstruction comes out mirrored relative to Siemens
       (established by matching the body centroid and the phantom's principal
       axis needs x -> -x, 0.30 mm and 0.10 deg residual).  Invariant #2 (README) requires
       this to be fixed in the OUTPUT and never by un-flipping detector channels, which
       would break the helical handedness.

    3. TABLE-REFERENCED ORIGIN.  Siemens references the patient frame to the table, not the
       isocentre (measured: 88.5 mm, exactly the recorded TableHeight).  Pass that as
       table_height_mm to write in the vendor's frame; 0.0 keeps our isocentre frame.

    Returns (array, origin_xyz, z_spacing) ready for SimpleITK.
    """
    arr = np.asarray(vol)
    z = np.asarray(z_targets_mm, dtype=float)
    if z.size != arr.shape[0]:
        raise ValueError(f"to_output_frame: {arr.shape[0]} slices but {z.size} z targets")

    if z.size > 1 and z[-1] < z[0]:
        arr = arr[::-1]
        z = z[::-1]
    if mirror_x:
        arr = arr[:, :, ::-1]
    arr = np.ascontiguousarray(arr)

    xy = float(fov_mm) / int(n_pixels)
    dz = float(abs(z[1] - z[0])) if z.size > 1 else 1.0
    origin = (-fov_mm / 2 + xy / 2,
              -fov_mm / 2 + xy / 2 - float(table_height_mm),
              float(z[0]))
    return arr, origin, dz


def z_average(vol, z_smooth_mm, z_spacing_mm):
    """
    Smooth a reconstructed volume along the z-axis to boost SNR.

    Applies a Gaussian kernel along the slice (z) axis with FWHM equal to
    z_smooth_mm.  For low-contrast insert visibility in a small phantom, an
    effective slice thickness of 1.5–3 mm typically gives 2–3× SNR over the
    native 0.4 mm slices at modest in-slice spatial-resolution cost.

    Parameters
    ----------
    vol           : ndarray [n_slices, n_y, n_x]
    z_smooth_mm   : float  FWHM of the z-direction Gaussian (mm).
                    0 disables smoothing (pass-through).
    z_spacing_mm  : float  z spacing between slices (mm).

    Returns
    -------
    vol_smooth : ndarray same shape as vol, same dtype
    """
    if z_smooth_mm <= 0:
        return vol
    from scipy.ndimage import gaussian_filter1d
    sigma_px = (z_smooth_mm / max(abs(z_spacing_mm), 1e-6)) / 2.355   # FWHM → σ
    if sigma_px < 0.3:
        return vol
    print(f"[z_average] Applying z-Gaussian: FWHM={z_smooth_mm:.2f} mm "
          f"→ σ={sigma_px:.2f} slices")
    return gaussian_filter1d(vol, sigma=sigma_px, axis=0,
                              mode='nearest').astype(vol.dtype)


def apply_hu_calibration(vol_native, calib):
    """
    Convert a native-unit volume to Hounsfield Units using a calibration dict
    produced by auto_hu_calibrate().

    HU = 1000 × (vol − mu_water) / (mu_water − mu_air)

    Returns float32 [n_slices, n_y, n_x] in HU.
    """
    mu_w = calib['mu_water']
    mu_a = calib['mu_air']
    denom = mu_w - mu_a
    if abs(denom) < 1e-6:
        raise ValueError(
            f"HU calibration denominator near zero "
            f"(mu_water={mu_w:.4f}, mu_air={mu_a:.4f}) — "
            f"auto_hu_calibrate may have failed"
        )
    return ((vol_native.astype(np.float32) - mu_w) / denom * 1000.0).astype(np.float32)