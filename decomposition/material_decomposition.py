"""
material_decomposition.py -- core image-domain material-decomposition library.

Solves the per-voxel linear model

        B = M x + n        ->        x = argmin (Mx - B)^T W (Mx - B) [+ prior]

  B  energy-channel linear attenuations (1/cm)
  M  material signatures, one column of <mu/rho> (cm^2/g) per basis material
  x  partial densities of the basis materials (g/cm^3)

Estimators (DecompConfig.estimator), simplest first:
  ols          plain least squares, x = (M^T M)^-1 M^T B
  wls          weighted least squares, W from the noise covariance measured on the data
  wls_denoise  wls, then an edge-preserving denoise of the maps
  wls_joint    penalised wls with an edge-preserving (TV) prior, solved iteratively

Compute functions take arrays, never paths.  decompose() is the single entry point, driven
by a serialisable DecompConfig with optional progress and cancel callbacks.  File I/O
(SimpleITK, imported lazily) is kept at the bottom of the file.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import material_library as mlib
from . import decomposition_modes as modes
from . import noise_estimation as ne
from . import denoising as dn

logger = logging.getLogger(__name__)

ProgressCB = Optional[Callable[[float, str], None]]
CancelCB = Optional[Callable[[], bool]]


# ============================================================================
# Config & result (serializable; GUI-ready)
# ============================================================================
@dataclass
class DecompConfig:
    """All knobs for a decomposition run. Serializable so a GUI can populate it."""
    mode: str = "phantom_ca_i"
    threshold_option: int = 1
    bin_domain: str = "exclusive"        # 'exclusive' (A-B,B-C,C-D,D) | 'cumulative' (A,B,C,D)

    # --- signature weighting: ONLY reaches the integrated cumulative windows ------
    # 'uniform' reproduces the published bin-averaged table and is the default for that
    # reason. 'spectrum' weights the integral by the photon fluence a counting detector
    # sees, which is a defensible alternative for [T_i, 140] -- windows the table does not
    # tabulate and this code integrates itself. Disjoint and monoenergetic channels are
    # unaffected either way. See material_library.Weighting.
    signature_weighting: str = "uniform"     # 'uniform' | 'spectrum'
    spectrum_kvp: float = 140.0
    spectrum_water_cm: float = 16.0

    # --- estimator & noise model (best -> simplest; defaults = best quality) ---
    estimator: str = "wls_joint"         # 'wls_joint'|'wls_denoise'|'wls'|'ols'
    noise_model: str = "spatial"         # 'spatial'|'global'
    noise_window: int = 5                # box size (voxels) for residual/covariance estimation
    flat_percentile: float = 40.0        # flattest fraction used for noise measurement
    noise_sample_stride: int = 24        # z-stride for the global covariance sample (memory)

    # --- edge-preserving denoise (wls_denoise, and the prior inside wls_joint) ---
    denoise_method: str = "tv"           # 'tv'|'nlm'|'bilateral'|'guided'|'gaussian'
    denoise_scale: float = 1.0           # strength multiplier on the *measured* map noise
    denoise_guide: bool = False          # cross-channel guiding (guide = highest-SNR channel, runtime)
    joint_iters: int = 10                # iterations for wls_joint
    joint_beta: float = 1.0              # proximity weight scale (mu = beta * mean diag(M^T W M))

    # --- HU->attenuation & unit calibration ---
    hu_input: bool = True                # inputs are HU volumes (else already linear attenuation)
    water_calibration: bool = True       # per-bin gain so a water ROI matches NIST water (unit scaling)
    water_hu_tol: float = 60.0           # |HU| < tol on threshold A defines the water ROI
    compute_residual: bool = True        # also return the per-voxel LS fit-residual norm
    z_chunk: int = 16                    # slices solved per chunk (memory bound)

    # --- I/O (used by the driver; the library decompose() takes arrays) ----
    input_format: str = "nifti"          # 'nifti' (our own recon) | 'dicom' (Siemens series)
    input_source: str = ""               # provenance label for outputs/reliability: 'own'|'wfbp'|'vmi'
    input_dir: Optional[str] = None
    input_pattern: str = "reconstruction_thr_{label}_HU.nii.gz"
    threshold_labels: Tuple[str, ...] = ("A", "B", "C", "D")
    input_paths: Optional[List[str]] = None
    calibration_pattern: str = "calibration_thr_{label}.json"
    output_dir: Optional[str] = None
    z_slab_mm: Optional[Tuple[float, float]] = None

    # --- DICOM input (input_format='dicom'): Siemens series live flat in one folder ----
    dicom_dir: Optional[str] = None      # the flat folder of DICOMs (defaults to input_dir)
    # ordered SeriesNumber (int) OR SeriesInstanceUID (str), one per threshold_label (A..D).
    # Identify which series are the 4 thresholds via inspect_dicom.py --dump-series first.
    dicom_series: Optional[List] = None

    def weighting(self) -> "mlib.Weighting":
        """The window weighting as the library object, built from the serialisable fields."""
        return mlib.parse_weighting(self.signature_weighting, self.spectrum_kvp,
                                    self.spectrum_water_cm)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecompConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if kw.get("threshold_labels") is not None:
            kw["threshold_labels"] = tuple(kw["threshold_labels"])
        if kw.get("z_slab_mm") is not None:
            kw["z_slab_mm"] = tuple(kw["z_slab_mm"])
        return cls(**kw)


@dataclass
class DecompResult:
    materials: List[str]
    material_maps: Dict[str, np.ndarray]   # name -> density map (g/cm^3), shape (Z,Y,X)
    M: np.ndarray                          # (n_bins, n_materials)
    stability: dict
    metadata: dict
    config: DecompConfig
    residual: Optional[np.ndarray] = None  # per-voxel LS residual norm, shape (Z,Y,X)


# ============================================================================
# Pure math -- forward model, bins, estimators
# ============================================================================
def cumulative_to_exclusive(vols: np.ndarray) -> np.ndarray:
    """
    Cumulative thresholds A>=20, B>=40, C>=56, D>=75 (axis 0, order A..D)
    -> exclusive energy bins [20-40, 40-56, 56-75, 75-140] = [A-B, B-C, C-D, D].
    Image-domain differencing only (invariant #3, README).
    """
    vols = np.asarray(vols)
    if vols.shape[0] != 4:
        raise ValueError(f"Expected 4 cumulative volumes on axis 0, got shape {vols.shape}")
    a, b, c, d = vols
    return np.stack([a - b, b - c, c - d, d], axis=0)


def hu_to_linear_attenuation(hu: np.ndarray, mu_water: float) -> np.ndarray:
    """HU -> linear attenuation (1/cm): mu = mu_water * (1 + HU/1000)."""
    return mu_water * (1.0 + np.asarray(hu, dtype=np.float64) / 1000.0)


def form_bin_measurements(vols_mu: np.ndarray, bin_domain: str) -> np.ndarray:
    """
    Per-channel linear attenuation (N, ...) -> measurement stack B (N, ...).
      'exclusive'  : subtract adjacent cumulative thresholds (A-B, ...) -- 4-threshold data only.
      'cumulative' : feed the thresholds as-is.
      'direct'     : feed the channels as-is (monoenergetic VMI, or any non-subtracted input).
    """
    if bin_domain == "exclusive":
        return cumulative_to_exclusive(vols_mu)
    if bin_domain in ("cumulative", "direct"):
        return np.asarray(vols_mu)
    raise ValueError(f"bin_domain must be 'exclusive'|'cumulative'|'direct', got '{bin_domain}'")


def solve_ols(B: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Ordinary least squares x = (M^T M)^-1 M^T B for every column of B at once.
    B (n_bins, n_vox), M (n_bins, n_materials) -> X (n_materials, n_vox).
    Uses the pseudoinverse (== (M^T M)^-1 M^T for full-rank tall M) for numerical stability.
    """
    return np.linalg.pinv(M) @ B


def _safe_inv(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def wls_operator(M: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Global-W WLS linear operator L = (M^T W M)^-1 M^T W, so x = L B."""
    MtW = M.T @ W
    return _safe_inv(MtW @ M) @ MtW


def solve_wls(B: np.ndarray, M: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted least squares with a single (global) weight matrix W = Sigma^-1."""
    return wls_operator(M, W) @ B


def voxel_wls(Bc: np.ndarray, M: np.ndarray, Wfield: np.ndarray,
              mu: float = 0.0, Zprox: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Per-voxel (optionally proximity-regularised) WLS.
      Bc     : (n_bins, n_vox)
      Wfield : (n_vox, n_bins, n_bins)   per-voxel inverse-covariance weights
    Solves (M^T W M + mu I) x = M^T W b + mu z  per voxel; returns (n_mat, n_vox).
    """
    Mt = M.T
    MtW = np.einsum("ib,vbc->vic", Mt, Wfield)        # (v, n_mat, n_bins)
    A = np.einsum("vic,cj->vij", MtW, M)              # (v, n_mat, n_mat)
    rhs = np.einsum("vic,cv->vi", MtW, Bc)            # (v, n_mat)
    if mu > 0:
        A = A + mu * np.eye(M.shape[1])
        if Zprox is not None:
            rhs = rhs + mu * Zprox.T
    X = np.linalg.solve(A, rhs[..., None])[..., 0]    # (v, n_mat)
    return X.T


# ---- stability / conditioning (M-only) ------------------------------------
def column_cosines(M: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of M's columns (1.0 => collinear => singular)."""
    norm = M / np.linalg.norm(M, axis=0, keepdims=True)
    return norm.T @ norm


def stability_verdict(kappa: float) -> str:
    if kappa < 10:
        return "excellent"
    if kappa < 50:
        return "good"
    if kappa < 200:
        return "usable"
    if kappa < 1000:
        return "poor"
    return "unstable"


def stability_report(M: np.ndarray, materials: Sequence[str]) -> dict:
    """Conditioning / noise diagnostics of the material matrix (depends only on M)."""
    materials = list(materials)
    s = np.linalg.svd(M, compute_uv=False)
    kappa = float(s[0] / s[-1])
    Mn = M / np.linalg.norm(M, axis=0, keepdims=True)
    sn = np.linalg.svd(Mn, compute_uv=False)
    inv = _safe_inv(M.T @ M)
    noise_gain = np.sqrt(np.clip(np.diag(inv), 0, None))
    return {
        "materials": materials,
        "condition_number": kappa,
        "condition_number_MtM": float(kappa ** 2),
        "condition_number_colnorm": float(sn[0] / sn[-1]),
        "singular_values": s.tolist(),
        "noise_amplification": {m: float(g) for m, g in zip(materials, noise_gain)},
        "column_cosines": column_cosines(M).tolist(),
        "verdict": stability_verdict(kappa),
    }


def reliability_report(M: np.ndarray, materials: Sequence[str]) -> dict:
    """
    Per-material reliability of the operator the solver actually inverts (M, or the effective
    diag(1/gains)@M). Flags which material(s) are likely DEGENERATE -- amplified by
    ill-conditioning / sitting in the near-null direction -- so a noise-dominated map is
    recognised as expected, not a bug. Generic: for VMIs (effective rank ~2, 3 materials) it
    auto-flags the under-determined material.
    """
    materials = list(materials)
    s = np.linalg.svd(M, compute_uv=False)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    kappa = float(s[0] / s[-1]) if s[-1] > 1e-30 else float("inf")
    amp = np.sqrt(np.clip(np.diag(_safe_inv(M.T @ M)), 0, None))
    eff_rank = int(np.sum(s > 1e-2 * s[0]))
    null_dom = materials[int(np.argmax(np.abs(Vt[-1])))]

    def _v(a: float) -> str:
        return "RELIABLE" if a < 1.0 else "OK" if a < 3.0 else "POOR" if a < 10.0 else "DEGENERATE"

    per = {m: {"noise_amplification": float(a), "verdict": _v(a)} for m, a in zip(materials, amp)}
    degenerate = [m for m, a in zip(materials, amp) if a >= 3.0]
    if not degenerate and eff_rank < len(materials):      # rank-deficient -> flag the worst
        degenerate = [materials[int(np.argmax(amp))]]
    return {
        "condition_number": kappa,
        "effective_rank": eff_rank,
        "n_materials": len(materials),
        "singular_values": s.tolist(),
        "per_material": per,
        "null_dominant_material": null_dom,
        "likely_degenerate": degenerate,
    }


def _print_reliability(rel: dict, mode: str, source: str, n_channels: int) -> None:
    """Prominent, greppable reliability block -> stdout (captured in the SLURM .out)."""
    L = ["=" * 62,
         f"[DECOMP RELIABILITY]  mode={mode}  source={source or '?'}  channels={n_channels}",
         f"  kappa(M_eff)={rel['condition_number']:.3g}   effective rank ~{rel['effective_rank']}"
         f" of {rel['n_materials']} materials",
         "  per-material noise amplification (higher = less reliable):"]
    for m, d in rel["per_material"].items():
        flag = ("   *** LIKELY DEGENERATE -- treat this map as unreliable, not a pipeline bug ***"
                if m in rel["likely_degenerate"] else "")
        L.append(f"     {m:16s} {d['noise_amplification']:9.3f}   {d['verdict']}{flag}")
    if rel["likely_degenerate"]:
        L.append(f"  => expect {', '.join(rel['likely_degenerate'])} to look noise-dominated for "
                 f"spectral (conditioning) reasons, not a bug.")
    L.append("=" * 62)
    print("\n".join(L))


# ============================================================================
# Water calibration (unit scaling; not a stability remedy, so estimator-agnostic)
# ============================================================================
def _water_reference(bin_domain: str, option: int, mu_water: Optional[np.ndarray]) -> np.ndarray:
    """Expected water linear attenuation (1/cm) per channel (density 1.0 g/cm^3)."""
    if bin_domain == "exclusive":
        return mlib.material_signature("Water", option) * mlib.material_info("Water").density_g_cm3
    if mu_water is not None:                       # cumulative/direct: per-channel physical water mu
        return np.asarray(mu_water, dtype=float)
    return mlib.material_signature("Water", option) * mlib.material_info("Water").density_g_cm3


def estimate_water_gains(B: np.ndarray, water_mask: np.ndarray,
                         reference: np.ndarray) -> np.ndarray:
    """Per-bin gain so median(B in water ROI) matches the NIST water reference."""
    gains = np.ones(B.shape[0], dtype=float)
    flat = B.reshape(B.shape[0], -1)
    m = water_mask.reshape(-1).astype(bool)
    if m.sum() < 10:
        warnings.warn("Water ROI too small for calibration; skipping (gains=1).")
        return gains
    for i in range(B.shape[0]):
        med = np.median(flat[i, m])
        if abs(med) > 1e-9:
            gains[i] = reference[i] / med
    return gains


# ============================================================================
# Estimator drivers (chunked over Z; data-adaptive weights)
# ============================================================================
def estimate_global_W(vols_mu: np.ndarray, gains: np.ndarray, config: DecompConfig) -> np.ndarray:
    """Sample strided slices, form the bins, estimate Sigma, return W = Sigma^-1 (one matrix)."""
    Z = vols_mu.shape[1]
    stride = max(1, int(config.noise_sample_stride))
    zc = np.arange(0, Z, stride)
    if zc.size < 3:
        zc = np.arange(Z)
    Bs = form_bin_measurements(vols_mu[:, zc], config.bin_domain) * gains[:, None, None, None]
    Sigma = ne.noise_covariance_global(Bs, config.noise_window, config.flat_percentile)
    return _safe_inv(Sigma)


def _shape(vols_mu):
    return vols_mu.shape[0], vols_mu.shape[1], vols_mu.shape[2], vols_mu.shape[3]


def _solve_linear_chunked(vols_mu, gains, M, L, config, n_mat, _p, cancel):
    """x = L @ B over Z-chunks (OLS or global-W WLS). Returns (maps, residual)."""
    n_bins, Z, Y, X = _shape(vols_mu)
    maps = np.zeros((n_mat, Z, Y, X), dtype=np.float32)
    residual = np.zeros((Z, Y, X), dtype=np.float32) if config.compute_residual else None
    chunk = max(1, int(config.z_chunk))
    for z0 in range(0, Z, chunk):
        if cancel and cancel():
            warnings.warn("decompose() cancelled; returning partial maps."); break
        z1 = min(z0 + chunk, Z)
        B = form_bin_measurements(vols_mu[:, z0:z1], config.bin_domain)
        B = (B * gains[:, None, None, None]).reshape(n_bins, -1)
        Xsol = L @ B
        maps[:, z0:z1] = Xsol.reshape(n_mat, z1 - z0, Y, X).astype(np.float32)
        if residual is not None:
            residual[z0:z1] = np.linalg.norm(M @ Xsol - B, axis=0).reshape(z1 - z0, Y, X)
        _p(0.10 + 0.80 * (z1 / Z), f"solved z {z1}/{Z}")
    return maps, residual


def _covariance_field_for_chunk(vols_mu, gains, config, z0, z1, n_bins):
    """Local inverse-covariance weights for a Z-chunk, using a halo to avoid boundary bias."""
    Z = vols_mu.shape[1]
    h = max(1, int(config.noise_window))
    a, b = max(0, z0 - h), min(Z, z1 + h)
    Bh = form_bin_measurements(vols_mu[:, a:b], config.bin_domain) * gains[:, None, None, None]
    field = ne.noise_covariance_field(Bh, config.noise_window)          # (nz_h,Y,X,nb,nb)
    sl = slice(z0 - a, z0 - a + (z1 - z0))
    Wf = np.linalg.inv(field[sl].reshape(-1, n_bins, n_bins))
    Bc = Bh[:, sl].reshape(n_bins, -1)
    return Bc, Wf


def _solve_spatial_chunked(vols_mu, gains, M, config, n_mat, _p, cancel):
    """Spatially-varying-W WLS: per-chunk local covariance field + per-voxel solve."""
    n_bins, Z, Y, X = _shape(vols_mu)
    maps = np.zeros((n_mat, Z, Y, X), dtype=np.float32)
    residual = np.zeros((Z, Y, X), dtype=np.float32) if config.compute_residual else None
    chunk = max(1, int(config.z_chunk))
    for z0 in range(0, Z, chunk):
        if cancel and cancel():
            warnings.warn("decompose() cancelled; returning partial maps."); break
        z1 = min(z0 + chunk, Z)
        Bc, Wf = _covariance_field_for_chunk(vols_mu, gains, config, z0, z1, n_bins)
        Xsol = voxel_wls(Bc, M, Wf)
        maps[:, z0:z1] = Xsol.reshape(n_mat, z1 - z0, Y, X).astype(np.float32)
        if residual is not None:
            residual[z0:z1] = np.linalg.norm(M @ Xsol - Bc, axis=0).reshape(z1 - z0, Y, X)
        _p(0.10 + 0.80 * (z1 / Z), f"solved z {z1}/{Z} (spatial noise)")
    return maps, residual


def _denoise_stack(maps: np.ndarray, config: DecompConfig) -> np.ndarray:
    """Edge-preserving denoise each material map, strength from its own measured noise."""
    sigmas = [ne.estimate_map_noise(maps[i], config.noise_window, config.flat_percentile)
              for i in range(maps.shape[0])]
    guide = dn.select_guide_index(list(maps), sigmas) if config.denoise_guide else None
    den = dn.denoise_maps(list(maps), sigmas, config.denoise_method, config.denoise_scale, guide)
    return np.stack(den).astype(np.float32)


def _solve_joint(vols_mu, gains, M, config, n_mat, _p, cancel):
    """
    Penalised WLS via half-quadratic splitting (plug-and-play):
      repeat:  z <- edge-preserving-denoise(x)   [TV prox, strength from measured noise]
               x <- argmin (Mx-B)^T W (Mx-B) + mu ||x - z||^2
    W is the data-adaptive noise weighting (global or spatial). Best quality; iterative.
    """
    n_bins, Z, Y, X = _shape(vols_mu)
    B = form_bin_measurements(vols_mu, config.bin_domain) * gains[:, None, None, None]
    Bflat = B.reshape(n_bins, -1)
    spatial = (config.noise_model == "spatial")
    tv = dn.DENOISERS["tv"]
    iters = max(1, int(config.joint_iters))

    if not spatial:
        W = estimate_global_W(vols_mu, gains, config)
        A0 = M.T @ W @ M
        mu = float(config.joint_beta) * float(np.mean(np.diag(A0)))
        invA = _safe_inv(A0 + mu * np.eye(n_mat))
        MtW = M.T @ W
        data_term = (invA @ MtW) @ Bflat                    # constant across iterations
        x = wls_operator(M, W) @ Bflat                      # WLS init
    else:
        maps0, _ = _solve_spatial_chunked(vols_mu, gains, M, config, n_mat,
                                          lambda *a: None, cancel)
        x = maps0.reshape(n_mat, -1)
        mu = float(config.joint_beta) * float(np.mean(np.diag(M.T @ M)))

    for it in range(iters):
        if cancel and cancel():
            warnings.warn("decompose() cancelled; returning partial maps."); break
        xvol = x.reshape(n_mat, Z, Y, X)
        z = np.empty_like(xvol)
        for c in range(n_mat):
            sig = ne.estimate_map_noise(xvol[c], config.noise_window, config.flat_percentile)
            z[c] = tv(xvol[c], sig, scale=config.denoise_scale)
        zflat = z.reshape(n_mat, -1)
        if not spatial:
            x = data_term + mu * (invA @ zflat)
        else:
            x = _joint_spatial_update(vols_mu, gains, M, config, n_mat, zflat, mu, cancel)
        _p(0.10 + 0.80 * ((it + 1) / iters), f"joint iter {it + 1}/{iters}")

    residual = None
    if config.compute_residual:
        residual = np.linalg.norm(M @ x - Bflat, axis=0).reshape(Z, Y, X).astype(np.float32)
    return x.reshape(n_mat, Z, Y, X).astype(np.float32), residual


def _joint_spatial_update(vols_mu, gains, M, config, n_mat, zflat, mu, cancel):
    """Data-update step of wls_joint under spatially-varying W (chunked, per-voxel)."""
    n_bins, Z, Y, X = _shape(vols_mu)
    out = np.zeros((n_mat, Z, Y, X), dtype=np.float64)
    zv = zflat.reshape(n_mat, Z, Y, X)
    chunk = max(1, int(config.z_chunk))
    for z0 in range(0, Z, chunk):
        if cancel and cancel():
            break
        z1 = min(z0 + chunk, Z)
        Bc, Wf = _covariance_field_for_chunk(vols_mu, gains, config, z0, z1, n_bins)
        Zc = zv[:, z0:z1].reshape(n_mat, -1)
        Xc = voxel_wls(Bc, M, Wf, mu=mu, Zprox=Zc)
        out[:, z0:z1] = Xc.reshape(n_mat, z1 - z0, Y, X)
    return out.reshape(n_mat, -1)


# ============================================================================
# Entry point
# ============================================================================
def decompose(volumes: np.ndarray, config: DecompConfig,
              mu_water: Optional[Sequence[float]] = None,
              water_mask: Optional[np.ndarray] = None,
              channels: Optional[Sequence] = None,
              progress: ProgressCB = None, cancel: CancelCB = None) -> DecompResult:
    """
    Decompose N reconstructed energy-channel volumes into per-material density maps.

    volumes  : (N, Z, Y, X) in HU (or linear attenuation if config.hu_input is False). N is the
               number of energy channels the data actually has -- NOT fixed at 4.
    channels : optional list of material_library.Channel describing each volume's energy (threshold
               window or monoenergetic VMI keV). If None, derived from config.threshold_option
               (back-compat: N = that option's threshold-bin count, e.g. 4).
    mu_water : optional per-channel water attenuation override for HU->mu; if None and hu_input, the
               physical per-channel water mu is generated from the channel energies.
    water_mask : optional bool (Z,Y,X) water ROI; auto-detected as |HU_channel0| < tol if None.
    """
    def _p(frac, msg):
        if progress:
            progress(float(frac), msg)

    volumes = np.asarray(volumes, dtype=np.float32)   # float32 storage; per-chunk math upcasts
    if volumes.ndim != 4:
        raise ValueError(f"volumes must be (N, Z, Y, X); got {volumes.shape}")
    N = int(volumes.shape[0])
    Z, Y, X = int(volumes.shape[1]), int(volumes.shape[2]), int(volumes.shape[3])

    spec = modes.get_mode(config.mode)
    materials = list(spec.materials)

    # Channels come from the loader (arbitrary count / kind); else derived from the threshold option.
    if channels is None:
        channels = mlib.threshold_channels(config.threshold_option)
    channels = list(channels)
    if len(channels) != N:
        raise ValueError(f"{N} volumes supplied but {len(channels)} channels described")

    # Monoenergetic (VMI) channels are not cumulative thresholds -> never subtract them.
    if any(c.kind == "mono" for c in channels) and config.bin_domain != "direct":
        if config.bin_domain == "exclusive":
            warnings.warn("bin_domain='exclusive' is meaningless for monoenergetic (VMI) channels "
                          "(not cumulative thresholds); coercing to 'direct'.")
        config = dataclasses.replace(config, bin_domain="direct")

    # M MUST describe the same quantity the measurement stack holds.  Fed as-is, a threshold
    # channel counts every photon ABOVE its threshold, so its signature is the average over
    # [T_i, E_max] -- not over the disjoint window the CSV tabulates, which is what the data
    # would hold only after differencing.  Done here, once, so no loader has to know.
    if config.bin_domain == "cumulative" and all(c.kind == "threshold" for c in channels):
        # The weighting is stamped on here and nowhere else, so the matrix and the water
        # attenuation used for HU -> mu are integrated the same way by construction.
        channels = mlib.as_cumulative_channels(channels, weighting=config.weighting())
        logger.info("cumulative bins: M built on the nested windows %s, %s weighted",
                    ", ".join(f"[{c.window_keV[0]:g},{c.window_keV[1]:g}]" for c in channels),
                    config.weighting().label)

    M = mlib.build_M(materials, channels)             # (N, n_mat); signatures generated per channel
    n_bins, n_mat = M.shape
    if n_bins != N:
        raise ValueError(f"M has {n_bins} rows but {N} volumes supplied")
    if n_bins < n_mat:
        raise ValueError(f"underdetermined: {n_bins} energy channels < {n_mat} materials "
                         f"({'/'.join(materials)}); provide >= {n_mat} channels or fewer materials")

    _p(0.02, f"mode '{config.mode}' [{'/'.join(materials)}] -- {n_bins} channels, stability audit")
    stability = stability_report(M, materials)
    logger.info("mode %s: %d channels, kappa=%.3g (%s)", config.mode, n_bins,
                stability["condition_number"], stability["verdict"])

    # HU -> linear attenuation (per cumulative threshold) --------------------
    # Filled one threshold at a time into a preallocated float32 buffer. This holds at most a
    # single float64 temporary (not a list of four + a float64 stack), so peak RAM stays bounded
    # on full volumes. The caller's `volumes` array is never mutated (the research harness reuses
    # it across estimators/domains), so we build a separate buffer rather than converting in place.
    if config.hu_input:
        if mu_water is None:
            mu_water = np.array([mlib.channel_water_mu(c) for c in channels], dtype=float)
        mu_water = np.asarray(mu_water, dtype=float)
        # Water ROI must be read in the HU domain, before the conversion.
        if config.water_calibration and water_mask is None:
            water_mask = np.abs(volumes[0]) < config.water_hu_tol
        vols_mu = np.empty_like(volumes)                      # float32
        for i in range(n_bins):
            vols_mu[i] = hu_to_linear_attenuation(volumes[i], mu_water[i])  # float64 calc -> f32
    else:
        vols_mu = volumes
        mu_water = None if mu_water is None else np.asarray(mu_water, dtype=float)
    volumes = None  # drop our reference to the HU input; vols_mu holds the attenuation

    gains = np.ones(n_bins)
    if config.water_calibration:
        if water_mask is None:
            warnings.warn("water_calibration on but no water ROI available; skipping.")
        else:
            B_probe = form_bin_measurements(vols_mu, config.bin_domain)
            reference = _water_reference(config.bin_domain, config.threshold_option, mu_water)
            gains = estimate_water_gains(B_probe, water_mask, reference)
            del B_probe

    # Reliability on the EFFECTIVE operator the solver inverts (M scaled by the per-channel gains).
    reliability = reliability_report(M / gains[:, None], materials)
    _print_reliability(reliability, config.mode, config.input_source, n_bins)

    # Estimator dispatch ------------------------------------------------------
    est = config.estimator
    _p(0.06, f"estimator '{est}', noise '{config.noise_model}', bin_domain '{config.bin_domain}'")
    if est == "ols":
        maps, residual = _solve_linear_chunked(vols_mu, gains, M, np.linalg.pinv(M),
                                               config, n_mat, _p, cancel)
    elif est in ("wls", "wls_denoise"):
        if config.noise_model == "global":
            W = estimate_global_W(vols_mu, gains, config)
            maps, residual = _solve_linear_chunked(vols_mu, gains, M, wls_operator(M, W),
                                                   config, n_mat, _p, cancel)
        else:
            maps, residual = _solve_spatial_chunked(vols_mu, gains, M, config, n_mat, _p, cancel)
    elif est == "wls_joint":
        maps, residual = _solve_joint(vols_mu, gains, M, config, n_mat, _p, cancel)
    else:
        raise ValueError(f"Unknown estimator '{est}' "
                         f"(use 'ols'|'wls'|'wls_denoise'|'wls_joint')")

    vols_mu = None  # free the attenuation volumes before any full-volume denoise
    if est == "wls_denoise":
        _p(0.92, f"edge-preserving denoise ({config.denoise_method})")
        maps = _denoise_stack(maps, config)

    material_maps = {m: maps[i] for i, m in enumerate(materials)}
    all_threshold = all(c.kind == "threshold" for c in channels)
    metadata = {
        "mode": config.mode,
        "display_name": spec.display_name,
        "clinical_question": spec.clinical_question,
        "materials": materials,
        "input_source": config.input_source,
        "n_channels": n_bins,
        "channels": [{"kind": c.kind, "label": c.label, "energy_keV": c.energy_keV,
                      "window_keV": (list(c.window_keV) if c.window_keV else None)}
                     for c in channels],
        "threshold_option": config.threshold_option if all_threshold else None,
        "bin_edges_keV": mlib.bin_edges(config.threshold_option) if all_threshold else None,
        "bin_domain": config.bin_domain,
        "estimator": config.estimator,
        "noise_model": config.noise_model,
        "denoise": ({"method": config.denoise_method, "scale": config.denoise_scale,
                     "guide": config.denoise_guide}
                    if est in ("wls_denoise", "wls_joint") else None),
        "mu_water_per_channel": None if mu_water is None else mu_water.tolist(),
        "water_calibration_gains": gains.tolist(),
        "reliability": reliability,
        "volume_shape": [int(Z), int(Y), int(X)],
    }
    _p(1.0, "done")
    return DecompResult(materials=materials, material_maps=material_maps, M=M,
                        stability=stability, metadata=metadata, config=config,
                        residual=residual)


def mode_stability(mode_key: str, option: int = 1) -> dict:
    """Stability report for a mode without any volumes (for the research/figures path)."""
    spec = modes.get_mode(mode_key)
    rep = stability_report(mlib.build_M(spec.materials, option), spec.materials)
    rep["mode"] = mode_key
    rep["display_name"] = spec.display_name
    return rep


# ============================================================================
# I/O layer (SimpleITK, lazy import -- kept separate from all compute above)
# ============================================================================
def _sitk():
    import SimpleITK as sitk
    return sitk


def _resolve_input_paths(config: DecompConfig) -> List[Path]:
    if config.input_paths:
        return [Path(p) for p in config.input_paths]
    if not config.input_dir:
        raise ValueError("config.input_dir or config.input_paths must be set")
    d = Path(config.input_dir)
    return [d / config.input_pattern.format(label=lbl) for lbl in config.threshold_labels]


def read_recon_calibration(config: DecompConfig) -> Optional[np.ndarray]:
    """Read per-threshold mu_water from the reconstruction calibration JSONs, if present."""
    if not config.input_dir:
        return None
    d = Path(config.input_dir)
    vals = []
    for lbl in config.threshold_labels:
        p = d / config.calibration_pattern.format(label=lbl)
        if not p.exists():
            return None
        with open(p) as f:
            cal = json.load(f)
        mw = cal.get("mu_water") or cal.get("mu_water_per_cm") or cal.get("muWater")
        if mw is None:
            return None
        vals.append(float(mw))
    return np.array(vals, dtype=float)


def _slab_slice(ref, z_slab_mm: Tuple[float, float]) -> Tuple[int, int]:
    """Map a physical z-range (mm) to numpy slice indices along axis 0 (slowest)."""
    nz = ref.GetSize()[2]
    z_of = [ref.TransformIndexToPhysicalPoint((0, 0, k))[2] for k in range(nz)]
    lo, hi = sorted(z_slab_mm)
    ks = [k for k, z in enumerate(z_of) if lo <= z <= hi]
    if not ks:
        raise ValueError(f"z_slab_mm {z_slab_mm} selects no slices "
                         f"(z range {min(z_of):.1f}..{max(z_of):.1f})")
    return min(ks), max(ks) + 1


def _is_dicom(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def build_dicom_index(folder: str, cache_path: Optional[str] = None) -> dict:
    """
    Index a flat folder of DICOMs -> {series_uid: {number, desc, files: [[z, path], ...] sorted}}.

    The Siemens export drops every series' files into one folder (survey: 6 series, one dir), so
    series are grouped by SeriesInstanceUID from the headers, not by path. Reading 11 k headers
    takes a couple of minutes; the result is cached to cache_path so repeated dev runs are instant.
    """
    sitk = _sitk()
    folder = str(folder)
    if cache_path and Path(cache_path).exists():
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if cached.get("folder") == folder:
                logger.info("using cached DICOM index %s", cache_path)
                return cached["series"]
        except (OSError, ValueError):
            pass

    series: Dict[str, dict] = {}
    n_seen = 0
    for root, _dirs, names in os.walk(folder):
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
            s = series.setdefault(uid, {"number": g("0020|0011"), "desc": g("0008|103e"),
                                        "files": []})
            s["files"].append([z, fp])
            n_seen += 1
            if n_seen % 2000 == 0:
                logger.info("  indexed %d DICOM headers ...", n_seen)
    for s in series.values():
        s["files"].sort(key=lambda zf: zf[0])
    logger.info("DICOM index: %d series, %d files under %s", len(series), n_seen, folder)
    if cache_path:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps({"folder": folder, "series": series}),
                                        encoding="utf-8")
        except OSError:
            pass
    return series


def _resolve_series(index: dict, selectors: Sequence) -> List[str]:
    """Map each selector (SeriesNumber int/str or SeriesInstanceUID) to a SeriesInstanceUID."""
    num_to_uid = {}
    for uid, s in index.items():
        num = s.get("number", "")
        if num:
            try:
                num_to_uid[str(int(float(num)))] = uid
            except ValueError:
                num_to_uid[num] = uid
    resolved = []
    for sel in selectors:
        key = str(sel).strip()
        if key in index:                     # already a UID
            resolved.append(key)
            continue
        try:
            key = str(int(float(key)))       # normalise "24"/"24.0"/24 -> "24"
        except ValueError:
            pass
        if key in num_to_uid:
            resolved.append(num_to_uid[key])
        else:
            raise ValueError(
                f"series selector {sel!r} not found. Available SeriesNumbers "
                f"{sorted(num_to_uid, key=lambda x: int(x) if x.isdigit() else 1 << 30)}")
    return resolved


def _slab_files(files: List[list], z_slab_mm: Optional[Tuple[float, float]]) -> List[str]:
    """Return the file paths whose z (mm) falls in z_slab_mm; all of them if no slab set."""
    paths = [f for _z, f in files]
    if not z_slab_mm:
        return paths
    lo, hi = sorted(z_slab_mm)
    keep = [f for z, f in files if lo <= z <= hi]
    if not keep:
        zs = [z for z, _f in files]
        raise ValueError(f"z_slab_mm {z_slab_mm} selects no slices "
                         f"(series z {min(zs):.1f}..{max(zs):.1f})")
    return keep


def load_threshold_volumes_dicom(config: DecompConfig):
    """
    Load the threshold volumes from a flat DICOM folder (Siemens export) -> the same
    (volumes (4,Z,Y,X) float32, reference sitk image, mu_water|None) contract as the NIfTI loader.

    config.dicom_series selects which series are the thresholds, in A..D order. RAM is bounded by
    reading only the z_slab_mm slices, converting to float32 per series, and freeing each series
    image before the next. mu_water is None (no recon calibration JSONs for Siemens data) -> the
    solver's water_calibration anchors the per-bin unit scale instead.
    """
    sitk = _sitk()
    folder = config.dicom_dir or config.input_dir
    if not folder or not config.dicom_series:
        raise ValueError("dicom input needs config.dicom_dir (or input_dir) and config.dicom_series "
                         "(ordered SeriesNumber/UID list, one per threshold label A..D)")
    if len(config.dicom_series) != len(config.threshold_labels):
        raise ValueError(f"dicom_series has {len(config.dicom_series)} entries but there are "
                         f"{len(config.threshold_labels)} threshold_labels")
    cache = None
    if config.output_dir:
        cache = str(Path(config.output_dir) / "dicom_index.json")
    index = build_dicom_index(folder, cache_path=cache)
    uids = _resolve_series(index, config.dicom_series)

    vols, ref = None, None
    for i, (lbl, uid) in enumerate(zip(config.threshold_labels, uids)):
        paths = _slab_files(index[uid]["files"], config.z_slab_mm)
        r = sitk.ImageSeriesReader()
        r.SetFileNames(paths)
        img = r.Execute()                                  # 3D, HU (rescale applied by GDCM)
        if ref is None:
            ref = img
        a = sitk.GetArrayFromImage(img).astype(np.float32)
        if img is not ref:
            del img
        if vols is None:                                   # preallocate (4,Z,Y,X); fill in place
            vols = np.empty((len(uids),) + a.shape, dtype=np.float32)
        elif a.shape != vols.shape[1:]:
            raise ValueError(f"Threshold series have mismatched shapes "
                             f"({a.shape} vs {vols.shape[1:]}) -- check dicom_series / z_slab_mm")
        vols[i] = a
        del a
        logger.info("threshold %s <- series #%s (%s) : %d slices",
                    lbl, index[uid]["number"], uid[-10:], vols.shape[1])
    return vols, ref, None


def load_threshold_volumes(config: DecompConfig):
    """
    Load the 4 threshold volumes -> (volumes (4,Z,Y,X) float32, reference sitk image, mu_water|None).
    Dispatches on config.input_format ('nifti' | 'dicom'). Applies config.z_slab_mm if set;
    the reference image carries the geometry used when saving results.
    """
    if getattr(config, "input_format", "nifti") == "dicom":
        return load_threshold_volumes_dicom(config)
    sitk = _sitk()
    paths = _resolve_input_paths(config)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing threshold volume(s):\n  " + "\n  ".join(missing))

    # Read one volume at a time into a preallocated (4,Z,Y,X) float32 buffer (avoids the list +
    # np.stack that briefly doubles peak RAM on full volumes). Slab-crop each before storing.
    img0 = sitk.ReadImage(str(paths[0]))
    ref = img0
    if config.z_slab_mm is not None:
        z0, z1 = _slab_slice(ref, config.z_slab_mm)
        logger.info("z_slab_mm %s -> slices [%d:%d]", config.z_slab_mm, z0, z1)
    else:
        z0, z1 = 0, ref.GetSize()[2]
    vols = None
    for i, p in enumerate(paths):
        img = img0 if i == 0 else sitk.ReadImage(str(p))
        a = sitk.GetArrayFromImage(img).astype(np.float32)[z0:z1]  # (Z,Y,X)
        if img is not img0:
            del img
        if vols is None:
            vols = np.empty((len(paths),) + a.shape, dtype=np.float32)
        elif a.shape != vols.shape[1:]:
            raise ValueError(f"Threshold volumes have mismatched shapes: {a.shape} vs {vols.shape[1:]}")
        vols[i] = a
        del a

    if config.z_slab_mm is not None:
        ref = ref[:, :, z0:z1]
    mu_water = read_recon_calibration(config)
    return vols, ref, mu_water


# ============================================================================
# N-channel energy-stack loaders -- one pipeline, three loaders (own / VMI / WFBP).
# Each returns (volumes (N,Z,Y,X) float32, channels: List[mlib.Channel], ref sitk image).
# The channel COUNT/kind comes from the data; decompose() builds the matching M.
# ============================================================================
def load_own_energy_stack(config: DecompConfig):
    """Our reconstruction: the threshold NIfTIs present in input_dir. Uses however many
    threshold_labels actually exist (3 works), each mapped to its threshold-option window."""
    sitk = _sitk()
    if not config.input_dir:
        raise ValueError("own input needs config.input_dir")
    d = Path(config.input_dir)
    edges = mlib.bin_edges(config.threshold_option)
    present = []
    for i, lbl in enumerate(config.threshold_labels):
        p = d / config.input_pattern.format(label=lbl)
        if p.exists():
            if i >= len(edges):
                raise ValueError(f"threshold '{lbl}' (#{i}) has no window in option "
                                 f"{config.threshold_option} ({len(edges)} bins)")
            present.append((mlib.Channel(kind="threshold", label=lbl, window_keV=edges[i],
                                         option=config.threshold_option, bin_index=i), p))
    if not present:
        raise FileNotFoundError(f"No threshold volumes in {d} "
                                f"(pattern {config.input_pattern.format(label='?')})")
    channels = [c for c, _ in present]
    paths = [p for _, p in present]
    img0 = sitk.ReadImage(str(paths[0]))
    ref = img0
    z0, z1 = ((0, ref.GetSize()[2]) if config.z_slab_mm is None
              else _slab_slice(ref, config.z_slab_mm))
    vols = None
    for j, p in enumerate(paths):
        img = img0 if j == 0 else sitk.ReadImage(str(p))
        a = sitk.GetArrayFromImage(img).astype(np.float32)[z0:z1]
        if img is not img0:
            del img
        if vols is None:
            vols = np.empty((len(paths),) + a.shape, dtype=np.float32)
        elif a.shape != vols.shape[1:]:
            raise ValueError(f"own volumes mismatched shapes: {a.shape} vs {vols.shape[1:]}")
        vols[j] = a
        del a
    if config.z_slab_mm is not None:
        ref = ref[:, :, z0:z1]
    logger.info("own recon: %d threshold channels %s", len(channels), [c.label for c in channels])
    return vols, channels, ref


def _classify_text(text: str, loose: bool = False):
    """
    Classify one string as a monoenergetic or threshold series.

    Real NAEOTOM SeriesDescriptions are 'MonoEnergeticPlus 70 keV' and
    'ProtocolModel WFBP_T1 Qr40f(3) 0.4 (0.4) [A,1]_0', so the keV figure is NOT
    adjacent to the word 'Mono' -- the product name sits between them.  Match 'mono'
    and the keV number independently instead of assuming they are neighbours.
    'VNC' and other spectral products match neither and are ignored.
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


def _classify_series(desc: str, path: str):
    """
    Series -> ('mono', keV) | ('threshold', n) | None.

    The SeriesDescription is tried ALONE first, then description+folder name.  One
    Siemens export folder holds several products -- the VMI export also carries
    WFBP_T1/T2 series -- so a folder whose name contains 'Mono' must never re-label a
    series whose own description says WFBP.

    Kept in step with reconstruction/recon_comparison.py.
    """
    return (_classify_text(desc or "")
            or _classify_text(f"{desc or ''} {path or ''}", loose=True))


def _read_series_volume(sitk, files, z_slab_mm):
    """Read one DICOM series (list of [z, path]) as a float32 (Z,Y,X) array (slab-limited)."""
    r = sitk.ImageSeriesReader()
    r.SetFileNames(_slab_files(files, z_slab_mm))
    img = r.Execute()
    return sitk.GetArrayFromImage(img).astype(np.float32), img


def load_siemens_energy_stack(config: DecompConfig, kind: str):
    """
    Siemens DICOM export -> auto-discovered channels of one product.
      kind='vmi'  : all 'Mono <keV>' monoenergetic series  -> mono channels (sorted by keV).
      kind='wfbp' : all 'WFBP T<n>' threshold series        -> threshold channels (sorted by n).
    Series are classified from SeriesDescription (or the folder name), so the channel COUNT is
    whatever the folder contains -- 2, 4, 8, or 13 VMIs -- never hardcoded.
    """
    sitk = _sitk()
    folder = config.dicom_dir or config.input_dir
    if not folder:
        raise ValueError("Siemens input needs config.dicom_dir")
    want = "mono" if kind == "vmi" else "threshold"
    cache = str(Path(config.output_dir) / "dicom_index.json") if config.output_dir else None
    index = build_dicom_index(folder, cache_path=cache)
    edges = mlib.bin_edges(config.threshold_option)

    found = []  # (sort_key, Channel, uid, files)
    for uid, s in index.items():
        path0 = s["files"][0][1] if s["files"] else ""
        spec = _classify_series(s.get("desc", ""), path0)
        if not spec or spec[0] != want:
            continue
        if want == "mono":
            e = spec[1]
            ch = mlib.Channel(kind="mono", label=f"{e:g}keV", energy_keV=e)
            found.append((e, ch, uid, s["files"]))
        else:
            n = spec[1]
            if n - 1 >= len(edges):
                continue
            ch = mlib.Channel(kind="threshold", label=f"T{n}", window_keV=edges[n - 1],
                              option=config.threshold_option, bin_index=n - 1)
            found.append((n, ch, uid, s["files"]))
    if not found:
        disc = [(s.get("number"), s.get("desc")) for s in index.values()]
        raise ValueError(f"No '{kind}' series found under {folder}. Discovered (number, desc): {disc}")

    found.sort(key=lambda t: t[0])
    channels = [c for _, c, _, _ in found]

    # A Siemens export puts each reconstruction in its own folder, and build_dicom_index
    # walks recursively.  Pointed at a PARENT holding several sets, every set matches and
    # they are concatenated into one oversized stack with repeated labels (T1,T1,T2,T2,...)
    # -- which decomposes without complaint and is silently wrong.  Refuse instead.
    labels = [c.label for c in channels]
    if len(set(labels)) != len(labels):
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        raise ValueError(
            f"{kind}: duplicate channel labels {dupes} under {folder} -- that folder holds "
            f"more than one {kind} reconstruction. Point the input at the specific series "
            f"folder. To see what is where:\n"
            f"  python -m decomposition.decompose --list-series '{folder}'")

    logger.info("Siemens %s: %d channels %s", kind, len(channels), [c.label for c in channels])

    vols, ref = None, None
    for i, (_key, ch, uid, files) in enumerate(found):
        a, img = _read_series_volume(sitk, files, config.z_slab_mm)
        if ref is None:
            ref = img
        elif img is not ref:
            del img
        if vols is None:
            vols = np.empty((len(found),) + a.shape, dtype=np.float32)
        elif a.shape != vols.shape[1:]:
            raise ValueError(f"Siemens {kind} series have mismatched shapes ({a.shape} vs "
                             f"{vols.shape[1:]}) -- differing matrix size / slab across series?")
        vols[i] = a
        del a
        logger.info("  %s <- series #%s (%s): %d slices",
                    ch.label, index[uid]["number"], uid[-8:], vols.shape[1])
    return vols, channels, ref


def load_energy_stack(config: DecompConfig):
    """
    Load an (N,Z,Y,X) volume stack + its channel descriptors + reference image, dispatching on
    config.input_source: 'own' (our NIfTI recon), 'wfbp' (Siemens threshold DICOM), 'vmi' (Siemens
    monoenergetic DICOM).  N is whatever the data provides -- the one shared entry point the driver
    uses so the pipeline is identical across all three approaches (only this loader differs).
    """
    src = (config.input_source or ("own" if config.input_format == "nifti" else "")).lower()
    if src == "own":
        return load_own_energy_stack(config)
    if src in ("vmi", "wfbp"):
        return load_siemens_energy_stack(config, src)
    raise ValueError(f"config.input_source must be 'own'|'wfbp'|'vmi' (got {config.input_source!r})")


def save_decomp_result(result: DecompResult, ref, out_dir) -> List[Path]:
    """Write one NIfTI per material (geometry copied from ref) + stability/config/metadata JSON."""
    sitk = _sitk()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def _save(arr, name):
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.CopyInformation(ref)
        p = out / name
        sitk.WriteImage(img, str(p))
        written.append(p)

    mode = result.config.mode
    for mat, arr in result.material_maps.items():
        _save(arr, f"decomp_{mode}_{mat}.nii.gz")
    if result.residual is not None:
        _save(result.residual, f"decomp_{mode}_residual.nii.gz")

    (out / f"decomp_{mode}_stability.json").write_text(json.dumps(result.stability, indent=2))
    (out / f"decomp_{mode}_metadata.json").write_text(json.dumps(result.metadata, indent=2))
    (out / f"decomp_{mode}_config.json").write_text(json.dumps(result.config.to_dict(), indent=2))
    written += [out / f"decomp_{mode}_stability.json",
                out / f"decomp_{mode}_metadata.json",
                out / f"decomp_{mode}_config.json"]
    return written
