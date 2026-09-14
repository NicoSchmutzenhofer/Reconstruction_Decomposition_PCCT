"""
noise_estimation.py -- no-reference estimate of the inter-channel noise covariance.

Measures per-channel noise and the full covariance in automatically selected flat regions,
so no ROI is drawn and no channel is assumed to be the noisiest.  Two granularities: one
matrix per volume (noise_covariance_global) or a windowed covariance field per voxel
(noise_covariance_field).  The off-diagonal terms matter, because exclusive bins formed by
subtracting cumulative images have correlated noise.  Only numpy is required for the global
path; scipy speeds up the box filter when present.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Local mean / box filter (scipy if available, else a numpy separable fallback)
# ---------------------------------------------------------------------------
def _box_mean(a: np.ndarray, size: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if size <= 1:
        return a
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(a, size=size, mode="nearest")
    except Exception:
        r = size // 2
        for ax in range(a.ndim):
            if a.shape[ax] == 1:
                continue
            aa = np.moveaxis(a, ax, -1)
            padded = np.pad(aa, [(0, 0)] * (aa.ndim - 1) + [(r, size - 1 - r)], mode="edge")
            cs = np.cumsum(padded, axis=-1)
            cs = np.concatenate([np.zeros(cs.shape[:-1] + (1,)), cs], axis=-1)
            win = (cs[..., size:] - cs[..., :-size]) / size
            a = np.moveaxis(win[..., : aa.shape[-1]], -1, ax)
        return a


def highpass_residual(vol: np.ndarray, size: int = 5) -> np.ndarray:
    """vol minus its local mean -> dominated by noise (plus edges)."""
    return np.asarray(vol, dtype=np.float64) - _box_mean(vol, size)


def local_gradient_magnitude(vol: np.ndarray) -> np.ndarray:
    grads = np.gradient(np.asarray(vol, dtype=np.float64))
    if isinstance(grads, np.ndarray):
        grads = [grads]
    return np.sqrt(np.sum([g * g for g in grads], axis=0))


def flat_mask(ref: np.ndarray, size: int = 5, percentile: float = 40.0,
              body_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Boolean mask of locally-flat voxels (low local gradient) -- where the high-pass residual
    is dominated by noise rather than structure. `percentile` keeps the flattest fraction.
    """
    grad = local_gradient_magnitude(_box_mean(ref, size))
    m = np.ones(ref.shape, bool) if body_mask is None else body_mask.astype(bool)
    if m.sum() == 0:
        m = np.ones(ref.shape, bool)
    thr = np.percentile(grad[m], percentile)
    return m & (grad <= thr)


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------
def noise_covariance_global(B: np.ndarray, size: int = 5, flat_percentile: float = 40.0,
                            body_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    One noise covariance Sigma (n_bins x n_bins) for the whole volume.

    B: (n_bins, ...) measurement volumes. Residuals are sampled over automatically-detected
    flat voxels; the returned matrix's *structure* (which bin is noisiest, and the inter-bin
    correlations) is what weighted least squares needs (it is invariant to an overall scale).
    """
    B = np.asarray(B, dtype=np.float64)
    nb = B.shape[0]
    R = np.stack([highpass_residual(B[i], size) for i in range(nb)], axis=0)
    fm = flat_mask(B.sum(0), size, flat_percentile, body_mask)
    Rf = R[:, fm]                                   # (n_bins, N_flat)
    if Rf.shape[1] < nb + 1:
        Rf = R.reshape(nb, -1)                      # fall back to all voxels
    Rf = Rf - Rf.mean(1, keepdims=True)
    Sigma = (Rf @ Rf.T) / max(Rf.shape[1] - 1, 1)
    Sigma += np.eye(nb) * (np.trace(Sigma) / nb) * 1e-6   # tiny ridge for invertibility
    return Sigma


def noise_covariance_field(B: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Per-voxel noise covariance field via windowed products of the high-pass residuals.

    B: (n_bins, ...spatial). Returns (...spatial, n_bins, n_bins). Intended to be called on a
    Z-chunk (optionally with a halo) so memory stays bounded on full volumes.
    """
    B = np.asarray(B, dtype=np.float64)
    nb = B.shape[0]
    R = [highpass_residual(B[i], size) for i in range(nb)]
    field = np.zeros(B.shape[1:] + (nb, nb), dtype=np.float64)
    for i in range(nb):
        for j in range(i, nb):
            c = _box_mean(R[i] * R[j], size)
            field[..., i, j] = c
            if i != j:
                field[..., j, i] = c
    # ridge so every local matrix is invertible
    tr = np.einsum("...ii->...", field) / nb
    field += np.eye(nb) * (tr[..., None, None] * 1e-3 + 1e-12)
    return field


def estimate_map_noise(vol: np.ndarray, size: int = 5, flat_percentile: float = 40.0,
                       body_mask: Optional[np.ndarray] = None) -> float:
    """Robust noise SD of a single (material) map, from its high-pass residual in flat regions."""
    r = highpass_residual(vol, size)
    fm = flat_mask(vol, size, flat_percentile, body_mask)
    vals = r[fm] if fm.sum() > 32 else r.ravel()
    mad = np.median(np.abs(vals - np.median(vals)))
    return float(1.4826 * mad)   # MAD -> Gaussian-equivalent SD (robust to outliers/edges)
