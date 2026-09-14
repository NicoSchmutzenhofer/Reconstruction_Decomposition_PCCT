"""
denoising.py -- edge-preserving denoisers for material maps.

Each denoiser is registered in DENOISERS and scales its strength from the measured map noise
instead of a fixed lambda.  Optional cross-channel guidance uses the highest-SNR channel,
chosen at run time, never a fixed one.  scikit-image and scipy are imported lazily; the
guided filter is pure numpy.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .noise_estimation import _box_mean

DENOISERS: Dict[str, Callable] = {}


def denoiser(name: str):
    def deco(fn: Callable) -> Callable:
        DENOISERS[name] = fn
        return fn
    return deco


def _clean(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


@denoiser("tv")
def _tv(vol: np.ndarray, sigma: float, scale: float = 1.0, **kw) -> np.ndarray:
    """Total-variation (edge-preserving); weight ~ measured noise."""
    from skimage.restoration import denoise_tv_chambolle
    w = max(scale * sigma, 1e-8)
    return denoise_tv_chambolle(_clean(vol), weight=w, channel_axis=None)


@denoiser("nlm")
def _nlm(vol: np.ndarray, sigma: float, scale: float = 1.0, patch_size: int = 3,
         patch_distance: int = 5, **kw) -> np.ndarray:
    """Non-local means (best texture/edge preservation, slower); h, sigma ~ measured noise."""
    from skimage.restoration import denoise_nl_means
    s = max(sigma, 1e-8)
    return denoise_nl_means(_clean(vol), h=scale * s, sigma=s, patch_size=patch_size,
                            patch_distance=patch_distance, fast_mode=True, channel_axis=None)


@denoiser("bilateral")
def _bilateral(vol: np.ndarray, sigma: float, scale: float = 1.0, sigma_spatial: float = 1.5,
               **kw) -> np.ndarray:
    """Edge-preserving bilateral, applied slice-by-slice (skimage bilateral is 2-D)."""
    from skimage.restoration import denoise_bilateral
    v = _clean(vol)
    out = np.empty_like(v)
    sc = max(scale * sigma, 1e-8)
    for k in range(v.shape[0]):
        out[k] = denoise_bilateral(v[k], sigma_color=sc, sigma_spatial=sigma_spatial,
                                   channel_axis=None)
    return out


@denoiser("gaussian")
def _gaussian(vol: np.ndarray, sigma: float, scale: float = 0.7, **kw) -> np.ndarray:
    """Plain Gaussian smoothing -- NOT edge-preserving; a baseline for comparison only.
    `scale` here is the spatial sigma (voxels), independent of the noise level."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(_clean(vol), sigma=max(scale, 1e-3))


def guided_filter(p: np.ndarray, guide: np.ndarray, radius: int = 2, eps: float = 1e-3) -> np.ndarray:
    """He et al. guided filter (pure numpy): smooth `p` using edges from `guide`."""
    p = _clean(p); I = _clean(guide)
    size = 2 * radius + 1
    mean_I = _box_mean(I, size); mean_p = _box_mean(p, size)
    var_I = _box_mean(I * I, size) - mean_I * mean_I
    cov_Ip = _box_mean(I * p, size) - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return _box_mean(a, size) * I + _box_mean(b, size)


@denoiser("guided")
def _guided(vol: np.ndarray, sigma: float, scale: float = 1.0, guide: Optional[np.ndarray] = None,
            radius: int = 2, **kw) -> np.ndarray:
    """Edge-preserving guided filter; guide defaults to self (like bilateral) if none given."""
    g = vol if guide is None else guide
    eps = (max(scale * sigma, 1e-8)) ** 2
    return guided_filter(vol, g, radius=radius, eps=eps)


# ---------------------------------------------------------------------------
def select_guide_index(maps: Sequence[np.ndarray], sigmas: Sequence[float]) -> int:
    """Highest-SNR channel = argmax(structure spread / noise SD). Chosen from the data."""
    snr = []
    for m, s in zip(maps, sigmas):
        spread = float(np.std(m))
        snr.append(spread / max(s, 1e-8))
    return int(np.argmax(snr))


def available_denoisers() -> List[str]:
    return list(DENOISERS)


def denoise_maps(maps: Sequence[np.ndarray], sigmas: Sequence[float], method: str = "tv",
                 scale: float = 1.0, guide_index: Optional[int] = None, **kw) -> List[np.ndarray]:
    """
    Denoise each material map with strength derived from its own measured noise `sigmas[i]`.

    If `guide_index` is given, the maps are denoised with the guided filter using that channel
    as the structural guide (cross-channel); otherwise each channel is denoised independently
    with `method`. Returns a new list of maps.
    """
    if method not in DENOISERS:
        raise KeyError(f"Unknown denoiser '{method}'. Available: {available_denoisers()}")
    out: List[np.ndarray] = []
    if guide_index is not None:
        guide = maps[guide_index]
        for i, (m, s) in enumerate(zip(maps, sigmas)):
            out.append(_guided(m, s, scale=scale, guide=guide, **kw))
        return out
    fn = DENOISERS[method]
    for m, s in zip(maps, sigmas):
        out.append(fn(m, s, scale=scale, **kw))
    return out
