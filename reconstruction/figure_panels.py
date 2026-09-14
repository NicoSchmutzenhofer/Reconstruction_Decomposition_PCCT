"""
figure_panels.py -- shared image-panel helpers for the reconstruction and decomposition figures.

Shows a slice at a physical z, cropped around a physical centre, under three rules:
  * one window for all panels that are compared (shared_window), so per-panel autoscaling
    cannot hide a noise difference;
  * ROIs drawn as outlines, never fills, so no judged pixel is covered;
  * a scale bar from each panel's own pixel size (add_scale_bar).
Also holds the reader-facing labels and print sizes, so both stages name things the same way.

Pure library: no I/O, no __main__; matplotlib is imported inside the functions.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["shared_window", "symmetric_window", "crop_box_px", "index_of_point",
           "show_slice", "outline_circle", "add_scale_bar", "window_label",
           "panel_letter", "greyed_panel",
           "pretty", "panel_figsize", "PRINT_FONT_PT", "TEXTWIDTH_IN",
           "SOURCE_LABELS", "ESTIMATOR_LABELS", "BASIS_LABELS", "VARIANT_LABELS"]


# ── reader-facing names ──────────────────────────────────────────────────────
# The keys below are internal: directory names, config values and dictionary keys. They
# are fine in a log and wrong in a figure, where the reader has never seen them and has no
# way to expand them. One map, imported by both stages, so a name cannot mean one thing in
# the reconstruction figures and another in the decomposition ones -- and so it matches the
# thesis tables, which use exactly these strings.
SOURCE_LABELS = {
    "own":   "Ours (SIRT, 100 it.)",
    "wfbp":  "Vendor WFBP",
    "vmi":   "Vendor VMI",
    "s2bin": "Vendor 2-bin",
}
ESTIMATOR_LABELS = {
    "ols":         "OLS",
    "wls":         "WLS",
    "wls_denoise": "WLS + denoise",
    "wls_joint":   "WLS + joint",
}
BASIS_LABELS = {
    "phantom_ca_i":       "Soft tissue / Ca / I",
    "phantom_water_ca_i": "Water / Ca / I",
    "siemens_2bin":       "Vendor 2-bin",
}
VARIANT_LABELS = {
    "fbp_ramlak":     "FBP Ram–Lak",
    "fbp_shepplogan": "FBP Shepp–Logan",
    "fbp_hann":       "FBP Hann",
    "sirt_25":        "SIRT ×25",
    "sirt_100":       "SIRT ×100",
}
_ALL_LABELS = {}
for _m in (SOURCE_LABELS, ESTIMATOR_LABELS, BASIS_LABELS, VARIANT_LABELS):
    _ALL_LABELS.update(_m)


def pretty(key, short=False):
    """
    Reader-facing name for an internal key, or the key itself if it has none.

    Falling back to the key rather than raising is deliberate: a new estimator or variant
    should appear in a figure under its own name and be noticed, not crash the run that
    produced the numbers. `short` drops the parenthetical from the source names, for axes
    where the long form does not fit.
    """
    lab = _ALL_LABELS.get(str(key), str(key))
    if short and " (" in lab:
        lab = lab.split(" (", 1)[0]
    return lab


# ── print sizing ─────────────────────────────────────────────────────────────
# A figure authored 6 in wide and included at .49\textwidth is scaled to about half, and
# its 8 pt tick labels reach the page at 4 pt. Font size alone cannot fix that: the figure
# has to be authored at roughly the width it is printed at, so the scale factor is ~1 and
# the point size in the code is the point size on the page.
# MEASURED, not assumed: \the\textwidth in the thesis reports 448.13095 pt, and a TeX
# point is 1/72.27 in, so 448.13095 / 72.27 = 6.2008 in. Re-measure if the class options
# or the font size change -- KOMA recomputes the type area from them, and a wrong value
# here silently rescales every font in every figure.
TEXTWIDTH_IN = 6.2008     # \textwidth of the thesis class, in inches
PRINT_FONT_PT = 9.0       # base font; nothing in a figure should fall below 8 pt on paper


def panel_figsize(n_panels, frac=0.49, height_in=3.6, gutter_in=0.10):
    """
    figsize for an n-panel row whose panels will each be placed at `frac`\\textwidth.

    Returns the size of the WHOLE figure; `_save_panels` crops one panel out of it, so
    each crop comes out near `frac * TEXTWIDTH_IN` and needs no further scaling in LaTeX.
    """
    per = max(frac, 0.05) * TEXTWIDTH_IN
    return (per * max(int(n_panels), 1) + gutter_in * (max(int(n_panels), 1) - 1),
            float(height_in))


# ── windows ──────────────────────────────────────────────────────────────────
def shared_window(arrays: Iterable[np.ndarray], lo_pct=1.0, hi_pct=99.0):
    """
    (vmin, vmax) over ALL the arrays together, from percentiles rather than min/max.

    Percentiles and not extremes because a single hot voxel -- a defect, a metal fleck, the
    edge of the couch -- sets min/max and compresses everything the figure is about into a
    few grey levels.  Computed jointly so panels stay comparable: a per-panel window makes
    a noisy map and a clean one look alike, which is the difference being measured.
    """
    vals = []
    for a in arrays:
        a = np.asarray(a, float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            # Subsample: the percentile of a 512x512x100 stack is the same to three digits
            # from 200k samples, and the figure code runs over ~24 runs.
            vals.append(a if a.size <= 200_000 else a[:: max(1, a.size // 200_000)])
    if not vals:
        return (0.0, 1.0)
    v = np.concatenate(vals)
    lo, hi = float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return (lo, lo + 1.0) if np.isfinite(lo) else (0.0, 1.0)
    return (lo, hi)


def symmetric_window(arrays: Iterable[np.ndarray], pct=99.0):
    """(-v, +v) for a difference image, so zero is the midpoint of a diverging colourmap."""
    vals = []
    for a in arrays:
        a = np.asarray(a, float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(np.abs(a if a.size <= 200_000 else a[:: max(1, a.size // 200_000)]))
    if not vals:
        return (-1.0, 1.0)
    v = float(np.percentile(np.concatenate(vals), pct))
    v = v if np.isfinite(v) and v > 0 else 1.0
    return (-v, v)


def window_label(vmin, vmax, unit="HU"):
    """'W 500 / L 50 HU' -- the radiological way to state what a greyscale panel shows."""
    return f"W {vmax - vmin:.0f} / L {0.5 * (vmax + vmin):.0f} {unit}"


# ── geometry ─────────────────────────────────────────────────────────────────
def index_of_point(img, x_mm, y_mm, z_mm):
    """(col, row) float index of a physical point on THIS image's grid."""
    p = img.TransformPhysicalPointToContinuousIndex(
        (float(x_mm), float(y_mm), float(z_mm)))
    return float(p[0]), float(p[1])


def crop_box_px(shape_yx, cx_px, cy_px, half_px):
    """
    (y0, y1, x0, x1) of a square crop, clipped to the array.

    Returned in the array's own index space and always non-empty, so a centre that falls
    outside the volume yields an edge crop rather than an exception -- the panel then shows
    the wrong place visibly, which is a better failure than a traceback in stage 2.
    """
    ny, nx = int(shape_yx[0]), int(shape_yx[1])
    h = max(1, int(round(half_px)))
    y0 = int(np.clip(round(cy_px) - h, 0, max(0, ny - 1)))
    y1 = int(np.clip(round(cy_px) + h, y0 + 1, ny))
    x0 = int(np.clip(round(cx_px) - h, 0, max(0, nx - 1)))
    x1 = int(np.clip(round(cx_px) + h, x0 + 1, nx))
    return y0, y1, x0, x1


# ── drawing ──────────────────────────────────────────────────────────────────
def show_slice(ax, arr, vmin=None, vmax=None, cmap="gray", pixel_mm=None,
               scale_bar_mm=20.0, title=None, title_size=9, bar_colour="w"):
    """
    One image panel: no axes, optional scale bar, optional short title.

    Axes are removed rather than left blank -- pixel indices are not a quantity the reader
    of a results figure wants, and the scale bar carries the only spatial information that
    matters.
    """
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=title_size)
    if pixel_mm and scale_bar_mm:
        add_scale_bar(ax, pixel_mm, scale_bar_mm, colour=bar_colour)
    return im


def outline_circle(ax, xy, r_px, colour="tab:red", lw=1.0, ls="-", zorder=5):
    """
    A ROI as a 1 px contour.

    Deliberately unfilled: an overlay with any alpha at all sits on top of the pixels the
    figure is asking the reader to judge, and on a material map -- where the quantity IS
    the brightness -- a tint changes the answer.
    """
    from matplotlib.patches import Circle
    ax.add_patch(Circle((float(xy[0]), float(xy[1])), float(r_px), fill=False,
                        color=colour, lw=lw, ls=ls, zorder=zorder))


def add_scale_bar(ax, pixel_mm, length_mm=20.0, colour="w", pad_frac=0.045, lw=2.4):
    """
    A bar of known physical length in the lower-right corner, labelled in mm.

    Drawn from the panel's OWN pixel size: the sources differ in spacing and the crops
    differ in extent, so one bar reused across panels would be wrong on most of them.
    Falls back to no bar when the panel is too small to hold one honestly.
    """
    ny, nx = ax.get_images()[0].get_array().shape[:2] if ax.get_images() else (0, 0)
    if not nx or not pixel_mm or pixel_mm <= 0:
        return
    n = float(length_mm) / float(pixel_mm)
    if n > 0.6 * nx:                      # would not fit: halve once, then give up
        length_mm, n = length_mm / 2.0, n / 2.0
        if n > 0.6 * nx:
            return
    x1 = nx * (1 - pad_frac)
    y1 = ny * (1 - pad_frac)
    ax.plot([x1 - n, x1], [y1, y1], "-", color=colour, lw=lw,
            solid_capstyle="butt", zorder=6)
    ax.text(x1 - n / 2, y1 - ny * 0.02, f"{length_mm:.0f} mm", color=colour,
            fontsize=7, ha="center", va="bottom", zorder=6)


def panel_letter(ax, letter, size=11, dx=-0.02, dy=1.04):
    """(a)/(b)/(c) outside the axes, so the thesis text can point at one panel."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def greyed_panel(ax, text, facecolour="0.92", textcolour="0.35", fontsize=10):
    """
    A panel that is deliberately empty, with the reason written in it.

    Used where a quantity does not exist rather than merely being absent -- two energy
    channels cannot yield a third material map.  An omitted panel reads as an oversight; a
    labelled empty one states the restriction, which is the point being made.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(facecolour)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("0.75")
        s.set_linewidth(0.8)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, color=textcolour, wrap=True)
