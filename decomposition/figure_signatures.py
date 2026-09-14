"""
figure_signatures.py -- mass attenuation of the basis materials against photon energy.

Plots <mu/rho> of the basis materials with the four threshold windows of the research mode
shaded.  It shows why water, soft tissue and calcium hydroxyapatite form an ill-conditioned
basis (smooth, nearly parallel curves) while iodine does not (a K-edge inside the detected
range).  Reads only data/mu_rho_mono.csv through material_library; no volumes, no GPU.

    python -m decomposition.figure_signatures [--out PATH] [--dpi N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from decomposition import material_library as mlib
from reconstruction import figure_panels as fp

# The four thresholds of the research mode, as the disjoint windows the published
# signature table tabulates. Shaded rather than drawn as lines: they are intervals on the
# energy axis, not values on the attenuation axis.
THRESHOLD_WINDOWS = ((20.0, 40.0), (40.0, 56.0), (56.0, 75.0), (75.0, 140.0))

# Ordered so the legend reads from the two agents down to the base materials, which is the
# order the conditioning argument is made in.
CURVES = (
    ("Iodine", "iodine", "tab:red", "-", 2.0),
    ("HA", "calcium hydroxyapatite", "tab:blue", "-", 1.8),
    ("Water", "water", "0.25", "--", 1.5),
    ("SoftTissue", "soft tissue", "0.55", ":", 1.5),
)


def _curve(material: str, energies: np.ndarray) -> np.ndarray:
    """mu/rho at each energy, with the K-edge left as a discontinuity."""
    return np.array([mlib.mono_mu_rho(material, float(e)) for e in energies], float)


def build(out_path: Path, dpi: int = 200) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = mlib.DETECTOR_RANGE_KEV
    # Dense grid, and the K-edge energies inserted twice with a hair either side so the
    # jump is drawn as a jump. On a plain linspace the edge is a steep diagonal line
    # between two samples, which is exactly the impression the figure has to avoid.
    grid = [np.linspace(lo, hi, 601)]
    for name, *_rest in CURVES:
        edge = mlib.material_info(name).k_edge_keV
        if edge and lo < edge < hi:
            grid.append(np.array([edge - 1e-3, edge + 1e-3]))
    energies = np.unique(np.concatenate(grid))

    fig, ax = plt.subplots(figsize=(fp.TEXTWIDTH_IN * 0.8, 3.6))

    # Bin labels along the BOTTOM: the curves all fall from left to right, so the top right
    # is the only place a legend fits, and a label there would be underneath it.
    for k, (e0, e1) in enumerate(THRESHOLD_WINDOWS):
        ax.axvspan(e0, e1, color="0.90" if k % 2 == 0 else "0.97", zorder=0, lw=0)
        ax.annotate(f"bin {k + 1}", ((e0 + e1) / 2, 0.015),
                    xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=fp.PRINT_FONT_PT - 2, color="0.35")

    for name, label, colour, style, width in CURVES:
        if not mlib.has_mono(name):
            continue
        y = _curve(name, energies)
        edge = mlib.material_info(name).k_edge_keV
        if edge and lo < edge < hi:                     # draw either side, never across
            for sel in (energies < edge, energies > edge):
                ax.plot(energies[sel], y[sel], style, color=colour, lw=width,
                        label=label if sel[0] else None)
            # Anchored below the edge and pointing up at it: above and to the right is
            # where the post-edge branch runs, and the label was landing on the curve it
            # is meant to identify.
            ax.annotate(f"K-edge {edge:.1f} keV",
                        (edge, mlib.mono_mu_rho(name, edge - 0.1)),
                        xytext=(20, -22), textcoords="offset points", ha="left",
                        fontsize=fp.PRINT_FONT_PT - 1, color=colour,
                        arrowprops={"arrowstyle": "->", "color": colour, "lw": 0.9,
                                    "shrinkB": 2})
        else:
            ax.plot(energies, y, style, color=colour, lw=width, label=label)

    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("photon energy  (keV)", fontsize=fp.PRINT_FONT_PT)
    ax.set_ylabel("mass attenuation coefficient\n" r"$\mu/\rho$  (cm$^2$/g)",
                  fontsize=fp.PRINT_FONT_PT)
    ax.tick_params(labelsize=fp.PRINT_FONT_PT - 1)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=fp.PRINT_FONT_PT - 1, frameon=True,
              framealpha=0.9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", default="output/research/material_signatures.png")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    print(f"wrote {build(Path(a.out), a.dpi)}")


if __name__ == "__main__":
    main()
