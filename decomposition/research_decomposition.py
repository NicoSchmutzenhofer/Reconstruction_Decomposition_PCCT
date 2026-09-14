"""
research_decomposition.py -- research and ablation experiments for the decomposition.

Separate from the production driver (decompose.py).  An experiment is one @experiment
function, so new ones are added without touching the core library.  Figures, tables and a
findings report go to output/research/.

  volume-free:
    stability_across_modes   kappa(M) per clinical mode
    material_cosines         collinearity heatmap of all material signatures
    threshold_option_scan    kappa per mode for each threshold option in mu_rho_binavg.csv
  need the volumes in output/reconstruction/ (skipped when absent):
    bin_domain_comparison    exclusive bins via subtraction vs cumulative M
    estimator_ladder         ols / wls / wls_denoise maps, flat-region noise and edge sharpness

    python -m decomposition.research_decomposition
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# --- import shim: work both as `-m decomposition.research_decomposition` and as a script ---
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decomposition import material_library as mlib
from decomposition import decomposition_modes as modes
from decomposition import noise_estimation as ne
from decomposition.material_decomposition import (
    DecompConfig, decompose, load_threshold_volumes, mode_stability, stability_report,
    column_cosines)

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = _REPO_ROOT / "output" / "research"

# --- experiment registry (the extension surface) ---------------------------
EXPERIMENTS: Dict[str, Callable] = {}


def experiment(name: str):
    def deco(fn):
        EXPERIMENTS[name] = fn
        return fn
    return deco


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"matplotlib unavailable ({e}); writing numbers only, skipping figures.")
        return None


@experiment("stability_across_modes")
def stability_across_modes(out_dir: Path, mode_keys: Optional[List[str]] = None,
                           option: int = 1) -> dict:
    keys = mode_keys or modes.list_modes()
    rows = sorted((mode_stability(k, option) for k in keys),
                  key=lambda r: r["condition_number"])
    table = [{"mode": r["mode"], "materials": "/".join(r["materials"]),
              "kappa": r["condition_number"], "kappa_colnorm": r["condition_number_colnorm"],
              "verdict": r["verdict"]} for r in rows]
    (out_dir / "stability_across_modes.json").write_text(json.dumps(table, indent=2))

    plt = _mpl()
    if plt:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        labels = [f"{r['mode']}\n{'/'.join(m[:4] for m in r['materials'])}" for r in rows]
        vals = [r["condition_number"] for r in rows]
        colors = ["#2a9d8f" if v < 200 else "#e9c46a" if v < 1000 else "#e76f51" for v in vals]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_yscale("log")
        ax.set_ylabel("condition number  kappa(M)")
        ax.axhline(200, ls="--", c="gray", lw=0.8)
        ax.axhline(1000, ls="--", c="gray", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Material-matrix conditioning per mode (Option {option})")
        fig.tight_layout()
        fig.savefig(out_dir / "stability_across_modes.png", dpi=140)
        plt.close(fig)
    return {"table": table}


@experiment("material_cosines")
def material_cosines(out_dir: Path, option: int = 1) -> dict:
    mats = mlib.available_materials()
    C = column_cosines(mlib.build_M(mats, option))
    (out_dir / "material_cosines.json").write_text(
        json.dumps({"materials": mats, "cosines": C.tolist()}, indent=2))
    plt = _mpl()
    if plt:
        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        im = ax.imshow(C, vmin=0.8, vmax=1.0, cmap="magma")
        ax.set_xticks(range(len(mats)))
        ax.set_xticklabels(mats, rotation=90, fontsize=7)
        ax.set_yticks(range(len(mats)))
        ax.set_yticklabels(mats, fontsize=7)
        fig.colorbar(im, label="cosine similarity (1.0 = collinear)")
        ax.set_title(f"Material-signature collinearity (Option {option})")
        fig.tight_layout()
        fig.savefig(out_dir / "material_cosines.png", dpi=140)
        plt.close(fig)
    return {"materials": mats}


@experiment("threshold_option_scan")
def threshold_option_scan(out_dir: Path, mode_keys: Optional[List[str]] = None) -> dict:
    keys = mode_keys or modes.list_modes()
    opts = mlib.available_options()
    table = []
    for k in keys:
        spec = modes.get_mode(k)
        row = {"mode": k, "materials": "/".join(spec.materials)}
        for o in opts:
            M = mlib.build_M(spec.materials, o)
            row[f"kappa_opt{o}"] = stability_report(M, spec.materials)["condition_number"]
        table.append(row)
    (out_dir / "threshold_option_scan.json").write_text(json.dumps(table, indent=2))
    return {"table": table, "options": opts}


@experiment("bin_domain_comparison")
def bin_domain_comparison(out_dir: Path, base_config: DecompConfig, loaded=None) -> dict:
    """Exclusive-via-subtraction vs cumulative-M on the same slab (needs volumes).
    `loaded` = optional preloaded (volumes, ref, mu_water) to share one load across experiments."""
    volumes, ref, mu_water = loaded if loaded is not None else load_threshold_volumes(base_config)
    water_mask = np.abs(volumes[0]) < base_config.water_hu_tol
    summary: dict = {}
    for domain in ("exclusive", "cumulative"):
        cfg = DecompConfig.from_dict({**base_config.to_dict(), "bin_domain": domain})
        res = decompose(volumes, cfg, mu_water=mu_water, water_mask=water_mask)
        summary[domain] = {
            "kappa": res.stability["condition_number"],
            "residual_mean": (float(np.mean(res.residual))
                              if res.residual is not None else None),
            "map_noise_sd_in_water": {m: float(np.std(arr[water_mask]))
                                      for m, arr in res.material_maps.items()},
        }
    (out_dir / "bin_domain_comparison.json").write_text(json.dumps(summary, indent=2))
    return summary


# --- no-reference image-quality metrics (no ground truth needed) -----------
def _edge_sharpness(vol: np.ndarray) -> float:
    """90th-percentile gradient magnitude -- higher = sharper edges (resolution proxy)."""
    g = np.gradient(np.asarray(vol, dtype=float))
    if isinstance(g, np.ndarray):
        g = [g]
    mag = np.sqrt(np.sum([gi * gi for gi in g], axis=0))
    return float(np.percentile(mag, 90))


def _map_metrics(m: np.ndarray) -> dict:
    return {"flat_noise_sd": ne.estimate_map_noise(m), "edge_sharpness": _edge_sharpness(m)}


def _panel(path: Path, panels: dict, materials, estimators) -> None:
    """panels[est][mat] = a precomputed 2-D mid-slice (so the caller need not keep full maps)."""
    plt = _mpl()
    if not plt:
        return
    ncol, nrow = len(materials), len(estimators)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    for r, est in enumerate(estimators):
        for c, mat in enumerate(materials):
            sl = panels[est][mat]
            lo, hi = np.percentile(sl, [2, 98])
            ax = axes[r][c]
            ax.imshow(sl, vmin=lo, vmax=hi if hi > lo else lo + 1e-6, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(mat, fontsize=9)
            if c == 0:
                ax.set_ylabel(est, fontsize=9)
    fig.suptitle("Estimator ladder -- mid slice per material (visual review)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@experiment("estimator_ladder")
def estimator_ladder(out_dir: Path, base_config: DecompConfig,
                     estimators=("ols", "wls", "wls_denoise"),
                     noise_model: str = "global", loaded=None) -> dict:
    """
    Run the estimator ladder on the loaded volume -> before/after panels + no-reference metrics
    (flat-region noise SD, edge sharpness) for radiologist-facing, ground-truth-free comparison.
    Needs the reconstructed volumes. 'wls_joint' is excluded by default (iterative + memory-heavy
    on a full volume); add it and set base_config.z_slab_mm to compare it on a slab.
    `loaded` = optional preloaded (volumes, ref, mu_water) to share one load across experiments.
    """
    volumes, ref, mu_water = loaded if loaded is not None else load_threshold_volumes(base_config)
    water_mask = np.abs(volumes[0]) < base_config.water_hu_tol
    materials = list(modes.get_mode(base_config.mode).materials)
    metrics, panels = {}, {}
    for est in estimators:
        cfg = DecompConfig.from_dict({**base_config.to_dict(),
                                      "estimator": est, "noise_model": noise_model})
        res = decompose(volumes, cfg, mu_water=mu_water, water_mask=water_mask)
        metrics[est] = {mat: _map_metrics(res.material_maps[mat]) for mat in materials}
        # Keep only the mid-slice per material for the visual panel, then drop the full maps
        # so we never hold more than one estimator's volumes in memory at a time.
        panels[est] = {mat: np.array(res.material_maps[mat][res.material_maps[mat].shape[0] // 2])
                       for mat in materials}
        del res
    (out_dir / "estimator_ladder_metrics.json").write_text(json.dumps(metrics, indent=2))
    _panel(out_dir / "estimator_ladder.png", panels, materials, estimators)
    return {"metrics": metrics, "estimators": list(estimators), "materials": materials}


def write_findings(out_dir: Path, results: dict) -> Path:
    lines = ["# Material decomposition -- research findings (Phase A)", "",
             "Auto-generated by research_decomposition.py. Figures alongside this file.", ""]
    sam = results.get("stability_across_modes", {}).get("table")
    if sam:
        lines += ["## Conditioning per mode (Option 1 = scan thresholds)", "",
                  "| mode | materials | kappa(M) | verdict |", "|---|---|---:|---|"]
        for r in sam:
            lines.append(f"| {r['mode']} | {r['materials']} | {r['kappa']:.0f} | {r['verdict']} |")
        lines += ["",
                  "Only bases containing a spectrally distinct (ideally K-edge) material are usable; "
                  "tissue-only bases are ill-conditioned (see material_cosines.png).", ""]
    tos = results.get("threshold_option_scan", {})
    if tos.get("table"):
        opts = tos["options"]
        lines += ["## Threshold-option scan (does bin placement help conditioning?)", "",
                  "| mode | " + " | ".join(f"kappa opt{o}" for o in opts) + " |",
                  "|---|" + "---:|" * len(opts)]
        for r in tos["table"]:
            lines.append("| " + r["mode"] + " | "
                         + " | ".join(f"{r.get(f'kappa_opt{o}'):.0f}" for o in opts) + " |")
        lines.append("")
    bdc = results.get("bin_domain_comparison")
    if bdc:
        lines += ["## Bin-domain comparison (exclusive vs cumulative)", "",
                  "| domain | kappa | residual mean | map noise SD (water) |",
                  "|---|---:|---:|---|"]
        for dom, v in bdc.items():
            noise = ", ".join(f"{m}={x:.3g}" for m, x in v["map_noise_sd_in_water"].items())
            rm = v["residual_mean"]
            rm_s = "" if rm is None else f"{rm:.3g}"
            lines.append(f"| {dom} | {v['kappa']:.0f} | {rm_s} | {noise} |")
        lines.append("")
    lad = results.get("estimator_ladder", {}).get("metrics")
    if lad:
        mats = results["estimator_ladder"]["materials"]
        lines += ["## Estimator ladder -- no-reference metrics (flat-region noise / edge sharpness)",
                  "", "Lower noise with preserved sharpness is better; see estimator_ladder.png.", "",
                  "| estimator | " + " | ".join(f"{m}" for m in mats) + " |",
                  "|" + "---|" * (len(mats) + 1)]
        for est, mm in lad.items():
            cells = " | ".join(f"{mm[m]['flat_noise_sd']:.3g} / {mm[m]['edge_sharpness']:.3g}"
                               for m in mats)
            lines.append(f"| {est} | {cells} |")
        lines.append("")
    p = out_dir / "decomposition_research_findings.md"
    p.write_text("\n".join(lines))
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "stability_across_modes": stability_across_modes(out),
        "material_cosines": material_cosines(out),
        "threshold_option_scan": threshold_option_scan(out),
    }
    # Volume-based experiments (need the reconstructed volumes; run on the cluster).
    # Set z_slab_mm for a fast targeted run of the (iterative) estimator ladder.
    # Light, memory-safe estimator for the full-volume comparisons (joint is slab-only).
    cfg = DecompConfig(mode="phantom_ca_i", estimator="wls", noise_model="global",
                       input_dir=str(_REPO_ROOT / "output" / "reconstruction"), output_dir=str(out))
    try:
        loaded = load_threshold_volumes(cfg)     # load once, shared across volume experiments
        results["bin_domain_comparison"] = bin_domain_comparison(out, cfg, loaded=loaded)
        results["estimator_ladder"] = estimator_ladder(out, cfg, loaded=loaded)
    except (FileNotFoundError, ImportError) as e:
        logger.warning("Skipping volume-based experiments (no volumes / SimpleITK): %s", e)
    findings = write_findings(out, results)
    print(f"Research outputs -> {out}")
    print(f"Findings -> {findings}")


if __name__ == "__main__":
    main()
