"""
processed_separation_investigation.py -- are the vendor's thresholds cumulative or already split?

Standalone diagnostic.  Measures the NOISE correlation between the four reconstructed
threshold volumes, from the high-pass residual in the automatically detected water region:
cumulative thresholds share photons and correlate strongly, disjoint bins are
near-uncorrelated.  Noise is used rather than signal because images of the same object
correlate either way, and a correlation is unaffected by the per-threshold gain calibration.

Input: output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz; CPU only.
Output: output/research/processed_separation/ (findings report, metrics JSON, correlation figure).
"""

import json
import sys
from pathlib import Path

import numpy as np

# Reuse the loader + label-free primitives from the image-domain study (no duplication).
from image_subtraction_investigation import (
    load_bin_volumes, auto_body_mask, auto_water_mask,
    noise_from_highpass, inter_bin_stats, off_diagonal_energy, LABELS)

sys.stdout.reconfigure(line_buffering=True)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _REPO_ROOT / "output" / "research" / "processed_separation"

NOISE_HP_SIGMA = 2.0          # in-plane high-pass sigma (px) for the noise estimate
CUM_CORR_HI = 0.6             # mean|off-diag| NOISE corr above this -> cumulative
EXC_CORR_LO = 0.3             # ... below this -> exclusive


def _cov_to_corr(C):
    """Covariance -> correlation matrix."""
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return np.asarray(C) / np.outer(d, d)


def _verdict(mean_offdiag_noise):
    if mean_offdiag_noise > CUM_CORR_HI:
        return ("cumulative",
                "the four thresholds share photons (nested A⊇B⊇C⊇D): their noise is strongly "
                "correlated, so the data are cumulative threshold counts. Forming exclusive energy "
                "bins requires subtraction (studied by the other two investigations, and subject to "
                "invariant #3); for decomposition, feed the cumulative thresholds directly with "
                "cumulative-averaged μ/ρ.")
    if mean_offdiag_noise < EXC_CORR_LO:
        return ("exclusive",
                "the four thresholds have near-independent noise, i.e. they are already DISJOINT "
                "energy bins (Siemens separated them). Do NOT subtract — feed the four bins directly "
                "into decomposition with the Excel's exclusive-window μ/ρ. The pipeline's cumulative "
                "assumption would be WRONG and must be revisited.")
    return ("ambiguous",
            "inter-threshold noise correlation is intermediate — partially shared. Likely cumulative "
            "with Siemens processing that partly decorrelates the bins, or exclusive bins with residual "
            "coupling from charge-sharing/pile-up corrections. Cross-check the raw descriptor metadata "
            "before deciding.")


def _save_fig(corr_signal, corr_noise, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, m, t in ((ax[0], corr_signal, "SIGNAL correlation\n(structure — high either way)"),
                    (ax[1], corr_noise, "NOISE correlation\n(the discriminator)")):
        im = a.imshow(m, vmin=-1, vmax=1, cmap="coolwarm")
        a.set_title(t, fontsize=9)
        a.set_xticks(range(len(LABELS))); a.set_yticks(range(len(LABELS)))
        a.set_xticklabels(LABELS); a.set_yticklabels(LABELS)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                a.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Inter-threshold correlation — cumulative (noise corr high) vs exclusive (noise corr ~0)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); print(f"[fig] {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Processed-data separation check (cumulative vs already-exclusive) ===")

    stack, xy_mm, z_mm, z0, origin_z = load_bin_volumes()
    nb = stack.shape[0]
    volA = stack[0]
    body = auto_body_mask(volA)
    water = auto_water_mask(volA, body)
    print(f"[mask] body={int(body.sum())} water={int(water.sum())}")

    # SIGNAL correlation (structure): high whether cumulative or exclusive.
    Xbody = np.stack([stack[b][body].ravel() for b in range(nb)], axis=1)
    corr_signal, _ = inter_bin_stats(Xbody)

    # NOISE correlation (the discriminator): from the in-plane high-pass residual in the flat ROI.
    Sigma = noise_from_highpass(stack, water, NOISE_HP_SIGMA)
    corr_noise = _cov_to_corr(Sigma)

    off_signal = off_diagonal_energy(corr_signal)
    off_noise = off_diagonal_energy(corr_noise)
    verdict, text = _verdict(off_noise)

    print(f"  mean|off-diag| SIGNAL corr = {off_signal:.3f}  (high either way)")
    print(f"  mean|off-diag| NOISE  corr = {off_noise:.3f}  -> {verdict.upper()}")

    metrics = {
        "mean_offdiag_signal_corr": off_signal,
        "mean_offdiag_noise_corr": off_noise,
        "signal_correlation": corr_signal.tolist(),
        "noise_correlation": corr_noise.tolist(),
        "noise_sd_per_threshold": np.sqrt(np.clip(np.diag(Sigma), 0, None)).tolist(),
        "thresholds": list(LABELS),
        "verdict": {"class": verdict, "explanation": text,
                    "cumulative_if_above": CUM_CORR_HI, "exclusive_if_below": EXC_CORR_LO},
    }
    (OUT_DIR / "processed_separation_metrics.json").write_text(json.dumps(metrics, indent=2))
    _save_fig(corr_signal, corr_noise, OUT_DIR / "processed_separation_correlation.png")
    _write_findings(metrics, OUT_DIR / "processed_separation_findings.md")
    print(f"\n[done] metrics  → {OUT_DIR / 'processed_separation_metrics.json'}")
    print(f"[done] findings → {OUT_DIR / 'processed_separation_findings.md'}")
    print(f"[verdict] {verdict}: noise off-diag corr = {off_noise:.3f}")


def _write_findings(m, path):
    cn = np.array(m["noise_correlation"])
    L = ["# Are the processed thresholds cumulative or already exclusive?", "",
         "*Standalone diagnostic. The gain-invariant inter-threshold NOISE correlation settles whether "
         "the Siemens-processed data are cumulative (nested, photons shared) or already exclusive "
         "(disjoint energy bins) — i.e. whether the subtraction studied by "
         "`image_subtraction_investigation.py` / `sinogram_subtraction_investigation.py` is the right "
         "operation at all.*", "",
         f"## Verdict: **{m['verdict']['class'].upper()}**", "",
         m["verdict"]["explanation"], "",
         f"- mean |off-diagonal| **noise** correlation = **{m['mean_offdiag_noise_corr']:.3f}** "
         f"(> {CUM_CORR_HI} → cumulative; < {EXC_CORR_LO} → exclusive)",
         f"- mean |off-diagonal| **signal** correlation = {m['mean_offdiag_signal_corr']:.3f} "
         f"(high either way — the four volumes image the same object, so it does NOT discriminate)", "",
         "## Why noise correlation is the right test",
         "Every photon counted in a higher cumulative threshold is also counted in the lower ones, so "
         "cumulative thresholds share quantum noise → strong positive noise correlation. Disjoint "
         "(exclusive) energy bins count different photons → independent noise → ~0 correlation. "
         "Per-threshold gain/HU calibration only rescales each threshold and cannot change a "
         "correlation, so this test is immune to the calibration that breaks the A≥B≥C≥D magnitude "
         "ordering (invariant #3). Reconstruction is per-threshold, so the correlation measured in the "
         "reconstructed volumes reflects the input threshold data.", "",
         "## Noise correlation matrix (thresholds " + " ".join(LABELS) + ")", "",
         "| | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for i, lab in enumerate(LABELS):
        L.append(f"| **{lab}** | " + " | ".join(f"{cn[i, j]:.3f}" for j in range(len(LABELS))) + " |")
    L += ["", "## Cross-checks (to corroborate)",
          "- **Metadata:** inspect the raw `.mat` descriptor with `mat_structure.py` for how the four "
          "thresholds are defined. Siemens `...THRESHOLD...T1...COUNT...` naming denotes cumulative "
          "threshold counts.",
          "- **Physics prior:** photon-counting detectors natively produce CUMULATIVE threshold counts "
          "(each energy comparator fires for every photon above its threshold); exclusive bins only "
          "exist after a downstream subtraction.",
          "- **Magnitude ordering** (raw, non-HU volumes): cumulative → A>B>C>D monotone; exclusive → "
          "mid-peaked. HU calibration normalises every threshold to water≈0, so use the raw volumes; "
          "per-threshold gain can distort this, so the noise-correlation test above is more robust.", "",
          "## Figure",
          "- `processed_separation_correlation.png` — signal vs noise correlation heatmaps.", "",
          "## Implication for material decomposition",
          "Match the M-matrix to the verdict: if **cumulative**, feed the four thresholds directly with "
          "cumulative-averaged μ/ρ (no subtraction — sidesteps invariant #3); if **exclusive**, feed the "
          "four bins directly with the Excel's exclusive-window μ/ρ (no subtraction, independent per-bin "
          "noise). Never subtract to convert between the two domains "
          "(see `sinogram_subtraction_investigation.py`)."]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
