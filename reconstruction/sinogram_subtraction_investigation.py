"""
sinogram_subtraction_investigation.py -- is threshold subtraction valid BEFORE reconstruction?

Standalone experiment; the production pipeline and invariant #3 (README) are unchanged.
Forms the exclusive energy windows on the sinogram,

        E1 = A - B,   E2 = B - C,   E3 = C - D,   E4 = D

reconstructs each with SIRT and answers two questions:
  1. Is the A >= B >= C >= D ordering violation zero-mean noise or a systematic bias?
     Decided by the projection-averaged exclusive value at each detector element.
  2. How far does subtract-then-reconstruct differ from reconstruct-then-subtract, per bin?
     The two are identical under FBP; SIRT is nonlinear and non-negative.

Needs the raw HDF5 sinograms and a CUDA GPU.  Holds at most two cumulative sinograms
(~16 GB each) and reconstructs only a z-slab.  Writes a findings report, metrics and figures
to output/research/sinogram_subtraction/.
"""

import json
import sys
from pathlib import Path

import numpy as np

from helical_reconstruction import (
    build_geom, detect_defect_channels, reconstruct_helical_stack,
    z_targets_for_full_scan, auto_hu_calibrate, apply_hu_calibration)
# Reuse the label-free analysis primitives from the image-domain study (no duplication).
from image_subtraction_investigation import (
    auto_body_mask, auto_water_mask, auto_detect_inserts, disk_mask,
    noise_from_highpass, cnr_sd)

sys.stdout.reconfigure(line_buffering=True)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _REPO_ROOT / "output" / "research" / "sinogram_subtraction"

# Raw data (same as the production driver).  Fill in your own files.
DATA_PATH = Path(r"/path/to/full_sinogram.raw.mat")
DESC_PATH = Path(r"/path/to/descriptor.raw.mat")

# Reconstruction settings — MIRROR the production driver so the comparison is fair.
N_PIXELS = 512
FOV_MM = 500.0
RECON_METHOD = 'sirt'          # nonlinear → the sino-vs-image comparison is meaningful
N_ITER = 100
FILTER_NAME = 'shepp-logan'    # only used if RECON_METHOD='fbp'
GEOMETRY_MODEL = 'curved'
Z_WEIGHTING = 'balanced'
WAVELET_RING_THRESHOLD = 2.0
SPIKE_MAD_K = 5.0
IPR_MAD_K = 6.0

# Study slab: SIRT × (4 cumulative + 3 exclusive) recons is heavy, so we reconstruct
# a slab centred on the phantom (the driver's example insert slice) rather than the
# full volume.  N_SLAB_SLICES is forced odd.  Increase for better noise statistics.
SLICE_IDX = 250                # phantom-with-inserts location (matches the driver)
N_SLAB_SLICES = 41             # ~16 mm at 0.4 mm spacing

# Negativity diagnostic
PROJ_STRIDE = 8                # projection subsample for the per-sample negativity fractions
OBJECT_PERCENTILE = 60.0       # rays above this (in the minuend) count as "through-object"

# Label-free metric knobs (same as the image-domain study)
INSERT_CONTRAST_HU = 45.0
INSERT_DIAM_MM = (4.0, 14.0)
NOISE_HP_SIGMA = 2.0

LABELS = ["A", "B", "C", "D"]
# Exclusive-window keV ranges, for labelling only.
EXCL_TAGS = ["20-40", "40-56", "56-75", "75-140"]


# ═══════════════════════════════════════════════════════════════════════
# IO  (raw .mat threshold sinograms)
# ═══════════════════════════════════════════════════════════════════════
def _load_threshold(f, logical_idx):
    """Threshold logical_idx (0=A..3=D) from an open HDF5 .mat, channels flipped.
    Physical storage is reversed (physical = 3 - logical); see invariants #1 and #2, README."""
    ref = f['data_full']['A'][3 - logical_idx, 0]
    return f[ref][:][:, :, ::-1].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Negativity structure — the empirical test of invariant #3
# ═══════════════════════════════════════════════════════════════════════
def negativity_stats(minuend, subtrahend=None, proj_stride=PROJ_STRIDE,
                     object_pct=OBJECT_PERCENTILE):
    """
    Characterise the exclusive sinogram E = minuend − subtrahend (E = minuend if
    subtrahend is None, i.e. the top window D).  Computed BEFORE any in-place
    subtraction, from cheap full-resolution projection reductions plus a
    projection subsample for the per-sample fractions (memory-lean).

    The headline discriminator is `neg_frac_meanmap`: the fraction of detector
    elements whose PROJECTION-AVERAGED exclusive value is negative.
      ~0   → per-sample negatives are zero-mean noise (reconstruction averages
             them out; sinogram subtraction is sound)
      high → a systematic per-element bias that survives reconstruction
             (sinogram subtraction is not valid without correction)
    """
    # Full-resolution mean over projections (streaming reductions, no large temp).
    mean_map = minuend.mean(axis=0, dtype=np.float64)
    if subtrahend is not None:
        mean_map = mean_map - subtrahend.mean(axis=0, dtype=np.float64)   # (n_rows, n_ch)

    # Per-sample fractions on a projection subsample.
    Am = minuend[::proj_stride]                                  # object reference (view)
    Es = Am - subtrahend[::proj_stride] if subtrahend is not None else Am
    neg = Es < 0
    thr_obj = float(np.percentile(Am, object_pct))
    obj = Am > thr_obj
    n_obj, n_air = int(obj.sum()), int((~obj).sum())

    std_map = Es.std(axis=0, dtype=np.float64)                   # per-element fluctuation (subsample)
    snr = np.abs(mean_map) / (std_map + 1e-12)
    scale = float(np.percentile(np.abs(mean_map), 95)) + 1e-12

    return {
        "neg_frac_sample": float(neg.mean()),
        "neg_frac_through_object": float((neg & obj).sum() / max(n_obj, 1)),
        "neg_frac_through_air": float((neg & ~obj).sum() / max(n_air, 1)),
        "neg_frac_meanmap": float((mean_map < 0).mean()),
        "neg_frac_meanmap_strict": float((mean_map < -0.02 * scale).mean()),
        "median_meanmap_snr": float(np.median(snr)),
        "mean_map_min": float(mean_map.min()),
        "mean_map_median": float(np.median(mean_map)),
        "_mean_map": mean_map,          # kept for the figure; stripped before JSON
        "_sample_hist": np.histogram(Es.ravel(), bins=200)[0].tolist(),
        "_sample_hist_edges": np.histogram(Es.ravel(), bins=200)[1].tolist(),
    }


def _verdict(meanmap_negs):
    """Interpret the worst exclusive bin's projection-mean negativity."""
    worst = max(meanmap_negs) if meanmap_negs else 0.0
    if worst < 0.02:
        return ("quantum-noise-like", "the projection-averaged exclusive sinogram is non-negative "
                "almost everywhere, so the per-sample A<B violations are zero-mean noise that "
                "reconstruction integrates away — sinogram subtraction recovers signal.")
    if worst < 0.15:
        return ("mostly-noise-with-structure", "most detector elements average non-negative, but a "
                "non-trivial minority stay negative — largely noise with some systematic component.")
    return ("systematic-gain-bias", "many detector elements have a persistently negative "
            "projection-averaged exclusive value, i.e. a bias that survives reconstruction — "
            "sinogram subtraction is not valid here without a per-threshold gain correction.")


# ═══════════════════════════════════════════════════════════════════════
# Reconstruction helper
# ═══════════════════════════════════════════════════════════════════════
def _recon_slab(sino, geom, z_sel):
    """SIRT (or configured method) reconstruction of the z-slab from a full sinogram."""
    return reconstruct_helical_stack(
        sino, geom, z_sel, method="astra", n_pixels=N_PIXELS,
        filter_name=FILTER_NAME, geometry_model=GEOMETRY_MODEL,
        z_weighting=Z_WEIGHTING, wavelet_ring_threshold=WAVELET_RING_THRESHOLD,
        algorithm=RECON_METHOD, n_iter=N_ITER).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Plotting  (lazy matplotlib)
# ═══════════════════════════════════════════════════════════════════════
def _save_negativity_fig(neg, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for k in range(4):
        mm = neg[k]["_mean_map"]
        v = np.percentile(np.abs(mm), 99) + 1e-9
        im = ax[0, k].imshow(mm, aspect="auto", cmap="coolwarm", vmin=-v, vmax=v)
        ax[0, k].set_title(f"E{k+1} [{EXCL_TAGS[k]} keV]  mean over proj\n"
                           f"meanmap<0: {100*neg[k]['neg_frac_meanmap']:.1f}%")
        ax[0, k].set_xlabel("channel"); ax[0, k].set_ylabel("row")
        fig.colorbar(im, ax=ax[0, k], fraction=0.046)
        edges = np.array(neg[k]["_sample_hist_edges"])
        cx = 0.5 * (edges[:-1] + edges[1:])
        ax[1, k].bar(cx, neg[k]["_sample_hist"], width=(edges[1] - edges[0]), color="steelblue")
        ax[1, k].axvline(0, color="k", lw=0.8)
        ax[1, k].set_yscale("log")
        ax[1, k].set_title(f"sample values  (neg {100*neg[k]['neg_frac_sample']:.1f}%)")
        ax[1, k].set_xlabel("exclusive sinogram value")
    fig.suptitle("Exclusive-sinogram negativity — is A−B a bias or zero-mean noise? "
                 "(top: projection mean per detector element; bottom: sample histogram)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); print(f"[fig] {path}")


def _save_panels(cum_stack, sin_stack, img_stack, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = cum_stack.shape[1] // 2
    rows = [("cumulative A..D", cum_stack, LABELS),
            ("sino-domain exclusive", sin_stack, EXCL_TAGS),
            ("image-domain exclusive", img_stack, EXCL_TAGS)]
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for ri, (name, vol, tags) in enumerate(rows):
        for ci in range(4):
            img = vol[ci, z]; lo, hi = np.percentile(img, [2, 98])
            ax[ri, ci].imshow(img, cmap="gray", vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
            ax[ri, ci].set_title(f"{name}\n[{tags[ci]}]", fontsize=8)
            ax[ri, ci].axis("off")
    fig.suptitle("Sinogram-domain vs image-domain exclusive bins (mid slice)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); print(f"[fig] {path}")


def _save_sino_vs_image(sin_stack, img_stack, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = sin_stack.shape[1] // 2
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for k in range(4):
        d = sin_stack[k, z] - img_stack[k, z]
        v = np.percentile(np.abs(d), 99) + 1e-9
        im = ax[k].imshow(d, cmap="coolwarm", vmin=-v, vmax=v)
        ax[k].set_title(f"E{k+1} [{EXCL_TAGS[k]}]  sino − image", fontsize=9)
        ax[k].axis("off"); fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.suptitle("Where subtract-then-SIRT differs from SIRT-then-subtract "
                 "(nonlinearity / non-negativity)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); print(f"[fig] {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    import scipy.io as sio
    import h5py

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Sinogram-domain threshold-separation investigation "
          f"[{RECON_METHOD.upper()} x{N_ITER}] ===")

    # Geometry + z-slab (balanced weighting spans 2 rotations → trim 1 rotation each end).
    desc = sio.loadmat(str(DESC_PATH), struct_as_record=True, squeeze_me=False)
    geom = build_geom(desc["descriptor"].flat[0], geo_dir=_REPO_ROOT / "geometry",
                      channels_flipped=True)
    z_targets, z_spacing = z_targets_for_full_scan(geom, oversample=1, end_margin_rotations=1.0)
    half = N_SLAB_SLICES // 2
    c = int(np.clip(SLICE_IDX, half, len(z_targets) - half - 1))
    z_sel = z_targets[c - half: c + half + 1]
    print(f"  reconstructable slices: {len(z_targets)}  |  study slab: "
          f"{len(z_sel)} slices at z={z_sel[0]:+.2f}..{z_sel[-1]:+.2f} mm")

    # Incremental load/recon/subtract: at most 2 cumulative sinograms in RAM at once.
    cum = {}                       # label -> reconstructed cumulative slab (raw attenuation)
    sinoexc = {}                   # bin idx 0..3 -> reconstructed sino-domain exclusive slab
    neg = {}                       # bin idx 0..3 -> negativity stats
    sino_prev, label_prev = None, None
    with h5py.File(str(DATA_PATH), "r") as f:
        for li, lab in enumerate(LABELS):
            sino = _load_threshold(f, li)
            print(f"  loaded {lab}: {sino.shape}  [{sino.min():.1f}, {sino.max():.1f}]")
            if li == 0:
                geom["spike_mask"] = detect_defect_channels(
                    sino, spike_mad_k=SPIKE_MAD_K, ipr_mad_k=IPR_MAD_K)
                print(f"  {int(geom['spike_mask'].sum())} defect channels masked (from A)")

            print(f"  reconstructing cumulative {lab} ...")
            cum[lab] = _recon_slab(sino, geom, z_sel)

            if sino_prev is not None:
                k = li - 1                         # exclusive bin E_{k+1} = prev − current
                print(f"  exclusive E{k+1} = {label_prev} - {lab}: negativity + reconstruct ...")
                neg[k] = negativity_stats(sino_prev, sino)
                sino_prev -= sino                  # in place: sino_prev now holds the exclusive bin
                sinoexc[k] = _recon_slab(sino_prev, geom, z_sel)
                del sino_prev
            sino_prev, label_prev = sino, lab

    # Top window E4 = D: sino-domain recon == cumulative D; negativity is trivial (no subtraction).
    sinoexc[3] = cum["D"]
    neg[3] = negativity_stats(sino_prev)           # sino_prev holds D
    del sino_prev

    # ── Masks / features from the (HU-calibrated) cumulative A volume ─────
    cumA = cum["A"]
    calib = auto_hu_calibrate(cumA, fov_mm=FOV_MM)
    cumA_hu = apply_hu_calibration(cumA, calib)
    body = auto_body_mask(cumA_hu)
    water = auto_water_mask(cumA_hu, body)
    water_med = float(np.median(cumA_hu[body])) if body.any() else 0.0
    xy_mm = FOV_MM / N_PIXELS
    feats = auto_detect_inserts(cumA_hu, body, water_med, INSERT_CONTRAST_HU, xy_mm, INSERT_DIAM_MM)
    print(f"  masks: body={int(body.sum())} water={int(water.sum())}  "
          f"features={len(feats)}  mu_water(A)={calib['mu_water']:.4g}")

    # ── Build the three stacks (raw attenuation; directly comparable) ────
    cum_stack = np.stack([cum[l] for l in LABELS])
    sin_stack = np.stack([sinoexc[k] for k in range(4)])
    img_stack = np.stack([cum["A"] - cum["B"], cum["B"] - cum["C"],
                          cum["C"] - cum["D"], cum["D"]])

    def _noise(stack):
        return np.sqrt(np.maximum(np.diag(noise_from_highpass(stack, water, NOISE_HP_SIGMA)), 0))

    sd_cum, sd_sin, sd_img = _noise(cum_stack), _noise(sin_stack), _noise(img_stack)

    # per-bin sino-vs-image agreement + CNR (label-free, on the auto features)
    nz, ny, nx = cum_stack.shape[1:]
    agree, cnr_sin, cnr_img = [], [], []
    wm_sin = [sin_stack[b][water].mean() for b in range(4)]
    wm_img = [img_stack[b][water].mean() for b in range(4)]
    for b in range(4):
        a, cimg = sin_stack[b][body].ravel(), img_stack[b][body].ravel()
        corr = float(np.corrcoef(a, cimg)[0, 1]) if a.size > 2 else float("nan")
        mad = float(np.mean(np.abs(sin_stack[b][body] - img_stack[b][body])))
        rel = mad / (float(np.mean(np.abs(img_stack[b][body]))) + 1e-12)
        agree.append({"corr": corr, "mean_abs_diff": mad, "rel_diff": rel})
        cs = [cnr_sd(sin_stack[b][disk_mask((nz, ny, nx), r["z"], r["cy"], r["cx"], r["r"])],
                     wm_sin[b], sd_sin[b]) for r in feats]
        ci = [cnr_sd(img_stack[b][disk_mask((nz, ny, nx), r["z"], r["cy"], r["cx"], r["r"])],
                     wm_img[b], sd_img[b]) for r in feats]
        cnr_sin.append(float(np.nanmean(cs)) if cs else float("nan"))
        cnr_img.append(float(np.nanmean(ci)) if ci else float("nan"))

    meanmap_negs = [neg[k]["neg_frac_meanmap"] for k in range(3)]   # E1..E3 (E4 = top window)
    verdict, verdict_text = _verdict(meanmap_negs)

    metrics = {
        "config": {"recon_method": RECON_METHOD, "n_iter": N_ITER, "z_weighting": Z_WEIGHTING,
                   "geometry_model": GEOMETRY_MODEL, "n_slab_slices": int(len(z_sel)),
                   "slab_z_mm": [float(z_sel[0]), float(z_sel[-1])]},
        "negativity": {f"E{k+1}": {kk: vv for kk, vv in neg[k].items() if not kk.startswith("_")}
                       for k in range(4)},
        "verdict": {"class": verdict, "explanation": verdict_text,
                    "meanmap_neg_fraction_E1_E3": meanmap_negs},
        "noise_sd": {"cumulative": sd_cum.tolist(), "sino_exclusive": sd_sin.tolist(),
                     "image_exclusive": sd_img.tolist()},
        "sino_vs_image_agreement": {f"E{b+1}": agree[b] for b in range(4)},
        "mean_cnr": {"sino_exclusive": cnr_sin, "image_exclusive": cnr_img, "n_features": len(feats)},
    }
    (OUT_DIR / "sinogram_subtraction_metrics.json").write_text(json.dumps(metrics, indent=2))

    _save_negativity_fig(neg, OUT_DIR / "sinogram_subtraction_negativity.png")
    _save_panels(cum_stack, sin_stack, img_stack, OUT_DIR / "sinogram_subtraction_panels.png")
    _save_sino_vs_image(sin_stack, img_stack, OUT_DIR / "sinogram_subtraction_sino_vs_image.png")
    _write_findings(metrics, OUT_DIR / "sinogram_subtraction_findings.md")
    print(f"\n[done] metrics  → {OUT_DIR / 'sinogram_subtraction_metrics.json'}")
    print(f"[done] findings → {OUT_DIR / 'sinogram_subtraction_findings.md'}")
    print(f"[verdict] {verdict}: worst E1..E3 projection-mean negativity "
          f"= {100*max(meanmap_negs):.1f}%")


def _write_findings(m, path):
    neg, agree = m["negativity"], m["sino_vs_image_agreement"]
    sd = m["noise_sd"]
    L = ["# Sinogram-domain threshold separation — findings", "",
         "*Standalone investigation. Does NOT modify invariant #3; it tests, in the "
         "projection domain, whether that invariant's premise holds for this data.*",
         f"*Reconstruction: {m['config']['recon_method'].upper()} ×{m['config']['n_iter']}, "
         f"{m['config']['z_weighting']} weighting, {m['config']['n_slab_slices']}-slice slab.*", "",
         "## Q1 — Is A−B a systematic bias or zero-mean noise?", "",
         f"**Verdict: {m['verdict']['class']}.** {m['verdict']['explanation']}", "",
         "| bin | sample neg % | neg % (object rays) | neg % (air rays) | **proj-mean neg %** | median mean/σ |",
         "|---|---:|---:|---:|---:|---:|"]
    for k in range(4):
        e = neg[f"E{k+1}"]
        L.append(f"| E{k+1} [{EXCL_TAGS[k]} keV] | {100*e['neg_frac_sample']:.1f} | "
                 f"{100*e['neg_frac_through_object']:.1f} | {100*e['neg_frac_through_air']:.1f} | "
                 f"**{100*e['neg_frac_meanmap']:.1f}** | {e['median_meanmap_snr']:.3f} |")
    L += ["",
          "The **proj-mean neg %** column is the discriminator: it is the fraction of detector "
          "elements whose exclusive value, *averaged over all projections*, is negative. Near-zero "
          "means the per-sample negatives cancel (noise) and reconstruction recovers signal; a large "
          "value is a per-element bias that survives reconstruction. (E4 = D is the top window, no "
          "subtraction, shown for reference.)", "",
          "## Q2 — Sinogram-domain vs image-domain exclusive bins (SIRT is nonlinear)", "",
          "| bin | noise SD sino | noise SD image | correlation | mean\\|Δ\\| | rel. diff |",
          "|---|---:|---:|---:|---:|---:|"]
    for b in range(4):
        L.append(f"| E{b+1} [{EXCL_TAGS[b]}] | {sd['sino_exclusive'][b]:.4g} | "
                 f"{sd['image_exclusive'][b]:.4g} | {agree[f'E{b+1}']['corr']:.4f} | "
                 f"{agree[f'E{b+1}']['mean_abs_diff']:.4g} | {agree[f'E{b+1}']['rel_diff']:.3f} |")
    L += ["",
          "Under FBP these two would be identical (linearity). Any difference here is the effect of "
          "the SIRT non-negativity constraint on near-zero / negative exclusive bins — largest where "
          "the exclusive signal sits near the clamp.", "",
          "## Mean CNR on auto-detected features (label-free)", "",
          f"features: {m['mean_cnr']['n_features']}",
          "| bin | CNR sino | CNR image |", "|---|---:|---:|"]
    for b in range(4):
        L.append(f"| E{b+1} [{EXCL_TAGS[b]}] | {m['mean_cnr']['sino_exclusive'][b]:.3g} | "
                 f"{m['mean_cnr']['image_exclusive'][b]:.3g} |")
    L += ["", "## Figures",
          "- `sinogram_subtraction_negativity.png` — per-bin projection-mean map + sample histogram (Q1).",
          "- `sinogram_subtraction_panels.png` — cumulative vs sino-exclusive vs image-exclusive (mid slice).",
          "- `sinogram_subtraction_sino_vs_image.png` — where the two subtraction domains diverge (Q2).", "",
          "## Relation to the image-domain study",
          "This complements `image_subtraction_investigation.py` / "
          "the image-domain study: that study "
          "asked whether image-domain separation improves *image quality* (it does not). This one "
          "asks whether the *projection-domain* subtraction is physically valid and whether the "
          "subtraction domain matters under nonlinear reconstruction — the inputs to material "
          "decomposition, not image quality."]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
