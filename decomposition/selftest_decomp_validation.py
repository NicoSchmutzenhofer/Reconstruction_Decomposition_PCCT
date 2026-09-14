"""
selftest_decomp_validation.py -- synthetic checks for the DEP-V5 validation metrics.

No data, no GPU.  Builds a phantom whose truth is known -- three layers, three HU-matched
pairs each, one pair per layer carrying no agent -- and verifies that the assignment and
every metric recovers it, including the failure modes they exist to catch.

    python -m decomposition.selftest_decomp_validation
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decomposition.decomp_validation import (
    AGENTS, BIN_DOMAIN_BY_SOURCE, CONC_RELATIVE, LAYER_BASE_HU_140, ROLES, RQ1_AGENTS,
    SOURCE_2BIN, ValidationConfig, assign_layers_and_roles, bin_domain_for,
    anti_correlated_agents, carrier_insert, cross_layer_metrics, doped_pairs,
    inverted_agents, is_rankable, linearity_metrics, measure_rois, response_fraction,
    response_metrics, rq1_summary, run_score, separation_metrics, specificity_metrics,
    _components_of_label, _mean_of_finite)

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<54s} {detail}")
    if not ok:
        FAILS.append(name)


def close(a, b, tol):
    return np.isfinite(a) and abs(a - b) <= tol


# ═════════════════════════════════════════════════════════════════════
# A synthetic DEP-V5: 3 layers x 3 pairs x 2 members
# ═════════════════════════════════════════════════════════════════════
NZ, N, PX = 90, 128, 1.0
LAYOUT = {}          # (layer_index) -> layer letter
TRUE_I = {"base": 0.0, "low": 0.0017, "high": 0.0017 * 3.1}       # g/cm^3
TRUE_H = {"base": 0.0, "low": 0.055, "high": 0.055 * 3.1}


def build():
    """Volumes + insert list with known truth. Layer k occupies slices 30k..30k+29."""
    rng = np.random.default_rng(11)
    hu = np.zeros((NZ, N, N), np.float32)
    iod = np.zeros((NZ, N, N), np.float32)
    ha = np.zeros((NZ, N, N), np.float32)
    labels = np.zeros((NZ, N, N), np.uint16)
    yy, xx = np.mgrid[:N, :N]

    # Layer index 0/1/2 maps to letters B/C/A -- deliberately NOT in order, so the test
    # proves the identification comes from HU rather than from the segmentation order.
    order = ["B", "C", "A"]
    inserts, gid = [], 0
    for li, letter in enumerate(order):
        LAYOUT[li] = letter
        k0, k1 = 30 * li, 30 * li + 29
        base = LAYER_BASE_HU_140[letter]
        hu[k0:k1 + 1] = base * 0.0 + 0.0            # body at 0 HU
        for pi, role in enumerate(ROLES):
            lab = li * 3 + pi + 1
            # HU: base material, plus the agent's increment. The two members are equal by
            # construction -- that IS the phantom's design.
            lvl = {"base": 0.0, "low": 55.0, "high": 165.0}[role]
            for m in range(2):
                ang = 2 * np.pi * (pi * 2 + m) / 6
                cy = N / 2 + (34 - 9 * pi) * np.sin(ang)
                cx = N / 2 + (34 - 9 * pi) * np.cos(ang)
                disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= 5.0 ** 2
                hu[k0:k1 + 1, disc] = base + lvl
                # member 0 carries iodine, member 1 carries calcium
                iod[k0:k1 + 1, disc] = TRUE_I[role] if m == 0 else 0.0
                ha[k0:k1 + 1, disc] = TRUE_H[role] if m == 1 else 0.0
                seed = (yy - cy) ** 2 + (xx - cx) ** 2 <= 2.0 ** 2
                labels[k0:k1 + 1, seed] = lab
                inserts.append({"cy": float(cy), "cx": float(cx), "k_lo": k0, "k_hi": k1,
                                "geom_id": gid, "pair_label": lab, "member": m,
                                "layer_index": li, "radius_mm": 5.0,
                                "truth_layer": letter, "truth_role": role,
                                "truth_agent": "Iodine" if m == 0 else "HA"})
                gid += 1
    hu += rng.normal(0, 4.0, hu.shape).astype(np.float32)
    iod += rng.normal(0, 2e-4, iod.shape).astype(np.float32)
    ha += rng.normal(0, 6e-3, ha.shape).astype(np.float32)
    return hu, {"Iodine": iod, "HA": ha}, labels, inserts


def main():
    print("=" * 70)
    print("decomp_validation -- synthetic self-test")
    print("=" * 70)
    cfg = ValidationConfig(insert_radius_mm=5.0)
    hu, maps, labels, inserts = build()

    print("\n[1] pair labels split into exactly two members")
    ok = True
    for lab in range(1, 10):
        comps = _components_of_label(labels, lab)
        ok &= len(comps) == 2
    check("every label yields 2 inserts", ok)
    check("18 inserts in total", len(inserts) == 18, f"got {len(inserts)}")

    print("\n[2] layer and role recovered from HU alone")
    hu_vals = [r["mean"] for r in measure_rois(hu, inserts, PX, cfg)]
    assign = assign_layers_and_roles(inserts, hu_vals)
    for ins, h in zip(inserts, hu_vals):
        a = assign[ins["layer_index"]]
        ins["layer"] = a["layer"]
        ins["role"] = a["role_of_pair"][ins["pair_label"]]
    check("layer identity correct (order deliberately scrambled)",
          all(i["layer"] == i["truth_layer"] for i in inserts),
          f"{ {li: a['layer'] for li, a in assign.items()} } vs {LAYOUT}")
    check("base/low/high correct", all(i["role"] == i["truth_role"] for i in inserts))

    print("\n[3] density measured at the right places")
    dens = {m: [r["mean"] for r in measure_rois(v, inserts, PX, cfg)]
            for m, v in maps.items()}
    noise = {m: [r["sd"] for r in measure_rois(v, inserts, PX, cfg)]
             for m, v in maps.items()}
    hi_i = [dens["Iodine"][i["geom_id"]] for i in inserts
            if i["truth_role"] == "high" and i["truth_agent"] == "Iodine"]
    check("iodine recovered in the iodine inserts",
          close(float(np.mean(hi_i)), TRUE_I["high"], 3e-4),
          f"{np.mean(hi_i):.5f} vs {TRUE_I['high']:.5f}")

    print("\n[4] specificity: base lesions contain no agent")
    spec = specificity_metrics(inserts, dens)
    check("iodine bias ~0 in base lesions", abs(spec["Iodine"]["mean"]) < 3e-4,
          f"{spec['Iodine']['mean']:+.2e}")
    check("CaHA bias ~0 in base lesions", abs(spec["HA"]["mean"]) < 6e-3,
          f"{spec['HA']['mean']:+.2e}")

    print("\n[5] separation of HU-matched pairs")
    sep = separation_metrics(inserts, dens, noise)
    doped = [s for s in sep if s["role"] in ("low", "high")]
    check("all doped pairs have opposite-sign agents",
          all(s["sign_ok"] for s in doped), f"{sum(s['sign_ok'] for s in doped)}/{len(doped)}")
    check("separation index is large for doped pairs",
          min(s["sep_index_Iodine"] for s in doped) > 3.0,
          f"min {min(s['sep_index_Iodine'] for s in doped):.1f}")
    base = [s for s in sep if s["role"] == "base"]
    check("base pairs show no separation",
          max(abs(s["delta_Iodine"]) for s in base) < 5e-4,
          f"max abs delta {max(abs(s['delta_Iodine']) for s in base):.2e}")

    print("\n[6] a common-mode failure is CAUGHT")
    # Both members respond to HU instead of material -> same sign in both maps.
    bad = {m: list(v) for m, v in dens.items()}
    for i in inserts:
        if i["truth_role"] != "base":
            bad["Iodine"][i["geom_id"]] = 0.5 * TRUE_I[i["truth_role"]]
            bad["HA"][i["geom_id"]] = 0.5 * TRUE_H[i["truth_role"]]
    bad_sep = [s for s in separation_metrics(inserts, bad, noise)
               if s["role"] in ("low", "high")]
    check("no doped pair passes the sign test",
          not any(s["sign_ok"] for s in bad_sep),
          f"{sum(bool(s['sign_ok']) for s in bad_sep)}/{len(bad_sep)} passed")

    print("\n[7] proportionality against the inferred 0 : 1 : 3.1")
    lin = linearity_metrics(inserts, dens)
    r2 = [e["r2"] for e in lin.values()]
    ratio = [e["measured_high_over_low"] for e in lin.values()]
    check("R^2 ~ 1 for every layer and agent", min(r2) > 0.99, f"min {min(r2):.4f}")
    check("measured high/low ~ 3.1", close(float(np.mean(ratio)), 3.1, 0.25),
          f"{np.mean(ratio):.2f}")

    print("\n[8] cross-layer consistency (base-material independence)")
    xl = cross_layer_metrics(inserts, dens)
    check("iodine CV small across A/B/C", xl["Iodine/high"]["cv"] < 0.06,
          f"CV {xl['Iodine/high']['cv']:.3f}")
    check("CaHA CV small across A/B/C", xl["HA/high"]["cv"] < 0.06,
          f"CV {xl['HA/high']['cv']:.3f}")

    print("\n[9] base-material crosstalk is CAUGHT")
    skew = {m: list(v) for m, v in dens.items()}
    for i in inserts:                      # layer A reads 40 % high -> CV must rise
        if i["truth_layer"] == "A":
            skew["Iodine"][i["geom_id"]] *= 1.4
    check("skewed layer raises the CV",
          cross_layer_metrics(inserts, skew)["Iodine/high"]["cv"] > 0.12,
          f"CV {cross_layer_metrics(inserts, skew)['Iodine/high']['cv']:.3f}")

    print("\n[10] unit-free metrics for the 4-bin vs 2-bin question")
    # leak_ratio: the agent's reading on the pair member that does NOT carry it, over the
    # member that does. Must be ~0 here, where each member carries exactly one agent.
    check("iodine leak ratio ~0 on a clean separation",
          max(s["leak_ratio_Iodine"] for s in doped) < 0.15,
          f"max {max(s['leak_ratio_Iodine'] for s in doped):.3f}")
    # The common-mode failure of [6] is the case leak_ratio exists to score: both members
    # read the same, so the ratio goes to 1.
    bad_leak = [s["leak_ratio_Iodine"] for s in bad_sep]
    check("a pure HU tracker drives the leak ratio to ~1",
          min(bad_leak) > 0.85, f"min {min(bad_leak):.3f}")
    check("leak_frac is dimensionless (unchanged by a unit rescale)",
          close(specificity_metrics(inserts, {m: [1000.0 * x for x in v]
                                              for m, v in dens.items()})["Iodine"]["leak_frac"],
                spec["Iodine"]["leak_frac"], 1e-9),
          f"{spec['Iodine']['leak_frac']:.4f}")
    # A SIGN-INVERTED map: every reading negated. The high-lesion reference then goes
    # negative, and while leak_frac took it signed, the ratio came out negative -- so on an
    # axis that says lower is better the most inverted run in the set scored best. A base
    # lesion holds no agent, so only distance from zero can mean anything there.
    flip = specificity_metrics(inserts, {m: [-x for x in v] for m, v in dens.items()})
    check("an inverted map cannot produce a negative leak fraction",
          flip["Iodine"]["leak_frac"] >= 0
          and close(flip["Iodine"]["leak_frac"], spec["Iodine"]["leak_frac"], 1e-9),
          f"inverted {flip['Iodine']['leak_frac']:+.4f} vs "
          f"upright {spec['Iodine']['leak_frac']:+.4f}")
    check("but the inversion is REPORTED, not silently absorbed",
          flip["Iodine"]["reference_negative"] and not spec["Iodine"]["reference_negative"],
          f"high_reference {flip['Iodine']['high_reference']:+.4f}")
    check("rq1_summary carries the flag to the figure and the table",
          rq1_summary({"rq1_iodine": {"specificity": flip}})["reference_negative"])
    # The same defect sat in three other places: every metric that picks "the member
    # carrying the agent" used a SIGNED max, which on a map below zero returns the member
    # with the LEAST agent. All four RQ1 numbers must survive a global sign flip unchanged.
    neg = {m: [-x for x in v] for m, v in dens.items()}
    up = separation_metrics(inserts, dens, noise)
    dn = separation_metrics(inserts, neg, noise)
    up_l = [s["leak_ratio_Iodine"] for s in up if s["role"] in ("low", "high")]
    dn_l = [s["leak_ratio_Iodine"] for s in dn if s["role"] in ("low", "high")]
    check("leak_ratio is unmoved by a global sign flip",
          all(close(a, b, 1e-9) for a, b in zip(up_l, dn_l)),
          f"upright max {max(up_l):.3f}, inverted max {max(dn_l):.3f}")
    check("and it cannot exceed 1, the value the axis calls total failure",
          max(dn_l) <= 1.0 + 1e-9, f"max {max(dn_l):.3f}")
    lu, ld = linearity_metrics(inserts, dens), linearity_metrics(inserts, neg)
    check("the linearity fit picks the same three inserts either way",
          all(close(abs(lu[k]["measured_high_over_low"]),
                    abs(ld[k]["measured_high_over_low"]), 1e-6) for k in lu),
          f"hi/lo {lu['Iodine/A']['measured_high_over_low']:.2f} vs "
          f"{ld['Iodine/A']['measured_high_over_low']:.2f}")
    xu, xd = cross_layer_metrics(inserts, dens), cross_layer_metrics(inserts, neg)
    check("cross-layer CV is unmoved by a global sign flip",
          all(close(xu[k]["cv"], xd[k]["cv"], 1e-9) for k in xu),
          f"Iodine/high CV {xu['Iodine/high']['cv']:.4f} vs {xd['Iodine/high']['cv']:.4f}")
    # The figures pick an INSERT by the same rule the metrics pick a value. If they drift,
    # the contact sheet outlines one lesion while the fit describes another -- and that
    # sheet exists precisely to audit the selection.
    hi_mem = [i for i in inserts if i["truth_role"] == "high"
              and i["pair_label"] == next(j["pair_label"] for j in inserts
                                          if j["truth_role"] == "high")]
    pick_up = carrier_insert(hi_mem, dens["Iodine"])
    pick_dn = carrier_insert(hi_mem, neg["Iodine"])
    check("the figures select the same insert as the metrics, either sign",
          pick_up["geom_id"] == pick_dn["geom_id"]
          and pick_up["truth_agent"] == "Iodine",
          f"geom_id {pick_up['geom_id']} carries {pick_up['truth_agent']}")

    print("\n[11] RQ1 restricts BOTH sides to iodine")
    full_run = {"density": dens, "density_sd": noise,
                "separation": separation_metrics(inserts, dens, noise),
                "rq1_iodine": {
                    "separation": separation_metrics(inserts, dens, noise,
                                                     agents=RQ1_AGENTS),
                    "specificity": specificity_metrics(inserts, dens, agents=RQ1_AGENTS),
                    "linearity": linearity_metrics(inserts, dens, agents=RQ1_AGENTS),
                    "cross_layer": cross_layer_metrics(inserts, dens, agents=RQ1_AGENTS)}}
    check("the restriction drops HA from every RQ1 block",
          not any(k.startswith("HA") for k in full_run["rq1_iodine"]["linearity"])
          and "HA" not in full_run["rq1_iodine"]["specificity"]
          and not any(k.startswith("HA")
                      for k in full_run["rq1_iodine"]["cross_layer"]),
          f"linearity keys {sorted(full_run['rq1_iodine']['linearity'])}")

    # The rigging this exists to prevent: a run whose CALCIUM map is excellent and whose
    # IODINE map is poor must not be flattered by averaging the two, because the 2-bin
    # side it is being compared with has no calcium map to average in.
    rigged = {"Iodine": [0.5 * v for v in dens["Iodine"]], "HA": list(dens["HA"])}
    for i in inserts:                       # iodine now barely separates the pairs
        if i["truth_role"] != "base":
            rigged["Iodine"][i["geom_id"]] = 0.9 * TRUE_I[i["truth_role"]] \
                if i["truth_agent"] == "Iodine" else 0.8 * TRUE_I[i["truth_role"]]
    both = run_score({"separation": separation_metrics(inserts, rigged, noise)},
                     agents=AGENTS).index
    iod_only = run_score({"separation": separation_metrics(inserts, rigged, noise,
                                                           agents=RQ1_AGENTS)},
                         agents=RQ1_AGENTS).index
    check("a good-HA / poor-iodine run scores worse on iodine alone",
          iod_only < both, f"iodine-only {iod_only:.2f} vs both-agent {both:.2f}")

    print("\n[12] a rank-limited run is scored but not ranked")
    vendor = {"density": {"Iodine": dens["Iodine"]},
              "density_sd": {"Iodine": noise["Iodine"]},
              "separation": separation_metrics(inserts,
                                               {"Iodine": dens["Iodine"]},
                                               {"Iodine": noise["Iodine"]})}
    vendor["rq1_iodine"] = {
        "separation": vendor["separation"],
        "specificity": specificity_metrics(inserts, {"Iodine": dens["Iodine"]},
                                           agents=RQ1_AGENTS),
        "linearity": linearity_metrics(inserts, {"Iodine": dens["Iodine"]},
                                       agents=RQ1_AGENTS),
        "cross_layer": cross_layer_metrics(inserts, {"Iodine": dens["Iodine"]},
                                           agents=RQ1_AGENTS)}
    check("sign_ok is ABSENT, not False, without a second agent",
          not any("sign_ok" in s for s in vendor["separation"]))
    check("excluded from the estimator ranking", not is_rankable(vendor)
          and is_rankable({"density": dens}))
    summary = rq1_summary(vendor)
    check("but every RQ1 metric is still produced",
          all(np.isfinite(summary[k]) for k in
              ("leak_ratio", "base_leak_frac", "cross_layer_cv", "ratio_error")),
          ", ".join(f"{k}={v:.3f}" for k, v in summary.items()))

    print("\n[13] each source is solved from its own measurement stack")
    # The three sources do not hold the same quantity: our thresholds are nested and fed
    # as-is, the vendor's are differenced into separate bins, and monoenergetic keV images
    # have nothing to subtract. One shared bin_domain would silently misread two of them.
    vcfg = ValidationConfig()
    got = {s: bin_domain_for(vcfg, s) for s in ("own", "wfbp", "vmi")}
    check("own -> cumulative, wfbp -> exclusive, vmi -> direct",
          got == {"own": "cumulative", "wfbp": "exclusive", "vmi": "direct"}, str(got))
    check("an unknown source falls back to cumulative, never to subtraction",
          bin_domain_for(vcfg, "something_new") == "cumulative")
    check("the global override reaches every source",
          {s: bin_domain_for(ValidationConfig(bin_domain="direct"), s)
           for s in ("own", "wfbp", "vmi")} == {"own": "direct", "wfbp": "direct",
                                                "vmi": "direct"})
    one = ValidationConfig()
    one.bin_domain_by_source["wfbp"] = "cumulative"
    check("a per-source override moves only that source",
          bin_domain_for(one, "wfbp") == "cumulative"
          and bin_domain_for(one, "own") == "cumulative"
          and bin_domain_for(one, "vmi") == "direct")
    check("the imported vendor source has no domain to solve in",
          SOURCE_2BIN not in BIN_DOMAIN_BY_SOURCE)

    print("\n[14] the pose check compares a common PHYSICAL z, not slice index")
    # An imported 120-slice slab against a 2232-slice vendor series reported a 141 mm
    # displacement for a phantom that had not moved: the two middle slices sat ~400 mm
    # apart, so the check compared different anatomy. A false alarm that large is worse
    # than no check, because it teaches the reader to ignore the real one.
    from decomposition.decomp_validation import _sitk
    from reconstruction.recon_comparison import _check_pose_against_reference
    sitk = _sitk()
    if sitk is None:
        check("SimpleITK unavailable -- pose check skipped", True)
    else:
        # NOT named PX: assigning it anywhere in main() would make the module-level PX
        # a local for the whole function and break the earlier sections.
        px, n = 1.0, 192
        yy, xx = np.mgrid[:n, :n]

        def _vol(nz, z0, dx_mm=0.0):
            cy, cx = n / 2, n / 2 + dx_mm / px
            img = np.full((n, n), -1000.0)
            img[(((yy - cy) / 60) ** 2 + ((xx - cx) / 85) ** 2) <= 1.0] = 40.0
            img[((yy - cy - 40) ** 2 + (xx - cx) ** 2) <= 14 ** 2] = 700.0   # breaks symmetry
            v = np.repeat(img[None], nz, axis=0).astype(np.float32)
            im = sitk.GetImageFromArray(v)
            im.SetSpacing((px, px, 0.4))
            im.SetOrigin((-96.0, -96.0, z0))
            return v, im, px

        full = _vol(400, -1600.0)
        slab = _vol(120, -1486.0)              # same object, a slab of it
        d = _check_pose_against_reference({"wfbp": full, "s2bin": slab})
        check("a slab of the SAME object reports no displacement",
              d.get("s2bin", {}).get("distance_mm", 9e9) < 1.0,
              f"{d.get('s2bin', {}).get('distance_mm', float('nan')):.2f} mm")
        moved = _vol(120, -1486.0, dx_mm=12.0)  # a slab that really has moved
        d2 = _check_pose_against_reference({"wfbp": full, "s2bin": moved})
        check("a genuinely shifted slab is still caught",
              abs(d2.get("s2bin", {}).get("distance_mm", 0.0) - 12.0) < 1.5,
              f"{d2.get('s2bin', {}).get('distance_mm', float('nan')):.2f} mm, truth 12.0")

    print("\n[15] a pair that could not be measured is dropped, not failed")
    # An ROI that falls outside a source's volume gives mean = sd = NaN, so both deltas are
    # NaN and `NaN * NaN < 0` is False -- the pair would be reported as a failed sign test,
    # i.e. as the solver tracking HU, on the strength of a measurement that never happened.
    # It also made every aggregate emit "Mean of empty slice" once per run.
    # One member each from two DIFFERENT pairs -- losing both members of one pair would
    # only prove that a single pair drops out.
    hi_pairs = sorted({i["pair_label"] for i in inserts if i["truth_role"] == "high"})[:2]
    lost = [next(i["geom_id"] for i in inserts if i["pair_label"] == p) for p in hi_pairs]
    gone = {m: list(v) for m, v in dens.items()}
    gone_sd = {m: list(v) for m, v in noise.items()}
    for m in AGENTS:
        for g in lost:
            gone[m][g] = float("nan")
            gone_sd[m][g] = float("nan")
    lost_sep = separation_metrics(inserts, gone, gone_sd)
    n_all = len([s for s in lost_sep if s["role"] in ("low", "high")])
    kept = doped_pairs(lost_sep)
    check("the unmeasured pairs are excluded from the average",
          len(kept) == n_all - 2, f"{len(kept)}/{n_all} pairs kept")
    check("they carry no sign_ok verdict at all",
          not any("sign_ok" in s for s in lost_sep if not s.get("measured", True)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sc = run_score({"separation": lost_sep})
    check("the surviving pairs still score 100 % sign-correct", close(sc.sign, 1.0, 1e-9),
          f"sign {sc.sign:.2f}, sep idx {sc.index:.2f}")
    check("and no empty-slice warning is emitted",
          not [w for w in caught if "empty slice" in str(w.message)],
          f"{len(caught)} warning(s)")
    check("an all-NaN average is NaN, not an error",
          not np.isfinite(_mean_of_finite([np.nan, None, np.nan]))
          and close(_mean_of_finite([np.nan, 2.0, 4.0]), 3.0, 1e-9))

    print("\n[16] image panels: one window, real outlines, honest scale bars")
    # The image figures make claims a plot cannot: that two panels are comparable. That
    # rests entirely on the window being computed ONCE over both, so it is checked here
    # rather than trusted -- a per-panel window would rescale a noisy map into looking
    # like a clean one, which is the difference those figures exist to show.
    from reconstruction import figure_panels as fpx
    quiet = np.random.default_rng(0).normal(0, 1.0, 8000)
    noisy = np.random.default_rng(1).normal(0, 6.0, 8000)
    w_joint = fpx.shared_window([quiet, noisy])
    w_quiet = fpx.shared_window([quiet])
    check("one window over both is wider than the quiet one alone",
          (w_joint[1] - w_joint[0]) > 2.5 * (w_quiet[1] - w_quiet[0]),
          f"joint {w_joint[1] - w_joint[0]:.1f} vs quiet {w_quiet[1] - w_quiet[0]:.1f}")
    spike = np.concatenate([quiet, [1e6]])
    check("a single hot voxel does not set the window",
          fpx.shared_window([spike])[1] < 10.0,
          f"vmax {fpx.shared_window([spike])[1]:.2f}")
    check("an all-NaN input still yields a usable window",
          all(np.isfinite(v) for v in fpx.shared_window([np.full(10, np.nan)])))
    lo, hi = fpx.symmetric_window([quiet])
    check("a difference window is centred on zero", close(lo, -hi, 1e-12), f"{lo:.2f}..{hi:.2f}")
    check("the window label reads as W/L",
          fpx.window_label(-200.0, 300.0) == "W 500 / L 50 HU",
          fpx.window_label(-200.0, 300.0))

    # A crop centred outside the array must clip, not raise: stage 2 would otherwise lose
    # every figure to one source whose grid does not reach the module.
    for cy, cx in ((50, 50), (-40, 300), (1e6, -1e6)):
        y0, y1, x0, x1 = fpx.crop_box_px((100, 120), cx, cy, 30)
        ok = 0 <= y0 < y1 <= 100 and 0 <= x0 < x1 <= 120
        if not ok:
            break
    check("a crop centred off the array clips instead of raising", ok,
          f"last box {(y0, y1, x0, x1)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fpx.show_slice(ax, np.zeros((100, 100)), vmin=0, vmax=1, pixel_mm=0.4,
                   scale_bar_mm=10.0)
    bars = list(ax.lines)
    check("the scale bar is drawn at its true pixel length",
          len(bars) == 1 and close(abs(np.ptp(bars[0].get_xdata())), 10.0 / 0.4, 1e-6),
          f"{abs(np.ptp(bars[0].get_xdata())):.1f} px for 10 mm at 0.4 mm/px")
    # Too wide for the panel: halved once and RELABELLED, never drawn at the wrong length.
    ax2 = fig.add_subplot(312)
    fpx.show_slice(ax2, np.zeros((40, 40)), vmin=0, vmax=1, pixel_mm=0.4,
                   scale_bar_mm=10.0)
    half = ax2.lines[0] if ax2.lines else None
    check("an over-wide bar is halved and its label follows",
          half is not None and close(abs(np.ptp(half.get_xdata())), 5.0 / 0.4, 1e-6)
          and any("5 mm" in t.get_text() for t in ax2.texts),
          f"{abs(np.ptp(half.get_xdata())):.1f} px, "
          f"labels {[t.get_text() for t in ax2.texts]}")
    ax3 = fig.add_subplot(313)
    fpx.show_slice(ax3, np.zeros((40, 40)), vmin=0, vmax=1, pixel_mm=8.0,
                   scale_bar_mm=1000.0)
    check("a bar that still cannot fit is omitted rather than drawn wrong", not ax3.lines)
    fpx.outline_circle(ax, (5, 5), 3.0)
    check("an ROI outline is an outline, never a fill",
          len(ax.patches) == 1 and not ax.patches[0].get_fill())
    plt.close(fig)

    print("\n[17] a run that passes every sign test on a map assigned to the WRONG "
          "material")
    # The failure block [6] catches a solver that moves both agents the SAME way. This is
    # the other half, and the sign test alone cannot see it: a solver that splits one
    # attenuation across two columns of OPPOSITE sign passes every pair while neither map
    # tracks the material it is named after.  Built here as the iodine map kept correct and
    # the CaHA map made a negative copy of the total agent content, responding more strongly
    # to iodine (a) than to calcium (b) -- which is the shape an ill-conditioned basis
    # produces, since the near-null direction of M has mixed signs by construction.
    a, b = 0.8 * TRUE_H["high"], 0.25 * TRUE_H["high"]
    rig = {"Iodine": list(dens["Iodine"]), "HA": [0.0] * len(inserts)}
    for i in inserts:
        r, g = i["truth_role"], i["geom_id"]
        frac = TRUE_I[r] / TRUE_I["high"] if TRUE_I["high"] else 0.0
        rig["HA"][g] = -(a if i["truth_agent"] == "Iodine" else b) * frac
    rig_sep = separation_metrics(inserts, rig, noise)
    rig_doped = [s for s in rig_sep if s["role"] in ("low", "high")]
    check("it passes the sign test on every pair, as the real data did",
          all(s["sign_ok"] for s in rig_doped),
          f"{sum(bool(s['sign_ok']) for s in rig_doped)}/{len(rig_doped)}")
    rig_lin = linearity_metrics(inserts, rig)
    rig_resp = response_metrics(rig_lin)
    check("but its CaHA map falls as calcium rises, in every layer",
          rig_resp["HA"]["anti_correlated"] and not rig_resp["HA"]["response_ok"],
          f"{rig_resp['HA']['n_positive']}/{rig_resp['HA']['n_layers']} layers rise")
    check("while the iodine map is still recognised as correct",
          rig_resp["Iodine"]["response_ok"],
          f"{rig_resp['Iodine']['n_positive']}/{rig_resp['Iodine']['n_layers']} layers rise")
    clean_resp = response_metrics(linearity_metrics(inserts, dens))
    check("a correct solve responds upward in both maps",
          all(clean_resp[m]["response_ok"] for m in AGENTS),
          f"{response_fraction({'response': clean_resp}):.2f} of fits")
    # The point of the metric: the ranking must no longer prefer this run.
    rig_run = {"separation": rig_sep, "response": rig_resp,
               "specificity": specificity_metrics(inserts, rig)}
    good_run = {"separation": sep, "response": clean_resp,
                "specificity": specificity_metrics(inserts, dens)}
    check("and the ranking puts the correct solve above it",
          run_score(good_run) > run_score(rig_run),
          f"good {tuple(round(v, 2) for v in run_score(good_run))} vs "
          f"rigged {tuple(round(v, 2) for v in run_score(rig_run))}")
    check("the defect is named for the agent it affects",
          anti_correlated_agents(rig_run) == ["HA"] and not anti_correlated_agents(good_run),
          f"{anti_correlated_agents(rig_run)}")
    # And the other orientation failure, kept separate: a map below zero on the lesions
    # carrying the most agent is not a concentration map, whatever its ratios look like.
    flip = {"Iodine": [-v for v in dens["Iodine"]], "HA": list(dens["HA"])}
    flip_run = {"separation": separation_metrics(inserts, flip, noise),
                "response": response_metrics(linearity_metrics(inserts, flip)),
                "specificity": specificity_metrics(inserts, flip)}
    check("a sign-inverted iodine map is flagged, not scored",
          inverted_agents(flip_run) == ["Iodine"] and not inverted_agents(good_run),
          f"{inverted_agents(flip_run)}")
    check("and it cannot outrank an upright solve",
          run_score(good_run) > run_score(flip_run),
          f"upright {run_score(good_run).upright:.0f} vs "
          f"inverted {run_score(flip_run).upright:.0f}")

    print("\n" + "=" * 70)
    print("FAILURES:", FAILS or "none")
    print("ALL CHECKS PASSED" if not FAILS else "SELF-TEST FAILED")
    print("=" * 70)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
