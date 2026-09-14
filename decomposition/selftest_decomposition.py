"""
selftest_decomposition.py -- synthetic self-test of the decomposition math.

Checks the parts most likely to be subtly wrong, on synthetic data.  Needs numpy; the denoise
and joint tests also use scipy and scikit-image.

    python -m decomposition.selftest_decomposition
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decomposition import material_library as mlib
from decomposition import noise_estimation as ne
from decomposition import denoising as dn
from decomposition.material_decomposition import (
    DecompConfig, cumulative_to_exclusive, decompose, form_bin_measurements,
    solve_ols, solve_wls, stability_report)


# --- Phase A ----------------------------------------------------------------
def test_cumulative_to_exclusive():
    v = np.arange(4 * 6, dtype=float).reshape(4, 2, 3)
    exc = cumulative_to_exclusive(v)
    assert np.allclose(exc[0], v[0] - v[1])
    assert np.allclose(exc[3], v[3])
    assert np.allclose(form_bin_measurements(v, "cumulative"), v)
    print("  cumulative_to_exclusive / form_bin_measurements  OK")


def test_ols_exact_recovery():
    M = mlib.build_M(["SoftTissue", "HA", "Iodine"], 1)
    rng = np.random.default_rng(1)
    x_true = rng.random((3, 2000)) * np.array([[1.0], [0.5], [0.02]])
    err = np.abs(solve_ols(M @ x_true, M) - x_true).max()
    assert err < 1e-6, err
    print(f"  OLS exact noise-free recovery  OK (max err {err:.2e})")


def test_kappa_matches_numpy():
    mats = ["SoftTissue", "HA", "Iodine"]
    rep = stability_report(mlib.build_M(mats, 1), mats)
    assert np.isclose(rep["condition_number"], np.linalg.cond(mlib.build_M(mats, 1)))
    print(f"  stability kappa matches numpy  OK (kappa={rep['condition_number']:.1f})")


# --- Phase B ----------------------------------------------------------------
def test_noise_detects_noisiest_bin():
    """The noise estimator must find whichever bin is worst -- no fixed assumption."""
    rng = np.random.default_rng(3)
    shape = (24, 32, 32)
    sd = np.array([1.0, 1.5, 4.0, 2.0])            # bin index 2 is the noisiest
    B = np.stack([np.full(shape, 5.0) + rng.normal(0, s, shape) for s in sd])
    Sigma = ne.noise_covariance_global(B, size=5, flat_percentile=80.0)
    assert int(np.argmax(np.diag(Sigma))) == 2, np.diag(Sigma)
    print(f"  noise estimator finds the noisiest bin  OK (diag={np.round(np.diag(Sigma), 2)})")


def test_wls_beats_ols():
    M = mlib.build_M(["SoftTissue", "HA", "Iodine"], 1)
    rng = np.random.default_rng(4)
    x_true = rng.random((3, 4000)) * np.array([[1.0], [0.4], [0.02]])
    sd = np.array([0.02, 0.02, 0.02, 0.5])         # last bin very noisy
    B = M @ x_true + rng.normal(0, 1, (4, 4000)) * sd[:, None]
    e_ols = np.mean((solve_ols(B, M) - x_true) ** 2)
    e_wls = np.mean((solve_wls(B, M, np.diag(1.0 / sd ** 2)) - x_true) ** 2)
    assert e_wls < e_ols, (e_wls, e_ols)
    print(f"  WLS beats OLS under heteroscedastic noise  OK (MSE {e_wls:.3g} < {e_ols:.3g})")


def test_denoise_preserves_edge():
    rng = np.random.default_rng(5)
    vol = np.zeros((16, 64, 64))
    vol[:, :, 32:] = 1.0                            # step edge
    noisy = vol + rng.normal(0, 0.2, vol.shape)
    den = dn.DENOISERS["tv"](noisy, ne.estimate_map_noise(noisy), scale=1.0)
    n0, n1 = np.std(noisy[:, :, :28]), np.std(den[:, :, :28])
    edge = den[:, :, 40].mean() - den[:, :, 24].mean()
    assert n1 < 0.6 * n0, (n0, n1)
    assert edge > 0.8, edge
    print(f"  edge-preserving denoise  OK (flat noise {n0:.3f}->{n1:.3f}, edge {edge:.2f})")


def _build_cumulative(x_true, M, noise_sd=0.0, seed=0):
    """Cumulative threshold volumes whose exclusive bins equal M @ x_true (+ optional noise)."""
    excl = np.tensordot(M, x_true, axes=([1], [0]))          # (n_bins, Z,Y,X)
    cum = np.cumsum(excl[::-1], axis=0)[::-1]                # reverse-cumsum -> cumulative A..D
    if noise_sd:
        cum = cum + np.random.default_rng(seed).normal(0, noise_sd, cum.shape)
    return cum


def test_estimators_end_to_end():
    M = mlib.build_M(["SoftTissue", "HA", "Iodine"], 1)
    Z, Y, X = 12, 24, 24
    x_true = np.zeros((3, Z, Y, X))
    x_true[0] = 1.0
    x_true[1, :, 8:16, 8:16] = 0.5                  # HA block
    x_true[2, :, 4:8, 4:8] = 0.015                  # iodine block

    def cfg(est, noise="global"):
        return DecompConfig(mode="phantom_ca_i", estimator=est, noise_model=noise,
                            hu_input=False, water_calibration=False, compute_residual=False,
                            joint_iters=3, z_chunk=6)

    # noise-free OLS is exact
    r = decompose(_build_cumulative(x_true, M), cfg("ols"))
    assert np.allclose(r.material_maps["HA"], x_true[1], atol=1e-4)
    print("  end-to-end OLS exact recovery (noise-free)  OK")

    # noisy: every estimator/noise-model combo runs; joint reduces flat-region noise vs OLS
    cumn = _build_cumulative(x_true, M, noise_sd=0.02, seed=7)
    bg = (slice(None), slice(0, 4), slice(16, 24))  # HA-free background
    n_ols = np.std(decompose(cumn, cfg("ols")).material_maps["HA"][bg])
    n_joint = np.std(decompose(cumn, cfg("wls_joint", "global")).material_maps["HA"][bg])
    assert n_joint < n_ols, (n_ols, n_joint)
    for est in ("wls", "wls_denoise"):
        for nm in ("global", "spatial"):
            decompose(cumn, cfg(est, nm))
    decompose(cumn, cfg("wls_joint", "spatial"))
    print(f"  all estimators x noise-models run; joint cuts flat noise  OK "
          f"({n_ols:.4f}->{n_joint:.4f})")


def test_window_averaging_reproduces_the_table():
    """
    The integrator must reproduce mu_rho_binavg.csv on the windows the CSV tabulates.

    This is the guard on the whole cumulative-M change: the cumulative windows are computed
    by the same routine, so if it agrees with the published table everywhere the table
    exists, the new windows are on the same footing as the old ones rather than a second,
    silently different physics table.  It also pins the finding the change rests on -- that
    the published averages are UNIFORM over energy, not spectrum-weighted.
    """
    import csv
    from pathlib import Path
    rows = list(csv.DictReader(
        open(Path(__file__).resolve().parent / "data" / "mu_rho_binavg.csv")))
    mats = [m for m in mlib.available_mono_materials()]
    err = []
    for r in rows:
        lo, hi = float(r["e_lo_keV"]), float(r["e_hi_keV"])
        for m in mats:
            err.append(abs(mlib.window_mu_rho(m, lo, hi) / float(r[m]) - 1.0))
    worst = 100 * max(err)
    assert worst < 0.5, f"window integration disagrees with the published table by {worst:.2f} %"
    print(f"  window <mu/rho> reproduces mu_rho_binavg.csv  OK "
          f"({len(err)} values, max {worst:.2f} %)")


def test_cumulative_channels():
    """Cumulative windows: nested, top bin unchanged, and better conditioned."""
    disj = mlib.threshold_channels(1)
    cum = mlib.threshold_channels(1, cumulative=True)
    e_max = max(hi for _lo, hi in mlib.bin_edges(1))
    assert all(c.window_keV[1] == e_max for c in cum), "cumulative windows must all end at E_max"
    assert [c.window_keV[0] for c in cum] == [c.window_keV[0] for c in disj]

    # The LAST bin is the same window either way, so its signature must not move at all.
    for m in ("Water", "HA", "Iodine"):
        a = mlib.signature_for_channel(m, disj[-1])
        b = mlib.signature_for_channel(m, cum[-1])
        assert abs(b / a - 1.0) < 5e-3, f"{m}: top window moved, {a:.5f} -> {b:.5f}"

    basis = ["Water", "HA", "Iodine"]
    kd = np.linalg.cond(mlib.build_M(basis, disj))
    kc = np.linalg.cond(mlib.build_M(basis, cum))
    assert kc < kd, f"cumulative M should be better conditioned, got {kc:.1f} vs {kd:.1f}"
    # Water mu (HU->attenuation, and the calibration reference) must follow the same
    # windows as M, or the forward model and the gain reference describe different things.
    assert abs(mlib.channel_water_mu(cum[0])
               - mlib.signature_for_channel("Water", cum[0])) < 1e-12
    print(f"  cumulative windows: nested, top bin fixed, kappa {kd:.1f} -> {kc:.1f}  OK")


def test_decompose_picks_the_matching_windows():
    """decompose() must pair the measurement stack with the M that describes it."""
    vols = np.zeros((4, 1, 2, 2), dtype=np.float32)
    for dom, want in (("cumulative", "cumulative"), ("exclusive", "disjoint")):
        cfg = DecompConfig(mode="phantom_water_ca_i", estimator="ols", bin_domain=dom,
                           hu_input=False, water_calibration=False)
        res = decompose(vols, cfg, channels=mlib.threshold_channels(1))
        got = res.metadata["channels"]
        widths = {tuple(c["window_keV"]) for c in got}
        nested = all(w[1] == 140.0 for w in widths)
        assert (nested if want == "cumulative" else not nested), \
            f"bin_domain={dom!r} built M on {sorted(widths)}"
    print("  decompose() matches M windows to the bin domain  OK")


def main():
    print("Running decomposition self-tests...")
    test_window_averaging_reproduces_the_table()
    test_cumulative_channels()
    test_decompose_picks_the_matching_windows()
    test_cumulative_to_exclusive()
    test_ols_exact_recovery()
    test_kappa_matches_numpy()
    test_noise_detects_noisiest_bin()
    test_wls_beats_ols()
    test_denoise_preserves_edge()
    test_estimators_end_to_end()
    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
