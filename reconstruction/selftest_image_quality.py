"""
selftest_image_quality.py -- synthetic validation of image_quality_metrics.py.

No scan data, no GPU: every case has a known answer, so run this before trusting a
comparison.  Checks:
  * NPS normalisation (true variance to ~0.02 %), shape (exact response of a known blur)
    and texture (smoothed noise has a lower f_av)
  * TTF50 and 10-90 % edge width of a Gaussian-blurred disc, to <8 %
  * NEQ, and d' scaling with contrast and noise
  * automatic ROI detection, and insert ROIs seeded from a hand-drawn label map
  * slab detection restricted to the phantom, and slice-thickness matching (sd / sqrt(n))
  * the noise region on a non-uniform phantom, and the water (absolute HU) reference

    python -m reconstruction.selftest_image_quality
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reconstruction import image_quality_metrics as iq

PIX = 0.5           # mm/pixel for the synthetic images
_FAILS: list[str] = []


def _chk(name, got, want, rtol):
    ok = abs(got - want) <= rtol * max(abs(want), 1e-12)
    print(f"  {'OK  ' if ok else 'FAIL'} {name:44s} got={got:12.5f} want={want:12.5f}")
    if not ok:
        _FAILS.append(name)


def _chk_true(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {name:44s} {detail}")
    if not cond:
        _FAILS.append(name)


def test_nps_normalisation(rng):
    print("\n[1] NPS normalisation (white noise)")
    sd = 20.0
    vol = rng.normal(0, sd, (40, 256, 256))
    patches = [(y, x) for y in range(32, 176, 48) for x in range(32, 176, 48)]
    r = iq.noise_power_spectrum(vol, patches, PIX)
    _chk("sigma recovered", r["noise_sd_hu"], sd, 0.02)
    _chk("int NPS == TRUE variance", r["variance_from_nps"], sd ** 2, 0.02)
    _chk("white NPS level = sd^2 dx^2", float(np.median(r["nps"])), sd ** 2 * PIX ** 2, 0.05)
    _chk_true("detrending loss is restored, not ignored",
              r["variance_measured"] < r["variance_from_nps"],
              f"detrended={r['variance_measured']:.2f} < corrected={r['variance_from_nps']:.2f}")
    return r


def test_nps_shape(rng, white):
    """
    Blur chosen so the spectrum falls ~76x across the band -- the realistic CT regime.

    The estimator's accuracy degrades as the spectrum steepens (spectral leakage from
    the finite patch): measured full-band error is ~4 % up to a 1000x fall and ~6 % at
    19000x.  Validating at a physically representative steepness is the point; a
    contrived 1e7x fall would only measure the leakage floor.
    """
    print("\n[2] NPS shape vs exact discrete blur")
    sd, sig_px = 20.0, 0.6
    # 60 slices x 25 patches = 1500 patches: enough that the estimate has converged onto
    # its systematic floor (~4 %) instead of fluctuating (a 360-patch sample scatters up
    # to 9 % between seeds and would make this check flaky).
    sm = ndimage.gaussian_filter(rng.normal(0, sd, (60, 256, 256)),
                                 sigma=(0, sig_px, sig_px), mode="wrap")
    patches = [(y, x) for y in range(32, 176, 32) for x in range(32, 176, 32)]
    r = iq.noise_power_spectrum(sm, patches, PIX)

    p = r["patch_px"]
    imp = np.zeros((p, p))
    imp[0, 0] = 1.0
    H = np.abs(np.fft.fft2(ndimage.gaussian_filter(imp, sig_px, mode="wrap")))
    fx = np.fft.fftfreq(p, d=PIX)
    _f, pred = iq.radial_average((sd ** 2 * PIX ** 2) * H ** 2, fx, fx)

    rel = float(np.max(np.abs(r["nps"] - pred)) / pred.max())
    _chk_true("NPS shape over the full band", rel < 0.06,
              f"max rel dev={rel:.4f}, spectrum falls {pred.max() / pred.min():.0f}x")
    _chk_true("smoothed noise is coarser than white",
              r["f_av"] < white["f_av"],
              f"f_av {r['f_av']:.4f} < {white['f_av']:.4f}")


def test_ttf():
    print("\n[3] TTF of a Gaussian-blurred disc")
    n = 512
    yy, xx = np.mgrid[:n, :n]
    cy = cx = n / 2 - 0.5
    r_mm = 12.0
    disc = ((((yy - cy) ** 2 + (xx - cx) ** 2) <= (r_mm / PIX) ** 2)
            .astype(np.float64) * 300.0)
    for sig_px in (1.5, 2.5):
        img = ndimage.gaussian_filter(disc, sigma=sig_px)
        t = iq.task_transfer_function(img, (cy, cx), r_mm, PIX)
        s = sig_px * PIX
        _chk(f"TTF50 (sigma={s:.2f} mm)", t["ttf50"],
             np.sqrt(np.log(2) / (2 * np.pi ** 2 * s ** 2)), 0.08)
        _chk(f"TTF10 (sigma={s:.2f} mm)", t["ttf10"],
             np.sqrt(np.log(10) / (2 * np.pi ** 2 * s ** 2)), 0.12)
        _chk(f"10-90 % edge width (sigma={s:.2f} mm)",
             t["edge_width_1090_mm"], 2.563 * s, 0.15)
        _chk(f"contrast (sigma={s:.2f} mm)", t["contrast_hu"], 300.0, 0.02)
    return disc, (cy, cx), r_mm


def test_detectability(white, disc, centre, r_mm):
    print("\n[4] NEQ and detectability index")
    t = iq.task_transfer_function(ndimage.gaussian_filter(disc, 2.0), centre, r_mm, PIX)
    f, nq = iq.neq(t["f"], t["ttf"], white["f"], white["nps"], contrast_hu=100.0)
    _chk_true("NEQ finite across the band", np.all(np.isfinite(nq)),
              f"{int(np.isfinite(nq).sum())}/{len(nq)} bins")
    d1 = iq.detectability_index(t["f"], t["ttf"], white["f"], white["nps"], 5.0, 25.0)["d_prime"]
    d2 = iq.detectability_index(t["f"], t["ttf"], white["f"], white["nps"], 5.0, 50.0)["d_prime"]
    d4 = iq.detectability_index(t["f"], t["ttf"], white["f"], white["nps"] * 4, 5.0, 50.0)["d_prime"]
    _chk("d' doubles with doubled contrast", d2 / d1, 2.0, 0.02)
    _chk("d' halves with 4x noise power", d4 / d2, 0.5, 0.02)


def test_roi_detection(rng):
    print("\n[5] automatic ROI detection")
    n = 512
    yy, xx = np.mgrid[:n, :n]
    cy = cx = n / 2 - 0.5
    phan = np.full((n, n), -1000.0)
    phan[((yy - cy) ** 2 + (xx - cx) ** 2) <= (100 / PIX) ** 2] = 0.0
    truth = [(-60, -60, 8, 300.0), (60, -60, 8, 150.0), (0, 70, 10, -80.0)]
    for dy, dx, rmm, hu in truth:
        m = ((yy - (cy + dy / PIX)) ** 2 + (xx - (cx + dx / PIX)) ** 2) <= (rmm / PIX) ** 2
        phan[m] = hu
    noisy = phan + rng.normal(0, 8.0, phan.shape)

    body = iq.detect_body_mask(noisy, PIX)
    ins = iq.detect_inserts(noisy, body, PIX)
    _chk_true("insert count", len(ins) == len(truth), f"found {len(ins)}, expected {len(truth)}")
    for d in ins:
        print(f"        r={d['radius_mm']:5.2f} mm  HU={d['mean_hu']:8.1f}  "
              f"circularity={d['circularity']:.2f}")
    pts = iq.background_patches(body, ins, PIX, patch_px=32)
    _chk_true("background patches found", len(pts) >= 4, f"{len(pts)} patches")
    stats = iq.roi_statistics(noisy[None], ins, body, PIX)
    _chk_true("CNR computed per insert",
              all(np.isfinite(i["cnr"]) for i in stats["inserts"]),
              " ".join(f"{i['cnr']:.1f}" for i in stats["inserts"]))


def test_slab_and_thickness(rng):
    print("\n[6] slab detection and slice-thickness matching")
    n = 256
    yy, xx = np.mgrid[:n, :n]
    body = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2) <= (80 / PIX) ** 2
    nz, k0, k1 = 60, 20, 39
    vol = np.empty((nz, n, n), dtype=np.float32)
    for k in range(nz):
        sl = np.full((n, n), -1000.0, dtype=np.float32)
        sl[body] = 0.0
        if k0 <= k <= k1:
            m = ((yy - n / 2 - 30) ** 2 + (xx - n / 2) ** 2) <= (10 / PIX) ** 2
            sl[m] = 300.0
        vol[k] = sl + rng.normal(0, 5.0, sl.shape)
    sd = iq.find_insert_slab(vol, PIX, np.arange(nz) * 0.4)
    _chk_true("insert slab located",
              abs(sd["k_lo"] - k0) <= 2 and abs(sd["k_hi"] - k1) <= 2,
              f"detected k=[{sd['k_lo']},{sd['k_hi']}], truth [{k0},{k1}]")

    test_labelmap_seeded_inserts(rng)
    test_noise_region(rng)
    test_water_reference(rng)
    test_slab_search_restriction(rng)

    native = rng.normal(0, 30.0, (100, 32, 32)).astype(np.float32)
    z_nat = np.arange(100) * 0.4
    matched, used = iq.match_slice_thickness(native, z_nat, 2.0, np.arange(10, 30) * 0.4)
    _chk("thickness-matched noise = sd/sqrt(n)", float(matched.std()),
         30.0 / np.sqrt(np.mean(used)), 0.10)
    print(f"        averaged {np.mean(used):.1f} native slices per 2.0 mm slice")


def test_labelmap_seeded_inserts(rng):
    """
    Hand-drawn seeds -> refined ROIs.

    Mimics a Slicer segmentation: one label per ROW of inserts (so each label must be
    split into components), positions roughly right but deliberately offset, and drawn
    radii wrong by -35 % to +55 %.  The refinement has to recover the true geometry from
    the image, because TTF needs the true centre and the background exclusion needs the
    true radius.
    """
    print("\n[8] insert ROIs seeded from a hand-drawn label map")
    pix = 500.0 / 512
    n = 512
    yy, xx = np.mgrid[:n, :n]
    cy = cx = n / 2 - 0.5
    img = np.full((n, n), -1000.0)
    img[((yy - cy) ** 2 + (xx - cx) ** 2) <= (100 / pix) ** 2] = 0.0

    r_true_mm = 6.0
    truth, labels = [], np.zeros((n, n), dtype=np.int32)
    for row, (dy, hu) in enumerate([(-45, 300.0), (0, 150.0), (45, -70.0)], start=1):
        for dx in (-50, -20, 20, 50):
            ty, tx = cy + dy / pix, cx + dx / pix
            img[((yy - ty) ** 2 + (xx - tx) ** 2) <= (r_true_mm / pix) ** 2] = hu
            truth.append((ty, tx))
            oy, ox = rng.uniform(-2.5, 2.5, 2)          # sloppy centre
            rs = (r_true_mm / pix) * rng.uniform(0.65, 1.55)   # sloppy radius
            labels[((yy - (ty + oy)) ** 2 + (xx - (tx + ox)) ** 2) <= rs ** 2] = row
    img = ndimage.gaussian_filter(img, 1.2) + rng.normal(0, 18.0, img.shape)

    ins = iq.inserts_from_labelmap(labels, img, pix, refine=True)
    _chk_true("row labels split into individual inserts", len(ins) == len(truth),
              f"{len(ins)} ROIs from {len({i['label'] for i in ins})} labels, "
              f"expected {len(truth)} from 3")
    _chk_true("every edge refined", all(i["refined"] for i in ins),
              f"{sum(i['refined'] for i in ins)}/{len(ins)}")

    err = [min(np.hypot(i["cy"] - t[0], i["cx"] - t[1]) for t in truth) * pix for i in ins]
    _chk_true("centres recovered to <0.5 mm", max(err) < 0.5,
              f"mean {np.mean(err):.2f} mm, max {np.max(err):.2f} mm")

    drawn = np.abs(np.array([i["seed_radius_mm"] for i in ins]) - r_true_mm).mean()
    refined = np.abs(np.array([i["radius_mm"] for i in ins]) - r_true_mm).mean()
    _chk_true("refinement beats the drawn radius by >5x", drawn / refined > 5,
              f"drawn err {drawn:.2f} mm -> refined err {refined:.2f} mm "
              f"({drawn / refined:.1f}x better)")


def test_slab_search_restriction(rng):
    """
    Regression test for a real failure: on a long clinical scan range an off-phantom
    structure out-scored the phantom and put the slab ~665 mm away from it.

    Also checks that with several insert layers, select='peak' takes the one carrying
    the MOST inserts, including when that layer sits hard against the end of the
    phantom (the layer most at risk of being missed).
    """
    print("\n[7] slab search restricted to the phantom")
    pix = 500.0 / 512
    n = 256
    yy, xx = np.mgrid[:n, :n]
    cy = cx = n / 2 - 0.5
    body = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (100 / pix) ** 2
    z = np.arange(-2200, -1300, 4.0)
    vol = np.empty((len(z), n, n), dtype=np.float32)

    def ring(sl, count, r_ring, r_ins, hu):
        for j in range(count):
            a = 2 * np.pi * j / count
            m = (((yy - (cy + r_ring * np.sin(a) / pix)) ** 2
                  + (xx - (cx + r_ring * np.cos(a) / pix)) ** 2) <= (r_ins / pix) ** 2)
            sl[m] = hu

    phantom_z = (-1515.5, -1408.3)
    for k, zz in enumerate(z):
        sl = np.full((n, n), -1000.0, dtype=np.float32)
        if phantom_z[0] <= zz <= phantom_z[1]:
            sl[body] = 0.0
            if -1500 <= zz <= -1470:
                ring(sl, 10, 70, 7, 250.0)          # fewer inserts
            if -1425 <= zz <= -1410:
                ring(sl, 18, 70, 7, 250.0)          # most inserts, at the phantom end
        if -2140 <= zz <= -2110:                    # off-phantom decoy
            sl[body] = 0.0
            ring(sl, 24, 60, 11, 900.0)
        vol[k] = sl + rng.normal(0, 12.0, sl.shape)

    unrestricted = iq.find_insert_slab(vol, pix, z, select="peak")
    _chk_true("unrestricted search is captured by the decoy (the bug)",
              unrestricted["z_lo_mm"] < -2000,
              f"slab at {unrestricted['z_lo_mm']:+.0f}..{unrestricted['z_hi_mm']:+.0f} mm")

    restricted = iq.find_insert_slab(vol, pix, z, search_z_mm=phantom_z, select="peak")
    inside = (phantom_z[0] - 1 <= restricted["z_lo_mm"]
              and restricted["z_hi_mm"] <= phantom_z[1] + 1)
    _chk_true("search_z_mm confines the slab to the phantom", inside,
              f"slab at {restricted['z_lo_mm']:+.0f}..{restricted['z_hi_mm']:+.0f} mm")
    _chk_true("select='peak' takes the layer with the most inserts",
              restricted["z_lo_mm"] >= -1430,
              f"chose {restricted['z_lo_mm']:+.0f}..{restricted['z_hi_mm']:+.0f} mm "
              f"(18-insert layer at -1425..-1410, not the 10-insert one at -1500..-1470)")
    _chk_true("every candidate layer is reported",
              len(restricted["candidates"]) >= 2,
              f"{len(restricted['candidates'])} candidates")


def test_noise_region(rng):
    """
    Where the noise is measured, on a phantom that is NOT uniform.

    Pins two failures that each produced plausible-looking but wrong noise numbers:
      * the body outline of an anthropomorphic phantom contains lung, bone and the table,
        so a patch on any of them measures anatomy -- the real run reported 54-250 HU
        "noise" with an f_av of 0.078 cyc/mm, i.e. a noise grain over a centimetre wide;
      * selecting the flattest FRACTION of a uniform region keeps speckle rather than a
        region, and the following erosion then destroyed it (1.4 % of a completely
        uniform body survived), silently forcing the fallback to the body outline.
    """
    print("\n[9] noise region on a non-uniform phantom")
    pix = 500.0 / 512
    n = 512
    yy, xx = np.mgrid[:n, :n]
    cy, cx = n / 2 - 30, n / 2 - 0.5

    # uniform phantom: the homogeneous region must survive essentially intact
    uni = np.full((n, n), -1000.0)
    uni[((yy - cy) ** 2 + (xx - cx) ** 2) <= (100 / pix) ** 2] = 0.0
    uni = uni + rng.normal(0, 12.0, uni.shape)
    body_u = iq.detect_body_mask(uni, pix)
    homo_u = iq.homogeneous_mask(uni, body_u, pix)
    _chk_true("uniform body is kept whole", homo_u.sum() > 0.6 * body_u.sum(),
              f"kept {100 * homo_u.sum() / body_u.sum():.0f}% of the body")

    # thorax phantom: lung, bone and table must all be rejected
    thx = np.full((n, n), -1000.0)
    thx[(((yy - cy) / (95 / pix)) ** 2 + ((xx - cx) / (150 / pix)) ** 2) <= 1.0] = 40.0
    for sx in (-70, 70):
        thx[(((yy - cy) / (60 / pix)) ** 2
             + ((xx - cx - sx / pix) / (48 / pix)) ** 2) <= 1.0] = -800.0
    thx[((yy - (cy + 70 / pix)) ** 2 + (xx - cx) ** 2) <= (18 / pix) ** 2] = 900.0
    thx[(yy > cy + 110 / pix) & (yy < cy + 120 / pix)] = 700.0
    truth_lung, truth_dense = thx < -400, thx > 400
    thx_n = thx + rng.normal(0, 12.0, thx.shape)
    body_t = iq.detect_body_mask(thx_n, pix)
    homo_t = iq.homogeneous_mask(thx_n, body_t, pix)
    _chk_true("lung and bone/table excluded",
              not (homo_t & truth_lung).any() and not (homo_t & truth_dense).any(),
              f"lung {int((homo_t & truth_lung).sum())} px, "
              f"dense {int((homo_t & truth_dense).sum())} px, "
              f"kept {100 * homo_t.sum() / body_t.sum():.0f}% of the body")

    # an insert left inside the region is counted as noise
    sd_true = 40.0
    flat_img = np.full((n, n), -1000.0)
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (100 / pix) ** 2
    flat_img[disc] = 0.0
    ins = []
    for j in range(6):
        a = 2 * np.pi * j / 6
        ty, tx = cy + 55 * np.sin(a) / pix, cx + 55 * np.cos(a) / pix
        flat_img[((yy - ty) ** 2 + (xx - tx) ** 2) <= (9 / pix) ** 2] = 400.0
        ins.append({"cy": ty, "cx": tx, "radius_px": 9 / pix, "radius_mm": 9.0})
    vol = np.repeat(flat_img[None], 12, axis=0) + rng.normal(0, sd_true, (12, n, n))
    body = iq.detect_body_mask(vol[6], pix)
    with_ins, _ = iq.region_noise_sd(vol, iq.noise_region(body, [], pix))
    without, _ = iq.region_noise_sd(vol, iq.noise_region(body, ins, pix))
    _chk("noise with inserts excluded", without, sd_true, 0.05)
    _chk_true("leaving inserts in inflates the noise", with_ins > without * 1.2,
              f"{with_ins:.1f} HU with inserts vs {without:.1f} HU without "
              f"(true {sd_true:.0f})")

    # the SD must not depend on square patches fitting
    sd_all, n_vox = iq.region_noise_sd(vol, iq.noise_region(body, ins, pix))
    _chk_true("SD measured without any patch geometry", n_vox > 1000 and np.isfinite(sd_all),
              f"{n_vox} voxels, sd={sd_all:.1f} HU")


def test_water_reference(rng):
    """
    The absolute HU check, and the two ways it can quietly lie.

    Pins:
      * that the offset survives at all -- region_noise_sd fits and subtracts a 2-D
        polynomial, whose constant term IS the offset, so the mean cannot be taken from
        the same code path;
      * that a region traced along the module boundary and filled inwards keeps the
        module WALL, which is not water. The contaminated mean looks entirely plausible,
        so it has to be detected rather than eyeballed.
    """
    print("\n[10] water reference (absolute HU)")
    pix = 500.0 / 512
    n = 384
    yy, xx = np.mgrid[:n, :n]
    cy = cx = n / 2 - 0.5
    r_out = 100.0 / pix
    offset, sd_true = 12.0, 30.0

    img = np.full((n, n), -1000.0)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disc = rr <= r_out
    img[disc] = offset
    ins = []
    for j in range(6):
        a = 2 * np.pi * j / 6
        ty, tx = cy + 55 * np.sin(a) / pix, cx + 55 * np.cos(a) / pix
        img[((yy - ty) ** 2 + (xx - tx) ** 2) <= (9 / pix) ** 2] = 400.0
        ins.append({"cy": ty, "cx": tx, "radius_px": 9 / pix, "radius_mm": 9.0})

    vol = np.repeat(img[None], 8, axis=0) + rng.normal(0, sd_true, (8, n, n))
    region = iq.noise_region(disc, ins, pix, edge_margin_mm=2.0)
    w = iq.water_reference(vol, region, pix, expected_hu=0.0)
    _chk("water offset recovered", w["offset_hu"], offset, 0.15)
    _chk_true("clean region reports no wall bias", abs(w["wall_bias_hu"]) < 3.0,
              f"wall_bias={w['wall_bias_hu']:.2f} HU")

    # The offset and the noise are different quantities measured on the same voxels:
    # the detrended SD must not see the offset, and the mean must not see the noise.
    sd, _ = iq.region_noise_sd(vol, region)
    _chk_true("detrended SD is blind to the offset the mean reports",
              abs(sd - sd_true) < 0.1 * sd_true,
              f"sd={sd:.1f} HU (true {sd_true:.0f}) while mean={w['mean_hu']:.1f} HU")

    # 5 mm of 300 HU wall just inside the traced boundary; 2 mm of it survives the
    # region's own erosion, and must show up rather than shift the answer silently.
    walled = img.copy()
    walled[(rr <= r_out) & (rr > r_out - 5.0 / pix)] = 300.0
    volw = np.repeat(walled[None], 8, axis=0) + rng.normal(0, sd_true, (8, n, n))
    ww = iq.water_reference(volw, region, pix, expected_hu=0.0)
    _chk_true("wall contamination is detected", abs(ww["wall_bias_hu"]) > 5.0,
              f"wall_bias={ww['wall_bias_hu']:.1f} HU, "
              f"mean={ww['mean_hu']:.1f} vs core={ww['core_mean_hu']:.1f}")
    _chk_true("the core is the value nearer the truth",
              abs(ww["core_mean_hu"] - offset) < abs(ww["mean_hu"] - offset),
              f"core off by {abs(ww['core_mean_hu'] - offset):.1f} HU, "
              f"full region off by {abs(ww['mean_hu'] - offset):.1f} HU")


def main() -> int:
    rng = np.random.default_rng(0)
    print("=" * 68)
    print("image_quality_metrics -- synthetic self-test")
    print("=" * 68)
    white = test_nps_normalisation(rng)
    test_nps_shape(rng, white)
    disc, centre, r_mm = test_ttf()
    test_detectability(white, disc, centre, r_mm)
    test_roi_detection(rng)
    test_slab_and_thickness(rng)
    print("\n" + "=" * 68)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): " + ", ".join(_FAILS))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
