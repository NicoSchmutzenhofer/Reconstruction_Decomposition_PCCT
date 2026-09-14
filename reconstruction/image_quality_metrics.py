"""
image_quality_metrics.py -- no-reference image-quality metrics for CT volumes.

Pure library: arrays in, numbers out; no I/O, no plotting, no __main__.  recon_comparison.py
is the driver.  Body, inserts, noise region and insert slab are detected automatically, so no
ROI is drawn per scan.  Noise and resolution are measured together, since either alone can be
improved by blurring:

    noise        noise_power_spectrum()        sigma and the NPS curve
    resolution   task_transfer_function()      TTF50 / TTF10, 10-90 % edge width
    both         neq(), detectability_index()

Conventions, fixed so the numbers stay comparable across image families:
  * NPS is two-sided, in HU^2 mm^2, normalised so its 2-D integral equals the variance;
    the radial NPS is the ring mean of the 2-D NPS.
  * f_av = int f NPS(f) df / int NPS(f) df on the radial curve; low f_av = coarse noise.
  * TTF is normalised to 1 at f = 0.
  * Spatial frequencies are in cycles/mm.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


# ═════════════════════════════════════════════════════════════════════
# Region detection -- automatic, so no ROI has to be drawn per scan
# ═════════════════════════════════════════════════════════════════════
def detect_body_mask(img_hu, pixel_mm, air_hu=-300.0, erode_mm=0.0):
    """
    Segment the phantom/patient body from air.

    Uses a fixed HU threshold rather than Otsu: these volumes are HU-calibrated, so
    -300 HU is a physically meaningful air/body boundary and is stable across
    thresholds and keV levels (Otsu drifts when the histogram shape changes between
    channels, which would silently move the ROIs between the images being compared).

    Returns a bool mask of the largest connected component, holes filled.
    """
    img = np.asarray(img_hu, dtype=np.float32)
    mask = img > air_hu
    if not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n > 1:                                  # keep the largest blob (drop table, noise)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = lab == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_fill_holes(mask)
    if erode_mm > 0:
        r = max(1, int(round(erode_mm / pixel_mm)))
        mask = ndimage.binary_erosion(mask, _disk(r))
    return mask


def _disk(radius):
    """Boolean disk structuring element of the given pixel radius."""
    r = int(max(1, radius))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


def detect_inserts(img_hu, body_mask, pixel_mm,
                   min_diameter_mm=4.0, max_diameter_mm=40.0,
                   contrast_mad_k=4.0, min_circularity=0.65):
    """
    Find the phantom's cylindrical inserts inside `body_mask`.

    An insert is a compact, roughly circular region whose HU differs from the modal
    body HU by more than contrast_mad_k robust standard deviations.  Detection is
    deliberately run on ONE high-contrast reference channel by the caller and the
    resulting geometry reused for every other channel of that family, so that any
    metric difference between channels comes from the images and not from the ROIs
    having moved.

    Returns a list of dicts: {cy, cx, radius_px, radius_mm, mean_hu, contrast_hu,
    circularity, area_px}, sorted by descending area.
    """
    img = np.asarray(img_hu, dtype=np.float32)
    if not body_mask.any():
        return []
    inside = img[body_mask]
    bg = float(np.median(inside))
    mad = float(np.median(np.abs(inside - bg))) * 1.4826
    if mad <= 0:
        mad = float(np.std(inside)) or 1.0

    # Smooth before thresholding so noise does not fragment the inserts.
    sm = ndimage.gaussian_filter(img, sigma=max(1.0, 0.6 / pixel_mm))
    cand = body_mask & (np.abs(sm - bg) > contrast_mad_k * mad)
    cand = ndimage.binary_opening(cand, _disk(max(1, int(round(0.6 / pixel_mm)))))

    lab, n = ndimage.label(cand)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        r_eq = np.sqrt(area / np.pi) * pixel_mm
        if not (min_diameter_mm / 2 <= r_eq <= max_diameter_mm / 2):
            continue
        # circularity 4*pi*A/P^2 -> 1.0 for a perfect disc
        per = float(np.sum(comp ^ ndimage.binary_erosion(comp)))
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0.0
        if circ < min_circularity:
            continue
        cy, cx = ndimage.center_of_mass(comp)
        out.append({
            "cy": float(cy), "cx": float(cx),
            "radius_px": float(np.sqrt(area / np.pi)),
            "radius_mm": float(r_eq),
            "mean_hu": float(img[comp].mean()),
            "contrast_hu": float(img[comp].mean() - bg),
            "circularity": float(circ),
            "area_px": area,
        })
    out.sort(key=lambda d: -d["area_px"])
    return out


def refine_insert_geometry(img_hu, cy, cx, r_guess_px, search=2.5):
    """
    Refine one insert's centre and radius from the image, starting from a rough guess.

    Made for hand-drawn seeds: a contour traced by eye has the right position but not
    the right edge, and both the TTF (which needs the true centre to build a clean
    radial edge profile) and the background exclusion (which needs the true radius) are
    sensitive to that.  The seed only has to overlap the insert.

    Method: estimate the inside and outside levels from the seed, threshold halfway
    between them, take the connected component containing the seed for a robust centre,
    then read the radius off the 50 % crossing of the radial profile about that centre --
    which is sub-pixel and does not care how wobbly the drawn outline was.

    Returns (cy, cx, radius_px, ok).  ok=False means the insert could not be separated
    from its surroundings (too little contrast in this channel) and the seed geometry
    was kept unchanged.
    """
    img = np.asarray(img_hu, dtype=np.float64)
    ny, nx = img.shape
    r = float(max(r_guess_px, 1.5))
    half = int(np.ceil(search * r)) + 2
    y0, y1 = max(0, int(cy) - half), min(ny, int(cy) + half + 1)
    x0, x1 = max(0, int(cx) - half), min(nx, int(cx) + half + 1)
    sub = img[y0:y1, x0:x1]
    if sub.size < 16:
        return float(cy), float(cx), r, False

    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inner = sub[rr < 0.5 * r]
    outer = sub[(rr > 1.5 * r) & (rr < search * r)]
    if inner.size < 4 or outer.size < 8:
        return float(cy), float(cx), r, False
    lo, hi = float(np.median(outer)), float(np.median(inner))
    if abs(hi - lo) < 1e-6:
        return float(cy), float(cx), r, False

    mid = 0.5 * (lo + hi)
    blob = (sub > mid) if hi > lo else (sub < mid)
    blob = ndimage.binary_opening(blob, _disk(1))
    lab, n = ndimage.label(blob)
    if n == 0:
        return float(cy), float(cx), r, False
    seed = lab[int(round(cy)) - y0, int(round(cx)) - x0]
    if seed == 0:                       # seed landed just off the insert -> nearest blob
        cand = [(abs(ndimage.center_of_mass(lab == i)[0] + y0 - cy)
                 + abs(ndimage.center_of_mass(lab == i)[1] + x0 - cx), i)
                for i in range(1, n + 1)]
        seed = min(cand)[1]
    comp = lab == seed
    ccy, ccx = ndimage.center_of_mass(comp)
    ccy, ccx = float(ccy + y0), float(ccx + x0)
    r_eq = float(np.sqrt(comp.sum() / np.pi))

    # sub-pixel radius from the 50 % crossing about the refined centre
    rr2 = np.sqrt((yy - ccy) ** 2 + (xx - ccx) ** 2)
    dr = 0.25
    nb = int(np.ceil(search * r / dr))
    idx = np.clip((rr2 / dr).astype(int), 0, nb - 1)
    cnt = np.bincount(idx.ravel(), minlength=nb)
    tot = np.bincount(idx.ravel(), weights=sub.ravel(), minlength=nb)
    ok_bins = cnt > 0
    prof = np.full(nb, np.nan)
    prof[ok_bins] = tot[ok_bins] / cnt[ok_bins]
    radii = (np.arange(nb) + 0.5) * dr
    good = np.isfinite(prof)
    frac = (prof[good] - lo) / (hi - lo)
    cross = _crossing(radii[good], frac, 0.5)
    if np.isfinite(cross) and 0.3 * r <= cross <= search * r:
        r_eq = float(cross)
    return ccy, ccx, r_eq, True


def label_z_extents(label_vol):
    """
    z-slice range each label value occupies: {value: (k_lo, k_hi, n_slices)}.

    A phantom whose inserts sit in several axial LAYERS needs this.  Collapsing such a
    segmentation onto one plane merges inserts that share an (x, y) position in different
    layers, and averaging the image over the whole slab dilutes every insert's contrast by
    its layer fraction -- an insert present in a third of the slices reads about a third of
    its true contrast.
    """
    lab = np.asarray(label_vol)
    if lab.ndim != 3:
        return {}
    out = {}
    for v in sorted(int(v) for v in np.unique(lab) if v > 0):
        kz = np.nonzero(np.any(lab == v, axis=(1, 2)))[0]
        if kz.size:
            out[v] = (int(kz.min()), int(kz.max()), int(kz.size))
    return out


def inserts_from_labelmap(label_vol, img_hu, pixel_mm, refine=True,
                          min_diameter_mm=3.0, max_diameter_mm=40.0,
                          air_hu=-300.0, vol_hu=None):
    """
    Turn a hand-drawn label map into insert ROIs, with edges refined from the image.

    Each label value is treated as a GROUP (e.g. one layer, or one row of inserts drawn as
    a single segment), so every label is split into its connected components first -- one
    ROI per component, not one per label.  Labels need not cover every insert; whatever is
    drawn is used, which is the point of allowing a partial annotation.

    label_vol : integer labels on the SAME grid as img_hu / vol_hu (resample beforehand).
    img_hu    : the 2-D image ROIs are measured on when vol_hu is not given.
    vol_hu    : the full (Z, Y, X) volume.  Supply it whenever the segmentation is 3-D:
                each label is then projected and measured over ITS OWN z-slices instead of
                the whole slab, which is what a multi-layer phantom requires.  Every ROI
                carries the z-range it belongs to, so downstream measurements can restrict
                themselves to the slices where that insert actually exists.
    """
    lab3d = np.asarray(label_vol)
    img_all = np.asarray(img_hu, dtype=np.float32)
    per_layer = vol_hu is not None and lab3d.ndim == 3
    vol = np.asarray(vol_hu, dtype=np.float32) if per_layer else None

    lab_flat = np.max(lab3d, axis=0) if lab3d.ndim == 3 else lab3d
    if lab_flat.shape != img_all.shape:
        raise ValueError(f"inserts_from_labelmap: label shape {lab_flat.shape} != image "
                         f"shape {img_all.shape} -- resample the segmentation first")

    extents = label_z_extents(lab3d) if lab3d.ndim == 3 else {}
    nz = vol.shape[0] if per_layer else 1
    out = []
    for value in sorted(int(v) for v in np.unique(lab_flat) if v > 0):
        if per_layer and value in extents:
            k0, k1, _ = extents[value]
            group = np.any(lab3d[k0:k1 + 1] == value, axis=0)
            img = vol[k0:k1 + 1].mean(axis=0)
        else:
            k0, k1 = 0, nz - 1
            group = lab_flat == value
            img = img_all
        # Reference level for the contrast figures: the surrounding TISSUE, not air.
        # Taking the median over everything unlabelled includes the air around the
        # phantom, which for a body occupying a fraction of the FOV is the majority of
        # the image -- every insert would then be reported at ~+1000 HU contrast.
        # Evaluated on THIS layer's image so the reference matches the measurement.
        outside = (lab_flat == 0) & (img > air_hu)
        bg = float(np.median(img[outside])) if outside.any() else float(np.median(img))
        comps, n = ndimage.label(group)
        for i in range(1, n + 1):
            comp = comps == i
            area = int(comp.sum())
            r_seed = np.sqrt(area / np.pi)
            if not (min_diameter_mm / 2 <= r_seed * pixel_mm <= max_diameter_mm / 2):
                continue
            cy, cx = ndimage.center_of_mass(comp)
            refined = False
            r_px = r_seed
            if refine:
                cy, cx, r_px, refined = refine_insert_geometry(img, cy, cx, r_seed)
            yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
            disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r_px * 0.7) ** 2
            mean_hu = float(img[disc].mean()) if disc.any() else float(img[comp].mean())
            out.append({
                "cy": float(cy), "cx": float(cx),
                "radius_px": float(r_px), "radius_mm": float(r_px * pixel_mm),
                "mean_hu": mean_hu, "contrast_hu": mean_hu - bg,
                "circularity": 1.0, "area_px": area,
                "label": value, "refined": bool(refined),
                "seed_radius_mm": float(r_seed * pixel_mm),
                "k_lo": int(k0), "k_hi": int(k1), "n_z": int(k1 - k0 + 1),
            })
    out.sort(key=lambda d: (d["label"], d["cx"], d["cy"]))
    return out


def homogeneous_mask(img_hu, body_mask, pixel_mm, window_mm=6.0, hu_band=75.0,
                     flat_tolerance=1.6, erode_mm=2.0):
    """
    Restrict a body mask to the parts that are actually UNIFORM.

    A noise measurement assumes the region under it contains nothing but noise.  That
    holds for a uniform cylinder, and fails completely for an anthropomorphic phantom:
    lung, bone, mediastinum and the patient table all sit inside the body outline, and a
    patch dropped on any of them measures anatomy.  The symptom is unmistakable once you
    know it -- a noise "SD" of a few hundred HU and an f_av of ~0.08 cyc/mm, i.e. a noise
    grain of over a centimetre, which is structure, not noise.

    Two criteria, both computed on the slab z-average so that quantum noise (already
    suppressed by sqrt(Z)) does not drive them:
      * HU band     -- within hu_band of the modal soft-tissue value, which removes lung
                       (very negative) and bone/table (very positive).
      * local flatness -- local standard deviation in a window_mm box no more than
                       flat_tolerance x its median, which removes edges, gradients
                       and residual structure while keeping uniform material whole.

    Returns a bool mask.  Prefer an explicitly drawn background region when one is
    available; this is the automatic fallback.
    """
    img = np.asarray(img_hu, dtype=np.float64)
    if not np.any(body_mask):
        return np.zeros(img.shape, dtype=bool)

    vals = img[body_mask]
    hist, edges = np.histogram(vals, bins=128)
    mode = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    band = body_mask & (np.abs(img - mode) < hu_band)
    if band.sum() < 64:
        band = body_mask.copy()

    w = max(3, int(round(window_mm / pixel_mm)) | 1)
    mean = ndimage.uniform_filter(img, size=w)
    mean_sq = ndimage.uniform_filter(img * img, size=w)
    local_sd = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    # Threshold RELATIVE to the typical local SD, not at a fixed percentile.  A percentile
    # keeps exactly that fraction by construction, and in uniform material -- where local
    # SD is nearly constant and the ranking is pure chance -- the survivors are speckle
    # rather than a region, which the following erosion then annihilates (measured: 1.4 %
    # of a completely uniform body surviving a "keep the flattest 50 %" rule).  A relative
    # threshold keeps essentially all of a homogeneous area as one connected region while
    # still rejecting lung, bone, table and edges, whose local SD is many times higher.
    ref = float(np.median(local_sd[band]))
    thr = ref * float(flat_tolerance) if ref > 0 else float(np.percentile(local_sd[band], 75))
    keep = band & (local_sd <= thr)
    keep = ndimage.binary_closing(keep, _disk(max(1, int(round(2.0 / pixel_mm)))))
    keep &= band
    if erode_mm > 0:
        keep = ndimage.binary_erosion(keep, _disk(max(1, int(round(erode_mm / pixel_mm)))))
    return keep


def noise_region(region_mask, inserts, pixel_mm, insert_margin_mm=6.0,
                 edge_margin_mm=10.0):
    """
    The region a noise measurement may use: `region_mask`, pulled back from its own edge
    and with every insert (plus a margin) cut out.

    Both exclusions are mandatory, not cosmetic.  An insert left inside contributes its
    full contrast to the "noise", which is how a 42 HU image comes to measure 51 HU -- and
    because the contrast is identical in every channel while the noise is not, it also
    compresses and can even invert the ranking the whole comparison exists to establish.
    """
    keep = np.asarray(region_mask, dtype=bool)
    if edge_margin_mm > 0:
        keep = ndimage.binary_erosion(keep, _disk(max(1, int(round(edge_margin_mm / pixel_mm)))))
    if inserts:
        ny, nx = keep.shape
        y, x = np.ogrid[:ny, :nx]
        for ins in inserts:
            r = ins["radius_px"] + insert_margin_mm / pixel_mm
            keep &= ((y - ins["cy"]) ** 2 + (x - ins["cx"]) ** 2) > r * r
    return keep


def background_patches(body_mask, inserts, pixel_mm, patch_px=64,
                       insert_margin_mm=6.0, edge_margin_mm=10.0, max_patches=64):
    """
    Tile the uniform background with square patches for the NPS.

    Patches must lie entirely inside the body, clear of every insert by
    insert_margin_mm and clear of the body edge by edge_margin_mm (beam hardening and
    the body boundary would otherwise leak structure into the noise estimate).

    Returns a list of (y0, x0) top-left origins.
    """
    ny, nx = body_mask.shape
    keep = noise_region(body_mask, inserts, pixel_mm, insert_margin_mm, edge_margin_mm)

    # Integral image -> a patch is valid iff every pixel under it is valid.
    ii = np.cumsum(np.cumsum(keep.astype(np.int32), axis=0), axis=1)
    p = int(patch_px)
    out = []
    step = max(p // 4, 4)   # 75 % overlap. Overlapping patches are correlated and so
                            # add less than N independent samples, but on a phantom with
                            # only a small uniform region they are the difference between
                            # a usable NPS and three patches.
    for y0 in range(0, ny - p + 1, step):
        for x0 in range(0, nx - p + 1, step):
            y1, x1 = y0 + p - 1, x0 + p - 1
            tot = ii[y1, x1]
            if y0 > 0:
                tot -= ii[y0 - 1, x1]
            if x0 > 0:
                tot -= ii[y1, x0 - 1]
            if y0 > 0 and x0 > 0:
                tot += ii[y0 - 1, x0 - 1]
            if tot == p * p:
                out.append((y0, x0))
    if len(out) > max_patches:                  # spread the sample over the whole region
        idx = np.linspace(0, len(out) - 1, max_patches).astype(int)
        out = [out[i] for i in idx]
    return out


def auto_background_patches(body_mask, inserts, pixel_mm, patch_px=64, min_patches=4,
                            min_patch_px=12, **kw):
    """
    background_patches() with automatic size reduction.

    The usable patch size is set by the phantom's geometry, not by preference: a vendor
    export reconstructed at a whole-body FOV puts a ~200 mm phantom on ~1 mm pixels, so
    a 64 px patch is 62 mm wide and no such square of clear background exists between
    the inserts.  Halve until enough patches fit rather than failing.

    Returns (patches, patch_px_used, tried) where `tried` is [(size, count), ...] for
    diagnostics.  Use the SAME patch_px_used for every image being compared -- the NPS
    frequency grid depends on it.
    """
    p = int(patch_px)
    tried = []
    while p >= int(min_patch_px):
        pts = background_patches(body_mask, inserts, pixel_mm, patch_px=p, **kw)
        tried.append((p, len(pts)))
        if len(pts) >= int(min_patches):
            return pts, p, tried
        p //= 2
    best = max(tried, key=lambda t: t[1]) if tried else (int(patch_px), 0)
    if best[1] > 0:
        return (background_patches(body_mask, inserts, pixel_mm, patch_px=best[0], **kw),
                best[0], tried)
    return [], int(patch_px), tried


# ═════════════════════════════════════════════════════════════════════
# Noise power spectrum
# ═════════════════════════════════════════════════════════════════════
def _poly_basis(ny, nx, order):
    """Design matrix of the 2-D polynomial basis used for patch detrending."""
    y, x = np.mgrid[0:ny, 0:nx]
    y = (y - ny / 2) / ny
    x = (x - nx / 2) / nx
    cols = [np.ones(ny * nx)]
    for i in range(1, order + 1):
        for j in range(i + 1):
            cols.append((y ** (i - j) * x ** j).ravel())
    return np.stack(cols, axis=1)


def _detrend_patch(patch, order=2):
    """Remove a low-order 2-D polynomial (residual cupping/shading) from a patch."""
    if order <= 0:
        return patch - patch.mean()
    ny, nx = patch.shape
    A = _poly_basis(ny, nx, order)
    coef, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    return patch - (A @ coef).reshape(ny, nx)


def _detrend_nps_response(p, order):
    """
    Multiplicative attenuation that polynomial detrending imposes on the NPS, per bin.

    Detrending is the linear projection R = I - A(A^T A)^-1 A^T onto the complement of
    the polynomial basis A, so for a white input the expected periodogram at frequency
    u is  N - g(u)^T (A^T A)^-1 conj(g(u)),  where g(u) collects the DFT of each basis
    image at u.  The result depends only on (p, order) -- never on the data -- so it can
    be divided straight back out.

    This correction matters here rather than being a nicety: detrending removes ~48 % of
    the lowest-frequency bin at order 2, and the low-frequency end is exactly where
    iteratively reconstructed and denoised noise concentrates its power.  Leaving it in
    would bias NEQ and d' (whose integrands weight low frequencies heavily) and would
    distort f_av for precisely the images the comparison is about.
    """
    if order <= 0:
        att = np.ones((p, p))
        att[0, 0] = 0.0                       # mean subtraction kills DC exactly
        return att
    N = p * p
    A = _poly_basis(p, p, order)
    M_inv = np.linalg.inv(A.T @ A)
    G = np.stack([np.fft.fft2(A[:, j].reshape(p, p)) for j in range(A.shape[1])], axis=0)
    # quadratic form g^T M^-1 conj(g) evaluated at every frequency bin at once
    quad = np.einsum("iyx,ij,jyx->yx", G, M_inv, np.conj(G)).real
    return np.clip(1.0 - quad / N, 0.0, 1.0)


def body_pose(img_hu, pixel_mm, origin_xy=(0.0, 0.0), air_hu=-300.0, dense_hu=300.0):
    """
    Where the body sits and how it is turned, in physical mm -- position AND orientation.

    A centroid alone is blind to rotation: turning an object about its own centre leaves the
    centroid exactly where it was.  So two further quantities are measured, and they fail in
    different ways on purpose:

      * PRINCIPAL AXIS from the second central moments.  Precise, but an axis is a line
        rather than an arrow, so it is only defined modulo 180 deg -- and on a near-circular
        body it is arbitrary, which is what `elongation` is reported for.
      * BEARING of the dense structure (spine, couch) from the centroid.  Much coarser, but
        a full 360 deg, so it resolves the ambiguity the principal axis leaves open.

    Assumes an identity direction matrix, which every volume this pipeline writes has.
    Returns a dict; `axis_deg` is NaN when no body is found.
    """
    img = np.asarray(img_hu, dtype=np.float64)
    mask = detect_body_mask(img, pixel_mm, air_hu=air_hu)
    out = {"n_px": int(mask.sum()), "centre_mm": None, "axis_deg": float("nan"),
           "elongation": float("nan"), "bearing_deg": float("nan"), "n_dense": 0}
    if not mask.any():
        return out
    ys, xs = np.nonzero(mask)
    px = np.stack([origin_xy[0] + xs * pixel_mm, origin_xy[1] + ys * pixel_mm], axis=1)
    centre = px.mean(axis=0)
    out["centre_mm"] = [float(centre[0]), float(centre[1])]

    evals, evecs = np.linalg.eigh(np.cov(px.T))
    major = evecs[:, int(np.argmax(evals))]
    out["axis_deg"] = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    lo, hi = float(np.min(evals)), float(np.max(evals))
    out["elongation"] = float(np.sqrt(hi / lo)) if lo > 0 else float("inf")

    dense = mask & (img > dense_hu)
    out["n_dense"] = int(dense.sum())
    if out["n_dense"] >= 20:
        dy, dx = np.nonzero(dense)
        v = np.array([origin_xy[0] + dx.mean() * pixel_mm,
                      origin_xy[1] + dy.mean() * pixel_mm]) - centre
        if np.hypot(*v) > 2.0:
            out["bearing_deg"] = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    return out


def pose_difference(a, b, weak_elongation=1.15):
    """
    Compare two body_pose results: translation in mm, rotation in degrees.

    The axis difference is wrapped into (-90, +90]: two axes 179 deg apart are 1 deg of
    rotation, not 179.  `reliable` is False when either body is too round for its principal
    axis to mean anything, so a meaningless number is never reported as a measurement.
    """
    out = {"translation_mm": None, "distance_mm": float("nan"),
           "axis_deg": float("nan"), "bearing_deg": float("nan"), "reliable": False}
    if not a or not b or a.get("centre_mm") is None or b.get("centre_mm") is None:
        return out
    d = np.asarray(a["centre_mm"], float) - np.asarray(b["centre_mm"], float)
    out["translation_mm"] = [float(d[0]), float(d[1])]
    out["distance_mm"] = float(np.hypot(*d))
    if np.isfinite(a["axis_deg"]) and np.isfinite(b["axis_deg"]):
        out["axis_deg"] = float((a["axis_deg"] - b["axis_deg"] + 90.0) % 180.0 - 90.0)
    if np.isfinite(a["bearing_deg"]) and np.isfinite(b["bearing_deg"]):
        out["bearing_deg"] = float((a["bearing_deg"] - b["bearing_deg"] + 180.0) % 360.0
                                   - 180.0)
    out["reliable"] = bool(min(a.get("elongation", 0.0), b.get("elongation", 0.0))
                           >= weak_elongation)
    return out


def region_noise_sd(volume, region_mask, detrend_order=2, slices=None):
    """
    Noise SD over an arbitrary region, WITHOUT requiring square patches.

    The NPS needs square blocks because it is a Fourier estimate, but the noise magnitude
    does not -- and on a phantom whose uniform material is a thin ring of matrix between
    inserts, no square of useful size fits while there are still tens of thousands of
    perfectly good voxels.  Tying the headline noise number to the patch geometry would
    throw those away and report nothing at all.

    A low-order 2-D polynomial is fitted over the region's own coordinates and subtracted
    (the same detrending the patches get, generalised to an arbitrary shape), so residual
    cupping or shading across the region does not masquerade as noise.

    Returns (sd, n_voxels_per_slice).
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim == 2:
        vol = vol[None]
    mask = np.asarray(region_mask, dtype=bool)
    if mask.sum() < 16:
        return float("nan"), 0
    ys, xs = np.nonzero(mask)
    ny, nx = mask.shape
    y = (ys - ny / 2) / ny
    x = (xs - nx / 2) / nx
    cols = [np.ones(ys.size)]
    for i in range(1, max(0, int(detrend_order)) + 1):
        for j in range(i + 1):
            cols.append(y ** (i - j) * x ** j)
    A = np.stack(cols, axis=1)
    pinv = np.linalg.pinv(A)

    idx = range(vol.shape[0]) if slices is None else slices
    acc, n = 0.0, 0
    for z in idx:
        v = vol[z][ys, xs]
        resid = v - A @ (pinv @ v)
        acc += float(np.mean(resid * resid))
        n += 1
    return (float(np.sqrt(acc / n)) if n else float("nan")), int(ys.size)


def water_reference(volume, region_mask, pixel_mm, expected_hu=0.0,
                    core_erosion_mm=5.0, slices=None):
    """
    Mean HU over a region of KNOWN material -- the only absolute accuracy check available.

    Every other metric here is relative (one reconstruction against another) or paired
    (a difference between two of them), so nothing says whether the HU scale itself is
    right.  A phantom whose body material is specified as water-equivalent supplies the
    missing reference: the region's mean must be `expected_hu`, and the deviation is a
    number no comparison between modalities can produce.

    NOTE this must NOT be read as accuracy for a reconstruction whose HU calibration was
    itself derived from the same material -- there it measures the calibration's internal
    consistency.  It is the caller's job to label it accordingly; the function only
    reports what it measured.

    `region_mask` is expected to be the noise region, i.e. already insert-excluded, so
    only the uniform material contributes.  The mean is deliberately taken WITHOUT the
    detrending that region_noise_sd applies: a polynomial fit absorbs exactly the
    constant offset being measured.

    A region traced along the module boundary and filled inwards keeps the module WALL
    inside it, and the wall is not water.  Rather than assume a wall thickness, the mean
    is also taken over a core eroded by a further `core_erosion_mm`; if the two disagree,
    `wall_bias_hu` is non-zero and the outer value is contaminated.

    Returns dict: mean_hu, offset_hu, core_mean_hu, wall_bias_hu, n_vox, n_core_vox.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim == 2:
        vol = vol[None]
    mask = np.asarray(region_mask, dtype=bool)
    out = {"mean_hu": float("nan"), "offset_hu": float("nan"),
           "core_mean_hu": float("nan"), "wall_bias_hu": float("nan"),
           "n_vox": int(mask.sum()), "n_core_vox": 0}
    if mask.sum() < 16:
        return out

    idx = list(range(vol.shape[0])) if slices is None else list(slices)
    if not idx:
        return out
    ys, xs = np.nonzero(mask)
    out["mean_hu"] = float(np.mean([np.mean(vol[z][ys, xs]) for z in idx]))
    out["offset_hu"] = out["mean_hu"] - float(expected_hu)

    if core_erosion_mm > 0:
        core = ndimage.binary_erosion(
            mask, _disk(max(1, int(round(core_erosion_mm / pixel_mm)))))
        if core.sum() >= 16:
            cy, cx = np.nonzero(core)
            out["core_mean_hu"] = float(np.mean([np.mean(vol[z][cy, cx]) for z in idx]))
            out["wall_bias_hu"] = out["mean_hu"] - out["core_mean_hu"]
            out["n_core_vox"] = int(core.sum())
    return out


def noise_power_spectrum(volume, patches, pixel_mm, patch_px=64, slices=None,
                         detrend_order=2):
    """
    2-D NPS from uniform patches, averaged over patches and slices.

    volume   : (Z, Y, X) HU
    patches  : [(y0, x0), ...] from background_patches()
    patch_px : MUST be the same patch_px passed to background_patches().  The origins
               were validated for that size; reading a larger block from them reaches
               back over the inserts and the body edge and inflates the "noise" by a
               factor of several -- so a mismatch is rejected rather than inferred.
    slices   : iterable of z indices (default: all)

    Returns dict with the 2-D NPS (HU^2 mm^2), its frequency axes, the radially
    averaged curve, and summary numbers.  After the detrending correction the
    normalisation satisfies
        int int NPS du dv == variance of the underlying noise field
    (verified to ~0.02 % by selftest_image_quality.py), which is slightly ABOVE the
    variance actually measured on the detrended patches -- the difference is the
    low-frequency power detrending removed and the correction puts back.

    Known limitation -- spectral leakage: a finite patch convolves the true NPS with
    the Fejer kernel of the 64x64 window, so power leaks from strong bins into weak
    ones.  This is negligible while the NPS varies by one or two decades across the
    band (measured error ~5 %, the realistic CT case) but inflates the far tail of a
    very steeply falling spectrum.  Compare curves over the range within ~2 decades of
    the peak and do not read the extreme tail quantitatively.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim == 2:
        vol = vol[None]
    if slices is None:
        slices = range(vol.shape[0])
    if not patches:
        raise ValueError("noise_power_spectrum: no background patches supplied")

    p = int(patch_px)
    ny, nx = vol.shape[1:]
    bad = [(y0, x0) for y0, x0 in patches if y0 + p > ny or x0 + p > nx]
    if bad:
        raise ValueError(
            f"noise_power_spectrum: {len(bad)} patch origins do not fit a {p}x{p} block "
            f"inside a {ny}x{nx} image -- patch_px must match the value given to "
            f"background_patches()")
    acc = np.zeros((p, p))
    n = 0
    var_acc = 0.0
    patch_var = []
    for z in slices:
        sl = vol[z]
        for (y0, x0) in patches:
            sub = sl[y0:y0 + p, x0:x0 + p]
            d = _detrend_patch(sub, detrend_order)
            v = float(np.mean(d * d))
            var_acc += v
            patch_var.append(v)
            acc += np.abs(np.fft.fft2(d)) ** 2
            n += 1
    if n == 0:
        raise ValueError("noise_power_spectrum: no usable patches")

    acc /= n
    dx = float(pixel_mm)
    nps2 = acc * (dx * dx) / (p * p)             # -> HU^2 mm^2, Parseval-consistent

    # Undo the (exactly known) low-frequency suppression caused by detrending.
    att = _detrend_nps_response(p, detrend_order)
    ok_att = att > 1e-3
    nps2 = np.where(ok_att, nps2 / np.where(ok_att, att, 1.0), 0.0)

    fx = np.fft.fftfreq(p, d=dx)
    f, nps_r = radial_average(nps2, fx, fx)

    df = 1.0 / (p * dx)
    variance_from_nps = float(nps2.sum() * df * df)

    # Uniformity diagnostics, REPORTED BUT NOT THRESHOLDED.
    #
    # An insert that was never detected stays inside the background, inflating the noise
    # SD and dragging f_av down (structure is low-frequency).  It is tempting to detect
    # that from the patch statistics, but measurement shows it cannot be done reliably:
    # with 16 of 18 inserts left in, the patch-mean spread rises to ~3.7x the white-noise
    # expectation, while CLEAN but strongly correlated noise (an iteratively reconstructed
    # or denoised image -- precisely what the vendor supplies) reaches ~14x. Any threshold
    # separating them would false-alarm on the images the comparison is about.
    #
    # So these are context, not alarms.  The guard that actually works is structural:
    # exclude every insert (supply a segmentation) and check the count against
    # --expect-inserts, then confirm on qc/roi_*.png.
    sd_patch = np.sqrt(np.asarray(patch_var, dtype=float))
    lo_p, hi_p = np.percentile(sd_patch, [10, 90])
    heterogeneity = float(hi_p / lo_p) if lo_p > 0 else float("inf")
    return {
        "nps_2d": np.fft.fftshift(nps2),
        "f_axis": np.fft.fftshift(fx),
        "f": f,
        "nps": nps_r,
        "n_patches": n,
        "patch_px": p,
        "pixel_mm": dx,
        "variance_measured": var_acc / n,
        "variance_from_nps": variance_from_nps,
        "noise_sd_hu": float(np.sqrt(var_acc / n)),
        "patch_sd_p10": float(lo_p), "patch_sd_p90": float(hi_p),
        "patch_heterogeneity": heterogeneity,
        **nps_summary(f, nps_r),
    }


def radial_average(img2d, fx, fy, drop_dc=True):
    """
    Ring-mean of a 2-D spectrum -> (f, radial_mean).  f in the same units as fx.

    drop_dc excludes the exact zero-frequency term.  It must stay on for an NPS:
    detrending forces the DC bin to zero, and since no other FFT bin falls in the
    first radial ring, leaving it in produces a spurious ~0 as the first point of the
    curve -- which then corrupts f_av and anything normalised to nps[0].
    """
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    r = np.sqrt(FX ** 2 + FY ** 2).ravel()
    v = np.asarray(img2d, dtype=np.float64).ravel()
    if drop_dc:
        keep = r > 0
        r, v = r[keep], v[keep]
    d = np.abs(np.diff(np.sort(fx)))
    df = float(np.min(d[d > 0]))
    nb = int(np.ceil(r.max() / df)) + 1
    idx = np.clip((r / df).astype(int), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    tot = np.bincount(idx, weights=v, minlength=nb)
    ok = cnt > 0
    f = (np.arange(nb) + 0.5) * df
    return f[ok], tot[ok] / cnt[ok]


def nps_summary(f, nps):
    """sigma-free texture descriptors of an NPS curve: mean and peak frequency."""
    f = np.asarray(f, dtype=float)
    nps = np.asarray(nps, dtype=float)
    tot = np.trapezoid(nps, f) if hasattr(np, "trapezoid") else np.trapz(nps, f)
    if tot <= 0:
        return {"f_av": float("nan"), "f_peak": float("nan")}
    num = np.trapezoid(f * nps, f) if hasattr(np, "trapezoid") else np.trapz(f * nps, f)
    return {"f_av": float(num / tot), "f_peak": float(f[int(np.argmax(nps))])}


# ═════════════════════════════════════════════════════════════════════
# Task transfer function (circular-edge method)
# ═════════════════════════════════════════════════════════════════════
def radial_edge_profile(img_hu, center_yx, pixel_mm, r_max_mm, bins_per_pixel=4):
    """
    Oversampled edge-spread function around a cylindrical insert.

    Every pixel within r_max_mm contributes its own radial distance, so binning finely
    in r yields an ESF sampled far below the pixel pitch -- this is what makes the
    circular-edge method able to measure resolution beyond the Nyquist of the grid.

    Returns (r_mm, esf) with r_mm ascending.
    """
    img = np.asarray(img_hu, dtype=np.float64)
    cy, cx = center_yx
    ny, nx = img.shape
    r_max_px = r_max_mm / pixel_mm
    y0, y1 = max(0, int(cy - r_max_px - 2)), min(ny, int(cy + r_max_px + 3))
    x0, x1 = max(0, int(cx - r_max_px - 2)), min(nx, int(cx + r_max_px + 3))
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel() * pixel_mm
    v = sub.ravel()
    keep = r <= r_max_mm
    r, v = r[keep], v[keep]

    dr = pixel_mm / float(bins_per_pixel)
    nb = int(np.ceil(r_max_mm / dr))
    idx = np.clip((r / dr).astype(int), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    tot = np.bincount(idx, weights=v, minlength=nb)
    ok = cnt > 0
    return (np.arange(nb)[ok] + 0.5) * dr, tot[ok] / cnt[ok]


def task_transfer_function(img_hu, center_yx, radius_mm, pixel_mm,
                           bins_per_pixel=4, smooth_bins=5):
    """
    TTF from one cylindrical insert, via the circular-edge method (AAPM TG-233).

    ESF -> (smooth) -> differentiate -> LSF -> |FFT| -> normalise to 1 at DC.

    Because TTF is normalised at DC it is a *relative* resolution measure, so it can
    be compared between images with completely different contrast scales -- which is
    what lets the same routine serve threshold images and monoenergetic images.

    Returns dict: f (cyc/mm), ttf, ttf50, ttf10, edge_width_1090_mm, contrast_hu.
    """
    r_max_mm = radius_mm * 2.2
    r, esf = radial_edge_profile(img_hu, center_yx, pixel_mm, r_max_mm, bins_per_pixel)
    if r.size < 16:
        raise ValueError("task_transfer_function: edge profile too short")

    inner = esf[r < radius_mm * 0.6]
    outer = esf[r > radius_mm * 1.5]
    if inner.size < 3 or outer.size < 3:
        raise ValueError("task_transfer_function: insert too small or too close to another")
    lo, hi = float(np.mean(outer)), float(np.mean(inner))
    contrast = hi - lo

    if smooth_bins and smooth_bins >= 3 and r.size > smooth_bins:
        w = int(smooth_bins) | 1
        esf_s = signal.savgol_filter(esf, w, 2)
    else:
        esf_s = esf

    dr = float(np.mean(np.diff(r)))
    lsf = np.gradient(esf_s, dr)
    lsf = np.abs(lsf)
    if lsf.max() <= 0:
        raise ValueError("task_transfer_function: flat edge (no contrast)")

    # Hann window centred on the LSF peak: suppresses the noisy tails without
    # touching the transition itself (untapered tails alias into the TTF).
    k = int(np.argmax(lsf))
    half = max(8, int(round((radius_mm * 0.8) / dr)))
    a, b = max(0, k - half), min(len(lsf), k + half + 1)
    seg = lsf[a:b] * np.hanning(b - a)

    n = int(2 ** np.ceil(np.log2(max(64, seg.size * 4))))     # zero-pad for a smooth curve
    F = np.abs(np.fft.rfft(seg, n))
    if F[0] <= 0:
        raise ValueError("task_transfer_function: zero DC")
    ttf = F / F[0]
    f = np.fft.rfftfreq(n, d=dr)

    # 10-90 % edge width straight off the normalised ESF: the intuitive companion to
    # TTF50, and it needs no Fourier transform to defend in the text.
    prof = (esf_s - lo) / contrast if abs(contrast) > 1e-9 else esf_s * 0
    if contrast < 0:
        prof = 1.0 - prof
    width = _crossing(r, prof, 0.10) - _crossing(r, prof, 0.90)

    return {
        "f": f, "ttf": ttf,
        "ttf50": _ttf_at(f, ttf, 0.50),
        "ttf10": _ttf_at(f, ttf, 0.10),
        "edge_width_1090_mm": float(abs(width)) if np.isfinite(width) else float("nan"),
        "contrast_hu": float(contrast),
        "r_mm": r, "esf": esf_s,
    }


def _crossing(x, y, level):
    """First x where y crosses `level`, linearly interpolated (y assumed monotone-ish)."""
    y = np.asarray(y, dtype=float)
    for i in range(len(y) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            t = (level - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def _ttf_at(f, ttf, level):
    """Frequency (cyc/mm) at which the TTF first falls to `level`."""
    for i in range(len(ttf) - 1):
        if ttf[i] >= level >= ttf[i + 1]:
            d = ttf[i] - ttf[i + 1]
            t = 0.0 if d == 0 else (ttf[i] - level) / d
            return float(f[i] + t * (f[i + 1] - f[i]))
    return float("nan")


# ═════════════════════════════════════════════════════════════════════
# Combined: NEQ and detectability index
# ═════════════════════════════════════════════════════════════════════
def neq(f_ttf, ttf, f_nps, nps, contrast_hu=1.0):
    """
    Noise-equivalent quanta: NEQ(f) = contrast^2 * TTF(f)^2 / NPS(f).

    Literally "resolution squared over noise", frequency by frequency -- the
    principled way to combine the two without assuming what is being looked for.
    Higher = more usable information at that spatial scale.  Units: mm^-2 when NPS
    is in HU^2 mm^2 and contrast in HU.

    Returns (f, neq) on the NPS frequency grid (the coarser of the two).
    """
    f = np.asarray(f_nps, dtype=float)
    t = np.interp(f, np.asarray(f_ttf, dtype=float), np.asarray(ttf, dtype=float))
    n = np.asarray(nps, dtype=float)
    out = np.full(f.shape, np.nan)
    ok = n > 0
    out[ok] = (contrast_hu ** 2) * (t[ok] ** 2) / n[ok]
    return f, out


def detectability_index(f_ttf, ttf, f_nps, nps,
                        task_diameter_mm=5.0, task_contrast_hu=50.0):
    """
    Ideal-observer detectability index d' for a flat circular lesion.

        d'^2 = 2*pi * int f * |W(f)|^2 * TTF(f)^2 / NPS(f) df

    with W(f) the 2-D Fourier transform of a disc of the stated diameter and contrast
    (a jinc).  One number per image, but it depends on the assumed task, so the task
    parameters are returned alongside and must be quoted with the value.
    """
    from scipy.special import j1

    f = np.asarray(f_nps, dtype=float)
    t = np.interp(f, np.asarray(f_ttf, dtype=float), np.asarray(ttf, dtype=float))
    n = np.asarray(nps, dtype=float)

    R = 0.5 * float(task_diameter_mm)
    a = 2 * np.pi * R * f
    W = np.where(a > 1e-9, task_contrast_hu * np.pi * R * R * 2 * j1(np.maximum(a, 1e-9)) / np.maximum(a, 1e-9),
                 task_contrast_hu * np.pi * R * R)
    integ = np.zeros_like(f)
    ok = n > 0
    integ[ok] = f[ok] * (W[ok] ** 2) * (t[ok] ** 2) / n[ok]
    d2 = 2 * np.pi * (np.trapezoid(integ, f) if hasattr(np, "trapezoid") else np.trapz(integ, f))
    return {
        "d_prime": float(np.sqrt(max(d2, 0.0))),
        "task_diameter_mm": float(task_diameter_mm),
        "task_contrast_hu": float(task_contrast_hu),
    }


# ═════════════════════════════════════════════════════════════════════
# Paired comparison (own vs WFBP) -- separate bias from independent noise
# ═════════════════════════════════════════════════════════════════════
def difference_analysis(vol_a, vol_b, pixel_mm, body_mask=None, smooth_mm=5.0):
    """
    Split (a - b) into a systematic and a noise part.

    A plain RMSE between two independent reconstructions of the same object is
    dominated by the fact that their noise is independent: for unbiased a and b it
    tends to sqrt(sigma_a^2 + sigma_b^2) even when both are perfect.  Low-pass
    filtering the difference averages that independent noise away and leaves the part
    that actually matters -- HU offsets, cupping, shading, geometric distortion.

    Returns dict with the systematic RMS/mean/max (HU), the total RMSE for reference,
    and the smoothed difference map for plotting.
    """
    a = np.asarray(vol_a, dtype=np.float32)
    b = np.asarray(vol_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"difference_analysis: shape mismatch {a.shape} vs {b.shape}")
    d = a - b
    sig = ndimage.gaussian_filter(d, sigma=(0, smooth_mm / pixel_mm, smooth_mm / pixel_mm))
    m = np.ones(a.shape, dtype=bool) if body_mask is None else np.broadcast_to(body_mask, a.shape)
    return {
        "rmse_total_hu": float(np.sqrt(np.mean(d[m] ** 2))),
        "systematic_rms_hu": float(np.sqrt(np.mean(sig[m] ** 2))),
        "systematic_mean_hu": float(np.mean(sig[m])),
        "systematic_max_abs_hu": float(np.max(np.abs(sig[m]))),
        "noise_rms_hu": float(np.sqrt(np.mean((d - sig)[m] ** 2))),
        "systematic_map": sig,
    }


def roi_statistics(volume, inserts, body_mask, pixel_mm, shrink=0.7):
    """
    Mean/SD in each insert (shrunk to avoid the partial-volume rim) and in the
    background -- the inputs to a Bland-Altman comparison and to CNR.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None]
    ny, nx = vol.shape[1:]
    y, x = np.ogrid[:ny, :nx]
    bg = body_mask.copy()
    out = {"inserts": []}
    for i, ins in enumerate(inserts):
        rr = ins["radius_px"] * shrink
        m = ((y - ins["cy"]) ** 2 + (x - ins["cx"]) ** 2) <= rr * rr
        bg &= ((y - ins["cy"]) ** 2 + (x - ins["cx"]) ** 2) > (ins["radius_px"] + 6.0 / pixel_mm) ** 2
        v = vol[:, m]
        out["inserts"].append({
            "index": i, "cy": ins["cy"], "cx": ins["cx"], "radius_mm": ins["radius_mm"],
            "mean_hu": float(v.mean()), "sd_hu": float(v.std()), "n_vox": int(v.size),
        })
    vb = vol[:, bg]
    out["background"] = {"mean_hu": float(vb.mean()), "sd_hu": float(vb.std()),
                         "n_vox": int(vb.size)}
    for ins in out["inserts"]:
        ins["cnr"] = ((ins["mean_hu"] - out["background"]["mean_hu"])
                      / out["background"]["sd_hu"]) if out["background"]["sd_hu"] > 0 else float("nan")
    return out


# ═════════════════════════════════════════════════════════════════════
# Slab / structure detection
# ═════════════════════════════════════════════════════════════════════
def find_insert_slab(volume, pixel_mm, z_positions_mm, air_hu=-300.0, smooth_mm=1.0,
                     search_z_mm=None, min_run=3, select="peak"):
    """
    Locate the z-range containing a phantom insert layer.

    The insert layer is a run of slices with strong in-plane structure inside the body;
    homogeneous layers have none.  Structure is scored as the FRACTION of body voxels
    lying far from the body median, after light smoothing so that quantum noise -- the
    very thing that differs between the channels -- does not drive the detection.  A
    spread statistic such as the MAD does not work here: inserts occupy only a few
    percent of the body area, and MAD is designed precisely to discard a small minority
    of deviant voxels, so it stays flat straight through the insert layer.

    search_z_mm : (lo, hi) restricting where a slab may be chosen.  **Supply this for a
        long clinical scan range.**  A Thx-Abdomen acquisition contains the table, the
        positioning aids and scan-end artefacts, any of which can out-score the phantom
        and put the slab hundreds of millimetres away from it.  Slices outside the range
        are not even scored, which is also much faster.

    select : 'peak'    -- the run containing the single most insert-covered slice.
                          Best when layers differ in how many inserts they carry, since
                          the score is essentially the insert area fraction.
             'longest' -- the longest run above threshold (the previous behaviour).

    Returns dict with the chosen k_lo/k_hi/z_lo_mm/z_hi_mm, the full per-slice `score`,
    and **`candidates`**: every run found, so a layer that was not selected is still
    visible in the log and the QC figure rather than being silently discarded.
    """
    vol = np.asarray(volume, dtype=np.float32)
    nz = vol.shape[0]
    z = np.asarray(z_positions_mm, dtype=float)
    score = np.zeros(nz)

    if search_z_mm is not None:
        lo, hi = sorted(float(v) for v in search_z_mm)
        in_range = (z >= lo) & (z <= hi)
        if not in_range.any():
            raise ValueError(
                f"find_insert_slab: search range {lo:.1f}..{hi:.1f} mm contains no "
                f"slices (volume spans {z.min():.1f}..{z.max():.1f} mm)")
    else:
        in_range = np.ones(nz, dtype=bool)

    sigma_px = max(0.5, smooth_mm / pixel_mm)
    # The body edge must be eroded away before scoring: smoothing drags the -1000 HU
    # air across the boundary, producing a rim of extreme values in EVERY slice, which
    # would otherwise score as "structure" and flatten the curve completely.
    erode_px = int(np.ceil(3 * sigma_px))
    for k in range(nz):
        if not in_range[k]:
            continue
        sl = vol[k]
        body = sl > air_hu
        if body.sum() < 100:
            continue
        body = ndimage.binary_erosion(body, _disk(erode_px))
        if body.sum() < 100:
            continue
        sm = ndimage.gaussian_filter(sl, sigma=sigma_px)
        v = sm[body]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        if mad <= 0:
            continue
        score[k] = float(np.mean(np.abs(v - med) > 5.0 * mad))

    if not np.any(score > 0):
        raise ValueError("find_insert_slab: no structure found in the searched range")

    # Bridge single-slice dropouts before thresholding at a fraction of the peak.
    score_s = ndimage.median_filter(score, size=3, mode="nearest")
    score_s[~in_range] = 0.0
    thr = 0.25 * float(score_s.max())
    above = score_s >= thr

    runs = []
    cur = None
    for k in range(nz + 1):
        hot = bool(above[k]) if k < nz else False
        if hot and cur is None:
            cur = k
        elif not hot and cur is not None:
            a, b = cur, k - 1
            if b - a + 1 >= int(min_run):
                runs.append({
                    "k_lo": int(a), "k_hi": int(b), "n_slices": int(b - a + 1),
                    "z_lo_mm": float(min(z[a], z[b])), "z_hi_mm": float(max(z[a], z[b])),
                    "peak_score": float(score_s[a:b + 1].max()),
                    "mean_score": float(score_s[a:b + 1].mean()),
                })
            cur = None
    if not runs:                       # nothing survived min_run -- fall back to the peak
        k = int(np.argmax(score_s))
        runs = [{"k_lo": k, "k_hi": k, "n_slices": 1,
                 "z_lo_mm": float(z[k]), "z_hi_mm": float(z[k]),
                 "peak_score": float(score_s[k]), "mean_score": float(score_s[k])}]

    key = (lambda r: r["n_slices"]) if select == "longest" else (lambda r: r["peak_score"])
    best = max(runs, key=key)
    return {"k_lo": best["k_lo"], "k_hi": best["k_hi"],
            "z_lo_mm": best["z_lo_mm"], "z_hi_mm": best["z_hi_mm"],
            "score": score, "threshold": float(thr),
            "candidates": sorted(runs, key=key, reverse=True),
            "select": select}


def match_slice_thickness(vol_native, z_native_mm, target_thickness_mm, target_z_mm):
    """
    Average native slices into thicker ones, reproducing a target slice profile.

    Averaging adjacent slices is physically what a thicker slice IS, so this matches
    the vendor's slice sensitivity profile rather than approximating it with a
    Gaussian.  Doing it this way also avoids the sqrt(N) noise shortcut, which is
    wrong here because SSR slices at neighbouring z share detector rows and are
    therefore correlated.

    Returns (vol_matched, n_used_per_slice).
    """
    zn = np.asarray(z_native_mm, dtype=float)
    out = np.zeros((len(target_z_mm), *vol_native.shape[1:]), dtype=np.float32)
    used = []
    half = 0.5 * float(target_thickness_mm)
    for i, zc in enumerate(target_z_mm):
        sel = np.where(np.abs(zn - zc) <= half)[0]
        if sel.size == 0:                      # thinner target than native spacing
            sel = np.array([int(np.argmin(np.abs(zn - zc)))])
        out[i] = vol_native[sel].mean(axis=0)
        used.append(int(sel.size))
    return out, used
