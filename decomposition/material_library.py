"""
material_library.py -- material X-ray signatures, the columns of M.

Builds one <mu/rho> (cm^2/g) per material and energy channel from checked-in tables in data/:
  mu_rho_binavg.csv   bin-averaged values per threshold option and bin (NIST, K-edge aware)
  mu_rho_mono.csv     monoenergetic values, for VMI channels and cumulative windows
  materials.csv       density, K-edge, category and description per material
Cumulative windows are integrated here, uniformly or spectrum-weighted (Weighting).  Adding a
material is one column in mu_rho_binavg.csv plus one row in materials.csv; it then appears in
available_materials() and works in every mode.  No volume I/O; the only state is a read cache.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent / "data"
_COEF_CSV = _DATA_DIR / "mu_rho_binavg.csv"
_MAT_CSV = _DATA_DIR / "materials.csv"

# Detector spectral range (keV) -- used to flag whether a K-edge is exploitable.
DETECTOR_RANGE_KEV: Tuple[float, float] = (20.0, 140.0)

_DESCRIPTOR_COLS = {"option", "bin", "e_lo_keV", "e_hi_keV"}


@dataclass(frozen=True)
class MaterialInfo:
    name: str
    density_g_cm3: float
    k_edge_keV: Optional[float]
    category: str
    description: str = ""

    @property
    def k_edge_in_range(self) -> bool:
        lo, hi = DETECTOR_RANGE_KEV
        return self.k_edge_keV is not None and lo <= self.k_edge_keV <= hi


@lru_cache(maxsize=1)
def _coef_rows() -> Tuple[dict, ...]:
    with open(_COEF_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No coefficient rows in {_COEF_CSV}")
    return tuple(rows)


@lru_cache(maxsize=1)
def load_materials() -> Dict[str, MaterialInfo]:
    mats: Dict[str, MaterialInfo] = {}
    with open(_MAT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            name = row["material"].strip()
            k = (row.get("k_edge_keV") or "").strip()
            mats[name] = MaterialInfo(
                name=name,
                density_g_cm3=float(row["density_g_cm3"]),
                k_edge_keV=float(k) if k else None,
                category=(row.get("category") or "").strip(),
                description=(row.get("description") or "").strip(),
            )
    return mats


def material_info(material: str) -> MaterialInfo:
    mats = load_materials()
    if material not in mats:
        raise KeyError(f"Unknown material '{material}'. Known: {sorted(mats)}")
    return mats[material]


def available_materials() -> List[str]:
    """Material names that have a signature column in mu_rho_binavg.csv."""
    return [c for c in _coef_rows()[0].keys() if c not in _DESCRIPTOR_COLS]


def available_options() -> List[int]:
    return sorted({int(r["option"]) for r in _coef_rows()})


def _option_rows(option: int) -> List[dict]:
    rows = [r for r in _coef_rows() if int(r["option"]) == option]
    if not rows:
        raise KeyError(f"Threshold option {option} not in {_COEF_CSV}. "
                       f"Available: {available_options()}")
    rows.sort(key=lambda r: int(r["bin"]))
    return rows


def bin_edges(option: int = 1) -> List[Tuple[float, float]]:
    """[(e_lo, e_hi), ...] keV for each energy bin of `option`, ordered by bin index."""
    return [(float(r["e_lo_keV"]), float(r["e_hi_keV"])) for r in _option_rows(option)]


def n_bins(option: int = 1) -> int:
    return len(_option_rows(option))


def material_signature(material: str, option: int = 1) -> np.ndarray:
    """<mu/rho> (cm^2/g) across the bins of `option`, ordered by bin index."""
    rows = _option_rows(option)
    if material not in rows[0]:
        raise KeyError(f"Unknown material '{material}'. Available: {available_materials()}")
    return np.array([float(r[material]) for r in rows], dtype=float)


def build_M(materials: Sequence[str], spec=1) -> np.ndarray:
    """
    Material-signature matrix M, shape (n_channels, n_materials).

    Columns are the materials' <mu/rho> (cm^2/g) in the given order -- the M of the forward
    model B = M x, with x = partial density (g/cm^3). `spec` selects the energy structure:
      - int  -> a threshold Option (back-compat): the bin-averaged windows of mu_rho_binavg.csv.
      - list of Channel -> arbitrary channels (threshold windows and/or monoenergetic VMI
        energies); each channel's signature is generated on demand (bin-avg lookup or mono
        interpolation), so the channel COUNT is whatever the data has -- not fixed at 4.
    """
    if len(materials) == 0:
        raise ValueError("Need at least one material to build M")
    if isinstance(spec, (int, np.integer)):
        return np.column_stack([material_signature(m, int(spec)) for m in materials])
    channels = list(spec)
    if not channels:
        raise ValueError("Need at least one channel to build M")
    return np.array([[signature_for_channel(m, ch) for m in materials] for ch in channels],
                    dtype=float)


def densities(materials: Sequence[str]) -> np.ndarray:
    """Bulk densities (g/cm^3) of the given materials, in order."""
    mats = load_materials()
    return np.array([mats[m].density_g_cm3 for m in materials], dtype=float)


# ============================================================================
# Channels -- arbitrary energy structure (threshold windows and/or VMI monoenergies)
# ============================================================================
# A channel describes one volume of an input stack.  The channel COUNT and kind come from the
# data (the loader), never a hardcoded 4: own recon / Siemens WFBP -> 'threshold' windows;
# Siemens VMI -> 'mono' energies.  build_M(materials, channels) generates the matching signature
# for each channel, or errors clearly if it cannot (defensive).
_MONO_CSV = _DATA_DIR / "mu_rho_mono.csv"


@dataclass(frozen=True)
class Weighting:
    """
    How an INTEGRATED window average is weighted over energy.

    Only reaches the cumulative windows, which this module integrates itself.  The
    disjoint windows are read from mu_rho_binavg.csv as published and a monoenergetic
    channel is a point value, so neither can be moved by this and neither should be.

    'uniform' is the default and the published table's own weighting (see window_mu_rho).
    'spectrum' weights by the photon fluence a counting detector actually sees, which is
    the sensitivity test the cumulative windows need: [20, 140] keV is 120 keV wide, and a
    flat mean over it gives the same weight to 20 keV, where iodine reads 25 cm^2/g and
    almost no photons survive the patient, as to 100 keV, where most of the counts are.

    Carried ON THE CHANNEL rather than passed through build_M so that every consumer --
    the matrix, the water attenuation used for HU -> mu, the run signature, run.json --
    reads one value and they cannot disagree.  A frozen dataclass so it stays hashable for
    the lru_cache on window_mu_rho.
    """
    kind: str = "uniform"          # 'uniform' | 'spectrum'
    kvp: float = 140.0             # tube potential (kV) = the bremsstrahlung endpoint
    water_cm: float = 16.0         # phantom path length used as the filtration

    @property
    def label(self) -> str:
        if self.kind == "uniform":
            return "uniform"
        return f"spectrum({self.kvp:g}kV, {self.water_cm:g}cm water)"

    def to_dict(self) -> dict:
        return ({"kind": "uniform"} if self.kind == "uniform"
                else {"kind": self.kind, "kvp": float(self.kvp),
                      "water_cm": float(self.water_cm)})


UNIFORM = Weighting()


def parse_weighting(kind: str, kvp: float = 140.0, water_cm: float = 16.0) -> Weighting:
    kind = (kind or "uniform").strip().lower()
    if kind not in ("uniform", "spectrum"):
        raise ValueError(f"weighting must be 'uniform' or 'spectrum', got {kind!r}")
    return Weighting(kind=kind, kvp=float(kvp), water_cm=float(water_cm))


def spectrum_weight(energies, w: Weighting) -> np.ndarray:
    """
    Relative photon fluence at each energy, for the window average of `w`.

    Kramers bremsstrahlung, N(E) proportional to (E_max - E)/E, hardened by `water_cm` of
    water as a stand-in for the phantom.  No factor of E: this detector counts photons, so
    the weight is the fluence itself and not the energy it carries -- that factor would
    belong to an energy-integrating detector.

    The tube's own filtration is NOT modelled, because the monoenergetic table holds no
    aluminium and inventing a coefficient for it would put a fabricated number under every
    signature.  Leaving it out makes the model SOFTER than the real beam, so the weighting
    it produces sits between uniform and the truth: if a result moves under this weight it
    moves at least this much under the real one, which is the direction a sensitivity test
    should err in.
    """
    e = np.asarray(energies, dtype=float)
    n = np.clip(w.kvp - e, 0.0, None) / np.maximum(e, 1e-6)
    if w.water_cm > 0:
        mu = np.array([mono_mu_rho("Water", float(x)) for x in e], dtype=float)
        n = n * np.exp(-mu * float(w.water_cm))       # rho_water = 1 g/cm^3
    return n


@dataclass(frozen=True)
class Channel:
    kind: str                                          # 'threshold' | 'mono'
    label: str = ""
    energy_keV: Optional[float] = None                 # mono
    window_keV: Optional[Tuple[float, float]] = None   # threshold (e_lo, e_hi)
    option: Optional[int] = None                       # threshold: source option in the bin-avg CSV
    bin_index: Optional[int] = None                    # threshold: 0-based bin index in that option
    weighting: Weighting = UNIFORM                     # only used when bin_index is None


@lru_cache(maxsize=1)
def _mono_table() -> Dict[str, tuple]:
    """{material: ((energy_keV, mu/rho), ...) sorted by energy} from mu_rho_mono.csv."""
    tab: Dict[str, list] = {}
    with open(_MONO_CSV, newline="") as f:
        for row in csv.DictReader(f):
            tab.setdefault(row["material"].strip(), []).append(
                (float(row["energy_keV"]), float(row["mu_rho_cm2_g"])))
    for m in tab:
        tab[m].sort(key=lambda t: t[0])
    return {m: tuple(v) for m, v in tab.items()}


def has_mono(material: str) -> bool:
    return material in _mono_table()


def available_mono_materials() -> List[str]:
    return sorted(_mono_table())


def _mono_segments(rows: Sequence[Tuple[float, float]]) -> List[list]:
    """Split (E, mu/rho) rows into monotone segments at K-edges (duplicate energies)."""
    segs, start = [], 0
    for i in range(1, len(rows)):
        if abs(rows[i][0] - rows[i - 1][0]) < 1e-9:   # edge: rows[i-1]=below end, rows[i]=above start
            segs.append(list(rows[start:i]))
            start = i
    segs.append(list(rows[start:]))
    return segs


def mono_mu_rho(material: str, energy_keV: float) -> float:
    """
    Monoenergetic <mu/rho> (cm^2/g) at `energy_keV`, log-log interpolated and K-edge-AWARE:
    interpolation never crosses an absorption edge (the table keeps below/above doublets), and at
    an exact edge energy the post-edge value is used.  Raises if the material has no mono data
    (e.g. StainlessSteel/CoCr -- missing element) or the energy is outside the tabulated range.
    """
    tab = _mono_table()
    if material not in tab:
        raise KeyError(f"No monoenergetic mu/rho for '{material}' -- cannot generate a VMI "
                       f"signature. Available: {available_mono_materials()}")
    rows = tab[material]
    E = float(energy_keV)
    chosen = None
    for seg in _mono_segments(rows):                  # later (upper) segment wins at an edge boundary
        if seg[0][0] - 1e-9 <= E <= seg[-1][0] + 1e-9:
            chosen = seg
    if chosen is None:
        raise ValueError(f"energy {E} keV outside tabulated range "
                         f"[{rows[0][0]}, {rows[-1][0]}] keV for '{material}'")
    xs = np.log(np.array([t[0] for t in chosen]))
    ys = np.log(np.array([t[1] for t in chosen]))
    return float(np.exp(np.interp(np.log(E), xs, ys)))


@lru_cache(maxsize=512)
def window_mu_rho(material: str, e_lo: float, e_hi: float, per_keV: int = 40,
                  weighting: Weighting = UNIFORM) -> float:
    """
    <mu/rho> (cm^2/g) over an ARBITRARY energy window, from the monoenergetic table.

    UNIFORM by default, not spectrum-weighted -- because that is demonstrably what
    mu_rho_binavg.csv already is.  A plain unweighted mean over each published window
    reproduces all 120 of its values (12 windows x 10 materials) to 0.37 % max, 0.10 % rms;
    fitting a free weighting curve instead recovers a flat one and does no better.  The
    sheet's "Source: 140 keV tube" names the energy RANGE, not a fluence weighting.  That
    default must not change: every validated number in the project rests on that table, and
    a different weighting would silently move all 120 published values at once.

    `weighting='spectrum'` is the deliberate exception, and it is not a correction to the
    table -- it never touches it.  It applies only where this function is asked for a
    window the table does not tabulate, which in practice is the CUMULATIVE set [T_i, 140]
    that the threshold channels actually deliver.  There the uniform mean is hard to
    defend: [20, 140] spans 120 keV and weights 20 keV, where iodine reads 25 cm^2/g and
    hardly a photon survives the patient, the same as 100 keV where most of the counts sit.
    The two settings therefore bracket the answer rather than one replacing the other, and
    both are recorded in the run signature so a result is always readable with the one that
    produced it.
    """
    lo, hi = float(min(e_lo, e_hi)), float(max(e_lo, e_hi))
    if hi - lo < 1e-9:
        return mono_mu_rho(material, lo)
    n = max(int(round((hi - lo) * max(int(per_keV), 1))) + 1, 33)
    grid = np.linspace(lo, hi, n)
    vals = np.array([mono_mu_rho(material, float(e)) for e in grid])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    if weighting.kind == "uniform":
        return float(trapz(vals, grid) / (hi - lo))
    wt = spectrum_weight(grid, weighting)
    norm = float(trapz(wt, grid))
    if not np.isfinite(norm) or norm <= 0.0:            # window entirely above the endpoint
        return float(trapz(vals, grid) / (hi - lo))
    return float(trapz(vals * wt, grid) / norm)


def threshold_channels(option: int = 1, cumulative: bool = False,
                       weighting: Weighting = UNIFORM) -> List[Channel]:
    """
    The threshold channels of a mu_rho_binavg.csv option.

    cumulative=False -> the DISJOINT windows the CSV tabulates ([20,40], [40,56], ...).
        Correct when the measurement stack has been differenced into separate energy bins
        (bin_domain='exclusive').

    cumulative=True  -> the NESTED windows the detector actually delivers ([20,140],
        [40,140], ...), each signature integrated from the monoenergetic table.  Correct
        when the thresholds are fed as-is (bin_domain='cumulative'), where every channel
        counts every photon ABOVE its threshold.  Pairing cumulative data with the disjoint
        table describes a quantity the data does not contain; it is also the worse-
        conditioned of the two (kappa 159 vs 76 for Water/HA/Iodine).

    `bin_index=None` marks the cumulative form, which is what makes signature_for_channel
    integrate rather than look the row up.
    """
    edges = bin_edges(option)
    if not cumulative:
        return [Channel(kind="threshold", label=f"T{i+1}", window_keV=e,
                        option=int(option), bin_index=i)
                for i, e in enumerate(edges)]
    e_max = max(hi for _lo, hi in edges)
    return [Channel(kind="threshold", label=f"T{i+1}", window_keV=(lo, e_max),
                    option=int(option), bin_index=None, weighting=weighting)
            for i, (lo, _hi) in enumerate(edges)]


def as_cumulative_channels(channels: Sequence[Channel],
                           weighting: Weighting = UNIFORM) -> List[Channel]:
    """
    Re-express threshold channels on their cumulative windows [T_i, E_max].

    Applied to whatever the loader produced, so every input path gets the same treatment
    and none of them has to know about the bin domain.  Monoenergetic channels and channels
    already cumulative are returned unchanged.

    `weighting` is stamped on the rewritten channels here because this is the one place a
    cumulative window is created; a channel that keeps its published bin index is never
    integrated and so is never weighted.
    """
    chs = list(channels)
    thr = [c for c in chs if c.kind == "threshold"]
    if not thr or any(c.bin_index is None for c in thr):
        return chs
    e_max = max((c.window_keV[1] for c in thr if c.window_keV), default=None)
    if e_max is None:
        return chs
    out = []
    for c in chs:
        if c.kind == "threshold" and c.window_keV:
            out.append(Channel(kind="threshold", label=c.label,
                               window_keV=(c.window_keV[0], float(e_max)),
                               option=c.option, bin_index=None, weighting=weighting))
        else:
            out.append(c)
    return out


def mono_channels(energies_keV: Sequence[float], labels: Optional[Sequence[str]] = None) -> List[Channel]:
    """Monoenergetic (VMI) channels at the given keV energies."""
    ekv = list(energies_keV)
    return [Channel(kind="mono", label=(labels[i] if labels else f"{e:g}keV"), energy_keV=float(e))
            for i, e in enumerate(ekv)]


def signature_for_channel(material: str, ch: Channel) -> float:
    """<mu/rho> (cm^2/g) of `material` for one channel, generated per the channel's kind."""
    if ch.kind == "threshold":
        if ch.bin_index is None:                      # cumulative window -> integrate
            if not ch.window_keV:
                raise ValueError(f"threshold channel '{ch.label}' has neither a bin index "
                                 f"nor a window to integrate over")
            return window_mu_rho(material, float(ch.window_keV[0]), float(ch.window_keV[1]),
                                 weighting=ch.weighting)
        return float(material_signature(material, ch.option)[ch.bin_index])
    if ch.kind == "mono":
        return mono_mu_rho(material, ch.energy_keV)
    raise ValueError(f"Unknown channel kind '{ch.kind}' (expected 'threshold' or 'mono')")


def channel_water_mu(ch: Channel) -> float:
    """Physical water linear attenuation (1/cm) for HU->mu of this channel (Water rho = 1)."""
    return float(signature_for_channel("Water", ch))
