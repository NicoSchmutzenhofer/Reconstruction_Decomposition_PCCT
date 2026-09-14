"""
build_mono_table.py -- generate the monoenergetic mass-attenuation table for VMI decomposition.

One-off data-prep tool.  Reads the per-element NIST XCOM sheets in
NIST_XrayMassCoef_Elements_BinSignature.xlsx and writes data/mu_rho_mono.csv, <mu/rho>
(cm^2/g) per library material on the NIST energy grid, so the runtime never needs the Excel:
  * elements (Iodine, Iron, Titanium, Tantalum, Platinum, Gold): the element's sheet,
    including the K-edge doublets, marked edge_side below/above;
  * composites (Water, Fat, SoftTissue, HA): mass-weighted sum of their elements on the
    common energies; light elements only, so no edge inside the CT range;
  * StainlessSteel and CoCr: skipped, as they need Ni, which has no sheet.

Needs pandas and openpyxl.

    python decomposition/build_mono_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).resolve().parent
_XLSX = _DIR / "NIST_XrayMassCoef_Elements_BinSignature.xlsx"
_OUT = _DIR / "data" / "mu_rho_mono.csv"

EMIN_KEV, EMAX_KEV = 15.0, 200.0     # margin around the 20-140 keV working range

# Mass fractions (from the Excel's documented compositions; ICRU adipose / ICRU-44 soft tissue /
# hydroxyapatite Ca10(PO4)6(OH)2 / water). Pure elements map to a single sheet.
COMPOSITION = {
    "Water":      {"H": 0.111894, "O": 0.888106},
    "Fat":        {"H": 0.114, "C": 0.598, "N": 0.007, "O": 0.278, "Na": 0.001, "S": 0.001, "Cl": 0.001},
    "SoftTissue": {"H": 0.102, "C": 0.143, "N": 0.034, "O": 0.708, "Na": 0.002, "P": 0.003,
                   "S": 0.003, "Cl": 0.002, "K": 0.003},
    "HA":         {"H": 0.002, "O": 0.414, "P": 0.185, "Ca": 0.399},
    "Iodine":     {"I": 1.0},
    "Iron":       {"Fe": 1.0},
    "Titanium":   {"Ti": 1.0},
    "Tantalum":   {"Ta": 1.0},
    "Platinum":   {"Pt": 1.0},
    "Gold":       {"Au": 1.0},
}
SKIP = {"StainlessSteel": "needs Ni (Z=28) -- no element sheet", "CoCr": "needs Ni (Z=28) -- no element sheet"}


def _sheet_for_symbol(xl):
    """Map element symbol -> sheet name, e.g. 'I' -> 'I (Z=53)'."""
    m = {}
    for s in xl.sheet_names:
        if "Z=" in s:
            m[s.split("(")[0].strip()] = s
    return m


def _read_element(xl, sheet):
    """(keV, mu/rho, edge_side) rows in [EMIN,EMAX], edge doublets preserved (below then above)."""
    df = pd.read_excel(xl, sheet_name=sheet, header=None)
    hdr = next(i for i in range(len(df)) if str(df.iloc[i, 0]).strip().startswith("Energy (MeV)"))
    kev = pd.to_numeric(df.iloc[hdr + 1:, 1], errors="coerce").values
    mur = pd.to_numeric(df.iloc[hdr + 1:, 2], errors="coerce").values
    out = []
    for j in range(len(kev)):
        e, v = kev[j], mur[j]
        if pd.isna(e) or pd.isna(v) or not (EMIN_KEV <= e <= EMAX_KEV):
            continue
        side = ""
        if j > 0 and pd.notna(kev[j - 1]) and abs(kev[j - 1] - e) < 1e-6:
            side = "above"                       # duplicate energy: this is the post-edge row
        elif j + 1 < len(kev) and pd.notna(kev[j + 1]) and abs(kev[j + 1] - e) < 1e-6:
            side = "below"                       # next row shares the energy: this is pre-edge
        out.append((round(float(e), 4), float(v), side))
    return out


def main():
    xl = pd.ExcelFile(_XLSX)
    sym2sheet = _sheet_for_symbol(xl)
    elem_rows = {sym: _read_element(xl, sh) for sym, sh in sym2sheet.items()}

    def elem_map(sym):
        """{keV: mu/rho} for an element with NO in-range edge (single-valued)."""
        d = {}
        for e, v, side in elem_rows[sym]:
            if side:
                raise RuntimeError(f"element {sym} has an in-range edge; cannot use as smooth composite constituent")
            d[e] = v
        return d

    rows = []  # (material, energy_keV, mu_rho, edge_side)
    for mat, comp in COMPOSITION.items():
        if len(comp) == 1 and list(comp.values())[0] == 1.0:      # pure element
            sym = next(iter(comp))
            for e, v, side in elem_rows[sym]:
                rows.append((mat, e, v, side))
        else:                                                     # composite: mass-weighted sum
            maps = {sym: elem_map(sym) for sym in comp}
            common = set.intersection(*[set(m) for m in maps.values()])
            for e in sorted(common):
                v = sum(w * maps[sym][e] for sym, w in comp.items())
                rows.append((mat, e, round(v, 6), ""))

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["material", "energy_keV", "mu_rho_cm2_g", "edge_side"])
        w.writerows(rows)

    # ---- report + sanity checks ----
    print(f"wrote {_OUT}  ({len(rows)} rows, {len(COMPOSITION)} materials)")
    for m, why in SKIP.items():
        print(f"  SKIPPED {m}: {why}")
    def at(mat, kev):
        cand = [(e, v) for (mm, e, v, s) in rows if mm == mat and abs(e - kev) < 1e-6 and s != "below"]
        return cand[0][1] if cand else None
    print("  sanity (cm^2/g):")
    print(f"    Iodine @80keV     = {at('Iodine', 80):.3f}   (NIST 3.51)")
    print(f"    Iodine @33.17 abv = {at('Iodine', 33.1694):.2f}   (NIST 35.82)")
    print(f"    Water  @80keV     = {at('Water', 80):.4f}   (NIST ~0.1837)")
    print(f"    Water  @60keV     = {at('Water', 60):.4f}   (NIST ~0.2059)")
    print(f"    SoftTissue @80keV = {at('SoftTissue', 80):.4f}")
    print(f"    HA @80keV         = {at('HA', 80):.4f}")


if __name__ == "__main__":
    main()
