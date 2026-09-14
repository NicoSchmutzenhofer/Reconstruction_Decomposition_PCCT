"""
probe_dose.py -- acquisition and dose parameters of every scan used in this work.

For each labelled DICOM folder, reads the tags in TAGS (kV, mA, CTDIvol, pitch, collimation,
kernel, ...) from up to SAMPLE files per series, reports a range where the files disagree
(e.g. under tube-current modulation) and prints the z-extent of the sampled slices.  DLP is
not in image headers: a radiation-dose structured report is flagged when present, as the
place to read it.  For our own reconstruction, the raw descriptor is searched for fields with
matching names.

    python -m reconstruction.probe_dose --dicom wfbp=/path/to/wfbp vmi=/path/to/vmi \\
                                        --desc /path/to/descriptor.mat [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (keyword, label, unit) -- keyword is the pydicom name, which is stable across versions
TAGS = [
    ("KVP", "tube potential", "kV"),
    ("XRayTubeCurrent", "tube current", "mA"),
    ("Exposure", "exposure", "mAs"),
    ("ExposureTime", "exposure time", "ms"),
    ("CTDIvol", "CTDIvol", "mGy"),
    ("SpiralPitchFactor", "pitch", ""),
    ("RevolutionTime", "rotation time", "s"),
    ("SingleCollimationWidth", "single collimation", "mm"),
    ("TotalCollimationWidth", "total collimation", "mm"),
    ("TableFeedPerRotation", "table feed per rotation", "mm"),
    ("SliceThickness", "slice thickness", "mm"),
    ("SpacingBetweenSlices", "slice increment", "mm"),
    ("ReconstructionDiameter", "reconstruction FOV", "mm"),
    ("ConvolutionKernel", "kernel", ""),
    ("ExposureModulationType", "dose modulation", ""),
    ("FilterType", "filter", ""),
]

# How many files to sample per series. Enough to catch modulation, cheap enough to run
# over an export of several thousand slices.
SAMPLE = 25


def _fmt(values, unit):
    """One value, or a range when the sampled files disagree."""
    vals = [v for v in values if v not in (None, "")]
    if not vals:
        return None
    uniq = sorted({str(v) for v in vals})
    if len(uniq) == 1:
        return f"{uniq[0]}{' ' + unit if unit else ''}"
    try:
        lo, hi = min(float(v) for v in vals), max(float(v) for v in vals)
        return f"{lo:g} to {hi:g}{' ' + unit if unit else ''}  ({len(uniq)} distinct)"
    except ValueError:
        return " / ".join(uniq)


def probe_dicom(folder: Path) -> dict:
    """Technique factors per DICOM series under `folder`."""
    import pydicom

    series: dict[str, dict] = {}
    rdsr = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
        except Exception:                             # noqa: BLE001  not a DICOM file
            continue
        sop = str(getattr(ds, "SOPClassUID", ""))
        if sop == "1.2.840.10008.5.1.4.1.1.88.67":    # X-Ray Radiation Dose SR
            rdsr.append(p)
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "?"))
        s = series.setdefault(uid, {
            "description": str(getattr(ds, "SeriesDescription", "")),
            "number": getattr(ds, "SeriesNumber", None),
            "n_files": 0,
            "z": [],
            "_vals": {k: [] for k, _l, _u in TAGS},
        })
        s["n_files"] += 1
        if s["n_files"] > SAMPLE:
            continue
        for key, _label, _unit in TAGS:
            s["_vals"][key].append(getattr(ds, key, None))
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            s["z"].append(float(ipp[2]))

    out = {"folder": str(folder), "rdsr": [str(p) for p in rdsr], "series": []}
    for uid, s in sorted(series.items(), key=lambda kv: (kv[1]["number"] or 0)):
        rec = {"series_uid": uid, "series_number": s["number"],
               "description": s["description"], "files": s["n_files"]}
        for key, label, unit in TAGS:
            v = _fmt(s["_vals"][key], unit)
            if v is not None:
                rec[label] = v
        if len(s["z"]) > 1:
            rec["reconstructed z-extent"] = f"{max(s['z']) - min(s['z']):.1f} mm (sampled)"
        out["series"].append(rec)
    return out


def probe_descriptor(path: Path) -> dict:
    """Scan parameters from the raw .mat descriptor -- the source behind `own`."""
    from scipy.io import loadmat

    d = loadmat(str(path), struct_as_record=True, squeeze_me=False)
    desc = d["descriptor"].flat[0]
    want = ("kv", "current", "voltage", "mas", "exposure", "pitch", "rotation",
            "collimat", "slice", "feed", "focal", "filter", "dose", "ctdi", "time")
    found: dict[str, str] = {}

    def walk(node, prefix=""):
        names = getattr(getattr(node, "dtype", None), "names", None)
        if not names:
            return
        for name in names:
            try:
                child = node[name].flat[0]
            except Exception:                          # noqa: BLE001
                continue
            path_ = f"{prefix}{name}"
            if getattr(getattr(child, "dtype", None), "names", None):
                walk(child, path_ + ".")
                continue
            if any(w in name.lower() for w in want):
                try:
                    val = child.item() if getattr(child, "size", 0) == 1 else child.ravel()[:6]
                except Exception:                      # noqa: BLE001
                    continue
                found[path_] = str(val)

    walk(desc)
    return {"descriptor": str(path), "fields": found}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--dicom", nargs="*", default=[], metavar="NAME=DIR",
                   help="one or more labelled DICOM folders, e.g. wfbp=/path")
    p.add_argument("--desc", help="raw .mat descriptor of our own acquisition")
    p.add_argument("--json", help="also write everything to this file")
    a = p.parse_args()

    report: dict = {}
    for item in a.dicom:
        name, _, folder = item.partition("=")
        if not folder:
            p.error(f"expected NAME=DIR, got {item!r}")
        print(f"\n{'=' * 78}\n{name}   {folder}\n{'=' * 78}")
        rep = probe_dicom(Path(folder))
        report[name] = rep
        if rep["rdsr"]:
            print(f"  radiation-dose structured report present: {len(rep['rdsr'])} file(s)")
            print("  read the reported DLP from it; the per-series values below are the "
                  "reconstruction's")
        for s in rep["series"]:
            print(f"\n  [{s['series_number']}] {s['description']}   ({s['files']} files)")
            for k, v in s.items():
                if k in ("series_uid", "series_number", "description", "files"):
                    continue
                print(f"      {k:26s} {v}")

    if a.desc:
        print(f"\n{'=' * 78}\nown (raw descriptor)   {a.desc}\n{'=' * 78}")
        rep = probe_descriptor(Path(a.desc))
        report["own"] = rep
        for k, v in sorted(rep["fields"].items()):
            print(f"  {k:44s} {v}")
        if not rep["fields"]:
            print("  no matching field found -- inspect with mat_structure.py")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
