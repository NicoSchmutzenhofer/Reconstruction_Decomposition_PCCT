"""
decompose.py -- production driver: one clinical mode over one input source.

Builds a DecompConfig from the command line (the constants below are the defaults), calls
decompose() and writes the material maps, their metadata with the per-material reliability
report, and the config that produced them (decomp_config.json) to
output/decomposition/<source>/<mode>/.  All sources share one pipeline; only the loader
(load_energy_stack) differs, and the channel count is read from the data:

    own   our reconstruction, output/reconstruction/reconstruction_thr_*_HU.nii.gz
    wfbp  Siemens WFBP threshold DICOMs
    vmi   Siemens monoenergetic (VMI) DICOMs, keV series found automatically

    python -m decomposition.decompose --source own
    python -m decomposition.decompose --source wfbp --wfbp-dir /path/to/wfbp
    python -m decomposition.decompose --source vmi  --vmi-dir  /path/to/vmi
    python -m decomposition.decompose --list-series /path/to/export   # what is where
    python -m decomposition.decompose --list-modes
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# --- import shim: work both as `-m decomposition.decompose` and as a script path ---
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decomposition.material_decomposition import (
    DecompConfig, decompose, load_energy_stack, save_decomp_result)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ============================== CONFIG ==============================
INPUT_SOURCE = os.environ.get("DECOMP_SOURCE", "own").lower()   # 'own'|'wfbp'|'vmi'; batch.sh sets it from $1
MODE = "phantom_ca_i"         # see decomposition_modes.list_modes()
THRESHOLD_OPTION = 1          # threshold windows for 'own'/'wfbp' (VMI ignores this)

# --- estimator: start simple (OLS = rawest baseline); advance later ---
ESTIMATOR = "ols"             # 'ols' -> 'wls' -> 'wls_denoise' -> 'wls_joint'
NOISE_MODEL = "global"        # ignored by OLS; used by wls*/joint
BIN_DOMAIN = "cumulative"     # thresholds: 'cumulative' (no subtraction) | 'exclusive'; VMI auto -> 'direct'
WATER_CALIBRATION = False     # baseline: physical per-channel mu_water, no per-bin gains
SIGNATURE_WEIGHTING = "uniform"   # 'uniform' | 'spectrum'; cumulative windows only
SPECTRUM_KVP = 140.0              # bremsstrahlung endpoint for 'spectrum'
SPECTRUM_WATER_CM = 16.0          # phantom path used as the filtration for 'spectrum'

# --- input locations -------------------------------------------------------------
OWN_DIR = str(_REPO_ROOT / "output" / "reconstruction")     # our NIfTI recon (INPUT_SOURCE='own')
# Siemens DICOM export folders.  WFBP and VMI are SEPARATE arguments because a Siemens
# export puts each reconstruction in its own folder -- one path cannot serve both, and a
# parent folder holding several sets would concatenate their channels into one stack.
# Point each at the folder holding exactly ONE reconstruction; use --list-series to check.
WFBP_DIR = ""                 # e.g. "/path/to/wfbp_export"
VMI_DIR = ""                  # e.g. "/path/to/vmi_export"
Z_SLAB_MM = None              # (z_lo, z_hi) mm to restrict to a slab; None = full volume

OUTPUT_ROOT = str(_REPO_ROOT / "output" / "decomposition")  # <source>/<mode>/ added below
# ====================================================================
#
# Every constant above is only a DEFAULT: the CLI below overrides each of them, so a run
# never requires editing this file.  `python -m decomposition.decompose --help` is the
# authoritative list, and each run records the exact configuration that produced it
# (decomp_config.json) -- which is what the thesis promises in sec:bg-software.


def _print_stability(result) -> None:
    s = result.stability
    mats = s["materials"]
    cos = s["column_cosines"]
    print("\n" + "=" * 66)
    print(f"STABILITY   mode '{result.config.mode}'   [{'/'.join(mats)}]")
    print("=" * 66)
    print(f"  condition number  kappa(M)     = {s['condition_number']:.1f}   -> {s['verdict'].upper()}")
    print(f"  kappa(M^T M) = kappa^2         = {s['condition_number_MtM']:.3g}")
    print(f"  column-normalised kappa        = {s['condition_number_colnorm']:.1f}")
    print(f"  singular values                = {['%.4g' % v for v in s['singular_values']]}")
    print("  per-material noise amplification  sqrt(diag((M^T M)^-1)):")
    for m, g in s["noise_amplification"].items():
        print(f"      {m:16s} {g:12.3f}")
    print("  column cosines (1.0 = collinear = singular):")
    print("           " + "".join(f"{m[:8]:>10s}" for m in mats))
    for i, m in enumerate(mats):
        print(f"    {m[:8]:>8s} " + "".join(f"{cos[i][j]:10.4f}" for j in range(len(mats))))
    print("=" * 66)


def build_parser() -> argparse.ArgumentParser:
    from decomposition.decomposition_modes import list_modes

    p = argparse.ArgumentParser(
        prog="python -m decomposition.decompose",
        description="Image-domain material decomposition over our reconstruction, "
                    "Siemens WFBP thresholds, or Siemens VMIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", default=INPUT_SOURCE, choices=["own", "wfbp", "vmi"],
                   help="input family (env DECOMP_SOURCE sets the default)")
    p.add_argument("--mode", default=MODE,
                   help=f"clinical mode -> basis materials. Available: {', '.join(list_modes())}")
    p.add_argument("--own-dir", default=OWN_DIR,
                   help="folder of our reconstruction_thr_*_HU.nii.gz")
    p.add_argument("--wfbp-dir", default=WFBP_DIR or None,
                   help="folder holding ONE Siemens WFBP threshold reconstruction")
    p.add_argument("--vmi-dir", default=VMI_DIR or None,
                   help="folder holding ONE Siemens monoenergetic (VMI) reconstruction")
    p.add_argument("--estimator", default=ESTIMATOR,
                   choices=["ols", "wls", "wls_denoise", "wls_joint"])
    p.add_argument("--noise-model", default=NOISE_MODEL, choices=["global", "spatial"])
    p.add_argument("--bin-domain", default=BIN_DOMAIN,
                   choices=["cumulative", "exclusive"],
                   help="thresholds only; VMI is forced to 'direct'")
    p.add_argument("--threshold-option", type=int, default=THRESHOLD_OPTION,
                   help="threshold window set for own/wfbp (VMI ignores it)")
    p.add_argument("--water-calibration", dest="water_calibration", action="store_true",
                   default=WATER_CALIBRATION)
    p.add_argument("--no-water-calibration", dest="water_calibration",
                   action="store_false")
    # Mirrors decomp_validation so the two entry points cannot disagree about how M was
    # integrated -- the same trap water_calibration fell into. Only reaches the cumulative
    # windows; the published disjoint rows and the monoenergetic values are untouched.
    p.add_argument("--signature-weighting", default=SIGNATURE_WEIGHTING,
                   choices=["uniform", "spectrum"],
                   help="weighting of the CUMULATIVE window averages of M. 'uniform' is "
                        "the published table's own weighting and the default")
    p.add_argument("--spectrum-kvp", type=float, default=SPECTRUM_KVP)
    p.add_argument("--spectrum-water-cm", type=float, default=SPECTRUM_WATER_CM)
    p.add_argument("--z-slab", metavar="LO,HI",
                   help="restrict to an axial slab in mm. Write it as --z-slab=-1515,-1408 "
                        "-- a value starting with '-' is otherwise parsed as a flag")
    p.add_argument("--output-root", default=OUTPUT_ROOT,
                   help="outputs go to <output-root>/<source>/<mode>/")
    p.add_argument("--config", metavar="FILE",
                   help="load a previously written decomp_config.json; explicit flags "
                        "still win over it")
    p.add_argument("--list-modes", action="store_true",
                   help="print the available clinical modes and exit")
    p.add_argument("--list-series", metavar="DIR",
                   help="print every DICOM series under DIR with its classification and "
                        "exit -- use it to find the right --wfbp-dir/--vmi-dir")
    return p


def _list_series(folder: str) -> None:
    from decomposition.material_decomposition import build_dicom_index, _classify_series

    index = build_dicom_index(folder)
    print(f"\n{len(index)} series under {folder}\n")
    print(f"{'#':>5s} {'files':>6s} {'class':<14s} description")
    print("-" * 78)
    for _uid, s in sorted(index.items(), key=lambda kv: str(kv[1].get("number"))):
        spec = _classify_series(s.get("desc", ""),
                                s["files"][0][1] if s["files"] else "")
        cls = f"{spec[0]}:{spec[1]:g}" if spec else "-"
        print(f"{str(s['number']):>5s} {len(s['files']):>6d} {cls:<14s} {s.get('desc', '')}")
    print("\nPass the folder holding exactly ONE reconstruction: several sets under one "
          "parent\nproduce duplicate channel labels and are rejected.")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.list_modes:
        from decomposition.decomposition_modes import MODES
        for key, spec in MODES.items():
            print(f"  {key:22s} {spec.display_name:32s} {'/'.join(spec.materials)}")
        return
    if args.list_series:
        _list_series(args.list_series)
        return

    overrides = {}
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))

    src = args.source.lower()
    dicom_dir = args.wfbp_dir if src == "wfbp" else args.vmi_dir
    if src in ("wfbp", "vmi") and not dicom_dir:
        raise SystemExit(
            f"--source {src} needs --{src}-dir (the folder holding that one Siemens "
            f"reconstruction).\nFind it with:\n"
            f"  python -m decomposition.decompose --list-series /path/to/export")

    z_slab = (tuple(float(v) for v in args.z_slab.split(",")) if args.z_slab
              else Z_SLAB_MM)

    cfg = DecompConfig(
        mode=args.mode, threshold_option=args.threshold_option,
        bin_domain=args.bin_domain, estimator=args.estimator,
        noise_model=args.noise_model, water_calibration=args.water_calibration,
        signature_weighting=args.signature_weighting, spectrum_kvp=args.spectrum_kvp,
        spectrum_water_cm=args.spectrum_water_cm,
        input_source=src,
        input_format=("nifti" if src == "own" else "dicom"),
        input_dir=args.own_dir, dicom_dir=dicom_dir, z_slab_mm=z_slab,
        # <source>/<mode>/ so own/wfbp/vmi and different clinical questions never
        # overwrite one another -- the comparison tables need them side by side.
        output_dir=str(Path(args.output_root) / src / args.mode),
    )
    if overrides:                       # --config first, explicit flags already applied
        merged = {**overrides, **{k: v for k, v in cfg.to_dict().items()
                                  if v is not None}}
        cfg = DecompConfig.from_dict(merged)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, "decomp_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")

    print(f"=== material decomposition   source='{src}'   mode='{cfg.mode}'   estimator='{cfg.estimator}' ===")
    if dicom_dir:
        print(f"  input: {dicom_dir}")
    volumes, channels, ref = load_energy_stack(cfg)
    print(f"  loaded {len(channels)} channels {[c.label for c in channels]}  volumes {tuple(volumes.shape)}")

    def prog(frac, msg):
        print(f"  [{frac * 100:5.1f}%] {msg}")

    result = decompose(volumes, cfg, channels=channels, progress=prog)
    _print_stability(result)

    written = save_decomp_result(result, ref, cfg.output_dir)
    print(f"\nWrote {len(written)} files to {cfg.output_dir}:")
    for p in written:
        print("   ", p.name)


if __name__ == "__main__":
    main()
