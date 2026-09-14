#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=DecompSens
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# Two sensitivity runs against the sign inversion of the material maps.
#
#   A  water calibration OFF (--no-water-calibration), on own and wfbp.  It is
#      ON in the reported runs; wfbp, whose water HU is already accurate, is
#      the control.
#   B  spectrum-weighted signatures (--signature-weighting spectrum), on own
#      only -- the one source solved on windows the code integrates itself.
#
# Paths and settings are read from batch_decomp_validation.sh and its stage-0
# files are reused.  Each run writes to its own out-root, so the reported
# results are never overwritten.  CPU only, minutes per run.
#
#   sbatch batch_decomp_sensitivity.sh              # both runs
#   sbatch batch_decomp_sensitivity.sh --force      # ignore cached runs
#
# Afterwards, compare the `sign ok` and `orient` columns with the reported run:
# the question is whether the inversion moves, not whether the numbers improve.
# ============================================================================

cd "$SLURM_SUBMIT_DIR"

# Configuration is READ from the main script rather than repeated here, so the
# two cannot drift apart when a data path is edited.  Only plain NAME="value"
# assignments are picked up; anything else in that file is ignored.
MAIN="batch_decomp_validation.sh"
if [ ! -f "$MAIN" ]; then
    echo "ERROR: $MAIN not found -- run this from the repository root." >&2
    exit 1
fi
eval "$(grep -E '^(OWN_DIR|WFBP_DIR|VMI_DIR|OUT_ROOT|MODE|SENSITIVITY_MODE|ESTIMATORS|TIMEOUT)=' "$MAIN")"
if [ -z "$OUT_ROOT" ] || [ -z "$OWN_DIR" ] || [ -z "$WFBP_DIR" ]; then
    echo "ERROR: could not read the paths from $MAIN." >&2
    exit 1
fi

BASE_ROOT="$OUT_ROOT"
ROOT_NOWATER="${BASE_ROOT}_sens_nowater"
ROOT_SPECTRUM="${BASE_ROOT}_sens_spectrum"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda/envs/MatDecomp

set -e

# Stage 0 writes these three, and none of them depends on the calibration or the
# weighting -- they are segmentation geometry on one reference grid, and stage 2
# loads all three together.  Copying them in is what keeps these runs at minutes
# instead of re-reading every DICOM series.  Keep the list complete: seeding a
# subset produces a root that passes the guard and then dies inside stage 2 on
# the first file that was left behind.
STAGE0_FILES="decomp_match.json roi_labels.nrrd _masks.npy"

seed_root () {
    local dst="$1"
    local f missing=""
    mkdir -p "$dst"
    for f in $STAGE0_FILES; do
        [ -f "$BASE_ROOT/$f" ] || missing="$missing $f"
    done
    if [ -z "$missing" ]; then
        for f in $STAGE0_FILES; do
            cp -n "$BASE_ROOT/$f" "$dst/"
        done
        echo "    seeded from $BASE_ROOT (stage 0 skipped)"
    else
        echo "    $BASE_ROOT is missing:$missing -- running stage 0"
        python -m decomposition.decomp_validation --stage 0 \
            --out-root "$dst" --own-dir "$OWN_DIR" \
            --wfbp-dir "$WFBP_DIR" --vmi-dir "$VMI_DIR" --sources own,wfbp
    fi
}

run_pair () {                       # <out-root> <sources> <extra flags...>
    local root="$1"; shift
    local sources="$1"; shift
    for stage in 1 2; do
        python -m decomposition.decomp_validation --stage "$stage" \
            --out-root "$root" --own-dir "$OWN_DIR" \
            --wfbp-dir "$WFBP_DIR" --vmi-dir "$VMI_DIR" \
            --sources "$sources" \
            --mode "$MODE" --sensitivity-mode "$SENSITIVITY_MODE" \
            --estimators "$ESTIMATORS" --timeout "$TIMEOUT" \
            --all-figures "$@"
    done
}

echo "=== 0/3  metric self-test (no data needed) ==="
python -m decomposition.selftest_decomp_validation

echo
echo "=== 1/3  A: water calibration OFF  (own + wfbp) ==="
echo "    -> $ROOT_NOWATER"
seed_root "$ROOT_NOWATER"
run_pair "$ROOT_NOWATER" "own,wfbp" --no-water-calibration "$@"

echo
echo "=== 2/3  B: spectrum-weighted signatures  (own only) ==="
echo "    -> $ROOT_SPECTRUM"
echo "    wfbp and vmi are excluded because no weighting can move them"
seed_root "$ROOT_SPECTRUM"
run_pair "$ROOT_SPECTRUM" "own" --signature-weighting spectrum "$@"

echo
echo "=== 3/3  done ==="
echo "  Compare the SUMMARY TABLE of each against the reported run:"
echo "    $BASE_ROOT/metrics/decomp_validation.json      reported"
echo "    $ROOT_NOWATER/metrics/decomp_validation.json   water calibration off"
echo "    $ROOT_SPECTRUM/metrics/decomp_validation.json  spectrum-weighted M"
echo "  The columns that answer the question are 'sign ok' and 'orient', and the"
echo "  per-run linearity figures, which show whether a map still falls as its"
echo "  own agent rises:"
echo "    $ROOT_NOWATER/figures/linearity_own__*.png"
echo "    $ROOT_SPECTRUM/figures/linearity_own__*.png"
echo "  Expect the spectrum run to be NOISIER: at [20,140] keV the weighting moves"
echo "  HA by -36 % and iodine by +7 %, and the condition number of the cumulative"
echo "  solve rises from 76.6 to 657.4. Noise and orientation are separate"
echo "  questions, which is the point of running it."
