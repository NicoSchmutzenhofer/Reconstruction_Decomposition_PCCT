#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=DecompValid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# Validate the material decomposition against the QRM DEP-V5 phantom
# (decomposition/decomp_validation.py: self-test, then stages 0-2).
#
# Stage 1 skips any source/mode/estimator that already ran with a matching
# configuration, so a re-run for the metrics only costs nothing.
#
#   sbatch batch_decomp_validation.sh                 # everything
#   sbatch batch_decomp_validation.sh --stage 2       # metrics + figures only
#   sbatch batch_decomp_validation.sh --estimators ols,wls    # skip the slow two
#   sbatch batch_decomp_validation.sh --force         # redo the solves
#
# CPU only -- no --gres line.  If the partition refuses a job without one, add:
#     #SBATCH --gres=gpu:rtx6000ada:1
#
# The segmentation is found as $WFBP_DIR/Segmentation_decomposition.nrrd; its
# label contract is in the docstring of decomp_validation.py.
# ============================================================================

WFBP_DIR="/path/to/wfbp_export/"
VMI_DIR="/path/to/vmi_export/"
OWN_DIR="output/reconstruction"

# --- RQ1: the scanner's own 2-BIN decomposition -----------------------------
# The export holding the Lung_Evaluation_PBV and VNC folders.  PBV is the IODINE
# map; VNC is the complementary base image (the scan with iodine removed) and is
# used for QC only.  Both are IMPORTED, never solved.
#
# RUN THE PROBE FIRST and set the match strings from what it prints -- an RGB
# colour rendering is a picture of a map, not a map, and cannot be scored:
#
#   python -m decomposition.decomp_validation --probe-siemens "$S2BIN_DIR"
#
# Leave empty to skip RQ1 and run the modality comparison alone.
S2BIN_DIR="/path/to/2bin_export/"
S2BIN_IODINE_MATCH="pbv"
S2BIN_VNC_MATCH="vnc"

OUT_ROOT="output/research/decomp_validation"

# Basis whose base column is the phantom body material (CTWater).  The second
# basis is run alongside purely so the difference MEASURES the base-material
# sensitivity instead of leaving it as a caveat.  Set SENSITIVITY_MODE="" to skip.
MODE="phantom_water_ca_i"
SENSITIVITY_MODE="phantom_ca_i"

# Each source is solved from its own measurement stack -- they do not hold the
# same quantity, so one setting cannot serve all three:
#   own   cumulative  our thresholds count every photon ABOVE the threshold
#   wfbp  exclusive   separate bins, by image-domain differencing
#   vmi   direct      monoenergetic keV; nothing nested to subtract
# That is the default and needs no flag. To override:
#   sbatch batch_decomp_validation.sh --bin-domain-map wfbp=cumulative
#   sbatch batch_decomp_validation.sh --bin-domain exclusive     # all sources

# ols and wls are fast; wls_denoise / wls_joint can fail to converge, so each run
# is capped and anything past the cap is REPORTED as truncated rather than
# returned as if it had converged.
ESTIMATORS="ols,wls,wls_denoise,wls_joint"
TIMEOUT=1800

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda/envs/MatDecomp
cd "$SLURM_SUBMIT_DIR"

# The vendor import is one extra source; without a folder it is simply absent and
# the run degrades to the modality comparison rather than failing.
SOURCES="own,wfbp,vmi"
# An ARRAY, not a string: the Siemens export paths contain spaces ("Thx.- Abdomen
# Staging_Standard - PNR_..."), and an unquoted string expansion word-splits on them, so
# argparse saw five arguments where one was meant. An array keeps each element intact.
S2BIN_ARGS=()
if [ -n "$S2BIN_DIR" ]; then
    SOURCES="$SOURCES,s2bin"
    S2BIN_ARGS=(--s2bin-dir "$S2BIN_DIR"
                --s2bin-iodine-match "$S2BIN_IODINE_MATCH"
                --s2bin-vnc-match "$S2BIN_VNC_MATCH")
fi

set -e   # a bad stage 0 makes stages 1-2 meaningless

echo "=== 0/4  metric self-test (no data needed) ==="
python -m decomposition.selftest_decomp_validation

echo
echo "=== 1/4  ROIs, layer and role assignment (CPU) ==="
python -m decomposition.decomp_validation --stage 0 \
    --out-root "$OUT_ROOT" \
    --own-dir  "$OWN_DIR" \
    --wfbp-dir "$WFBP_DIR" \
    --vmi-dir  "$VMI_DIR" \
    --sources "$SOURCES" "${S2BIN_ARGS[@]}" \
    "$@"

echo
echo "=== 2/4  decompose: sources x bases x estimators (CPU) ==="
echo "    sources: $SOURCES"
echo "    runs with a matching config are skipped"
python -m decomposition.decomp_validation --stage 1 \
    --out-root "$OUT_ROOT" \
    --own-dir  "$OWN_DIR" \
    --wfbp-dir "$WFBP_DIR" \
    --vmi-dir  "$VMI_DIR" \
    --sources "$SOURCES" "${S2BIN_ARGS[@]}" \
    --mode "$MODE" --sensitivity-mode "$SENSITIVITY_MODE" \
    --estimators "$ESTIMATORS" --timeout "$TIMEOUT" \
    "$@"

echo
echo "=== 3/4  metrics + figures (CPU) ==="
python -m decomposition.decomp_validation --stage 2 \
    --out-root "$OUT_ROOT" \
    --own-dir  "$OWN_DIR" \
    --wfbp-dir "$WFBP_DIR" \
    --vmi-dir  "$VMI_DIR" \
    --sources "$SOURCES" "${S2BIN_ARGS[@]}" \
    "$@"

echo
echo "=== done ==="
echo "  QC (check this FIRST):"
echo "    $OUT_ROOT/qc/qc_roi_placement.png    circles must sit on the inserts in EVERY row"
echo "                                        the s2bin row matters most: separate acquisition"
echo "  RQ1  4 bins vs 2 bins (iodine only, both sides):"
echo "    $OUT_ROOT/figures/bins_4_vs_2.png        <- the headline; 4-bin envelope vs 2-bin point"
echo "    $OUT_ROOT/figures/rq1_iodine_maps.png    the same claim as an image, both normalised"
echo "    $OUT_ROOT/figures/rq1_capability.png     4 bins give a CaHA map; 2 bins cannot"
echo "  RQ2  does the modality matter:"
echo "    $OUT_ROOT/figures/source_comparison.png"
echo "    $OUT_ROOT/figures/estimator_tradeoff.png  bias / noise / runtime per estimator"
echo "    $OUT_ROOT/figures/separation_best.png  linearity_best.png  mode_comparison.png"
echo "  The maps themselves:"
echo "    $OUT_ROOT/figures/material_maps.png      source x material, one window per material"
echo "    $OUT_ROOT/figures/material_space.png     insert voxels in the (iodine, CaHA) plane"
echo "    $OUT_ROOT/figures/contact_sheet_*.png    the 0 : 1 : 3.1 levels as inserts"
echo "    $OUT_ROOT/figures/roi_placement.png      the QC grid, cropped for the thesis"
echo "  For LaTeX:"
echo "    $OUT_ROOT/figures/panels/                one file per panel (--no-split-figures to skip)"
echo "    $OUT_ROOT/figures/CAPTIONS.md            draft captions + a stitching snippet"
echo "    $OUT_ROOT/metrics/decomp_validation.json"
