#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=Test2Bin
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# THROWAWAY -- delete this file once the 2-bin import is confirmed working.
#
# Checks the RQ1 (4-bin vs 2-bin) additions on the cluster, where the data is.
# Nothing here touches output/research/decomp_validation: step 3 writes to its
# own scratch root, so a broken run cannot corrupt real results.
#
#   sbatch batch_test_2bin.sh
#
# CPU only -- no --gres, so it does not hold a GPU. If the partition refuses a
# job without one, add:  #SBATCH --gres=gpu:rtx6000ada:1
#
# Steps:
#   1  self-tests            no data needed; must be green before anything else
#   2  probe the 2-bin export   <- THE step that matters
#   3  end-to-end import into a scratch out-root
#
# Step 2 answers the one question that can stop this: was the iodine map
# exported as a quantitative MONOCHROME2 image, or as an RGB colour blend? An
# RGB series is a picture of a map, not a map, and carries no densities. Read
# its output before trusting anything downstream.
# ============================================================================

# --- FILL THIS IN: the export holding the Lung_Evaluation_PBV and VNC folders ---
S2BIN_DIR=""

# The 4-bin reference, needed by step 3 only (the segmentation was drawn on it).
WFBP_DIR="/path/to/wfbp_export/"
VMI_DIR="/path/to/vmi_export/"

SCRATCH="output/research/_test_2bin"     # delete this folder with the script

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda/envs/MatDecomp
cd "$SLURM_SUBMIT_DIR"

set -e

echo "=== 1/3  self-tests (no data, no GPU) ==="
python -m decomposition.selftest_decomp_validation
python -m decomposition.selftest_decomposition
python -m reconstruction.selftest_image_quality

if [ -z "$S2BIN_DIR" ]; then
    echo
    echo "S2BIN_DIR is empty -- self-tests done, stopping here."
    echo "Set it at the top of this file to the folder holding the"
    echo "Lung_Evaluation_PBV and VNC folders, then re-submit."
    exit 0
fi

echo
echo "=== 2/3  what is actually in the 2-bin export ==="
echo "    $S2BIN_DIR"
echo
echo "--- every series, with geometry ---"
python -m reconstruction.recon_comparison --list-series "$S2BIN_DIR"
echo
echo "--- photometric interpretation, rescale, value range ---"
echo "    CHECK: the PBV series must be MONOCHROME2 with 1 sample/pixel."
echo "    If it is RGB, it cannot be scored and a numeric re-export is needed."
python -m decomposition.decomp_validation --probe-siemens "$S2BIN_DIR" \
    --out-root "$SCRATCH"

echo
echo "=== 3/3  end-to-end: import the vendor maps into a scratch root ==="
echo "    -> $SCRATCH  (safe to delete; the real run uses a different root)"
echo
echo "--- stage 0: ROIs from the segmentation ---"
python -m decomposition.decomp_validation --stage 0 \
    --out-root "$SCRATCH" \
    --wfbp-dir "$WFBP_DIR" \
    --vmi-dir  "$VMI_DIR"

echo
echo "--- stage 1: the vendor import ONLY (no solves, seconds) ---"
python -m decomposition.decomp_validation --stage 1 \
    --out-root "$SCRATCH" \
    --wfbp-dir "$WFBP_DIR" \
    --vmi-dir  "$VMI_DIR" \
    --sources s2bin \
    --s2bin-dir "$S2BIN_DIR"

echo
echo "--- stage 2: measure it (the 4-vs-2 figure needs 4-bin runs too, so it"
echo "    is expected to be SKIPPED here -- this step proves the import, the"
echo "    ROI resample and the QC path, not the comparison) ---"
python -m decomposition.decomp_validation --stage 2 \
    --out-root "$SCRATCH" \
    --wfbp-dir "$WFBP_DIR" \
    --vmi-dir  "$VMI_DIR" \
    --sources s2bin \
    --s2bin-dir "$S2BIN_DIR"

echo
echo "=== done ==="
echo "  Look at, in this order:"
echo "    1. the probe table above -- MONOCHROME2, and a sane value range?"
echo "    2. $SCRATCH/qc/qc_roi_placement.png"
echo "       the s2bin row is the one that matters: it is a SEPARATE acquisition,"
echo "       so the ROIs landing on the inserts there is an assumption being tested"
echo "    3. the summary table -- the vendor run should read near zero iodine on the"
echo "       BASE lesions and clearly positive on the high ones. If not, the series"
echo "       selection or the rescale is wrong, not the decomposition."
echo
echo "  Then set S2BIN_DIR in batch_decomp_validation.sh and delete:"
echo "    rm -rf $SCRATCH batch_test_2bin.sh"
