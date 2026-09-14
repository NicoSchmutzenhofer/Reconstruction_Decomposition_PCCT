#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=Reconstruction
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# (Re)build our own per-threshold volumes -- the SSR helical reconstruction.
#
#   sbatch batch_reconstruction.sh
#
# This is the ONLY stage that needs the GPU (ASTRA FBP_CUDA / SIRT_CUDA).
# Everything downstream (decomposition, both comparisons) reads what this
# writes, so run it first whenever output/reconstruction/ is stale.
#
# Writes to output/reconstruction/ :
#     reconstruction_thr_{A,B,C,D}.nii.gz        raw attenuation
#     reconstruction_thr_{A,B,C,D}_HU.nii.gz     <- the contract with decomposition
#     calibration_thr_{A,B,C,D}.json             cached mu_water / mu_air
#     reconstruction_4thr_multienergy.nii.gz     4D stack
#     reconstruction_4thr_summary.png
#     invariant_log.json                         <- read this after every run
#
# All settings live at the top of reconstruction/python_reconstruction.py, not
# here -- the driver takes no arguments.  The values in force are echoed below
# so the log records what actually ran.
#
# Runtime: FAST_MODE=False with RECON_METHOD='sirt' and N_ITER=100 is HOURS
# (iterative is ~2*N_ITER x slower per slice than FBP).  Set RECON_METHOD='fbp'
# in the driver for a quick pass.
# ============================================================================

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda/envs/MatDecomp
cd "$SLURM_SUBMIT_DIR"

DRIVER="reconstruction/python_reconstruction.py"
OUT_DIR="output/reconstruction"

set -e

echo "=== reconstruction settings in force ==="
grep -E "^(FAST_MODE|N_PIXELS|FOV_MM|RECON_METHOD|N_ITER|FILTER_NAME|GEOMETRY_MODEL|Z_WEIGHTING|Z_SMOOTH_MM|ENABLE_HU_CALIBRATION|OUTPUT_MIRROR_X|TABLE_HEIGHT_MM|PATIENT_POSITION|MAR_STRENGTH)\s*=" "$DRIVER"

# A preview writes previews, not volumes -- submitting one to a GPU queue by
# accident costs a slot and produces nothing downstream can read.
if grep -qE "^FAST_MODE\s*=\s*True" "$DRIVER"; then
    echo
    echo "ABORT: FAST_MODE = True in $DRIVER."
    echo "       That is the slab preview -- it writes no NIfTI volumes."
    echo "       Set FAST_MODE = False for a full reconstruction."
    exit 1
fi

# The HU calibration is cached per threshold and reused across runs.  That is
# what you want while tuning geometry or the output frame; it is NOT what you
# want after changing preprocessing, filtering or the reconstruction method,
# since mu_water then really has moved.  Pass --recalibrate to drop the cache.
if [ "$1" = "--recalibrate" ]; then
    echo
    echo "dropping cached calibration (calibration_thr_*.json)"
    rm -f "$OUT_DIR"/calibration_thr_*.json
fi

echo
echo "=== full helical reconstruction, 4 thresholds (GPU) ==="
python "$DRIVER"

echo
echo "=== done ==="
echo "  Check FIRST:"
echo "    $OUT_DIR/invariant_log.json               soft-warned invariants"
echo "    $OUT_DIR/reconstruction_4thr_summary.png  all four thresholds"
echo "  Then, downstream:"
echo "    sbatch batch_compare.sh                   recon vs Siemens WFBP/VMI"
echo "    sbatch batch.sh own                       material decomposition"
echo "    sbatch batch_decomp_validation.sh         decomposition vs phantom"
