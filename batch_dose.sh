#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=ProbeDose
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# Acquisition and dose parameters of every scan (reconstruction/probe_dose.py).
#
# Reads the technique factors from the vendor DICOM exports and from the raw
# descriptor behind our own reconstruction.  own, wfbp and vmi come from one
# acquisition; the 2-bin product is a separate acquisition with its own dose.
#
#   sbatch batch_dose.sh
#
# CPU only, seconds, read-only.  A CTDIvol that varies within a series means the
# folder mixes acquisitions.  DLP is not in image headers: read it from the dose
# report if the export has one, otherwise quote CTDIvol x irradiated length --
# the printed z-extent is only a lower bound on that length.
# ============================================================================

# --- the same three folders batch_decomp_validation.sh uses -----------------
WFBP_DIR="/path/to/wfbp_export/"
VMI_DIR="/path/to/vmi_export/"
S2BIN_DIR="/path/to/2bin_export/"

# --- the descriptor python_reconstruction.py reads --------------------------
DESC="/path/to/descriptor.raw.mat"

OUT="output/research/dose/scan_parameters.json"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda/envs/MatDecomp
cd "$SLURM_SUBMIT_DIR"

mkdir -p "$(dirname "$OUT")"

python -m reconstruction.probe_dose \
    --dicom "wfbp=$WFBP_DIR" "vmi=$VMI_DIR" "s2bin=$S2BIN_DIR" \
    --desc "$DESC" \
    --json "$OUT"

echo
echo "----------------------------------------------------------------------"
echo "wrote $OUT"
echo "own, wfbp and vmi come from one acquisition and share its dose;"
echo "the two-bin product is a separate acquisition with its own."
