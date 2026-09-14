#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=ProbeOrigin
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# Do our reconstructed volumes and the Siemens export share an in-plane frame?
#
# CPU only and seconds to run -- no --gres line, so it does not hold a GPU.
# If the partition refuses a job without one, add back:
#     #SBATCH --gres=gpu:rtx6000ada:1
#
#   sbatch batch_probe_origin.sh
#
# Paths and the segmentation are taken from match_config.json (written by
# stage 0), so nothing needs to be edited here after a comparison run.
# Override with --wfbp-dir / --vmi-dir / --segmentation if needed; any extra
# argument is forwarded.
# ============================================================================

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda/envs/MatDecomp
cd "$SLURM_SUBMIT_DIR"

python -m reconstruction.probe_dicom_origin \
    --out-root output/research/recon_comparison \
    "$@"
