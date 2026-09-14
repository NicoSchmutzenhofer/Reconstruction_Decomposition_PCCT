#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=BinSepImage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# Image-domain threshold-separation investigation (standalone):
# reconstruction/image_subtraction_investigation.py, settings at its top.
#
# Input : output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz
# Compute: CPU.  The --gres line is only needed for RECON_FALLBACK=True, which
#          reconstructs a slab from the raw .mat when the NIfTIs are absent.
# RAM   : ~20 GB transient; 64G is safe.
# Output: output/research/image_subtraction/
#
# Submit with:  sbatch batch_image.sh
# ============================================================================

# 1. Initialize Conda using the system path (same for all users)
source /opt/miniconda3/etc/profile.d/conda.sh

# 2. Activate the environment (user-dependent path)
conda activate /path/to/conda/envs/MatDecomp

# 3. Run from the repo root (the directory sbatch was submitted from)
cd "$SLURM_SUBMIT_DIR"

# 4. Run the investigation (standalone script; knobs are at the top of it)
python reconstruction/image_subtraction_investigation.py
