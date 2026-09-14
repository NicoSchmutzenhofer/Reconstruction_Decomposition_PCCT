#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=SinoThresholdSep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# Sinogram-domain threshold-separation investigation (standalone):
# reconstruction/sinogram_subtraction_investigation.py, settings at its top.
#
# Subtracts the cumulative threshold sinograms (E1=A-B, E2=B-C, E3=C-D, E4=D)
# before SIRT, tests whether the A>=B>=C>=D ordering violation is noise or bias
# (invariant #3, README), and compares subtract-then-SIRT with SIRT-then-subtract.
#
# Needs : the raw .mat sinograms (paths set in the script) + a CUDA GPU.
# RAM   : peak ~32 GB (two 16 GB sinograms held at once); 64G is safe.
# Time  : scales with N_SLAB_SLICES and N_ITER (defaults 41 slices, SIRT x100);
#         shrink N_SLAB_SLICES for a quick first pass.
# Output: output/research/sinogram_subtraction/
#
# Submit with:  sbatch batch_sinogram.sh
# ============================================================================

# 1. Initialize Conda using the system path (same for all users)
source /opt/miniconda3/etc/profile.d/conda.sh

# 2. Activate the environment (user-dependent path)
conda activate /path/to/conda/envs/MatDecomp

# 3. Run from the repo root (the directory sbatch was submitted from)
cd "$SLURM_SUBMIT_DIR"

# 4. Run the investigation (standalone script; SIRT settings are at the top of it)
python reconstruction/sinogram_subtraction_investigation.py
