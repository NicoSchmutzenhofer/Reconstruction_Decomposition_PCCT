#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=ThreshDomainCheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# "Are the processed thresholds cumulative or already exclusive?" diagnostic.
#
# Computes the gain-invariant inter-threshold NOISE correlation from the four
# reconstructed threshold volumes and prints a cumulative-vs-exclusive verdict.
# Settles whether the subtraction studied by the other two investigations is the
# right operation, or whether Siemens already separated the bins.
#
# Input : output/reconstruction/reconstruction_thr_{A,B,C,D}_HU.nii.gz.
# Compute: CPU-only — reads the finished NIfTIs, no ASTRA, no GPU. The --gres line
#         is kept only for scheduler consistency with the other jobs; drop it for a
#         pure CPU slot if your cluster allows CPU jobs on this partition.
# Output: output/research/processed_separation/
#           processed_separation_findings.md, processed_separation_metrics.json,
#           processed_separation_correlation.png
#
# Submit with:  sbatch batch_processed.sh
# ============================================================================

# 1. Initialize Conda using the system path (same for all users)
source /opt/miniconda3/etc/profile.d/conda.sh

# 2. Activate the environment (user-dependent path)
conda activate /path/to/conda/envs/MatDecomp

# 3. Run from the repo root (the directory sbatch was submitted from)
cd "$SLURM_SUBMIT_DIR"

# 4. Run the diagnostic (standalone script; knobs at the top of it)
python reconstruction/processed_separation_investigation.py
