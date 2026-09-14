#!/bin/bash
##SBATCH --nodelist=your-node
#SBATCH --job-name=ReconComparison
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# Reconstruction-stage image-quality comparison, ours vs Siemens WFBP vs VMI
# (reconstruction/recon_comparison.py, stages 0-2).
#
# Stage 1, the GPU sweep, skips any variant that already exists with a matching
# configuration, so a re-run after a crash or for the metrics only is cheap.
# Extra arguments are forwarded; `--help` there is the full list.
#
#   sbatch batch_compare.sh                       # all stages, paths from below
#   sbatch batch_compare.sh --stage 2             # metrics only (CPU, seconds)
#   sbatch batch_compare.sh --force               # redo the sweep from scratch
#   sbatch batch_compare.sh --slab=-1500,-1470    # override the detected slab
#                                                 (use = for negative values)
#
# WFBP_DIR and VMI_DIR must each hold exactly ONE reconstruction; a parent with
# several is rejected.  To see what is where:
#   python -m reconstruction.recon_comparison --list-series /path/to/export
# ============================================================================

# --- the three input paths (edit these) -------------------------------------
WFBP_DIR="/path/to/wfbp_export/"
VMI_DIR="/path/to/vmi_export/"
DATA_PATH="/path/to/full_sinogram.raw.mat"
DESC_PATH="/path/to/descriptor.raw.mat"

OUT_ROOT="output/research/recon_comparison"

# --- phantom extent, read off the WFBP volume in Slicer ---------------------
# Without this the insert-layer search covers the whole Thx-Abdomen range, where the
# table / positioning aids / scan-end artefacts can out-score the phantom -- the first
# run picked a slab at z ~ -2125 mm, roughly 665 mm away from the phantom.
# Set PHANTOM_Z="" (or pass --no-phantom-z) to search the whole range again.
PHANTOM_Z="-1515.5,-1408.3"
EXPECT_INSERTS=18

# --- hand-drawn insert prior (optional) -------------------------------------
# A Slicer segmentation of the inserts. Positions only need to be roughly right and
# a PARTIAL annotation is fine: each label is split into its individual inserts and
# every edge is then refined from the image (~10x better radii than the drawn
# outline in testing). Leave empty to fall back to automatic detection.
# Auto-discovered as $WFBP_DIR/Segmentation.nrrd when left empty.
SEGMENTATION=""
# Label value of the segment marking the UNIFORM region.  It does DOUBLE duty:
#   * it is where the noise is measured -- on this anthropomorphic phantom the body
#     outline contains lung, bone and the table, none of which are noise
#   * it is the WATER REFERENCE -- the phantom body is CTWater, specified at 0 HU, so
#     its mean HU is the only absolute accuracy number in the whole comparison.  Every
#     other metric is relative or paired and cannot see an HU-scale error at all.
# Label 4 in Segmentation.nrrd is the module boundary ring; the interior is filled
# automatically, so tracing the boundary is enough.  Painting the module solid (over the
# inserts or around them) is equivalent -- verified.  Annotated inserts + a 6 mm margin
# come out in every case.  The stage-0 log prints the labels it found, so use that
# rather than guessing if this ever moves.
BACKGROUND_LABEL="4"

# An ARRAY, not a string. SEGMENTATION defaults to a path under $WFBP_DIR, which contains
# spaces ("Thx.- Abdomen Staging_Standard - PNR_..."); an unquoted string expansion splits
# on them, and argparse then sees several arguments where one was meant.
PHANTOM_ARGS=()
if [ -n "$PHANTOM_Z" ]; then
    # NOTE the = form: a value starting with "-" is otherwise taken as a flag
    PHANTOM_ARGS+=("--phantom-z=$PHANTOM_Z")
fi
if [ -n "$EXPECT_INSERTS" ]; then
    PHANTOM_ARGS+=(--expect-inserts "$EXPECT_INSERTS")
fi
if [ -n "$SEGMENTATION" ]; then
    PHANTOM_ARGS+=(--segmentation "$SEGMENTATION")
fi
if [ -n "$BACKGROUND_LABEL" ]; then
    PHANTOM_ARGS+=(--background-label "$BACKGROUND_LABEL")
fi

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/conda/envs/MatDecomp
cd "$SLURM_SUBMIT_DIR"

set -e   # stop on the first failure: a bad stage 0 makes stages 1-2 meaningless

echo "=== 0/3  metric self-test (no data needed) ==="
python -m reconstruction.selftest_image_quality

echo
echo "=== 1/3  probe Siemens geometry + locate the insert slab (CPU) ==="
echo "    WFBP: $WFBP_DIR"
echo "    VMI : $VMI_DIR"
python -m reconstruction.recon_comparison --stage 0 \
    --wfbp-dir "$WFBP_DIR" \
    --vmi-dir  "$VMI_DIR" \
    --out-root "$OUT_ROOT" \
    "${PHANTOM_ARGS[@]}" \
    "$@"

echo
echo "=== 2/3  reconstruction sweep, 5 settings x 4 thresholds (GPU) ==="
echo "    existing variants with a matching config are skipped"
python -m reconstruction.recon_comparison --stage 1 \
    --out-root  "$OUT_ROOT" \
    --data-path "$DATA_PATH" \
    --desc-path "$DESC_PATH" \
    "$@"

echo
echo "=== 3/3  metrics: NPS / TTF / NEQ / d' / bias (CPU) ==="
python -m reconstruction.recon_comparison --stage 2 \
    --out-root "$OUT_ROOT" \
    "${PHANTOM_ARGS[@]}" \
    "$@"

echo
echo "=== done ==="
echo "  QC (check these FIRST):"
echo "    $OUT_ROOT/qc/stage0_slab_detection.png   slab must contain the inserts"
echo "    $OUT_ROOT/qc/roi_own.nrrd  roi_wfbp.nrrd  roi_vmi.nrrd   (load in Slicer ON TOP of the data)"
echo "  Results:"
echo "    $OUT_ROOT/figures/panels/                      one file per panel, for LaTeX"
echo "    $OUT_ROOT/figures/CAPTIONS.md                  draft captions + a stitching snippet"
echo "    $OUT_ROOT/figures/tradeoff_per_threshold.png   <- the headline figure"
echo "    $OUT_ROOT/figures/modality_comparison.png      ours vs WFBP vs VMI at matched noise"
echo "    $OUT_ROOT/figures/insert_localisation.png      how well each modality finds a known object"
echo "    $OUT_ROOT/figures/nps_curves.png  vmi_vs_kev.png  bland_altman_own_vs_wfbp.png"
echo "    $OUT_ROOT/metrics/metrics.json"
