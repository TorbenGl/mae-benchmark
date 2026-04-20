#!/bin/bash
# =============================================================================
# Node007 Local-Storage Pipeline
# =============================================================================
# Step 1: Build Arrow cache from Parquet on /scratch (runs ON node007)
# Step 2: Submit 3 throughput benchmarks (bs512 / bs1024 / bs2048) that
#         depend on Step 1 completing successfully.
#
# Usage:
#   export WANDB_API_KEY=your_key
#   bash slurm/submit_node007_local_pipeline.sh
#
# Override source / destination paths if needed:
#   SRC_PATH=/datasets/imagenet21k DST_PATH=/scratch/imagenet21k_arrow bash ...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRESET_DIR="$SCRIPT_DIR/presets/throughput"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

SRC_PATH="${SRC_PATH:-/datasets/imagenet21k}"
DST_PATH="${DST_PATH:-/scratch/imagenet21k_arrow}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"

echo "========================================================"
echo " Node007 Local-Storage Pipeline"
echo "  Source (Parquet) : $SRC_PATH"
echo "  Dest   (Arrow)   : $DST_PATH"
echo "========================================================"

# ---------------------------------------------------------------------------
# Step 1 — Build Arrow cache on node007 /scratch
# Idempotent: save_dataset_to_scratch.py skips if DST_PATH already exists.
# ---------------------------------------------------------------------------
CACHE_JOB=$(sbatch --parsable \
    --job-name=build-arrow-n007 \
    --partition=gpu-node \
    --nodelist=node007 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=32G \
    --time=01:00:00 \
    --output="$LOG_DIR/build-arrow-n007-%j.out" \
    --error="$LOG_DIR/build-arrow-n007-%j.err" \
    --wrap="source $VENV_DIR/bin/activate && \
            python $REPO_DIR/save_dataset_to_scratch.py \
                --src $SRC_PATH \
                --dst $DST_PATH \
                --num_proc 16")

echo "Step 1 submitted: build-arrow-n007 -> job $CACHE_JOB"

# ---------------------------------------------------------------------------
# Step 2 — Submit 3 throughput benchmarks, each waiting for Step 1
# ---------------------------------------------------------------------------
DEPS="--dependency=afterok:${CACHE_JOB}"

JOB_512=$(sbatch --parsable $DEPS \
    --export=ALL,DATA_PATH="$DST_PATH" \
    "$PRESET_DIR/throughput_node007_bs512_real.sh")
echo "Step 2 submitted: bs512  -> job $JOB_512  (depends on $CACHE_JOB)"

JOB_1024=$(sbatch --parsable $DEPS \
    --export=ALL,DATA_PATH="$DST_PATH" \
    "$PRESET_DIR/throughput_node007_bs1024_real.sh")
echo "Step 2 submitted: bs1024 -> job $JOB_1024 (depends on $CACHE_JOB)"

JOB_2048=$(sbatch --parsable $DEPS \
    --export=ALL,DATA_PATH="$DST_PATH" \
    "$PRESET_DIR/throughput_node007_bs2048_real.sh")
echo "Step 2 submitted: bs2048 -> job $JOB_2048 (depends on $CACHE_JOB)"

echo ""
echo "Done. 4 jobs queued."
echo "  Cache build : $CACHE_JOB"
echo "  bs512       : $JOB_512"
echo "  bs1024      : $JOB_1024"
echo "  bs2048      : $JOB_2048"
echo ""
echo "Monitor : squeue -u \$USER"
echo "Logs    : $LOG_DIR/"
