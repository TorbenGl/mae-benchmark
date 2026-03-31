#!/bin/bash
# =============================================================================
# MAE Benchmark — Smoke Test
#
# Submits 2 jobs (node005 and node007) each running a single training batch
# via --fast_dev_run. Use this to verify data path and Python environment
# before firing off all 12 locality benchmark presets.
#
# Usage:
#   export DATA_PATH=/datasets/imagenet21k
#   export WANDB_API_KEY=your_key        # or omit to run W&B in offline mode
#   bash slurm/smoke_test.sh
#
# PREREQUISITES:
#   - node005 must be undrained: scontrol update nodename=node005 state=resume
#   - Dataset cache must be built: python prebuild_dataset_cache.py
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

mkdir -p "$SCRIPT_DIR/logs"

echo "Smoke test: submitting 2 jobs (node005 + node007) with --fast_dev_run"
echo "  Partition : gpu-node"
echo "  Data path : $DATA_PATH"
echo ""

# --- node005 (remote storage): ViT-Base, batch=256 ---
JOB1=$(sbatch --parsable \
    --job-name=mae-smoke-n005 \
    --partition=gpu-node \
    --nodelist=node005 \
    --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 \
    --cpus-per-task=8 --mem=15G --time=0:10:00 \
    --output="$SCRIPT_DIR/logs/smoke-n005-%j.out" \
    --error="$SCRIPT_DIR/logs/smoke-n005-%j.err" \
    --wrap="source $VENV_DIR/bin/activate && \
            python $REPO_DIR/train_benchmark.py \
                --model mae_vit_base_patch16 \
                --batch_size 256 \
                --max_steps 200 \
                --warmup_epochs 0 \
                --blr 1e-3 \
                --data_path '$DATA_PATH' \
                --output_dir $REPO_DIR/outputs/smoke_node005 \
                --gpu_label h200_node005_remote \
                --precision 16-mixed \
                --num_workers 7 \
                --image_col jpg --label_col cls \
                --fast_dev_run")
echo "  Submitted node005 smoke job -> $JOB1"

# --- node007 (local storage): ViT-Base, batch=256 ---
JOB2=$(sbatch --parsable \
    --job-name=mae-smoke-n007 \
    --partition=gpu-node \
    --nodelist=node007 \
    --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 \
    --cpus-per-task=16 --mem=30G --time=0:10:00 \
    --output="$SCRIPT_DIR/logs/smoke-n007-%j.out" \
    --error="$SCRIPT_DIR/logs/smoke-n007-%j.err" \
    --wrap="source $VENV_DIR/bin/activate && \
            python $REPO_DIR/train_benchmark.py \
                --model mae_vit_base_patch16 \
                --batch_size 256 \
                --max_steps 200 \
                --warmup_epochs 0 \
                --blr 1e-3 \
                --data_path '$DATA_PATH' \
                --output_dir $REPO_DIR/outputs/smoke_node007 \
                --gpu_label h200_node007_local \
                --precision 16-mixed \
                --num_workers 14 \
                --image_col jpg --label_col cls \
                --fast_dev_run")
echo "  Submitted node007 smoke job -> $JOB2"

echo ""
echo "Monitor:  squeue -j $JOB1,$JOB2"
echo "Logs:     $SCRIPT_DIR/logs/smoke-n005-$JOB1.out"
echo "          $SCRIPT_DIR/logs/smoke-n007-$JOB2.out"
echo ""
echo "Once both jobs complete successfully, run the full benchmark:"
echo "  bash slurm/submit_all.sh"
