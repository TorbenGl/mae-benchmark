#!/bin/bash
# =============================================================================
# MAE Benchmark — Smoke Test
#
# Submits 2 jobs (1 full H200, 1 MIG) that each run a single training batch
# via --fast_dev_run. Use this to verify partition names, data path, and Python
# environment before firing off all 12 benchmark presets.
#
# Usage:
#   bash slurm/smoke_test.sh
#
# Requires the same env vars as submit_all.sh:
#   export DATA_PATH=/path/to/imagenet21k
#   export WANDB_API_KEY=your_key        # or omit to run W&B in offline mode
#
# Partition names must already be configured (run submit_all.sh --configure first).
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets"

# Guard: partition placeholders must be replaced first
if grep -qr "PARTITION_NAME" "$PRESETS_DIR" 2>/dev/null; then
    echo "ERROR: Partition placeholders not yet replaced."
    echo "Run:  bash slurm/submit_all.sh --configure"
    exit 1
fi

# Read the configured partition names back out of a preset
FULL_PART=$(grep -m1 '^#SBATCH --partition=' "$PRESETS_DIR/h200_full_vit_s_bs64.sh" | cut -d= -f2)
MIG_PART=$(grep -m1  '^#SBATCH --partition=' "$PRESETS_DIR/h200_mig_vit_s_bs16.sh"  | cut -d= -f2)

VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

mkdir -p "$SCRIPT_DIR/logs"

echo "Smoke test: submitting 2 jobs (1 full H200, 1 MIG) with --fast_dev_run"
echo "  Full H200 partition : $FULL_PART"
echo "  MIG  H200 partition : $MIG_PART"
echo "  Data path           : $DATA_PATH"
echo ""

# --- Full H200: ViT-Small, batch=64 ---
JOB1=$(sbatch --parsable \
    --job-name=mae-smoke-full \
    --partition="$FULL_PART" \
    --nodelist=node005 \
    --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 \
    --cpus-per-task=4 --mem=12G --time=0:10:00 \
    --output="$SCRIPT_DIR/logs/smoke-full-%j.out" \
    --error="$SCRIPT_DIR/logs/smoke-full-%j.err" \
    --wrap="source $VENV_DIR/bin/activate && \
            python $REPO_DIR/train_benchmark.py \
                --model mae_vit_small_patch16 \
                --batch_size 64 \
                --max_steps 200 \
                --warmup_epochs 0 \
                --blr 1e-3 \
                --data_path '$DATA_PATH' \
                --output_dir $REPO_DIR/outputs/smoke_full \
                --gpu_label h200_full \
                --precision 16-mixed \
                --num_workers 4 \
                --image_col jpg --label_col cls \
                --fast_dev_run")
echo "  Submitted full H200 smoke job -> $JOB1"

# --- MIG: ViT-Small, batch=16 ---
JOB2=$(sbatch --parsable \
    --job-name=mae-smoke-mig \
    --partition="$MIG_PART" \
    --nodelist=node006 \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:1g.16gb:1 \
    --cpus-per-task=4 --mem=12G --time=0:10:00 \
    --output="$SCRIPT_DIR/logs/smoke-mig-%j.out" \
    --error="$SCRIPT_DIR/logs/smoke-mig-%j.err" \
    --wrap="source $VENV_DIR/bin/activate && \
            python $REPO_DIR/train_benchmark.py \
                --model mae_vit_small_patch16 \
                --batch_size 16 \
                --max_steps 200 \
                --warmup_epochs 0 \
                --blr 1e-3 \
                --data_path '$DATA_PATH' \
                --output_dir $REPO_DIR/outputs/smoke_mig \
                --gpu_label h200_mig \
                --precision 16-mixed \
                --num_workers 4 \
                --image_col jpg --label_col cls \
                --fast_dev_run")
echo "  Submitted MIG smoke job       -> $JOB2"

echo ""
echo "Monitor:  squeue -j $JOB1,$JOB2"
echo "Logs:     $SCRIPT_DIR/logs/smoke-full-$JOB1.out"
echo "          $SCRIPT_DIR/logs/smoke-mig-$JOB2.out"
echo ""
echo "Once both jobs complete successfully, run the full benchmark:"
echo "  bash slurm/submit_all.sh"
