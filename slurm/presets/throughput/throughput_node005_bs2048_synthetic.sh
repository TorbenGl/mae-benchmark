#!/bin/bash
# =============================================================================
# Throughput Benchmark — ViT-Base, batch=2048, node005 (remote storage), synthetic data
# Synthetic mode: random tensors, no disk I/O — pure GPU throughput baseline.
# Note: torch.randn is CPU-bound; 8 workers may show nonzero io/dataloader_wait_ms
# at bs2048 if tensor generation can't keep pace with the GPU. This is expected.
# node005: 64 CPUs, 157 GB RAM, H200 NVL 140 GB, remote (off-rack) storage
# NOTE: node005 must be undrained before submitting:
#   scontrol update nodename=node005 state=resume
# =============================================================================
#SBATCH --job-name=throughput-n005-bs2048-synthetic
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node005
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/throughput-n005-bs2048-synthetic-%j.out
#SBATCH --error=logs/throughput-n005-bs2048-synthetic-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark_throughput.py" \
    --model mae_vit_base_patch16 \
    --batch_size 2048 \
    --max_steps 200 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_mode synthetic \
    --prefetch_factor 4 \
    --num_workers 8 \
    --output_dir "$REPO_DIR/outputs/throughput_node005" \
    --node_label n005 \
    --precision bf16-mixed
