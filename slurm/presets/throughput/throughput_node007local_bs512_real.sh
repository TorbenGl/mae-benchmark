#!/bin/bash
# =============================================================================
# Throughput Benchmark — ViT-Base, batch=512, node007 LOCAL scratch, real data
# Data source: /scratch/imagenet21k_arrow (Arrow format, load_from_disk)
# OOM budget: 56 workers × 4 prefetch × 512 imgs × 0.6 MB = 69 GB prefetch buffer
# node007: 64 CPUs, 295 GB RAM, H200 NVL 140 GB, local NVMe/SSD scratch
#
# PREREQUISITE: run slurm/build_local_arrow_node007.sh first (one-time, ~4–10 h).
# Compare against throughput-n007-bs512-real (NFS) to measure storage locality gain.
# =============================================================================
#SBATCH --job-name=throughput-n007local-bs512-real
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=63
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs/throughput-n007local-bs512-real-%j.out
#SBATCH --error=logs/throughput-n007local-bs512-real-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"

source "$VENV_DIR/bin/activate"

# Redirect all HF and temp I/O to local scratch — keeps NFS traffic to logs only
export HF_HOME=/scratch/.hf_home
export HF_DATASETS_CACHE=/scratch/.hf_home/datasets
export TMPDIR=/scratch/tmp
mkdir -p /scratch/.hf_home /scratch/tmp

echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"

python "$REPO_DIR/train_benchmark_throughput.py" \
    --model mae_vit_base_patch16 \
    --batch_size 512 \
    --max_steps 5000 \
    --warmup_epochs 0 \
    --blr 1e-3 \
    --data_mode real \
    --data_path /scratch/imagenet21k_arrow \
    --prefetch_factor 4 \
    --num_workers 56 \
    --output_dir "$REPO_DIR/outputs/throughput_node007local" \
    --node_label n007local \
    --precision bf16-mixed \
    --compile \
    --image_col jpg --label_col cls
