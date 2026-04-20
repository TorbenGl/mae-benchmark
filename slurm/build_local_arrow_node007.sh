#!/bin/bash
# =============================================================================
# Build local Arrow dataset on node007 scratch disk — ONE-TIME OPERATION.
#
# Reads ImageNet-21k from NFS ($DATA_PATH) and saves Arrow format to
# /scratch/imagenet21k_arrow. After this completes, load_from_disk() reads
# Arrow natively with zero cache overhead — all I/O stays on local scratch.
#
# Space: Arrow output ~1.1–1.2 TB, fits within the 2 TB local scratch.
# Time:  Expect 4–10 hours (1.1 TB read over NFS + local write).
#
# After completion:
#   ls -lh /scratch/imagenet21k_arrow/
#   bash slurm/submit_throughput_local.sh
# =============================================================================
#SBATCH --job-name=build-local-arrow-n007
#SBATCH --partition=gpu-node
#SBATCH --nodelist=node007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/build-local-arrow-n007-%j.out
#SBATCH --error=logs/build-local-arrow-n007-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet21k}"

mkdir -p "$REPO_DIR/logs"

source "$VENV_DIR/bin/activate"
echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME"
echo "Source : $DATA_PATH"
echo "Dest   : /scratch/imagenet21k_arrow"
echo "Started: $(date)"

python "$REPO_DIR/build_local_arrow.py" \
    --src  "$DATA_PATH" \
    --dst  /scratch/imagenet21k_arrow \
    --num_proc 8

echo "Finished: $(date)"
echo ""
echo "Verify with:"
echo "  python -c \"from datasets import load_from_disk; ds = load_from_disk('/scratch/imagenet21k_arrow'); print(len(ds), 'examples')\""
echo ""
echo "Then submit benchmarks:"
echo "  bash slurm/submit_throughput_local.sh"
