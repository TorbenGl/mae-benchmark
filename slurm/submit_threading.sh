#!/bin/bash
# =============================================================================
# Submit all 28 threading benchmark presets to SLURM.
#
# Compares single-thread (56 workers × 1 thread) vs multi-thread (16 workers ×
# 4 threads) DataLoader workers across both locality nodes and 7 batch sizes.
#
# PREREQUISITES:
#   1. Undrain node005 if needed:
#        scontrol update nodename=node005 state=resume
#
#   2. Set your W&B key (if not already in ~/.bashrc):
#        export WANDB_API_KEY=your_key_here
#
#   3. Set your ImageNet path:
#        export DATA_PATH=/datasets/imagenet21k
#
#   4. Submit:
#        bash slurm/submit_threading.sh
#
# W&B gpu_label values:
#   h200_n005_remote_st / h200_n005_remote_mt
#   h200_n007_local_st  / h200_n007_local_mt
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/threading"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting 28 threading benchmark presets..."
echo "  node005 (remote storage): 7 st + 7 mt = 14 jobs"
echo "  node007 (local storage):  7 st + 7 mt = 14 jobs"
echo ""

submitted=0
for script in \
    threading_node005_st_vit_b_bs32.sh   threading_node005_st_vit_b_bs64.sh   \
    threading_node005_st_vit_b_bs128.sh  threading_node005_st_vit_b_bs256.sh  \
    threading_node005_st_vit_b_bs512.sh  threading_node005_st_vit_b_bs1024.sh \
    threading_node005_st_vit_b_bs2048.sh \
    threading_node005_mt_vit_b_bs32.sh   threading_node005_mt_vit_b_bs64.sh   \
    threading_node005_mt_vit_b_bs128.sh  threading_node005_mt_vit_b_bs256.sh  \
    threading_node005_mt_vit_b_bs512.sh  threading_node005_mt_vit_b_bs1024.sh \
    threading_node005_mt_vit_b_bs2048.sh \
    threading_node007_st_vit_b_bs32.sh   threading_node007_st_vit_b_bs64.sh   \
    threading_node007_st_vit_b_bs128.sh  threading_node007_st_vit_b_bs256.sh  \
    threading_node007_st_vit_b_bs512.sh  threading_node007_st_vit_b_bs1024.sh \
    threading_node007_st_vit_b_bs2048.sh \
    threading_node007_mt_vit_b_bs32.sh   threading_node007_mt_vit_b_bs64.sh   \
    threading_node007_mt_vit_b_bs128.sh  threading_node007_mt_vit_b_bs256.sh  \
    threading_node007_mt_vit_b_bs512.sh  threading_node007_mt_vit_b_bs1024.sh \
    threading_node007_mt_vit_b_bs2048.sh ; do

    path="$PRESETS_DIR/$script"
    job_id=$(sbatch --parsable "$path")
    echo "  Submitted $script -> job $job_id"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. $submitted jobs submitted."
echo "Monitor: squeue -u $USER"
echo "W&B:     https://wandb.ai/your-entity/mae-slurm-benchmark"
