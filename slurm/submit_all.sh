#!/bin/bash
# =============================================================================
# Submit all 12 locality benchmark presets to SLURM.
#
# PREREQUISITES:
#   1. Undrain node005 if needed (it may be drained after a job failure):
#        scontrol update nodename=node005 state=resume
#
#   2. Set your W&B key (if not already in ~/.bashrc):
#        export WANDB_API_KEY=your_key_here
#
#   3. Set your ImageNet path:
#        export DATA_PATH=/datasets/imagenet21k
#
#   4. Submit:
#        bash slurm/submit_all.sh
#
# Presets: ViT-Base, batch sizes 32/64/128/256/512/1024/2048, on node005 and node007.
# Both nodes use partition: gpu-node
# See slurm/USAGE.md for full details.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/locality"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting 14 locality benchmark presets..."
echo "  node005 (remote storage): 7 jobs"
echo "  node007 (local storage):  7 jobs"
echo ""

submitted=0
for script in \
    locality_node005_vit_b_bs32.sh   locality_node005_vit_b_bs64.sh   \
    locality_node005_vit_b_bs128.sh  locality_node005_vit_b_bs256.sh  \
    locality_node005_vit_b_bs512.sh  locality_node005_vit_b_bs1024.sh \
    locality_node005_vit_b_bs2048.sh \
    locality_node007_vit_b_bs32.sh   locality_node007_vit_b_bs64.sh   \
    locality_node007_vit_b_bs128.sh  locality_node007_vit_b_bs256.sh  \
    locality_node007_vit_b_bs512.sh  locality_node007_vit_b_bs1024.sh \
    locality_node007_vit_b_bs2048.sh ; do

    path="$PRESETS_DIR/$script"
    job_id=$(sbatch --parsable "$path")
    echo "  Submitted $script -> job $job_id"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. $submitted jobs submitted."
echo "Monitor: squeue -u $USER"
echo "W&B:     https://wandb.ai/your-entity/mae-slurm-benchmark"
