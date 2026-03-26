#!/bin/bash
# =============================================================================
# Submit all 12 MAE benchmark presets to SLURM.
#
# SETUP (run once before submitting):
#   1. Set your partition names:
#        export H200_FULL_PART=your_h200_full_partition
#        export H200_MIG_PART=your_h200_mig_partition
#        bash slurm/submit_all.sh --configure
#
#   2. Set your W&B key (if not already in ~/.bashrc):
#        export WANDB_API_KEY=your_key_here
#
#   3. Set your ImageNet path:
#        export DATA_PATH=/path/to/imagenet
#
#   4. Submit:
#        bash slurm/submit_all.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets"

# --configure: apply partition names to all scripts
if [[ "${1:-}" == "--configure" ]]; then
    if [[ -z "${H200_FULL_PART:-}" || -z "${H200_MIG_PART:-}" ]]; then
        echo "ERROR: set H200_FULL_PART and H200_MIG_PART before running --configure"
        exit 1
    fi
    sed -i "s/H200_FULL_PARTITION_NAME/$H200_FULL_PART/g" "$PRESETS_DIR"/h200_full_*.sh
    sed -i "s/H200_MIG_PARTITION_NAME/$H200_MIG_PART/g"   "$PRESETS_DIR"/h200_mig_*.sh
    echo "Partition names applied."
    echo "  Full H200 partition : $H200_FULL_PART"
    echo "  MIG  H200 partition : $H200_MIG_PART"
    exit 0
fi

# Guard: warn if placeholders are still present
if grep -qr "PARTITION_NAME" "$PRESETS_DIR" 2>/dev/null; then
    echo "ERROR: Partition placeholders not yet replaced."
    echo "Run:  bash slurm/submit_all.sh --configure"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting 12 MAE benchmark presets..."
echo ""

submitted=0
for script in \
    h200_full_vit_s_bs64.sh  h200_full_vit_s_bs512.sh \
    h200_full_vit_b_bs64.sh  h200_full_vit_b_bs256.sh \
    h200_full_vit_l_bs32.sh  h200_full_vit_l_bs128.sh \
    h200_mig_vit_s_bs16.sh   h200_mig_vit_s_bs64.sh  \
    h200_mig_vit_b_bs8.sh    h200_mig_vit_b_bs32.sh  \
    h200_mig_vit_l_bs4.sh    h200_mig_vit_l_bs16.sh  ; do

    path="$PRESETS_DIR/$script"
    job_id=$(sbatch --parsable "$path")
    echo "  Submitted $script -> job $job_id"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. $submitted jobs submitted."
echo "Monitor: squeue -u $USER"
echo "W&B:     https://wandb.ai/your-entity/mae-slurm-benchmark"
