#!/bin/bash
# Submit all 6 locality presets for node005 (remote storage).
# PREREQUISITE: node005 must be undrained first:
#   scontrol update nodename=node005 state=resume

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/locality"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting 6 locality presets for node005 (remote storage)..."
echo ""

submitted=0
for script in \
    locality_node005_vit_b_bs32.sh   \
    locality_node005_vit_b_bs64.sh   \
    locality_node005_vit_b_bs128.sh  \
    locality_node005_vit_b_bs256.sh  \
    locality_node005_vit_b_bs512.sh  \
    locality_node005_vit_b_bs1024.sh ; do

    job_id=$(sbatch --parsable "$PRESETS_DIR/$script")
    echo "  Submitted $script -> job $job_id"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. $submitted jobs submitted."
echo "Monitor: squeue -u $USER"
