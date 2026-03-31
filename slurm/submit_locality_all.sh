#!/bin/bash
# Submit all 12 locality presets (node005 + node007, all batch sizes).
# PREREQUISITE: node005 must be undrained first:
#   scontrol update nodename=node005 state=resume

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRESETS_DIR="$SCRIPT_DIR/presets/locality"

mkdir -p "$SCRIPT_DIR/logs"

echo "Submitting all 12 locality presets..."
echo "  node005 (remote storage): 6 jobs"
echo "  node007 (local storage):  6 jobs"
echo ""

submitted=0
for script in "$PRESETS_DIR"/locality_*.sh; do
    job_id=$(sbatch --parsable "$script")
    echo "  Submitted $(basename "$script") -> job $job_id"
    submitted=$((submitted + 1))
done

echo ""
echo "Done. $submitted jobs submitted."
echo "Monitor: squeue -u $USER"
