#!/usr/bin/env bash
# run_task1.sh — Run the Task 1 pipeline and tee output to a timestamped log file.
#
# Usage:
#   bash run_task1.sh [args passed to run_task1.py]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/task1_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"

python "$SCRIPT_DIR/src/task1/run_task1.py" "$@" 2>&1 | tee "$LOG_FILE"
