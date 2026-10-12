#!/usr/bin/env bash
set -euo pipefail

QA_RESULTS_PATH="${1:-/path/to/qa_results.json}"
DATA_PATH="${2:-./input_data/candidate_aware_answercentric_full.jsonl}"
DATA_STATS="${3:-./input_data/candidate_aware_answercentric_full_stats.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "$DATA_PATH")"

python "$SCRIPT_DIR/prepare_candidate_aware_answercentric_dataset.py" \
  --in_path "$QA_RESULTS_PATH" \
  --out_path "$DATA_PATH" \
  --out_stats "$DATA_STATS" 
