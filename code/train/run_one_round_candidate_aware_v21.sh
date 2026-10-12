#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/path/to/Qwen2.5-7B-Instruct}"
QA_RESULTS_PATH="${2:-/path/to/qa_results.json}"
OUT_ROOT="${3:-./outputs/candidate_aware_open_source}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="$OUT_ROOT/data"
TRAIN_OUT="$OUT_ROOT/train"
mkdir -p "$DATA_DIR" "$TRAIN_OUT"

DATA_PATH="$DATA_DIR/candidate_aware_answercentric_full.jsonl"
DATA_STATS="$DATA_DIR/candidate_aware_answercentric_full_stats.json"

echo "[DATA] Build candidate-aware answer-centric CE/UL dataset"
python "$SCRIPT_DIR/prepare_candidate_aware_answercentric_dataset.py" \
  --in_path "$QA_RESULTS_PATH" \
  --out_path "$DATA_PATH" \
  --out_stats "$DATA_STATS" 

bash "$SCRIPT_DIR/train_2gpu_candidate_aware.sh" "$MODEL_PATH" "$DATA_PATH" "$TRAIN_OUT"

echo "[DONE] adapter dir: $TRAIN_OUT/adapter"
echo "[DONE] dataset stats: $DATA_STATS"
