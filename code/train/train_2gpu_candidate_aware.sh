#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/path/to/Qwen2.5-7B-Instruct}"
DATA_PATH="${2:-./input_data/candidate_aware_answercentric_full.jsonl}"
OUT_DIR="${3:-./outputs/candidate_aware_open_source}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC="${NPROC_PER_NODE:-2}"
MAX_STEPS_VALUE="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS_VALUE="${NUM_TRAIN_EPOCHS:-1.0}"

torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/train_hf.py" \
  --objective balanced_ul \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_path "$DATA_PATH" \
  --output_dir "$OUT_DIR" \
  --device_map ddp \
  --max_steps "$MAX_STEPS_VALUE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS_VALUE"
