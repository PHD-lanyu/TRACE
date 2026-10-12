#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

MODEL_DIR="${MODEL_DIR:-./merged_model}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./results/test-ncmc}"
DATASETS="${DATASETS:-confiqa_mc}"
TP_SIZE="${TP_SIZE:-1}"
RAG="${RAG:-true}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS=()

if [[ -n "${CONFIQA_QA_FILE:-}" ]]; then
  EXTRA_ARGS+=(--confiqa-qa-file "$CONFIQA_QA_FILE")
fi

python "$SCRIPT_DIR/main_knowledge.py" \
  --model "$MODEL_DIR" \
  --output-dir "$OUT_DIR" \
  --data-path "$DATA_ROOT" \
  --datasets "$DATASETS" \
  --tensor-parallel-size "$TP_SIZE" \
  --rag "$RAG" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
