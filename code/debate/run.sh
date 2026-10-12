#!/usr/bin/env bash
set -euo pipefail

# Example launcher for the ConFiQA QA pipeline. Run from any directory.
# Recommended usage:
#   Terminal A:
#     CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve
#   Terminal B:
#     bash code/debate/run.sh data/conflict_qa/ConFiQA-QA-train.json outputs/confiqa_qa

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INPUT_JSON="${1:-$REPO_ROOT/data/conflict_qa/ConFiQA-QA-train.json}"
OUTPUT_DIR="${2:-$REPO_ROOT/outputs/confiqa_qa_debate}"
RESPONDER_MODEL="${RESPONDER_MODEL:-/path/to/Qwen2.5-7B-Instruct}"
CRITIC_MODEL="${CRITIC_MODEL:-qwen3:32b}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-r1:8b}"
OLLAMA_BASE="${OLLAMA_BASE:-http://127.0.0.1:11434}"
export OLLAMA_CONTEXT_LENGTH=8192

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python "$SCRIPT_DIR/main_experiment_ca.py" \
  --input-json "$INPUT_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --responder-model "$RESPONDER_MODEL" \
  --critic-model "$CRITIC_MODEL" \
  --judge-model "$JUDGE_MODEL" \
  --responder-device cuda:0 \
  --critic-base-url "$OLLAMA_BASE" \
  --judge-base-url "$OLLAMA_BASE" \
  --max-rounds 5 \
  --max-examples 4800 \
  --temperature 0
