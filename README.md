# TRACE

Source code for **“Parameters vs. Context: TRACE Fine-Tuning for Robust Retrieval-Augmented Generation.”** TRACE uses debate traces to train a retrieval-augmented QA model to select reliable knowledge when its parametric knowledge and retrieved context disagree. Debate is used to build training data; evaluation runs the fine-tuned model in a single pass.

![TRACE framework](framework.pdf)

The original [framework diagram](trace-overview.pdf) shows the full training objective. The pipeline is:

1. Collect query-only answers and multi-round responder/critic/judge debate traces under correct and misleading contexts.
2. Turn correct candidates, incorrect candidates, and answer shifts into source-labeled training records (`internal`, `external`, or `both`).
3. Fine-tune a LoRA adapter with positive answer supervision, answer-span unlikelihood for wrong candidates, and answer-completeness regularization (answer-tail reinforcement and early-termination suppression).
4. Merge the adapter and evaluate the model on knowledge-conflict datasets.

## Repository layout

```text
TRACE-main/
├── code/
│   ├── debate/          # debate collection; responder, critic, and judge utilities
│   ├── train/           # trace-to-record conversion and LoRA training
│   └── test/            # adapter merge, vLLM evaluation, and metrics
├── data/                # supplied ConFiQA zip archives; extract before running
├── framework.png        # README preview of trace-overview.pdf
├── trace-overview.pdf
├── requirements.txt
└── README.md
```

Generated debate traces, training data, checkpoints, and evaluation results go under `outputs/` when using the commands below. They are not included in this repository.

## Environment

Use Linux with a CUDA-compatible GPU for the full training and vLLM evaluation workflow. Python 3.10 is a starting point; install a PyTorch build matching your CUDA driver before installing the remaining packages.

```bash
conda create -n trace python=3.10 -y
conda activate trace
# Install the CUDA-matched PyTorch build from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

The debate example also uses an [Ollama](https://ollama.com/) server for the critic and judge. The local responder model, Ollama model weights, and GPU capacity must be supplied separately. To use the legacy OpenAI backend instead, install `openai<1` and set `OPENAI_API_KEY`; this dependency is optional for the commands below.

## Data

The included archives contain `ConFiQA-QA`, `ConFiQA-MR`, and `ConFiQA-MC` train/test JSON files. Extract them into the layout expected by the scripts:

```bash
mkdir -p data/conflict_qa
unzip data/ConFiQA-QA.zip -d data/conflict_qa
unzip data/ConFiQA-MR.zip -d data/conflict_qa
unzip data/ConFiQA-MC.zip -d data/conflict_qa
```

The evaluator also accepts `confiqa_sc`, `explainpe`, `hotpotqa`, `2wiki`, and `musique`; their files are not in the supplied archives. See `DATASETS` in [main_knowledge.py](code/test/main_knowledge.py) for the required relative paths.

Run the following commands from the repository root. Paths passed to scripts are resolved relative to the current working directory unless absolute paths are used.

## 1. Collect debate traces

Start Ollama in one terminal and run the example launcher in another. Set `RESPONDER_MODEL` to a local Hugging Face model directory. The defaults for critic and judge are `qwen3:32b` and `deepseek-r1:8b`; override them with `CRITIC_MODEL` and `JUDGE_MODEL` if needed.

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

```bash
RESPONDER_MODEL=/path/to/Qwen2.5-7B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
bash code/debate/run.sh \
  data/conflict_qa/ConFiQA-QA-train.json \
  outputs/confiqa_qa_debate
```

The launcher defaults to five rounds and 4,800 examples. For a short check or to change settings, call [main_experiment_ca.py](code/debate/main_experiment_ca.py) directly with `--max-examples`, `--max-rounds`, and the other arguments shown by `--help`. The main output is `outputs/confiqa_qa_debate/qa_results.json`; `retrieval_material.jsonl` is also saved there.

## 2. Build records and fine-tune

This command converts the debate results into candidate-aware records and trains the LoRA adapter:

```bash
NPROC_PER_NODE=2 \
bash code/train/run_one_round_candidate_aware_v21.sh \
  /path/to/Qwen2.5-7B-Instruct \
  outputs/confiqa_qa_debate/qa_results.json \
  outputs/trace_candidate_aware
```

The records and statistics are written to `outputs/trace_candidate_aware/data/`. The adapter is saved to `outputs/trace_candidate_aware/train/adapter`. Use `NPROC_PER_NODE=1` for one GPU. `NUM_TRAIN_EPOCHS` defaults to `1.0`, and `MAX_STEPS=-1` leaves epoch control active. The launcher defaults to Hugging Face offline mode; set `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0` if the base model must be downloaded.

To build records without training, run [run_prepare_candidate_aware_only.sh](code/train/run_prepare_candidate_aware_only.sh) with a debate JSON path, output JSONL path, and statistics path.

## 3. Merge the adapter

```bash
python code/test/merge_lora_adapter.py \
  --base-model /path/to/Qwen2.5-7B-Instruct \
  --adapter outputs/trace_candidate_aware/train/adapter \
  --output outputs/trace_merged \
  --dtype bfloat16 \
  --device-map auto
```

## 4. Evaluate

```bash
MODEL_DIR=outputs/trace_merged \
DATA_ROOT=data \
OUT_DIR=outputs/evaluation \
DATASETS=confiqa_mr,confiqa_mc \
CUDA_VISIBLE_DEVICES=0 \
TP_SIZE=1 \
bash code/test/test-ncmc.sh
```
