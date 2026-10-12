import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE_MAP = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base causal language model")
    parser.add_argument("--base-model", required=True, help="Base model path or Hugging Face model id")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Directory used to save the merged model")
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_tokenizer(base_model: str, adapter: str, trust_remote_code: bool):
    source = adapter if os.path.exists(os.path.join(adapter, "tokenizer_config.json")) else base_model
    return AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code, use_fast=True)


def main():
    args = parse_arguments()
    dtype = DTYPE_MAP[args.dtype]
    tokenizer = load_tokenizer(args.base_model, args.adapter, args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output)
    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
