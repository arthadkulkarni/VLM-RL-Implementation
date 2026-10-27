"""Merge a LoRA adapter into its base model and save a full HF checkpoint.

LoRA training saves adapter-only checkpoints (<step>/actor/lora_adapter); the
pipeline serves them on the shared base directly, so merging is only needed
for tools that want a standalone model (e.g. questioner_drift_diag.py).

  python scripts_Qwen3-VL-8B/merge_lora.py --base <base model> --adapter <.../actor/lora_adapter> --output <dir>
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", required=True, help="Base model the adapter was trained on (HF format).")
    p.add_argument("--adapter", required=True, help="peft adapter directory (adapter_config.json + weights).")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.output, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.output)
    print(f"Merged {args.adapter} into {args.base} -> {args.output}")


if __name__ == "__main__":
    main()
