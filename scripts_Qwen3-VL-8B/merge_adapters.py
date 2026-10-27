"""Merge one role's per-round LoRA adapters into its round adapter
(RISE_ADAPTER_MERGE in main.sh); see verl/utils/lora_merge.py for the modes.

  python scripts_Qwen3-VL-8B/merge_adapters.py --mode ema --decay 0.9 \\
      --adapters <round 1 raw adapter> ... <round k raw adapter> --output <round k dir>
"""

import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.lora_merge import MERGE_MODES, merge_adapters, round_weights, save_adapter  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=[m for m in MERGE_MODES if m != "none"], required=True)
    p.add_argument("--decay", type=float, default=0.9, help="EMA decay (weight kept by the merged history).")
    p.add_argument("--adapters", nargs="+", required=True, help="Every round's raw adapter so far, oldest first.")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    start = time.time()
    weights = round_weights(args.mode, len(args.adapters), args.decay)
    config, tensors = merge_adapters(args.adapters, weights, device=args.device)
    info = {
        "mode": args.mode,
        "decay": args.decay if args.mode == "ema" else None,
        "round": len(args.adapters),
        "sources": [os.path.abspath(path) for path in args.adapters],
        "weights": weights,
        "space": "delta W = (alpha/r) B A, truncated to rank r by SVD",
    }
    save_adapter(args.output, config, tensors, info)
    print(f"[merge-adapters] {args.mode} of {len(args.adapters)} rounds (weights {[round(w, 4) for w in weights]}) "
          f"-> {args.output} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
