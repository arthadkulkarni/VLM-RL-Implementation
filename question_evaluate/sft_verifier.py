"""SFT control for the self-play Verifier: fine-tune on CoMET-Bench ground-truth
labels for the same candidates validate_pseudo_labels.py scores, so the
result can be set against self-play training on majority-vote pseudo-labels.
Standalone; the training loop is untouched.

Trains on the --split train clips only (the split is by clip, see
pseudo_label_eval.assign_splits); evaluate the saved checkpoint with
validate_pseudo_labels.py --split eval, as the base and self-play models.

Input is evaluate.py's candidate prompt (build_candidate_prompt + the
candidate's montage); the target is "\\boxed{<GT label>}" and the loss is on
the target tokens only. Precision, sharding, optimizer and clipping follow the
self-play solver's actor (cot_config.yaml + verl's FSDP worker): bf16 params
with fp32 reduce, FULL_SHARD per decoder layer / vision block, gradient
checkpointing, vision tower trained, AdamW lr 5e-6 wd 0.01, grad norm 1.0,
global batch 64, constant LR with no warmup.

  torchrun --nproc_per_node 4 question_evaluate/sft_verifier.py --model <base> --out_dir <dir>
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from question_evaluate.pseudo_label_eval import assign_splits, build_items, load_queries  # noqa: E402
from question_evaluate.validate_pseudo_labels import DATA_ROOT  # noqa: E402
from verl.utils.fsdp_utils import get_fsdp_wrap_policy  # noqa: E402
from video_verifier.prompts import build_candidate_prompt, candidate_image, load_graph  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--annotations", default=f"{DATA_ROOT}/datasets/Vision-Learning-Model-Video-Dataset/data/annotations.json")
    p.add_argument("--manifest", default=f"{DATA_ROOT}/storage_RISE_small/videos_5min/manifest.json")
    p.add_argument("--graph_dir", default=f"{DATA_ROOT}/storage_RISE_small/video_graphs")
    p.add_argument("--eval_fraction", type=float, default=0.5)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--max_candidates", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="Stop after this many optimizer steps (smoke tests); 0 = no cap.")
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_pixels", type=int, default=2097152)
    p.add_argument("--min_pixels", type=int, default=262144)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def encode(item, processor, args):
    """Prompt + GT target, with the loss masked to the target tokens."""
    graph = load_graph(item["graph_path"])
    image = candidate_image(item["graph_path"], graph, item["candidate"], args.max_pixels, args.min_pixels)
    prompt = processor(text=[build_candidate_prompt(item["problem"])], images=[image], return_tensors="pt")
    target = processor.tokenizer(f"\\boxed{{{item['label']}}}<|im_end|>", add_special_tokens=False,
                                 return_tensors="pt").input_ids
    input_ids = torch.cat([prompt["input_ids"], target], dim=1)
    labels = torch.cat([torch.full_like(prompt["input_ids"], -100), target], dim=1)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "pixel_values": prompt["pixel_values"],
        "image_grid_thw": prompt["image_grid_thw"],
    }


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    log = (lambda msg: print(f"[sft {time.strftime('%H:%M:%S')}] {msg}", flush=True)) if rank == 0 else (lambda msg: None)

    queries = assign_splits(load_queries(args.annotations, args.manifest, args.graph_dir), args.eval_fraction, args.split_seed)
    items = [item for item in build_items(queries, args.max_candidates) if item["split"] == "train"]
    accum = args.global_batch_size // world
    steps_per_epoch = len(items) // args.global_batch_size
    log(f"{len(items)} train candidates from {len({i['clip'] for i in items})} clips, "
        f"GT positive rate {sum(i['label'] for i in items) / max(len(items), 1):.1%}; "
        f"{steps_per_epoch} steps/epoch x {args.epochs} epochs, {accum} micro-batches/GPU/step")
    if steps_per_epoch == 0:
        raise SystemExit(f"fewer train candidates ({len(items)}) than one global batch ({args.global_batch_size})")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=os.getenv("RISE_ATTN_IMPLEMENTATION", "sdpa"),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=get_fsdp_wrap_policy(model),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
        device_id=torch.cuda.current_device(),
        forward_prefetch=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay, fused=True)

    step = 0
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(items)))
        random.Random(args.seed + epoch).shuffle(order)
        for s in range(steps_per_epoch):
            batch = order[s * args.global_batch_size:(s + 1) * args.global_batch_size][rank::world]
            start, total = time.time(), torch.zeros(2, device="cuda")
            for idx in batch:
                inputs = {k: v.cuda() for k, v in encode(items[idx], processor, args).items()}
                loss = model(**inputs).loss
                (loss / accum).backward()
                total += torch.tensor([loss.item(), 1.0], device="cuda")
            grad_norm = model.clip_grad_norm_(args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            dist.all_reduce(total)
            step += 1
            log(f"epoch {epoch + 1} step {step}: loss {total[0].item() / total[1].item():.4f}, "
                f"grad_norm {grad_norm.item():.3f}, {time.time() - start:.0f}s")
            if step == args.max_steps:
                break
        if step == args.max_steps:
            break

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state_dict = model.state_dict()
    if rank == 0:
        model.module.save_pretrained(args.out_dir, state_dict=state_dict, safe_serialization=True)
        processor.save_pretrained(args.out_dir)
        with open(os.path.join(args.out_dir, "sft_control.json"), "w") as f:
            json.dump(vars(args) | {"train_candidates": len(items), "steps": step}, f, indent=2)
        log(f"saved {args.out_dir}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
