"""Replay one questioner training step's actor update exactly as the trainer
applies it, from the rows the trainer logs with RISE_UPDATE_LOG
(verl/trainer/ray_trainer.py:log_update_batch), and measure the first
category-token distribution after each optimizer step.

What it reproduces (dp_actor.update_policy + FSDP + AdamW): the batch, in
logged order, is split into world_size contiguous device shards; each shard
into mini-batches of --mini_batch rows; optimizer step k averages mini-batch
k over all devices. Loss per rollout: verl's PPO clipped loss (clip 0.2 /
0.3, dual clip 3.0) against the starting policy's log-probs, token-mean per
sequence, mean over the mini-batch. Gradients are clipped to norm
--max_grad_norm, then AdamW (betas 0.9/0.999, eps 1e-8, bias-corrected,
weight decay left out: 5e-8 per step) carries its moments across steps.
Weights are fp32: questioner_update_diag.py showed bf16 rounding does not
change the step's effect on the category choice.

With --step2_parts, optimizer step 2 is run three times from the same
post-step-1 state (weights and Adam moments): full loss, the category-name
token's loss only, and every other token's only. The parts keep the full
loss's 1/length token weights, so before clipping they sum to the full
gradient.

It also prints, without the GPU part, each mini-batch's advantage sum per
declared category, which is often enough to see where a step goes.

  python3 scripts_Qwen3-VL-8B/questioner_update_replay.py --update_log <jsonl> --graph <graph.json> --model <dir>
"""

import argparse
import collections
import json
import re

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from video_verifier.prompts import CATEGORIES, build_questioner_system_prompt, build_questioner_user_prompt, load_graph


def declared_category(response):
    match = re.search(r"<category>\s*([a-z]+)", response)
    return match.group(1) if match else "none"


def category_token_index(tok, response):
    """Index (within the response's tokens) of the first token of the
    category name, as in questioner_update_diag.py."""
    start = response.find("<category>")
    if start < 0:
        return None
    name_start = start + len("<category>")
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if e > name_start:
            return i
    return None


def mini_batches(rows, mini_batch):
    """Rows of each optimizer step, in the order the trainer takes them."""
    world_size = rows[0]["world_size"]
    shard = len(rows) // world_size
    steps = shard // mini_batch
    return [
        [rows[w * shard + k * mini_batch + i] for w in range(world_size) for i in range(mini_batch)]
        for k in range(steps)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--update_log", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--step", type=int, default=None, help="trainer global_step to replay (default: the first logged)")
    p.add_argument("--mini_batch", type=int, default=8, help="global_batch_size_per_device")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--clip_low", type=float, default=0.2)
    p.add_argument("--clip_high", type=float, default=0.3)
    p.add_argument("--clip_dual", type=float, default=3.0)
    p.add_argument("--advantages_only", action="store_true", help="print the per-mini-batch advantage sums and stop")
    p.add_argument("--step2_parts", action="store_true",
                   help="run optimizer step 2 from the same state with the full loss, the category token only, and the rest only")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.update_log)]
    step = args.step if args.step is not None else min(r["step"] for r in rows)
    rows = sorted((r for r in rows if r["step"] == step), key=lambda r: r["position"])
    batches = mini_batches(rows, args.mini_batch)
    print(f"step {step}: {len(rows)} rollouts, world_size {rows[0]['world_size']}, {len(batches)} optimizer steps")
    for k, batch in enumerate(batches):
        sums, counts = collections.defaultdict(float), collections.Counter()
        for r in batch:
            sums[declared_category(r["response"])] += r["advantage"]
            counts[declared_category(r["response"])] += 1
        print(f"  optimizer step {k + 1}: advantage sum by category: "
              + ", ".join(f"{c}={sums[c]:+.2f} (n={counts[c]})" for c in sorted(counts)))
    if args.advantages_only:
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32, device_map="auto")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    params = []
    for name, param in model.named_parameters():
        param.requires_grad_("visual" not in name)
        if param.requires_grad:
            params.append(param)

    graph = load_graph(args.graph)
    messages = [
        {"role": "system", "content": build_questioner_system_prompt()},
        {"role": "user", "content": build_questioner_user_prompt(graph, 120)},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    n_prompt = len(tok(prompt).input_ids)
    cat_ids = {c: tok(c, add_special_tokens=False).input_ids[0] for c in CATEGORIES}
    probe_ids = tok(prompt + "<category>", return_tensors="pt").input_ids.cuda()

    def category_probs():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = model(input_ids=probe_ids, logits_to_keep=1).logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        return {c: probs[i].item() for c, i in cat_ids.items()}

    def response_logp(ids):
        n_response = ids.shape[1] - n_prompt
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = model(input_ids=ids, logits_to_keep=n_response + 1).logits[0, :-1].float()
        return torch.log_softmax(logits, -1).gather(-1, ids[0, n_prompt:, None])[:, 0]

    ids = {}
    old_logp = {}
    cat_idx = {}
    with torch.no_grad():
        for r in rows:
            ids[r["position"]] = tok(prompt + r["response"] + tok.eos_token, return_tensors="pt").input_ids.cuda()
            old_logp[r["position"]] = response_logp(ids[r["position"]]).detach()
            cat_idx[r["position"]] = category_token_index(tok, r["response"])

    show = lambda probs: ", ".join(f"{c}={probs[c]:.3f}" for c in ("bounded", "sequential", "causal", "dynamic"))
    report = {"step": step, "base": category_probs()}
    print(f"base: {show(report['base'])}")
    exp_avg = [torch.zeros_like(param) for param in params]
    exp_avg_sq = [torch.zeros_like(param) for param in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def optimizer_step(k, batch, part="full"):
        """AdamW step k on one mini-batch; part restricts the loss to the
        category token ("category") or every other token ("rest")."""
        model.zero_grad(set_to_none=True)
        clipped = 0
        for r in batch:
            logp = response_logp(ids[r["position"]])
            ratio = torch.exp(logp - old_logp[r["position"]])
            a = r["advantage"]
            loss1 = -a * ratio
            loss2 = -a * ratio.clamp(1 - args.clip_low, 1 + args.clip_high)
            loss = torch.maximum(loss1, loss2)
            if a < 0:
                loss = torch.minimum(loss, torch.full_like(loss, -a * args.clip_dual))
            clipped += (loss1 != loss).float().mean().item()
            mask = torch.ones_like(loss)
            if part != "full":
                cat_mask = torch.zeros_like(loss)
                if cat_idx[r["position"]] is not None:
                    cat_mask[cat_idx[r["position"]]] = 1
                mask = cat_mask if part == "category" else 1 - cat_mask
            if mask.any():
                ((loss * mask).sum() / loss.numel() / len(batch)).backward()
        with torch.no_grad():
            grads = [param.grad if param.grad is not None else torch.zeros_like(param) for param in params]
            norm = sum(g.pow(2).sum().item() for g in grads) ** 0.5
            scale = min(1.0, args.max_grad_norm / (norm + 1e-6))
            for param, g, m, v in zip(params, grads, exp_avg, exp_avg_sq):
                g = g * scale
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / (1 - beta1 ** k)
                v_hat = v / (1 - beta2 ** k)
                param.add_(m_hat / (v_hat.sqrt() + eps), alpha=-args.lr)
            del grads
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return norm, clipped / len(batch)

    for k, batch in enumerate(batches, start=1):
        if args.step2_parts and k == 2:
            break
        norm, clipped = optimizer_step(k, batch)
        probs = category_probs()
        report[f"after_optimizer_step_{k}"] = probs
        print(f"after optimizer step {k} (grad norm {norm:.3f}, clipped token fraction {clipped:.2f}): {show(probs)}")

    if args.step2_parts:
        # post-step-1 weights and moments, restored before each step-2 variant
        saved = [[t.detach().cpu() for t in ts] for ts in (params, exp_avg, exp_avg_sq)]
        for part in ("full", "category", "rest"):
            with torch.no_grad():
                for ts, copies in zip((params, exp_avg, exp_avg_sq), saved):
                    for t, c in zip(ts, copies):
                        t.copy_(c)
            norm, clipped = optimizer_step(2, batches[1], part)
            probs = category_probs()
            report[f"after_optimizer_step_2_{part}"] = probs
            print(f"after optimizer step 2, {part} loss (grad norm {norm:.3f}, clipped token fraction {clipped:.2f}): {show(probs)}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
