"""Why does the questioner's first GRPO update move P(category) against the
category rewards? Recomputes the step-1 policy gradient from a run's logged
rollouts (QUESTIONER_REWARD_LOG, call 1: raw response + reward), then
applies equal-size steps along different directions and measures the first
category-token distribution (as in questioner_drift_diag.py) after each:

  sgd    the gradient itself, scaled to the Adam step's norm -- what the
         rewards ask for;
  adam   AdamW's first step from a cold state, -lr * g / (|g| + eps): every
         parameter moves by ~lr in the sign of its gradient (clipping does
         not change it), with fp32 master weights;
  adam_bf16  the same step written into bf16 weights, which is what the
         trainer applies: fsdp.torch_dtype null loads the actor in
         mp_param_dtype (bf16) and AdamW updates those weights in place, so a
         5e-6 step survives rounding only on weights with |w| < ~2e-3
         (8.4% of the language model's);

each for three losses at ratio 1 (the PPO loss's gradient on its first
mini-batch): `full` (verl's per-sequence token mean of -A * log p, one
rollout per micro-batch), `category` (only the first category-name token),
and `rest` (every other response token). Parameters are fp32, like FSDP's
master weights: a 5e-6 step is rounded away on most bf16 weights. fp32
weights + grads (+ a bf16 copy) need ~85 GB, so the sbatch spreads the model over 2 GPUs.

Advantages: rewards z-scored over all call-1 rollouts (one graph, one
prompt, so this matches GRPO's per-group baseline in expectation).

  python3 scripts_Qwen3-VL-8B/questioner_update_diag.py --reward_log <jsonl> --graph <graph.json> --model <dir>
"""

import argparse
import json
import statistics

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from video_verifier.prompts import CATEGORIES, build_questioner_system_prompt, build_questioner_user_prompt, load_graph


def category_token_index(tok, response):
    """Index (within the response's tokens) of the first token of the
    category name, i.e. the first token starting at or after "<category>"."""
    start = response.find("<category>")
    if start < 0:
        return None
    name_start = start + len("<category>")
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if e > name_start:
            return i
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward_log", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--call", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--steps", type=float, nargs="+", default=[1, 4], help="step sizes, in Adam first steps")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.reward_log)]
    rows = [r for r in rows if r["call"] == args.call and r.get("response")]
    scores = [r["overall"] for r in rows]
    mean, std = statistics.mean(scores), statistics.pstdev(scores)
    adv = [(s - mean) / (std + 1e-6) for s in scores]
    by_cat = {}
    for r, a in zip(rows, adv):
        by_cat.setdefault(r["category"], []).append(a)
    print(f"{len(rows)} call-{args.call} rollouts; advantage sum by category: "
          + ", ".join(f"{c}={sum(v):+.2f} (n={len(v)})" for c, v in sorted(by_cat.items())))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32, device_map="auto")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    params = []
    for name, param in model.named_parameters():
        param.requires_grad_("visual" not in name)  # the questioner trains with the vision tower frozen
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

    samples = []
    for r, a in zip(rows, adv):
        ids = tok(prompt + r["response"] + tok.eos_token, return_tensors="pt").input_ids.cuda()
        samples.append((ids, a, category_token_index(tok, r["response"])))

    def gradient(part):
        """Accumulate the step-1 policy gradient for one loss part into .grad."""
        model.zero_grad(set_to_none=True)
        for ids, a, cat_idx in samples:
            n_response = ids.shape[1] - n_prompt
            # no autocast weight cache: a bf16 copy of every weight would not fit beside fp32 weights + grads
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):  # logits only where response tokens are predicted
                logits = model(input_ids=ids, logits_to_keep=n_response + 1).logits[0, :-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, ids[0, n_prompt:, None])[:, 0]
            mask = torch.ones_like(logp)
            if part != "full" and cat_idx is not None:
                cat_mask = torch.zeros_like(logp)
                cat_mask[cat_idx] = 1
                mask = cat_mask if part == "category" else 1 - cat_mask
            loss = -(a * logp * mask).sum() / mask.sum().clamp(min=1) / len(samples)
            loss.backward()

    def step(param, kind, sgd_scale):
        """This parameter's step direction (times lr), from its .grad."""
        g = param.grad
        return -g / (g.abs() + 1e-8) if kind == "adam" else -g * sgd_scale

    # The checkpoint is bf16, so these fp32 weights are exactly representable
    # in bf16 and a bf16 copy restores them exactly.
    original = {param: param.detach().to(torch.bfloat16) for param in params}

    def apply(kind, scale, sgd_scale):
        with torch.no_grad():
            for param in params:
                if param.grad is not None:
                    if kind == "adam_bf16":
                        param.copy_((param + scale * step(param, "adam", sgd_scale)).to(torch.bfloat16))
                    else:
                        param.add_(step(param, kind, sgd_scale), alpha=scale)

    def restore():
        with torch.no_grad():
            for param in params:
                param.copy_(original[param])

    base = category_probs()
    show = lambda probs: ", ".join(f"{c}={probs[c]:.3f}" for c in ("bounded", "sequential", "causal", "dynamic"))
    print(f"base: {show(base)}")
    report = {"base": base}
    for part in ("full", "category", "rest"):
        gradient(part)
        with torch.no_grad():
            grads = [param.grad for param in params if param.grad is not None]
            # |Adam step|^2 = number of parameters with a nonzero gradient
            # per-tensor .item(): the model is spread over GPUs
            adam_norm = sum((g != 0).sum().item() for g in grads) ** 0.5
            g_norm = sum(g.pow(2).sum().item() for g in grads) ** 0.5
        sgd_scale = adam_norm / g_norm  # sgd step: the gradient direction at the Adam step's norm
        for kind in ("sgd", "adam", "adam_bf16"):
            for steps in args.steps:
                scale = args.lr * steps
                apply(kind, scale, sgd_scale)
                probs = category_probs()
                restore()
                report[f"{part}/{kind}/x{steps:g}"] = probs
                print(f"{part:8s} {kind:9s} x{steps:<3g}: {show(probs)}  "
                      f"d bounded {probs['bounded'] - base['bounded']:+.3f}  d sequential {probs['sequential'] - base['sequential']:+.3f}")
        del grads  # otherwise this pass's gradients stay alive through the next pass
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
