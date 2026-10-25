"""Did questioner GRPO steps move the policy toward or away from its rewarded
rollouts? Rescores the run's own rollouts (QUESTIONER_REWARD_LOG rows for
--graph) under each checkpoint and checks that the change in log p from the
first model follows each rollout's advantage. Also prints the first
category-token distribution and the category mix of fresh samples.

  python3 scripts_Qwen3-VL-8B/questioner_drift_diag.py --graph <graph.json> \
      --reward_log <questioner_rewards.jsonl> --models base=<dir> step2=<dir> ...

Advantages are the reward z-scored over all rollouts of one reward call for
this graph. The log has no GRPO uid (_balance_batch reorders the batch before
reward), but rollouts of one prompt share a baseline in expectation, so the
pooled score has the same sign as GRPO's per-group one on average. Rows logged
before the "response" field existed are rebuilt from category/question/bounds.
"""

import argparse
import collections
import json
import os
import re
import statistics

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from video_verifier.prompts import CATEGORIES, build_questioner_system_prompt, build_questioner_user_prompt, fmt_time, load_graph


def load_rollouts(reward_log, graph_path):
    """[(call, response, advantage, category)] for one graph's logged rollouts."""
    by_call = collections.defaultdict(list)
    with open(reward_log) as f:
        for line in f:
            row = json.loads(line)
            if os.path.abspath(row.get("graph_path") or "") == os.path.abspath(graph_path):
                by_call[row["call"]].append(row)
    rollouts = []
    for call, rows in sorted(by_call.items()):
        scores = [row["overall"] for row in rows]
        mean, std = statistics.mean(scores), statistics.pstdev(scores)
        for row in rows:
            response = row.get("response")
            if response is None:
                response = f"<category>{row['category']}</category>\n<question>{row['question']}</question>"
                if row.get("time_bounds"):
                    start, end = row["time_bounds"]
                    response += f"\n<time_bounds>{fmt_time(start)}, {fmt_time(end)}</time_bounds>"
            rollouts.append((call, response, (row["overall"] - mean) / (std + 1e-6), row["category"]))
    return rollouts


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return cov / var if var else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--reward_log", required=True, help="QUESTIONER_REWARD_LOG of the run that trained --models")
    p.add_argument("--models", nargs="+", required=True, help="name=path; the first is the reference")
    p.add_argument("--num_samples", type=int, default=32)
    args = p.parse_args()

    graph = load_graph(args.graph)
    rollouts = load_rollouts(args.reward_log, args.graph)
    assert rollouts, f"no rows for {args.graph} in {args.reward_log}"
    messages = [
        {"role": "system", "content": build_questioner_system_prompt()},
        {"role": "user", "content": build_questioner_user_prompt(graph, 120)},
    ]
    ref_logp = None
    for spec in args.models:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        n_prompt = prompt_ids.shape[1]

        print(f"\n=== {name}: {path} (prompt {n_prompt} tokens, {len(rollouts)} logged rollouts)")
        with torch.no_grad():
            logp = []
            for _, resp, _, _ in rollouts:
                ids = tok(prompt + resp + tok.eos_token, return_tensors="pt").input_ids.cuda()
                lp = torch.log_softmax(model(input_ids=ids).logits[0, n_prompt - 1 : -1].float(), -1)
                logp.append(lp.gather(-1, ids[0, n_prompt:, None]).sum().item())
            if ref_logp is None:
                ref_logp = logp
            else:
                # The policy followed its rewards iff log p rose on adv > 0 rollouts and fell on adv < 0 ones.
                for call in sorted({r[0] for r in rollouts}):
                    idx = [i for i, r in enumerate(rollouts) if r[0] == call]
                    adv = [rollouts[i][2] for i in idx]
                    delta = [logp[i] - ref_logp[i] for i in idx]
                    pos = [d for a, d in zip(adv, delta) if a > 0]
                    neg = [d for a, d in zip(adv, delta) if a < 0]
                    print(
                        f"  step-{call} rollouts: corr(adv, dlogp)={pearson(adv, delta):+.3f}  "
                        f"mean dlogp adv>0={statistics.mean(pos) if pos else float('nan'):+.2f} (n={len(pos)})  "
                        f"adv<0={statistics.mean(neg) if neg else float('nan'):+.2f} (n={len(neg)})"
                    )
                by_cat = collections.defaultdict(list)
                for (_, _, a, cat), d in zip(rollouts, (l - r for l, r in zip(logp, ref_logp))):
                    by_cat[cat].append((a, d))
                print("  by category (mean adv, mean dlogp): " + ", ".join(
                    f"{c}=({statistics.mean(a for a, _ in v):+.2f}, {statistics.mean(d for _, d in v):+.2f})"
                    for c, v in sorted(by_cat.items())))

            ids = tok(prompt + "<category>", return_tensors="pt").input_ids.cuda()
            probs = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)
            cat_p = {c: probs[tok(c, add_special_tokens=False).input_ids[0]].item() for c in CATEGORIES}
            print("  P(first category token): " + ", ".join(f"{c}={v:.3f}" for c, v in cat_p.items()))

            out = model.generate(prompt_ids.repeat(args.num_samples, 1), do_sample=True, temperature=1.0, top_p=0.99,
                                 max_new_tokens=160, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            texts = tok.batch_decode(out[:, n_prompt:], skip_special_tokens=True)
            cats = collections.Counter((re.search(r"<category>\s*(\w+)", t) or [None, "none"])[1] for t in texts)
            with_bounds = sum("<time_bounds>" in t for t in texts)
            print(f"  {args.num_samples} samples: categories={dict(cats)}  with <time_bounds>={with_bounds}")
            for t in texts[:3]:
                print("   | " + t.replace("\n", " ")[:220])
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
