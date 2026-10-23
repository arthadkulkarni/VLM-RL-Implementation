"""Did questioner GRPO steps move the policy toward or away from its rewarded
rollouts? For each checkpoint, score fixed step-1 rollouts (one rewarded -1,
two rewarded ~+0.07), the category-token distribution, and the category mix of
fresh samples on the same graph prompt.

  python3 scripts_Qwen3-VL-8B/questioner_drift_diag.py --graph <graph.json> --models base=<dir> step2=<dir> ...
"""

import argparse
import collections
import re

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from video_verifier.prompts import CATEGORIES, build_questioner_system_prompt, build_questioner_user_prompt, load_graph

# Step-1 rollouts from job 3317242's [questioner-debug] lines, with their reward.
ROLLOUTS = {
    "bounded(-1.00)": "<category>bounded</category>\n<question>What does the woman in the white shirt do while the man "
    "in the dark jacket is shown eating and then leaning forward in distress?</question>",
    "sequential(+0.07)a": "<category>sequential</category>\n<question>When does the woman in the white shirt first "
    "appear in the shelter and tend to a sick person, and what happens immediately after?</question>",
    "sequential(+0.08)b": "<category>sequential</category>\n<question>When does the woman in the white shirt first "
    "tend to a wounded man in the shelter, and what happens immediately afterward?</question>",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--models", nargs="+", required=True, help="name=path")
    p.add_argument("--num_samples", type=int, default=32)
    args = p.parse_args()

    graph = load_graph(args.graph)
    messages = [
        {"role": "system", "content": build_questioner_system_prompt()},
        {"role": "user", "content": build_questioner_user_prompt(graph, 120)},
    ]
    for spec in args.models:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        n_prompt = prompt_ids.shape[1]

        print(f"\n=== {name}: {path} (prompt {n_prompt} tokens)")
        with torch.no_grad():
            for label, resp in ROLLOUTS.items():
                ids = tok(prompt + resp + tok.eos_token, return_tensors="pt").input_ids.cuda()
                logp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
                tgt = ids[0, 1:]
                tok_lp = logp.gather(-1, tgt[:, None])[:, 0][n_prompt - 1:]
                print(f"  logp {label:20s} total={tok_lp.sum().item():8.2f}  mean/token={tok_lp.mean().item():.3f}")

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
