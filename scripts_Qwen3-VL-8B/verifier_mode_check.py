"""Can a cheap Verifier mode (direct / greedy, video_verifier/scoring.py)
stand in for G sampled judgments in the questioner reward? Re-plans every
format-valid query of a run's QUESTIONER_REWARD_LOG exactly as /hello does
(flat C(x, q), evenly spaced to --max_candidates), judges each candidate in
all three modes on one GPU, and reports against `sample`:

  - candidate c_i: Pearson corr. As a noise floor, `sample`'s two halves
    (odd vs even votes, G/2 each) are compared the same way here and below;
  - query d(x, q): Pearson corr;
  - questioner advantages: the run's rewards with d(x, q) swapped in,
    z-scored per reward call (as in questioner_drift_diag.py), Pearson corr;
  - wall time and generated tokens per mode.

  python3 scripts_Qwen3-VL-8B/verifier_mode_check.py --reward_log <jsonl> --model <dir> [--out report.json]
"""

import argparse
import collections
import json
import os
import statistics
import time

import vllm
from transformers import AutoTokenizer

from frozen_planner.planner import plan
from verl.utils.vllm_utils import VLLMHijack
from video_verifier import scoring
from video_verifier.prompts import build_candidate_problem, candidate_image, evenly_spaced, load_graph, tally_votes

CANDIDATE_MAX_PIXELS = 2097152
CANDIDATE_MIN_PIXELS = 262144


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return float("nan")
    mx, my = statistics.mean(x for x, _ in pairs), statistics.mean(y for _, y in pairs)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    var = (sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs)) ** 0.5
    return cov / var if var else float("nan")


def difficulty(scores):
    scores = [c for c in scores if c is not None]
    return sum(min(c, 1 - c) for c in scores) / len(scores) if scores else 0.0


def zscore(xs):
    mean, std = statistics.mean(xs), statistics.pstdev(xs)
    return [(x - mean) / (std + 1e-6) for x in xs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward_log", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--samples", type=int, default=10, help="G for the `sample` reference")
    p.add_argument("--max_candidates", type=int, default=64)
    p.add_argument("--max_model_len", type=int, default=12288)
    p.add_argument("--out", default=None)
    p.add_argument("--cache_dir", default=None, help="per-mode scores are saved here and reused, so a rerun skips finished modes")
    p.add_argument("--chunk", type=int, default=256, help="prompts per generate call; only scores are kept, not outputs")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.reward_log)]
    queries = {}  # one judgment set per distinct query; rows that repeat it share it
    for row in rows:
        if row.get("format") == 1.0 and row.get("category"):
            key = (row["question"], row["category"], json.dumps(row.get("time_bounds")), row["graph_path"])
            queries.setdefault(key, None)
    prompts = {mode: [] for mode in scoring.MODES}
    owners = []
    for q, (question, category, bounds, graph_path) in enumerate(queries):
        graph = load_graph(graph_path)
        candidates = plan(graph, {"category": category, "time_bounds": json.loads(bounds)})
        for candidate in evenly_spaced(candidates, args.max_candidates):
            image = candidate_image(graph_path, graph, candidate, CANDIDATE_MAX_PIXELS, CANDIDATE_MIN_PIXELS)
            problem = build_candidate_problem(question, candidate, graph)
            for mode in scoring.MODES:
                prompts[mode].append({"prompt": scoring.verifier_prompt(problem, mode), "multi_modal_data": {"image": image}})
            owners.append(q)
    print(f"{len(queries)} queries, {len(owners)} candidates")

    tok = AutoTokenizer.from_pretrained(args.model)
    VLLMHijack.hijack()  # vision-tower SDPA (RISE_FORCE_VIT_SDPA): no xformers on GH200, as in the reward server
    llm = vllm.LLM(
        model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
        disable_mm_preprocessor_cache=True, enable_prefix_caching=False, limit_mm_per_prompt={"image": 1, "video": 0},
    )
    stop = [tok.eos_token_id]
    scores, report = {}, {"queries": len(queries), "candidates": len(owners), "modes": {}}
    halves = [[], []]
    for mode in ("direct", "greedy", "sample"):
        cache = os.path.join(args.cache_dir, f"{mode}.json") if args.cache_dir else None
        if cache and os.path.exists(cache):
            with open(cache) as f:
                saved = json.load(f)
            if len(saved["scores"]) == len(owners):
                scores[mode], report["modes"][mode] = saved["scores"], saved["report"]
                halves = saved.get("halves", halves)
                print(f"[{mode}] reused {cache}")
                continue
        t0, tokens, scores[mode] = time.time(), 0, []
        params = scoring.sampling_params(mode, args.samples, stop)
        for start in range(0, len(prompts[mode]), args.chunk):
            outputs = llm.generate(prompts[mode][start:start + args.chunk], params)
            scores[mode] += [scoring.score_output(mode, out)[0] for out in outputs]
            tokens += sum(len(o.token_ids) for out in outputs for o in out.outputs)
            if mode == "sample":
                for i in (0, 1):
                    halves[i] += [tally_votes([o.text for o in out.outputs[i::2]])[1] for out in outputs]
            del outputs
        wall = time.time() - t0
        report["modes"][mode] = {"wall_sec": round(wall, 1), "generated_tokens": tokens,
                                 "unscored": sum(c is None for c in scores[mode])}
        print(f"[{mode}] {wall:.0f}s, {tokens} generated tokens, {report['modes'][mode]['unscored']} unscored")
        if cache:
            os.makedirs(args.cache_dir, exist_ok=True)
            with open(cache, "w") as f:
                json.dump({"scores": scores[mode], "report": report["modes"][mode],
                           **({"halves": halves} if mode == "sample" else {})}, f)

    def per_query(mode_scores):
        by_query = collections.defaultdict(list)
        for q, c in zip(owners, mode_scores):
            by_query[q].append(c)
        return [difficulty(by_query[q]) for q in range(len(queries))]

    d = {mode: per_query(scores[mode]) for mode in scoring.MODES}
    d_halves = [per_query(half) for half in halves]
    index = {key: q for q, key in enumerate(queries)}

    def advantages(query_d):
        """Per reward call, the run's rewards with d(x, q) = query_d, z-scored."""
        by_call = collections.defaultdict(list)
        for row in rows:
            if row.get("format") == 1.0 and row.get("category"):
                key = (row["question"], row["category"], json.dumps(row.get("time_bounds")), row["graph_path"])
                by_call[row["call"]].append(row["overall"] - row["difficulty"] + query_d[index[key]] - row["penalty"])
            else:
                by_call[row["call"]].append(row["overall"])
        return [a for call in sorted(by_call) for a in zscore(by_call[call])]

    ref_adv = advantages(d["sample"])
    # Noise floor of `sample` itself: two independent G/2-vote halves.
    report["sample_split_half"] = {
        "candidate_corr": round(pearson(*halves), 3),
        "advantage_corr": round(pearson(*(advantages(half) for half in d_halves)), 3),
    }
    for mode in ("direct", "greedy"):
        report["modes"][mode].update({
            "candidate_corr_vs_sample": round(pearson(scores[mode], scores["sample"]), 3),
            "query_d_corr_vs_sample": round(pearson(d[mode], d["sample"]), 3),
            "advantage_corr_vs_sample": round(pearson(advantages(d[mode]), ref_adv), 3),
            "mean_d": round(statistics.mean(d[mode]), 4),
            "speedup_vs_sample": round(report["modes"]["sample"]["wall_sec"] / max(report["modes"][mode]["wall_sec"], 1e-6), 1),
        })
    report["modes"]["sample"]["mean_d"] = round(statistics.mean(d["sample"]), 4)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
