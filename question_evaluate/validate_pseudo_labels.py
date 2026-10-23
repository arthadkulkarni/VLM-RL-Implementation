"""Validate the Verifier's majority-vote pseudo-labels (evaluate.py) against
CoMET-Bench ground truth. Standalone; the training loop is untouched.

For every labeled candidate (see pseudo_label_eval.py), sample G judgments
with evaluate.py's prompt and sampling settings, majority-vote them into the
pseudo-label and its agreement, and report accuracy overall and per category,
with and without the SUPERVISOR_MIN_SCORE/MAX_SCORE band, and against vote
agreement. Writes to --out_dir:

  records.jsonl     one line per candidate: GT label, category, split, votes
  report.json/.md   the numbers
  calibration.png   accuracy vs agreement

Run it with --model set to the base model, a self-play solver checkpoint, or
an SFT-control checkpoint (sft_verifier.py), and --split eval to compare them
on clips the SFT control never trained on. --records re-reports from an
existing records.jsonl without a GPU.
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from question_evaluate.pseudo_label_eval import (  # noqa: E402
    assign_splits,
    build_items,
    load_queries,
    plot_calibration,
    report_markdown,
    summarize,
)


DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--annotations", default=f"{DATA_ROOT}/datasets/Vision-Learning-Model-Video-Dataset/data/annotations.json")
    p.add_argument("--manifest", default=f"{DATA_ROOT}/storage_RISE_small/videos_5min/manifest.json")
    p.add_argument("--graph_dir", default=f"{DATA_ROOT}/storage_RISE_small/video_graphs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", help="Verifier model (HF format). Required unless --records is given.")
    p.add_argument("--name", help="Label for the report title, e.g. base, selfplay_step4, sft. Default: the model path.")
    p.add_argument("--records", help="Re-report from this records.jsonl instead of sampling.")
    p.add_argument("--split", choices=["all", "train", "eval"], default="all")
    p.add_argument("--eval_fraction", type=float, default=0.5, help="Share of clips in the eval split.")
    p.add_argument("--split_seed", type=int, default=0)
    # evaluate.py's defaults.
    p.add_argument("--num_samples", type=int, default=9, help="Judgments sampled per candidate (G).")
    p.add_argument("--max_candidates", type=int, default=64)
    p.add_argument("--min_score", type=float, default=float(os.getenv("SUPERVISOR_MIN_SCORE", "0.3")))
    p.add_argument("--max_score", type=float, default=float(os.getenv("SUPERVISOR_MAX_SCORE", "0.8")))
    p.add_argument("--max_pixels", type=int, default=2097152)
    p.add_argument("--min_pixels", type=int, default=262144)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=12288)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Judge only the first N candidates (smoke tests); 0 = all.")
    args = p.parse_args()
    if not (args.model or args.records):
        p.error("--model or --records is required")
    return args


def sample_votes(items, args):
    """G judgments per item, sampled exactly as evaluate.py samples them."""
    import vllm
    from transformers import AutoTokenizer

    from verl.utils.vllm_utils import VLLMHijack
    from video_verifier.prompts import build_candidate_prompt, candidate_image, load_graph, parse_binary

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    VLLMHijack.hijack()
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        seed=args.seed,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    params = vllm.SamplingParams(
        max_tokens=1024, temperature=1.0, top_p=1.0, top_k=40,
        stop_token_ids=[tokenizer.eos_token_id], n=args.num_samples,
    )
    start = time.time()
    for b in range(0, len(items), args.batch_size):
        batch = items[b:b + args.batch_size]
        chats = []
        for item in batch:
            graph = load_graph(item["graph_path"])
            image = candidate_image(item["graph_path"], graph, item["candidate"], args.max_pixels, args.min_pixels)
            chats.append({"prompt": build_candidate_prompt(item["problem"]), "multi_modal_data": {"image": image}})
        for item, response in zip(batch, model.generate(chats, sampling_params=params, use_tqdm=False)):
            parsed = [parse_binary(output.text) for output in response.outputs]
            item["votes"] = [v for v in parsed if v is not None]
            item["unparsed"] = sum(v is None for v in parsed)
        print(f"[validate] judged {min(b + args.batch_size, len(items))}/{len(items)} candidates "
              f"({time.time() - start:.0f}s)", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.records:
        with open(args.records) as f:
            records = [json.loads(line) for line in f]
        model_name = records[0].get("model", "?") if records else "?"
    else:
        queries = assign_splits(load_queries(args.annotations, args.manifest, args.graph_dir),
                                args.eval_fraction, args.split_seed)
        items = build_items(queries, args.max_candidates)
        if args.split != "all":
            items = [item for item in items if item["split"] == args.split]
        if args.limit:
            items = items[:args.limit]
        print(f"[validate] {len({i['query_id'] for i in items})} queries, {len(items)} candidates "
              f"(split={args.split}), GT positive rate {sum(i['label'] for i in items) / max(len(items), 1):.1%}")
        sample_votes(items, args)
        model_name = args.model
        records = [
            {key: item[key] for key in ("query_id", "clip", "split", "category", "label", "votes", "unparsed")}
            | {"candidate_id": item["candidate"]["candidate_id"], "model": model_name}
            for item in items
        ]
        with open(os.path.join(args.out_dir, "records.jsonl"), "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    if args.split != "all":
        records = [r for r in records if r["split"] == args.split]
    summary = summarize(records, args.min_score, args.max_score)
    summary.update(model=model_name, split=args.split, num_samples=args.num_samples)
    summary["name"] = args.name or model_name
    title = f"Verifier pseudo-labels vs CoMET-Bench GT: {summary['name']}, split={args.split}"
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write(report_markdown(summary, title))
    plot_calibration(summary, os.path.join(args.out_dir, "calibration.png"), "Accuracy vs vote agreement")
    print(report_markdown(summary, title))


if __name__ == "__main__":
    main()
