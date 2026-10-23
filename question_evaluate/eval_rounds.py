"""Self-evolution curve: score every round's verifier adapter (A_V) on the
CoMET-Bench eval split (validate_pseudo_labels.py's metrics) and write a
performance-vs-round table. Round 0 is the base model.

Rounds come from a LoRA run's --model_save_root (main.sh, RISE_LORA=1):
  served  round_adapters/solver/round_NNN, the merged adapters the run served
          (RISE_ADAPTER_MERGE=linear/ema)
  raw     each micro iteration's trained adapter,
          <MODEL_ABBR>_s_b<big>/global_step_<step>/actor/lora_adapter
  auto    served if the run saved round adapters, else raw (what a run with
          RISE_ADAPTER_MERGE=none served)

One base-model vLLM engine serves every round as a LoRA request. Rounds can be
split over GPUs with --shard/--num_shards (one process per GPU, then
--table_only); finished rounds are skipped, so an interrupted run resumes.
Writes <out_dir>/round_NNN/{records.jsonl,report.*} and <out_dir>/rounds.{csv,md}.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from question_evaluate.pseudo_label_eval import vote_outcome  # noqa: E402
from question_evaluate.validate_pseudo_labels import (  # noqa: E402
    DATA_ROOT,
    build_engine,
    labeled_items,
    sample_votes,
    write_records,
    write_report,
)


def discover_rounds(model_save_root, lineage):
    """[(round, adapter dir)] in round order, round 1 first."""
    served = sorted(glob.glob(os.path.join(model_save_root, "round_adapters", "solver", "round_*")))
    served = [d for d in served if os.path.exists(os.path.join(d, "adapter_config.json"))]
    if lineage == "served" or (lineage == "auto" and served):
        if not served:
            raise SystemExit(f"no round adapters under {model_save_root}/round_adapters/solver")
        return [(int(re.search(r"round_(\d+)$", d).group(1)), d) for d in served]

    raw = []
    for path in glob.glob(os.path.join(model_save_root, "*_s_b*", "global_step_*", "actor", "lora_adapter")):
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            big = int(re.search(r"_s_b(\d+)", path).group(1))
            step = int(re.search(r"global_step_(\d+)", path).group(1))
            raw.append(((big, step), path))
    if not raw:
        raise SystemExit(f"no solver LoRA adapters under {model_save_root}/*_s_b*/global_step_*/actor/lora_adapter")
    return [(i + 1, path) for i, (_, path) in enumerate(sorted(raw))]


def positive_stats(records):
    """Recall and precision of pseudo-label 1 against GT positives (voted candidates)."""
    tp = fn = fp = 0
    for r in records:
        pred, _ = vote_outcome(r["votes"])
        if pred is None:
            continue
        tp += pred == 1 and r["label"] == 1
        fn += pred == 0 and r["label"] == 1
        fp += pred == 1 and r["label"] == 0
    return tp / max(tp + fn, 1), tp / max(tp + fp, 1)


COLUMNS = [
    ("round", "Round"),
    ("accuracy", "Acc."),
    ("balanced_accuracy", "Bal. acc."),
    ("positive_recall", "Pos. recall"),
    ("positive_precision", "Pos. precision"),
    ("in_band_accuracy", "Acc. in band"),
    ("kept_fraction", "Kept by band"),
    ("voted_fraction", "Voted"),
    ("candidates", "Candidates"),
    ("adapter", "Adapter"),
]


def table_rows(out_dir):
    rows = []
    for report_path in sorted(glob.glob(os.path.join(out_dir, "round_*", "report.json"))):
        round_dir = os.path.dirname(report_path)
        with open(report_path) as f:
            summary = json.load(f)
        with open(os.path.join(round_dir, "records.jsonl")) as f:
            records = [json.loads(line) for line in f]
        recall, precision = positive_stats(records)
        overall = summary["overall"]
        rows.append({
            "round": int(re.search(r"round_(\d+)$", round_dir).group(1)),
            "accuracy": overall["all_voted"].get("accuracy"),
            "balanced_accuracy": overall["all_voted"].get("balanced_accuracy"),
            "positive_recall": recall,
            "positive_precision": precision,
            "in_band_accuracy": overall["in_band"].get("accuracy"),
            "kept_fraction": overall["kept_fraction"],
            "voted_fraction": overall["voted_fraction"],
            "candidates": overall["candidates"],
            "adapter": summary.get("adapter") or "base",
        })
    return rows


def write_table(out_dir, header):
    rows = table_rows(out_dir)
    with open(os.path.join(out_dir, "rounds.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key for key, _ in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    def cell(key, value):
        if value is None:
            return "–"
        if key in ("round", "candidates", "adapter"):
            return str(value)
        return f"{value:.1%}"

    lines = [f"# {header}", "", "| " + " | ".join(label for _, label in COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(cell(key, row[key]) for key, _ in COLUMNS) + " |" for row in rows]
    lines += ["", "Round 0 is the base model. Accuracy counts voted candidates; "
              "\"in band\" is what evaluate.py keeps for solver training."]
    with open(os.path.join(out_dir, "rounds.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_save_root", required=True, help="The run's MODEL_SAVE_ROOT (STORAGE_PATH/models).")
    p.add_argument("--model", required=True, help="The shared base model the adapters were trained on.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lineage", choices=["auto", "served", "raw"], default="auto")
    p.add_argument("--skip_base", action="store_true", help="Do not score the base model as round 0.")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--table_only", action="store_true", help="Only (re)write rounds.csv/.md from finished rounds.")
    p.add_argument("--no_table", action="store_true", help="Skip the table (a shard of a multi-GPU run).")
    # validate_pseudo_labels.py's data and sampling options; scored on --split eval.
    p.add_argument("--annotations", default=f"{DATA_ROOT}/datasets/Vision-Learning-Model-Video-Dataset/data/annotations.json")
    p.add_argument("--manifest", default=f"{DATA_ROOT}/storage_RISE_small/videos_5min/manifest.json")
    p.add_argument("--graph_dir", default=f"{DATA_ROOT}/storage_RISE_small/video_graphs")
    p.add_argument("--split", choices=["all", "train", "eval"], default="eval")
    p.add_argument("--eval_fraction", type=float, default=0.5)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=9)
    p.add_argument("--max_candidates", type=int, default=64)
    p.add_argument("--min_score", type=float, default=float(os.getenv("SUPERVISOR_MIN_SCORE", "0.3")))
    p.add_argument("--max_score", type=float, default=float(os.getenv("SUPERVISOR_MAX_SCORE", "0.8")))
    p.add_argument("--max_pixels", type=int, default=2097152)
    p.add_argument("--min_pixels", type=int, default=262144)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=12288)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    header = f"Self-evolution: verifier on CoMET-Bench ({args.split} split) by round, {args.model_save_root}"

    if not args.table_only:
        rounds = [(0, None)] * (not args.skip_base) + discover_rounds(args.model_save_root, args.lineage)
        todo = [(r, path) for r, path in rounds[args.shard::args.num_shards]
                if not os.path.exists(os.path.join(args.out_dir, f"round_{r:03d}", "report.json"))]
        print(f"[rounds] {len(rounds)} rounds ({args.lineage} lineage); shard {args.shard}/{args.num_shards} "
              f"scores {[r for r, _ in todo]}", flush=True)
        if todo:
            from verl.utils.lora_utils import adapter_rank, vllm_adapter

            ranks = [adapter_rank(path) for _, path in todo if path]
            engine, params = build_engine(args.model, args, max(ranks) if ranks else 0)
            items = labeled_items(args)
            for r, path in todo:
                _, request = vllm_adapter(path)
                if request is not None:
                    request.lora_int_id = r + 1  # one id per adapter
                print(f"[rounds] round {r}: {path or 'base model'}", flush=True)
                sample_votes(engine, params, items, args, lora_request=request)
                round_dir = os.path.join(args.out_dir, f"round_{r:03d}")
                records = write_records(items, args.model, round_dir)
                summary = write_report(records, args, f"round {r}", args.model, round_dir)
                with open(os.path.join(round_dir, "report.json"), "w") as f:
                    json.dump(summary | {"round": r, "adapter": path}, f, indent=2)

    if not args.no_table:
        write_table(args.out_dir, header)


if __name__ == "__main__":
    main()
