#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description:
    Builds per-candidate pseudo-labels for Verifier (solver) training from
    generated video queries. Runs as a batch job, often in parallel across
    multiple GPUs.

    For each generated query q over video x:
      1. Validity: structural checks (declared category is one of the Planner's
         categories; the Planner finds at least one candidate in the graph),
         then a temperature-0 judge over the video graph's captions/entities
         decides whether q is well-posed, grounded in the graph, and of the
         declared category.
      2. The Frozen Planner enumerates C(x, q) = {c_1, ..., c_K}.
      3. Each candidate c_i (its description + segment montage frames) is judged
         G times by the Verifier prompt; G binary judgments are majority-voted
         independently per candidate into the pseudo-label ŷ_i.

    Output: one row per candidate with its Verifier problem text, montage
    image, ŷ_i ("1"/"0") and vote agreement, consumed by upload.py.

Setup:
    pip install stopit transformers torch vllm

Example Usage (in a shell script):
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --model "Qwen/Qwen3-4B-Base" --suffix 0 --save_name "my_experiment" &

Input items (<save_name>_<suffix>.json) carry:
    question, declared_category (a Planner category), graph_path (a
    video_graph_builder graph JSON, built with montages), optional
    time_bounds [start_sec, end_sec], optional question_type.
'''

import json
import vllm
from transformers import AutoTokenizer
import argparse
import os
import time
from datetime import datetime
import base64
from io import BytesIO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from verl.utils.vllm_utils import VLLMHijack
from frozen_planner.planner import plan
from video_verifier.prompts import (
    build_candidate_problem,
    build_candidate_prompt,
    build_graph_context,
    build_validity_prompt,
    candidate_image,
    evenly_spaced,
    extract_category_match,
    load_graph,
    normalize_category,
    parse_binary,
    tally_votes,
)

SUPERVISOR_VALIDITY_ENABLED = os.getenv("SUPERVISOR_VALIDITY_ENABLED", "1") == "1"
# Candidates whose vote agreement (majority count / valid votes) falls outside
# this band are dropped: too-unanimous candidates carry little training signal.
SUPERVISOR_MIN_SCORE = float(os.getenv("SUPERVISOR_MIN_SCORE", "0.3"))
SUPERVISOR_MAX_SCORE = float(os.getenv("SUPERVISOR_MAX_SCORE", "0.8"))

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Build per-candidate Verifier pseudo-labels using vLLM.")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the model in Hugging Face format.")
parser.add_argument("--num_samples", type=int, default=9, help="Verifier judgments sampled per candidate (G).")
parser.add_argument("--suffix", type=str, default="0", help="A unique suffix for file naming, often the GPU index.")
parser.add_argument("--save_name", type=str, required=True, help="A base name for input and output files.")
parser.add_argument("--gpu_mem_util", type=float, default=0.85, help="GPU memory utilization passed to vLLM.")
parser.add_argument("--max_model_len", type=int, default=12288, help="Maximum model length passed to vLLM.")
parser.add_argument("--batch_size", type=int, default=256, help="Maximum number of prompts per vLLM generate batch.")
parser.add_argument("--max_pixels", type=int, default=2097152, help="Maximum pixels for each image before feeding vLLM.")
parser.add_argument("--min_pixels", type=int, default=262144, help="Minimum pixels for each image before feeding vLLM.")
parser.add_argument(
    "--max_candidates", type=int, default=64,
    help="Cap on candidates judged per query (evenly spaced over C(x, q) in timeline order); 0 = no cap.",
)
parser.add_argument(
    "--max_context_segments", type=int, default=120,
    help="Cap on timeline segments listed in the validity judge's graph context (evenly spaced).",
)
args = parser.parse_args()

# --- Constants and Paths ---
STORAGE_PATH = os.getenv("STORAGE_PATH", "../storage_RISE_Qwen3-VL-8B")
INPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
OUTPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}_results.json"


def get_image_size(image):
    if image is None:
        return None
    return {"width": int(image.width), "height": int(image.height)}


def image_to_b64(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def engine_is_dead(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "engine core" in message
        or "engine died" in message
        or "enginecore encountered an issue" in message
        or "shutting down" in message
    )


def is_addr_in_use_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "eaddrinuse" in message or "address already in use" in message


def build_vllm_model(args):
    last_exc = None
    for attempt in range(3):
        try:
            return vllm.LLM(
                model=args.model,
                tokenizer=args.model,
                gpu_memory_utilization=args.gpu_mem_util,
                max_model_len=args.max_model_len,
                seed=int(args.suffix),
                # One candidate image per prompt, never video (skips max-size video profiling).
                limit_mm_per_prompt={"image": 1, "video": 0},
            )
        except Exception as exc:
            last_exc = exc
            if not is_addr_in_use_error(exc) or attempt == 2:
                raise
            wait_s = 2 + attempt
            print(f"[vllm-init-retry] address already in use, retrying in {wait_s}s (attempt {attempt + 1}/3)")
            time.sleep(wait_s)
    raise last_exc


def extract_boxed_binary(text):
    return parse_binary(text) or 0


# --- Main Script Logic ---

# 1. Load and Prepare Data
print(f"[{args.suffix}] Loading data from: {INPUT_FILE}")
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    # Clean up the input file immediately after loading to save space
    os.remove(INPUT_FILE)
except FileNotFoundError:
    print(f"[{args.suffix}] ERROR: Input file not found. Exiting.")
    exit()

# Structural validity: category parses, graph loads, the Planner finds candidates.
query_items = []
structural_rejects = {}
for item in data:
    question = item.get("question", "")
    if not question:
        continue
    category = normalize_category(item.get("declared_category"))
    query_item = {
        "question": question,
        "question_type": item.get("question_type", ""),
        "declared_category": category or str(item.get("declared_category") or "unknown"),
        "time_bounds": item.get("time_bounds"),
        "graph_path": item.get("graph_path", ""),
        "category_match": 0,
        "valid": 0,
        "validity_reason": "",
    }
    reason = None
    if category is None:
        reason = "skipped_unknown_category"
    else:
        try:
            graph = load_graph(query_item["graph_path"])
            query = {"category": category, "time_bounds": query_item["time_bounds"]}
            query_item["candidates"] = plan(graph, query)
            if not query_item["candidates"]:
                reason = "skipped_no_candidates"
        except Exception as exc:
            print(f"[{args.suffix}] WARNING: planning failed for '{question[:50]}...': {exc}")
            reason = "skipped_graph_or_plan_failed"
    if reason:
        query_item["validity_reason"] = reason
        structural_rejects[reason] = structural_rejects.get(reason, 0) + 1
    query_items.append(query_item)

eligible_items = [item for item in query_items if not item["validity_reason"]]
print(
    f"[{args.suffix}] {len(eligible_items)}/{len(query_items)} queries passed structural checks; "
    f"rejects: {structural_rejects}"
)
if not eligible_items:
    print(f"[{args.suffix}] No valid queries found. Exiting.")
    with open(OUTPUT_FILE, "w") as f:
        json.dump([], f)
    exit()

# 2. Initialize Model and Tokenizer
print(f"[{now()}][{args.suffix}] Initializing vLLM for model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
VLLMHijack.hijack()
model = build_vllm_model(args)
sample_params = vllm.SamplingParams(
    max_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

judge_sample_params = vllm.SamplingParams(
    max_tokens=512,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    stop_token_ids=[tokenizer.eos_token_id],
    n=1,
)

# 3. Joint validity + category verification over the video graph
if SUPERVISOR_VALIDITY_ENABLED:
    validity_chats = [
        {
            "prompt": build_validity_prompt(
                item["question"],
                item["declared_category"],
                item["time_bounds"],
                build_graph_context(load_graph(item["graph_path"]), args.max_context_segments),
            ),
        }
        for item in eligible_items
    ]
    validity_start = time.time()
    print(
        f"[{now()}][{args.suffix}] Running joint validity+category verification for "
        f"{len(validity_chats)} generated queries..."
    )
    validity_responses = model.generate(validity_chats, sampling_params=judge_sample_params, use_tqdm=True)
    for debug_idx, response in enumerate(validity_responses[:3]):
        raw_text = response.outputs[0].text if response.outputs else ""
        print(f"[{args.suffix}] [validity-debug-{debug_idx}] {raw_text}")
    for item, response in zip(eligible_items, validity_responses):
        raw_text = response.outputs[0].text if response.outputs else ""
        valid = extract_boxed_binary(raw_text)
        item["category_match"] = extract_category_match(raw_text, final_valid=valid)
        item["valid"] = valid
        item["validity_reason"] = raw_text.strip()
    validity_elapsed = time.time() - validity_start
    verifier_items = [item for item in eligible_items if item["valid"] == 1]
    print(
        f"[{now()}][{args.suffix}] Joint validity+category verification kept "
        f"{len(verifier_items)}/{len(eligible_items)} queries in {validity_elapsed:.1f}s."
    )
    if not verifier_items:
        print(f"[{now()}][{args.suffix}] No valid queries remain after joint validity+category verification.")
        with open(OUTPUT_FILE, "w") as f:
            json.dump([], f, indent=4)
        print(f"[{now()}][{args.suffix}] Saved empty results to: {OUTPUT_FILE}")
        print(f"[{now()}][{args.suffix}] Script finished.")
        exit()
else:
    for item in eligible_items:
        item["category_match"] = 1
        item["valid"] = 1
        item["validity_reason"] = "skipped_validity_filter_disabled"
    verifier_items = eligible_items
    print(
        f"[{now()}][{args.suffix}] Joint validity+category verification disabled; "
        f"passing {len(verifier_items)} queries directly to candidate judging."
    )

# 4. Build one Verifier prompt per candidate: C(x, q) from the Planner, each
# shown through its segment montage(s).
candidate_entries = []
for item in verifier_items:
    graph = load_graph(item["graph_path"])
    all_candidates = item.pop("candidates")
    candidates = evenly_spaced(all_candidates, args.max_candidates)
    if len(candidates) < len(all_candidates):
        print(
            f"[{args.suffix}] Capped candidates {len(all_candidates)} -> {len(candidates)} "
            f"for '{item['question'][:50]}...'"
        )
    for candidate in candidates:
        try:
            image = candidate_image(
                item["graph_path"], graph, candidate, max_pixels=args.max_pixels, min_pixels=args.min_pixels
            )
        except Exception as exc:
            print(f"[{args.suffix}] WARNING: skipping candidate {candidate['candidate_id']}: {exc}")
            continue
        problem = build_candidate_problem(item["question"], candidate, graph)
        prompt = build_candidate_prompt(problem)
        candidate_entries.append({
            "query": item,
            "candidate": candidate,
            "problem": problem,
            "image_pil": image,
            "image_size": get_image_size(image),
            "chat": {"prompt": prompt, "multi_modal_data": {"image": image}},
        })
print(
    f"[{now()}][{args.suffix}] Built {len(candidate_entries)} candidate prompts "
    f"from {len(verifier_items)} queries."
)

# 5. Sample G judgments per candidate.
solver_gen_start = time.time()
BATCH_SIZE = args.batch_size
total_prompts = len(candidate_entries)
print(
    f"[{now()}][{args.suffix}] Starting candidate judgment generation "
    f"({total_prompts} candidates × {args.num_samples} samples = {total_prompts * args.num_samples} sequences, "
    f"batch_size={BATCH_SIZE})..."
)


def entry_metadata(entry, index):
    return {
        "prompt_index": index,
        "question": entry["query"]["question"],
        "candidate_id": entry["candidate"]["candidate_id"],
        "image_size": entry["image_size"],
    }


responses = []
for batch_start in range(0, total_prompts, BATCH_SIZE):
    batch_entries = candidate_entries[batch_start:batch_start + BATCH_SIZE]
    batch = [entry["chat"] for entry in batch_entries]
    batch_end = min(batch_start + BATCH_SIZE, total_prompts)
    print(
        f"[{now()}][{args.suffix}] Generating candidate batch "
        f"{batch_start//BATCH_SIZE + 1}/{(total_prompts + BATCH_SIZE - 1)//BATCH_SIZE} "
        f"(prompts {batch_start}-{batch_end-1})..."
    )
    try:
        batch_responses = model.generate(batch, sampling_params=sample_params, use_tqdm=True)
    except Exception as e:
        debug_path = OUTPUT_FILE.replace(
            "_results.json",
            f"_failed_batch_{batch_start}_{batch_end - 1}.json",
        )
        debug_payload = {
            "suffix": args.suffix,
            "model": args.model,
            "batch_start": batch_start,
            "batch_end": batch_end,
            "batch_size": len(batch),
            "items": [entry_metadata(entry, batch_start + offset) for offset, entry in enumerate(batch_entries)],
        }
        with open(debug_path, "w") as f:
            json.dump(debug_payload, f, indent=2, ensure_ascii=False)
        print(f"[{now()}][{args.suffix}] Saved failed batch metadata to: {debug_path}")
        if engine_is_dead(e):
            raise
        recovered_responses = []
        for local_idx, single_prompt in enumerate(batch):
            single_index = batch_start + local_idx
            try:
                single_output = model.generate([single_prompt], sampling_params=sample_params, use_tqdm=False)
                recovered_responses.extend(single_output)
            except Exception as single_exc:
                single_debug_path = OUTPUT_FILE.replace(
                    "_results.json",
                    f"_failed_item_{single_index}.json",
                )
                with open(single_debug_path, "w") as f:
                    json.dump(
                        {
                            "suffix": args.suffix,
                            "model": args.model,
                            **entry_metadata(batch_entries[local_idx], single_index),
                            "error": str(single_exc),
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"[{now()}][{args.suffix}] Saved failed item metadata to: {single_debug_path}")
                if engine_is_dead(single_exc):
                    raise
                print(f"[{now()}][{args.suffix}] Skipping prompt {single_index}: {single_exc}")
                recovered_responses.append(None)
        batch_responses = recovered_responses
    responses.extend(batch_responses)
solver_gen_elapsed = time.time() - solver_gen_start
print(
    f"[{now()}][{args.suffix}] Candidate judgment generation complete for "
    f"{len(responses)}/{total_prompts} candidates in {solver_gen_elapsed:.1f}s ({solver_gen_elapsed/60:.1f}min)."
)

# 6. Majority-vote each candidate's G judgments independently into ŷ_i.
results_all = []
vote_skips = {"failed": 0, "no_votes": 0, "tie": 0, "out_of_band": 0}
grade_start = time.time()
print(f"[{now()}][{args.suffix}] Majority-voting candidate judgments...")
for response, entry in zip(responses, candidate_entries):
    if response is None:
        vote_skips["failed"] += 1
        continue
    votes, consistency = tally_votes([output.text for output in response.outputs])
    if consistency is None:
        vote_skips["no_votes"] += 1
        continue
    if consistency == 0.5:
        vote_skips["tie"] += 1
        continue
    pseudo_label = 1 if consistency > 0.5 else 0
    score = max(consistency, 1 - consistency)
    if score < SUPERVISOR_MIN_SCORE or score > SUPERVISOR_MAX_SCORE:
        vote_skips["out_of_band"] += 1
        continue

    query_item = entry["query"]
    candidate = entry["candidate"]
    results_all.append({
        "question": entry["problem"],
        "answer": str(pseudo_label),
        "score": score,
        "image": image_to_b64(entry["image_pil"]),
        "question_type": query_item["question_type"],
        "query": query_item["question"],
        "declared_category": query_item["declared_category"],
        # upload.py balances on declared_skill; the category takes its place.
        "declared_skill": query_item["declared_category"],
        "category_match": query_item["category_match"],
        "valid": query_item["valid"],
        "validity_reason": query_item["validity_reason"],
        # No answer supervisor in the per-candidate pipeline; upload.py filters on this.
        "supervisor_correct": 1,
        "graph_path": query_item["graph_path"],
        "time_bounds": query_item["time_bounds"],
        "candidate_id": candidate["candidate_id"],
        "candidate": {key: candidate[key] for key in ("kind", "segment_ids", "spans", "captions", "entity_ids")},
        "results": votes,
    })

grade_elapsed = time.time() - grade_start
print(
    f"[{now()}][{args.suffix}] Majority-vote kept {len(results_all)}/{total_prompts} candidates "
    f"(score band [{SUPERVISOR_MIN_SCORE}, {SUPERVISOR_MAX_SCORE}]); skipped: {vote_skips}"
)

# 7. Save Final Results
print(f"[{now()}][{args.suffix}] Voting complete in {grade_elapsed:.1f}s ({grade_elapsed/60:.1f}min).")
print(f"[{now()}][{args.suffix}] Saving {len(results_all)} candidate rows to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results_all, f, indent=4)

print(f"[{now()}][{args.suffix}] Script finished.")
