import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from verl.utils.vllm_utils import VLLMHijack
from video_verifier.prompts import (
    build_questioner_system_prompt,
    build_questioner_user_prompt,
    load_graph,
    parse_questioner_output,
)
import glob
import json
import os
import random
STORAGE_PATH = os.getenv("STORAGE_PATH")
QUESTION_GENERATE_SHUFFLE = os.getenv("QUESTION_GENERATE_SHUFFLE", "1") == "1"
QUESTION_GENERATE_SHUFFLE_SEED = int(os.getenv("QUESTION_GENERATE_SHUFFLE_SEED", "42"))


def load_graph_paths(graph_dir, max_samples=None):
    """The questioner's input pool: one video graph JSON per video, as written
    by video_graph_builder (montages live in subdirectories and are skipped)."""
    graph_paths = sorted(os.path.abspath(path) for path in glob.glob(os.path.join(graph_dir, "*.json")))
    if not graph_paths:
        raise FileNotFoundError(f"No video graph .json files found in {graph_dir}")
    if max_samples:
        graph_paths = graph_paths[:max_samples]
    if QUESTION_GENERATE_SHUFFLE:
        print(f"Shuffling video graphs with seed={QUESTION_GENERATE_SHUFFLE_SEED}")
        random.Random(QUESTION_GENERATE_SHUFFLE_SEED).shuffle(graph_paths)
    print(f"Total video graphs loaded: {len(graph_paths)}")
    return graph_paths


def extract_boxed(text):
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break   # no more \boxed{…}

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
        i = j

    return results

def get_response_mask(response_ids, eos_token_id, dtype):
    batch_size, seq_len = response_ids.shape
    mask = torch.ones((batch_size, seq_len), dtype=dtype)
    for i in range(batch_size):
        for j in range(seq_len):
            if response_ids[i][j] == eos_token_id:
                mask[i][j:] = 0
                break
    return mask


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
            )
        except Exception as exc:
            last_exc = exc
            if not is_addr_in_use_error(exc) or attempt == 2:
                raise
            wait_s = 2 + attempt
            print(f"[vllm-init-retry] address already in use, retrying in {wait_s}s (attempt {attempt + 1}/3)")
            import time
            time.sleep(wait_s)
    raise last_exc


def build_sample_indices(total_samples, start_index, num_samples, suffix):
    requested = max(0, int(num_samples))
    if total_samples <= 0 or requested <= 0:
        return [], "empty"

    if start_index < total_samples:
        # Wrap around rather than truncating at the end of the graph list, so a
        # small graph pool (e.g. the mini run's single graph) still yields
        # num_samples questions -- each a fresh temperature-1.0 sample.
        end_index = start_index + requested
        print(
            f"Using contiguous slice [{start_index}, {end_index}) (wrapping) "
            f"from dataset of size {total_samples}"
        )
        return [i % total_samples for i in range(start_index, end_index)], "slice"

    seed_material = f"{QUESTION_GENERATE_SHUFFLE_SEED}:{start_index}:{suffix}"
    rng = random.Random(seed_material)
    all_indices = list(range(total_samples))
    if total_samples >= requested:
        sampled_indices = rng.sample(all_indices, requested)
        replacement = False
    else:
        sampled_indices = [rng.choice(all_indices) for _ in range(requested)]
        replacement = True

    print(
        f"start_index ({start_index}) is out of range for dataset size {total_samples}. "
        f"Fallback to random sampling {len(sampled_indices)} items from the dataset "
        f"(replacement={replacement}, seed='{seed_material}')."
    )
    return sampled_indices, "random"

def main(args):
    # breakpoint()
    VLLMHijack.hijack()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = build_vllm_model(args)
    
    print(f"Loading video graphs from {args.graph_dir}...")
    graph_paths = load_graph_paths(args.graph_dir, max_samples=args.max_samples)
    total_samples = len(graph_paths)
    start_index = max(0, int(args.start_index))
    sample_indices, sampling_mode = build_sample_indices(
        total_samples=total_samples,
        start_index=start_index,
        num_samples=args.num_samples,
        suffix=args.suffix,
    )

    system_prompt = build_questioner_system_prompt()
    sample_params = vllm.SamplingParams(
        max_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Process each sampled video graph
    results = []
    target_count = len(sample_indices)
    print(f"Question generation mode: {sampling_mode}, target_count={target_count}")
    for offset, i in enumerate(sample_indices):
        graph_path = graph_paths[i]
        print(f"Processing sample {offset+1}/{target_count} (dataset_idx={i}, graph={graph_path})")
        try:
            graph = load_graph(graph_path)
        except Exception as e:
            print(f"Failed to load graph {graph_path}: {e}")
            continue

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{build_questioner_user_prompt(graph, args.max_context_segments)}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        try:
            completions: List[RequestOutput] = model.generate([{"prompt": prompt}], sampling_params=sample_params)
        except Exception as e:
            print(f"Generation failed for dataset_idx={i}: {e}")
            results.append({
                "dataset_idx": i,
                "declared_category": "error",
                "question": "",
                "time_bounds": None,
                "graph_path": graph_path,
                "error": str(e),
            })
            if engine_is_dead(e):
                print("vLLM engine died during question generation. Saving partial results and stopping early.")
                break
            continue

        for completion in completions:
            response = completion.outputs[0].text
            parsed = parse_questioner_output(response)
            results.append({
                "declared_category": parsed["category"] or "unknown",
                # Empty when unparseable; evaluate.py drops empty questions.
                "question": parsed["question"] or "",
                "time_bounds": parsed["time_bounds"],
                "graph_path": graph_path,
                "raw_response": response,
            })

    # Save results
    output_file = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Generated {len(results)} questions and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name or path")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default=f"{STORAGE_PATH}/video_graphs",
        help="Directory of video graph JSONs from video_graph_builder (built with montages)",
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in dataset for slicing")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum video graphs to load")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file")
    parser.add_argument("--save_name", type=str, default="vqa_generated", help="Base name for output file")
    parser.add_argument("--gpu_mem_util", type=float, default=0.8, help="GPU memory utilization passed to vLLM")
    parser.add_argument("--max_model_len", type=int, default=12288, help="Maximum model length passed to vLLM")
    parser.add_argument(
        "--max_context_segments",
        type=int,
        default=120,
        help="Cap on timeline segments listed in the questioner's graph context (evenly spaced)",
    )
    args = parser.parse_args()

    main(args) 
