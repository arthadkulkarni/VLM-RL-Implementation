#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Reward / graph-building vLLM server.

Endpoints used by the questioner reward (train_examples/reward_function/cot_val.py):
    /judge_validity  temperature-0 validity + category judge over the video graph
    /hello           per-candidate Verifier sampling: for each query, every Planner
                     candidate is judged G times, returning its consistency c_i
Endpoints used by video_graph_builder: /caption_entities, /same_entity.

Prompts are shared with question_evaluate/evaluate.py via video_verifier.prompts.

    python start_vllm_server.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import sys
import threading
import time
import torch
from transformers import AutoTokenizer
import base64
import io
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from verl.utils.vllm_utils import VLLMHijack
from video_graph_builder.montage import stack_montages
from video_graph_builder.parsing import parse_caption_entities
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

# /hello judges at most this many candidates per query (evenly spaced over C(x, q)).
MAX_CANDIDATES = int(os.getenv("RISE_MAX_CANDIDATES", "64"))
# /judge_validity lists at most this many timeline segments in the graph context.
MAX_CONTEXT_SEGMENTS = int(os.getenv("RISE_MAX_CONTEXT_SEGMENTS", "120"))
CANDIDATE_MAX_PIXELS = int(os.getenv("RISE_CANDIDATE_MAX_PIXELS", "2097152"))
CANDIDATE_MIN_PIXELS = int(os.getenv("RISE_CANDIDATE_MIN_PIXELS", "262144"))
# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
parser.add_argument(
    '--max_model_len',
    type=int,
    default=12288,
    help='Cap vLLM max model length to keep KV cache requirements bounded.',
)
parser.add_argument(
    '--hello_batch_size',
    type=int,
    default=256,
    help='Maximum number of /hello prompts submitted to one vLLM generate call.',
)
args = parser.parse_args()

# ------------------------- vLLM Initialization ------------------------ #
# (This section remains unchanged)
print('[init] Loading model...')

# See VLLMHijack.hijack() for the RISE_FORCE_VIT_SDPA vision-tower attention patch.
VLLMHijack.hijack()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    max_model_len=args.max_model_len,
    disable_mm_preprocessor_cache=True,
    enable_prefix_caching=False,
    # Every endpoint sends exactly one image and never video; without this
    # vLLM profiles the encoder with a maximum-size video at startup.
    limit_mm_per_prompt={"image": 1, "video": 0},
)

sample_params = vllm.SamplingParams(
    max_tokens=2048,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10, # G: Verifier judgments sampled per candidate
)

judge_sample_params = vllm.SamplingParams(
    max_tokens=256,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    stop_token_ids=[tokenizer.eos_token_id],
    n=1,
)

caption_sample_params = vllm.SamplingParams(
    max_tokens=512,
    temperature=0.2,
    top_p=0.9,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=1,
)

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# (This section remains unchanged)
stop_event = threading.Event()    # Event to stop the thread globally
pause_event = threading.Event()   # Event to pause the thread during requests

def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1) # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

# RISE_GPU_IDLE_WORKER=0 turns the idle worker off, e.g. for graph building,
# where the builder's own DINOv2/RAFT pass shares GPU 0 with a server.
idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
if os.getenv("RISE_GPU_IDLE_WORKER", "1") == "1":
    idle_thread.start()
else:
    print('[idle_worker] disabled (RISE_GPU_IDLE_WORKER=0).')
generation_lock = threading.Lock()

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok", "port": int(args.port)})


def base64_to_pil(b64_string):
    if not b64_string:
        return None
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def extract_boxed_binary(text):
    return parse_binary(text) or 0


def build_caption_entities_chat(montage_img):
    prompt = (
        "<|im_start|>system\n"
        "You are a precise video segment annotator. You are shown one image that horizontally "
        "concatenates a few frames sampled from a short video segment, left = earliest, right = "
        "latest. Describe what happens across the segment (not just one frame), and list the "
        "distinct people/objects/entities visible.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Respond with a fenced ```json block containing exactly this shape:\n"
        "{\"caption\": \"one or two sentence description of the action across the segment\", "
        "\"entities\": [{\"name\": \"short entity name\", \"type\": \"person|object|animal|other\", "
        "\"description\": \"short visual description usable to re-identify this entity in another segment\"}]}\n"
        "Only include entities you can actually see. Output nothing outside the fenced json block.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"image": montage_img}}


def engine_is_dead(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "engine core" in message
        or "engine died" in message
        or "enginecore encountered an issue" in message
        or "shutting down" in message
    )


def dump_failed_batch(name, batch_name, batch_start, batch_end, items):
    debug_path = name.replace(
        ".json",
        f"_{batch_name}_failed_batch_{batch_start}_{batch_end - 1}.json",
    )
    with open(debug_path, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"[server] Saved failed batch metadata to {debug_path}")


def generate_with_fallback(name, batch_name, prompts, sampling_params, batch_items, use_tqdm=True):
    if not prompts:
        return []

    outputs = []
    total = len(prompts)
    chunk_size = args.hello_batch_size if batch_name == "hello" else len(prompts)
    total_batches = (total + chunk_size - 1) // chunk_size

    for batch_idx, batch_start in enumerate(range(0, total, chunk_size), start=1):
        batch = prompts[batch_start:batch_start + chunk_size]
        items = batch_items[batch_start:batch_start + chunk_size]
        batch_end = batch_start + len(batch)
        print(
            f"[server] /{batch_name} generating batch {batch_idx}/{total_batches} "
            f"(prompts {batch_start}-{batch_end - 1}, size={len(batch)})"
        )
        try:
            outputs.extend(model.generate(batch, sampling_params=sampling_params, use_tqdm=use_tqdm))
        except Exception as e:
            dump_failed_batch(name, batch_name, batch_start, batch_end, items)
            if engine_is_dead(e):
                raise
            print(f"[server] /{batch_name} batch failed, retrying one-by-one: {e}")
            for local_idx, single_prompt in enumerate(batch):
                single_item = items[local_idx]
                single_index = batch_start + local_idx
                try:
                    single_output = model.generate([single_prompt], sampling_params=sampling_params, use_tqdm=False)
                    outputs.extend(single_output)
                except Exception as single_exc:
                    dump_failed_batch(
                        name,
                        batch_name,
                        single_index,
                        single_index + 1,
                        [single_item],
                    )
                    if engine_is_dead(single_exc):
                        raise
                    print(f"[server] /{batch_name} skipping prompt {single_index}: {single_exc}")
                    outputs.append(None)
    return outputs

@app.route('/hello', methods=['GET'])
def hello():
    '''Questioner difficulty signal: for each query, the Planner enumerates C(x, q),
    each candidate is judged G times, and the per-candidate consistency c_i
    (fraction of parseable judgments that accept) is returned as candidate_scores.
    cot_val.py turns these into d(x, q).'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    results_all = [
        {
            'question': item.get('question', ''),
            'candidate_scores': [],
            'num_candidates': 0,
            'reason': '',
        }
        for item in data
    ]
    chats, chat_items, owners = [], [], []
    for idx, item in enumerate(data):
        question = item.get('question', '')
        category = normalize_category(item.get('declared_category'))
        if not question or category is None:
            results_all[idx]['reason'] = 'missing question or unknown category'
            continue
        try:
            graph = load_graph(item.get('graph_path', ''))
            candidates = plan(graph, {'category': category, 'time_bounds': item.get('time_bounds')})
        except Exception as e:
            results_all[idx]['reason'] = f'planning failed: {e}'
            continue
        candidates = evenly_spaced(candidates, MAX_CANDIDATES)
        results_all[idx]['num_candidates'] = len(candidates)
        for candidate in candidates:
            try:
                image = candidate_image(
                    item['graph_path'], graph, candidate,
                    max_pixels=CANDIDATE_MAX_PIXELS, min_pixels=CANDIDATE_MIN_PIXELS,
                )
            except Exception as e:
                print(f"[server][warning] skipping candidate {candidate['candidate_id']}: {e}")
                continue
            problem = build_candidate_problem(question, candidate, graph)
            chats.append({'prompt': build_candidate_prompt(problem), 'multi_modal_data': {'image': image}})
            chat_items.append({
                'prompt_index': len(chat_items),
                'question': question,
                'candidate_id': candidate['candidate_id'],
                'image_size': {'width': image.width, 'height': image.height},
            })
            owners.append(idx)
    print(f'[server] Prepared {len(chats)} candidate prompts for {len(data)} queries.')

    with generation_lock:
        responses = generate_with_fallback(name, 'hello', chats, sample_params, chat_items, use_tqdm=True)
    print('[server] Generation completed.')

    for idx, response in zip(owners, responses):
        if response is None:
            continue
        _, consistency = tally_votes([out.text for out in response.outputs])
        if consistency is not None:
            results_all[idx]['candidate_scores'].append(consistency)
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})


@app.route('/judge_validity', methods=['GET'])
def judge_validity():
    '''Query validity: structural checks (known category, graph loads, the Planner
    finds candidates), then a temperature-0 judge over the graph's text view.'''
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received validity request for task file: {name}')

    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    results_all = []
    valid_chats, valid_chat_items, valid_indices = [], [], []
    for idx, item in enumerate(data):
        question = item.get('question', '')
        category = normalize_category(item.get('declared_category'))
        result = {
            'question': question,
            'declared_category': category or item.get('declared_category', 'unknown'),
            'category_match': 0,
            'valid': 0,
            'reason': '',
        }
        results_all.append(result)
        if not question or category is None:
            result['reason'] = 'missing question or unknown category'
            continue
        try:
            graph = load_graph(item.get('graph_path', ''))
            if not plan(graph, {'category': category, 'time_bounds': item.get('time_bounds')}):
                result['reason'] = 'no candidates'
                continue
        except Exception as e:
            result['reason'] = f'planning failed: {e}'
            continue
        valid_chats.append({
            'prompt': build_validity_prompt(
                question, category, item.get('time_bounds'), build_graph_context(graph, MAX_CONTEXT_SEGMENTS)
            ),
        })
        valid_chat_items.append({'prompt_index': idx, 'question': question, 'declared_category': category})
        valid_indices.append(idx)

    if valid_chats:
        with generation_lock:
            responses = generate_with_fallback(
                name,
                "judge_validity",
                valid_chats,
                judge_sample_params,
                valid_chat_items,
                use_tqdm=True,
            )
        debug_responses = [response for response in responses if response is not None]
        for debug_idx, response in enumerate(debug_responses[:3]):
            raw_text = response.outputs[0].text if response.outputs else ""
            print(f"[server][validity-debug-{debug_idx}] {raw_text}")
        for idx, response in zip(valid_indices, responses):
            if response is None:
                results_all[idx]['reason'] = 'generation failed'
                continue
            raw_text = response.outputs[0].text if response.outputs else ""
            valid = parse_binary(raw_text) or 0
            results_all[idx]['valid'] = valid
            results_all[idx]['category_match'] = extract_category_match(raw_text, final_valid=valid)
            results_all[idx]['reason'] = raw_text.strip()

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    pause_event.clear()
    print(f'[server] Processed validity {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed validity {name}, results saved to {out_path}.'})


def _describe_mention(mention):
    name = str(mention.get("name", "")).strip() or "unnamed"
    description = str(mention.get("description", "")).strip()
    kind = str(mention.get("type", "")).strip()
    return f"{name} ({kind})" + (f": {description}" if description else "")


def build_same_entity_chat(pair_img, mention_a, mention_b):
    prompt = (
        "<|im_start|>system\n"
        "You are a strict video entity re-identification judge. You are shown one image with two "
        "labelled strips: Segment A on top and Segment B below, each a few frames from a short video "
        "segment, left = earliest. Decide whether the entity described in Segment A and the entity "
        "described in Segment B are the same individual -- the same physical person, animal or "
        "object instance -- not merely the same kind of thing. Judge from what you see: appearance "
        "(color, markings, clothing, distinctive features) must be consistent. Ignore changes in "
        "position, pose, scale, lighting, background and the wording of the descriptions.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"Entity in Segment A: {_describe_mention(mention_a)}\n"
        f"Entity in Segment B: {_describe_mention(mention_b)}\n"
        "In one short sentence compare their appearance, then end with \\boxed{1} if they are the "
        "same individual or \\boxed{0} if they are not.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"image": pair_img}}

@app.route('/caption_entities', methods=['GET'])
def caption_entities():
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received caption_entities request for task file: {name}')

    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    segment_ids = [item.get('segment_id', '') for item in data]
    images = [item.get('image', '') for item in data]

    pil_images = []
    for img_b64 in images:
        if img_b64:
            try:
                pil_images.append(base64_to_pil(img_b64))
            except Exception as e:
                print(f"[warning] Image decode failed in caption_entities: {e}")
                pil_images.append(None)
        else:
            pil_images.append(None)

    valid_chats = []
    valid_chat_items = []
    valid_indices = []
    results_all = [
        {
            'segment_id': segment_id,
            'caption': '',
            'entities': [],
            'reason': 'missing segment_id or image',
        }
        for segment_id in segment_ids
    ]

    for idx, (segment_id, img) in enumerate(zip(segment_ids, pil_images)):
        if segment_id and img:
            valid_chats.append(build_caption_entities_chat(img))
            valid_chat_items.append({
                "prompt_index": idx,
                "segment_id": segment_id,
                "image_size": {"width": img.width, "height": img.height},
            })
            valid_indices.append(idx)

    if valid_chats:
        with generation_lock:
            responses = generate_with_fallback(
                name,
                "caption_entities",
                valid_chats,
                caption_sample_params,
                valid_chat_items,
                use_tqdm=True,
            )
        for idx, response in zip(valid_indices, responses):
            if response is not None:
                raw_text = response.outputs[0].text if response.outputs else ""
                parsed = parse_caption_entities(raw_text)
                results_all[idx] = {
                    "segment_id": segment_ids[idx],
                    "caption": parsed["caption"],
                    "entities": parsed["entities"],
                    "reason": "",
                }
            else:
                results_all[idx] = {
                    "segment_id": segment_ids[idx],
                    "caption": "",
                    "entities": [],
                    "reason": "generation failed",
                }

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    pause_event.clear()
    print(f'[server] Processed caption_entities {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed caption_entities {name}, results saved to {out_path}.'})


@app.route('/same_entity', methods=['GET'])
def same_entity():
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received same_entity request for task file: {name}')

    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    montages = {}
    for segment_id, img_b64 in data.get('montages', {}).items():
        try:
            montages[segment_id] = base64_to_pil(img_b64)
        except Exception as e:
            print(f"[warning] Image decode failed in same_entity for {segment_id}: {e}")
            montages[segment_id] = None

    pairs = data.get('pairs', [])
    results_all = [
        {'pair_id': pair.get('pair_id', ''), 'same': 0, 'reason': 'missing montage'}
        for pair in pairs
    ]

    valid_chats = []
    valid_chat_items = []
    valid_indices = []
    for idx, pair in enumerate(pairs):
        img_a = montages.get(pair.get('segment_a'))
        img_b = montages.get(pair.get('segment_b'))
        if img_a and img_b:
            valid_chats.append(build_same_entity_chat(
                stack_montages(img_a, img_b), pair.get('mention_a', {}), pair.get('mention_b', {}),
            ))
            valid_chat_items.append({"prompt_index": idx, "pair_id": pair.get('pair_id', '')})
            valid_indices.append(idx)

    if valid_chats:
        with generation_lock:
            responses = generate_with_fallback(
                name,
                "same_entity",
                valid_chats,
                judge_sample_params,
                valid_chat_items,
                use_tqdm=True,
            )
        for idx, response in zip(valid_indices, responses):
            if response is not None:
                raw_text = response.outputs[0].text if response.outputs else ""
                results_all[idx] = {
                    "pair_id": pairs[idx].get('pair_id', ''),
                    "same": extract_boxed_binary(raw_text),
                    "reason": raw_text.strip(),
                }
            else:
                results_all[idx]["reason"] = "generation failed"

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    pause_event.clear()
    print(f'[server] Processed same_entity {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed same_entity {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=False)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        if idle_thread.is_alive():
            idle_thread.join()
        print('[main] Application shutdown complete.')
