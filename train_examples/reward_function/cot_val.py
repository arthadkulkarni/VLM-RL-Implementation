# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This reward function is for regular [CoT] -> [Answer] GRPO finetuning
'''
import re, os, json, glob
from typing import Dict, List, Optional
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from collections import Counter
STORAGE_PATH = os.getenv("STORAGE_PATH")
if STORAGE_PATH is None:
    STORAGE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
SUPERVISOR_VALIDITY_ENABLED = os.getenv("SUPERVISOR_VALIDITY_ENABLED", "1") == "1"
QUESTIONER_DEBUG_SAMPLES = int(os.getenv("QUESTIONER_DEBUG_SAMPLES", "3"))
SKILL_AWARE_ENABLED = os.getenv("SKILL_AWARE_ENABLED", "1") == "1"
SKILL_BALANCE_ENABLED = os.getenv("SKILL_BALANCE_ENABLED", "1") == "1"
SKILL_BALANCE_WEIGHT = float(os.getenv("SKILL_BALANCE_WEIGHT", "0.2"))
REWARD_REQUEST_RETRIES = int(os.getenv("REWARD_REQUEST_RETRIES", "12"))
REWARD_REQUEST_RETRY_SLEEP_SEC = float(os.getenv("REWARD_REQUEST_RETRY_SLEEP_SEC", "5"))
# Number of judge/reward vLLM servers started by vllm_service_init/start.sh on ports
# 6000..6000+N-1. Must match RISE_REWARD_SERVERS used there.
NUM_REWARD_SERVERS = int(os.getenv("RISE_REWARD_SERVERS", "4"))
# Scale on the duplicate-share penalty. The share is in (0, 1] while difficulty
# d is in [0, 0.5], so unscaled a fully duplicated batch (e.g. one graph repeated)
# drives every valid query to <= -0.5 and swamps d entirely.
DUPLICATE_PENALTY_WEIGHT = float(os.getenv("RISE_DUPLICATE_PENALTY_WEIGHT", "0.2"))

TEMP_RESULTS_DIR = os.path.join(STORAGE_PATH, "temp_results")
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

# The questioner's query categories (the Frozen Planner's taxonomy). Named
# "skills" for continuity with RISE's skill-balance machinery.
ALLOWED_SKILLS = [
    "causal",
    "sequential",
    "synchronous",
    "bounded",
    "static",
    "dynamic",
    "identity",
    "negative",
]
SKILL_ALIASES = {skill: skill for skill in ALLOWED_SKILLS}
_SKILL_COUNTS_CACHE = {"signature": None, "counts": Counter()}


def normalize_skill_label(skill: Optional[str]) -> Optional[str]:
    if skill is None:
        return None
    normalized = str(skill).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return SKILL_ALIASES.get(normalized)


def get_recent_skill_counts() -> Counter:
    summary_dir = os.path.join(STORAGE_PATH, "local_parquet")
    if not os.path.isdir(summary_dir):
        return Counter()

    summary_files = sorted(glob.glob(os.path.join(summary_dir, "*_train_summary.json")))
    signature = tuple(
        (path, os.path.getmtime(path), os.path.getsize(path))
        for path in summary_files
        if os.path.exists(path)
    )
    if signature == _SKILL_COUNTS_CACHE["signature"]:
        return _SKILL_COUNTS_CACHE["counts"]

    counts = Counter()
    for path, _, _ in signature:
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[skill-balance] failed to load summary {path}: {exc}")
            continue
        skill_counts = payload.get("declared_skill_counts_after_balance", {}) or payload.get("declared_skill_counts_after_filter", {})
        for skill, value in skill_counts.items():
            normalized = normalize_skill_label(skill)
            if normalized:
                try:
                    counts[normalized] += int(value)
                except Exception:
                    continue

    _SKILL_COUNTS_CACHE["signature"] = signature
    _SKILL_COUNTS_CACHE["counts"] = counts
    return counts


def compute_skill_balance_bonus(skill: Optional[str], counts: Counter) -> float:
    normalized = normalize_skill_label(skill)
    if not normalized or normalized not in ALLOWED_SKILLS:
        return 0.0
    total = sum(counts.get(label, 0) for label in ALLOWED_SKILLS)
    if total <= 0:
        return 0.0
    target = total / len(ALLOWED_SKILLS)
    current = counts.get(normalized, 0)
    if current >= target:
        return 0.0
    return (target - current) / max(target, 1.0)

def duplicate_share_per_problem(keys):
    """Diversity penalty, placeholder: each query's share of the batch taken by
    exact duplicates of it (same category, normalized question text, video, and
    time bounds). Open design item (reference doc Section 4d): replace with a
    distance over taxonomy fields + involved graph entities, clustered like the
    original BLEU-based cluster_share_per_problem."""
    if not keys:
        return []
    total = len(keys)
    counts = Counter(keys)
    return [counts[key] / total for key in keys]


def duplicate_key(item, graph_path):
    question = " ".join(str(item.get("question", "")).lower().split())
    time_bounds = tuple(item["time_bounds"]) if item.get("time_bounds") else None
    return (item.get("declared_skill"), question, graph_path, time_bounds)


def format_reward(predict: str) -> float:
    pattern = re.compile(
        r"^\s*<category>.+?</category>\s*"
        r"<question>.+?</question>\s*"
        r"(?:<time_bounds>.+?</time_bounds>\s*)?$",
        re.DOTALL
    )
    return 1.0 if pattern.fullmatch(predict.strip()) else 0.0


def parse_time_bounds(text):
    """"start, end" in seconds -> [start, end], or None if malformed."""
    numbers = re.findall(r"\d+(?:\.\d+)?", text or "")
    if len(numbers) != 2:
        return None
    start, end = float(numbers[0]), float(numbers[1])
    return [start, end] if start <= end else None


def match(generation):
    match_obj = re.search(r"<category>(.*?)</category>.*?<question>(.*?)</question>", generation, re.DOTALL)
    if not match_obj:
        return None
    bounds_obj = re.search(r"<time_bounds>(.*?)</time_bounds>", generation, re.DOTALL)
    return {
        "declared_skill": normalize_skill_label(match_obj.group(1)),
        "question": match_obj.group(2).strip(),
        "time_bounds_raw": bounds_obj.group(1).strip() if bounds_obj else None,
        "time_bounds": parse_time_bounds(bounds_obj.group(1)) if bounds_obj else None,
    }


def compute_format_components(predict: str) -> Dict[str, float]:
    normalized_predict = predict.strip()
    structure_ok = format_reward(normalized_predict)
    parsed = match(normalized_predict) if structure_ok else None

    skill_ok = 1.0 if parsed and parsed.get("declared_skill") in ALLOWED_SKILLS else 0.0
    question_ok = 1.0 if parsed and parsed.get("question") else 0.0
    # time_bounds must parse when given, and is required for `bounded`.
    if not parsed:
        time_bounds_ok = 0.0
    elif parsed["time_bounds_raw"] is not None:
        time_bounds_ok = 1.0 if parsed["time_bounds"] is not None else 0.0
    else:
        time_bounds_ok = 0.0 if parsed.get("declared_skill") == "bounded" else 1.0

    overall_ok = 1.0 if structure_ok and skill_ok and question_ok and time_bounds_ok else 0.0

    return {
        "format": overall_ok,
        "format_structure": structure_ok,
        "format_skill": skill_ok,
        "format_question": question_ok,
        "format_time_bounds": time_bounds_ok,
    }

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=NUM_REWARD_SERVERS):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def fetch(index, path, endpoint):
    url = f"http://127.0.0.1:{6000+index}/{endpoint}"
    last_exc = None
    for attempt in range(1, REWARD_REQUEST_RETRIES + 1):
        try:
            response = requests.get(url, params={"name": path}, timeout=1800)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= REWARD_REQUEST_RETRIES:
                break
            print(
                f"[reward-fetch] retrying endpoint={endpoint} port={6000+index} "
                f"attempt={attempt}/{REWARD_REQUEST_RETRIES} after error: {exc}"
            )
            time.sleep(REWARD_REQUEST_RETRY_SLEEP_SEC)
    raise last_exc

def generate_results(data, endpoint="hello"):
    n = NUM_REWARD_SERVERS
    datas = split_list(data, n)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(n)]
    for i in range(n):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(fetch, i, random_names[i], endpoint) for i in range(n)]

        for future in as_completed(futures):
            print(future.result())

    for i in range(n):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    for i in range(n):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results


def _shorten_text(text: str, limit: int = 160) -> str:
    text = "" if text is None else str(text).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def difficulty_from_candidates(candidate_scores):
    """d(x, q) = (1/K) * sum_i min(c_i, 1 - c_i) over the per-candidate
    consistency scores c_i (reference doc Section 4a); 0 when no candidate
    was judged."""
    if not candidate_scores:
        return 0.0
    return sum(min(c, 1 - c) for c in candidate_scores) / len(candidate_scores)


def compute_score(predicts: List[str], ground_truths: List[str], questions: List[str], description_answers: List[str], format_weight: float = 0.1, images: Optional[List[str]] = None) -> List[Dict[str, float]]:
    """Questioner reward. Each sample's ground_truth carries its video's graph
    path (the questioner parquet's answer column); images are unused."""
    print("Computing rewards")
    results = []
    format_components = []
    skill_counts = get_recent_skill_counts() if (SKILL_AWARE_ENABLED and SKILL_BALANCE_ENABLED) else Counter()
    for predict, graph_path in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_info = compute_format_components(predict)
        item = match(predict) or {"question": "", "declared_skill": None, "time_bounds": None}
        item["graph_path"] = str(graph_path)
        results.append(item)
        format_components.append(format_info)

    def server_input(item):
        return {
            "question": item["question"],
            "declared_category": item.get("declared_skill"),
            "time_bounds": item.get("time_bounds"),
            "graph_path": item["graph_path"],
        }

    gated_indices = [
        idx for idx, item in enumerate(results)
        if format_components[idx]["format"] == 1.0 and item.get("question")
    ]

    validity_results = [
        {
            "question": item.get("question", ""),
            "declared_category": item.get("declared_skill"),
            "category_match": 0,
            "valid": 0,
            "reason": "skipped_format_gate",
        }
        for item in results
    ]
    if SUPERVISOR_VALIDITY_ENABLED and gated_indices:
        fetched_validity_results = generate_results(
            [server_input(results[idx]) for idx in gated_indices], endpoint="judge_validity"
        )
        for idx, item in zip(gated_indices, fetched_validity_results):
            validity_results[idx] = item
        invalid_examples = [item.get("question", "") for item in validity_results if item.get("valid", 0) != 1][:3]
        print(
            f"[reward] validity enabled: valid={sum(1 for item in validity_results if item.get('valid', 0) == 1)}/"
            f"{len(validity_results)}, invalid_examples={invalid_examples}"
        )
    elif SUPERVISOR_VALIDITY_ENABLED:
        print("[reward] validity enabled: valid=0/0, invalid_examples=[]")

    difficulty_results = [{"question": "", "candidate_scores": [], "num_candidates": 0} for _ in results]
    if gated_indices:
        fetched_difficulty_results = generate_results([server_input(results[idx]) for idx in gated_indices])
        for idx, item in zip(gated_indices, fetched_difficulty_results):
            difficulty_results[idx] = item

    penalties = [0.0 for _ in results]
    if gated_indices:
        penalty_values = duplicate_share_per_problem(
            [duplicate_key(results[idx], results[idx]["graph_path"]) for idx in gated_indices]
        )
        for idx, penalty in zip(gated_indices, penalty_values):
            penalties[idx] = DUPLICATE_PENALTY_WEIGHT * penalty

    scores = []
    for i in range(len(results)):
        format_info = format_components[i]
        valid = 1 if validity_results[i].get("valid", 0) == 1 else 0
        validity_bonus = 0.1 if valid == 1 else 0.0
        declared_skill = results[i].get("declared_skill")
        skill_match = 1 if validity_results[i].get("category_match", 0) == 1 else 0
        skill_balance_bonus = 0.0
        if valid == 1 and SKILL_AWARE_ENABLED and SKILL_BALANCE_ENABLED:
            skill_balance_bonus = compute_skill_balance_bonus(declared_skill, skill_counts)
        penalty = penalties[i]
        candidate_scores = difficulty_results[i].get("candidate_scores") or []
        skill_indicator_metrics = {
            f"skill_{skill}": 1.0 if declared_skill == skill else 0.0
            for skill in ALLOWED_SKILLS
        }

        if format_info["format"] != 1.0:
            difficulty_score = -1.0
            final_score = -1.0
        else:
            difficulty_score = difficulty_from_candidates(candidate_scores) - penalty
            final_score = difficulty_score + validity_bonus + SKILL_BALANCE_WEIGHT * skill_balance_bonus
        score_item = {
            "overall": final_score,
            "format": format_info["format"],
            "format_skill": format_info["format_skill"],
            "format_time_bounds": format_info["format_time_bounds"],
            "validity": valid,
            "skill_match": skill_match,
            "skill_balance_bonus": skill_balance_bonus,
            "difficulty": difficulty_score,
            "penalty": penalty,
            "num_candidates": float(difficulty_results[i].get("num_candidates", 0)),
            "judged_candidates": float(len(candidate_scores)),
        }
        score_item.update(skill_indicator_metrics)
        scores.append(score_item)
    if QUESTIONER_DEBUG_SAMPLES > 0:
        debug_count = min(QUESTIONER_DEBUG_SAMPLES, len(scores))
        print(f"[questioner-debug] showing {debug_count}/{len(scores)} samples")
        for i in range(debug_count):
            item = results[i]
            score_item = scores[i]
            print(
                f"[questioner-debug-{i}] "
                f"category={item.get('declared_skill') or 'unknown'} | "
                f"time_bounds={item.get('time_bounds')} | "
                f"question={_shorten_text(item.get('question', ''))} | "
                f"overall={score_item['overall']:.4f} | "
                f"format={score_item['format']:.1f} | "
                f"validity={score_item['validity']} | "
                f"category_match={score_item['skill_match']} | "
                f"skill_bonus={score_item['skill_balance_bonus']:.4f} | "
                f"candidates={int(score_item['judged_candidates'])}/{int(score_item['num_candidates'])} | "
                f"difficulty={score_item['difficulty']:.4f}"
            )
    return scores
