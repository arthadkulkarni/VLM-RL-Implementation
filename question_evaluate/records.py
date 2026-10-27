"""The rows each stage of question generation -> evaluation -> upload writes,
as pure functions so the declared category's path through all three can be
tested without vLLM (tests/test_category_pipeline.py).

  question_generate.py  generated_record      response text -> generated query
  evaluate.py           structural_query_item generated query -> Planner-checked query
                        candidate_result_row  query + voted candidate -> result row
  upload.py             upload_row            result row -> solver training row
                        apply_skill_balanced_sampling
"""

from collections import Counter

from frozen_planner.planner import plan
from video_verifier.prompts import load_graph, normalize_category, parse_questioner_output


# Query categories (the Frozen Planner's taxonomy); must match ALLOWED_SKILLS in
# train_examples/reward_function/cot_val.py, which reads the counts upload.py writes.
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


def generated_record(response, graph_path, focus_unit=None):
    """One questioner completion. focus_unit is the focus window a long
    video's prompt showed in full (None otherwise); evaluate.py's validity
    judge is shown the same view."""
    parsed = parse_questioner_output(response)
    return {
        "declared_category": parsed["category"] or "unknown",
        # Empty when unparseable; evaluate.py drops empty questions.
        "question": parsed["question"] or "",
        "time_bounds": parsed["time_bounds"],
        "graph_path": graph_path,
        "focus_unit": focus_unit,
        "raw_response": response,
    }


def structural_query_item(item):
    """Structural validity: category parses, graph loads, the Planner finds
    candidates. Returns (query_item, reject reason or None), or None for an
    empty question."""
    question = item.get("question", "")
    if not question:
        return None
    category = normalize_category(item.get("declared_category"))
    query_item = {
        "question": question,
        "question_type": item.get("question_type", ""),
        "declared_category": category or str(item.get("declared_category") or "unknown"),
        "time_bounds": item.get("time_bounds"),
        "graph_path": item.get("graph_path", ""),
        "focus_unit": item.get("focus_unit"),
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
            print(f"WARNING: planning failed for '{question[:50]}...': {exc}")
            reason = "skipped_graph_or_plan_failed"
    if reason:
        query_item["validity_reason"] = reason
    return query_item, reason


def candidate_result_row(query_item, candidate, problem, pseudo_label, score, votes, image_b64):
    return {
        "question": problem,
        "answer": str(pseudo_label),
        "score": score,
        "image": image_b64,
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
    }


def normalize_skill_label(skill):
    if skill is None:
        return None
    normalized = str(skill).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return SKILL_ALIASES.get(normalized)


def upload_row(data):
    return {
        'problem': data['question'],
        'answer': data['answer'],
        'score': data['score'],
        'images': data.get('image', ''),
        'problem_type': data.get('question_type', 'unknown'),
        'declared_skill': normalize_skill_label(data.get('declared_skill')) or str(data.get('declared_skill', 'unknown')),
        # evaluate.py writes the judge's verdict as category_match.
        'skill_match': int(data.get('category_match', 1)),
        'valid': int(data.get('valid', 1)),
    }


def count_skills(rows, key):
    counts = Counter()
    for row in rows:
        normalized = normalize_skill_label(row.get(key))
        if normalized:
            counts[normalized] += 1
        else:
            counts["unknown"] += 1
    return counts


def apply_skill_balanced_sampling(rows, target_size, rng):
    if not rows or target_size <= 0:
        return rows, {
            "requested_target": target_size,
            "per_skill_quota": 0,
            "selected_before_backfill": 0,
            "backfill_count": 0,
        }

    per_skill_quota = max(target_size // len(ALLOWED_SKILLS), 0)
    selected = []
    selected_ids = set()

    skill_to_rows = {skill: [] for skill in ALLOWED_SKILLS}
    unknown_rows = []
    for idx, row in enumerate(rows):
        normalized = normalize_skill_label(row.get("declared_skill"))
        if normalized in ALLOWED_SKILLS:
            skill_to_rows[normalized].append((idx, row))
        else:
            unknown_rows.append((idx, row))

    for skill in ALLOWED_SKILLS:
        bucket = list(skill_to_rows[skill])
        rng.shuffle(bucket)
        chosen = bucket[:per_skill_quota]
        selected.extend(row for _, row in chosen)
        selected_ids.update(idx for idx, _ in chosen)

    selected_before_backfill = len(selected)
    if len(selected) < min(target_size, len(rows)):
        remaining = [(idx, row) for idx, row in enumerate(rows) if idx not in selected_ids]
        rng.shuffle(remaining)
        need = min(target_size, len(rows)) - len(selected)
        chosen_backfill = remaining[:need]
        selected.extend(row for _, row in chosen_backfill)
    else:
        chosen_backfill = []

    return selected, {
        "requested_target": target_size,
        "per_skill_quota": per_skill_quota,
        "selected_before_backfill": selected_before_backfill,
        "backfill_count": len(chosen_backfill),
    }
