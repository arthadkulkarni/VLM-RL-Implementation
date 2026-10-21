"""Prompts and parsing shared by the questioner (question_generate), the
offline pseudo-labeller (question_evaluate/evaluate.py) and the online reward
server (vllm_service_init/start_vllm_server.py), so all of them describe
videos, categories and candidates with the same text.

  - questioner: generate one query over a video graph's text view, in the
    <category><question>[<time_bounds>] schema cot_val.py rewards;

  - validity: a temperature-0 judge over a text view of the video graph decides
    whether a query is well-posed, grounded, and of its declared category;
  - candidates: one Verifier prompt per Planner candidate c_i, shown through its
    segment montage(s), asking for a binary \\boxed{1}/\\boxed{0} judgment;
  - votes: G sampled judgments per candidate -> consistency c_i and ŷ_i.
"""

import json
import math
import os
import re

from mathruler.grader import extract_boxed_content
from PIL import Image

from frozen_planner.planner import CATEGORIES
from video_graph_builder.montage import stack_montages

CATEGORY_CONTEXTS = {
    "static": (
        "Category `static`: asks about an appearance-stable state of a scene -- what is present, "
        "where, what it looks like -- that holds over a whole scene. "
        "Not static: questions about a change, motion, or an action unfolding."
    ),
    "dynamic": (
        "Category `dynamic`: asks about a single action, motion, or change happening within one span of the video. "
        "Not dynamic: questions relating two separate events."
    ),
    "identity": (
        "Category `identity`: asks about a specific person or object (an entity) and where/when it appears. "
        "The entity must be identifiable across the video."
    ),
    "bounded": (
        "Category `bounded`: asks about what happens within an explicit time window of the video. "
        "The query must state or imply the time window."
    ),
    "negative": (
        "Category `negative`: asks about an event or state that does NOT occur in the video. "
        "It should mention people, objects, or settings the video contains, but the queried event itself is absent, "
        "so no part of the video satisfies it."
    ),
    "causal": (
        "Category `causal`: asks about an event that causes or leads to a later event, where both involve "
        "a shared person or object."
    ),
    "sequential": (
        "Category `sequential`: asks about the order of two events -- what happens before or after another event."
    ),
    "synchronous": (
        "Category `synchronous`: asks about two events that happen at the same time or overlap in time."
    ),
}

IMAGE_PLACEHOLDER = "<|image_pad|>"


def normalize_category(category):
    if category is None:
        return None
    normalized = str(category).strip().lower()
    return normalized if normalized in CATEGORIES else None


def parse_binary(text):
    """Boxed binary judgment -> 1 / 0, or None if absent or not binary.
    (mathruler returns the string 'None' when there is no box.)"""
    boxed = extract_boxed_content(text or "")
    normalized = str(boxed).strip().lower()
    if normalized in {"1", "yes", "true", "valid", "correct"}:
        return 1
    if normalized in {"0", "no", "false", "invalid", "incorrect"}:
        return 0
    return None


def extract_category_match(text, final_valid):
    if not text:
        return final_valid
    match = re.search(r"category\s*match\s*[:：]\s*([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return final_valid


def fmt_time(seconds):
    minutes, secs = divmod(max(0.0, float(seconds)), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"


def evenly_spaced(items, limit):
    """At most `limit` items, evenly spaced and in their original order; limit <= 0 keeps all."""
    if limit <= 0 or len(items) <= limit:
        return list(items)
    return [items[(i * len(items)) // limit] for i in range(limit)]


def process_image_for_vllm(image, max_pixels, min_pixels):
    image.load()
    if image.mode != "RGB":
        image = image.convert("RGB")
    total_pixels = image.width * image.height
    if total_pixels > max_pixels or total_pixels < min_pixels:
        target = max_pixels if total_pixels > max_pixels else min_pixels
        resize_factor = math.sqrt(target / float(total_pixels))
        new_size = (max(1, int(image.width * resize_factor)), max(1, int(image.height * resize_factor)))
        image = image.resize(new_size, resample=Image.LANCZOS)
    return image


# --- Video graph access ---
_GRAPH_CACHE = {}


def load_graph(graph_path):
    if graph_path not in _GRAPH_CACHE:
        with open(graph_path, "r") as f:
            _GRAPH_CACHE[graph_path] = json.load(f)
    return _GRAPH_CACHE[graph_path]


def load_montage(graph_path, segment):
    """A segment's montage; montage_path is relative to the graph's directory."""
    rel_path = segment.get("montage_path")
    if not rel_path:
        raise KeyError(
            f"segment {segment['segment_id']} has no montage_path; rebuild the graph with "
            f"video_graph_builder (--overwrite) so montages are saved"
        )
    path = os.path.join(os.path.dirname(os.path.abspath(graph_path)), rel_path)
    return Image.open(path).convert("RGB")


def build_graph_context(graph, max_segments):
    """Text view of a video graph for the validity judge: timeline captions
    and the linked entities."""
    segments = {seg["segment_id"]: seg for seg in graph["segments"]}
    timeline = graph.get("timeline") or [
        seg["segment_id"] for seg in sorted(graph["segments"], key=lambda s: s["start_sec"])
    ]
    shown = evenly_spaced(timeline, max_segments)
    lines = [f"Video timeline ({len(shown)} of {len(timeline)} segments shown):"]
    for segment_id in shown:
        seg = segments[segment_id]
        lines.append(f"[{fmt_time(seg['start_sec'])}-{fmt_time(seg['end_sec'])}] {seg.get('caption', '').strip()}")
    entities = graph.get("entities", [])
    if entities:
        lines.append(f"Entities ({len(entities)}):")
        for entity in entities[:50]:
            lines.append(
                f"- {entity.get('name', '')} ({entity.get('type', '')}): {entity.get('description', '').strip()} "
                f"[appears in {len(entity.get('segment_ids', []))} segments]"
            )
    return "\n".join(lines)


def build_validity_prompt(question, declared_category, time_bounds, graph_context):
    category_context = CATEGORY_CONTEXTS.get(declared_category, "")
    bounds_line = (
        f"Time bounds: {fmt_time(time_bounds[0])}-{fmt_time(time_bounds[1])}\n" if time_bounds else ""
    )
    return (
        "<|im_start|>system\n"
        "You are a strict video query validity judge. "
        "You are given a text description of a video (a timeline of segment captions and the people/objects in it), "
        "a query about the video, and the query's declared category. "
        "Only decide whether the query is well-posed, grounded in the video description (it refers to people, objects, "
        "or settings the description contains), and whether the declared category matches the query. "
        "Do not answer the query. "
        "Use the following definition for the declared category:\n"
        f"{category_context}\n"
        "You may first give a brief reason, then you must output a line in the form 'Category Match: 1' or 'Category Match: 0'. "
        "Finally, you must put the final decision inside \\boxed{} exactly once at the end. "
        "Output \\boxed{1} only if the query is well-posed, grounded in the video description, and the declared category is correct. "
        "Output \\boxed{0} otherwise.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{graph_context}\n\n"
        f"Query: {question}\n"
        f"Declared Category: {declared_category}\n"
        f"{bounds_line}"
        f"The only valid categories are: {'; '.join(CATEGORIES)}.\n"
        "Judge whether this query is well-posed and grounded in the video description, and whether the declared category "
        "matches the query. If either condition fails, output \\boxed{0}. Do not answer the query. "
        "You may briefly explain why, then output 'Category Match: 1' or 'Category Match: 0', and end with \\boxed{1} or \\boxed{0}.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def candidate_image(graph_path, graph, candidate, max_pixels, min_pixels):
    """A single candidate's montage, or a pair's two montages stacked (A on top)."""
    segments = {seg["segment_id"]: seg for seg in graph["segments"]}
    montages = [load_montage(graph_path, segments[sid]) for sid in candidate["segment_ids"]]
    image = montages[0] if len(montages) == 1 else stack_montages(montages[0], montages[1])
    return process_image_for_vllm(image, max_pixels=max_pixels, min_pixels=min_pixels)


def build_candidate_problem(question, candidate, graph):
    """The Verifier's problem text for one candidate. Stored as the solver
    training row's problem, so training sees exactly what was pseudo-labelled."""
    names = {entity["entity_id"]: entity.get("name", "") for entity in graph.get("entities", [])}
    lines = [f"Query: {question}", f"Query category: {candidate['category']}", "Candidate:"]
    labels = ["Segment A", "Segment B"] if candidate["kind"] == "pair" else ["Segment"]
    for label, (start, end), caption in zip(labels, candidate["spans"], candidate["captions"]):
        lines.append(f"{label} [{fmt_time(start)}-{fmt_time(end)}]: {caption.strip()}")
    shared = [names.get(eid, eid) for eid in candidate.get("entity_ids", [])]
    if shared:
        lines.append(f"Entities: {', '.join(shared)}")
    image_note = (
        "The image shows Segment A's frames (top) and Segment B's frames (bottom), each left to right in time."
        if candidate["kind"] == "pair"
        else "The image shows the segment's frames, left to right in time."
    )
    return (
        "\n".join(lines) + "\n"
        f"{image_note} "
        "Judge whether this candidate satisfies the query. "
        "Output \\boxed{1} if it does and \\boxed{0} if it does not."
    )


def build_candidate_prompt(problem):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{IMAGE_PLACEHOLDER}<|vision_end|>"
        f"{problem} "
        "Please reason step by step carefully based on the image. "
        "After completing your reasoning, you MUST output the final judgment, 1 or 0, "
        "strictly inside \\boxed{}. "
        "The final judgment MUST appear inside \\boxed{}, and nowhere else. "
        "If there is no boxed judgment, your response is considered incorrect.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def tally_votes(texts):
    """G sampled judgments for one candidate -> (votes, c_i) where c_i is the
    fraction of parseable votes that accept, or None if none parse."""
    votes = [vote for vote in (parse_binary(text) for text in texts) if vote is not None]
    if not votes:
        return votes, None
    return votes, sum(votes) / len(votes)


def video_duration(graph):
    return max((seg["end_sec"] for seg in graph["segments"]), default=0.0)


def build_questioner_system_prompt():
    category_lines = "\n".join(f"- {CATEGORY_CONTEXTS[category]}" for category in CATEGORIES)
    return (
        "You are an intelligent Question Generator. Your task is to create a **difficult** query about a video, "
        "given a text description of it: a timeline of timestamped segment captions and the people/objects in it.\n\n"
        "**Requirements (must follow exactly):**\n"
        "1. Read the whole timeline and entity list carefully.\n"
        "2. Generate **exactly one query** whose answer is the part(s) of the video that satisfy it, so it can be "
        "answered by checking segments of the video one at a time.\n"
        "3. Choose the category from **only one** of the following, using these definitions as the decision boundary:\n"
        f"{category_lines}\n"
        "4. The query must be grounded in the video: it refers to people, objects, or settings the description contains. "
        "For `negative`, the queried event itself must not happen anywhere in the video.\n"
        "5. The query must require temporal reasoning, not just restating one caption.\n"
        "6. For `bounded`, give the time window in seconds in a <time_bounds> block, inside the video's duration. "
        "Other categories may omit <time_bounds>; include it only if the query itself restricts the time window.\n"
        "7. Do **not** answer the query, and do **not** add commentary, explanations, or extra text.\n"
        "8. **Output must be strictly in this format, with nothing else:**\n"
        "<category>C</category>\n"
        "<question>Q</question>\n"
        "<time_bounds>START, END</time_bounds>   (seconds; required for `bounded`, otherwise optional)\n\n"
        "**Example of correct output:**\n"
        "<category>sequential</category>\n"
        "<question>What does the man do right after he puts the kettle on the stove?</question>\n\n"
        "**Example of correct output:**\n"
        "<category>bounded</category>\n"
        "<question>Which object does the woman pick up during this part of the video?</question>\n"
        "<time_bounds>12.0, 30.5</time_bounds>"
    )


def build_questioner_user_prompt(graph, max_context_segments):
    return (
        f"Video duration: {video_duration(graph):.1f} seconds.\n"
        f"{build_graph_context(graph, max_context_segments)}\n\n"
        "Generate one new, challenging query about this video. Remember to format the output exactly as instructed."
    )


def parse_time_bounds(text):
    """"start, end" in seconds -> [start, end], or None if malformed."""
    numbers = re.findall(r"\d+(?:\.\d+)?", text or "")
    if len(numbers) != 2:
        return None
    start, end = float(numbers[0]), float(numbers[1])
    return [start, end] if start <= end else None


def parse_questioner_output(text):
    """The last <category>/<question>/<time_bounds> blocks of a questioner
    response; missing fields are None."""
    def last(tag):
        found = re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
        return found[-1].strip() if found else None

    time_bounds_raw = last("time_bounds")
    return {
        "category": normalize_category(last("category")),
        "question": last("question"),
        "time_bounds_raw": time_bounds_raw,
        "time_bounds": parse_time_bounds(time_bounds_raw) if time_bounds_raw is not None else None,
    }
