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
  - coarse units: for long videos, one Verifier prompt per Planner coarse unit
    (a window of consecutive segments) asking whether any part of the query
    happens there; only accepted units' candidates are judged;
  - votes: G sampled judgments per candidate -> consistency c_i and ŷ_i.
"""

import json
import math
import os
import re

from mathruler.grader import extract_boxed_content
from PIL import Image

from frozen_planner.planner import CATEGORIES, PlannerConfig, coarse_units
from video_graph_builder.montage import stack_montage_rows, stack_montages

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


FOCUS_SEP = "#focus="


# Focus windows of the hierarchical view: >= 60 s, at most 96 per video, so a
# window stays ~20 segments even at 3 hours and fits the view's focus budget.
# Finer than the overview's parts (the Planner's default coarse units, <= 32).
FOCUS_CONFIG = PlannerConfig(coarse_min_sec=60.0, max_coarse_units=96)


def focus_windows(graph):
    """The windows a long video's questioner prompt can show in full; a
    focus_unit indexes this list."""
    return coarse_units(graph, FOCUS_CONFIG)


def graph_ref(graph_path, focus_unit=None):
    """A questioner sample's video: the graph path, plus the focus window
    (focus_windows) its prompt shows in full when the video is too long to show
    whole. Carried as the questioner parquet's answer / reward ground_truth
    string."""
    return graph_path if focus_unit is None else f"{graph_path}{FOCUS_SEP}{int(focus_unit)}"


def split_graph_ref(ref):
    """graph_ref's inverse: -> (graph_path, focus_unit or None)."""
    ref = str(ref)
    if FOCUS_SEP in ref:
        path, focus = ref.rsplit(FOCUS_SEP, 1)
        return path, int(focus)
    return ref, None


def _timeline(graph):
    return graph.get("timeline") or [seg["segment_id"] for seg in sorted(graph["segments"], key=lambda s: s["start_sec"])]


def needs_focus(graph, max_segments):
    """Whether build_graph_context shows this video hierarchically (its
    timeline exceeds max_segments) -- then each prompt has a focus unit."""
    return 0 < max_segments < len(_timeline(graph))


# Character budget of the hierarchical view. Captions run ~3.4 characters per
# Qwen3-VL token, so 20000 is ~6k tokens; with the questioner system prompt
# the whole prompt measured <= ~6.9k tokens on 1-3 hour videos, under
# data.max_prompt_length = 8192, past which a prompt is silently dropped.
MAX_CONTEXT_CHARS = int(os.getenv("RISE_MAX_CONTEXT_CHARS", "20000"))
# Long-video view only: captions clipped to CLIP_CHARS; the entity list keeps
# the LONG_VIEW_ENTITIES most frequent, descriptions clipped to
# ENTITY_CLIP_CHARS, leaving room for the overview's captions.
CLIP_CHARS = 200
LONG_VIEW_ENTITIES = 40
# ... and keeps at least this many when the budget is tight.
MIN_LONG_VIEW_ENTITIES = 15
ENTITY_CLIP_CHARS = 100


def _clip(text, limit):
    text = (text or "").strip()
    return text if limit is None or len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _entity_lines(entities, clip=None):
    return [
        f"- {entity.get('name', '')} ({entity.get('type', '')}): {_clip(entity.get('description', ''), clip)} "
        f"[appears in {len(entity.get('segment_ids', []))} segments]"
        for entity in entities
    ]


def build_graph_context(graph, max_segments, focus_unit=None, max_chars=None):
    """Text view of a video graph, shared by the questioner prompt and the
    validity judge so both see the same video.

    A timeline of at most max_segments segments is listed whole. A longer one
    is shown hierarchically: an overview of the Planner's coarse units
    (frozen_planner coarse_units, default PlannerConfig) with a few evenly
    spaced captions each, and one focus window (focus_windows[focus_unit])
    listed segment by segment. Both
    together use at most max_segments caption lines (the focus unit at most
    half) and at most max_chars characters (default MAX_CONTEXT_CHARS): the
    overview gives up captions first, then entities, then the focus, so the prompt stays the
    same size however long the video is. focus_unit=None spreads every line
    over the overview.
    """
    segments = {seg["segment_id"]: seg for seg in graph["segments"]}
    timeline = _timeline(graph)
    entities = graph.get("entities", [])

    def caption_line(sid, clip=None):
        seg = segments[sid]
        return f"[{fmt_time(seg['start_sec'])}-{fmt_time(seg['end_sec'])}] {_clip(seg.get('caption', ''), clip)}"

    if not needs_focus(graph, max_segments):
        shown = evenly_spaced(timeline, max_segments)
        lines = [f"Video timeline ({len(shown)} of {len(timeline)} segments shown):"]
        lines += [caption_line(sid) for sid in shown]
        if entities:
            lines.append(f"Entities ({len(entities)}):")
            lines += _entity_lines(entities[:50])
        return "\n".join(lines)

    max_chars = MAX_CONTEXT_CHARS if max_chars is None else max_chars
    units = coarse_units(graph)
    windows = focus_windows(graph)
    focus = windows[focus_unit] if focus_unit is not None and 0 <= focus_unit < len(windows) else None
    # Most widespread first: on a long video these are the recurring people and objects.
    ranked = sorted(entities, key=lambda e: -len(e.get("segment_ids", [])))

    # Part summaries from the graph build (video_graph_builder/part_summaries.py),
    # matched to today's units by their segments; a part without one (older
    # graph, failed summary) falls back to captions.
    summaries = {
        tuple(part.get("segment_ids", [])): part.get("summary", "").strip() for part in graph.get("parts") or []
    }
    unit_summary = [summaries.get(tuple(unit["segment_ids"]), "") for unit in units]

    def render(per_unit, focus_lines, n_entities):
        """per_unit lines per part: its summary (if any) counts as one, the
        rest are evenly spaced captions."""
        focus_shown = evenly_spaced(focus["segment_ids"], focus_lines) if focus else []
        if any(unit_summary):
            shows = "a summary" + (" and a few segment captions" if per_unit > 1 else "")
        else:
            shows = f"{'a few' if per_unit > 1 else 'one'} of its segment captions"
        lines = [
            f"Video overview: {len(timeline)} segments in {len(units)} parts. Each part shows {shows}"
            + ("; the focus window is shown in full detail below." if focus else ".")
        ]
        for k, unit in enumerate(units):
            start, end = unit["span"]
            lines.append(f"Part {k + 1} [{fmt_time(start)}-{fmt_time(end)}], {len(unit['segment_ids'])} segments:")
            captions = per_unit
            if unit_summary[k]:
                lines.append(f"  Summary: {unit_summary[k]}")
                captions -= 1
            if captions > 0:  # evenly_spaced treats a limit of 0 as "keep all"
                lines += [f"  {caption_line(sid, CLIP_CHARS)}" for sid in evenly_spaced(unit["segment_ids"], captions)]
        if focus:
            start, end = focus["span"]
            lines.append(
                f"Focus [{fmt_time(start)}-{fmt_time(end)}] "
                f"({len(focus_shown)} of {len(focus['segment_ids'])} segments shown):"
            )
            lines += [caption_line(sid, CLIP_CHARS) for sid in focus_shown]
        if entities:
            lines.append(f"Entities ({len(entities)}, most frequent first):")
            lines += _entity_lines(ranked[:n_entities], ENTITY_CLIP_CHARS)
        return "\n".join(lines)

    focus_lines = min(len(focus["segment_ids"]), max_segments // 2) if focus else 0
    per_unit = max(1, (max_segments - focus_lines) // len(units))
    n_entities = min(len(ranked), LONG_VIEW_ENTITIES)
    text = render(per_unit, focus_lines, n_entities)
    while len(text) > max_chars:
        if per_unit > 1:
            per_unit -= 1
        elif n_entities > MIN_LONG_VIEW_ENTITIES:
            n_entities = max(MIN_LONG_VIEW_ENTITIES, n_entities - 5)
        elif focus_lines > 1:
            focus_lines -= 1
        else:
            break
        text = render(per_unit, focus_lines, n_entities)
    return text


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


def coarse_unit_image(graph_path, graph, unit, max_segments, max_pixels, min_pixels):
    """A coarse unit's view: the montages of up to max_segments of its
    segments, evenly spaced, stacked in time order."""
    segments = {seg["segment_id"]: seg for seg in graph["segments"]}
    shown = [segments[sid] for sid in evenly_spaced(unit["segment_ids"], max_segments)]
    image = stack_montage_rows(
        [load_montage(graph_path, seg) for seg in shown],
        [f"[{fmt_time(seg['start_sec'])}-{fmt_time(seg['end_sec'])}]" for seg in shown],
        max_pixels=max_pixels,
    )
    return process_image_for_vllm(image, max_pixels=max_pixels, min_pixels=min_pixels)


def build_coarse_problem(question, category, unit, graph):
    """The Verifier's problem text for one coarse unit: does any event or
    moment the query is about happen in this window? A relational query's two
    events may fall in different windows, so a window showing either one
    counts. Accepted units are expanded into their fine candidates."""
    segments = {seg["segment_id"]: seg for seg in graph["segments"]}
    start, end = unit["span"]
    lines = [
        f"Query: {question}",
        f"Query category: {category}",
        f"Video window [{fmt_time(start)}-{fmt_time(end)}], segment by segment:",
    ]
    for sid in unit["segment_ids"]:
        seg = segments[sid]
        lines.append(f"[{fmt_time(seg['start_sec'])}-{fmt_time(seg['end_sec'])}] {seg.get('caption', '').strip()}")
    return (
        "\n".join(lines) + "\n"
        "The image shows frames from some of these segments, each row left to right in time, rows in time order. "
        "Judge whether any event or moment the query is about -- all of it, or one of the events it relates -- "
        "happens in this window. "
        "Output \\boxed{1} if it does and \\boxed{0} if it does not."
    )


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


def needs_more_votes(texts, initial_samples):
    """Adaptive sampling's stopping rule: after the first initial_samples
    judgments of a candidate, stop only if every one parsed and they all agree
    -- the candidate is then taken as unanimous (c_i = 0 or 1). Otherwise the
    rest of the G samples are drawn. A split candidate can agree by chance in
    its first few votes, so this trades some accuracy in d(x, q) for cost."""
    votes, _ = tally_votes(texts)
    return len(votes) < initial_samples or len(set(votes)) > 1


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


def build_questioner_user_prompt(graph, max_context_segments, focus_unit=None):
    """The questioner's user turn. For a long video (see build_graph_context)
    the query should be grounded in the focus part, and may relate it to
    events elsewhere in the overview."""
    focus_note = (
        "Ground the query in the focus window; it may relate an event there to one elsewhere in the overview. "
        if needs_focus(graph, max_context_segments) and focus_unit is not None
        else ""
    )
    return (
        f"Video duration: {video_duration(graph):.1f} seconds.\n"
        f"{build_graph_context(graph, max_context_segments, focus_unit)}\n\n"
        f"{focus_note}"
        "Generate one new, challenging query about this video. Remember to format the output exactly as instructed."
    )


def parse_time_bounds(text):
    """"start, end" -> [start, end] in seconds, or None if malformed. Each time
    is seconds or [H:]MM:SS(.s), the format the timeline in the prompt uses."""
    times = re.findall(r"\d+(?::\d+){0,2}(?:\.\d+)?", text or "")
    if len(times) != 2:
        return None
    start, end = (sum(float(part) * 60 ** i for i, part in enumerate(reversed(t.split(":")))) for t in times)
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
