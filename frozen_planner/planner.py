"""Frozen Planner: C(x, q) = {c_1, ..., c_K}.

Deterministic, non-learned candidate enumeration over a video graph written by
video_graph_builder.build_video_graph. No model calls happen here -- the
Planner only decides *which* segments / segment pairs are worth asking about,
from graph structure alone. Whether a candidate actually answers q is the
Verifier's job, and the Verifier is the only place judgment happens.

Single-event categories yield one candidate per matching segment. Relational
categories yield segment pairs by pure structural enumeration: causal pairs
must share a linked entity, sequential pairs must be close on the timeline,
synchronous pairs must be close in time. Pair plausibility is deliberately
NOT delegated to an LLM.
"""

from dataclasses import dataclass
from typing import Optional

SINGLE_EVENT_CATEGORIES = ("static", "dynamic", "identity", "bounded")
RELATIONAL_CATEGORIES = ("causal", "sequential", "synchronous")
CATEGORIES = SINGLE_EVENT_CATEGORIES + RELATIONAL_CATEGORIES


@dataclass
class PlannerConfig:
    # sequential: pair timeline segments at most this many positions apart.
    window_segments: int = 3
    # causal: pairs must share a linked entity; optionally also cap their
    # timeline distance (None = unbounded). Unbounded, one entity present
    # throughout the video gives ~N^2/2 pairs -- measure_k.py reports it.
    max_entity_gap_segments: Optional[int] = None
    # synchronous: pair segments whose spans overlap or are at most this many
    # seconds apart (timeline segments never overlap, so this is what admits
    # neighbours across a cut).
    sync_tolerance_sec: float = 1.0


class GraphView:
    """Index over one video graph. Tolerates graphs built before the
    scene/action hierarchy (no "level"/"timeline" keys): every segment is then
    treated as both a scene and a finest-level segment.
    """

    def __init__(self, graph):
        self.video_id = graph["video_id"]
        self.segments = {seg["segment_id"]: seg for seg in graph["segments"]}
        self.timeline = graph.get("timeline") or [
            seg["segment_id"] for seg in sorted(graph["segments"], key=lambda s: s["start_sec"])
        ]
        self.position = {segment_id: i for i, segment_id in enumerate(self.timeline)}
        self.scenes = [
            seg["segment_id"] for seg in graph["segments"] if seg.get("level", "scene") == "scene"
        ]

    def entities(self, segment_id):
        return set(self.segments[segment_id].get("entity_ids", []))


def _single_candidate(view, category, segment_id):
    seg = view.segments[segment_id]
    return {
        "category": category,
        "kind": "single",
        "segment_ids": [segment_id],
        "captions": [seg.get("caption", "")],
        "spans": [[seg["start_sec"], seg["end_sec"]]],
        "entity_ids": sorted(view.entities(segment_id)),
        "basis": [],
    }


def _pair_candidate(view, category, first_id, second_id, basis):
    first, second = view.segments[first_id], view.segments[second_id]
    return {
        "category": category,
        "kind": "pair",
        "segment_ids": [first_id, second_id],
        "captions": [first.get("caption", ""), second.get("caption", "")],
        "spans": [[first["start_sec"], first["end_sec"]], [second["start_sec"], second["end_sec"]]],
        "entity_ids": sorted(view.entities(first_id) & view.entities(second_id)),
        "basis": sorted(basis),
    }


def _overlaps(seg, time_bounds):
    lo, hi = time_bounds
    return seg["end_sec"] >= lo and seg["start_sec"] <= hi


def single_event_segment_ids(view, category, time_bounds=None):
    """Which segments are eligible for a single-event category. The mapping is
    structural only:
      static   -> scene nodes (appearance-stable spans from the DINOv2 cuts)
      dynamic  -> finest-level timeline nodes (RAFT motion spans where a
                  scene was split, else the scene itself)
      identity -> timeline nodes that mention at least one linked entity
      bounded  -> timeline nodes overlapping time_bounds=(start_sec, end_sec);
                  every timeline node when q carries no bounds
    """
    if category == "static":
        return list(view.scenes)
    if category == "dynamic":
        return list(view.timeline)
    if category == "identity":
        return [sid for sid in view.timeline if view.entities(sid)]
    if category == "bounded":
        if time_bounds is None:
            return list(view.timeline)
        return [sid for sid in view.timeline if _overlaps(view.segments[sid], time_bounds)]
    raise ValueError(f"not a single-event category: {category}")


def sequential_pairs(view, config):
    """sequential: (earlier, later) timeline pairs at most window_segments apart."""
    timeline = view.timeline
    return {
        (first_id, second_id): {"proximity"}
        for i, first_id in enumerate(timeline)
        for second_id in timeline[i + 1:i + 1 + config.window_segments]
    }


def causal_pairs(view, config):
    """causal: (earlier, later) timeline pairs that share a linked entity,
    optionally at most max_entity_gap_segments apart."""
    pairs = {}
    timeline = view.timeline
    for i, first_id in enumerate(timeline):
        first_entities = view.entities(first_id)
        if not first_entities:
            continue
        stop = len(timeline) if config.max_entity_gap_segments is None else i + 1 + config.max_entity_gap_segments
        for second_id in timeline[i + 1:stop]:
            if first_entities & view.entities(second_id):
                pairs[(first_id, second_id)] = {"shared_entity"}
    return pairs


def synchronous_pairs(view, config):
    """synchronous: timeline pairs whose spans overlap or lie within
    sync_tolerance_sec of each other."""
    pairs = {}
    timeline = view.timeline
    for i, first_id in enumerate(timeline):
        first = view.segments[first_id]
        for second_id in timeline[i + 1:]:
            second = view.segments[second_id]
            # timeline is start-ordered, so once the gap exceeds the tolerance
            # no later segment can come back within it
            if second["start_sec"] - first["end_sec"] > config.sync_tolerance_sec:
                break
            pairs[(first_id, second_id)] = {"proximity"}
    return pairs


PAIR_ENUMERATORS = {
    "causal": causal_pairs,
    "sequential": sequential_pairs,
    "synchronous": synchronous_pairs,
}


def plan(graph, query, config=None):
    """C(x, q). `graph` is one video-graph JSON dict, `query` is
    {"category": one of CATEGORIES, "time_bounds": optional (start_sec, end_sec),
    ...}. The category comes from q -- the Planner does not classify questions,
    since that would be judgment. Returns the candidate list; order is
    deterministic (timeline order).
    """
    config = config or PlannerConfig()
    view = GraphView(graph) if not isinstance(graph, GraphView) else graph
    category = query["category"]

    if category in SINGLE_EVENT_CATEGORIES:
        candidates = [
            _single_candidate(view, category, sid)
            for sid in single_event_segment_ids(view, category, query.get("time_bounds"))
        ]
    elif category in PAIR_ENUMERATORS:
        candidates = [
            _pair_candidate(view, category, a, b, basis)
            for (a, b), basis in PAIR_ENUMERATORS[category](view, config).items()
        ]
    else:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")

    for k, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"{view.video_id}_{category}_c{k:04d}"
    return candidates
