"""Frozen Planner: C(x, q) = {c_1, ..., c_K}.

Deterministic, non-learned candidate enumeration over a video graph written by
video_graph_builder.build_video_graph. No model calls happen here -- the
Planner only decides *which* segments / segment pairs are worth asking about,
from graph structure alone. Whether a candidate actually answers q is the
Verifier's job, and the Verifier is the only place judgment happens.

Single-event categories yield one candidate per matching segment. Relational
categories yield segment pairs by pure structural enumeration: causal pairs
must share a linked entity (each appearance paired with that entity's next
few appearances), sequential pairs must be close on the timeline,
synchronous pairs must be close in time. Pair plausibility is deliberately
NOT delegated to an LLM.
"""

from dataclasses import dataclass

SINGLE_EVENT_CATEGORIES = ("static", "dynamic", "identity", "bounded", "negative")
RELATIONAL_CATEGORIES = ("causal", "sequential", "synchronous")
CATEGORIES = SINGLE_EVENT_CATEGORIES + RELATIONAL_CATEGORIES


@dataclass
class PlannerConfig:
    # sequential: pair timeline segments at most this many positions apart.
    window_segments: int = 3
    # causal: pair each appearance of a linked entity with that entity's next
    # this-many appearances on the timeline. Keeps K linear in video length
    # (<= this * total appearances) while still pairing an entity's last
    # appearance before it leaves with its first after it returns, however
    # far apart. Pairing all appearances instead is ~N^2/2 for one entity
    # present throughout (258,840 pairs for 720 segments).
    causal_next_appearances: int = 2
    # synchronous: pair segments whose spans overlap or are at most this many
    # seconds apart (timeline segments never overlap, so this is what admits
    # neighbours across a cut).
    sync_tolerance_sec: float = 1.0
    # Coarse-to-fine search (coarse_units / within_units): the timeline is cut
    # into consecutive windows at least coarse_min_sec long, widened so a video
    # has at most max_coarse_units of them. The coarse pass then costs
    # O(max_coarse_units) Verifier calls however long the video is.
    coarse_min_sec: float = 60.0
    max_coarse_units: int = 32


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
    """Positive-length overlap with [lo, hi] -- a segment that only touches a
    bound is outside it. A point bound (lo == hi) selects the segment
    containing that instant (start inclusive)."""
    lo, hi = time_bounds
    if lo == hi:
        return seg["start_sec"] <= lo < seg["end_sec"]
    return seg["end_sec"] > lo and seg["start_sec"] < hi


def single_event_segment_ids(view, category, time_bounds=None):
    """Which segments are eligible for a single-event category. The mapping is
    structural only:
      static   -> scene nodes (appearance-stable spans from the DINOv2 cuts)
      dynamic  -> finest-level timeline nodes (RAFT motion spans where a
                  scene was split, else the scene itself)
      identity -> timeline nodes that mention at least one linked entity
      bounded  -> timeline nodes overlapping time_bounds=(start_sec, end_sec);
                  every timeline node when q carries no bounds
      negative -> every timeline node; q asserts something absent, so the
                  Verifier is expected to reject all of them
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
    if category == "negative":
        return list(view.timeline)
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
    where the later segment is among that entity's next
    causal_next_appearances appearances after the earlier one."""
    appearances = {}
    for segment_id in view.timeline:
        for entity_id in sorted(view.entities(segment_id)):
            appearances.setdefault(entity_id, []).append(segment_id)

    pairs = {}
    for segment_ids in appearances.values():
        for i, first_id in enumerate(segment_ids):
            for second_id in segment_ids[i + 1:i + 1 + config.causal_next_appearances]:
                pairs[(first_id, second_id)] = {"shared_entity"}
    # timeline order, so candidate ids stay deterministic and time-sorted
    return dict(sorted(pairs.items(), key=lambda item: (view.position[item[0][0]], view.position[item[0][1]])))


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


def coarse_units(graph, config=None):
    """Coarse level of the coarse-to-fine search: consecutive runs of timeline
    segments, each spanning at least unit_sec, where unit_sec =
    max(coarse_min_sec, duration / max_coarse_units). Cuts fall on timeline
    segment boundaries, so every finest-level segment is in exactly one unit; a
    short remainder is merged into the last unit. Structural only, like the
    rest of the Planner -- which units are worth expanding is the Verifier's
    call. Returns [{"unit_id", "segment_ids", "span": [start_sec, end_sec]}] in
    timeline order.
    """
    config = config or PlannerConfig()
    view = GraphView(graph) if not isinstance(graph, GraphView) else graph
    if not view.timeline:
        return []
    segments = [view.segments[sid] for sid in view.timeline]
    duration = segments[-1]["end_sec"] - segments[0]["start_sec"]
    unit_sec = max(config.coarse_min_sec, duration / max(1, config.max_coarse_units))

    # Close a unit once it spans unit_sec: every unit but the last is then at
    # least unit_sec long, so there are at most max_coarse_units of them.
    groups = [[]]
    for seg in segments:
        if groups[-1] and groups[-1][-1]["end_sec"] - groups[-1][0]["start_sec"] >= unit_sec:
            groups.append([])
        groups[-1].append(seg)
    if len(groups) > 1 and groups[-1][-1]["end_sec"] - groups[-1][0]["start_sec"] < unit_sec / 2:
        groups[-2].extend(groups.pop())

    return [
        {
            "unit_id": f"{view.video_id}_unit{k:03d}",
            "segment_ids": [seg["segment_id"] for seg in group],
            "span": [group[0]["start_sec"], group[-1]["end_sec"]],
        }
        for k, group in enumerate(groups)
    ]


def within_units(candidates, units):
    """Fine level of the coarse-to-fine search: the candidates of C(x, q) whose
    every span overlaps one of `units` (the coarse units the Verifier
    accepted). A pair needs both segments covered, so a sequential or causal
    pair whose events fall in two different accepted units is kept. Overlap
    (not containment) also keeps scene nodes, which can straddle a unit cut.
    """
    spans = [unit["span"] for unit in units]
    return [
        candidate for candidate in candidates
        if all(any(_overlaps({"start_sec": s, "end_sec": e}, span) for span in spans) for s, e in candidate["spans"])
    ]

