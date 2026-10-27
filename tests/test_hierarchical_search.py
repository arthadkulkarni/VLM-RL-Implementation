"""Coarse-to-fine candidate search: frozen_planner.coarse_units cuts the
timeline into a bounded number of windows, within_units keeps the fine
candidates inside the windows the Verifier accepted, and the Verifier's
coarse-unit view (prompt text, stacked montages) covers the whole window."""

from PIL import Image

from frozen_planner.planner import CATEGORIES, PlannerConfig, coarse_units, plan, within_units
from video_graph_builder.montage import GUTTER, LABEL_HEIGHT, stack_montage_rows, stack_montages
from video_verifier.prompts import build_coarse_problem, coarse_unit_image


def long_graph(minutes, segments_per_scene=3, segment_sec=6.0):
    """A synthetic video: scenes of `segments_per_scene` action segments, each
    `segment_sec` long (~10 timeline segments per minute, like the real 5-min
    graphs), with one entity that recurs every scene and one per scene."""
    segments, timeline, entities = [], [], []
    recurring = {"entity_id": "ent_r", "name": "man", "type": "person", "description": "", "segment_ids": []}
    t, scene = 0.0, 0
    while t < minutes * 60:
        scene_id = f"long_scene{scene:04d}"
        scene_seg = {
            "segment_id": scene_id, "level": "scene", "parent_id": None, "child_ids": [],
            "start_sec": t, "end_sec": t + segments_per_scene * segment_sec,
            "caption": f"scene {scene}", "entity_ids": ["ent_r"], "montage_path": "m.png",
        }
        segments.append(scene_seg)
        local = {"entity_id": f"ent_{scene}", "name": f"cup {scene}", "type": "object", "description": "",
                 "segment_ids": []}
        for j in range(segments_per_scene):
            action_id = f"{scene_id}_act{j:02d}"
            segments.append({
                "segment_id": action_id, "level": "action", "parent_id": scene_id, "child_ids": [],
                "start_sec": t, "end_sec": t + segment_sec, "caption": f"action {scene}.{j}",
                "entity_ids": ["ent_r", local["entity_id"]], "montage_path": "m.png",
            })
            scene_seg["child_ids"].append(action_id)
            timeline.append(action_id)
            recurring["segment_ids"].append(action_id)
            local["segment_ids"].append(action_id)
            t += segment_sec
        entities.append(local)
        scene += 1
    return {"video_id": "long", "segments": segments, "timeline": timeline, "entities": [recurring] + entities}


def test_units_partition_the_timeline_in_order():
    graph = long_graph(minutes=10)
    units = coarse_units(graph)
    assert [sid for unit in units for sid in unit["segment_ids"]] == graph["timeline"]
    for unit in units[:-1]:
        assert unit["span"][1] - unit["span"][0] >= 60.0
    for earlier, later in zip(units, units[1:]):
        assert earlier["span"][1] <= later["span"][0]


def test_unit_count_stays_bounded_on_long_videos():
    for minutes, expected_unit_sec in ((5, 60.0), (60, 112.5), (120, 225.0)):
        graph = long_graph(minutes=minutes)
        units = coarse_units(graph)
        assert len(units) <= 32
        # widened to duration / max_coarse_units once that exceeds coarse_min_sec; a unit closes
        # on the first segment that reaches unit_sec, and a short remainder joins the last one
        lengths = [u["span"][1] - u["span"][0] for u in units]
        assert min(lengths[:-1]) >= expected_unit_sec
        assert max(lengths) < 1.5 * expected_unit_sec + 2 * 6.0
    assert len(coarse_units(long_graph(minutes=120), PlannerConfig(max_coarse_units=8))) <= 8


def test_short_remainder_merges_into_last_unit():
    graph = long_graph(minutes=2, segments_per_scene=1, segment_sec=10.0)  # 120 s + nothing
    graph["timeline"].append("tail")
    graph["segments"].append({"segment_id": "tail", "level": "scene", "start_sec": 120.0, "end_sec": 125.0,
                              "caption": "", "entity_ids": []})
    units = coarse_units(graph)
    assert len(units) == 2
    assert units[-1]["segment_ids"][-1] == "tail"


def test_within_units_all_or_none():
    graph = long_graph(minutes=10)
    units = coarse_units(graph)
    for category in CATEGORIES:
        candidates = plan(graph, {"category": category})
        assert within_units(candidates, units) == candidates
        assert within_units(candidates, []) == []


def test_within_units_restricts_to_accepted_windows():
    graph = long_graph(minutes=10)
    units = coarse_units(graph)
    accepted = [units[3]]
    lo, hi = accepted[0]["span"]

    dynamic = within_units(plan(graph, {"category": "dynamic"}), accepted)
    assert [c["segment_ids"][0] for c in dynamic] == accepted[0]["segment_ids"]

    # Scene nodes may straddle a unit cut; overlap keeps them.
    static = within_units(plan(graph, {"category": "static"}), accepted)
    assert static and all(c["spans"][0][1] > lo and c["spans"][0][0] < hi for c in static)

    # A pair needs both segments in accepted units ...
    sequential = within_units(plan(graph, {"category": "sequential"}), accepted)
    assert sequential and all(set(c["segment_ids"]) <= set(accepted[0]["segment_ids"]) for c in sequential)
    # ... and is kept when its two segments fall in two different accepted units.
    two = within_units(plan(graph, {"category": "sequential"}), [units[3], units[4]])
    assert any(c["segment_ids"][0] in units[3]["segment_ids"] and c["segment_ids"][1] in units[4]["segment_ids"]
               for c in two)


def test_hierarchy_cuts_judged_candidates_on_long_videos():
    """The point of the search: on a 1-hour video, two accepted windows leave
    far fewer fine candidates than C(x, q), and all of them are in the
    windows the Verifier picked (not evenly spread over the hour)."""
    graph = long_graph(minutes=60)
    units = coarse_units(graph)
    for category in ("dynamic", "sequential", "causal", "synchronous"):
        candidates = plan(graph, {"category": category})
        kept = within_units(candidates, units[10:12])
        assert 0 < len(kept) < len(candidates) / 10


def test_coarse_problem_lists_every_segment():
    graph = long_graph(minutes=5)
    unit = coarse_units(graph)[1]
    problem = build_coarse_problem("When does the man pick up the cup?", "sequential", unit, graph)
    captions = {seg["segment_id"]: seg["caption"] for seg in graph["segments"]}
    for sid in unit["segment_ids"]:
        assert captions[sid] in problem
    assert "\\boxed{1}" in problem and "Query category: sequential" in problem


def test_coarse_unit_image_stacks_a_few_montages(tmp_path):
    graph = long_graph(minutes=5)
    Image.new("RGB", (300, 60), (10, 20, 30)).save(tmp_path / "m.png")
    graph_path = str(tmp_path / "long.json")
    unit = coarse_units(graph)[0]
    assert len(unit["segment_ids"]) > 3
    image = coarse_unit_image(graph_path, graph, unit, max_segments=3, max_pixels=10**7, min_pixels=1)
    # 3 rows of 60 px, each under a label, rows separated by a gutter (resized to multiples of 28 by
    # process_image_for_vllm, so compare loosely).
    assert abs(image.height - (3 * (60 + LABEL_HEIGHT) + 2 * GUTTER)) < 28


def test_stack_montages_unchanged_for_pairs():
    top, bottom = Image.new("RGB", (200, 50)), Image.new("RGB", (150, 70))
    pair = stack_montages(top, bottom)
    assert pair.size == (200, 50 + 70 + 2 * LABEL_HEIGHT + GUTTER)
    assert stack_montage_rows([top, bottom], ["Segment A", "Segment B"]).size == pair.size


def test_adaptive_stops_only_on_unanimous_parsed_votes():
    from video_verifier.prompts import needs_more_votes

    yes, no, junk = "so \\boxed{1}", "so \\boxed{0}", "no box here"
    assert not needs_more_votes([no] * 4, 4)
    assert not needs_more_votes([yes] * 4, 4)
    assert needs_more_votes([no, no, yes, no], 4)  # split
    assert needs_more_votes([no, no, no, junk], 4)  # one unparsed: not proven unanimous
    assert needs_more_votes([junk] * 4, 4)
