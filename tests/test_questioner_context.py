"""The questioner / validity-judge view of a video (build_graph_context):
unchanged for videos that fit, hierarchical (overview of every coarse unit +
one focus unit in full) and size-bounded for long ones, with the focus unit
carried from the questioner parquet to the judge as a graph_ref."""

import json
import os
import subprocess
import sys

import pandas as pd

from frozen_planner.planner import coarse_units
from question_evaluate.records import generated_record, structural_query_item
from test_hierarchical_search import long_graph
from video_verifier.prompts import (
    MAX_CONTEXT_CHARS,
    MIN_LONG_VIEW_ENTITIES,
    build_graph_context,
    build_questioner_user_prompt,
    focus_windows,
    graph_ref,
    load_graph,
    needs_focus,
    split_graph_ref,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KITCHEN = os.path.join(os.path.dirname(__file__), "fixtures", "kitchen_graph.json")
SEQUENTIAL = "<category>sequential</category>\n<question>What does the man do after the kettle?</question>"


def test_short_video_keeps_the_flat_timeline():
    graph = load_graph(KITCHEN)
    assert not needs_focus(graph, 120)
    context = build_graph_context(graph, 120)
    assert context.startswith("Video timeline (4 of 4 segments shown):")
    assert build_graph_context(graph, 120, focus_unit=0) == context  # focus is ignored when it all fits
    assert "focus window" not in build_questioner_user_prompt(graph, 120, 0)


def test_graph_ref_round_trip():
    assert split_graph_ref(graph_ref("/a/b.json")) == ("/a/b.json", None)
    assert split_graph_ref(graph_ref("/a/b.json", 7)) == ("/a/b.json", 7)
    assert graph_ref("/a/b.json", 7) == "/a/b.json#focus=7"


def test_long_video_overview_and_focus():
    graph = long_graph(minutes=60)
    units, windows = coarse_units(graph), focus_windows(graph)
    assert needs_focus(graph, 120) and len(windows) >= len(units)
    context = build_graph_context(graph, 120, focus_unit=4)
    for k, unit in enumerate(units):
        assert f"Part {k + 1} [" in context
    captions = {seg["segment_id"]: seg["caption"] for seg in graph["segments"]}
    focus_block = context[context.index("Focus ["):]
    assert all(captions[sid] in focus_block for sid in windows[4]["segment_ids"])
    assert len(context) <= MAX_CONTEXT_CHARS
    assert "Focus:" not in build_graph_context(graph, 120)
    assert "Ground the query in the focus window" in build_questioner_user_prompt(graph, 120, 4)


def test_context_size_is_bounded_regardless_of_length():
    sizes = [len(build_graph_context(long_graph(minutes=m), 120, focus_unit=0)) for m in (30, 60, 120, 180)]
    assert max(sizes) <= MAX_CONTEXT_CHARS


def test_char_budget_shrinks_overview_then_entities_then_focus():
    graph = long_graph(minutes=60)
    for seg in graph["segments"]:
        seg["caption"] = "x" * 150
    overview = lambda text: text.count("\n  [")
    entities = lambda text: text.count("\n- ")
    focus = lambda text: text[text.index("Focus ["):].count("\n[")
    view = lambda max_chars: build_graph_context(graph, 120, focus_unit=2, max_chars=max_chars)
    roomy, parts = view(10**6), len(coarse_units(graph))

    # 1. overview captions go first, down to one per part
    step1 = view(12000)
    assert len(step1) <= 12000 and overview(step1) == parts < overview(roomy)
    assert (entities(step1), focus(step1)) == (entities(roomy), focus(roomy))
    # 2. then the entity list, down to MIN_LONG_VIEW_ENTITIES
    step2 = view(9500)
    assert len(step2) <= 9500 and entities(step2) == MIN_LONG_VIEW_ENTITIES < entities(roomy)
    assert focus(step2) == focus(roomy)
    # 3. then the focus; the floor is returned even over budget
    step3 = view(9000)
    assert len(step3) <= 9000 and focus(step3) < focus(roomy)
    floor = view(1)
    assert (overview(floor), entities(floor), focus(floor)) == (parts, MIN_LONG_VIEW_ENTITIES, 1)


def test_parquet_one_row_per_focus_unit_for_long_videos(tmp_path):
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    (graph_dir / "kitchen.json").write_text(open(KITCHEN).read())
    long = long_graph(minutes=60)
    (graph_dir / "long.json").write_text(json.dumps(long))
    out = tmp_path / "q.parquet"
    subprocess.run(
        [sys.executable, "video_graph_builder/make_questioner_parquet.py", "--graph_dir", str(graph_dir),
         "--output", str(out)],
        cwd=REPO, env={**os.environ, "PYTHONPATH": REPO}, check=True, capture_output=True,
    )
    refs = [split_graph_ref(ref) for ref in pd.read_parquet(out)["answer"]]
    assert (str(graph_dir / "kitchen.json"), None) in refs
    assert sorted(f for p, f in refs if p.endswith("long.json")) == list(range(len(focus_windows(long))))


def test_focus_unit_reaches_the_validity_item():
    record = generated_record(SEQUENTIAL, KITCHEN, focus_unit=3)
    query_item, reason = structural_query_item(record)
    assert reason is None and query_item["focus_unit"] == 3
    assert structural_query_item(generated_record(SEQUENTIAL, KITCHEN))[0]["focus_unit"] is None
