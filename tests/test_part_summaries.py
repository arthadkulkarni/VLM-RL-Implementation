"""Per-part summaries (video_graph_builder/part_summaries.py) and their use
in the long-video overview: one request per Planner coarse unit, stored as
graph["parts"], shown in place of sampled captions when the budget is tight,
and ignored where they no longer match the units or the video is short."""

import copy

import video_graph_builder.graph_client as graph_client
from frozen_planner.planner import coarse_units
from test_hierarchical_search import long_graph
from video_graph_builder.part_summaries import (
    MAX_PART_ENTITIES,
    MAX_SUMMARY_CHARS,
    attach_parts,
    build_summary_prompt,
    parse_summary,
    part_requests,
    summarize_graphs,
)
from video_verifier.prompts import build_graph_context, load_graph
from test_questioner_context import KITCHEN


def summarized(graph):
    requests = part_requests(graph)
    return attach_parts(graph, requests, {r["part_id"]: f"summary of {r['part_id']}" for r in requests})


def test_one_request_per_unit_in_order():
    graph = long_graph(minutes=30)
    requests, units = part_requests(graph), coarse_units(graph)
    assert [r["segment_ids"] for r in requests] == [u["segment_ids"] for u in units]
    captions = {s["segment_id"]: s["caption"] for s in graph["segments"]}
    first = requests[0]
    assert [c for _, _, c in first["captions"]] == [captions[sid] for sid in first["segment_ids"]]
    # the recurring entity is named first, and names are capped
    assert first["entity_names"][0] == "man" and len(first["entity_names"]) <= MAX_PART_ENTITIES


def test_summary_prompt_and_parse():
    request = part_requests(long_graph(minutes=10))[1]
    prompt = build_summary_prompt(request)
    assert all(caption in prompt for _, _, caption in request["captions"])
    assert "Recurring people/objects in this part: man" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert parse_summary("  The man\n walks.  ") == "The man walks."
    assert len(parse_summary("x" * 1000)) == MAX_SUMMARY_CHARS


def test_summarize_graphs_batches_all_parts(monkeypatch):
    graphs = [long_graph(minutes=10), long_graph(minutes=20)]
    sent = []

    def fake_batch(items, num_servers, base_port=None):
        sent.extend(items)
        return [{"part_id": item["part_id"], "summary": f"s-{item['part_id']}"} for item in items]

    monkeypatch.setattr(graph_client, "summarize_parts_batch", fake_batch)
    n = summarize_graphs(graphs, num_servers=2)
    assert n == len(sent) == sum(len(coarse_units(g)) for g in graphs)
    assert all(part["summary"] == f"s-{part['part_id']}" for g in graphs for part in g["parts"])


def test_overview_shows_summaries():
    graph = summarized(long_graph(minutes=60))
    context = build_graph_context(graph, 120, focus_unit=3)
    assert context.count("  Summary: ") == len(coarse_units(graph))
    assert "Each part shows a summary" in context


def test_tight_budget_keeps_summaries_and_drops_captions():
    graph = summarized(long_graph(minutes=60))
    for seg in graph["segments"]:
        seg["caption"] = "x" * 150
    tight = build_graph_context(graph, 120, focus_unit=3, max_chars=1)
    # floor: one line per part, which for a summarized part is its summary alone
    assert tight.count("  Summary: ") == len(coarse_units(graph))
    assert tight.count("\n  [") == 0


def test_stale_or_missing_summary_falls_back_to_captions():
    graph = summarized(long_graph(minutes=60))
    graph["parts"][0]["segment_ids"] = graph["parts"][0]["segment_ids"][:-1]  # no longer matches unit 0
    graph["parts"][1]["summary"] = ""
    context = build_graph_context(graph, 120, focus_unit=3, max_chars=1)
    assert context.count("  Summary: ") == len(coarse_units(graph)) - 2
    assert context.count("\n  [") == 2


def test_short_video_ignores_parts():
    graph = copy.deepcopy(load_graph(KITCHEN))
    plain = build_graph_context(graph, 120)
    assert build_graph_context(summarized(graph), 120) == plain
