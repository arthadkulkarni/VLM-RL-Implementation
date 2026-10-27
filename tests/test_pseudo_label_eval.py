"""GT labeling and pseudo-label scoring for validate_pseudo_labels.py."""

import json
import os

import pytest

from question_evaluate.pseudo_label_eval import (
    assign_splits,
    build_items,
    calibration,
    candidate_label,
    load_queries,
    plot_calibration,
    report_markdown,
    shift_query_text,
    summarize,
    vote_outcome,
)


GRAPH_DIR = os.path.join(os.path.dirname(__file__), "fixtures")  # kitchen_graph.json, 20 s, 4 segments


def test_shift_query_text_into_clip_time():
    text = "Between 00:25:00 and 00:30:00, what happens? Also 00:10:00 and 01:00:00."
    assert shift_query_text(text, offset=1500, duration=300) == (
        "Between 00:00:00 and 00:05:00, what happens? Also 00:00:00 and 00:05:00."
    )


def test_candidate_label_single_and_pair():
    gt = [[5.0, 9.0]]
    assert candidate_label({"spans": [[5.0, 9.0]]}, gt) == 1
    assert candidate_label({"spans": [[0.0, 5.0]]}, gt) == 0  # only touches the interval
    assert candidate_label({"spans": [[0.0, 6.0], [5.0, 9.0]]}, gt) == 1
    assert candidate_label({"spans": [[0.0, 6.0], [9.0, 14.0]]}, gt) == 0  # one side outside
    assert candidate_label({"spans": [[9.0, 14.0]]}, [[10.0, 10.0]]) == 1  # point interval inside


def write_benchmark(tmp_path, queries, start_sec=100):
    annotations = [{"url": "kitchen_graph.mp4", "queries": queries}]
    manifest = [{"source": "videos/kitchen_graph.mp4", "clip": "kitchen_graph.mp4", "start_sec": start_sec}]
    (tmp_path / "ann.json").write_text(json.dumps(annotations))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return str(tmp_path / "ann.json"), str(tmp_path / "manifest.json")


def query(text, types, spans, status="verified"):
    return {
        "query_text": text, "status": status, "query_types": types,
        "annotations": [{"start_timestamp": s, "end_timestamp": e} for s, e in spans],
    }


def test_load_queries_filters_and_shifts(tmp_path):
    ann, manifest = write_benchmark(tmp_path, [
        query("Kettle on stove?", ["static", "periodical"], [("00:01:45", "00:01:49")]),  # clip 5-9 s
        query("Elsewhere in the video?", ["dynamic"], [("00:30:00", "00:30:05")]),  # outside the clip
        query("Never happens?", ["negative"], []),  # no annotations: kept, all 0
        query("Unverified?", ["static"], [("00:01:45", "00:01:49")], status="unverified"),
        query("Only periodical?", ["periodical"], [("00:01:45", "00:01:49")]),
    ])
    queries = load_queries(ann, manifest, GRAPH_DIR)
    assert [q["question"] for q in queries] == ["Kettle on stove?", "Never happens?"]
    assert queries[0]["categories"] == ["static"]
    assert queries[0]["gt_intervals"] == [[5, 9]]
    assert queries[1]["gt_intervals"] == []


def test_build_items_labels_from_gt(tmp_path):
    ann, manifest = write_benchmark(tmp_path, [
        query("Kettle?", ["static", "sequential"], [("00:01:45", "00:01:49"), ("00:01:49", "00:01:54")]),
    ])
    items = build_items(assign_splits(load_queries(ann, manifest, GRAPH_DIR), 0.5, 0), max_candidates=64)
    static = {i["candidate"]["segment_ids"][0]: i["label"] for i in items if i["category"] == "static"}
    assert static == {"kitchen_scene000": 0, "kitchen_scene001": 1, "kitchen_scene002": 1, "kitchen_scene003": 0}
    sequential = {tuple(i["candidate"]["segment_ids"]): i["label"] for i in items if i["category"] == "sequential"}
    assert sequential[("kitchen_scene001", "kitchen_scene002")] == 1
    assert sequential[("kitchen_scene000", "kitchen_scene001")] == 0
    assert all(i["problem"].startswith("Query: Kettle?") for i in items)


def test_vote_outcome_matches_evaluate():
    assert vote_outcome([1, 1, 0]) == (1, pytest.approx(2 / 3))
    assert vote_outcome([0, 0, 0, 1]) == (0, 0.75)
    assert vote_outcome([1, 0]) == (None, None)  # tie: dropped
    assert vote_outcome([]) == (None, None)  # nothing parseable: dropped


def records():
    def rec(category, label, votes):
        return {"category": category, "label": label, "votes": votes}
    return [
        rec("static", 1, [1] * 9),           # agreement 1.0, right, out of band
        rec("static", 0, [1] * 9),           # agreement 1.0, wrong, out of band
        rec("static", 0, [0] * 6 + [1] * 3),  # 0.667, right, in band
        rec("dynamic", 1, [0] * 7 + [1] * 2),  # 0.778, wrong, in band
        rec("dynamic", 0, [0, 1]),           # tie
    ]


def test_summarize_band_and_categories():
    s = summarize(records(), 0.3, 0.8)
    overall = s["overall"]
    assert overall["candidates"] == 5
    assert overall["voted_fraction"] == 0.8
    assert overall["kept_fraction"] == 0.4
    assert overall["all_voted"]["accuracy"] == 0.5
    assert overall["in_band"]["accuracy"] == 0.5
    assert overall["out_of_band"]["accuracy"] == 0.5
    assert s["per_category"]["static"]["all_voted"]["accuracy"] == pytest.approx(2 / 3)
    assert s["per_category"]["dynamic"]["in_band"] == {
        "n": 1, "accuracy": 0.0, "balanced_accuracy": 0.0, "positive_rate": 1.0, "majority_baseline": 1.0,
    }


def test_calibration_bins():
    bins = {tuple(b["bin"]): b for b in calibration(records())}
    assert bins[(0.6, 0.7)]["n"] == 1 and bins[(0.6, 0.7)]["accuracy"] == 1.0
    assert bins[(0.7, 0.8)]["n"] == 1 and bins[(0.7, 0.8)]["accuracy"] == 0.0
    assert bins[(0.9, 1.0)]["n"] == 2 and bins[(0.9, 1.0)]["accuracy"] == 0.5  # 1.0 lands in the last bin
    assert bins[(0.5, 0.6)]["n"] == 0


def test_report_and_plot_render(tmp_path):
    s = summarize(records(), 0.3, 0.8)
    md = report_markdown(s, "test")
    assert "| **overall** | 5 | 80.0% | 50.0%" in md
    assert "| [0.9, 1.0] | 2 | 1.000 | 50.0% |" in md
    plot_calibration(s, tmp_path / "c.png", "test")
    assert (tmp_path / "c.png").stat().st_size > 0
