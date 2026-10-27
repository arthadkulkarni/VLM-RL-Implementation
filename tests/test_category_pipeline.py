"""The questioner's declared category must reach the solver training row
unchanged: question_generate -> evaluate (structural check, judge verdict,
per-candidate rows) -> upload (row mapping, skill-balanced sampling)."""

import os
import random

import pytest

from question_evaluate.records import (
    ALLOWED_SKILLS,
    apply_skill_balanced_sampling,
    candidate_result_row,
    count_skills,
    generated_record,
    structural_query_item,
    upload_row,
)


GRAPH_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "kitchen_graph.json")

SEQUENTIAL = (
    "<category>sequential</category>\n"
    "<question>What does the man do right after he puts the kettle on the stove?</question>"
)
BOUNDED = (
    "<category>bounded</category>\n"
    "<question>Which object does the man pick up during this part of the video?</question>\n"
    "<time_bounds>00:09.0, 00:14.0</time_bounds>"
)


def solver_rows(response, category_match=1):
    """One questioner response through every stage up to upload_row."""
    record = generated_record(response, GRAPH_PATH)
    query_item, reason = structural_query_item(record)
    assert reason is None
    # evaluate.py step 3: the joint validity + category judge's verdict.
    query_item["category_match"] = category_match
    query_item["valid"] = 1
    rows = []
    for candidate in query_item.pop("candidates"):
        assert candidate["category"] == query_item["declared_category"]
        row = candidate_result_row(query_item, candidate, "problem", 1, 0.6, [1, 1, 0], "")
        rows.append(upload_row(row))
    assert rows
    return rows


def test_sequential_stays_sequential():
    record = generated_record(SEQUENTIAL, GRAPH_PATH)
    assert record["declared_category"] == "sequential"
    query_item, _ = structural_query_item(record)
    assert query_item["declared_category"] == "sequential"
    assert all(c["kind"] == "pair" for c in query_item["candidates"])

    rows = solver_rows(SEQUENTIAL)
    assert {row["declared_skill"] for row in rows} == {"sequential"}


@pytest.mark.parametrize("target", [4, 1500])
def test_balanced_sampling_keeps_sequential_label(target):
    sequential = solver_rows(SEQUENTIAL)
    bounded = solver_rows(BOUNDED) * 5  # bounded dominates, as in the mini runs
    selected, _ = apply_skill_balanced_sampling(bounded + sequential, target, random.Random(42))

    kept = [row for row in selected if row in sequential]
    assert kept, "skill balancing dropped every sequential row"
    assert all(row["declared_skill"] == "sequential" for row in kept)
    assert count_skills(selected, "declared_skill")["sequential"] == len(kept)


@pytest.mark.parametrize("category", ALLOWED_SKILLS)
def test_every_category_survives(category):
    bounds = "\n<time_bounds>5.0, 14.0</time_bounds>" if category == "bounded" else ""
    response = f"<category>{category}</category>\n<question>A query about the man and the kettle?</question>{bounds}"
    assert {row["declared_skill"] for row in solver_rows(response)} == {category}


def test_judge_category_verdict_reaches_skill_match():
    assert {row["skill_match"] for row in solver_rows(SEQUENTIAL, category_match=0)} == {0}
    assert {row["skill_match"] for row in solver_rows(SEQUENTIAL, category_match=1)} == {1}
