"""Verifier modes in video_verifier/scoring.py: a judgment's top logprobs at
its answer token become c_i = P(1) / (P(1) + P(0)), wherever the answer sits
in the generated tokens."""

import math
from types import SimpleNamespace

import pytest

from video_verifier.prompts import build_candidate_prompt
from video_verifier.scoring import (
    BOX_OPEN,
    answer_position,
    binary_prob,
    build_direct_prompt,
    score_output,
    verifier_prompt,
)


def logprob_position(probs):
    """vLLM-style {token_id: Logprob} from {decoded token: probability}."""
    return {i: SimpleNamespace(logprob=math.log(p), decoded_token=tok) for i, (tok, p) in enumerate(probs.items())}


def request_output(tokens, positions, text=""):
    """A RequestOutput with one completion whose i-th token is tokens[i] (also
    the top entry at position i) and whose logprobs are `positions`."""
    logprobs = []
    for token, probs in zip(tokens, positions):
        position = logprob_position(probs or {token: 1.0})
        chosen = next(i for i, lp in position.items() if lp.decoded_token == token)
        logprobs.append({**{k: v for k, v in position.items() if k != chosen}, chosen: position[chosen]})
    token_ids = [next(k for k, v in position.items() if v.decoded_token == token) for token, position in zip(tokens, logprobs)]
    return SimpleNamespace(outputs=[SimpleNamespace(text=text or "".join(tokens), token_ids=token_ids, logprobs=logprobs)])


def test_binary_prob_normalizes_over_one_and_zero():
    assert binary_prob({"1": math.log(0.6), "0": math.log(0.2), "x": math.log(0.2)}) == pytest.approx(0.75)
    assert binary_prob({" 1": math.log(0.3), "0": math.log(0.3)}) == pytest.approx(0.5)
    assert binary_prob({"x": 0.0}) is None


def test_answer_position_after_last_box():
    tokens = ["I", " think", " \\", "boxed", "{", "0", "}", " no", " \\", "boxed", "{", "1", "}"]
    assert answer_position(tokens) == (11, "")
    assert answer_position(["no", " box"]) is None
    assert answer_position(["\\", "boxed", "{"]) is None  # nothing after the box


def test_answer_position_box_merged_with_answer():
    assert answer_position(["\\", "boxed", "{1", "}"]) == (2, "{")


def test_direct_reads_first_token():
    out = request_output(["1"], [{"1": 0.7, "0": 0.1, "2": 0.2}])
    c, drawn = score_output("direct", out)
    assert c == pytest.approx(0.875) and drawn == 1


def test_greedy_reads_answer_token_even_when_merged():
    tokens = ["Reason", ".", " \\", "boxed", "{0", "}"]
    positions = [None, None, None, None, {"{0": 0.6, "{1": 0.2, "{": 0.2}, None]
    c, drawn = score_output("greedy", request_output(tokens, positions))
    assert c == pytest.approx(0.25) and drawn == 1
    assert score_output("greedy", request_output(["no", " box"], [None, None])) == (None, 1)


def test_sample_mode_is_vote_fraction():
    out = SimpleNamespace(outputs=[SimpleNamespace(text=t) for t in ("\\boxed{1}", "\\boxed{0}", "\\boxed{1}", "x")])
    c, drawn = score_output("sample", out)
    assert c == pytest.approx(2 / 3) and drawn == 4
    assert score_output("sample", None) == (None, 0)


def test_direct_prompt_prefills_the_box():
    problem = "Query: q\nJudge whether this candidate satisfies the query."
    direct = build_direct_prompt(problem)
    assert direct.endswith(f"<|im_start|>assistant\n{BOX_OPEN}")
    assert "step by step" not in direct and problem in direct
    assert verifier_prompt(problem, "greedy") == verifier_prompt(problem, "sample") == build_candidate_prompt(problem)
