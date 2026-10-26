"""How a Verifier judgment of one candidate (or coarse unit) becomes its
consistency c_i, shared by the reward server and
scripts_Qwen3-VL-8B/verifier_mode_check.py so both score identically.

  sample  G sampled step-by-step judgments; c_i = fraction that accept
          (the original Verifier; build_candidate_prompt).
  direct  no reasoning: the assistant turn is pre-filled with "\\boxed{" and
          one token is generated; c_i = P(1) / (P(1) + P(0)) at that token.
          One prompt pass per candidate instead of G decodes.
  greedy  one temperature-0 step-by-step judgment; c_i is read the same way
          at the token after its last "\\boxed{".

direct and greedy return a probability, not a vote fraction, so d(x, q) =
mean min(c_i, 1 - c_i) is computed from the same formula but means "how
unsure is the Verifier" rather than "how often do its samples disagree".
"""

import math

from video_verifier.prompts import IMAGE_PLACEHOLDER, build_candidate_prompt, tally_votes

MODES = ("sample", "direct", "greedy")
# Top logprobs returned per generated token (vLLM's default max_logprobs).
TOP_LOGPROBS = 20
BOX_OPEN = "\\boxed{"
YES_TOKENS = ("1", " 1")
NO_TOKENS = ("0", " 0")


def build_direct_prompt(problem):
    """build_candidate_prompt without the reasoning instruction, with the
    assistant turn pre-filled up to the judgment."""
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{IMAGE_PLACEHOLDER}<|vision_end|>"
        f"{problem} "
        "Answer with only the final judgment, 1 or 0, inside \\boxed{}.<|im_end|>\n"
        f"<|im_start|>assistant\n{BOX_OPEN}"
    )


def verifier_prompt(problem, mode):
    return build_direct_prompt(problem) if mode == "direct" else build_candidate_prompt(problem)


def sampling_params(mode, n, stop_token_ids):
    """vLLM sampling for one judgment round: n sampled judgments in `sample`
    mode, one deterministic judgment with top logprobs otherwise."""
    import vllm

    if mode == "sample":
        return vllm.SamplingParams(
            max_tokens=2048, temperature=1.0, top_p=1.0, top_k=40, stop_token_ids=stop_token_ids, n=n,
        )
    return vllm.SamplingParams(
        max_tokens=1 if mode == "direct" else 2048, temperature=0.0, top_p=1.0, top_k=-1,
        stop_token_ids=stop_token_ids, n=1, logprobs=TOP_LOGPROBS,
    )


def binary_prob(top_logprobs):
    """{decoded token: logprob} at the judgment position -> P(1) / (P(1) + P(0)),
    or None if neither appears in the top logprobs."""
    p_yes = sum(math.exp(top_logprobs[t]) for t in YES_TOKENS if t in top_logprobs)
    p_no = sum(math.exp(top_logprobs[t]) for t in NO_TOKENS if t in top_logprobs)
    if p_yes + p_no == 0:
        return None
    return p_yes / (p_yes + p_no)


def answer_position(tokens):
    """Where the judgment of the last "\\boxed{" is, in a list of decoded
    tokens: (index of the token holding it, the text of that token before the
    judgment). The second part is non-empty when the box opens mid-token
    (e.g. a merged "{1"); alternatives at that position must then be read past
    the same prefix. None when there is no box or nothing follows it."""
    text, ends = "", []
    for token in tokens:
        text += token
        ends.append(len(text))
    cut = text.rfind(BOX_OPEN)
    if cut < 0:
        return None
    box_end = cut + len(BOX_OPEN)
    for i, end in enumerate(ends):
        if end > box_end:
            return i, text[end - len(tokens[i]):box_end]
    return None


def _top_logprobs(position_logprobs, prefix=""):
    """vLLM's {token_id: Logprob} at one position -> {decoded token: logprob},
    keeping tokens that start with `prefix` and reading them past it."""
    return {
        lp.decoded_token[len(prefix):]: lp.logprob
        for lp in position_logprobs.values()
        if lp.decoded_token is not None and lp.decoded_token.startswith(prefix)
    }


def score_output(mode, request_output):
    """One vLLM RequestOutput (None if generation failed) -> (c_i or None,
    judgments drawn)."""
    if request_output is None:
        return None, 0
    outputs = request_output.outputs
    if mode == "sample":
        return tally_votes([out.text for out in outputs])[1], len(outputs)
    out = outputs[0]
    if not out.logprobs:
        return None, 1
    if mode == "direct":
        return binary_prob(_top_logprobs(out.logprobs[0])), 1
    tokens = [position[token_id].decoded_token or "" for token_id, position in zip(out.token_ids, out.logprobs)]
    found = answer_position(tokens)
    if found is None:
        return None, 1
    pos, prefix = found
    return binary_prob(_top_logprobs(out.logprobs[pos], prefix)), 1
