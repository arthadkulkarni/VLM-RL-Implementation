"""Frozen Aggregator: Verifier-accepted candidates -> final answer Ŷ.

Sits after the Verifier and before the GRPO update. Deterministic and
non-learned, like the Planner -- no model calls. Its only job is turning
accepted candidates into the reportable answer Ŷ (interval list + count)
that is scored against held-out CoMET-Bench ground truth. Ŷ is not used in
the training reward.

Input is the Planner's C(x, q) plus the Verifier's verdicts. One Verifier
prompt judges one candidate, so GRPO yields n sampled verdicts per
candidate; those are combined into one verdict per candidate first.

Output:
  - records: one (start, end, summary, score) per accepted candidate, the
    minimal record list that format_for_prompt serializes when accepted
    evidence is fed back into a prompt as context;
  - Ŷ: the accepted intervals after deduplication, and their count.

Two resolution steps:
  - Relational candidates (causal/sequential/synchronous) are segment pairs;
    each resolves to one reportable interval by a per-category rule
    (hull / first / second), which should follow CoMET-Bench's convention.
  - Separately accepted candidates that describe the same event are
    collapsed: any two whose intervals have IoU >= dedup_iou keep only the
    higher-scoring one, so the count doesn't double-count.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

from frozen_planner.planner import RELATIONAL_CATEGORIES

# How an accepted pair (first, second) becomes one reportable interval:
#   hull   -> [first.start, second.end]
#   first  -> first's span (e.g. the cause)
#   second -> second's span (e.g. the effect)
RELATIONAL_OUTPUTS = ("hull", "first", "second")


class Record(NamedTuple):
    start: float
    end: float
    summary: str
    score: float


@dataclass
class AggregatorConfig:
    # Used only when a verdict carries no explicit "accepted" flag.
    accept_threshold: float = 0.5
    # Accepted intervals with IoU >= this are one event; the higher-scoring
    # one is kept.
    dedup_iou: float = 0.5
    # Per-category pair -> interval rule. Set to CoMET-Bench's convention;
    # "hull" is a placeholder until that is confirmed.
    relational_output: dict = field(
        default_factory=lambda: {category: "hull" for category in RELATIONAL_CATEGORIES}
    )


# Joins a relational candidate's two captions when the Verifier gives no
# summary: ordered relations read first -> second, synchronous as co-occurring.
_PAIR_JOINERS = {"causal": " -> ", "sequential": " -> ", "synchronous": " | "}


def combine_rollouts(verdicts, config):
    """n sampled verdicts for one candidate -> one verdict: mean score;
    accepted by strict majority of the samples (each sample's own decision --
    its "accepted" flag if present, else its score vs accept_threshold; ties
    reject); summary from the highest-scoring sample that has one."""
    scores = [float(v.get("score", 0.0)) for v in verdicts]
    votes = [_is_accepted(v, config) for v in verdicts]
    combined = {"score": sum(scores) / len(scores), "accepted": sum(votes) * 2 > len(votes)}
    with_summary = [v for v in verdicts if str(v.get("summary") or "").strip()]
    if with_summary:
        combined["summary"] = max(with_summary, key=lambda v: float(v.get("score", 0.0)))["summary"]
    return combined


def _is_accepted(verdict, config):
    if verdict is None:
        return False
    if "accepted" in verdict:
        return bool(verdict["accepted"])
    return float(verdict.get("score", 0.0)) >= config.accept_threshold


def resolve_interval(candidate, config):
    """A candidate's reportable interval: a single event's own span, or a
    relational pair resolved by config.relational_output."""
    spans = candidate["spans"]
    if candidate["kind"] != "pair":
        return spans[0][0], spans[0][1]
    rule = config.relational_output.get(candidate["category"], "hull")
    if rule == "first":
        return spans[0][0], spans[0][1]
    if rule == "second":
        return spans[1][0], spans[1][1]
    if rule == "hull":
        return min(s[0] for s in spans), max(s[1] for s in spans)
    raise ValueError(f"unknown relational_output {rule!r}; expected one of {RELATIONAL_OUTPUTS}")


def to_record(candidate, verdict, config):
    start, end = resolve_interval(candidate, config)
    summary = str(verdict.get("summary") or "").strip()
    if not summary:
        joiner = _PAIR_JOINERS.get(candidate["category"], " ")
        summary = joiner.join(caption.strip() for caption in candidate["captions"] if caption.strip())
    return Record(start, end, summary, float(verdict.get("score", 0.0)))


def build_records(candidates, verdicts, config=None):
    """Accepted candidates as Records, in time order. `verdicts` maps
    candidate_id -> a verdict {"score": float, "accepted": optional bool,
    "summary": optional str}, or a list of them (the n GRPO samples for that
    candidate), which combine_rollouts reduces. A candidate with no verdict
    is rejected."""
    config = config or AggregatorConfig()
    records = []
    for candidate in candidates:
        verdict = verdicts.get(candidate["candidate_id"])
        if isinstance(verdict, list):
            verdict = combine_rollouts(verdict, config) if verdict else None
        if _is_accepted(verdict, config):
            records.append(to_record(candidate, verdict, config))
    return sorted(records, key=lambda r: (r.start, r.end, -r.score))


def _timestamp(seconds):
    minutes, secs = divmod(max(0.0, seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"


def format_for_prompt(records):
    """One line per record, time-ordered:
        [00:12.0-00:15.0] (0.87) a man picks up a cup
    """
    return "\n".join(
        f"[{_timestamp(r.start)}-{_timestamp(r.end)}] ({r.score:.2f}) {r.summary}" for r in records
    )


def interval_iou(a, b):
    """IoU of two (start, end) intervals; two identical zero-length intervals
    count as 1."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else float(tuple(a) == tuple(b))


def dedup(records, iou_threshold):
    """Greedy, highest score first: keep a record unless its interval has
    IoU >= iou_threshold with one already kept. Returned in time order."""
    kept = []
    for record in sorted(records, key=lambda r: (-r.score, r.start, r.end)):
        if all(interval_iou((record.start, record.end), (k.start, k.end)) < iou_threshold for k in kept):
            kept.append(record)
    return sorted(kept, key=lambda r: (r.start, r.end))


def aggregate(candidates, verdicts, config=None):
    """C(x, q) + Verifier verdicts -> {"records", "intervals", "count"}.
    All candidates come from one question. `intervals` and `count` form Ŷ,
    scored against held-out ground truth -- not a training reward."""
    config = config or AggregatorConfig()
    categories = {c["category"] for c in candidates}
    if len(categories) > 1:
        raise ValueError(f"candidates mix categories {sorted(categories)}; aggregate one C(x, q) at a time")

    records = build_records(candidates, verdicts, config)
    answer = dedup(records, config.dedup_iou)
    return {
        "records": records,
        "intervals": [[r.start, r.end] for r in answer],
        "count": len(answer),
    }
