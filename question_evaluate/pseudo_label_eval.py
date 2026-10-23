"""Ground truth for the Verifier's pseudo-labels: the labeled CoMET-Bench
queries on clips that have video graphs, each run through the Planner exactly
as evaluate.py runs a generated query, with every candidate labeled from the
annotated intervals. Shared by validate_pseudo_labels.py (vote, then score
against the labels) and sft_verifier.py (train on the labels).

Choices (2026-10-22):
  - Data: annotations.json queries with status "verified", on the clips cut by
    prepare_video_subset.py (manifest.json) that have a graph. Annotated times
    are in source-video seconds and are shifted into clip time.
  - A query with annotations but none inside its clip is skipped (its event
    happens elsewhere in the source video); a query with no annotations is
    kept, all its candidates labeled 0.
  - Label: a single candidate is 1 if its segment overlaps any GT interval; a
    pair is 1 if both of its segments do.
  - Category: one Planner run per query type that is a Planner category
    (CoMET's `periodical` has none), reported under that category.
"""

import json
import os
import random
import re

from frozen_planner.planner import CATEGORIES, plan
from video_verifier.prompts import build_candidate_problem, evenly_spaced, load_graph, video_duration


HMS = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")


def clip_key(url):
    """Annotation entry -> video id: the YouTube/Bilibili id, or the file
    stem when the url is a bare filename (the numbered benchmark videos)."""
    m = re.search(r"(?:v=|youtu\.be/|video/)([\w-]+)", url)
    return m.group(1) if m else os.path.splitext(os.path.basename(url))[0]


def hms_to_sec(text):
    return sum(int(part) * 60 ** i for i, part in enumerate(reversed(text.split(":"))))


def sec_to_hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def shift_query_text(text, offset, duration):
    """Rewrite HH:MM:SS times in a query from source-video time into clip time,
    clamped to the clip, so they agree with the clip-time captions."""
    return HMS.sub(lambda m: sec_to_hms(min(max(hms_to_sec(m.group(0)) - offset, 0), duration)), text)


def overlaps(span, interval):
    """Positive-length overlap, or containment of a point interval -- the
    Planner's rule for time_bounds (frozen_planner.planner._overlaps)."""
    (start, end), (lo, hi) = span, interval
    if lo == hi:
        return start <= lo < end
    return end > lo and start < hi


def candidate_label(candidate, gt_intervals):
    return int(all(any(overlaps(span, iv) for iv in gt_intervals) for span in candidate["spans"]))


def load_queries(annotations_path, manifest_path, graph_dir):
    """Verified CoMET queries on every manifest clip that has a graph, in clip time."""
    with open(annotations_path) as f:
        annotations = {clip_key(entry["url"]): entry for entry in json.load(f)}
    with open(manifest_path) as f:
        manifest = json.load(f)

    queries = []
    for clip in manifest:
        clip_id = os.path.splitext(clip["clip"])[0]
        graph_path = os.path.abspath(os.path.join(graph_dir, f"{clip_id}.json"))
        entry = annotations.get(os.path.splitext(os.path.basename(clip["source"]))[0])
        if entry is None or not os.path.exists(graph_path):
            continue
        offset, duration = clip["start_sec"], video_duration(load_graph(graph_path))
        for qi, query in enumerate(entry["queries"]):
            if query.get("status") != "verified":
                continue
            source = [(hms_to_sec(a["start_timestamp"]), hms_to_sec(a["end_timestamp"])) for a in query.get("annotations", [])]
            in_clip = sorted({(max(s - offset, 0), min(e - offset, duration)) for s, e in source
                              if overlaps((offset, offset + duration), (s, e))})
            if source and not in_clip:
                continue
            categories = [t for t in query.get("query_types", []) if t in CATEGORIES]
            if not categories:
                continue
            queries.append({
                "query_id": f"{clip_id}#{qi}",
                "clip": clip_id,
                "graph_path": graph_path,
                "question": shift_query_text(query["query_text"], offset, duration),
                "categories": categories,
                "gt_intervals": [list(iv) for iv in in_clip],
            })
    return queries


def assign_splits(queries, eval_fraction, seed):
    """Split by clip, so SFT never trains on a clip it is evaluated on."""
    clips = sorted({q["clip"] for q in queries})
    random.Random(seed).shuffle(clips)
    eval_clips = set(clips[:round(len(clips) * eval_fraction)])
    for q in queries:
        q["split"] = "eval" if q["clip"] in eval_clips else "train"
    return queries


def build_items(queries, max_candidates):
    """One item per (query, Planner category, candidate), capped per run as
    evaluate.py caps C(x, q)."""
    items = []
    for q in queries:
        graph = load_graph(q["graph_path"])
        for category in q["categories"]:
            candidates = plan(graph, {"category": category, "time_bounds": None})
            for candidate in evenly_spaced(candidates, max_candidates):
                items.append({
                    "query_id": q["query_id"],
                    "clip": q["clip"],
                    "split": q["split"],
                    "graph_path": q["graph_path"],
                    "category": category,
                    "question": q["question"],
                    "candidate": candidate,
                    "problem": build_candidate_problem(q["question"], candidate, graph),
                    "label": candidate_label(candidate, q["gt_intervals"]),
                })
    return items


# --- Scoring the majority vote against the labels ---

CALIBRATION_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def vote_outcome(votes):
    """evaluate.py's majority vote: (pseudo-label, agreement score), or
    (None, None) with no parseable vote or a tie, which evaluate.py drops."""
    if not votes:
        return None, None
    consistency = sum(votes) / len(votes)
    if consistency == 0.5:
        return None, None
    return int(consistency > 0.5), max(consistency, 1 - consistency)


def accuracy_stats(rows):
    """rows: (label, pred). Accuracy, balanced accuracy, GT positive rate and
    the always-majority-class baseline, which plain accuracy must beat."""
    n = len(rows)
    if not n:
        return {"n": 0}
    pos = [pred for label, pred in rows if label == 1]
    neg = [pred for label, pred in rows if label == 0]
    recalls = [sum(p == 1 for p in pos) / len(pos)] if pos else []
    recalls += [sum(p == 0 for p in neg) / len(neg)] if neg else []
    return {
        "n": n,
        "accuracy": sum(label == pred for label, pred in rows) / n,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "positive_rate": len(pos) / n,
        "majority_baseline": max(len(pos), len(neg)) / n,
    }


def summarize_group(records, min_score, max_score):
    voted = []
    for r in records:
        pred, score = vote_outcome(r["votes"])
        if pred is not None:
            voted.append((r["label"], pred, score))
    in_band = [(label, pred) for label, pred, score in voted if min_score <= score <= max_score]
    out_band = [(label, pred) for label, pred, score in voted if not min_score <= score <= max_score]
    return {
        "candidates": len(records),
        "voted_fraction": len(voted) / len(records) if records else 0.0,
        "kept_fraction": len(in_band) / len(records) if records else 0.0,
        "all_voted": accuracy_stats([(label, pred) for label, pred, _ in voted]),
        "in_band": accuracy_stats(in_band),
        "out_of_band": accuracy_stats(out_band),
    }


def calibration(records):
    bins = []
    for lo, hi in zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:]):
        rows = []
        for r in records:
            pred, score = vote_outcome(r["votes"])
            if pred is not None and lo <= score and (score < hi or hi == CALIBRATION_EDGES[-1]):
                rows.append((r["label"], pred, score))
        bins.append({
            "bin": [lo, hi],
            "n": len(rows),
            "mean_agreement": sum(s for _, _, s in rows) / len(rows) if rows else None,
            "accuracy": sum(label == pred for label, pred, _ in rows) / len(rows) if rows else None,
        })
    return bins


def summarize(records, min_score, max_score):
    categories = [c for c in CATEGORIES if any(r["category"] == c for r in records)]
    return {
        "band": [min_score, max_score],
        "overall": summarize_group(records, min_score, max_score),
        "per_category": {c: summarize_group([r for r in records if r["category"] == c], min_score, max_score)
                         for c in categories},
        "calibration": calibration(records),
    }


def _pct(stats, key):
    return f"{stats[key]:.1%}" if stats.get("n") else "–"


def report_markdown(summary, title):
    lo, hi = summary["band"]
    lines = [
        f"# {title}",
        "",
        f"Band: agreement in [{lo}, {hi}] (SUPERVISOR_MIN_SCORE / SUPERVISOR_MAX_SCORE). Agreement is "
        "max(c, 1 - c) over parseable votes, so it is always >= 0.5. Ties and candidates with no "
        "parseable vote get no pseudo-label, as in evaluate.py.",
        "",
        "## Pseudo-label accuracy",
        "",
        "| Category | Candidates | Voted | Acc. (all voted) | Bal. acc. | GT pos. | Majority baseline "
        "| Kept by band | Acc. (in band) | Acc. (out of band) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, group in [("**overall**", summary["overall"])] + list(summary["per_category"].items()):
        a, b, o = group["all_voted"], group["in_band"], group["out_of_band"]
        lines.append(
            f"| {name} | {group['candidates']} | {group['voted_fraction']:.1%} | {_pct(a, 'accuracy')} "
            f"| {_pct(a, 'balanced_accuracy')} | {_pct(a, 'positive_rate')} | {_pct(a, 'majority_baseline')} "
            f"| {group['kept_fraction']:.1%} | {_pct(b, 'accuracy')} (n={b['n']}) | {_pct(o, 'accuracy')} (n={o['n']}) |"
        )
    lines += [
        "",
        "## Accuracy vs vote agreement",
        "",
        "| Agreement | Candidates | Mean agreement | Accuracy |",
        "|---|---|---|---|",
    ]
    for b in summary["calibration"]:
        lo_b, hi_b = b["bin"]
        right = "]" if hi_b == CALIBRATION_EDGES[-1] else ")"
        mean = f"{b['mean_agreement']:.3f}" if b["n"] else "–"
        acc = f"{b['accuracy']:.1%}" if b["n"] else "–"
        lines.append(f"| [{lo_b:.1f}, {hi_b:.1f}{right} | {b['n']} | {mean} | {acc} |")
    lines += ["", "A calibrated vote has accuracy close to its mean agreement in every row."]
    return "\n".join(lines) + "\n"


def plot_calibration(summary, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, series, band = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#f0efec"
    points = [b for b in summary["calibration"] if b["n"]]
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    lo, hi = summary["band"]
    ax.axvspan(max(lo, 0.5), hi, color=band, zorder=0)
    ax.text((max(lo, 0.5) + hi) / 2, 0.03, "kept by band", ha="center", color=ink2, fontsize=8)
    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle=(0, (4, 3)), linewidth=1, color=ink2, zorder=1,
            label="calibrated (accuracy = agreement)")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=ink2)
    xs = [b["mean_agreement"] for b in points]
    ys = [b["accuracy"] for b in points]
    ax.plot(xs, ys, color=series, linewidth=2, marker="o", markersize=7,
            markeredgecolor=surface, markeredgewidth=2, zorder=3)
    for x, y, b in zip(xs, ys, points):
        ax.annotate(f"n={b['n']}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", color=ink2, fontsize=7)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0, 1.09)
    ax.set_xlabel("Vote agreement (mean per bin)", color=ink2, fontsize=9)
    ax.set_ylabel("Pseudo-label accuracy", color=ink2, fontsize=9)
    ax.set_title(title, color=ink, fontsize=10, loc="left")
    ax.grid(color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=ink2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
