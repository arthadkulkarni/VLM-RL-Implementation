"""Measure K = |C(x, q)| per category over a directory of video graphs, before
training. Relational categories are the ones to watch: sequential and
synchronous (proximity) pairs are bounded by ~N * window, but causal
(shared-entity) pairs can grow ~N^2 when one linked entity spans most of the
video. Anything that looks superlinear is flagged.

Usage:
  python -m frozen_planner.measure_k --graph_dir <dir> [--out report.json]
"""

import argparse
import glob
import json
import math
import os
import statistics

from frozen_planner.planner import (
    CATEGORIES,
    RELATIONAL_CATEGORIES,
    GraphView,
    PlannerConfig,
    plan,
)


def _percentile(values, pct):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1)]


def _loglog_slope(ns, ks):
    """Least-squares slope of log K vs log N. ~1 means linear growth in the
    number of timeline segments, ~2 means pairwise blow-up."""
    points = [(math.log(n), math.log(k)) for n, k in zip(ns, ks) if n > 1 and k > 0]
    if len({x for x, _ in points}) < 3:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    var_x = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / var_x


def measure_video(graph, config):
    view = GraphView(graph)
    n = len(view.timeline)
    row = {"video_id": view.video_id, "n_timeline": n, "n_scenes": len(view.scenes), "k": {}}

    for category in CATEGORIES:
        candidates = plan(view, {"category": category}, config)
        row["k"][category] = len(candidates)

    # Share of the timeline covered by the single most widespread entity --
    # the driver of shared-entity pair growth (m segments -> m(m-1)/2 pairs).
    coverage = {}
    for sid in view.timeline:
        for entity_id in view.entities(sid):
            coverage[entity_id] = coverage.get(entity_id, 0) + 1
    row["max_entity_coverage"] = max(coverage.values()) / n if coverage and n else 0.0
    return row


def summarize(rows, k_budget):
    summary, flags = {}, []
    coverage = [r["max_entity_coverage"] for r in rows]
    summary["max_entity_coverage"] = {"median": statistics.median(coverage), "max": max(coverage)}
    ns = [r["n_timeline"] for r in rows]
    for category in CATEGORIES:
        ks = [r["k"][category] for r in rows]
        stats = {
            "min": min(ks),
            "median": statistics.median(ks),
            "p90": _percentile(ks, 90),
            "max": max(ks),
            "mean_k_per_segment": statistics.fmean(k / n for k, n in zip(ks, ns) if n),
        }
        if category in RELATIONAL_CATEGORIES:
            slope = _loglog_slope(ns, ks)
            stats["loglog_slope_k_vs_n"] = slope
            if slope is not None and slope > 1.3:
                flags.append(f"{category}: K grows superlinearly in N (log-log slope {slope:.2f})")
        if stats["max"] > k_budget:
            flags.append(f"{category}: max K={stats['max']} exceeds per-question budget {k_budget}")
        summary[category] = stats

    if len(set(ns)) < 3:
        flags.append(
            f"only {len(set(ns))} distinct timeline lengths across {len(rows)} graphs -- "
            f"too few to estimate how K scales with N"
        )
    return summary, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    parser.add_argument("--window_segments", type=int, default=PlannerConfig.window_segments)
    parser.add_argument("--max_entity_gap_segments", type=int, default=None)
    parser.add_argument("--sync_tolerance_sec", type=float, default=PlannerConfig.sync_tolerance_sec)
    parser.add_argument("--k_budget", type=int, default=128, help="Flag categories whose max K exceeds this.")
    args = parser.parse_args()

    config = PlannerConfig(
        window_segments=args.window_segments,
        max_entity_gap_segments=args.max_entity_gap_segments,
        sync_tolerance_sec=args.sync_tolerance_sec,
    )

    paths = sorted(glob.glob(os.path.join(args.graph_dir, "*.json")))
    if not paths:
        raise SystemExit(f"no graph JSONs under {args.graph_dir}")
    rows = []
    for path in paths:
        with open(path) as f:
            rows.append(measure_video(json.load(f), config))

    summary, flags = summarize(rows, args.k_budget)

    print(f"[measure-k] {len(rows)} graphs, config={config}")
    print(f"{'category':<12} {'min':>5} {'med':>6} {'p90':>5} {'max':>5} {'K/N':>6} {'slope':>6}")
    for category in CATEGORIES:
        s = summary[category]
        slope = s.get("loglog_slope_k_vs_n")
        print(
            f"{category:<12} {s['min']:>5} {s['median']:>6} {s['p90']:>5} {s['max']:>5} "
            f"{s['mean_k_per_segment']:>6.2f} {'-' if slope is None else f'{slope:.2f}':>6}"
        )
    coverage = summary["max_entity_coverage"]
    print(
        f"[measure-k] most widespread entity covers median {100 * coverage['median']:.0f}% / "
        f"max {100 * coverage['max']:.0f}% of a video's timeline (drives causal K)"
    )
    for flag in flags:
        print(f"[measure-k] FLAG {flag}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": vars(config), "summary": summary, "flags": flags, "videos": rows}, f, indent=2)
        print(f"[measure-k] report -> {args.out}")


if __name__ == "__main__":
    main()
