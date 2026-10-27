"""Per-part summaries of a video graph, for the long-video overview in the
questioner / validity-judge view (video_verifier/prompts.build_graph_context).

A part is one of the Planner's coarse units (frozen_planner coarse_units,
default PlannerConfig). Its summary is written by the caption servers' text
model from the part's timestamped segment captions, given the names of the
linked entities it contains so names stay consistent across parts. Stored as
graph["parts"] = [{"part_id", "span", "segment_ids", "summary"}].

build_video_graph.py summarizes as part of the build; for graphs built
before that, backfill with the caption servers running:

  python -m video_graph_builder.part_summaries --graph_dir <dir> --num_servers 4 [--base_port 6000]
"""

import argparse
import glob
import json
import os

from frozen_planner.planner import coarse_units
from video_verifier.prompts import fmt_time

# Entity names offered per part (most frequent in the whole video first).
MAX_PART_ENTITIES = 10
# Longest summary kept; the prompt asks for at most ~60 words.
MAX_SUMMARY_CHARS = 400


def part_requests(graph):
    """One summary request per coarse unit of `graph` (which needs segments
    with captions and entity_ids, a timeline, and entities)."""
    segments = {seg["segment_id"]: seg for seg in graph["segments"]}
    frequency = {e["entity_id"]: len(e.get("segment_ids", [])) for e in graph.get("entities", [])}
    names = {e["entity_id"]: e.get("name", "") for e in graph.get("entities", [])}
    requests = []
    for unit in coarse_units(graph):
        entity_ids = {eid for sid in unit["segment_ids"] for eid in segments[sid].get("entity_ids", [])}
        ranked = sorted(entity_ids, key=lambda eid: (-frequency.get(eid, 0), eid))
        requests.append({
            "part_id": unit["unit_id"],
            "span": unit["span"],
            "segment_ids": unit["segment_ids"],
            "captions": [
                [segments[sid]["start_sec"], segments[sid]["end_sec"], segments[sid].get("caption", "")]
                for sid in unit["segment_ids"]
            ],
            "entity_names": [names[eid] for eid in ranked[:MAX_PART_ENTITIES] if names.get(eid)],
        })
    return requests


def build_summary_prompt(request):
    """Text-only chat prompt for one part's summary."""
    start, end = request["span"]
    caption_lines = "\n".join(
        f"[{fmt_time(s)}-{fmt_time(e)}] {(caption or '').strip()}" for s, e, caption in request["captions"]
    )
    names = ", ".join(request.get("entity_names") or []) or "none"
    return (
        "<|im_start|>system\n"
        "You summarize one part of a long video from the captions of its consecutive segments.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Video part [{fmt_time(start)}-{fmt_time(end)}], segment captions in time order:\n"
        f"{caption_lines}\n"
        f"Recurring people/objects in this part: {names}\n"
        "Write a summary of this part in at most 60 words: the main events in the order they happen, "
        "who or what is involved, and where. Use the names above for those people/objects. "
        "Only use what the captions say. Output only the summary text.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_summary(text):
    text = " ".join((text or "").split())
    return text if len(text) <= MAX_SUMMARY_CHARS else text[: MAX_SUMMARY_CHARS - 3].rstrip() + "..."


def attach_parts(graph, requests, summaries_by_id):
    """Store graph["parts"]; a part whose summary failed gets an empty one
    (the overview then falls back to its captions)."""
    graph["parts"] = [
        {
            "part_id": request["part_id"],
            "span": request["span"],
            "segment_ids": request["segment_ids"],
            "summary": summaries_by_id.get(request["part_id"], ""),
        }
        for request in requests
    ]
    return graph


def summarize_graphs(graphs, num_servers, base_port=None):
    """Summarize the parts of several graphs in one request batch; sets each
    graph's "parts". Returns the number of parts requested."""
    from video_graph_builder.graph_client import summarize_parts_batch

    per_graph = [part_requests(graph) for graph in graphs]
    items = [request for requests in per_graph for request in requests]
    results = summarize_parts_batch(items, num_servers=num_servers, base_port=base_port)
    by_id = {r.get("part_id"): r.get("summary", "") for r in results}
    for graph, requests in zip(graphs, per_graph):
        attach_parts(graph, requests, by_id)
    return len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--base_port", type=int, default=int(os.getenv("RISE_CAPTION_SERVER_BASE_PORT", "6000")))
    parser.add_argument("--graphs_per_batch", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Re-summarize graphs that already have parts.")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.graph_dir, "*.json")))
    pending = []
    for path in paths:
        with open(path) as f:
            graph = json.load(f)
        if args.overwrite or not graph.get("parts"):
            pending.append((path, graph))
    print(f"[part-summaries] {len(pending)}/{len(paths)} graphs to summarize")
    for i in range(0, len(pending), args.graphs_per_batch):
        batch = pending[i:i + args.graphs_per_batch]
        n = summarize_graphs([graph for _, graph in batch], args.num_servers, args.base_port)
        for path, graph in batch:
            with open(path, "w") as f:
                json.dump(graph, f, indent=2)
        print(f"[part-summaries] batch {i // args.graphs_per_batch + 1}: {len(batch)} graphs, {n} parts")


if __name__ == "__main__":
    main()
