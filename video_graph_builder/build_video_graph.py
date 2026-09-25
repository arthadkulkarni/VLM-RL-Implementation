import argparse
import base64
import glob
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from video_graph_builder.entity_linker import (
    candidate_mention_pairs,
    candidate_tracklet_pairs,
    canonical_entities,
    cluster_mentions,
)
from video_graph_builder.graph_client import caption_entities_batch, same_entity_batch
from video_graph_builder.montage import build_montage, sample_frame_indices
from video_graph_builder.segmenter import (
    merge_short_segments,
    build_segment_hierarchy,
    scene_spans_from_boundaries,
)
from video_graph_builder.similarity_curves import SimilarityCurves
from video_graph_builder.video_io import sample_video

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _video_id(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


def find_videos(video_dir):
    paths = []
    for ext in VIDEO_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
    return sorted(paths)


def build_segments_for_video(sc, video_path, fps, min_seconds, num_montage_frames, montage_frame_height, timings):
    """Returns (video_id, segments, timeline). `segments` holds nodes at both
    levels -- "scene" (DINOv2 appearance cuts) and "action" (RAFT motion cuts
    inside one scene) -- linked by parent_id / child_ids. `timeline` is the
    ordered, non-overlapping list of finest-level segment_ids covering the
    video: a scene's actions where it was split, else the scene itself (see
    build_segment_hierarchy for the rule). The video is decoded once; stage
    wall times (seconds) are recorded into `timings`.
    """
    t0 = time.time()
    sampled = sample_video(video_path, fps=fps)
    timings["decode"] = time.time() - t0
    timings["sampled_frames"] = len(sampled.frames)

    t0 = time.time()
    scene_result = sc.compute_scene(sampled)
    scene_spans = scene_spans_from_boundaries(scene_result["boundaries"], scene_result["num_sampled_frames"])
    scene_spans = merge_short_segments(scene_spans, max(2, round(min_seconds * fps)))
    timings["scene_dino"] = time.time() - t0

    t0 = time.time()
    action_results = sc.compute_action_for_segments(sampled, segments=scene_spans)
    hierarchy = build_segment_hierarchy(scene_spans, action_results, min_seconds=min_seconds, fps=fps)
    timings["action_raft"] = time.time() - t0

    t0 = time.time()
    frame_indices = sampled.frame_indices
    video_fps = sampled.video_fps
    video_id = _video_id(video_path)

    last_sampled_idx = len(frame_indices) - 1

    def make_segment(segment_id, level, start_idx, end_idx, parent_id):
        # end_idx is the next segment's first frame (a boundary at i means
        # frame i starts the new segment), so it's excluded from the montage --
        # otherwise the caption describes the next scene. The video's final
        # segment ends on the last sampled frame, which it does own.
        last_in_span = end_idx if end_idx >= last_sampled_idx else max(start_idx, end_idx - 1)
        sample_idxs = sample_frame_indices(start_idx, last_in_span, num_montage_frames)
        montage = build_montage(
            [Image.fromarray(sampled.frames[min(idx, last_sampled_idx)]) for idx in sample_idxs],
            max_frame_height=montage_frame_height,
        )

        start_raw = frame_indices[min(start_idx, len(frame_indices) - 1)]
        end_raw = frame_indices[min(end_idx, len(frame_indices) - 1)]
        start_sec = start_raw / video_fps
        end_sec = end_raw / video_fps
        return {
            "segment_id": segment_id,
            "level": level,
            "parent_id": parent_id,
            "child_ids": [],
            "start_frame": int(start_raw),
            "end_frame": int(end_raw),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "start_time": sc.seconds_to_timestamp(start_sec),
            "end_time": sc.seconds_to_timestamp(end_sec),
            "montage": montage,
        }

    segments = []
    timeline = []
    for i, scene in enumerate(hierarchy):
        scene_seg = make_segment(f"{video_id}_scene{i:03d}", "scene", *scene["span"], parent_id=None)
        segments.append(scene_seg)
        if not scene["actions"]:
            timeline.append(scene_seg["segment_id"])
            continue
        for j, (start_idx, end_idx) in enumerate(scene["actions"]):
            action_seg = make_segment(
                f"{scene_seg['segment_id']}_act{j:02d}", "action", start_idx, end_idx,
                parent_id=scene_seg["segment_id"],
            )
            scene_seg["child_ids"].append(action_seg["segment_id"])
            segments.append(action_seg)
            timeline.append(action_seg["segment_id"])

    timings["montage"] = time.time() - t0
    return video_id, segments, timeline


def _mention_fields(mention):
    return {key: mention.get(key, "") for key in ("name", "type", "description")}


def caption_and_link(videos, num_servers, base_port, link_window_sec, max_reid_pairs):
    """Caption and entity-link the segments of several videos at once, so each
    VLM round (captions, local same-entity pairs, re-id pairs) is one request
    batch spread over all servers instead of a few prompts per video.

    videos: [{"video_id", "segments", ...}] (montages still attached, from
    prepare_video). Sets each segment's caption/entity_ids, stores the video's
    canonical entities under "entities", and returns per-round stats.
    """
    montages_b64 = {
        seg["segment_id"]: _pil_to_base64(seg.pop("montage"))
        for video in videos for seg in video["segments"]
    }
    stats = {}

    t0 = time.time()
    request_items = [{"segment_id": segment_id, "image": image} for segment_id, image in montages_b64.items()]
    raw_results = caption_entities_batch(request_items, num_servers=num_servers, base_port=base_port)
    parsed_by_id = {item.get("segment_id"): item for item in raw_results}
    stats["caption"] = (time.time() - t0, len(request_items))

    for video in videos:
        mentions = []
        for seg in video["segments"]:
            parsed = parsed_by_id.get(seg["segment_id"], {"caption": "", "entities": []})
            seg["caption"] = parsed.get("caption", "")
            for entity in parsed.get("entities", []):
                mentions.append({**entity, "segment_id": seg["segment_id"]})
        video["mentions"] = mentions
        video["segments_by_id"] = {seg["segment_id"]: seg for seg in video["segments"]}

    # Cross-segment identity is decided visually by the VLM, pair by pair --
    # text similarity of names/descriptions can't tell "yellow rectangle" vs
    # "yellow square" (same object) from "yellow square" vs "cyan square".
    # Stage 1 links nearby mentions into tracklets; stage 2 re-identifies
    # tracklets across the whole video (see entity_linker). Each stage judges
    # every video's pairs in one batch.
    def judge(pairs_per_video):
        requests = []
        for v, (video, pairs) in enumerate(zip(videos, pairs_per_video)):
            mentions = video["mentions"]
            for k, (i, j) in enumerate(pairs):
                requests.append({
                    "pair_id": f"{v}:{k}",
                    "segment_a": mentions[i]["segment_id"],
                    "segment_b": mentions[j]["segment_id"],
                    "mention_a": _mention_fields(mentions[i]),
                    "mention_b": _mention_fields(mentions[j]),
                })
        results = same_entity_batch(requests, montages_b64, num_servers=num_servers, base_port=base_port)
        same_by_id = {r.get("pair_id"): r.get("same", 0) == 1 for r in results}
        return [
            [(i, j, same_by_id.get(f"{v}:{k}", False)) for k, (i, j) in enumerate(pairs)]
            for v, pairs in enumerate(pairs_per_video)
        ], len(requests)

    t0 = time.time()
    local_verdicts, n_local = judge([
        candidate_mention_pairs(video["mentions"], video["segments_by_id"], link_window_sec)
        for video in videos
    ])
    stats["link_local"] = (time.time() - t0, n_local)

    reid_pairs, reid_totals = [], []
    for video, verdicts in zip(videos, local_verdicts):
        video["tracklet_labels"] = cluster_mentions(video["mentions"], verdicts)
        pairs, total = candidate_tracklet_pairs(
            video["mentions"], video["tracklet_labels"], video["segments_by_id"], verdicts, max_reid_pairs,
        )
        reid_pairs.append(pairs)
        reid_totals.append(total)

    t0 = time.time()
    reid_verdicts, n_reid = judge(reid_pairs)
    stats["link_reid"] = (time.time() - t0, n_reid)

    for video, local, reid, pairs, total in zip(videos, local_verdicts, reid_verdicts, reid_pairs, reid_totals):
        mentions = video.pop("mentions")
        labels = cluster_mentions(mentions, local + reid)
        canonical = canonical_entities(mentions, labels)

        print(
            f"[video-graph] {video['video_id']} entity linking: {len(mentions)} mentions; "
            f"local {len(local)} pairs ({sum(v[2] for v in local)} same) -> "
            f"{len(set(video.pop('tracklet_labels')))} tracklets; "
            f"re-id {len(pairs)}/{total} pairs ({sum(v[2] for v in reid)} same) -> "
            f"{len(canonical)} entities"
        )
        if total > len(pairs):
            print(
                f"[video-graph] WARNING re-id cap hit: judged {len(pairs)} of {total} tracklet "
                f"pairs (--max_reid_pairs {max_reid_pairs}); entities that reappear may stay split"
            )

        segment_entity_ids = {seg["segment_id"]: [] for seg in video["segments"]}
        for entity in canonical:
            for segment_id in entity["segment_ids"]:
                if segment_id in segment_entity_ids:
                    segment_entity_ids[segment_id].append(entity["entity_id"])
        for seg in video["segments"]:
            seg["entity_ids"] = segment_entity_ids[seg["segment_id"]]
        video.pop("segments_by_id")
        video["entities"] = canonical

    return stats


def assemble_graph(video_id, fps, segments, entities, timeline):
    edges = []
    seen_adjacent = set()

    def add_adjacent(src, dst):
        if (src, dst) not in seen_adjacent:
            seen_adjacent.add((src, dst))
            edges.append({"type": "temporal_adjacent", "src": src, "dst": dst})

    # Adjacency along each level: consecutive scenes, and consecutive finest-level
    # segments (which crosses scene boundaries, e.g. last action -> next scene's first).
    scene_ids = [seg["segment_id"] for seg in segments if seg["level"] == "scene"]
    for ids in (scene_ids, timeline):
        for i in range(len(ids) - 1):
            add_adjacent(ids[i], ids[i + 1])

    for seg in segments:
        for child_id in seg["child_ids"]:
            edges.append({
                "type": "contains_segment",
                "src": seg["segment_id"],
                "dst": child_id,
            })
    for seg in segments:
        for entity_id in seg["entity_ids"]:
            edges.append({
                "type": "contains_entity",
                "segment_id": seg["segment_id"],
                "entity_id": entity_id,
            })

    return {
        "video_id": video_id,
        "fps": fps,
        "timeline": timeline,
        "segments": segments,
        "entities": entities,
        "edges": edges,
    }


def save_montages(segments, out_dir, video_id):
    """Write each segment's montage to <out_dir>/montages/<video_id>/ and
    record its path, relative to out_dir, as montage_path -- the Verifier
    (question_evaluate/evaluate.py) shows candidates through these."""
    rel_dir = os.path.join("montages", video_id)
    os.makedirs(os.path.join(out_dir, rel_dir), exist_ok=True)
    for seg in segments:
        rel_path = os.path.join(rel_dir, f"{seg['segment_id']}.png")
        seg["montage"].save(os.path.join(out_dir, rel_path))
        seg["montage_path"] = rel_path


def prepare_video(sc, video_path, args):
    """GPU/CPU-local half of graph building (decode, scene/action cuts,
    montages) -- runs in the background while the VLM servers caption the
    previous batch."""
    t0 = time.time()
    timings = {}
    video_id, segments, timeline = build_segments_for_video(
        sc, video_path, args.fps, args.min_segment_seconds, args.num_montage_frames,
        args.montage_frame_height, timings,
    )
    save_montages(segments, args.out_dir, video_id)
    timings["prepare_total"] = time.time() - t0
    print(
        f"[profile] {video_id}: sampled_frames={timings['sampled_frames']} segments={len(segments)} "
        f"timeline={len(timeline)} decode={timings['decode']:.1f}s scene_dino={timings['scene_dino']:.1f}s "
        f"action_raft={timings['action_raft']:.1f}s montage={timings['montage']:.1f}s "
        f"prepare_total={timings['prepare_total']:.1f}s"
    )
    return {"video_id": video_id, "video_path": video_path, "segments": segments, "timeline": timeline}


def prepare_batch(sc, video_paths, args):
    prepared, failed = [], []
    for video_path in video_paths:
        try:
            prepared.append(prepare_video(sc, video_path, args))
        except Exception as e:
            print(f"[video-graph] FAILED preparing {video_path}: {e}")
            failed.append(video_path)
    return prepared, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min_segment_seconds", type=float, default=1.0)
    parser.add_argument("--num_montage_frames", type=int, default=3)
    parser.add_argument(
        "--montage_frame_height",
        type=int,
        default=360,
        help="Downscale montage frames to at most this height (0 = native). Sets the image-token "
        "cost of every caption, same-entity and Verifier call.",
    )
    parser.add_argument(
        "--videos_per_batch",
        type=int,
        default=8,
        help="Videos whose caption / same-entity requests are sent to the servers together. The "
        "next batch is decoded and segmented in the background meanwhile.",
    )
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--base_port", type=int, default=int(os.getenv("RISE_CAPTION_SERVER_BASE_PORT", "6000")))
    parser.add_argument(
        "--link_window_sec",
        type=float,
        default=10.0,
        help="Stage-1 entity linking: directly compare mentions whose segments are at most this "
        "far apart, forming tracklets. Longer-range identity is handled by stage-2 re-id.",
    )
    parser.add_argument(
        "--max_reid_pairs",
        type=int,
        default=2000,
        help="Stage-2 entity re-identification: cap on tracklet pairs judged per video. Every "
        "same-type pair is judged below the cap; a warning is printed when it is hit.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild a video's graph even if <out_dir>/<video_id>.json already exists.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sc = SimilarityCurves(device=args.device)

    video_paths = find_videos(args.video_dir)
    print(f"[video-graph] found {len(video_paths)} videos under {args.video_dir}")

    pending = []
    for video_path in video_paths:
        cached_path = os.path.join(args.out_dir, f"{_video_id(video_path)}.json")
        if os.path.exists(cached_path) and not args.overwrite:
            print(f"[video-graph] skipping {video_path}, cached graph already at {cached_path}")
            continue
        pending.append(video_path)

    batch_size = max(1, args.videos_per_batch)
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    failed = []
    run_start = time.time()

    # One worker: batch b+1 is decoded/segmented while batch b is captioned
    # and linked on the VLM servers.
    with ThreadPoolExecutor(max_workers=1) as prep_pool:
        future = prep_pool.submit(prepare_batch, sc, batches[0], args) if batches else None
        for b in range(len(batches)):
            wait_start = time.time()
            prepared, prep_failed = future.result()
            prep_wait = time.time() - wait_start
            failed.extend(prep_failed)
            if b + 1 < len(batches):
                future = prep_pool.submit(prepare_batch, sc, batches[b + 1], args)
            if not prepared:
                continue

            batch_start = time.time()
            try:
                stats = caption_and_link(
                    prepared, args.num_servers, args.base_port, args.link_window_sec, args.max_reid_pairs,
                )
            except Exception as e:
                print(f"[video-graph] FAILED captioning/linking batch {b + 1}: {e}")
                failed.extend(video["video_path"] for video in prepared)
                continue

            for video in prepared:
                graph = assemble_graph(video["video_id"], args.fps, video["segments"], video["entities"], video["timeline"])
                out_path = os.path.join(args.out_dir, f"{graph['video_id']}.json")
                with open(out_path, "w") as f:
                    json.dump(graph, f, indent=2)
                print(
                    f"[video-graph] {graph['video_id']}: "
                    f"{sum(seg['level'] == 'scene' for seg in graph['segments'])} scenes, "
                    f"{sum(seg['level'] == 'action' for seg in graph['segments'])} actions, "
                    f"{len(graph['entities'])} entities -> {out_path}"
                )

            print(
                f"[profile] batch {b + 1}/{len(batches)}: videos={len(prepared)} "
                f"waited_for_prepare={prep_wait:.1f}s "
                + " ".join(f"{name}={secs:.1f}s/{n}req" for name, (secs, n) in stats.items())
                + f" vlm_total={time.time() - batch_start:.1f}s"
            )

    print(f"[profile] run: {len(pending) - len(failed)}/{len(pending)} videos in {time.time() - run_start:.1f}s")

    # Keep going past a bad video so one failure doesn't waste the rest of the
    # run, but exit non-zero so SLURM / the calling script see the failure.
    if failed:
        print(f"[video-graph] {len(failed)}/{len(video_paths)} videos failed:")
        for video_path in failed:
            print(f"[video-graph]   {video_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
