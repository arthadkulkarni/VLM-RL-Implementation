import argparse
import base64
import glob
import io
import json
import os
import sys
import time

from PIL import Image

from video_graph_builder.entity_linker import link_entities
from video_graph_builder.graph_client import caption_entities_batch
from video_graph_builder.montage import build_montage, sample_frame_indices
from video_graph_builder.segmenter import (
    merge_short_segments,
    build_segment_hierarchy,
    scene_spans_from_boundaries,
)
from video_graph_builder.similarity_curves import SimilarityCurves
from video_graph_builder.video_io import VideoReader

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


def build_segments_for_video(sc, video_path, fps, min_seconds, num_montage_frames):
    """Returns (video_id, segments, timeline). `segments` holds nodes at both
    levels -- "scene" (DINOv2 appearance cuts) and "action" (RAFT motion cuts
    inside one scene) -- linked by parent_id / child_ids. `timeline` is the
    ordered, non-overlapping list of finest-level segment_ids covering the
    video: a scene's actions where it was split, else the scene itself (see
    build_segment_hierarchy for the rule).
    """
    scene_result = sc.compute_scene(video_path, fps=fps)
    scene_spans = scene_spans_from_boundaries(scene_result["boundaries"], scene_result["num_sampled_frames"])
    scene_spans = merge_short_segments(scene_spans, max(2, round(min_seconds * fps)))
    action_results = sc.compute_action_for_segments(video_path, fps=fps, segments=scene_spans)
    hierarchy = build_segment_hierarchy(scene_spans, action_results, min_seconds=min_seconds, fps=fps)

    frame_indices = scene_result["frame_indices"]
    vr = VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    video_id = _video_id(video_path)

    def make_segment(segment_id, level, start_idx, end_idx, parent_id):
        sample_idxs = sample_frame_indices(start_idx, end_idx, num_montage_frames)
        raw_idxs = [frame_indices[min(idx, len(frame_indices) - 1)] for idx in sample_idxs]
        raw_frames = vr.get_batch(raw_idxs).asnumpy()
        montage = build_montage([Image.fromarray(f) for f in raw_frames])

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

    return video_id, segments, timeline


def caption_and_link(segments, num_servers, base_port):
    request_items = [
        {"segment_id": seg["segment_id"], "image": _pil_to_base64(seg["montage"])}
        for seg in segments
    ]
    raw_results = caption_entities_batch(request_items, num_servers=num_servers, base_port=base_port)
    parsed_by_id = {item.get("segment_id"): item for item in raw_results}

    mentions = []
    for seg in segments:
        parsed = parsed_by_id.get(seg["segment_id"], {"caption": "", "entities": []})
        seg["caption"] = parsed.get("caption", "")
        for entity in parsed.get("entities", []):
            mentions.append({**entity, "segment_id": seg["segment_id"]})
        del seg["montage"]

    canonical_entities = link_entities(mentions)

    segment_entity_ids = {seg["segment_id"]: [] for seg in segments}
    for entity in canonical_entities:
        for segment_id in entity["segment_ids"]:
            if segment_id in segment_entity_ids:
                segment_entity_ids[segment_id].append(entity["entity_id"])
    for seg in segments:
        seg["entity_ids"] = segment_entity_ids[seg["segment_id"]]

    return segments, canonical_entities


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


def build_graph_for_video(sc, video_path, args):
    video_id, segments, timeline = build_segments_for_video(
        sc, video_path, args.fps, args.min_segment_seconds, args.num_montage_frames
    )
    segments, entities = caption_and_link(segments, args.num_servers, args.base_port)
    return assemble_graph(video_id, args.fps, segments, entities, timeline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min_segment_seconds", type=float, default=1.0)
    parser.add_argument("--num_montage_frames", type=int, default=3)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--base_port", type=int, default=int(os.getenv("RISE_CAPTION_SERVER_BASE_PORT", "6000")))
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

    failed = []
    for video_path in video_paths:
        cached_path = os.path.join(args.out_dir, f"{_video_id(video_path)}.json")
        if os.path.exists(cached_path) and not args.overwrite:
            print(f"[video-graph] skipping {video_path}, cached graph already at {cached_path}")
            continue

        start_time = time.time()
        try:
            graph = build_graph_for_video(sc, video_path, args)
        except Exception as e:
            print(f"[video-graph] FAILED on {video_path}: {e}")
            failed.append(video_path)
            continue

        out_path = os.path.join(args.out_dir, f"{graph['video_id']}.json")
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)

        elapsed = time.time() - start_time
        print(
            f"[video-graph] {graph['video_id']}: "
            f"{sum(seg['level'] == 'scene' for seg in graph['segments'])} scenes, "
            f"{sum(seg['level'] == 'action' for seg in graph['segments'])} actions, "
            f"{len(graph['entities'])} entities -> {out_path} ({elapsed:.1f}s)"
        )

    # Keep going past a bad video so one failure doesn't waste the rest of the
    # run, but exit non-zero so SLURM / the calling script see the failure.
    if failed:
        print(f"[video-graph] {len(failed)}/{len(video_paths)} videos failed:")
        for video_path in failed:
            print(f"[video-graph]   {video_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
