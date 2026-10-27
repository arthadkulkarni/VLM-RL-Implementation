"""Download a small, topic-stratified subset of
ArthadK/Vision-Learning-Model-Video-Dataset and cut one fixed-length clip per
video for build_video_graphs.sh.

Clips are stream-copied (no re-encode), so they start at the keyframe at or
before the chosen offset. Run on the login node (needs network, no GPU):

  python3 scripts_Qwen3-VL-8B/prepare_video_subset.py \
    --raw_dir ../datasets/Vision-Learning-Model-Video-Dataset \
    --clip_dir ../storage_RISE_small/videos_5min --num_videos 32
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

import av
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "ArthadK/Vision-Learning-Model-Video-Dataset"


def video_topics(raw_dir):
    """Map a video file stem to its annotated topic, keyed by YouTube/Bilibili id."""
    path = hf_hub_download(REPO_ID, "data/annotations.json", repo_type="dataset", local_dir=raw_dir)
    topics = {}
    with open(path) as f:
        for entry in json.load(f):
            m = re.search(r"(?:v=|youtu\.be/|video/)([\w-]+)", entry.get("url", ""))
            if m:
                topics[m.group(1)] = entry.get("topic") or "unknown"
    return topics


def stratum(stem, topics):
    if stem in topics:
        return topics[stem]
    m = re.match(r"^\d+_(.+)_\d+$", stem)  # e.g. 0032_order_322
    return m.group(1) if m else "unknown"


def pick_videos(files, topics, num_videos, max_mb, seed):
    groups = defaultdict(list)
    for f in files:
        if f.size is None or f.size > max_mb * 1e6 or f.size < 5e6:
            continue
        stem = os.path.splitext(os.path.basename(f.path))[0]
        if not re.fullmatch(r"[\w-]+", stem):  # skip names with spaces/query strings
            continue
        groups[stratum(stem, topics)].append(f.path)
    rng = random.Random(seed)
    for paths in groups.values():
        paths.sort()
        rng.shuffle(paths)
    # Round-robin over strata so small topics are represented.
    picked, order = [], sorted(groups, key=lambda g: -len(groups[g]))
    while len(picked) < num_videos and any(groups.values()):
        for g in order:
            if groups[g] and len(picked) < num_videos:
                picked.append((g, groups[g].pop()))
    return picked


def cut_clip(src, dst, clip_sec, rng):
    with av.open(src) as inp:
        stream = inp.streams.video[0]
        duration = float(inp.duration / av.time_base) if inp.duration else None
        if duration is None or duration < 60:
            return None
        start = 0 if duration <= clip_sec else rng.randint(0, int(duration - clip_sec))
        out_path = dst.format(t=start)
        with av.open(out_path, "w") as out:
            ostream = out.add_stream_from_template(stream)
            if start:
                inp.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
            first = None
            for packet in inp.demux(stream):
                if packet.dts is None:
                    continue
                t = float(packet.pts * stream.time_base) if packet.pts is not None else float(packet.dts * stream.time_base)
                if first is None:
                    first = packet.dts
                if t > start + clip_sec:
                    break
                packet.dts -= first
                if packet.pts is not None:
                    packet.pts -= first
                packet.stream = ostream
                out.mux(packet)
        return out_path, start, duration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--clip_dir", required=True)
    p.add_argument("--num_videos", type=int, default=32)
    p.add_argument("--clip_sec", type=int, default=300)
    p.add_argument("--max_mb", type=float, default=900)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.clip_dir, exist_ok=True)
    topics = video_topics(args.raw_dir)
    files = [f for f in HfApi().list_repo_tree(REPO_ID, path_in_repo="videos", repo_type="dataset")
             if f.path.endswith(".mp4")]
    picked = pick_videos(files, topics, args.num_videos, args.max_mb, args.seed)
    rng = random.Random(args.seed)
    manifest = []
    for i, (group, path) in enumerate(picked):
        stem = os.path.splitext(os.path.basename(path))[0]
        src = hf_hub_download(REPO_ID, path, repo_type="dataset", local_dir=args.raw_dir)
        dst = os.path.join(args.clip_dir, f"{stem}_t{{t}}_{args.clip_sec // 60}min.mp4")
        result = cut_clip(src, dst, args.clip_sec, rng)
        if result is None:
            print(f"[{i + 1}/{len(picked)}] skip {stem}: too short or no duration")
            continue
        out_path, start, duration = result
        manifest.append({"source": path, "topic": group, "clip": os.path.basename(out_path),
                         "start_sec": start, "source_duration_sec": round(duration, 1)})
        print(f"[{i + 1}/{len(picked)}] {group:12s} {stem} -> {os.path.basename(out_path)} "
              f"({os.path.getsize(out_path) / 1e6:.0f} MB, source {duration / 60:.0f} min)")
    with open(os.path.join(args.clip_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"done: {len(manifest)} clips in {args.clip_dir}")


if __name__ == "__main__":
    main()
