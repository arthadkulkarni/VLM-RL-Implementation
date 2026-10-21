from dataclasses import dataclass

import av
import numpy as np


@dataclass
class SampledVideo:
    """The frames the graph builder works from: every `interval`-th raw frame
    (interval = int(video_fps / fps)), decoded once. `frame_indices[k]` is the
    raw frame index of `frames[k]`; everything downstream (scene/action curves,
    montages, timestamps) indexes into these."""

    frames: list
    frame_indices: list
    video_fps: float
    total_frames: int


def sample_video(video_path, fps=1.0):
    """Stream-decode `video_path` with PyAV (decord ships no aarch64 wheels,
    and this runs on GH200 nodes), keeping only the frames sampled at `fps`.

    Every frame still has to be decoded (inter-coded video can't skip), but
    only kept frames are converted to RGB and held in memory: a 102-minute
    720x540 video at 30 fps keeps ~6k frames (~7 GB) instead of 184k (~214 GB).
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        video_fps = float(stream.average_rate) if stream.average_rate else 30.0
        interval = max(1, int(video_fps / fps))

        frames, frame_indices = [], []
        total = 0
        for idx, frame in enumerate(container.decode(stream)):
            if idx % interval == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                frame_indices.append(idx)
            total = idx + 1
    finally:
        container.close()

    return SampledVideo(frames=frames, frame_indices=frame_indices, video_fps=video_fps, total_frames=total)
