import av
import numpy as np


class _FrameBatch:
    """Minimal stand-in for decord's NDArray batch result."""

    def __init__(self, frames):
        self._frames = frames

    def asnumpy(self):
        return self._frames


class VideoReader:
    """decord.VideoReader-compatible wrapper backed by PyAV (`av`), used
    instead of decord because decord ships no aarch64/ARM64 wheels and this
    pipeline runs on GH200 (aarch64) nodes. Decodes the whole video once into
    memory on construction so frame_indices-based random access (get_batch)
    works the same way decord's did -- fine for the short clips this module
    processes, not intended for very long videos.
    """

    def __init__(self, video_path):
        container = av.open(video_path)
        stream = container.streams.video[0]
        self._avg_fps = float(stream.average_rate) if stream.average_rate else 30.0
        self._frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        container.close()

    def __len__(self):
        return len(self._frames)

    def get_avg_fps(self):
        return self._avg_fps

    def get_batch(self, indices):
        selected = np.stack([self._frames[i] for i in indices], axis=0)
        return _FrameBatch(selected)
