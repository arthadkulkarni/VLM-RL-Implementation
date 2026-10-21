import argparse
import os

import av
import numpy as np


def _frame(width, height, bg_color, shape_color, shape_xy, shape_size):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = bg_color
    x, y = shape_xy
    half = shape_size // 2
    x0, x1 = max(0, x - half), min(width, x + half)
    y0, y1 = max(0, y - half), min(height, y + half)
    arr[y0:y1, x0:x1] = shape_color
    return arr


def make_clip(path, width, height, fps, num_frames, bg_colors, shape_colors):
    """Two-scene synthetic clip: first half has a square sliding left->right
    on bg_colors[0], second half has a square sliding top->bottom on
    bg_colors[1] -- gives scene-detection (DINO embedding shift at the
    midpoint) and action/optical-flow (continuous motion within each half)
    signal for video_graph_builder's SimilarityCurves to pick up.
    """
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    half = num_frames // 2
    shape_size = min(width, height) // 4

    for i in range(num_frames):
        if i < half:
            t = i / max(1, half - 1)
            x = int(shape_size / 2 + t * (width - shape_size))
            y = height // 2
            bg, fg = bg_colors[0], shape_colors[0]
        else:
            t = (i - half) / max(1, num_frames - half - 1)
            x = width // 2
            y = int(shape_size / 2 + t * (height - shape_size))
            bg, fg = bg_colors[1], shape_colors[1]

        arr = _frame(width, height, bg, fg, (x, y), shape_size)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_clips", type=int, default=2)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--clip_fps", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=160)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    palette = [
        ((30, 30, 120), (230, 230, 30)),
        ((120, 30, 30), (30, 230, 230)),
        ((30, 120, 30), (230, 30, 230)),
        ((90, 90, 90), (250, 150, 20)),
    ]

    for i in range(args.num_clips):
        bg0, fg0 = palette[(2 * i) % len(palette)]
        bg1, fg1 = palette[(2 * i + 1) % len(palette)]
        out_path = os.path.join(args.out_dir, f"synthetic_clip_{i:02d}.mp4")
        make_clip(
            out_path,
            args.width,
            args.height,
            args.clip_fps,
            args.num_frames,
            bg_colors=(bg0, bg1),
            shape_colors=(fg0, fg1),
        )
        print(f"[make-synthetic-videos] wrote {out_path}")


if __name__ == "__main__":
    main()
