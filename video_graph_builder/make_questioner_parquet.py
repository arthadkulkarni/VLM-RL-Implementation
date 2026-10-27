"""Build the questioner's training parquet from a directory of video graphs.
The `answer` column holds a graph_ref (verl/utils/dataset.py builds the
prompt from it and passes it to cot_val.py as ground_truth) and `problem`
the video id.

A video whose timeline fits in --max_context_segments gets one row, its
graph path. A longer one is shown hierarchically (build_graph_context) with
one focus window in full, so it gets one row per window,
"<path>#focus=<k>": each epoch then asks about every part of the video once.
"""

import argparse
import glob
import os

import pandas as pd

from video_verifier.prompts import focus_windows, graph_ref, load_graph, needs_focus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--output", required=True, help="Output .parquet path")
    parser.add_argument(
        "--max_context_segments", type=int, default=int(os.getenv("RISE_MAX_CONTEXT_SEGMENTS", "120")),
        help="Must match the questioner's RISE_MAX_CONTEXT_SEGMENTS (verl/utils/dataset.py).",
    )
    args = parser.parse_args()

    graph_paths = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(args.graph_dir, "*.json")))
    if not graph_paths:
        raise FileNotFoundError(f"No video graph .json files found in {args.graph_dir}")

    rows, long_videos = [], 0
    for path in graph_paths:
        video_id = os.path.splitext(os.path.basename(path))[0]
        graph = load_graph(path)
        if needs_focus(graph, args.max_context_segments):
            long_videos += 1
            rows += [{"problem": video_id, "answer": graph_ref(path, k)} for k in range(len(focus_windows(graph)))]
        else:
            rows.append({"problem": video_id, "answer": graph_ref(path)})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"Wrote {len(rows)} rows from {len(graph_paths)} video graphs ({long_videos} long, one row per focus window) "
          f"to {args.output}")


if __name__ == "__main__":
    main()
