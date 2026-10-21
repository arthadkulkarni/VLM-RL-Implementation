"""Build the questioner's training parquet from a directory of video graphs:
one row per video, with the graph's absolute .json path in the `answer` column
(verl/utils/dataset.py builds the prompt from it and passes it to cot_val.py as
ground_truth) and the video id in `problem`.
"""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--output", required=True, help="Output .parquet path")
    args = parser.parse_args()

    graph_paths = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(args.graph_dir, "*.json")))
    if not graph_paths:
        raise FileNotFoundError(f"No video graph .json files found in {args.graph_dir}")

    rows = [{"problem": os.path.splitext(os.path.basename(p))[0], "answer": p} for p in graph_paths]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"Wrote {len(rows)} video graphs to {args.output}")


if __name__ == "__main__":
    main()
