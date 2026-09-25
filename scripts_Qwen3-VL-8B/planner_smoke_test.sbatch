#!/bin/bash
# Frozen Planner smoke test: synthetic clips -> build_video_graphs.sh (current
# scene/action hierarchy + captioning) -> frozen_planner on the fresh graphs.
# Validates the graph-builder -> Planner handoff on real builder output. Uses
# its own STORAGE_PATH so the older video-graph smoke graphs are left alone.
# Nothing downstream consumes C(x, q) yet (no Verifier), so this stops at
# enumerating candidates and measuring K.
# Submit from the repo root: sbatch scripts_Qwen3-VL-8B/planner_smoke_test.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
# Without this SLURM allocates 1 CPU, which the 4 vLLM servers, their image
# preprocessing and the builder/trainer all end up sharing.
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --job-name=rise-planner-smoke
#SBATCH --output=logs/planner_smoke/%x_%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/planner_smoke

echo "=== $(date) starting on $(hostname) ==="
source ../envs/RISE/bin/activate

export PYTHONUNBUFFERED=1
# See VLLMHijack.hijack() in verl/utils/vllm_utils.py.
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"
export RISE_NUM_GPUS="${RISE_NUM_GPUS:-4}"
export MODEL_DIR="${MODEL_DIR:-../pretrained_LM}"
export DATA_DIR="${DATA_DIR:-../datasets}"
export STORAGE_PATH="${STORAGE_PATH:-../storage_RISE_planner_smoke}"

VIDEO_DIR="${STORAGE_PATH}/synthetic_videos"
GRAPH_DIR="${STORAGE_PATH}/video_graphs"
export VIDEO_GRAPH_OUT_DIR="${GRAPH_DIR}"

echo "=== generating synthetic clips in ${VIDEO_DIR} ==="
python3 scripts_Qwen3-VL-8B/make_synthetic_videos.py \
  --out_dir "${VIDEO_DIR}" \
  --num_clips "${RISE_PLANNER_SMOKE_NUM_CLIPS:-4}"

echo "=== building video graphs into ${GRAPH_DIR} ==="
bash scripts_Qwen3-VL-8B/build_video_graphs.sh "${VIDEO_DIR}"

echo "=== running frozen planner on ${GRAPH_DIR} ==="
python3 - "${GRAPH_DIR}" <<'EOF'
import glob, json, os, sys
from frozen_planner.planner import CATEGORIES, plan

graph_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(graph_dir, "*.json")))
assert paths, f"no graphs under {graph_dir}"
for path in paths:
    graph = json.load(open(path))
    # Fresh graphs must carry the hierarchy; GraphView's flat fallback would
    # otherwise hide a schema regression.
    assert "timeline" in graph, f"{path}: no timeline -- built with pre-hierarchy code?"
    levels = sorted({seg["level"] for seg in graph["segments"]})
    print(f"--- {graph['video_id']}: timeline={len(graph['timeline'])} levels={levels} "
          f"entities={len(graph['entities'])}")
    for category in CATEGORIES:
        candidates = plan(graph, {"category": category})
        for c in candidates:
            assert all(c["captions"]), f"{c['candidate_id']}: empty caption"
        print(f"  {category:<12} K={len(candidates)}")
        if candidates:
            c = candidates[0]
            print(f"    e.g. {c['segment_ids']} basis={c['basis']} entities={c['entity_ids']}")
            for caption in c["captions"]:
                print(f"         caption: {caption[:120]}")
EOF

python3 -m frozen_planner.measure_k --graph_dir "${GRAPH_DIR}" --out "${STORAGE_PATH}/measure_k.json"
echo "=== planner smoke test done ==="
