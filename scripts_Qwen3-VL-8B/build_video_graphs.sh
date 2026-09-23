#!/bin/bash
set -euo pipefail

# Standalone, one-time video-graph-building stage: segments each video under
# --video_dir, captions and entity-links the segments via the Qwen3-VL-8B
# vLLM servers, and writes one graph JSON per video. Run this before
# scripts_Qwen3-VL-8B/main.sh's training loop, since the video corpus doesn't
# change per training round -- do not call this from inside main.sh.
#
# Usage: bash scripts_Qwen3-VL-8B/build_video_graphs.sh <video_dir>

cd "$(dirname "$0")/.."

VIDEO_DIR="${1:?Usage: build_video_graphs.sh <video_dir>}"

export DATA_DIR="${DATA_DIR:-../datasets}"
export MODEL_DIR="${MODEL_DIR:-../pretrained_LM}"
STORAGE_TAG="${STORAGE_TAG:-RISE_Qwen3-VL-8B}"
export STORAGE_PATH="${STORAGE_PATH:-../storage_${STORAGE_TAG}}"
BASE_MODEL="${MODEL_DIR}/Qwen3-VL-8B-Instruct"

OUT_DIR="${VIDEO_GRAPH_OUT_DIR:-${STORAGE_PATH}/video_graphs}"
NUM_SERVERS="${RISE_NUM_GPUS:-4}"
# vllm_service_init/start.sh always binds servers to ports 6000..6000+n-1
# (its port base is not configurable) -- keep this in sync with that script.
BASE_PORT=6000

source scripts_Qwen3-VL-8B/runtime_cleanup.sh
trap runtime_cleanup_all EXIT

mkdir -p "${OUT_DIR}"

echo "[build-video-graphs] launching ${NUM_SERVERS} caption servers on ports ${BASE_PORT}..$((BASE_PORT + NUM_SERVERS - 1))"
RISE_REWARD_GPU_OFFSET=0 RISE_REWARD_SERVERS="${NUM_SERVERS}" \
  bash vllm_service_init/start.sh "${BASE_MODEL}" "video_graph_build"

python -m video_graph_builder.build_video_graph \
  --video_dir "${VIDEO_DIR}" \
  --out_dir "${OUT_DIR}" \
  --num_servers "${NUM_SERVERS}" \
  --base_port "${BASE_PORT}" \
  --fps "${RISE_VIDEO_GRAPH_FPS:-1.0}" \
  --min_segment_seconds "${RISE_VIDEO_GRAPH_MIN_SEGMENT_SECONDS:-1.0}" \
  --num_montage_frames "${RISE_VIDEO_GRAPH_MONTAGE_FRAMES:-3}"

echo "[build-video-graphs] done, graphs written to ${OUT_DIR}"
