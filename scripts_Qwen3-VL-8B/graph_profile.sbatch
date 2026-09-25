#!/bin/bash
# Graph-builder profile: build the video graph(s) under $PROFILE_VIDEO_DIR with
# per-stage [profile] timings, then measure Planner K on the result.
# Submit from the repo root:
#   PROFILE_VIDEO_DIR=../storage_RISE_mini/videos_5min sbatch scripts_Qwen3-VL-8B/graph_profile.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
# Without this SLURM allocates 1 CPU, which the 4 vLLM servers, their image
# preprocessing and the builder all end up sharing.
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --job-name=rise-graph-profile
#SBATCH --output=logs/graph_profile/%x_%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/graph_profile

echo "=== $(date) starting on $(hostname), cpus=$(nproc) ==="
source ../envs/RISE/bin/activate

export PYTHONUNBUFFERED=1
# See VLLMHijack.hijack() in verl/utils/vllm_utils.py.
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"
export RISE_NUM_GPUS="${RISE_NUM_GPUS:-4}"
export MODEL_DIR="${MODEL_DIR:-../pretrained_LM}"
export STORAGE_PATH="${STORAGE_PATH:-../storage_RISE_mini}"

VIDEO_DIR="${PROFILE_VIDEO_DIR:?set PROFILE_VIDEO_DIR}"
GRAPH_DIR="${STORAGE_PATH}/video_graphs"
export VIDEO_GRAPH_OUT_DIR="${GRAPH_DIR}"

echo "=== building video graphs for ${VIDEO_DIR} into ${GRAPH_DIR} ==="
start=$(date +%s)
bash scripts_Qwen3-VL-8B/build_video_graphs.sh "${VIDEO_DIR}"
echo "[profile] build_video_graphs.sh wall (incl. server startup): $(( $(date +%s) - start ))s"

echo "=== VLM server request timings ==="
grep -hE "generating batch|Adding requests: 100%.*\]$|Processed prompts: 100%" \
  "${STORAGE_PATH}"/temp_results/vllm_server_600*.log | tr '\r' '\n' | grep -E "generating batch|100%" | tail -40 || true

echo "=== measuring Planner K ==="
python3 -m frozen_planner.measure_k --graph_dir "${GRAPH_DIR}" --out "${STORAGE_PATH}/measure_k.json"

echo "=== graph profile done ==="
