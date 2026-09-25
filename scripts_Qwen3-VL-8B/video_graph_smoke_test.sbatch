#!/bin/bash
# Video-graph pipeline smoke test: generates a couple of short synthetic
# clips (no real video corpus is staged on the cluster yet -- see
# make_synthetic_videos.py), then runs build_video_graphs.sh end-to-end
# (segmenting, RAFT/DINOv2 similarity curves, Qwen3-VL-8B captioning via the
# vllm_service_init caption servers, entity linking) to validate the pipeline
# before pointing it at a real video directory.
# Submit from the repo root: sbatch scripts_Qwen3-VL-8B/video_graph_smoke_test.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
# Without this SLURM allocates 1 CPU, which the 4 vLLM servers, their image
# preprocessing and the builder/trainer all end up sharing.
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --job-name=rise-video-graph-smoke
#SBATCH --output=logs/video_graph_smoke/%x_%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/video_graph_smoke

echo "=== $(date) starting on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

source ../envs/RISE/bin/activate

export PYTHONUNBUFFERED=1

# See VLLMHijack.hijack() in verl/utils/vllm_utils.py -- avoids the
# xformers ModuleNotFoundError on GH200 without flash-attn/xformers built.
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"

# 4 GPUs/node on DeltaAI; build_video_graphs.sh launches one caption server
# per GPU (RISE_REWARD_GPU_OFFSET=0), so all 4 are used for captioning here.
export RISE_NUM_GPUS="${RISE_NUM_GPUS:-4}"

export MODEL_DIR="${MODEL_DIR:-../pretrained_LM}"
export DATA_DIR="${DATA_DIR:-../datasets}"
STORAGE_TAG="RISE_video_graph_smoke"
export STORAGE_PATH="${STORAGE_PATH:-../storage_${STORAGE_TAG}}"

VIDEO_DIR="${STORAGE_PATH}/synthetic_videos"
mkdir -p "${VIDEO_DIR}"

echo "=== generating synthetic test clips in ${VIDEO_DIR} ==="
python3 scripts_Qwen3-VL-8B/make_synthetic_videos.py \
  --out_dir "${VIDEO_DIR}" \
  --num_clips "${RISE_VIDEO_GRAPH_SMOKE_NUM_CLIPS:-2}"

echo "=== running build_video_graphs.sh against ${VIDEO_DIR} (RISE_NUM_GPUS=${RISE_NUM_GPUS}) ==="
bash scripts_Qwen3-VL-8B/build_video_graphs.sh "${VIDEO_DIR}"
echo "=== video graph smoke test exit code: $? ==="
