#!/bin/bash
# Minimal end-to-end smoke test: 1 questioner step + 1 solver step, tiny
# batch/rollout sizes, sdpa attention (no flash-attn build needed).
# Submit from the repo root: sbatch scripts_Qwen3-VL-8B/smoke_test.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
# Without this SLURM allocates 1 CPU, which the 4 vLLM servers, their image
# preprocessing and the builder/trainer all end up sharing.
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --job-name=rise-smoke
#SBATCH --output=logs/smoke_test/%x_%j.log

set -euo pipefail

# Note: BASH_SOURCE[0] can't be used here to locate the repo root the way
# other scripts in this repo do it — sbatch copies this file into a spool
# dir before executing it, so BASH_SOURCE would point there instead of the
# real script path. Slurm already sets the job's initial CWD to the
# directory sbatch was invoked from ($SLURM_SUBMIT_DIR), so use that.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/smoke_test

echo "=== $(date) starting on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

source ../envs/RISE/bin/activate

# Redirected stdout is block-buffered by default, which can hide 30+ minutes
# of real progress behind silence in the log (as happened in job 3178940).
# Force line buffering so the log reflects what's actually happening live.
export PYTHONUNBUFFERED=1

echo "=== verifying torch CUDA on a real GPU ==="
python3 -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('matmul ok:', (torch.randn(4, 4, device='cuda') @ torch.randn(4, 4, device='cuda')).shape)
"

# No flash-attn build in this venv; run with sdpa attention and padding_free
# disabled (padding_free needs flash-attn's varlen kernels).
export RISE_ATTN_IMPLEMENTATION="${RISE_ATTN_IMPLEMENTATION:-sdpa}"
export RISE_ACTOR_PADDING_FREE="${RISE_ACTOR_PADDING_FREE:-false}"

# Without flash-attn or xformers, vLLM's Qwen2.5-VL/Qwen3-VL vision tower
# defaults to an XFORMERS attention backend on Hopper/GH200 and crashes with
# `ModuleNotFoundError: No module named 'xformers'` during profiling. See
# VLLMHijack.hijack() in verl/utils/vllm_utils.py, which patches the vision
# tower onto TORCH_SDPA when this is set.
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"

# 4 GPUs/node on DeltaAI (see cluster env memory), not the repo's default 8.
export RISE_NUM_GPUS="${RISE_NUM_GPUS:-4}"

# GH200 nodes report 288 logical CPUs; letting Ray auto-detect that makes it
# prestart ~288 worker processes and can hang ray.init() itself (see the
# comment in verl/trainer/main.py). A couple of GPU actors don't need that
# many CPU workers, so cap it well below the node's core count.
export RISE_RAY_NUM_CPUS="${RISE_RAY_NUM_CPUS:-16}"

# One big round, one micro iteration, one GRPO step for both stages.
export BIG_ROUNDS=1
export MICRO_ITERS=1
export MICRO_STEPS=1

# Tiny batch/rollout sizes validated to get through data loading + vLLM
# health checks on a 4-GPU node (job 3178183).
export RISE_ROLLOUT_BATCH_SIZE=8
export RISE_GLOBAL_BATCH_SIZE=4
export RISE_ROLLOUT_N=2

# Keep the solver's question-generation stage small too.
export QUESTION_NUM_SAMPLES_PER_MICRO="${QUESTION_NUM_SAMPLES_PER_MICRO:-32}"

export PROJECT_NAME="${PROJECT_NAME:-RISE_smoke}"
export WANDB_MODE=offline

echo "=== running RISE smoke test (RISE_ATTN_IMPLEMENTATION=${RISE_ATTN_IMPLEMENTATION}, RISE_NUM_GPUS=${RISE_NUM_GPUS}) ==="
bash scripts_Qwen3-VL-8B/main.sh
echo "=== smoke test exit code: $? ==="
