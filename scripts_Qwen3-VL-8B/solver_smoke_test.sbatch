#!/bin/bash
# Solver-only smoke test: 1 GRPO step on the solver stage, using the base
# model as both solver and questioner checkpoint (question_generate runs
# against the untrained questioner just to produce a small pseudo-labeled
# batch). Validates the rollout/train path after the TP=2 -> TP=1 vLLM
# rollout fix (see rise_gpu_config memory / job 3181231 segfault).
# Submit from the repo root: sbatch scripts_Qwen3-VL-8B/solver_smoke_test.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
# Without this SLURM allocates 1 CPU, which the 4 vLLM servers, their image
# preprocessing and the builder/trainer all end up sharing.
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --job-name=rise-solver-smoke
#SBATCH --output=logs/smoke_test/%x_%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/smoke_test

echo "=== $(date) starting on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

source scripts_Qwen3-VL-8B/use_env.sh

export PYTHONUNBUFFERED=1

# No flash-attn build in this venv; run with sdpa attention and padding_free
# disabled (padding_free needs flash-attn's varlen kernels).
export RISE_ATTN_IMPLEMENTATION="${RISE_ATTN_IMPLEMENTATION:-sdpa}"
export RISE_ACTOR_PADDING_FREE="${RISE_ACTOR_PADDING_FREE:-false}"

# See VLLMHijack.hijack() in verl/utils/vllm_utils.py.
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"

# 4 GPUs/node on DeltaAI, not the repo's default 8.
export RISE_NUM_GPUS="${RISE_NUM_GPUS:-4}"

# Cap Ray's auto-detected CPU worker count (GH200 nodes report 288 logical CPUs).
export RISE_RAY_NUM_CPUS="${RISE_RAY_NUM_CPUS:-16}"

# One GRPO step.
export MICRO_STEPS=1

# Tiny batch/rollout sizes validated on a 4-GPU node (job 3178183).
export RISE_ROLLOUT_BATCH_SIZE=8
export RISE_GLOBAL_BATCH_SIZE=4
export RISE_ROLLOUT_N=2
export QUESTION_NUM_SAMPLES_PER_MICRO="${QUESTION_NUM_SAMPLES_PER_MICRO:-32}"

export MODEL_DIR="${MODEL_DIR:-../pretrained_LM}"
export DATA_DIR="${DATA_DIR:-../datasets}"
export PROJECT_NAME="${PROJECT_NAME:-RISE_solver_smoke}"
export STORAGE_PATH="${STORAGE_PATH:-../storage_${PROJECT_NAME}}"
export WANDB_MODE=offline

BASE_MODEL="${MODEL_DIR}/Qwen3-VL-8B-Instruct"

mkdir -p "${STORAGE_PATH}/evaluation" "${STORAGE_PATH}/generated_question" \
  "${STORAGE_PATH}/local_parquet" "${STORAGE_PATH}/temp_results" \
  "${STORAGE_PATH}/wandb" "${STORAGE_PATH}/models"

echo "=== running solver-only smoke test (base model as solver+questioner, RISE_NUM_GPUS=${RISE_NUM_GPUS}) ==="
bash scripts_Qwen3-VL-8B/solver_train.sh \
  "${BASE_MODEL}" \
  "${BASE_MODEL}" \
  "solver_smoke_test" \
  1
echo "=== solver smoke test exit code: $? ==="
