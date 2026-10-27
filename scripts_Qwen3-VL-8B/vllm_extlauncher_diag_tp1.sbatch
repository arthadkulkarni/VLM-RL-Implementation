#!/bin/bash
# TP=1 control for the vLLM external_launcher hang confirmed in job 3181193
# (TP=2 external_launcher hung at "Initializing a V1 LLM engine" even with
# enforce_eager=True). Single GPU, single rank, no cross-rank NCCL group
# beyond world_size=1 -- isolates whether TP itself (not external_launcher
# or the RISE hijack) is the trigger.
# Submit from the repo root: sbatch scripts_Qwen3-VL-8B/vllm_extlauncher_diag_tp1.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --time=00:15:00
#SBATCH --job-name=rise-vllm-extlauncher-diag-tp1
#SBATCH --output=logs/smoke_test/%x_%j.log

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/smoke_test

echo "=== $(date) starting on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

echo "=== TP=1, enforce_eager=1 attempt ==="
DIAG_TP_SIZE=1 DIAG_ENFORCE_EAGER=1 timeout 420 torchrun --nproc_per_node=1 --standalone \
  scripts_Qwen3-VL-8B/vllm_extlauncher_diag.py
echo "=== TP=1, enforce_eager=1 exit code: $? ==="
