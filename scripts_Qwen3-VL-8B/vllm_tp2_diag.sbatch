#!/bin/bash
# Isolated diagnostic: does raw vLLM TP=2 engine startup hang on this node?
# Submit from the repo root: sbatch scripts_Qwen3-VL-8B/vllm_tp2_diag.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:2
#SBATCH --time=00:20:00
#SBATCH --job-name=rise-vllm-tp2-diag
#SBATCH --output=logs/smoke_test/%x_%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/smoke_test

echo "=== $(date) starting on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nvidia-smi topo -m || true

source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1

export RISE_ATTN_IMPLEMENTATION="${RISE_ATTN_IMPLEMENTATION:-sdpa}"
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export VLLM_LOGGING_LEVEL=DEBUG

echo "=== enforce_eager=1 attempt ==="
DIAG_ENFORCE_EAGER=1 timeout 600 python3 scripts_Qwen3-VL-8B/vllm_tp2_diag.py
echo "=== enforce_eager=1 exit code: $? ==="
