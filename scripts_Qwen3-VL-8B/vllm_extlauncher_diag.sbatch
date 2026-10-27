#!/bin/bash
# Faithful repro of the rollout engine construction that stalled in job
# 3180735 (FSDP-style NCCL group + vLLM external_launcher + RISE_FORCE_VIT_SDPA),
# without the ~1.5h of dataset filtering / reward-server startup the full
# smoke test needs to reach that point.
#
# Earlier 420-480s timeouts on this Lustre-backed venv weren't long enough to
# tell "genuinely deadlocked" apart from "just slow imports/engine init on a
# cold node" -- every run so far has shown multi-minute stalls at plain
# Python imports, not just at NCCL/engine init. This run uses a much longer
# timeout to settle that question.
#
# Submit from the repo root: sbatch scripts_Qwen3-VL-8B/vllm_extlauncher_diag.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:2
#SBATCH --time=00:45:00
#SBATCH --job-name=rise-vllm-extlauncher-diag-long
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

echo "=== enforce_eager=1 attempt, 30 min timeout ==="
DIAG_ENFORCE_EAGER=1 timeout 1800 torchrun --nproc_per_node=2 --standalone \
  scripts_Qwen3-VL-8B/vllm_extlauncher_diag.py
echo "=== enforce_eager=1 exit code: $? ==="
