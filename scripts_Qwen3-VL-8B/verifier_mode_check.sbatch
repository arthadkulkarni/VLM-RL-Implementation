#!/bin/bash
# Compare the Verifier modes (sample / direct / greedy) on a questioner run's
# logged queries with verifier_mode_check.py. RUN is the run's STORAGE_PATH
# (it must hold questioner_rewards.jsonl); defaults to the
# questioner_check.sbatch run. Submit from the repo root.
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --job-name=rise-verifier-modes
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
RUN="${RUN:-${HDD_ROOT}/storage_RISE_q_check_cat}"
PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}" python3 scripts_Qwen3-VL-8B/verifier_mode_check.py \
  --reward_log "${RUN}/questioner_rewards.jsonl" \
  --model "${MODEL:-${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct}" \
  --cache_dir "${RUN}/verifier_mode_check_cache" \
  --out "${RUN}/verifier_mode_check_${SLURM_JOB_ID:-local}.json"
