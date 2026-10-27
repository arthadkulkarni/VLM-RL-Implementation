#!/bin/bash
# questioner_update_diag.py on a questioner run's step-1 rollouts: does the
# first update's direction (plain gradient vs AdamW's first step) explain the
# category collapse? RUN is the run's STORAGE_PATH (it must hold
# questioner_rewards.jsonl with raw responses); defaults to the
# hierarchical-search q-check run. Submit from the repo root.
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --job-name=rise-q-update
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
RUN="${RUN:-${HDD_ROOT}/storage_RISE_q_check_hier}"
PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}" python3 scripts_Qwen3-VL-8B/questioner_update_diag.py \
  --reward_log "${RUN}/questioner_rewards.jsonl" \
  --graph "$(ls "${RUN}"/video_graphs/*.json | head -1)" \
  --model "${MODEL:-${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct}"
