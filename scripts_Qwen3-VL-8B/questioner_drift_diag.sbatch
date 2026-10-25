#!/bin/bash
# Run questioner_drift_diag.py on the base model and a questioner run's
# checkpoints. RUN is the run's STORAGE_PATH (it must hold
# questioner_rewards.jsonl); Q and STEPS pick the checkpoints. Defaults to the
# questioner_check.sbatch run. Submit from the repo root.
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --job-name=rise-q-drift
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
RUN="${RUN:-${HDD_ROOT}/storage_RISE_q_check_cat}"
Q="${Q:-${RUN}/models/q_check}"
models=("base=${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct")
for step in ${STEPS:-2}; do
  models+=("step${step}=${Q}/global_step_${step}/actor/huggingface")
done
PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}" python3 scripts_Qwen3-VL-8B/questioner_drift_diag.py \
  --graph "$(ls "${RUN}"/video_graphs/*.json | head -1)" \
  --reward_log "${RUN}/questioner_rewards.jsonl" \
  --models "${models[@]}"
