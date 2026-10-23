#!/bin/bash
# Run questioner_drift_diag.py on the base model and the mini e2e run's
# questioner checkpoints. Submit from the repo root.
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
RUN="${NVME_ROOT}/storage_RISE_mini_e2e"
Q="${RUN}/models/Qwen3-VL-8B-Instruct_q_b1"
PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}" python3 scripts_Qwen3-VL-8B/questioner_drift_diag.py \
  --graph "${RUN}/video_graphs/0032_order_322_t89_5min.json" \
  --models "base=${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct" \
           "step2=${Q}/global_step_2/actor/huggingface" \
           "step4=${Q}/global_step_4/actor/huggingface"
