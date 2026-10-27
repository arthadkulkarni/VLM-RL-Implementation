#!/bin/bash
# Score the Verifier's majority-vote pseudo-labels against CoMET-Bench GT
# (question_evaluate/validate_pseudo_labels.py). Submit from the repo root:
#   NAME=base SPLIT=eval sbatch scripts_Qwen3-VL-8B/validate_pseudo_labels.sbatch
#   NAME=selfplay_s4 MODEL=<.../global_step_4/actor/huggingface> SPLIT=eval sbatch ...
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=rise-pl-val
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/diag
source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
MODEL="${MODEL:-${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct}"
NAME="${NAME:-base}"
SPLIT="${SPLIT:-all}"
echo "=== $(date) ${NAME} (${MODEL}) split=${SPLIT}, commit $(git rev-parse --short HEAD) ==="

python3 question_evaluate/validate_pseudo_labels.py \
  --model "${MODEL}" --name "${NAME}" --split "${SPLIT}" \
  --out_dir "${OUT_DIR:-${HDD_ROOT}/pseudo_label_validation/${NAME}_${SPLIT}}"
