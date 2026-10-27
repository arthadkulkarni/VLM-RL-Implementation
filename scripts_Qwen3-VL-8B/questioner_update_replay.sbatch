#!/bin/bash
# questioner_update_replay.py on its own, from a run's RISE_UPDATE_LOG dump:
# by default optimizer step 2 split into full / category-token-only / rest-only
# (--step2_parts). RUN is the run's STORAGE_PATH (it must hold update_log.jsonl
# and video_graphs/); defaults to the 3451840 replay run. Submit from the repo root.
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=01:30:00
#SBATCH --job-name=rise-q-replay2
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
RUN="${RUN:-${HDD_ROOT}/storage_RISE_q_check_replay}"
echo "=== $(date) starting on $(hostname), commit $(git rev-parse --short HEAD) ==="
PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}" python3 scripts_Qwen3-VL-8B/questioner_update_replay.py \
  --update_log "${RUN}/update_log.jsonl" \
  --graph "$(ls "${RUN}"/video_graphs/*.json | head -1)" \
  --model "${MODEL:-${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct}" \
  ${REPLAY_ARGS:---step2_parts}
echo "=== $(date) done ==="
