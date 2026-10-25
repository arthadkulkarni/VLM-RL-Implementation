#!/bin/bash
# Two questioner GRPO steps from the base model on the mini graph, then
# questioner_drift_diag.py on base vs the new step 2. Checks the fixes in
# 0980592: entropy_loss > 0, step-1/step-2 rollouts differ, `bounded` rollouts
# pass the format gate, and the policy moves toward its rewarded rollouts.
#
# STORAGE_PATH defaults to hdd: the nvme allocation is near its quota.
# Submit from the repo root:
#   sbatch scripts_Qwen3-VL-8B/questioner_check.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --job-name=rise-q-check
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/diag

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
echo "=== $(date) starting on $(hostname), commit $(git rev-parse --short HEAD) ==="
source scripts_Qwen3-VL-8B/use_env.sh

# Runtime settings as in mini_run.sbatch.
export PYTHONUNBUFFERED=1
export PROJECT_NAME="${PROJECT_NAME:-RISE_q_check}"
export STORAGE_PATH="${STORAGE_PATH:-${HDD_ROOT}/storage_RISE_q_check}"
export RISE_VLLM_CACHE_DIR="${RISE_VLLM_CACHE_DIR:-${NVME_ROOT}/vllm_cache}"
export WANDB_MODE=offline
export RISE_ATTN_IMPLEMENTATION="${RISE_ATTN_IMPLEMENTATION:-sdpa}"
export RISE_ACTOR_PADDING_FREE="${RISE_ACTOR_PADDING_FREE:-false}"
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"
export RISE_NUM_GPUS="${RISE_NUM_GPUS:-4}"
export RISE_RAY_NUM_CPUS="${RISE_RAY_NUM_CPUS:-16}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-300}"
export RISE_RAY_TMPDIR="${RISE_RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}"
export RISE_ROLLOUT_BATCH_SIZE="${RISE_ROLLOUT_BATCH_SIZE:-8}"
export RISE_GLOBAL_BATCH_SIZE="${RISE_GLOBAL_BATCH_SIZE:-4}"
export RISE_ROLLOUT_N="${RISE_ROLLOUT_N:-4}"

BASE="${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct"
GRAPH_DIR="${STORAGE_PATH}/video_graphs"
mkdir -p "${GRAPH_DIR}"
rsync -a "${MINI_GRAPH_DIR:-${HDD_ROOT}/storage_RISE_mini/video_graphs}/" "${GRAPH_DIR}/"

export RISE_QUESTIONER_TRAIN_FILES="${STORAGE_PATH}/questioner_videos_mini.parquet"
python3 - "${GRAPH_DIR}" "${RISE_QUESTIONER_TRAIN_FILES}" 64 <<'EOF'
import glob, os, sys
import pandas as pd
graph_dir, out, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
paths = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(graph_dir, "*.json")))
assert paths, f"no graphs in {graph_dir}"
records = [{"problem": os.path.splitext(os.path.basename(p))[0], "answer": p} for p in paths]
records = [records[i % len(records)] for i in range(max(rows, len(records)))]
pd.DataFrame(records).to_parquet(out, index=False)
print(f"questioner parquet: {len(records)} rows from {len(paths)} graphs -> {out}")
EOF

# Base model as both the reward-side solver and the starting questioner.
bash scripts_Qwen3-VL-8B/questioner_train.sh "${BASE}" "${BASE}" q_check 2

PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES=0 \
  python3 scripts_Qwen3-VL-8B/questioner_drift_diag.py \
  --graph "$(ls "${GRAPH_DIR}"/*.json | head -1)" \
  --reward_log "${STORAGE_PATH}/questioner_rewards.jsonl" \
  --models "base=${BASE}" "step2=${STORAGE_PATH}/models/q_check/global_step_2/actor/huggingface"
echo "=== $(date) done ==="
