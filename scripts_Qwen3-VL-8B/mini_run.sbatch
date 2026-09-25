#!/bin/bash
# Mini end-to-end RISE run on prebuilt video graphs: main.sh's full
# questioner/solver alternation with MICRO_ITERS=2 x MICRO_STEPS=2 and tiny
# batches -- a proof of mechanism (graphs -> questioner + Planner/Verifier
# reward -> question generation/evaluation -> solver -> checkpoint handoff),
# not a training curve.
#
# Everything hot lives on /work/nvme: the venv, the base model, STORAGE_PATH
# (checkpoints) and the vLLM compile cache. On /work/hdd, 4 servers importing
# torch/vLLM at once took ~13 min per launch (job 3207016).
#
# Submit from the repo root, with MINI_GRAPH_DIR holding the graph JSONs (and
# their montages/ subdir) to train on:
#   MINI_GRAPH_DIR=../storage_RISE_mini/video_graphs sbatch scripts_Qwen3-VL-8B/mini_run.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
# Without this SLURM allocates 1 CPU, which the 4 vLLM servers, their image
# preprocessing and the trainer all end up sharing.
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --job-name=rise-mini
#SBATCH --output=logs/mini_run/%x_%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/mini_run

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
echo "=== $(date) starting on $(hostname), cpus=$(nproc) ==="
source "${RISE_ENV_DIR:-${NVME_ROOT}/envs/RISE}/bin/activate"
echo "python: $(command -v python)"

export PYTHONUNBUFFERED=1
export MODEL_DIR="${MODEL_DIR:-${NVME_ROOT}/pretrained_LM}"
export PROJECT_NAME="${PROJECT_NAME:-RISE_mini}"
export STORAGE_PATH="${STORAGE_PATH:-${NVME_ROOT}/storage_RISE_mini}"
export RISE_VLLM_CACHE_DIR="${RISE_VLLM_CACHE_DIR:-${NVME_ROOT}/vllm_cache}"
export WANDB_MODE=offline

# Same runtime settings as smoke_test.sbatch: no flash-attn/xformers in this
# venv, 4 GPUs/node, Ray capped well below the node's 288 logical CPUs.
export RISE_ATTN_IMPLEMENTATION="${RISE_ATTN_IMPLEMENTATION:-sdpa}"
export RISE_ACTOR_PADDING_FREE="${RISE_ACTOR_PADDING_FREE:-false}"
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"
export RISE_NUM_GPUS="${RISE_NUM_GPUS:-4}"
export RISE_RAY_NUM_CPUS="${RISE_RAY_NUM_CPUS:-16}"

export BIG_ROUNDS=1
export MICRO_ITERS="${MICRO_ITERS:-2}"
export MICRO_STEPS="${MICRO_STEPS:-2}"
export RISE_ROLLOUT_BATCH_SIZE="${RISE_ROLLOUT_BATCH_SIZE:-8}"
export RISE_GLOBAL_BATCH_SIZE="${RISE_GLOBAL_BATCH_SIZE:-4}"
# n=2 gave identical rollout pairs (zero GRPO advantage) in job 3207357.
export RISE_ROLLOUT_N="${RISE_ROLLOUT_N:-4}"
# Per GPU: question_generate.bash runs one generator per GPU, so 8 -> 32 questions.
export QUESTION_NUM_SAMPLES_PER_MICRO="${QUESTION_NUM_SAMPLES_PER_MICRO:-8}"

# Stage the graphs into STORAGE_PATH (montage paths are relative to the graph dir).
GRAPH_DIR="${STORAGE_PATH}/video_graphs"
mkdir -p "${GRAPH_DIR}"
rsync -a "${MINI_GRAPH_DIR:?set MINI_GRAPH_DIR}/" "${GRAPH_DIR}/"

# The questioner dataloader drops incomplete batches, and each micro iteration
# needs MICRO_STEPS full batches, so repeat the (few) graphs up to
# MINI_QUESTIONER_ROWS rows.
export RISE_QUESTIONER_TRAIN_FILES="${STORAGE_PATH}/questioner_videos_mini.parquet"
python3 - "${GRAPH_DIR}" "${RISE_QUESTIONER_TRAIN_FILES}" "${MINI_QUESTIONER_ROWS:-64}" <<'EOF'
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

start=$(date +%s)
bash scripts_Qwen3-VL-8B/main.sh
echo "=== mini run finished in $(( $(date +%s) - start ))s ==="
