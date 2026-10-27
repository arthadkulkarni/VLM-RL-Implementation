#!/bin/bash
# Backfill per-part summaries (video_graph_builder/part_summaries.py) into
# existing video graphs. By default it summarizes a COPY of SRC_GRAPH_DIR's
# graphs into GRAPH_DIR, so summaries can be read before the originals are
# touched; set GRAPH_DIR=SRC_GRAPH_DIR to backfill in place.
# Submit from the repo root:
#   sbatch scripts_Qwen3-VL-8B/part_summaries.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --job-name=rise-part-summaries
#SBATCH --output=logs/graph_profile/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/graph_profile
source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
SRC_GRAPH_DIR="${SRC_GRAPH_DIR:-${HDD_ROOT}/storage_RISE_small/video_graphs}"
GRAPH_DIR="${GRAPH_DIR:-${HDD_ROOT}/storage_RISE_parts_check/video_graphs}"
export STORAGE_PATH="${STORAGE_PATH:-$(dirname "${GRAPH_DIR}")}"
NUM_SERVERS="${RISE_NUM_GPUS:-1}"

if [ "${GRAPH_DIR}" != "${SRC_GRAPH_DIR}" ]; then
  mkdir -p "${GRAPH_DIR}"
  cp "${SRC_GRAPH_DIR}"/*.json "${GRAPH_DIR}/"
fi

source scripts_Qwen3-VL-8B/runtime_cleanup.sh
trap runtime_cleanup_all EXIT
RISE_GPU_IDLE_WORKER=0 RISE_REWARD_GPU_OFFSET=0 RISE_REWARD_SERVERS="${NUM_SERVERS}" \
  bash vllm_service_init/start.sh "${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct" "part_summaries"

python -m video_graph_builder.part_summaries --graph_dir "${GRAPH_DIR}" --num_servers "${NUM_SERVERS}" --base_port 6000
echo "=== $(date) done: summaries in ${GRAPH_DIR} ==="
