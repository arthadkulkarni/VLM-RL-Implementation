#!/bin/bash
# Self-evolution curve of a LoRA run (question_evaluate/eval_rounds.py): every
# round's verifier adapter scored on the CoMET-Bench eval split, rounds spread
# over the node's GPUs, then rounds.csv/.md. Rerun to resume. Submit from the
# repo root with the run's MODEL_SAVE_ROOT:
#   MODEL_SAVE_ROOT=<STORAGE_PATH>/models sbatch scripts_Qwen3-VL-8B/eval_rounds.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --job-name=rise-eval-rounds
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/diag
source scripts_Qwen3-VL-8B/use_env.sh
export PYTHONUNBUFFERED=1
export RISE_FORCE_VIT_SDPA="${RISE_FORCE_VIT_SDPA:-1}"

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
MODEL_SAVE_ROOT="${MODEL_SAVE_ROOT:?set MODEL_SAVE_ROOT to the STORAGE_PATH/models of the run}"
OUT_DIR="${OUT_DIR:-$(dirname "${MODEL_SAVE_ROOT}")/eval_rounds}"
NUM_GPUS="${NUM_GPUS:-4}"
args=(--model "${MODEL_DIR:-${NVME_ROOT}/pretrained_LM}/Qwen3-VL-8B-Instruct"
      --model_save_root "${MODEL_SAVE_ROOT}" --out_dir "${OUT_DIR}" --lineage "${LINEAGE:-auto}")
echo "=== $(date) ${MODEL_SAVE_ROOT} -> ${OUT_DIR}, commit $(git rev-parse --short HEAD) ==="

pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="${gpu}" VLLM_CACHE_ROOT="${RISE_VLLM_CACHE_DIR:-${NVME_ROOT}/vllm_cache}/eval_rounds_gpu${gpu}" \
    python3 question_evaluate/eval_rounds.py "${args[@]}" --shard "${gpu}" --num_shards "${NUM_GPUS}" --no_table &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
python3 question_evaluate/eval_rounds.py "${args[@]}" --table_only
exit "${status}"
