#!/bin/bash
# Two questioner GRPO steps from the base model on the mini graph, as LoRA
# (MODE=lora) or full fine-tuning (MODE=full) with otherwise identical
# settings, to compare peak GPU memory and check LoRA training. In LoRA mode
# it then checks the adapter-only checkpoint and merges it. Submit from the
# repo root:
#   MODE=lora sbatch scripts_Qwen3-VL-8B/lora_smoke.sbatch
#   MODE=full sbatch scripts_Qwen3-VL-8B/lora_smoke.sbatch
#SBATCH --account=bffz-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --job-name=rise-lora-smoke
#SBATCH --output=logs/diag/%x_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/diag
MODE="${MODE:?set MODE=lora or MODE=full}"

NVME_ROOT="${NVME_ROOT:-/work/nvme/bffz/akulkarni8/rise}"
HDD_ROOT="${HDD_ROOT:-/work/hdd/bffz/akulkarni8/rise}"
echo "=== $(date) MODE=${MODE} on $(hostname), commit $(git rev-parse --short HEAD) ==="
source scripts_Qwen3-VL-8B/use_env.sh

# Runtime settings as in mini_run.sbatch, with a smaller batch.
export PYTHONUNBUFFERED=1
export PROJECT_NAME="RISE_lora_smoke_${MODE}"
export STORAGE_PATH="${STORAGE_PATH:-${HDD_ROOT}/storage_RISE_lora_smoke_${MODE}}"
export RISE_VLLM_CACHE_DIR="${RISE_VLLM_CACHE_DIR:-${NVME_ROOT}/vllm_cache}"
export WANDB_MODE=offline
export RISE_ATTN_IMPLEMENTATION=sdpa RISE_ACTOR_PADDING_FREE=false RISE_FORCE_VIT_SDPA=1
export RISE_NUM_GPUS=4 RISE_RAY_NUM_CPUS=16 RAY_raylet_start_wait_time_s=300
export RISE_RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
export RISE_ROLLOUT_BATCH_SIZE="${RISE_ROLLOUT_BATCH_SIZE:-4}"
export RISE_GLOBAL_BATCH_SIZE="${RISE_GLOBAL_BATCH_SIZE:-2}"  # 2 updates per step
export RISE_ROLLOUT_N="${RISE_ROLLOUT_N:-4}"
if [ "${MODE}" = "lora" ]; then
  export RISE_LORA=1
fi

BASE="${NVME_ROOT}/pretrained_LM/Qwen3-VL-8B-Instruct"
GRAPH_DIR="${STORAGE_PATH}/video_graphs"
mkdir -p "${GRAPH_DIR}"
rsync -a "${MINI_GRAPH_DIR:-${HDD_ROOT}/storage_RISE_mini/video_graphs}/" "${GRAPH_DIR}/"
export RISE_QUESTIONER_TRAIN_FILES="${STORAGE_PATH}/questioner_videos_mini.parquet"
python3 - "${GRAPH_DIR}" "${RISE_QUESTIONER_TRAIN_FILES}" <<'EOF'
import glob, os, sys
import pandas as pd
paths = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(sys.argv[1], "*.json")))
pd.DataFrame([{"problem": os.path.basename(p), "answer": p} for p in paths] * 32).to_parquet(sys.argv[2], index=False)
EOF

bash scripts_Qwen3-VL-8B/questioner_train.sh "${BASE}" "${BASE}" smoke 2

STEP="${STORAGE_PATH}/models/smoke/global_step_2/actor"
echo "=== checkpoint ==="
du -sh "${STEP}"/* | sort -h
if [ "${MODE}" = "lora" ]; then
  MERGED="${STORAGE_PATH}/merged_step2"
  python3 scripts_Qwen3-VL-8B/merge_lora.py --base "${BASE}" --adapter "${STEP}/lora_adapter" --output "${MERGED}"
  python3 - "${BASE}" "${MERGED}" "${STEP}/lora_adapter" <<'EOF'
# The merge changes exactly the adapted language-model projections.
import json, sys
from safetensors import safe_open
base, merged, adapter = sys.argv[1:]
def index(path):
    return json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
def tensor(path, weight_map, name):
    with safe_open(f"{path}/{weight_map[name]}", "pt") as f:
        return f.get_tensor(name)
bmap, mmap = index(base), index(merged)
assert set(bmap) == set(mmap), "merged model has different weights"
targets = json.load(open(f"{adapter}/adapter_config.json"))["target_modules"]
changed = [n for n in bmap if not tensor(base, bmap, n).equal(tensor(merged, mmap, n))]
expected = [n for n in bmap if "visual" not in n and any(f".{t}.weight" in n for t in targets)]
print(f"[merge] {len(changed)} of {len(bmap)} weights changed; {len(expected)} adapted projections")
print("[merge] RESULT:", "PASS" if sorted(changed) == sorted(expected) else f"FAIL changed={sorted(set(changed) ^ set(expected))[:5]}")
EOF
  rm -rf "${MERGED}"
fi

echo "=== per-step metrics ==="
grep -a -E "Training step: global_step=|\s(pg_loss|entropy_loss|grad_norm|params_with_grad|non_lora_params_with_grad|lora_B_norm|max_memory_allocated_gb|max_memory_reserved_gb|overall):" \
  "logs/diag/rise-lora-smoke_${SLURM_JOB_ID}.log" | sed 's/.*\[A//' || true
grep -a -E "LoRA r=|Synced LoRA adapter" "logs/diag/rise-lora-smoke_${SLURM_JOB_ID}.log" | sed 's/.*\[0m //' | sort | uniq || true
echo "=== $(date) done ==="
