#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"

solver_model_path=$1
questioner_model_path=$2
experiment_name=$3
train_steps=${4:-${GLOBAL_STEP:-5}}
load_checkpoint_path=${5:-}

# Data and storage roots inherited from main.sh. They can also be overridden when this
# stage script is launched directly.
data_dir="${DATA_DIR:-../datasets}"
export PROJECT_NAME="${PROJECT_NAME:-RISE_Qwen3-VL-8B}"
export STORAGE_PATH="${STORAGE_PATH:-../storage_${PROJECT_NAME}}"
model_save_root="${MODEL_SAVE_ROOT:-${STORAGE_PATH}/models}"

source scripts_Qwen3-VL-8B/runtime_cleanup.sh
trap runtime_cleanup_all EXIT

mkdir -p "${model_save_root}"

# Distributed training timeout for long VLM rollout/reward steps.
export TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC:-3600}"

# Keep checkpoint loading/saving on GPU unless users need CPU offload for memory.
export VERL_CKPT_CPU_OFFLOAD="${VERL_CKPT_CPU_OFFLOAD:-0}"

# Explicit role marker used by the dataset pipeline. In questioner training, each
# row is one video; the prompt is built from its graph and the graph path is the
# only thing passed to the reward (see verl/utils/dataset.py).
export RISE_TRAINING_ROLE=questioner
export QUESTIONER_MASK_SOURCE_QA=1

# Unique ID for the vLLM reward servers used in this questioner stage.
export RUN_ID="${RUN_ID:-$(date +%s%N)}"

# GPUs are split between questioner training and solver-reward judge servers on the
# same node. RISE_NUM_GPUS is the total available; RISE_QUESTIONER_TRAIN_GPUS (default
# half) go to training on IDs 0..train_gpus-1, and the remainder run one reward server
# each on IDs train_gpus..num_gpus-1 (ports 6000..6000+reward_gpus-1).
num_gpus="${RISE_NUM_GPUS:-8}"
train_gpus="${RISE_QUESTIONER_TRAIN_GPUS:-$((num_gpus / 2))}"
reward_gpus=$((num_gpus - train_gpus))
if [ "${train_gpus}" -lt 1 ] || [ "${reward_gpus}" -lt 1 ]; then
  echo "ERROR: need at least 1 training GPU and 1 reward-server GPU (got num_gpus=${num_gpus}, train_gpus=${train_gpus})" >&2
  exit 1
fi
export RISE_REWARD_SERVERS="${reward_gpus}"
train_cuda_devices="$(seq -s, 0 $((train_gpus - 1)))"

echo "Train questioner: ${experiment_name}"
echo "Solver model for reward: ${solver_model_path}"
echo "Questioner model: ${questioner_model_path}"
echo "Target steps: ${train_steps}"
echo "GPU split: ${train_gpus} training (${train_cuda_devices}), ${reward_gpus} reward server(s)"

RISE_REWARD_GPU_OFFSET="${train_gpus}" RISE_REWARD_SERVERS="${reward_gpus}" \
  bash vllm_service_init/start.sh "${solver_model_path}" "${RUN_ID}"

# Questioner training data: one row per video graph (answer = graph .json path).
# Built from the graphs written by build_video_graphs.sh unless provided.
questioner_train_files="${RISE_QUESTIONER_TRAIN_FILES:-${STORAGE_PATH}/questioner_videos.parquet}"
if [ -z "${RISE_QUESTIONER_TRAIN_FILES:-}" ]; then
  python3 video_graph_builder/make_questioner_parquet.py \
    --graph_dir "${STORAGE_PATH}/video_graphs" --output "${questioner_train_files}"
fi

trainer_args=(
  config=train_examples/cot_config.yaml

  # Unlabeled video pool used to train the questioner.
  "data.train_files=${questioner_train_files}"
  "data.val_files=${data_dir}/parquet/zli12321__mmstar" # not used dataset, just for compatibility
  data.prompt_key=problem
  data.answer_key=answer
  data.image_key=images
  "worker.actor.model.model_path=${questioner_model_path}"

  # padding_free relies on flash-attn's varlen kernels (via RISE_ATTN_IMPLEMENTATION);
  # a smoke test without flash-attn built should set this to false.
  "worker.actor.padding_free=${RISE_ACTOR_PADDING_FREE:-true}"

  # The questioner trains on half the GPUs (the rest serve rewards), so FSDP
  # shards the 8B model's fp32 weights + Adam state over only 2 GPUs on a
  # 4-GPU node (~48 GB each). Left resident, the rollout engine can't re-claim
  # its KV cache after the first update (CUDA OOM in cumem wake_up, job
  # 3207356). Offload both between phases; on GH200 the CPU<->GPU copy runs
  # over NVLink-C2C.
  "worker.actor.offload.offload_params=${RISE_QUESTIONER_OFFLOAD_PARAMS:-true}"
  "worker.actor.offload.offload_optimizer=${RISE_QUESTIONER_OFFLOAD_OPTIMIZER:-true}"

  # Questioner prompts are text-only (video-graph context), so the vision tower
  # never gets gradients and has no optimizer state; resuming a checkpoint then
  # fails under FSDP unless it is frozen (which sets use_orig_params=True).
  "worker.actor.model.freeze_vision_tower=${RISE_QUESTIONER_FREEZE_VISION:-true}"

  # Maximum context length for questioner rollouts.
  worker.rollout.max_model_len=12288

  # Overridable batch/rollout sizing (defaults match the full-scale config; a smoke
  # test can shrink these via env vars to run a fast, low-resource iteration).
  "data.rollout_batch_size=${RISE_ROLLOUT_BATCH_SIZE:-256}"
  "worker.actor.global_batch_size=${RISE_GLOBAL_BATCH_SIZE:-64}"
  # Number of rollouts per image for GRPO.
  "worker.rollout.n=${RISE_ROLLOUT_N:-8}"

  "trainer.project_name=${PROJECT_NAME:-RISE}"
  "trainer.max_steps=${train_steps}"
  "trainer.save_freq=${train_steps}"
  "trainer.experiment_name=${experiment_name}"
  "trainer.save_checkpoint_path=${model_save_root}/${experiment_name}"
  # max_steps is the real stop condition; enough epochs guarantee a small
  # (e.g. 1-batch) dataloader still reaches it instead of ending early.
  "trainer.total_epochs=${train_steps}"

  # The questioner is trained on the low GPU IDs while solver reward servers use the
  # remaining GPU IDs on the same node (see GPU split above).
  "trainer.n_gpus_per_node=${train_gpus}"
  trainer.val_before_train=false
  trainer.val_only=false
)

if [ -n "${load_checkpoint_path}" ]; then
  trainer_args+=("trainer.load_checkpoint_path=${load_checkpoint_path}")
fi

echo "Training questioner..."
CUDA_VISIBLE_DEVICES="${train_cuda_devices}" python3 -m verl.trainer.main "${trainer_args[@]}"

# Merge the step the trainer actually saved, which can differ from the
# requested one if training ended early.
latest_step_file="${model_save_root}/${experiment_name}/latest_global_step.txt"
if [ ! -f "${latest_step_file}" ]; then
  echo "ERROR: questioner trainer saved no checkpoint (missing ${latest_step_file})" >&2
  exit 1
fi
saved_step="$(tr -d '[:space:]' < "${latest_step_file}")"
step_dir="${model_save_root}/${experiment_name}/global_step_${saved_step}"
python scripts_Qwen3-VL-8B/model_merger.py --local_dir "${step_dir}/actor"

if [ "${saved_step}" != "${train_steps}" ]; then
  echo "ERROR: questioner trainer stopped at global_step_${saved_step}, expected global_step_${train_steps}" >&2
  exit 1
fi

if [ ! -f "${step_dir}/actor/huggingface/config.json" ]; then
  echo "ERROR: merged questioner checkpoint is incomplete: ${step_dir}/actor/huggingface" >&2
  exit 1
fi

echo "Questioner training finished: ${step_dir}/actor/huggingface"
