# LoRA mode (RISE_LORA=1), sourced by main.sh and the stage scripts.
#
# The questioner and the solver/verifier are two LoRA adapters, A_Q and A_V,
# on one shared frozen base model. main.sh hands each stage the base model
# path plus the adapters to use through the environment:
#   RISE_QUESTIONER_ADAPTER  A_Q to continue training / generate questions with
#   RISE_SOLVER_ADAPTER      A_V to continue training / judge with
# Empty means a fresh adapter (training) or the plain base model (serving).
# Checkpoints are adapter-only: <step>/actor/lora_adapter.
#
# With RISE_ADAPTER_MERGE (main.sh) the adapters that generate questions,
# pseudo-label and judge are each role's merged round adapter instead:
#   RISE_QUESTIONER_SERVE_ADAPTER / RISE_SOLVER_SERVE_ADAPTER
# default to the training adapters above when unset or empty.
#
# RISE_LORA_RANK / RISE_LORA_ALPHA size fresh adapters (default 16 / 32).

lora_enabled() {
  [ "${RISE_LORA:-0}" = "1" ]
}

# The adapter a vLLM consumer should serve (its RISE_LORA_PATH): $1 in LoRA
# mode, nothing otherwise.
lora_serve_path() {
  if lora_enabled; then
    echo "${1:-}"
  fi
}

# The adapters each role is served with: its round adapter, else its training adapter.
questioner_serve_adapter() {
  echo "${RISE_QUESTIONER_SERVE_ADAPTER:-${RISE_QUESTIONER_ADAPTER:-}}"
}
solver_serve_adapter() {
  echo "${RISE_SOLVER_SERVE_ADAPTER:-${RISE_SOLVER_ADAPTER:-}}"
}

# Trainer overrides that train a LoRA adapter, continuing from adapter $1 if given.
lora_trainer_args() {
  local adapter="${1:-}"
  echo "worker.actor.model.lora.enabled=true"
  echo "worker.actor.model.lora.rank=${RISE_LORA_RANK:-16}"
  echo "worker.actor.model.lora.alpha=${RISE_LORA_ALPHA:-32}"
  if [ -n "${adapter}" ]; then
    echo "worker.actor.model.lora.adapter_path=${adapter}"
  fi
}

# Finish a trained stage: check the trainer saved step $3 under
# $2/latest_global_step.txt, merge it into HF format (full fine-tuning) or
# check its adapter (LoRA), and print the result's path. $1 names the role.
finish_stage_checkpoint() {
  local role="$1" exp_root="$2" expected_step="$3"
  local latest_step_file="${exp_root}/latest_global_step.txt"
  if [ ! -f "${latest_step_file}" ]; then
    echo "ERROR: ${role} trainer saved no checkpoint (missing ${latest_step_file})" >&2
    return 1
  fi
  # The step the trainer actually saved can differ from the requested one if
  # training ended early.
  local saved_step step_dir
  saved_step="$(tr -d '[:space:]' < "${latest_step_file}")"
  step_dir="${exp_root}/global_step_${saved_step}"
  if ! lora_enabled; then
    python scripts_Qwen3-VL-8B/model_merger.py --local_dir "${step_dir}/actor" >&2
  fi

  if [ "${saved_step}" != "${expected_step}" ]; then
    echo "ERROR: ${role} trainer stopped at global_step_${saved_step}, expected global_step_${expected_step}" >&2
    return 1
  fi

  local artifact="${step_dir}/actor/huggingface" marker="config.json"
  if lora_enabled; then
    artifact="${step_dir}/actor/lora_adapter" marker="adapter_config.json"
  fi
  if [ ! -f "${artifact}/${marker}" ]; then
    echo "ERROR: ${role} checkpoint is incomplete: ${artifact}" >&2
    return 1
  fi
  echo "${artifact}"
}
