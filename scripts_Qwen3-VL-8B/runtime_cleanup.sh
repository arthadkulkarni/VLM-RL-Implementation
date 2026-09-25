#!/bin/bash

# Kill the reward vLLM servers by process group. start.sh launches each one
# under setsid and records its pgid; the group also holds vLLM's EngineCore
# children, which `pkill -f start_vllm_server.py` misses. In job 3207357 those
# survived into question generation and held ~80GB on GPUs 2-3.
stop_vllm_server_groups() {
  local pgid_file="${VLLM_SERVER_PGID_FILE:-/tmp/${USER}/visplay_vllm_server_pgids_default}"
  local pgids=()
  if [ -f "${pgid_file}" ]; then
    mapfile -t pgids < <(grep -E '^[0-9]+$' "${pgid_file}" || true)
  fi

  local pgid
  for pgid in "${pgids[@]}"; do
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  done
  pkill -f "vllm_service_init/start_vllm_server.py" 2>/dev/null || true

  local deadline=$((SECONDS + ${RISE_CLEANUP_TIMEOUT_SEC:-60}))
  local alive
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    alive=0
    for pgid in "${pgids[@]}"; do
      if pgrep -g "${pgid}" >/dev/null 2>&1; then
        alive=1
      fi
    done
    [ "${alive}" = "0" ] && break
    sleep 2
  done
  for pgid in "${pgids[@]}"; do
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  done

  if [ -f "${pgid_file}" ]; then
    : > "${pgid_file}"
  fi
}

# Block until no process holds GPU memory, so the next stage's vLLM sees the
# full device. Warns and continues after RISE_GPU_FREE_TIMEOUT_SEC.
wait_for_gpus_free() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local deadline=$((SECONDS + ${RISE_GPU_FREE_TIMEOUT_SEC:-120}))
  local busy
  while true; do
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cE '[0-9]' || true)"
    [ "${busy}" = "0" ] && return 0
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      echo "WARNING: ${busy} process(es) still hold GPU memory after cleanup:" >&2
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
      return 0
    fi
    sleep 3
  done
}

runtime_cleanup_all() {
  echo "Stopping Ray and vLLM processes..."

  command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true

  stop_vllm_server_groups
  wait_for_gpus_free
}
