#!/bin/bash
set -euo pipefail

model_path=$1
run_id=$2
pgid_file="${VLLM_SERVER_PGID_FILE:-/tmp/${USER}/visplay_vllm_server_pgids_default}"
export VLLM_DISABLE_COMPILE_CACHE=1
vllm_server_max_model_len="${VLLM_SERVER_MAX_MODEL_LEN:-12288}"
vllm_server_gpu_mem_util="${VLLM_SERVER_GPU_MEM_UTIL:-0.8}"
vllm_server_log_dir="${VLLM_SERVER_LOG_DIR:-${STORAGE_PATH:-../storage_vllm}/temp_results}"
health_timeout="${VLLM_SERVER_HEALTH_TIMEOUT_SEC:-1800}"
health_poll_interval="${VLLM_SERVER_HEALTH_POLL_INTERVAL_SEC:-5}"

mkdir -p "$vllm_server_log_dir"
mkdir -p "$(dirname "${pgid_file}")"
: > "${pgid_file}"

start_one_server() {
  local cuda_id="$1"
  local port="$2"
  local log_file="${vllm_server_log_dir}/vllm_server_${port}.log"
  local pgid=""
  echo "[vllm-start] launching port=${port} cuda=${cuda_id} log=${log_file}"
  setsid env CUDA_VISIBLE_DEVICES="$cuda_id" python vllm_service_init/start_vllm_server.py \
    --port "$port" \
    --model_path "$model_path" \
    --max_model_len "$vllm_server_max_model_len" \
    --gpu_mem_util "$vllm_server_gpu_mem_util" \
    >"$log_file" 2>&1 &
  pgid=$!
  printf "%s\n" "${pgid}" >> "${pgid_file}"
  echo "[vllm-start] run_id=${run_id} port=${port} recorded pgid=${pgid}"
}

wait_for_server() {
  local port="$1"
  local deadline=$(( $(date +%s) + health_timeout ))
  while true; do
    if python - "$port" <<'PY'
import json
import sys
from urllib.request import urlopen

port = sys.argv[1]
try:
    with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") == "ok":
        raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
    then
      echo "[vllm-start] port=${port} passed /healthz"
      return 0
    fi

    if [ "$(date +%s)" -ge "${deadline}" ]; then
      echo "[vllm-start] ERROR: port=${port} did not become healthy within ${health_timeout}s" >&2
      return 1
    fi
    sleep "${health_poll_interval}"
  done
}

# GPU offset where reward servers start (training uses GPUs 0..offset-1) and how many
# reward servers to launch (one per remaining GPU, ports 6000..6000+n-1). Must match
# RISE_REWARD_SERVERS read by train_examples/reward_function/cot_val.py.
gpu_offset="${RISE_REWARD_GPU_OFFSET:-4}"
num_servers="${RISE_REWARD_SERVERS:-4}"

ports=()
for ((i = 0; i < num_servers; i++)); do
  cuda_id=$((gpu_offset + i))
  port=$((6000 + i))
  ports+=("$port")
  start_one_server "$cuda_id" "$port"
done

for port in "${ports[@]}"; do
  wait_for_server "$port"
done

echo "[vllm-start] all judge servers are healthy"
