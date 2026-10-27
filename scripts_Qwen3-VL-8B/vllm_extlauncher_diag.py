"""Faithful standalone repro of the rollout engine construction in
verl/workers/rollout/vllm_rollout_spmd.py: a pre-existing torch.distributed
NCCL group (like the one FSDPWorker.__init__ creates) + vLLM's
external_launcher executor + RISE_FORCE_VIT_SDPA hijack + the exact
LLM(...) kwargs used there. Run with torchrun --nproc_per_node=2 so each
rank matches one Ray WorkerDict process, without any Ray/FSDP/dataset
overhead.
"""

import os
import time

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])


def log(msg):
    print(f"[rank {rank} pid {os.getpid()} {time.strftime('%X')}] {msg}", flush=True)


log(f"=== starting, world_size={world_size} ===")
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
log("=== process group initialized ===")

from verl.utils.vllm_utils import VLLMHijack  # noqa: E402

VLLMHijack.hijack()
log("=== hijack applied ===")

from vllm import LLM, SamplingParams  # noqa: E402

model_path = "../pretrained_LM/Qwen3-VL-8B-Instruct"
enforce_eager = os.environ.get("DIAG_ENFORCE_EAGER", "0") == "1"
tp_size = int(os.environ.get("DIAG_TP_SIZE", "2"))

log(f"=== building LLM(tp={tp_size}, enforce_eager={enforce_eager}, external_launcher) ===")
t0 = time.time()
llm = LLM(
    model=model_path,
    skip_tokenizer_init=False,
    trust_remote_code=False,
    load_format="dummy",
    dtype="bfloat16",
    seed=1,
    max_model_len=8192,
    distributed_executor_backend="external_launcher",
    tensor_parallel_size=tp_size,
    gpu_memory_utilization=0.6,
    max_num_batched_tokens=8192,
    disable_log_stats=False,
    enforce_eager=enforce_eager,
    disable_custom_all_reduce=True,
    disable_mm_preprocessor_cache=True,
    enable_chunked_prefill=False,
    enable_sleep_mode=True,
)
log(f"=== LLM built in {time.time() - t0:.1f}s ===")

if rank == 0:
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0.0))
    log(f"=== generate output: {out[0].outputs[0].text!r} ===")

log("=== diag finished ===")
