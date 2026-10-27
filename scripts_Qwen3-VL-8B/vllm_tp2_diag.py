"""Standalone diagnostic: does vLLM TP=2 engine init hang on this node,
independent of the RISE/Ray/FSDP pipeline?

Isolates the exact point where smoke test job 3180735 stalled (after actor
FSDP init, during the rollout engine's process-group + CUDA-graph-capture
startup) without needing the ~1.5h of dataset filtering / reward-server
startup the full pipeline goes through first.
"""

import os
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

print(f"=== starting diag at {time.strftime('%X')} ===", flush=True)

from vllm import LLM, SamplingParams  # noqa: E402

model_path = "../pretrained_LM/Qwen3-VL-8B-Instruct"
attn_impl = os.environ.get("RISE_ATTN_IMPLEMENTATION", "sdpa")
enforce_eager = os.environ.get("DIAG_ENFORCE_EAGER", "1") == "1"

print(f"=== building LLM(tp=2, enforce_eager={enforce_eager}) at {time.strftime('%X')} ===", flush=True)
t0 = time.time()
llm = LLM(
    model=model_path,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.6,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    enforce_eager=enforce_eager,
    trust_remote_code=False,
    dtype="bfloat16",
)
print(f"=== LLM built in {time.time() - t0:.1f}s at {time.strftime('%X')} ===", flush=True)

out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0.0))
print("=== generate output:", out[0].outputs[0].text, flush=True)
print(f"=== diag finished at {time.strftime('%X')} ===", flush=True)
