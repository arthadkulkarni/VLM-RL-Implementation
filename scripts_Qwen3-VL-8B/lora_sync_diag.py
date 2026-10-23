"""Does the LoRA weight sync give vLLM the actor's adapter? One GPU, the real
pieces: the 8B base wrapped with a LoRA adapter under FSDP, vLLMRollout with
LoRA slots, and FSDPVLLMShardingManager doing the sync.

For each of two syncs (the second after changing the adapter weights, as a
training step would), greedy-generate with the synced adapter and check vLLM's
per-token logprobs against the FSDP model's own for the same tokens. Also
checks that the adapter changes vLLM's output at all (vs serving no adapter),
and that the file-based path the other vLLM consumers use (lora_utils.vllm_adapter
on a saved adapter) matches the tensor-based sync.

  python3 -m torch.distributed.run --standalone --nproc_per_node 1 scripts_Qwen3-VL-8B/lora_sync_diag.py --model <base>
"""

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers import AutoModelForImageTextToText, AutoTokenizer
from vllm import SamplingParams

from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.utils.lora_utils import apply_lora, vllm_adapter
from verl.workers.actor.config import LoraConfig
from verl.workers.rollout.config import RolloutConfig
from verl.workers.rollout.vllm_rollout_spmd import vLLMRollout
from verl.workers.sharding_manager import FSDPVLLMShardingManager


PROMPTS = ["Describe what happens when you put a kettle on the stove.", "List three uses of a mug."]


def actor_logprobs(model, prompt_ids, response_ids):
    """The FSDP actor's logprob of each response token (teacher forcing)."""
    ids = torch.tensor([prompt_ids + response_ids], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt_ids) - 1:-1].float()
    return torch.log_softmax(logits, -1).gather(-1, torch.tensor(response_ids, device="cuda")[:, None])[:, 0]


def vllm_logprobs(engine, prompt_ids, lora_request, n_tokens):
    params = SamplingParams(temperature=0.0, max_tokens=n_tokens, logprobs=0, detokenize=False)
    outputs = engine.generate([{"prompt_token_ids": p} for p in prompt_ids], params, lora_request=lora_request,
                              use_tqdm=False)
    results = []
    for out in outputs:
        tokens = list(out.outputs[0].token_ids)
        results.append((tokens, torch.tensor([lp[t].logprob for lp, t in zip(out.outputs[0].logprobs, tokens)])))
    return results


def compare(tag, model, engine, prompt_ids, lora_request, n_tokens):
    """Max |logp_vllm - logp_actor| over the greedy tokens of each prompt."""
    worst = 0.0
    results = vllm_logprobs(engine, prompt_ids, lora_request, n_tokens)
    for p_ids, (tokens, v_lp) in zip(prompt_ids, results):
        a_lp = actor_logprobs(model, p_ids, tokens).cpu()
        worst = max(worst, (v_lp - a_lp).abs().max().item())
    print(f"[diag] {tag}: max |logp_vllm - logp_actor| = {worst:.4f} over {len(prompt_ids)}x{n_tokens} greedy tokens",
          flush=True)
    return worst, [tokens for tokens, _ in results]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--b_std", type=float, default=0.02, help="Std of the random lora_B, so the adapter is not a no-op.")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--tol", type=float, default=0.25, help="Allowed max logprob gap (bf16 kernels differ).")
    args = p.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=os.getenv("RISE_ATTN_IMPLEMENTATION", "sdpa")
    )
    model = apply_lora(model, LoraConfig(enabled=True, rank=args.rank, alpha=2 * args.rank), materialize=True)
    model = model.to(torch.bfloat16)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=args.b_std)
    fsdp = FSDP(model, auto_wrap_policy=get_fsdp_wrap_policy(model), use_orig_params=True,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16), device_id=torch.cuda.current_device())
    fsdp.eval()

    config = RolloutConfig(tensor_parallel_size=1, gpu_memory_utilization=0.5, max_num_batched_tokens=4096,
                           enforce_eager=True)
    config.prompt_length, config.response_length = 512, args.tokens
    rollout = vLLMRollout(args.model, config, tokenizer, lora_rank=args.rank)
    manager = FSDPVLLMShardingManager(fsdp, rollout.inference_engine,
                                      init_device_mesh("cuda", (1, 1), mesh_dim_names=("dp", "tp")))
    prompt_ids = [tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
                  for text in PROMPTS]

    failures = []
    with manager:
        gap, adapter_tokens = compare("sync 1, adapter", fsdp, rollout.inference_engine, prompt_ids,
                                      manager.lora_request, args.tokens)
        _, base_tokens = compare("sync 1, no adapter (vs adapter actor)", fsdp, rollout.inference_engine,
                                 prompt_ids, None, args.tokens)
        failures += [] if gap < args.tol else ["sync 1 logprobs disagree"]
        failures += [] if adapter_tokens != base_tokens else ["adapter does not change vLLM's output"]

    # A training step changes the adapter; the next sync must carry it.
    with torch.no_grad(), FSDP.summon_full_params(fsdp, writeback=True):
        for name, param in fsdp.named_parameters():
            if "lora_B" in name:
                param.mul_(-1.0)
    with manager:
        gap, flipped_tokens = compare("sync 2 (lora_B negated), adapter", fsdp, rollout.inference_engine,
                                      prompt_ids, manager.lora_request, args.tokens)
        failures += [] if gap < args.tol else ["sync 2 logprobs disagree"]
        failures += [] if flipped_tokens != adapter_tokens else ["sync 2 output unchanged"]

        # The consumers' path: the same adapter saved to disk, served by file.
        with tempfile.TemporaryDirectory() as tmp:
            with FSDP.state_dict_type(fsdp, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
                fsdp.module.save_pretrained(tmp, state_dict=fsdp.state_dict())
            _, file_request = vllm_adapter(tmp)
            file_request.lora_int_id = 999
            _, file_tokens = compare("adapter loaded from file", fsdp, rollout.inference_engine, prompt_ids,
                                     file_request, args.tokens)
            failures += [] if file_tokens == flipped_tokens else ["file-loaded adapter differs from synced one"]

    print("[diag] RESULT:", "PASS" if not failures else "FAIL " + "; ".join(failures), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
