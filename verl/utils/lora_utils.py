# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LoRA helpers shared by the FSDP actor, the vLLM weight sync, checkpointing and
the vLLM consumers outside the trainer (question generation, evaluation, the
reward servers). Adapters use peft's default adapter name and its on-disk
format, so one adapter directory works for training, vLLM and merging.
"""

import json
import os
from typing import Dict, Optional, Tuple


ADAPTER_DIR = "lora_adapter"  # under a checkpoint's actor/ directory
# Language-model linears that vLLM wraps when LoRA is enabled; their base
# weights live under `<module>.base_layer.*` in the wrapped vLLM model.
VLLM_LORA_WRAPPED = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# vLLM's accepted max_lora_rank values (vllm.config.LoRAConfig).
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def is_peft_model(model) -> bool:
    return hasattr(model, "peft_config")


def apply_lora(model, lora_config, materialize: bool):
    """Wrap `model` with a LoRA adapter and freeze everything else.

    A fresh adapter follows `lora_config`; with `lora_config.adapter_path` the
    adapter's own config is used, and its weights are loaded only when
    `materialize` (the rank that holds real weights under rank0 init; the
    others build the same structure on the meta device and receive weights
    from FSDP's sync_module_states).
    """
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model

    if lora_config.adapter_path:
        if materialize:
            return PeftModel.from_pretrained(
                model, lora_config.adapter_path, is_trainable=True, autocast_adapter_dtype=False
            )
        peft_config = PeftConfig.from_pretrained(lora_config.adapter_path)
        peft_config.inference_mode = False
    else:
        peft_config = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=list(lora_config.target_modules),
            bias="none",
        )
    # Adapter weights in the base dtype: FSDP needs one dtype per flat parameter.
    return get_peft_model(model, peft_config, autocast_adapter_dtype=False)


def lora_b_norm(fsdp_module) -> float:
    """L2 norm of all lora_B weights, read from the live (sharded) parameters
    and summed over ranks; 0 until the adapter has been trained. Collective."""
    import torch
    import torch.distributed as dist

    local = sum(
        param.detach().float().pow(2).sum() for name, param in fsdp_module.named_parameters() if "lora_B" in name
    )
    total = torch.as_tensor(local, dtype=torch.float32, device="cuda")
    if dist.is_initialized():
        dist.all_reduce(total)
    return total.sqrt().item()


def peft_config_dict(model) -> dict:
    """The active adapter's config as plain JSON types (vLLM's PEFTHelper.from_dict)."""
    config = model.peft_config["default"].to_dict()
    return {key: sorted(value) if isinstance(value, set) else value for key, value in config.items()}


def is_lora_key(name: str) -> bool:
    return ".lora_" in name


def peft_adapter_name(name: str) -> str:
    """Live PeftModel parameter name -> peft's saved adapter name."""
    return name.replace(".default.", ".")


def hf_base_name(name: str) -> str:
    """Live PeftModel base parameter name -> the original HF name."""
    return name.removeprefix("base_model.model.").replace(".base_layer.", ".")


def vllm_base_name(hf_name: str) -> str:
    """HF name -> the name vLLM's load_weights expects once LoRA wraps the layer."""
    for module in VLLM_LORA_WRAPPED:
        for suffix in ("weight", "bias"):
            if hf_name.endswith(f".{module}.{suffix}") and not hf_name.startswith(("visual.", "model.visual.")):
                return hf_name[: -len(suffix)] + f"base_layer.{suffix}"
    return hf_name


def split_peft_state(state: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    """A PeftModel state dict -> (base weights named for a LoRA-enabled vLLM
    model, adapter tensors named as in a saved peft adapter)."""
    base, lora = {}, {}
    for name, tensor in state.items():
        if is_lora_key(name):
            lora[peft_adapter_name(name)] = tensor
        else:
            base[vllm_base_name(hf_base_name(name))] = tensor
    return base, lora


def vllm_max_lora_rank(rank: int) -> int:
    for allowed in VLLM_LORA_RANKS:
        if allowed >= rank:
            return allowed
    raise ValueError(f"LoRA rank {rank} exceeds vLLM's maximum {VLLM_LORA_RANKS[-1]}")


def vllm_lora_engine_kwargs(rank: int) -> dict:
    return {"enable_lora": True, "max_lora_rank": vllm_max_lora_rank(rank), "max_loras": 1}


def adapter_rank(adapter_path: str) -> int:
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        return int(json.load(f)["r"])


def adapter_from_env(var: str = "RISE_LORA_PATH") -> Optional[str]:
    """The adapter a vLLM consumer script should serve on top of its base
    model, from the environment (set per call site by the stage scripts); None
    when unset or empty, i.e. the plain base model."""
    path = os.getenv(var, "").strip()
    return path or None


def vllm_adapter(adapter_path: Optional[str]):
    """(extra vllm.LLM kwargs, lora_request for generate) for serving
    `adapter_path` on the base model, or ({}, None) without an adapter."""
    if not adapter_path:
        return {}, None
    from vllm.lora.request import LoRARequest

    request = LoRARequest(lora_name=os.path.basename(os.path.normpath(adapter_path)), lora_int_id=1,
                          lora_path=os.path.abspath(adapter_path))
    return vllm_lora_engine_kwargs(adapter_rank(adapter_path)), request
