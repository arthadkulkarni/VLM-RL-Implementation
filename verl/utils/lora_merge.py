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
Merging LoRA adapters across self-play rounds (scripts_Qwen3-VL-8B/merge_adapters.py).

Adapters are averaged in weight-update space, dW = (alpha / r) * B @ A per
module, and the average is truncated back to rank r with an SVD. A weighted
sum of k rank-r updates has rank <= k*r, so the SVD is exact and cheap: QR of
the stacked factors and an SVD of a (k*r x k*r) core, never of dW itself.

Round weights (newest last) are computed over every round's raw adapter, so
truncation error does not compound from one merge to the next:
  linear  running mean, merged_k = (1/k) * sum_j new_j
  ema     merged_1 = new_1, merged_k = decay * merged_{k-1} + (1 - decay) * new_k
"""

import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
from safetensors.torch import load_file, save_file


MERGE_MODES = ("none", "linear", "ema")


def round_weights(mode: str, num_rounds: int, decay: float = 0.9) -> List[float]:
    """Weight of each round's raw adapter in the merged adapter, oldest first."""
    if num_rounds < 1:
        raise ValueError("need at least one round")
    if mode == "none":
        return [0.0] * (num_rounds - 1) + [1.0]
    if mode == "linear":
        return [1.0 / num_rounds] * num_rounds
    if mode == "ema":
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        weights = [decay ** (num_rounds - 1)]  # new_1 enters with weight 1, then decays
        weights += [(1 - decay) * decay ** (num_rounds - 1 - j) for j in range(1, num_rounds)]
        return weights
    raise ValueError(f"unknown merge mode {mode!r}; expected one of {MERGE_MODES}")


def load_adapter(path: str) -> Tuple[dict, Dict[str, torch.Tensor]]:
    with open(os.path.join(path, "adapter_config.json")) as f:
        config = json.load(f)
    if config.get("use_rslora") or config.get("use_dora"):
        raise ValueError(f"{path}: rsLoRA/DoRA adapters are not supported")
    return config, load_file(os.path.join(path, "adapter_model.safetensors"))


def merge_module(factors: Sequence[Tuple[torch.Tensor, torch.Tensor, float]], rank: int, scale: float):
    """factors: (A_j, B_j, c_j) with dW = sum_j c_j * B_j @ A_j. Returns (A, B)
    of rank `rank` with scale * B @ A the best rank-`rank` approximation of dW."""
    b_cat = torch.cat([c * b.float() for a, b, c in factors], dim=1)  # out x (k r)
    a_cat = torch.cat([a.float() for a, b, c in factors], dim=0)  # (k r) x in
    q_b, r_b = torch.linalg.qr(b_cat)
    q_a, r_a = torch.linalg.qr(a_cat.T)
    u, s, vh = torch.linalg.svd(r_b @ r_a.T)
    keep = min(rank, s.numel())
    root = (s[:keep] / scale).sqrt()
    b = (q_b @ u[:, :keep]) * root
    a = root[:, None] * (vh[:keep] @ q_a.T)
    if keep < rank:  # pad to the configured rank so every round has the same shapes
        b = torch.cat([b, b.new_zeros(b.shape[0], rank - keep)], dim=1)
        a = torch.cat([a, a.new_zeros(rank - keep, a.shape[1])], dim=0)
    return a, b


def merge_adapters(paths: Sequence[str], weights: Sequence[float], device: str = "cpu"):
    """Weighted merge of peft adapters with the same target modules. Returns
    (config, tensors) for the result, at the newest adapter's rank and alpha."""
    paths, weights = zip(*[(path, w) for path, w in zip(paths, weights) if w != 0.0])
    adapters = [load_adapter(path) for path in paths]
    config, newest = adapters[-1]
    rank, scale = config["r"], config["lora_alpha"] / config["r"]
    modules = sorted({name.rsplit(".lora_", 1)[0] for name in newest})
    for path, (_, tensors) in zip(paths, adapters):
        if {name.rsplit(".lora_", 1)[0] for name in tensors} != set(modules):
            raise ValueError(f"{path} adapts different modules than {paths[-1]}")

    merged = {}
    for module in modules:
        factors = []
        for (cfg, tensors), w in zip(adapters, weights):
            a, b = tensors[f"{module}.lora_A.weight"].to(device), tensors[f"{module}.lora_B.weight"].to(device)
            factors.append((a, b, w * cfg["lora_alpha"] / cfg["r"]))
        a, b = merge_module(factors, rank, scale)
        dtype = newest[f"{module}.lora_A.weight"].dtype
        merged[f"{module}.lora_A.weight"] = a.to(dtype).cpu().contiguous()
        merged[f"{module}.lora_B.weight"] = b.to(dtype).cpu().contiguous()
    return config, merged


def save_adapter(path: str, config: dict, tensors: Dict[str, torch.Tensor], info: dict) -> None:
    os.makedirs(path, exist_ok=True)
    save_file(tensors, os.path.join(path, "adapter_model.safetensors"))
    with open(os.path.join(path, "merge_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:  # last: marks the round complete
        json.dump(config, f, indent=2)
