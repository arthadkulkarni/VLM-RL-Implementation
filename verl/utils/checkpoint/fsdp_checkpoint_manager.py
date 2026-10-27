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

import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from ..lora_utils import ADAPTER_DIR, is_peft_model
from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge

    For a LoRA (peft) model only the trainable adapter is saved: the shards and
    optimizer state cover adapter parameters only, and rank 0 also writes the
    full adapter in peft's format to lora_adapter/ (merge it into the base with
    scripts_Qwen3-VL-8B/merge_lora.py).
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)
        self.is_lora = is_peft_model(model._fsdp_wrapped_module)

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        # every rank download its own checkpoint
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading model from {os.path.abspath(model_path)}.")
        print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
        model_state_dict = torch.load(model_path, weights_only=False)
        optim_state_dict = torch.load(optim_path, weights_only=False)
        extra_state_dict = torch.load(extra_path, weights_only=False)

        state_dict_options = StateDictOptions(cpu_offload=True, ignore_frozen_params=self.is_lora, strict=not self.is_lora)
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
            options=state_dict_options,
        )
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

        # recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

    def save_checkpoint(self, path: str):
        path = self.local_mkdir(path)
        dist.barrier()

        # Only save model shard; optimizer/extra_state not needed (model_merger.py only reads model files)
        ckpt_cpu_offload_env = os.getenv("VERL_CKPT_CPU_OFFLOAD", "1").strip().lower()
        ckpt_cpu_offload = ckpt_cpu_offload_env not in {"0", "false", "no", "off"}
        if self.rank == 0:
            print(
                f"[rank-{self.rank}]: VERL_CKPT_CPU_OFFLOAD={ckpt_cpu_offload_env}, "
                f"save cpu_offload={ckpt_cpu_offload}."
            )
        state_dict_options = StateDictOptions(cpu_offload=ckpt_cpu_offload, ignore_frozen_params=self.is_lora)
        model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer, options=state_dict_options)
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_state_dict = {
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "rng": self.get_rng_state(),
        }

        print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
        print(f"[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
        torch.save(model_state_dict, model_path)
        torch.save(optim_state_dict, optim_path)
        torch.save(extra_state_dict, extra_path)
        del model_state_dict  # immediately free memory
        del optim_state_dict

        # wait for everyone to dump to local
        dist.barrier()

        module = self.model._fsdp_wrapped_module
        if self.is_lora:  # collective: every rank joins the gather, rank 0 writes
            # Not full_state_dict=True: with ignore_frozen_params it raises a
            # KeyError on ranks other than 0 (torch 2.8). Gather the sharded
            # adapter tensors instead; they are small.
            adapter_state = {
                name: (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).cpu()
                for name, tensor in get_model_state_dict(
                    self.model, options=StateDictOptions(ignore_frozen_params=True)
                ).items()
            }
            if self.rank == 0:
                module.save_pretrained(os.path.join(path, ADAPTER_DIR), state_dict=adapter_state)
                print(f"[rank-{self.rank}]: Saved LoRA adapter to {os.path.abspath(os.path.join(path, ADAPTER_DIR))}.")
            del adapter_state
            module = module.get_base_model()

        if self.rank == 0:
            hf_path = os.path.join(path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)
            assert isinstance(module, PreTrainedModel)
            module.config.save_pretrained(hf_path)
            module.generation_config.save_pretrained(hf_path)
            self.processing_class.save_pretrained(hf_path)

        dist.barrier()

