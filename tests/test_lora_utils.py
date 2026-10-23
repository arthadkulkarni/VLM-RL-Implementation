"""LoRA name mapping between peft, HF and vLLM (verl/utils/lora_utils.py).

The tiny-model tests need torch, so run them with the project env loaded:
  source scripts_Qwen3-VL-8B/use_env.sh && python3 -m pytest tests/test_lora_utils.py
"""

import pytest

from verl.utils.lora_utils import (
    hf_base_name,
    peft_adapter_name,
    split_peft_state,
    vllm_base_name,
    vllm_max_lora_rank,
)


def test_names():
    live = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    assert hf_base_name(f"{live}.base_layer.weight") == "model.language_model.layers.0.self_attn.q_proj.weight"
    assert peft_adapter_name(f"{live}.lora_A.default.weight") == f"{live}.lora_A.weight"
    assert vllm_base_name("model.language_model.layers.0.mlp.down_proj.weight") == (
        "model.language_model.layers.0.mlp.down_proj.base_layer.weight"
    )
    # vLLM wraps only the language model's projections.
    for name in ("model.language_model.norm.weight", "lm_head.weight", "model.visual.blocks.0.attn.qkv.weight"):
        assert vllm_base_name(name) == name


def test_max_lora_rank():
    assert [vllm_max_lora_rank(r) for r in (1, 8, 12, 16, 64, 100)] == [1, 8, 16, 16, 64, 128]
    with pytest.raises(ValueError):
        vllm_max_lora_rank(1024)


@pytest.fixture(scope="module")
def tiny_model():
    pytest.importorskip("peft")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, vocab_size=512, rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 3, 3],
                                                      "rope_type": "default"},
        ),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
            deepstack_visual_indexes=[0], num_position_embeddings=64,
        ),
    )
    return Qwen3VLForConditionalGeneration(config)


def lora_config(**overrides):
    from verl.workers.actor.config import LoraConfig

    return LoraConfig(enabled=True, rank=8, alpha=16, **overrides)


def test_apply_lora_trains_only_text_adapters(tiny_model):
    import copy

    from verl.utils.lora_utils import apply_lora

    base_keys = set(tiny_model.state_dict())
    model = apply_lora(copy.deepcopy(tiny_model), lora_config(), materialize=True)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable and all(".lora_" in name for name in trainable)
    assert not any("visual" in name for name in trainable)
    # Every module named in target_modules got an adapter: 7 per layer, 2 layers, A and B.
    assert len(trainable) == 7 * 2 * 2

    base, lora = split_peft_state(model.state_dict())
    # The base weights cover exactly the original model, under vLLM's wrapped names.
    assert {name.replace(".base_layer.", ".") for name in base} == base_keys
    assert sum(".base_layer." in name for name in base) == 7 * 2
    assert len(lora) == len(trainable) and all(".default." not in name for name in lora)


def test_adapter_save_load_roundtrip(tiny_model, tmp_path):
    import copy

    import torch
    from safetensors.torch import load_file

    from verl.utils.lora_utils import apply_lora

    model = apply_lora(copy.deepcopy(tiny_model), lora_config(), materialize=True)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()
    model.save_pretrained(tmp_path, state_dict=model.state_dict())

    _, lora = split_peft_state(model.state_dict())
    saved = load_file(tmp_path / "adapter_model.safetensors")
    # The tensors pushed to vLLM are named exactly as peft saves them.
    assert set(saved) == set(lora)

    reloaded = apply_lora(copy.deepcopy(tiny_model), lora_config(adapter_path=str(tmp_path)), materialize=True)
    for name, tensor in split_peft_state(reloaded.state_dict())[1].items():
        assert torch.equal(tensor, lora[name])
    assert all(p.requires_grad for n, p in reloaded.named_parameters() if ".lora_" in n)

