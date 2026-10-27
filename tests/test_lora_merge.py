"""Adapter merging across rounds (verl/utils/lora_merge.py). Needs torch:
  source scripts_Qwen3-VL-8B/use_env.sh && python3 -m pytest tests/test_lora_merge.py
"""

import json

import pytest


torch = pytest.importorskip("torch")

from verl.utils.lora_merge import merge_adapters, merge_module, round_weights  # noqa: E402


def test_round_weights():
    assert round_weights("none", 3) == [0.0, 0.0, 1.0]
    assert round_weights("linear", 4) == [0.25] * 4
    assert round_weights("ema", 1, 0.9) == [1.0]
    assert round_weights("ema", 3, 0.5) == pytest.approx([0.25, 0.25, 0.5])
    for k in range(1, 6):
        assert sum(round_weights("ema", k, 0.8)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        round_weights("ema", 2, 1.0)


def random_factors(k, rank, out=48, inp=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(rank, inp, generator=g), torch.randn(out, rank, generator=g), 0.5 + j) for j in range(k)]


def delta(factors):
    return sum(c * b @ a for a, b, c in factors)


def test_merge_module_exact_when_rank_fits():
    factors = random_factors(k=2, rank=3)  # total rank 6 <= 8
    a, b = merge_module(factors, rank=8, scale=2.0)
    assert a.shape == (8, 40) and b.shape == (48, 8)
    torch.testing.assert_close(2.0 * b @ a, delta(factors), rtol=1e-4, atol=1e-4)


def test_merge_module_is_best_rank_r_approximation():
    factors = random_factors(k=4, rank=4)  # total rank 16 > 4
    a, b = merge_module(factors, rank=4, scale=1.0)
    u, s, vh = torch.linalg.svd(delta(factors))
    best = u[:, :4] * s[:4] @ vh[:4]
    torch.testing.assert_close(b @ a, best, rtol=1e-3, atol=1e-3)


def write_adapter(path, factors, r, alpha):
    from safetensors.torch import save_file

    path.mkdir()
    tensors = {}
    for i, (a, b, _) in enumerate(factors):
        name = f"base_model.model.layers.{i}.q_proj"
        tensors[f"{name}.lora_A.weight"], tensors[f"{name}.lora_B.weight"] = a.contiguous(), b.contiguous()
    save_file(tensors, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha, "target_modules": ["q_proj"]}))


def test_merge_adapters_weights_delta_w(tmp_path):
    rounds = [random_factors(k=2, rank=4, seed=s) for s in range(3)]  # 2 modules per adapter
    for i, factors in enumerate(rounds):
        write_adapter(tmp_path / f"r{i}", factors, r=4, alpha=8)
    weights = round_weights("ema", 3, 0.5)
    config, merged = merge_adapters([str(tmp_path / f"r{i}") for i in range(3)], weights)
    assert config["r"] == 4
    for m in range(2):
        a = merged[f"base_model.model.layers.{m}.q_proj.lora_A.weight"]
        b = merged[f"base_model.model.layers.{m}.q_proj.lora_B.weight"]
        target = sum(w * 2.0 * rounds[i][m][1] @ rounds[i][m][0] for i, w in enumerate(weights))
        u, s, vh = torch.linalg.svd(target)
        torch.testing.assert_close(2.0 * b @ a, u[:, :4] * s[:4] @ vh[:4], rtol=1e-3, atol=1e-3)
