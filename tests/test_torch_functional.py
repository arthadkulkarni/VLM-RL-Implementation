"""verl/utils/torch_functional.py."""

import pytest


def test_constant_schedule_without_warmup_starts_at_full_lr():
    torch = pytest.importorskip("torch")
    from verl.utils.torch_functional import get_constant_schedule_with_warmup

    def lrs(warmup):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=warmup)
        out = []
        for _ in range(4):
            out.append(opt.param_groups[0]["lr"])  # the LR the next optimizer.step() uses
            sched.step()
        return out

    assert lrs(0) == [1.0, 1.0, 1.0, 1.0]
    assert lrs(2) == [0.0, 0.5, 1.0, 1.0]
