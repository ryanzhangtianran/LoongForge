# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resuming ZeRO + fused AdamW keeps ``step`` on the parameter device."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer

from loongforge.embodied.distributed.checkpoint import _move_fused_step_to_param_device
from loongforge.embodied.optimizer.optimizer import _MultiDtypeZeroOptimizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")


def _build(params):
    zeros = [
        ZeroRedundancyOptimizer([p], optimizer_class=torch.optim.AdamW, lr=1e-3, fused=True)
        for p in params
    ]
    return _MultiDtypeZeroOptimizer(zeros)


def _step(opt, params):
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()


def test_resume_zero_fused_adamw(tmp_path):
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    try:
        params = [
            torch.nn.Parameter(torch.randn(4, device="cuda")),
            torch.nn.Parameter(torch.randn(4, device="cuda", dtype=torch.bfloat16)),
        ]
        opt = _build(params)
        _step(opt, params)
        opt.consolidate_state_dict(to=0)
        saved = opt.state_dict()

        resumed = _build(params)
        resumed.load_state_dict(saved)
        with pytest.raises(RuntimeError, match="state_steps is on cpu"):
            _step(resumed, params)  # the bug this guards against

        resumed = _build(params)
        resumed.load_state_dict(saved)
        _move_fused_step_to_param_device(resumed)
        _step(resumed, params)
        for zero in resumed._optimizers:
            for state in zero.optim.state.values():
                assert state["step"].device.type == "cuda"
                assert state["step"].item() == 2
    finally:
        dist.destroy_process_group()
