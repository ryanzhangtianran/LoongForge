# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PI05Pytorch.load_pretrained's checkpoint layouts.

The base weights (lerobot pi05_base) have unprefixed keys. A checkpoint LoongForge saved from PI05Policy holds every
tensor twice, under ``model.`` and under the ``pi05.`` alias; it must load as well, so that a new run can start from
an earlier run's checkpoint.
"""

import torch
from safetensors.torch import save_file

from loongforge.embodied.model.pi05.modeling_pi05 import PI05Pytorch


class _TinyPI05(torch.nn.Module):
    """PI05Pytorch's loader on a module small enough to build in a test."""

    load_pretrained = PI05Pytorch.load_pretrained

    def __init__(self):
        super().__init__()
        self.action_in_proj = torch.nn.Linear(3, 2)


def _weights():
    return {"action_in_proj.weight": torch.randn(2, 3), "action_in_proj.bias": torch.randn(2)}


def test_loads_base_layout(tmp_path):
    w = _weights()
    save_file(w, str(tmp_path / "model.safetensors"))
    m = _TinyPI05().load_pretrained(str(tmp_path))
    assert torch.equal(m.action_in_proj.weight, w["action_in_proj.weight"])


def test_loads_own_policy_checkpoint(tmp_path):
    w = _weights()
    own = {**{f"model.{k}": v for k, v in w.items()}, **{f"pi05.{k}": v.clone() for k, v in w.items()}}
    save_file(own, str(tmp_path / "model.safetensors"))
    m = _TinyPI05().load_pretrained(str(tmp_path))
    assert torch.equal(m.action_in_proj.weight, w["action_in_proj.weight"])
    assert torch.equal(m.action_in_proj.bias, w["action_in_proj.bias"])
