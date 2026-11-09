# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RelativeActionTransform: a[t+k] - s[t] on the leading dims, grippers untouched, invertible."""

import numpy as np
import torch

from loongforge.embodied.data.datasets.pi05.transforms.pi05_transform import RelativeActionTransform
from tools.embodied.compute_relative_action_stats import relative_chunks


def test_apply_subtracts_state_on_leading_dims_only():
    state = torch.arange(4, dtype=torch.float32)                # [0, 1, 2, 3]
    action = torch.ones(3, 4) * 10                              # chunk of 3 steps
    out = RelativeActionTransform(dims=2).apply({"observation.state": state, "action": action.clone()})
    assert torch.equal(out["action"][:, :2], torch.tensor([[10.0, 9.0]] * 3))
    assert torch.equal(out["action"][:, 2:], action[:, 2:])


def test_unapply_restores_absolute_actions():
    t = RelativeActionTransform(dims=14)
    state = torch.randn(16)
    action = torch.randn(50, 16)
    data = t.apply({"observation.state": state, "action": action.clone()})
    assert torch.allclose(t.unapply(data)["action"], action)


def test_relative_chunks_matches_loader_padding():
    s = np.arange(6, dtype=np.float64).reshape(3, 2)             # 3 frames, 2 dims
    a = s + 100
    x = relative_chunks(s.copy(), a.copy(), dims=1, horizon=2).reshape(3, 2, 2)
    # frame 2, offset 1 is past the end -> padded with the last action
    assert np.array_equal(x[2, 1], [a[2, 0] - s[2, 0], a[2, 1]])
    assert np.array_equal(x[0, 1], [a[1, 0] - s[0, 0], a[1, 1]])
