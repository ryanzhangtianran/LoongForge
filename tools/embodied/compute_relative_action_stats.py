# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Write ``relative_action`` normalization stats into a LeRobot v2.x dataset's meta/stats.json.

For every frame t and chunk offset k < horizon, the sample is
``a[min(t+k, T-1)]`` with its first ``dims`` entries replaced by ``a[...] - s[t]``,
exactly what ``RelativeActionTransform`` sees on top of the v2 loader's last-action
padding. Stats are min/max/mean/std/q01/q99 over all such samples (openpi convention).

usage: python tools/embodied/compute_relative_action_stats.py <dataset_dir> --dims 14 [--horizon 50]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def relative_chunks(state: np.ndarray, action: np.ndarray, dims: int, horizon: int) -> np.ndarray:
    """(T, D) absolute state/action of one episode -> (T * horizon, D) relative samples."""
    T = len(action)
    idx = np.minimum(np.arange(T)[:, None] + np.arange(horizon)[None, :], T - 1)  # (T, H)
    chunks = action[idx]  # (T, H, D), padded with the last action like the loader
    chunks[:, :, :dims] -= state[:, None, :dims]
    return chunks.reshape(-1, action.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--dims", type=int, required=True, help="leading action dims made relative")
    ap.add_argument("--horizon", type=int, default=50)
    args = ap.parse_args()

    stats_path = args.dataset_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    samples = []
    for f in sorted((args.dataset_dir / "data").rglob("episode_*.parquet")):
        t = pq.read_table(f, columns=["observation.state", "action"])
        s = np.stack(t.column("observation.state").to_pylist()).astype(np.float64)
        a = np.stack(t.column("action").to_pylist()).astype(np.float64)
        samples.append(relative_chunks(s, a, args.dims, args.horizon))
    x = np.concatenate(samples)
    assert 0 < args.dims <= x.shape[-1], (args.dims, x.shape)

    stats["relative_action"] = {
        "min": x.min(0).tolist(), "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist(),
        "count": [int(len(x))],
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote relative_action stats over {len(x)} chunk samples to {stats_path}")
    print("q01", np.round(stats["relative_action"]["q01"], 4))
    print("q99", np.round(stats["relative_action"]["q99"], 4))


if __name__ == "__main__":
    main()
