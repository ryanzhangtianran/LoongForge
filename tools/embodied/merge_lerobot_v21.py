# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Merge several LeRobot v2.1 datasets (same features) into one multi-task dataset.

Why a physical merge instead of ``MultiLeRobotV2Dataset``: its stats aggregation keeps only
min/max/mean/std, dropping the q01/q99 and ``relative_action`` stats pi05 normalizes with.

What it does:
  * task text: renamed via ``--task-map`` ({dataset_dir_name: {old task: new task}}); a dataset
    listed in the map must cover all its tasks. Global task_index is rebuilt.
  * mixture: per-dataset sampling weight ~ frames ** alpha (alpha=0.5: square-root, 1: natural,
    0: uniform per dataset). Implemented by repeating whole episodes; repeat factors are
    normalized so the smallest is 1, i.e. no data is dropped. Fractional repeats pick a
    seeded random subset of episodes for the extra copy.
  * holdout: ``--holdout-per-dataset K`` keeps K random episodes per source out of training and
    writes them (no repeats) to ``<out>_holdout`` with the *training* stats.json, so eval loss
    is normalized exactly like training.
  * parquet rewritten with new episode_index/index/task_index; videos are relative symlinks
    to the source files (``--copy-videos`` to copy). Per-episode info.json dicts
    (camera_parameters, high_level_instruction, instruction_segments, ...) are re-keyed.
  * stats.json (state/action incl. q01/q99) over the merged training frames, repeats included
    (= the distribution the sampler sees); optional relative_action stats via
    compute_relative_action_stats.py.
  * meta/merge_manifest.json: per output episode -> (source dataset, source episode, copy).

usage:
  python tools/embodied/merge_lerobot_v21.py <src_dir>... --out <out_dir> \
      --task-map tools/embodied/ersi_task_map.json --alpha 0.5 --holdout-per-dataset 3 \
      --relative-action-dims 14
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PER_EPISODE_INFO_KEYS = ("camera_parameters", "high_level_instruction", "instruction_segments",
                         "h5_path", "intervention_info", "key_frame", "take_over", "aid_info")
INDEX_COLS = ("episode_index", "index", "task_index")


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, rows):
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def stats_of(x: np.ndarray, quantiles: bool):
    d = {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
         "std": x.std(0).tolist(), "count": [int(len(x))]}
    if quantiles:
        d["q01"] = np.quantile(x, 0.01, axis=0).tolist()
        d["q99"] = np.quantile(x, 0.99, axis=0).tolist()
    return d


class Source:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.info = json.loads((root / "meta/info.json").read_text())
        assert self.info["codebase_version"] == "v2.1", (root, self.info["codebase_version"])
        self.episodes = read_jsonl(root / "meta/episodes.jsonl")
        self.ep_stats = {e["episode_index"]: e for e in read_jsonl(root / "meta/episodes_stats.jsonl")}
        self.tasks = {t["task_index"]: t["task"] for t in read_jsonl(root / "meta/tasks.jsonl")}
        self.video_keys = [k for k, v in self.info["features"].items() if v.get("dtype") == "video"]
        self.length = {e["episode_index"]: e["length"] for e in self.episodes}

    def data_path(self, ep):
        return self.root / self.info["data_path"].format(
            episode_chunk=ep // self.info["chunks_size"], episode_index=ep)

    def video_path(self, ep, key):
        return self.root / self.info["video_path"].format(
            episode_chunk=ep // self.info["chunks_size"], video_key=key, episode_index=ep)


def check_compatible(sources):
    ref = sources[0]
    def sig(s):
        return {k: (v.get("dtype"), v.get("shape")) for k, v in s.info["features"].items()}
    for s in sources[1:]:
        assert sig(s) == sig(ref), f"features differ: {s.name} vs {ref.name}\n{sig(s)}\n{sig(ref)}"
        assert s.info["fps"] == ref.info["fps"], (s.name, s.info["fps"], ref.info["fps"])


def plan(sources, alpha, holdout, seed):
    """-> (train [(src, ep, copy)], holdout [(src, ep, 0)], report rows)."""
    rng = np.random.default_rng(seed)
    kept, held = {}, {}
    for s in sources:
        eps = [e["episode_index"] for e in s.episodes]
        ho = set(rng.choice(eps, size=holdout, replace=False).tolist()) if holdout else set()
        assert len(ho) < len(eps), (s.name, holdout, len(eps))
        kept[s.name] = [e for e in eps if e not in ho]
        held[s.name] = sorted(ho)
    length = {s.name: s.length for s in sources}
    frames = {s.name: sum(length[s.name][e] for e in kept[s.name]) for s in sources}
    total = sum(frames.values())
    w = {n: f ** alpha for n, f in frames.items()}
    wsum = sum(w.values())
    rep = {n: (w[n] / wsum) * total / frames[n] for n in frames}
    rmin = min(rep.values())
    rep = {n: r / rmin for n, r in rep.items()}

    train, report = [], []
    for s in sources:
        eps, r = kept[s.name], rep[s.name]
        whole, frac = int(math.floor(r)), r - math.floor(r)
        extra = set(rng.choice(eps, size=int(round(frac * len(eps))), replace=False).tolist())
        n_frames = 0
        for e in eps:
            for c in range(whole + (1 if e in extra else 0)):
                train.append((s, e, c))
                n_frames += length[s.name][e]
        report.append({"dataset": s.name, "episodes": len(eps), "holdout": len(held[s.name]),
                       "frames": frames[s.name], "repeat": r, "merged_frames": n_frames})
    merged_total = sum(r["merged_frames"] for r in report)
    for r in report:
        r["share_natural"] = r["frames"] / total
        r["share_target"] = w[r["dataset"]] / wsum
        r["share_merged"] = r["merged_frames"] / merged_total
    ho_list = [(s, e, 0) for s in sources for e in held[s.name]]
    return train, ho_list, report


def build_tasks(sources, task_map):
    """-> (global task list, {(src name, old task_index): new task_index})."""
    texts, remap = [], {}
    for s in sources:
        m = task_map.get(s.name)
        for ti, t in sorted(s.tasks.items()):
            if m is not None:
                assert t in m, f"{s.name}: task {t!r} not in --task-map"
                t = m[t]
            if t not in texts:
                texts.append(t)
            remap[(s.name, ti)] = texts.index(t)
    return texts, remap


def write_dataset(out: Path, entries, sources, task_texts, task_remap, copy_videos, workers):
    ref = sources[0]
    chunks = ref.info["chunks_size"]
    data_tpl, video_tpl = ref.info["data_path"], ref.info["video_path"]
    video_keys = ref.video_keys

    # global frame offsets
    offsets, total = [], 0
    for s, e, _ in entries:
        offsets.append(total)
        total += s.length[e]

    def one(i):
        s, e, _ = entries[i]
        t = pq.read_table(s.data_path(e))
        n = t.num_rows
        old_task = t.column("task_index").to_numpy()
        new_task = np.array([task_remap[(s.name, int(x))] for x in old_task], dtype=np.int64)
        cols = {}
        for name in t.schema.names:
            if name == "episode_index":
                cols[name] = pa.array(np.full(n, i, dtype=np.int64))
            elif name == "index":
                cols[name] = pa.array(np.arange(offsets[i], offsets[i] + n, dtype=np.int64))
            elif name == "task_index":
                cols[name] = pa.array(new_task)
            else:
                cols[name] = t.column(name)
        dst = out / data_tpl.format(episode_chunk=i // chunks, episode_index=i)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), dst)
        for k in video_keys:
            src_v = s.video_path(e, k)
            dst_v = out / video_tpl.format(episode_chunk=i // chunks, video_key=k, episode_index=i)
            dst_v.parent.mkdir(parents=True, exist_ok=True)
            assert src_v.exists(), src_v
            if copy_videos:
                shutil.copy2(src_v, dst_v)
            else:
                os.symlink(os.path.relpath(src_v.resolve(), dst_v.parent.resolve()), dst_v)
        st = t.column("observation.state").to_numpy(zero_copy_only=False)
        ac = t.column("action").to_numpy(zero_copy_only=False)
        # per-episode stats: source stats, with index columns recomputed
        ep_st = json.loads(json.dumps(s.ep_stats[e]["stats"]))
        for name, arr in (("episode_index", np.full(n, i)), ("task_index", new_task),
                          ("index", np.arange(offsets[i], offsets[i] + n))):
            if name in ep_st:
                ep_st[name] = stats_of(arr.astype(np.float64)[:, None], False)
        ep_row = {"episode_index": i,
                  "tasks": sorted({task_texts[x] for x in new_task.tolist()}), "length": n}
        return np.stack(st).astype(np.float32), np.stack(ac).astype(np.float32), ep_row, \
            {"episode_index": i, "stats": ep_st}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, range(len(entries))))

    states = np.concatenate([r[0] for r in results])
    actions = np.concatenate([r[1] for r in results])
    (out / "meta").mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "meta/episodes.jsonl", [r[2] for r in results])
    write_jsonl(out / "meta/episodes_stats.jsonl", [r[3] for r in results])
    write_jsonl(out / "meta/tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(task_texts)])

    info = json.loads(json.dumps(ref.info))
    n_ep = len(entries)
    info.update({
        "total_episodes": n_ep, "total_frames": int(total), "total_tasks": len(task_texts),
        "total_videos": n_ep * len(video_keys), "total_chunks": (n_ep + chunks - 1) // chunks,
        "splits": {"train": f"0:{n_ep}"},
    })
    for key in PER_EPISODE_INFO_KEYS:
        merged = {}
        for i, (s, e, _) in enumerate(entries):
            v = s.info.get(key)
            if isinstance(v, dict) and str(e) in v:
                merged[str(i)] = v[str(e)]
        info[key] = merged
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    manifest = [{"episode_index": i, "source": s.name, "source_episode": e, "copy": c}
                for i, (s, e, c) in enumerate(entries)]
    (out / "meta/merge_manifest.json").write_text(json.dumps(manifest, indent=1))
    return states, actions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", type=Path, nargs="+", help="LeRobot v2.1 dataset dirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task-map", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight ~ frames**alpha (0.5 = sqrt)")
    ap.add_argument("--holdout-per-dataset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relative-action-dims", type=int, default=0,
                    help=">0: also write relative_action stats (compute_relative_action_stats.py)")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--copy-videos", action="store_true", help="copy instead of relative symlinks")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    assert not args.out.exists(), f"{args.out} already exists"
    holdout_out = args.out.with_name(args.out.name + "_holdout")
    if args.holdout_per_dataset:
        assert not holdout_out.exists(), f"{holdout_out} already exists"

    sources = [Source(p) for p in args.sources]
    assert len({s.name for s in sources}) == len(sources), "duplicate source dir names"
    check_compatible(sources)
    task_map = {}
    if args.task_map:
        task_map = {k: v for k, v in json.loads(args.task_map.read_text()).items() if not k.startswith("_")}
    task_texts, task_remap = build_tasks(sources, task_map)

    train, held, report = plan(sources, args.alpha, args.holdout_per_dataset, args.seed)
    print(f"{'dataset':32s} {'eps':>4s} {'ho':>3s} {'frames':>8s} {'repeat':>6s} "
          f"{'natural':>7s} {'target':>7s} {'merged':>7s}")
    for r in report:
        print(f"{r['dataset']:32s} {r['episodes']:4d} {r['holdout']:3d} {r['frames']:8d} {r['repeat']:6.2f} "
              f"{r['share_natural']*100:6.1f}% {r['share_target']*100:6.1f}% {r['share_merged']*100:6.1f}%")
    print(f"tasks: {len(task_texts)}; train episodes {len(train)}, "
          f"frames {sum(r['merged_frames'] for r in report)}; holdout episodes {len(held)}")

    states, actions = write_dataset(args.out, train, sources, task_texts, task_remap,
                                    args.copy_videos, args.workers)
    stats = {"observation.state": stats_of(states, True), "action": stats_of(actions, True)}
    (args.out / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    (args.out / "meta/merge_report.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "report": report}, indent=2))
    print(f"wrote {args.out}")

    if args.relative_action_dims:
        subprocess.run([sys.executable, str(Path(__file__).with_name("compute_relative_action_stats.py")),
                        str(args.out), "--dims", str(args.relative_action_dims),
                        "--horizon", str(args.horizon)], check=True)

    if held:
        write_dataset(holdout_out, held, sources, task_texts, task_remap, args.copy_videos, args.workers)
        # normalize eval exactly like training
        shutil.copy(args.out / "meta/stats.json", holdout_out / "meta/stats.json")
        print(f"wrote {holdout_out} (stats.json copied from training set)")


if __name__ == "__main__":
    main()
