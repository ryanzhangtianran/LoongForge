# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Convert a legacy ERSI LeRobot v2.1 dataset (state[159], action[40], full-res videos)
to the agibot_to_lerobot_compose default layout (state[16], action[16], downscaled videos).

Videos are resized by a uniform factor (default 0.5: head 1280x800 -> 640x400, wrists
1280x1056 -> 640x528), keeping the aspect ratio. ``info.json["camera_parameters"]`` intrinsics
(Fx/Fy/Cx/Cy) are rescaled to match; distortion coefficients are resolution-independent.

usage: python tools/embodied/convert_ersi_legacy_schema.py <legacy_dir> <out_dir> [--scale 0.5]
then: python tools/embodied/compute_relative_action_stats.py <out_dir> --dims 14
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_IDX = np.r_[30:44, 0:2]   # arm joint positions + measured grippers
ACTION_IDX = np.r_[16:30, 0:2]  # arm joint targets + gripper labels
NAMES = [f"idx{i}_arm_l_joint{j}" for i, j in zip(range(21, 28), range(1, 8))] + \
        [f"idx{i}_arm_r_joint{j}" for i, j in zip(range(61, 68), range(1, 8))] + \
        ["left_gripper", "right_gripper"]
CAM_MAP = {
    "observation.images.top_head": "observation.images.cam0_head",
    "observation.images.hand_left": "observation.images.cam1_hand_left",
    "observation.images.hand_right": "observation.images.cam2_hand_right",
}
SCALE = 0.5
# camera_parameters key -> legacy video key (for source width)
INTRINSIC_CAM = {
    "intrinsic_head_front_rgb": "observation.images.top_head",
    "intrinsic_hand_left_rgb": "observation.images.hand_left",
    "intrinsic_hand_right_rgb": "observation.images.hand_right",
}


def rescale_intrinsics(camera_parameters: dict, src_sizes: dict, dst_sizes: dict, scale: float):
    """Rescale pinhole intrinsics in place to the resized videos.

    Some ERSI datasets already store head intrinsics at half resolution although the stored
    video is full resolution. The principal point tells which: Cx ~= W/2 at the scale the
    intrinsics were calibrated for. Values already at the target scale are left unchanged.
    ``imageSize`` is set to the output [width, height]. Returns per-camera scaled/kept counts.
    """
    log = {}
    for per_ep in camera_parameters.values():
        for cam, p in per_ep.items():
            if cam not in INTRINSIC_CAM or not isinstance(p, dict):
                continue
            src_w, _ = src_sizes[INTRINSIC_CAM[cam]]
            dst_w, dst_h = dst_sizes[INTRINSIC_CAM[cam]]
            # nearest of {src_w, dst_w} to 2*Cx decides the scale the intrinsics refer to
            at_src = abs(2 * p["Cx"] - src_w) < abs(2 * p["Cx"] - dst_w)
            if not at_src:
                assert abs(2 * p["Cx"] - dst_w) < 0.25 * dst_w, (cam, p["Cx"], src_w)
            else:
                for k in ("Fx", "Fy", "Cx", "Cy"):
                    p[k] = p[k] * scale
            p["imageSize"] = [dst_w, dst_h]
            log.setdefault(cam, {"scaled": 0, "kept": 0})["scaled" if at_src else "kept"] += 1
    return log


def convert_parquet(src: Path, dst: Path):
    t = pq.read_table(src)
    s = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    a = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
    s16, a16 = s[:, STATE_IDX], a[:, ACTION_IDX]
    cols = {}
    for name in t.schema.names:
        if name == "observation.state":
            cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(s16.ravel(), pa.float32()), 16)
        elif name == "action":
            cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(a16.ravel(), pa.float32()), 16)
        else:
            cols[name] = t.column(name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dst)
    return s16, a16


def convert_video(src: Path, dst: Path, size):
    dst.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"scale={w}:{h}:flags=area", "-c:v", "libx265", "-preset", "medium", "-crf", "23",
           "-g", "2", "-pix_fmt", "yuv420p", "-x265-params", "log-level=none", "-threads", "4", str(dst)]
    subprocess.run(cmd, check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
                          "-show_entries", "stream=width,height,nb_read_frames", "-of", "csv=p=0", str(dst)],
                         check=True, capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1]), int(out[2])


def stats_of(x: np.ndarray, quantiles: bool):
    d = {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
         "std": x.std(0).tolist(), "count": [int(len(x))]}
    if quantiles:
        d["q01"] = np.quantile(x, 0.01, axis=0).tolist()
        d["q99"] = np.quantile(x, 0.99, axis=0).tolist()
    return d


def main(legacy: Path, out: Path, scale: float = SCALE, workers: int = 96):
    assert not out.exists(), f"{out} already exists"
    info = json.loads((legacy / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["fps"] == 30
    # features shape is [height, width, channel]
    src_sizes = {k: (info["features"][k]["shape"][1], info["features"][k]["shape"][0]) for k in CAM_MAP}
    dst_sizes = {}
    for k, (w, h) in src_sizes.items():
        dw, dh = w * scale, h * scale
        assert dw == int(dw) and dh == int(dh) and int(dw) % 2 == 0 and int(dh) % 2 == 0, (k, w, h, scale)
        dst_sizes[k] = (int(dw), int(dh))
    episodes = [json.loads(l) for l in (legacy / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    ep_stats = [json.loads(l) for l in (legacy / "meta/episodes_stats.jsonl").read_text().splitlines() if l.strip()]
    assert len(episodes) == len(ep_stats) == info["total_episodes"]

    # 1. parquet + per-episode state/action stats
    all_s, all_a, new_ep_stats = [], [], []
    for ep, st in zip(episodes, ep_stats):
        idx = ep["episode_index"]
        rel = info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
        s16, a16 = convert_parquet(legacy / rel, out / rel)
        assert len(s16) == ep["length"], (idx, len(s16), ep["length"])
        all_s.append(s16); all_a.append(a16)
        new_stats = {}
        for k, v in st["stats"].items():
            if k == "observation.state": new_stats[k] = stats_of(s16, False)
            elif k == "action": new_stats[k] = stats_of(a16, False)
            else: new_stats[CAM_MAP.get(k, k)] = v
        new_ep_stats.append({**st, "stats": new_stats})
    print(f"parquet done: {len(all_s)} episodes, {sum(map(len, all_s))} frames")

    # 2. videos (parallel ffmpeg), verify frame counts
    jobs = []
    for ep in episodes:
        idx = ep["episode_index"]
        for old_key, new_key in CAM_MAP.items():
            fmt = dict(episode_chunk=idx // info["chunks_size"], episode_index=idx)
            src = legacy / info["video_path"].format(video_key=old_key, **fmt)
            dst = out / info["video_path"].format(video_key=new_key, **fmt)
            jobs.append((idx, new_key, src, dst, ep["length"], dst_sizes[old_key]))
    shapes = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (idx, key, _, _, length, size), (w, h, n) in zip(
                jobs, pool.map(lambda j: convert_video(j[2], j[3], j[5]), jobs)):
            assert n == length, f"episode {idx} {key}: {n} frames vs {length} rows"
            assert (w, h) == size, (idx, key, w, h, size)
            shapes.setdefault(key, (h, w, 3))
    print(f"videos done: {len(jobs)} files, shapes {shapes}")

    # 3. meta
    feats = {}
    for k, v in info["features"].items():
        if k in CAM_MAP:
            feats[CAM_MAP[k]] = {**v, "shape": list(shapes[CAM_MAP[k]])}
        elif k in ("observation.state", "action"):
            feats[k] = {"dtype": "float32", "shape": [16], "names": NAMES}
        else:
            feats[k] = v
    info["features"] = feats
    if isinstance(info.get("camera_parameters"), dict):
        log = rescale_intrinsics(info["camera_parameters"], src_sizes, dst_sizes, scale)
        print(f"intrinsics: {log}")
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    with (out / "meta/episodes_stats.jsonl").open("w") as f:
        for st in new_ep_stats:
            f.write(json.dumps(st) + "\n")
    S, A = np.concatenate(all_s), np.concatenate(all_a)
    (out / "meta/stats.json").write_text(json.dumps(
        {"observation.state": stats_of(S, True), "action": stats_of(A, True)}, indent=2))
    for name in ("tasks.jsonl", "episodes.jsonl"):
        shutil.copy(legacy / "meta" / name, out / "meta" / name)
    for name in ("static_trim_summary.json", "validation.json"):  # provenance reports, if present
        if (legacy / name).exists():
            shutil.copy(legacy / name, out / name)
    print("meta done; stats.json q01/q99 over", len(S), "frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("legacy_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--scale", type=float, default=SCALE, help="uniform video resize factor")
    ap.add_argument("--workers", type=int, default=96, help="parallel ffmpeg processes")
    args = ap.parse_args()
    main(args.legacy_dir, args.out_dir, args.scale, args.workers)
