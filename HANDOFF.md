# HANDOFF — pi05 on AgiBot G2 (ERSI) with LoongForge

更新：2026-11-09。记录 H800 开发机上 pi05 微调的现状、环境要求和待办。

本文件用两个变量代替具体机器路径，自己按本机取值替换（不要把真实路径提交回仓库）：

| 变量 | 含义 |
|---|---|
| `$LF` | LoongForge 仓库目录 |
| `$DATA_ROOT` | 数据、模型、venv、checkpoints 所在的盘;`/root/datasets -> $DATA_ROOT/datasets` |

## 1. 环境

- 硬件：2x H800 80GB，驱动 550.90.07（CUDA 12.4 驱动 API），无 RT core，不做渲染评测。
  - 第 4/6 节的性能数字都是单卡测的；多卡请显式 `GPUS_PER_NODE`。
- Python 环境：`$DATA_ROOT/envs/loongforge`（Python 3.12.16，torch 2.11.0+cu128，numpy 1.26.4，
  pandas 2.2.3，lerobot 0.5.0，transformers 5.3.0，flash_attn 2.8.3.post1，torchao 0.16.0，tensorboard，pytest）。
  - numpy 钉 1.26.4 与 `docker/Dockerfile` 一致；pandas 必须 <3。
  - torch 2.11 自带 cuDNN 9.19 在 550 驱动上懒加载初始化失败，训练必须 `CUDA_MODULE_LOADING=EAGER`。
- CUDA toolkit：`/usr/local/cuda -> /usr/local/cuda-12.4`，且 `nvcc` 不在 PATH 里（需要编译扩展时先
  `export PATH=/usr/local/cuda/bin:$PATH` 并确认 12.4 下的 toolkit 完整）。
- 网络：huggingface.co 不通，hf-mirror.com / modelscope.cn / pypi / github 可用。
- 存储分布：代码仓库（`$LF`）和数据盘（`$DATA_ROOT`）在**两个不同的网络盘**上，别混；
  `$DATA_ROOT` 下放数据、模型、venv、checkpoints、wandb，且 `/root/datasets -> $DATA_ROOT/datasets`。
- uv 操作一律显式 `--python $DATA_ROOT/envs/loongforge/bin/python`。

### 1.1 重建 venv

`docker/setup_venv_cci.sh`：从零重建 venv 的完整脚本（apt 依赖、torch cu128、LoongForge[gpu]、dmuon、
lerobot 0.5.0、flash-attn 2.8.3 sm_90、numpy/pandas/pyarrow/tensorboard 钉版、pytest），末尾自带导入校验。
两个路径都可用环境变量覆盖：`LF`（仓库，默认 `/workspace/LoongForge`）、
`VENV`（目标 venv，默认 `/workspace/venv`）。

脚本自己 `apt-get install cuda-toolkit-12-8` 并把 `CUDA_HOME` 指向 12.8（torch 是 cu128 轮子，
flash-attn 也对着 12.8 编），与本机现存的 12.4 toolkit 无关，不要为了"对齐 12.4"去改它。

### 1.2 凭证

- wandb：凭证在 `~/.netrc`，可用（`--wandb-mode online --wandb-project <name>`）。
- HF token 已随 `~/.cache/huggingface` 一起被删，需要时重新 `hf auth login`。tokenizer 已本地化，日常训练不需要 HF。

## 2. 资产路径

| 资产 | 路径 |
|---|---|
| 训练数据（新 schema，16/16 维，3 相机，stats 含 relative_action） | `$DATA_ROOT/datasets/ersi/02_pick_biggest_block` |
| 训练数据（legacy 159/40 维，带 videos） | `$DATA_ROOT/datasets/pulled/pick_biggest_block` |
| 训练数据（legacy 159/40 维，无 videos / 无 stats.json） | `$DATA_ROOT/datasets/ersi_raw/02_pick_biggest_block` |
| 其它 ERSI 数据集（本文件未覆盖，用前先核对 schema） | `$DATA_ROOT/datasets/ersi_mixed`、`ersi_mixed_eval` |
| pi05 base 权重（lerobot safetensors） | `$DATA_ROOT/models/models--lerobot--pi05_base` |
| PaliGemma tokenizer | `LoongForge/models/paligemma-3b-pt-224`（来自 ModelScope `AI-ModelScope/paligemma-3b-pt-224`，未进 git） |
| legacy -> 新 schema 转换脚本 | `tools/embodied/convert_ersi_legacy_schema.py`（切 16 维、相机改名、视频缩到 256p hevc、重算 stats.json；布局同 `geniesim_benchmark.dataset.convert.agibot_to_lerobot_compose` 默认值） |

openpi 源码、venv、缓存、所有 openpi checkpoint 已全部删除。

## 3. 数据 schema（LeRobot v2.1）

- `observation.state` [16]：14 臂关节（legacy state[30:44]）+ 2 夹爪测量值（legacy [0:2]）。
- `action` [16]：14 臂关节绝对目标（legacy action[16:30]）+ 2 夹爪标签（legacy [0:2]），绝对值，差分只在训练时算。
- 相机 key 字母序即 base 在前：`cam0_head`(256x410)、`cam1_hand_left`(256x310)、`cam2_hand_right`(256x310)，hevc，30 fps。
- `meta/stats.json`：state/action 的全量 min/max/mean/std/q01/q99，以及 `relative_action`（差分动作统计，由 `tools/embodied/compute_relative_action_stats.py --dims 14 --horizon 50` 生成）。
  - v2.1 加载器没有 stats.json 时会把 q01/q99 别名成 min/max，所以这个文件必须有。

## 4. 训练

```bash
cd $LF
CUDA_MODULE_LOADING=EAGER LOONGFORGE_PATH=$PWD GPUS_PER_NODE=1 \
PER_DEVICE_BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 NUM_WORKERS=8 \
TOKENIZER_PATH=$PWD/models/paligemma-3b-pt-224 \
CHECKPOINT_PATH=$DATA_ROOT/models/models--lerobot--pi05_base \
DATA_PATH=$DATA_ROOT/datasets/ersi/02_pick_biggest_block \
OUTPUT_DIR=<out> TRAIN_ITERS=<n> SAVE_INTERVAL=<k> \
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
  --lerobotdataset-version v2.1 \
  --fp8 --fp8-backend torchao --optimizer TorchFusedAdamW \
  data.num_images=3 model.action_dim=16 model.state_dim=16 model.relative_action_dims=14
```

- 动作空间两种都支持，训练和评测必须成对使用：
  - 绝对动作：`model.relative_action_dims=0`（默认），评测用 `tools/embodied/ersi_server_abs.yaml`。
  - 差分动作：`model.relative_action_dims=14`，评测用 `tools/embodied/ersi_server_rel.yaml`，
    数据集 `meta/stats.json` 必须含 `relative_action`。
  - **配错会静默算错**（用 `_rel` 服务绝对动作 ckpt 会把 state 多加一遍），所以
    `ersi_server.sh` 的 `CONFIG` 没有默认值，必须显式二选一。

- 选定配置：16x1 + TorchAO FP8 + TorchFusedAdamW（见 6b）。不开 FP8 时 batch 16 会 OOM，退回 12x1。
- `--optimizer TorchFusedAdamW` 是 torch 自带的 fused AdamW，无 TE/apex 依赖；脚本里写死的 `--optimizer AdamW` 被末尾的 `"$@"` 覆盖。
- `model.action_dim` 只影响 loss 切片和输出切片；benchmark 时用 32 以和 openpi 的 loss 口径一致。
- compile 默认开（multi_group，max-autotune-no-cudagraphs），首次编译约 4 分钟（开 FP8 约 9 分钟），inductor 缓存在 `/tmp/torchinductor_root`，同形状再启动约 1 分钟。改 batch 或相机数会重编译。
- batch 选择（单卡，3 相机，compile）：

| 配置 | 每步 | 每样本 | 峰值显存 |
|---|---|---|---|
| 8x1 | 0.50 s | 62.5 ms | 55 GB |
| 12x1 | 0.59–0.64 s | 53 ms | 65 GB |
| 16x1 bf16 | OOM | | |
| 16x1 FP8 | 0.645 s | 40 ms | 65 GB |
| 16x1 FP8 + TorchFusedAdamW（选定） | 0.59 s | 37 ms | 65 GB |

## 5. 差分动作（本地改动，已 rebase 到上游 master，尚未提交到远端）

- `model.relative_action_dims`（`Pi05ModelConfig`，默认 0=绝对）：前 N 维 action 训练为 `a[t+k] - s[t]`，夹爪绝对。
- `RelativeActionTransform`（`pi05_transform.py`）在归一化前执行；builder 用 `stats["relative_action"]` 归一化，缺失直接报错。
- `PI05Policy.predict_action` 在开关打开时用 relative_action 统计反归一化，再把 state 加回前 N 维；eval server 走这条路，
  `dataset_statistics_path` 必须指向含 `relative_action` 的 stats.json，且 client 必须传 state。
- 与 openpi `DeltaActions` 逐项核对：mean/std 完全一致，q01/q99 差 <0.005（openpi 用直方图近似分位数）。
- 单测（2026-11-09 在当前 venv 通过，5 passed）：
  ```bash
  PYTHONPATH=. CUDA_MODULE_LOADING=EAGER $DATA_ROOT/envs/loongforge/bin/python -m pytest \
    tests/embodied/unit/test_pi05_relative_action.py tests/embodied/unit/test_pi05_load_pretrained.py -q
  ```

### 5.1 与上游 #313 的合并要点

上游 PR #313（`[vla] fix: add pi0.5 finetune and eval on LIBERO and RoboTwin`）改了同一批文件，rebase 时解了 3 处冲突：

- `pi05_transform.py`：上游新增 RoboTwin 的 `data.use_aloha_pi_delta_actions`（env-space 绝对关节 → pi-space delta）。
  它和 `model.relative_action_dims` 都重写 action 目标，**同时开启会叠加出错**，已加显式互斥报错。
- `modeling_pi05.py`：上游把硬编码的 `_q99_unnormalize_actions` 换成通用 `Normalizer`，并给 `predict_action`
  加了 `normalization_mode` / `image_normalize_mode` 参数。已改为按 `relative_action_dims` 选择
  `stats["relative_action"]` 或 `stats["action"]`，再走上游的 Normalizer 反归一化，最后加回 state。
  两个新参数的默认值（`q99` / `identity`）与合并前行为一致，绝对动作路径不变。
- `pi05_factory.py`：本地那条 `pi05.` 别名去重已被上游实现进 `load_pretrained`（还多了 `model.` 前缀兼容和
  `canonical_state_dict`），且 eval 侧的 `_load_pi05_pretrained` 在上游已成死代码，故直接采用上游版本。

## 6. 与 openpi-PyTorch 的 benchmark（单卡，batch 8，200 步，bf16，绝对动作，同一归一化）

| | LoongForge eager | LoongForge compile | openpi-PyTorch |
|---|---|---|---|
| 每步 | 0.73 s | 0.50 s | 0.70 s |
| 峰值显存 | 64 GB | 55 GB | 33 GB（整模型重计算） |
| loss 151–200 步 | 0.0125 | 0.0125 | 0.0133 |

逐步 loss 相关系数 0.945，实现对齐。README 的 2.80x 是对另一基线（推测为 lerobot fp32）的数字，不是对 openpi-PyTorch。

## 6b. FP8 量化训练（已试跑，可选）

pi05 支持 FP8 训练（`--fp8`，后端 `te` 或 `torchao`）。本 venv 没装 TransformerEngine，只能用 TorchAO（torchao 0.16）；
TE 后端还要求 `model.compile_model=false`。在第 4 节命令后追加：

```
--fp8 --fp8-backend torchao            # 可选 --fp8-torchao-recipe rowwise，--fp8-min-dim 1024
```

转换目标由 `PI05Policy.default_fp8_targets()` 决定：视觉塔、投影层、action/time 投影和 adaRMS 保持 bf16/fp32；
默认 `--fp8-min-dim 2048` 下 458 个 Linear 转了 218 个（PaliGemma LM 全部 + expert 的 MLP/o_proj/q_proj）。

实测（单卡，compile，差分动作，step>10 取中位数）：

| 配置 | 每步 | 每样本 | 峰值显存 |
|---|---|---|---|
| bf16 12x1 | 0.59 s | 49 ms | 65 GB |
| FP8 12x1 | 0.59 s | 49 ms | 58 GB |
| FP8 16x1 | 0.645 s | 40 ms | 65 GB |
| FP8 16x1 + TorchFusedAdamW | 0.59 s | 37 ms | 65 GB |
| bf16 16x1 | OOM | | |

同 batch 下 FP8 不省时间（GEMM 太小），但省约 7 GB 显存，正好让 batch 16 放得下，
每样本吞吐比 bf16 12x1 高约 18%；再换 `--optimizer TorchFusedAdamW`（torch 自带 fused AdamW，无额外依赖）到 37 ms/样本，合计约 25%。loss 曲线 40-60 步内与 bf16 同量级，没做更长的精度对比。
首次 compile 约 9 分钟（FP8 kernel autotune 更多）。要用就是第 4 节命令改 `PER_DEVICE_BATCH_SIZE=16` 加上面两个参数。

## 7. 待办

- [x] 2026-11-09：rebase 到上游 master（20 个提交，含 #313），3 处冲突已解，差分动作改动为本地提交
      `WIP: pi05 relative-action training + ERSI eval path, save_start_iter`。备份分支 `backup/pre-pull-20261109`。
- [ ] 把那个 WIP 提交拆成正式提交并提 PR：差分动作、`save_start_iter`、`num_inference_steps`；
      仍未入 git 的新增文件（`HANDOFF.md`、`docker/setup_venv_cci.sh`、`tools/embodied/`、两个 pi05 单测）；
      `models/` 加进 `.gitignore`。
- [ ] G2 仿真评测在另一台带 RTX 的机器上；LoongForge eval server 还没有 G2 的 payload builder，需要接。
- [ ] 正式训练建议 16x1 + FP8 + TorchFusedAdamW、relative_action_dims=14、开 wandb；先跑 200 步和 bf16 12x1 对一下 loss 曲线。
