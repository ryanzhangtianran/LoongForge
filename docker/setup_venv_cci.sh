#!/usr/bin/env bash
# Rebuild the LoongForge VLA venv on the CCI container (H800, driver 550 => CUDA 12.x only).
# Run as root. Afterwards: source "$VENV/bin/activate"
# Runtime notes (see HANDOFF.md): export CUDA_MODULE_LOADING=EAGER on driver 550 (cuDNN 9.19 lazy-load bug),
# UV_PROJECT_ENVIRONMENT="$VENV", tokenizer from ModelScope AI-ModelScope/paligemma-3b-pt-224.
# Override LF (repo) and VENV (target venv) to relocate; defaults match the current box.
set -euxo pipefail

export PATH=/root/.local/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
LF=${LF:-/workspace/LoongForge}
VENV=${VENV:-/workspace/venv}

# 0. system
apt-get update && apt-get install -y git ffmpeg bc tree iproute2 cuda-toolkit-12-8
command -v uv >/dev/null || pip install uv

# 1. venv + torch cu128 (torchcodec pin drags in the matching torch build)
uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
uv pip install "torchcodec==0.11.1" torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. LoongForge
[ -d $LF ] || git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git $LF
uv pip install -e "$LF[gpu]"
uv pip uninstall nvidia-modelopt || true

# 3. dmuon + lerobot (mirrors docker/Dockerfile install_lerobot)
uv pip install ninja packaging setuptools wheel
git clone https://github.com/X-Square-Robot/dmuon.git /workspace/dmuon
git -C /workspace/dmuon checkout --detach 32e97c9096571e82f4ce733f4ac726899e42b368
uv pip install --no-build-isolation /workspace/dmuon && rm -rf /workspace/dmuon
git clone --depth 1 --branch v0.5.0 https://github.com/huggingface/lerobot.git /workspace/lerobot
pushd /workspace/lerobot
sed -i '/evdev/d' requirements-ubuntu.txt 2>/dev/null || true
sed -i -E 's/"torch>=([0-9][0-9.]*),<[0-9][0-9.]*"/"torch>=\1"/; s/"torchvision>=([0-9][0-9.]*),<[0-9][0-9.]*"/"torchvision>=\1"/' pyproject.toml
uv pip install "transformers==5.3.0"
uv pip install --no-build-isolation .
popd && rm -rf /workspace/lerobot
uv pip install "numpy==1.26.4" "scipy==1.15.3"
uv pip install --no-deps --reinstall "torchcodec==0.11.1" --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps --reinstall "diffusers==0.38.0"

# 4. flash-attn 2, sm_90 only; container cgroup is 14 CPUs / 240 GB, -j64 gets OOM-killed
FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=14 uv pip install --no-build-isolation "flash-attn==2.8.3.post1"

# 5. fixes found on first install (not in the Dockerfile because the NGC image already had them)
uv pip install "pyarrow==20.0.0" "pandas==2.2.3"          # numpy 1.26 compatible
uv pip install "tensorboard==2.16.2" "protobuf==3.20.2"   # TB>=2.17 needs protobuf>=5, wandb needs <7
uv pip uninstall pynvml || true; uv pip install nvidia-ml-py
uv pip install pytest                                     # tests/embodied/unit

# 6. verify
python -W ignore -c "
import torch, flash_attn, lerobot, dmuon, diffusers, pandas, pyarrow, torchcodec, wandb, loongforge
from torch.utils.tensorboard import SummaryWriter
from lerobot.policies.pi_gemma import PiGemmaForCausalLM
from loongforge.embodied.model.pi05 import modeling_pi05
from loongforge.embodied.train.utils import logging
assert torch.__version__.endswith('+cu128') and torchcodec.__version__.endswith('+cu128'), (torch.__version__, torchcodec.__version__)
assert diffusers.__version__ == '0.38.0'
print('OK', torch.__version__, flash_attn.__version__, torch.cuda.get_device_name())"
