#!/bin/bash
# MOSS 容器化部署 —— 前置准备（一次性）
#
# 覆盖两个场景：
#   1) 本机重装：权重已在本地，跳过下载
#   2) 新机器从零：下载权重 + 校验 + 安装 NVIDIA Container Toolkit
#
# 用法：
#   bash scripts/moss-container/setup.sh
#
# 注意：本脚本不启动容器、不占用 GPU。启动交给 run-moss.sh。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${MOSS_MODEL_DIR:-$HOME/models/MOSS-Transcribe-Diarize}"
IMAGE="${MOSS_IMAGE:-vllm/vllm-openai:v0.34.0}"
EXPECTED_SHA256="9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4"
SHARD="$MODEL_DIR/model-00000-of-00001.safetensors"

log() { echo "[setup] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

log "项目根目录: $ROOT"
log "模型目录:   $MODEL_DIR"

# ---------- 1. 基础命令 ----------
for c in docker curl python3 nvidia-smi; do
  command -v "$c" >/dev/null 2>&1 || die "缺少命令: $c"
done

# ---------- 2. GPU 探测与参数档位 ----------
read -r GPU_NAME GPU_CC VRAM_MB < <(nvidia-smi \
  --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits \
  | head -1 | tr -d ' ' | tr ',' ' ')
[[ -n "${GPU_CC:-}" ]] || die "无法读取 GPU compute capability"
VRAM_GI=$((VRAM_MB / 1024))
log "GPU: $GPU_NAME (sm_${GPU_CC//./}, ${VRAM_GI} GiB)"

# Turing/sm75 必须走保守档；sm80+ 可用 FlashAttention-2 / bf16
if awk -v c="$GPU_CC" 'BEGIN{exit !(c<8.0)}'; then
  TIER="turing"
  log "档位: Turing（TRITON_ATTN + enforce-eager + skip-mm-profiling）"
else
  TIER="modern"
  log "档位: Modern（FLASH_ATTN，放宽并发）"
fi

# ---------- 3. NVIDIA Container Toolkit ----------
if docker info 2>/dev/null | grep -qi nvidia; then
  log "NVIDIA Container Toolkit: 已就绪"
else
  log "NVIDIA Container Toolkit: 未配置"
  cat <<'EOF'

  Docker 当前没有 NVIDIA runtime。请在有 sudo 权限的终端执行：

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

  完成后重新运行本脚本。

EOF
  die "NVIDIA Container Toolkit 未配置（脚本不会代输 sudo 密码）"
fi

# ---------- 4. 拉取镜像 ----------
log "拉取镜像 $IMAGE（约 10GB，可耐心等待）"
docker pull "$IMAGE"

# ---------- 5. 权重 ----------
if [[ -f "$SHARD" ]]; then
  ACTUAL=$(sha256sum "$SHARD" | awk '{print $1}')
  if [[ "$ACTUAL" = "$EXPECTED_SHA256" ]]; then
    log "权重已存在且 SHA256 正确，跳过下载"
  else
    die "权重 SHA256 不匹配（实际 $ACTUAL），请删除后重新下载"
  fi
else
  log "权重不存在，开始下载（1.8GB，走 hf-mirror）"
  mkdir -p "$MODEL_DIR"
  # 下载路径由 MOSS_MODEL_DIR 控制（旧版本脚本硬编码为 ~/models/...）
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  MOSS_MODEL_DIR="$MODEL_DIR" \
  python3 "$ROOT/scripts/download_moss_shard.py"
  [[ -f "$SHARD" ]] || die "下载脚本未产出 $SHARD"
fi

# ---------- 6. 写出参数档位 ----------
STATE="$ROOT/scripts/moss-container/.deploy-state"
cat > "$STATE" <<EOF
IMAGE=$IMAGE
MODEL_DIR=$MODEL_DIR
GPU_CC=$GPU_CC
VRAM_GI=$VRAM_GI
TIER=$TIER
EOF
log "参数档位已写入 $STATE"
log "完成。启动服务请运行：bash scripts/moss-container/run-moss.sh"
