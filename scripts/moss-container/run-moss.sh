#!/bin/bash
# MOSS 容器化部署 —— 启动/停止/状态（单容器，无需 compose）
#
# 用法：
#   bash scripts/moss-container/run-moss.sh start     # 启动（自动选参数档位）
#   bash scripts/moss-container/run-moss.sh stop      # 停止并移除容器
#   bash scripts/moss-container/run-moss.sh status    # 查看状态
#   bash scripts/moss-container/run-moss.sh logs      # 跟随日志
#
# 依赖 setup.sh 先执行（镜像 + 权重 + GPU 档位）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE="$ROOT/scripts/moss-container/.deploy-state"
CONTAINER="${MOSS_CONTAINER:-moss-vllm}"
PORT="${MOSS_PORT:-9010}"

[[ -f "$STATE" ]] || { echo "请先运行 setup.sh" >&2; exit 1; }
# shellcheck source=/dev/null
source "$STATE"

usage() { echo "用法: $0 {start|stop|status|logs}" >&2; }

case "${1:-}" in
  stop)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    echo "已停止 $CONTAINER"
    ;;
  status)
    docker ps -a --filter "name=^/${CONTAINER}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  logs)
    exec docker logs -f "$CONTAINER"
    ;;
  start)
    # 幂等：已运行则直接返回
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      echo "$CONTAINER 已在运行"
      exit 0
    fi
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

    # 按 GPU 档位选参数：
    # - Turing(sm75)：FA2 不支持、显存小 → TRITON_ATTN + enforce-eager + 低并发
    # - Modern(sm80+)：FA2 + bf16 + 放宽并发
    if [[ "$TIER" = "turing" ]]; then
      EXTRA_ARGS=(
        --max-num-seqs 1
        --max-num-batched-tokens 2048
        --max-model-len 4096
        --gpu-memory-utilization 0.95
        --dtype half
        --skip-mm-profiling
        --enforce-eager
        --attention-backend TRITON_ATTN
      )
    else
      EXTRA_ARGS=(
        --max-num-seqs 16
        --max-model-len 32768
        --gpu-memory-utilization 0.90
        --dtype bfloat16
        --attention-backend FLASH_ATTN
      )
    fi

    echo "启动 $CONTAINER（档位 $TIER，端口 $PORT）"

    # WSL2 下把宿主 CUDA 库挂进容器，并让动态链接器优先使用它。
    # 镜像自带 libcuda.so.1（旧版）与宿主驱动不匹配会报 CUDA Error 500。
    # 关键点：必须提供 libcuda.so.1 符号链接，仅挂含 .so.1.1 的目录无效。
    DOCKER_ARGS=()
    if [[ -d /usr/lib/wsl/lib ]]; then
      CUDA_LIB_DIR="${HOME}/.cache/moss-wsl-cuda"
      mkdir -p "$CUDA_LIB_DIR"
      ln -sf /usr/lib/wsl/lib/libcuda.so.1.1 "$CUDA_LIB_DIR/libcuda.so.1"
      ln -sf /usr/lib/wsl/lib/libcuda.so.1.1 "$CUDA_LIB_DIR/libcuda.so"
      ln -sf /usr/lib/wsl/lib/libcuda.so.1.1 "$CUDA_LIB_DIR/libcuda.so.1.1"
      echo "  检测到 WSL，挂载 CUDA 库: $CUDA_LIB_DIR"
      DOCKER_ARGS+=(
        -v "$CUDA_LIB_DIR:/wsl-cuda-libs:ro"
        -v /usr/lib/wsl/drivers:/usr/lib/wsl/drivers:ro
        -e LD_LIBRARY_PATH=/wsl-cuda-libs
      )
    fi

    docker run -d \
      --name "$CONTAINER" \
      --gpus all \
      --restart unless-stopped \
      --shm-size 8g \
      -p "127.0.0.1:${PORT}:8000" \
      -v "$MODEL_DIR:/models/MOSS-Transcribe-Diarize:ro" \
      "${DOCKER_ARGS[@]}" \
      -e HF_HUB_OFFLINE=1 \
      "$IMAGE" \
      /models/MOSS-Transcribe-Diarize \
      --served-model-name MOSS-Transcribe-Diarize \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      "${EXTRA_ARGS[@]}"

    echo "等待服务就绪（首次加载模型约 1-3 分钟）…"
    for i in $(seq 1 90); do
      if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "READY: http://127.0.0.1:${PORT}/v1"
        exit 0
      fi
      sleep 2
    done
    echo "超时未就绪，请查看：bash scripts/moss-container/run-moss.sh logs" >&2
    exit 1
    ;;
  *)
    usage
    exit 2
    ;;
esac
