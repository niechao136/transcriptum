# Transcriptum —— 多阶段构建
#
# 目标：单个镜像同时提供 API 与前端静态资源（同源，免 CORS），
# 内置 ffmpeg（音频归一化必需），以非 root 运行。
#
# 构建：
#   docker build -t transcriptum:0.1.0 .
# 运行：
#   docker run -d -p 8080:8080 -v transcriptum-data:/data transcriptum:0.1.0

# ---------- 阶段 1：前端构建 ----------
FROM node:20-alpine AS frontend

WORKDIR /build
# 先只拷依赖清单，命中缓存
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ---------- 阶段 2：Python 依赖 ----------
FROM python:3.11-slim AS deps

# 国内构建时可传 --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# 默认走官方源，保持镜像构建的便携性
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /wheels
COPY backend/requirements.txt ./requirements.txt
RUN pip wheel --wheel-dir /wheels -r requirements.txt


# ---------- 阶段 3：运行时 ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="transcriptum" \
      org.opencontainers.image.description="会议与评估记录系统（本地/内网部署）" \
      org.opencontainers.image.licenses="Proprietary"

# ffmpeg/ffprobe：音频归一化与时长探测，缺了归一化直接失败
# 国内构建可传 --build-arg APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/debian
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian|$APT_MIRROR|g; s|http://security.debian.org/debian-security|$APT_MIRROR-security|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSCRIPTUM_CONFIG=/app/config.yaml \
    TRANSCRIPTUM_STATIC_DIR=/app/static \
    TRANSCRIPTUM_DATA_DIR=/data \
    TRANSCRIPTUM_DB_PATH=/data/transcriptum.db \
    PORT=8080

# 非 root 运行：数据卷归 appuser，避免容器以 root 写宿主机目录
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data /app \
    && chown -R appuser:appuser /data /app

WORKDIR /app

COPY --from=deps /wheels /wheels
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=appuser:appuser backend/transcriptum ./transcriptum
COPY --chown=appuser:appuser config.yaml ./config.yaml
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
COPY --from=frontend --chown=appuser:appuser /build/dist ./static

RUN chmod +x /entrypoint.sh

USER appuser

EXPOSE 8080

# 健康检查直连 /api/health，失败则容器标记 unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
