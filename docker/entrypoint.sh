#!/bin/sh
# 容器入口：初始化数据库后按子命令启动。
set -eu

# 数据目录必须可写，否则 sqlite 会在运行期才报错，排查困难
DATA_DIR="${TRANSCRIPTUM_DATA_DIR:-/data}"
mkdir -p "$DATA_DIR"
if [ ! -w "$DATA_DIR" ]; then
    echo "FATAL: 数据目录不可写：$DATA_DIR" >&2
    echo "       请检查挂载卷的属主（容器内 uid 10001）" >&2
    exit 1
fi

PY="python -m transcriptum --config ${TRANSCRIPTUM_CONFIG:-/app/config.yaml}"

# 任何子命令前都先确保 schema 是最新的（幂等）
$PY --init-db >/dev/null

case "${1:-serve}" in
    serve)
        # WORKER=0 时只起 API（多副本部署时让单独一个容器跑 worker）
        if [ "${TRANSCRIPTUM_RUN_WORKER:-1}" = "0" ]; then
            exec $PY --port "${PORT:-8080}" --no-worker
        fi
        exec $PY --port "${PORT:-8080}"
        ;;
    worker)
        exec $PY worker
        ;;
    purge)
        shift
        exec $PY --purge "$@"
        ;;
    shell)
        exec /bin/sh
        ;;
    *)
        exec "$@"
        ;;
esac
