"""CLI 入口：启动 API + worker、单独跑 worker、初始化库、跑保留期清理。

容器部署时 API 与 worker 可拆成两类进程（WORKER 容器只跑 worker 子命令），
本地开发默认同进程启动，行为保持不变。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from .config import Config
from .db import Db, connect, init_db
from .service import seed_frameworks
from .worker import Worker, WorkerDeps


def _bootstrap(config_path: str) -> tuple[Config, Db]:
    cfg = Config.load(config_path)
    conn = connect(cfg.db_path)
    init_db(conn)
    db = Db(conn)
    with db.tx():
        seed_frameworks(db)
    return cfg, db


def _run_worker_only(cfg: Config, db: Db) -> int:
    """只跑 worker（容器编排里作为独立部署单元）。"""
    deps = WorkerDeps.build(cfg, db)
    w = Worker(deps)

    stop = threading.Event()

    def _shutdown(signum, _frame):
        logging.getLogger("transcriptum").info("收到信号 %s，准备退出 worker", signum)
        w.stop()
        stop.set()

    # 容器 stop 发 SIGTERM：要能优雅退出，不能硬杀导致任务卡在 running
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print("worker 已启动（独立进程，单进程轮询）")
    log = logging.getLogger("transcriptum")
    while not stop.is_set():
        try:
            # run_once 返回是否有活干；空闲时短暂休眠，避免空转打满 CPU
            busy = w.run_once()
        except Exception:  # noqa: BLE001
            # 单条任务失败不应让 worker 进程整体退出（容器里会被反复重启）
            log.exception("worker 主循环异常")
            busy = False
        if not busy:
            stop.wait(cfg.queue.poll_interval_s)
    print("worker 已退出")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("transcriptum")
    p.add_argument("--config", default=os.environ.get("TRANSCRIPTUM_CONFIG", "config.yaml"))
    p.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    p.add_argument("--no-worker", action="store_true", help="只起 API，不起 worker")
    p.add_argument("--init-db", action="store_true")
    p.add_argument("--purge", action="store_true", help="执行保留期清理后退出")
    p.add_argument("--days", type=int, default=None,
                   help="purge 时覆盖保留天数（默认取配置 audio_retention_days）")
    p.add_argument(
        "command",
        nargs="?",
        choices=["serve", "worker", "purge"],
        help="子命令：serve（默认）/ worker / purge",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg, db = _bootstrap(args.config)

    if args.init_db:
        print(f"数据库已初始化：{cfg.db_path}")
        return 0

    if args.purge or args.command == "purge":
        from .service import RecordingService

        deps = WorkerDeps.build(cfg, db)
        svc = RecordingService(cfg, db, deps)
        print(svc.purge_expired(days=args.days))
        return 0

    if args.command == "worker":
        return _run_worker_only(cfg, db)

    import uvicorn

    from .api import create_app

    app = create_app(cfg)
    if not args.no_worker:
        deps = WorkerDeps.build(cfg, db)
        w = Worker(deps)
        threading.Thread(target=w.run_forever, daemon=True).start()
        print("worker 已启动（单进程轮询）")

    # 容器内必须监听 0.0.0.0，否则宿主机访问不到
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
