"""CLI 入口：启动 API + worker、初始化库、跑保留期清理。"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from .config import Config
from .db import Db, connect, init_db
from .service import seed_frameworks
from .worker import Worker, WorkerDeps


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("transcriptum")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-worker", action="store_true", help="只起 API，不起 worker")
    p.add_argument("--init-db", action="store_true")
    p.add_argument("--purge", action="store_true", help="执行保留期清理后退出")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = Config.load(args.config)
    conn = connect(cfg.db_path)
    init_db(conn)
    db = Db(conn)
    with db.tx():
        seed_frameworks(db)

    if args.init_db:
        print(f"数据库已初始化：{cfg.db_path}")
        return 0

    if args.purge:
        from .service import RecordingService

        deps = WorkerDeps.build(cfg, db)
        svc = RecordingService(cfg, db, deps)
        print(svc.purge_expired())
        return 0

    import uvicorn

    from .api import create_app

    app = create_app(cfg)
    if not args.no_worker:
        deps = WorkerDeps.build(cfg, db)
        w = Worker(deps)
        threading.Thread(target=w.run_forever, daemon=True).start()
        print("worker 已启动（单进程轮询）")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
