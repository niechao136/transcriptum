# Docker 部署指南

单镜像包含 API、Worker 和构建好的前端静态资源。镜像以非 root（uid 10001）运行。

---

## 1. 构建

```bash
docker build -t transcriptum:0.1.0 .
```

多阶段构建：Node 20 构建前端 → python:3.11-slim 安装依赖 → 运行时镜像（内置 ffmpeg）。
首次构建需联网拉取 node/python 基础镜像、npm 包和 pip 包。

无法直连 Docker Hub 时，可在 `daemon.json` 配 registry-mirrors（srv-02 已配好）：
```json
{"registry-mirrors": ["https://docker.m.daocloud.io", "https://docker.1ms.run"]}
```

## 2. 单机启动（API + Worker 同容器）

```bash
cp .env.example .env      # 按需修改模型端点与白名单
docker compose up -d
curl http://127.0.0.1:8080/api/health
curl http://127.0.0.1:8080/            # 前端页面
```

- 数据卷 `transcriptum-data` 挂载到 `/data`，保存 SQLite 库和音频；
- 容器启动会通过 entrypoint 自动执行幂等的 `--init-db`；
- 默认使用 mock 适配器，无需模型即可验证全链路。

## 3. 接入内网模型

编辑 `.env`（**白名单是唯一授权来源**，非 localhost 端点必须显式加入）：

```env
TRANSCRIPTUM_ASR_ADAPTER=moss
TRANSCRIPTUM_ASR_BASE_URL=http://gpu-box.internal:9000
TRANSCRIPTUM_LLM_ADAPTER=openai_compatible
TRANSCRIPTUM_LLM_BASE_URL=http://gpu-box.internal:8000/v1
TRANSCRIPTUM_ALLOWED_EGRESS=localhost,127.0.0.1,*.internal
ASR_API_KEY=...
LLM_API_KEY=...
```

如果模型跑在宿主机上，Linux 用 `host.docker.internal` 需要在 compose 里加
`extra_hosts: ["host.docker.internal:host-gateway"]`（本仓库已加）。

**注意**：当前 srv-02 只有 2 核、≈1GB 可用内存，不足以运行 MOSS 0.9B。
等带 RTX 2060 Super 的机器就绪后，把模型服务部署到那台机，再让此容器指向它。

## 4. API / Worker 分离（多副本）

```bash
docker compose --profile split up -d
```

- `transcriptum` 服务设 `TRANSCRIPTUM_RUN_WORKER=0`，只提供 API，可 `docker compose up -d --scale transcriptum=3`；
- `transcriptum-worker` 服务执行 `worker` 子命令，单独轮询队列（SQLite 队列表保证同一任务只被一个 worker 领取）。

## 5. 运维

```bash
docker compose logs -f transcriptum           # 日志
docker compose exec transcriptum shell        # 进容器（sh）
docker compose exec transcriptum python -m transcriptum --config /app/config.yaml --purge --days 30
docker inspect --format '{{.State.Health.Status}}' <container>   # 健康检查
```

## 6. 验证

```bash
# 镜像内自带脚本：容器内跑冒烟（用 mock）
docker run --rm --network host transcriptum:0.1.0 \
  python /app/scripts/smoke_e2e.py --base http://127.0.0.1:8080
```

> 注意：`scripts/` 未拷入镜像，冒烟脚本请从宿主机运行
> `python scripts/smoke_e2e.py --base http://127.0.0.1:8080`。
> 镜像内置 `/api/health` 健康检查，前端页面在 `/`。

## 7. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 启动即报“数据目录不可写” | 挂载卷属主不对，需 `chown 10001:10001 <host_dir>` |
| 配置启动报 `egress not allowed` | 模型端点不在白名单，加入 `TRANSCRIPTUM_ALLOWED_EGRESS` |
| 前端 404 | 镜像构建时前端阶段失败；检查 build log |
| 上传后转写卡住 | 查看 worker 日志：`docker compose logs -f transcriptum` |
| 中文文件名下载 | 已用 RFC 5987 处理，无需额外配置 |
