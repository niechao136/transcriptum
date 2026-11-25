# Transcriptum + MOSS 公网访问部署说明

参照 RAGFlow（`docs/` 中 WSL 端口转发方案）的架构，把本地 WSL 里的
Transcriptum 前端/后端 + MOSS 转写服务转发到上海服务器，供公网访问。

## 访问地址

**http://139.224.13.57:8081**

## 架构

```
公网浏览器
   │  http://139.224.13.57:8081
   ▼
上海服务器 srv-02 (139.224.13.57)
   │  nginx :8081（sites-available/moss）
   ▼
   127.0.0.1:18081
   │  SSH 反向隧道（WSL 主动连出，-R）
   ▼
本地 WSL
   ├─ 127.0.0.1:8080  Transcriptum（FastAPI，同时托管前端 dist 与 /api）
   │      │ openai_audio adapter → http://127.0.0.1:9010/v1
   │      ▼
   └─ 127.0.0.1:9010  vLLM + MOSS-Transcribe-Diarize（RTX 2070S, sm_75）
```

端口规划（避让已有服务）：

| 位置 | 端口 | 用途 |
|---|---|---|
| 服务器公网 | 8081 | nginx 入口（RAGFlow 占 80） |
| 服务器回环 | 18081 | 隧道远端（18080 已被 RAGFlow 隧道占用） |
| WSL | 8080 | Transcriptum |
| WSL | 9010 | MOSS vLLM |

## 组件与自启

全部为 systemd **用户级**服务（`loginctl enable-linger` 已开启，会话关闭不影响）。

| 单元 | 作用 |
|---|---|
| `moss-serve.service` | vLLM 加载 MOSS，监听 9010 |
| `transcriptum.service` | 后端 + 前端，监听 8080；`ExecStartPre` 等 MOSS 就绪 |
| `moss-tunnel.service` | SSH 反向隧道 18081→8080；`ExecStartPre` 等后端就绪 |

服务器侧：`/etc/nginx/sites-available/moss`（已 enabled），后端未就绪时返回
503 友好页并 15 秒自动刷新，不再是裸 502。

## 常用运维命令

```bash
# —— 本地 WSL ——
systemctl --user status moss-serve transcriptum moss-tunnel
systemctl --user restart moss-tunnel          # 改服务器 SSH 后
systemctl --user restart transcriptum
systemctl --user restart moss-serve           # 重新加载模型，约 1-3 分钟
journalctl --user -u moss-tunnel -f
journalctl --user -u transcriptum -f

# —— 上海服务器 srv-02 ——
nginx -t && systemctl reload nginx
tail -f /var/log/nginx/error.log
ss -ltn | grep -E ':(8081|18081)\b'
```

## 验收记录（2026-11-25）

| 检查项 | 结果 |
|---|---|
| WSL 本地 `/`（UI） | 200 |
| WSL 本地 `/api/health` | 200 |
| 服务器 nginx 8081 `/` | 200（0.057s） |
| **公网 http://139.224.13.57:8081** | **200（0.09s）** |
| 公网端到端转写（4 句 / 2 说话人） | ✅ S01/S02 时间戳正确 |
| 音频 Range 流式播放 | 206 |
| 公网连续压测 | 10/10 |
| 隧道被杀后自愈 | 12 秒内重建 ✅ |
| 冷启动（`wsl --shutdown` 后） | 全部单元自动 active ✅ |

## 前提依赖：WSL 保活（关键）

WSL2 默认 `vmIdleTimeout=60s`：**Windows 侧没有任何会话句柄时，60 秒后 WSL 会
自动 `systemctl poweroff`**，三个服务随虚拟机一起消失（表现为公网 503）。

已具备的两道防线：

1. `C:\Users\niech\.wslconfig` 中 `vmIdleTimeout=86400000`（24 小时）。
2. 启动项 `WSL-KeepAlive.vbs`，登录时静默拉起
   `wsl.exe -d Ubuntu -e sleep infinity`，让 Windows 侧始终持有会话。

> ⚠️ 若发现公网突然 503，先查 Windows 侧是否还有 `wsl.exe ... sleep infinity` 进程：
> ```powershell
> Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" | Select ProcessId,CommandLine
> ```
> 没有就手动拉起（或重新登录触发启动项）：
> ```powershell
> Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','sleep','infinity' -WindowStyle Hidden
> ```
> 多个保活进程并存无害，**不要用 `pkill`/`taskkill` 按名字批量清理**，
> 容易误杀会话导致 WSL 立即关机。

验证：静默 150 秒（远超原 60 秒阈值）后公网仍 200 ✅

## 注意事项

1. **本机关机/休眠/换网络 → 公网立即不可用**，这是本地部署的固有代价。
2. **无任何认证**：任何知道 IP 的人都能打开并使用（含转写与导出）。
   如需收紧，可给 nginx 加 IP 白名单（不要用 basic auth，RAGFlow 的教训是
   前端 XHR 会与 basic auth 冲突；此项目前端同理）。
3. **仅单用户/小并发**：受 8GB 显存限制，MOSS 配置为 `max-num-seqs 1`，
   并发转写会排队。
4. 安全组已放行 TCP 8081；若后续更换端口，需同步加安全组规则。
