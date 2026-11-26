# MOSS 容器化部署说明

把当前 WSL 上手工搭好的 MOSS 推理服务固化为**单容器 + 两个脚本**，不用 compose。

覆盖两个场景：
1. **本机重装** —— 权重已在本地，跳过下载，几分钟内恢复；
2. **新机器从零** —— 自动探测 GPU、装好前置条件、下载权重并校验。

公网层（SSH 反向隧道 + nginx）**不在这套封装内**，保持现状。

---

## 为什么改成容器

当前裸机部署中，以下步骤全是因为 WSL 没有 sudo 而不得不做的绕路：

- 用 `zig cc` 冒充 C 编译器（Triton JIT 需要）
- 手工 `apt-get download libpython3.12-dev` 解压出 `Python.h`
- 静态 ffmpeg / ffprobe 逐个找镜像

容器里以 root 直接 `apt install gcc python3-dev ffmpeg` 即可，这些绕路全部消失。而且 vLLM 官方镜像 `vllm/vllm-openai:v0.34.0` 已把 CUDA/torch/vLLM 固化好，无需自建。

> 代价：需要宿主机安装 **NVIDIA Container Toolkit**（当前 WSL 的 Docker 还没有）。

---

## GPU 自动适配

`setup.sh` 读取 `nvidia-smi` 的 `compute_cap` 和显存，写入 `.deploy-state`；`run-moss.sh` 据此选参数：

| 档位 | 条件 | 参数要点 |
|---|---|---|
| `turing` | compute capability < 8.0（如 RTX 2060S/2070S） | `TRITON_ATTN` + `enforce-eager` + `skip-mm-profiling`，`half`，`max-num-seqs 1`，`max-model-len 4096` |
| `modern` | compute capability ≥ 8.0（30/40 系等） | `FLASH_ATTN`、`bfloat16`、`max-num-seqs 16`、`max-model-len 32768` |

理由：FA2 仅支持 sm_80+；8GB Turing 卡上 MM profiling 会把可用 KV cache 压成负数。

---

## 依赖清单

| 类别 | 内容 | 由谁提供 |
|---|---|---|
| Docker | 29.9+ | 宿主 |
| GPU 运行时 | NVIDIA Container Toolkit | `setup.sh` 检测、给出安装命令 |
| 镜像 | `vllm/vllm-openai:v0.34.0`（约 10GB） | `setup.sh` 拉取 |
| 编译器 | gcc + python3-dev | 官方镜像内已包含 |
| ffmpeg/ffprobe | 宿主机已有（应用侧用） | 不在容器内 |
| 模型权重 | 1.82GB safetensors + 15 个配套文件 | `download_moss_shard.py` |
| 端口 | 9010（容器映射）→ 应用配置 `asr.base_url` | `run-moss.sh` |

---

## 使用步骤

### 一次性准备

```bash
bash scripts/moss-container/setup.sh
```

它会：
1. 探测 GPU 与显存，决定档位；
2. 检查 NVIDIA Container Toolkit（缺失时打印安装命令并停止，**不代输 sudo 密码**）；
3. 拉取官方镜像；
4. 校验或下载权重；**下载器会同时补齐 config/tokenizer/processor 等 15 个必需配套文件**（旧版只下权重，新机器会缺文件起不来）；
5. 写出 `scripts/moss-container/.deploy-state`。

本机重装时，权重校验通过即跳过下载。

### 日常运维

```bash
bash scripts/moss-container/run-moss.sh start    # 启动（幂等，自动选档）
bash scripts/moss-container/run-moss.sh status   # 状态
bash scripts/moss-container/run-moss.sh logs     # 跟随日志
bash scripts/moss-container/run-moss.sh stop     # 停止并移除容器
```

启动后等待 `/health` 返回 200（首次加载模型约 1–3 分钟）。容器加了 `--restart unless-stopped`，开机自启。

### 路径覆盖

```bash
MOSS_MODEL_DIR=/data/models/moss bash scripts/moss-container/setup.sh
MOSS_PORT=9011 bash scripts/moss-container/run-moss.sh start
```

---

## 与应用的衔接

应用（Transcriptum）仍按现有方式部署，配置里保持：

```yaml
asr:
  adapter: openai_audio
  base_url: http://127.0.0.1:9010/v1
```

容器只是把 `:9010` 上原来的裸机 vLLM 换成 docker 映射，应用侧无需改动。

---

## 当前状态

已完整实测通过（容器化部署已在本机跑通）：

- `scripts/moss-container/setup.sh`
- `scripts/moss-container/run-moss.sh`
- `scripts/download_moss_shard.py`（新增 15 个配套文件下载）

## API 文档访问地址

容器同时绑定两个端口，均为 `127.0.0.1` → 容器 `8000`：

| 用途 | 端口 | 说明 |
|---|---|---|
| 应用调用 | 9010 | Transcriptum 的 `asr.base_url` 指向此处 |
| 文档/调试 | 9090 | 专供浏览器访问 Swagger，避开 Windows 侧端口冲突 |

### 为什么要单独一个 9090

Windows 上 **Logitech G Hub（`lghub_agent.exe`）占用了 9010**，浏览器打开 `127.0.0.1:9010` 会打到 G Hub 并返回 `426 Upgrade Required`，而不是 vLLM。因此容器额外映射 `9090`，Windows 浏览器用这个端口即可。

### 本机（Windows 浏览器）

- Swagger UI：http://127.0.0.1:9090/docs
- ReDoc：http://127.0.0.1:9090/redoc
- OpenAPI：http://127.0.0.1:9090/openapi.json

### 公网

- Swagger UI：http://139.224.13.57:8082/docs
- ReDoc：http://139.224.13.57:8082/redoc
- OpenAPI：http://139.224.13.57:8082/openapi.json

链路：`8082` nginx → 隧道 `127.0.0.1:18082` → WSL `9090` → 容器 `8000`。

> 备用入口：`http://139.224.13.57:8081/moss/docs`（复用已放行端口，无需开安全组）。
> 但该入口**只能浏览，不能用 "Try it out"** —— OpenAPI 里的 path 是根相对路径，页面从 `/moss/` 发起调用会打到 Transcriptum 而非 MOSS。要交互调试请用 8082。

### 云端相关 systemd 单元

| 单元 | 作用 |
|---|---|
| `moss-tunnel.service` | `18081` → WSL `8080`（Transcriptum 应用） |
| `moss-docs-tunnel.service` | `18082` → WSL `9090`（MOSS 文档/API） |

## 实测验证记录（2026-11-26）

已在本机（Ubuntu 24.04 WSL2 / RTX 2070 SUPER / 驱动 610.88）完整验证：

| 检查项 | 结果 |
|---|---|
| NVIDIA Container Toolkit 1.14（阿里云镜像装 deb） | ✅ |
| `docker run --gpus all hello-world` | ✅ |
| 镜像拉取 `vllm/vllm-openai:v0.34.0`（36.3GB / 40 层） | ✅ |
| setup.sh（GPU 探测、权重 SHA256 校验、状态写入） | ✅ |
| 容器启动 + `/health` 200 | ✅ |
| `/v1/models` 返回模型 ID | ✅ |
| 真实中文语音转录：4 句 / 2 说话人 | ✅（6.2s / 19s 音频） |
| Transcriptum 端到端：`status=transcribed`，`duration_ms=18498` | ✅ |

### 验证中发现并已修复的两个问题

**1. WSL2 下容器内 CUDA 报错 `Error 500: named symbol not found`**

镜像自带 `libcuda.so.1`（580 版）与宿主驱动（610.88）不匹配。`run-moss.sh` 现在会检测 `/usr/lib/wsl/lib`，建立一个带正确符号链接（`libcuda.so.1` 等）的目录挂进容器并优先加载。

关键点：**只挂载含 `libcuda.so.1.1` 的驱动目录无效**，必须有 `libcuda.so.1` 这个符号链接，否则加载器仍回退到镜像自带库。

**2. Docker 镜像源选择**

- `docker.1panel.live` 在 3.87GiB 大层上反复 `unexpected EOF`
- 最终用 `docker.m.daocloud.io` 单源成功拉完
- daemon 里若配置了失效的自定义 DNS（如 `223.5.5.5` 在容器网络不可达），会导致 pull 卡死——移除后恢复

> 现网运行的仍是容器版 MOSS（容器名 `moss-vllm`）。原来的裸机 systemd 服务 `moss-serve` 已停止但单元文件保留，可随时切回。

### 安全提醒

排查镜像拉取时曾临时写过含 sudo 密码的脚本，已删除；正式脚本只打印安装命令，不含任何凭据。建议按之前提醒修改该密码。
</content>