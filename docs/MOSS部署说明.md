# MOSS 部署说明（交给执行 agent）

**目标**：在有 RTX 2060 Super 的机器上部署 MOSS-Transcribe-Diarize 0.9B 模型服务，
让 Transcriptum 通过 OpenAI-compatible `/v1/audio/transcriptions` 接口调用它，
实现“转写 + 说话人分离（一句一人）”。

**文档性质**：这份是给 agent 直接执行的任务说明。请按顺序做，每步都有验收条件。
凡是需要“改代码/改配置”的地方，先报告再动手。

---

## 0. 先读这段：三个会决定成败的事实

1. **RTX 2060 是 Turing 架构（sm_75，计算能力 7.5）**。
   主流新版本 vLLM 与 SGLang 已经**不再支持 Turing**。不要直接照官方最新 README
   装最新版然后撞墙。必须先按第 1 节确认版本兼容性。若确认装不上，按第 6 节走
   HuggingFace Transformers 原生方案，不要反复升级/降级浪费时间。

2. **官方输出格式是 `[start][Sxx]文本[end]`**，结束时间戳紧跟文本：
   ```
   [0.48][S01]Welcome everyone[1.66][12.26][S02]The pipeline is ready[13.81]
   ```
   不是“结束时间等于下一句开头”。Transcriptum 的解析器已按官方格式修好并有回归测试，
   但**部署方不要改输出格式**，也不要加前后缀说明文字。

3. **必须传 `max_new_tokens`，否则长音频会被截断**。
   官方默认 5120，cookbook 建议 65536。Transcriptum 已通过
   `TRANSCRIPTUM_ASR_MAX_NEW_TOKENS=65536` 传给模型服务，模型服务侧也要允许这个参数。

---

## 1. 任务清单（按序执行）

| 步骤 | 动作 | 验收条件 |
|---|---|---|
| 1 | 确认 GPU、驱动、CUDA、显存 | `nvidia-smi` 输出含 RTX 2060，驱动 ≥ 525，可用显存 ≥ 6GB |
| 2 | 选服务方案并安装（优先 vLLM，失败转 Transformers） | 服务进程起来，日志无 CUDA OOM |
| 3 | 下载模型权重 | 本地目录含 `config.json` 与 `*.safetensors` |
| 4 | 启动模型服务 | `curl /v1/models` 返回模型 id |
| 5 | 用一段 WAV 实测接口 | 返回 `verbose_json`，text 带 `[Sxx]` 前缀 |
| 6 | 用 `probe_moss.py` 验证 Transcriptum 解析 | 脚本打印“解析出 N 句”，N ≥ 1 |
| 7 | 把 Transcriptum 配置指向该服务 | health 显示 `asr.adapter=moss` 或 `openai_audio` |
| 8 | 跑端到端冒烟 | 33 passed, 0 failed |

---

## 2. 第 1 步：环境确认（必须先做）

在 **2060S 那台机器**执行：

```bash
nvidia-smi
nvcc --version 2>/dev/null || echo "没有 nvcc（不影响 Docker 方案）"
free -h
df -h /
python3 --version
docker --version 2>/dev/null || echo "没有 docker"
```

**硬性门槛**：

- GPU 名称含 `RTX 2060`，显存 8GB（2060 Super 是 8GB）；
- 驱动版本 ≥ 525.60（支持 CUDA 12.x）；
- 磁盘剩余 ≥ 20GB（模型权重约 2GB + 依赖）；
- 内存 ≥ 8GB。

任一项不满足就**停下来报告**，不要继续装依赖。

### 2.1 关键兼容性检查（不要跳过）

RTX 2060 的 compute capability 是 **7.5（Turing）**。请先确认你准备安装的服务版本支持它：

```bash
# 若用 vLLM：查官方文档的 GPU 支持表（搜 "compute capability 7.5" / "Turing"）
# 若用 SGLang-omni：同上，README 的 supported hardware 一节
```

判断规则：

- **支持 sm_75** → 走第 3 节服务方案；
- **明确要求 compute capability ≥ 8.0** → 跳过第 3/4 节，直接走第 6 节 Transformers 原生方案。

> 这一步是为了避免把时间耗在“最新版装不上 → 降版本 → 又装不上”的循环里。
> 我无法在此环境实测 2060S（当前机器无 NVIDIA GPU），所以这条必须由你在目标机上验证。

---

## 3. 第 2-4 步：服务方案 A —— vLLM / SGLang（如果支持 sm_75）

### 3.1 环境

```bash
sudo apt-get update && sudo apt-get install -y python3-venv ffmpeg libsndfile1
python3 -m venv ~/moss-venv
source ~/moss-venv/bin/activate
pip install -U pip
```

### 3.2 安装服务（二选一，优先 A1）

**A1. vLLM**（若版本支持 Turing）：

```bash
pip install "vllm>=0.6,<0.8"   # 具体版本以 2.1 兼容性检查为准
```

**A2. SGLang-omni**（官方推荐跑 MOSS 的方案）：

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
pip install -e ".[audio]"
```

> 国内网络：pip 加 `-i https://mirrors.aliyun.com/pypi/simple`，
> git 走代理或用镜像。若依赖下载超过 5 分钟仍无进展，报告网络情况。

### 3.3 下载模型

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内必需，否则可能连不上 huggingface.co
pip install -U huggingface_hub
huggingface-cli download OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --local-dir /opt/models/MOSS-Transcribe-Diarize
ls -lh /opt/models/MOSS-Transcribe-Diarize | head
```

验收：目录里有 `config.json`、`generation_config.json`、`*.safetensors`。

### 3.4 启动服务

按你装的服务选对应命令（**Audio Qwen2.5-Omni 系列参数**）：

```bash
# vLLM 示例
vllm serve /opt/models/MOSS-Transcribe-Diarize \
  --host 0.0.0.0 --port 9000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --dtype half
```

```bash
# SGLang-omni 示例
python -m sglang_omni.launch_server \
  --model-path /opt/models/MOSS-Transcribe-Diarize \
  --host 0.0.0.0 --port 9000 \
  --max-model-len 65536 \
  --dtype half
```

**关键参数解释**：

- `--max-model-len 65536`：必须 ≥ Transcriptum 传的 `max_new_tokens`，否则请求被拒；
- `--dtype half`：2060S 8GB 显存建议 fp16；若 OOM 再降 `--gpu-memory-utilization`；
- `--gpu-memory-utilization 0.85`：留点余量给系统。

启动验收：

```bash
curl -s http://127.0.0.1:9000/v1/models | head -c 500
```

应返回 JSON 且包含模型 id。**记下这个 id**，Transcriptum 的 `TRANSCRIPTUM_ASR_MODEL` 必须和它一致。

---

## 4. 第 5 步：实测接口（必须做，别跳过）

造一段测试音频：

```bash
ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -ac 1 -ar 16000 /tmp/probe.wav
```

调用（按实际返回的模型 id 替换）：

```bash
curl -s http://127.0.0.1:9000/v1/audio/transcriptions \
  -F file=@/tmp/probe.wav \
  -F model=MOSS-Transcribe-Diarize \
  -F response_format=verbose_json \
  -F 'max_new_tokens=65536' \
  -F 'timestamp_granularities[]=segment' | head -c 1200
```

**验收要点**：

1. HTTP 200；
2. JSON 里有 `segments` 数组；
3. 每个 `segment.text` 形如 `[S01]实际文字` —— **说话人前缀在 text 里**；
4. `start`/`end` 是秒数（浮点）。

如果服务返回的是自由文本而非 `verbose_json`，把原始返回贴出来报告 ——
Transcriptum 也支持解析 `[start][Sxx]text[end]` 全文格式，但需要确认是哪一种。

---

## 5. 第 6-7 步：接上 Transcriptum

Transcriptum 已经支持这个接口。在 **Transcriptum 容器/进程**设置：

```bash
export TRANSCRIPTUM_ASR_ADAPTER=openai_audio
export TRANSCRIPTUM_ASR_BASE_URL=http://<2060S机器IP>:9000/v1
export TRANSCRIPTUM_ASR_MODEL=MOSS-Transcribe-Diarize   # 用第 3.4 步记下的 id
export TRANSCRIPTUM_ASR_MAX_NEW_TOKENS=65536
export TRANSCRIPTUM_ASR_TIMEOUT_S=600
# 白名单必须包含模型端点（白名单是唯一授权来源）
export TRANSCRIPTUM_ALLOWED_EGRESS=localhost,127.0.0.1,<2060S机器IP或域名>
```

如果用 Docker Compose，把上面的变量写进 `.env` 再 `docker compose up -d`。

**注意**：`openai_audio` 适配器已经会剥离 `segment.text` 里的 `[Sxx]` 前缀并把说话人
交给对齐层；不要为了“看起来干净”去改适配器。

### 5.1 用 probe 脚本验证解析

在能访问 Transcriptum 代码库的机器上：

```bash
python scripts/probe_moss.py \
  --base http://<2060S机器IP>:9000
```

脚本会走 submit/poll/fetch 三段式；**但注意它默认按“作业式接口”探测**，
而 vLLM/SGLang 的 `/v1/audio/transcriptions` 是**同步接口**，没有 job 端点。
因此：

- 如果目标服务是标准 OpenAI-compatible 同步接口，**跳过 probe 脚本的 job 流程**，
  直接以第 4 步的 curl 结果为准；
- 只有当你另建了 job 包装层时，probe 脚本才适用。

更直接的验证方式：跑 Transcriptum 的端到端冒烟（下一节）。

### 5.2 端到端验收

```bash
python scripts/smoke_e2e.py --base http://<Transcriptum地址>
```

期望：**33 passed, 0 failed**。并且第 3 步“产出句子”里 speaker 为 `[Sxx]` 而非空。

---

## 6. 备选方案 B：Transformers 原生推理（当 vLLM/SGLang 不支持 sm_75 时）

如果第 2.1 步确认服务不支持 Turing，不要硬装，改用这个方案：

```bash
source ~/moss-venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece libsndfile1 soundfile
export HF_ENDPOINT=https://hf-mirror.com
```

写一个简单的 FastAPI 包装（伪代码，交给 agent 实现）：

```python
# /v1/audio/transcriptions 接收 multipart: file, model, max_new_tokens
# 1. soundfile 读音频，重采样到 16k mono
# 2. processor = AutoProcessor.from_pretrained(MODEL_PATH)
#    model = AutoModelForCausalLM.from_pretrained(
#        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda")
# 3. inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to("cuda")
# 4. ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
# 5. text = processor.batch_decode(ids, skip_special_tokens=True)[0]
# 6. 返回 {"text": text, "segments": []} 或自行切成 segment
#    —— text 必须保持 MOSS 原样 [start][Sxx]...[end]，不要清洗
```

包装层必须满足：

- 响应 `Content-Type: application/json`；
- 返回 `text` 字段（MOSS 全文），Transcriptum 会自己解析；
- 若返回 `segments`，每个 element 要有 `start`/`end`/`text`；
- 支持 `max_new_tokens` 表单字段。

这个方案 RTF 会比 vLLM 慢，但 8GB 显存跑 0.9B fp16 通常可行。
**性能不达标就报告实测 RTF，不要改模型。**

---

## 7. 已知坑（提前避开）

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 2060S 只有 8GB | 降 `--gpu-memory-utilization` 到 0.7；或 `--dtype float16` 确认没用 fp32 |
| 服务返回 400 说 max tokens 超限 | 服务 `--max-model-len` 小于 65536 | 启动服务时调大，或把 Transcriptum 的 `max_new_tokens` 降到服务允许值 |
| 说话人没出来 / 混在正文里 | 输出被改格式 | 确认 text 保留 `[Sxx]` 前缀；Transcriptum 依赖它 |
| 中文术语全错 | 没加热词 | 在 Transcriptum 的术语表维护“正确写法 + 常见错写”，hotwords 会自动注入 prompt |
| 长录音只转出前面一段 | max_new_tokens 太小 | 调到 65536；仍截则分段上传 |
| 连接被拒 | 白名单没配 | `TRANSCRIPTUM_ALLOWED_EGRESS` 必须含模型端点域名/IP |
| 模型 id 不匹配 | 配置写了错的 model | 用 `/v1/models` 返回的确切 id 填 `TRANSCRIPTUM_ASR_MODEL` |

---

## 8. 交付给执行 agent 的报告模板

部署完成后，请按这个格式回填：

```
GPU 型号 / 驱动 / 显存：
选用方案（vLLM / SGLang / Transformers）与版本：
模型权重路径：
服务监听地址与 /v1/models 返回的 model id：
/v1/audio/transcriptions 实测返回片段（前 300 字）：
Transcriptum health 输出：
smoke_e2e.py 结果（通过/失败数）：
实测 RTF（音频时长 / 处理耗时）：
遇到的问题与处理方式：
```

---

## 9. 相关文件

- `scripts/probe_moss.py` —— 作业式接口探测（注意第 5.1 节的适用限制）
- `scripts/smoke_e2e.py` —— 端到端验收
- `backend/tests/test_moss_official.py` —— 官方输出格式回归测试
- `docs/Docker部署指南.md` —— Transcriptum 容器部署
- `config.yaml` —— `asr.max_new_tokens` / `asr.timeout_s`

---

## 10. 我在本环境无法验证的部分（如实说明）

- RTX 2060（Turing sm_75）上 vLLM/SGLang 能否运行 —— 本机无 NVIDIA GPU；
- 真实模型的输出是否严格符合官方 README 的格式 —— 只按官方样例做了解析测试；
- 8GB 显存下的实际 RTF 与长音频是否 OOM。

这三项必须在目标机上实测。解析器已按官方格式修正并加了 7 项回归测试，
但真实模型若有细微差别（比如多空格、额外换行），按第 8 节模板把原始返回贴回来，我补解析器。
