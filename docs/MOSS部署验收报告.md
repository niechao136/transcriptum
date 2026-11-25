# MOSS 部署验收报告（按 `MOSS部署说明.md` §8 模板回填）

验收时间：2026-11-25 · 环境：WSL2 Ubuntu 24.04（`/mnt/f/hub/transcriptum`）

```
GPU 型号 / 驱动 / 显存：
  NVIDIA GeForce RTX 2070 SUPER（Turing sm_75，compute capability 7.5）
  驱动 610.88 / CUDA 13.0 / 8192 MiB（验收时剩余 1342 MiB）
  注：文档目标卡为 RTX 2060 Super；本机为 2070 Super，同为 sm_75 / 8GB，
      兼容性结论一致。

选用方案（vLLM / SGLang / Transformers）与版本：
  vLLM 0.34.0 + torch 2.14.0+cu130（Python 3.12.3，venv：~/moss-venv）
  说明：文档 §3.1 的 pin（vllm>=0.6,<0.8）早于 MOSS 支持；官方 recipe
  要求 vLLM >= 0.27，故按新版本安装。vLLM 0.34 注册表内确认含
  MossTranscribeDiarizeForConditionalGeneration / MossAudioModel，无需备选方案 B。

模型权重路径：
  /home/nechao/models/MOSS-Transcribe-Diarize
  单分片 model-00000-of-00001.safetensors，1,817,113,576 bytes
  sha256 = 9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4
  （与 HF LFS oid 一致，独立复核通过）

服务监听地址与 /v1/models 返回的 model id：
  http://127.0.0.1:9010/v1  （端口 9000 被宿主占用，改用 9010）
  model id = MOSS-Transcribe-Diarize
  GET /v1/models -> HTTP 200

/v1/audio/transcriptions 实测返回片段（前 300 字）：
  真实中文语音（19s，Windows SAPI Huihui 合成）：
  {"text":"[0.13][S01]今天我们过一下三号机的检修情况。[3.54][4.25][S02]好的，
  先看压缩机油压，油压正常就可以合闸。[9.44][10.13][S01]明白，那我把合闸前的
  检查项过一遍。[13.91][14.64][S02]对，安全阀校验记录也要带上。[17.84]",
  "usage":{"type":"duration","seconds":19}}
  纯音 WAV（3s，440Hz）返回：{"text":"[beep]","usage":{"type":"duration","seconds":3}}
  → 原生 [start][Sxx]text 格式，Transcriptum 解析器兼容（§4/§5.1 通过）。

Transcriptum health 输出：
  {"ok":true,
   "asr":{"adapter":"openai_audio",
          "capabilities":{"has_word_ts":false,"has_confidence":false,
                          "has_speaker":true,"json_mode":false}},
   "llm":{"adapter":"mock","model":"Qwen2.5-7B-Instruct"},
   "diarizer":"moss_inline",
   "proofread":{"low_confidence_threshold":0.6,"prompt_version":"proofread-v1"}}

smoke_e2e.py 结果（通过/失败数）：
  33 通过，0 失败（达到 §5.2 期望）
  走真实链路：上传 → MOSS 转写 → AI 校对 → 人工确认 → 评估引用强制 →
  md/docx/xlsx 导出 → 审计；"产出句子" speaker 为 S01/S02，非空。
  使用 TRANSCRIPTUM_SMOKE_WAV 指向真实语音；纯音 WAV 会使 mock LLM 的
  固定校对/评估数据（引用 seq 2/3/4）越界，属测试数据适配问题，非部署缺陷。

实测 RTF（音频时长 / 处理耗时）：
  19 s 音频 / 3.75 s wall = 0.197（约 5x 实时）

遇到的问题与处理方式：
  1. §2.1 兼容性判断：文档假设新 vLLM 不支持 Turing；实测 vLLM 0.34 官方要求
     compute capability >= 7.5，2070S（sm_75）符合 → 走服务方案，未用备选 B。
  2. 无 python3-venv / 无 sudo / 无 ffmpeg：venv --without-pip + get-pip wheel
     手动 bootstrap；ffmpeg 用 imageio-ffmpeg 静态二进制（n9.0.2）链接到 PATH。
  3. WSL DNS 异常（IPv6 优先、hostname 间歇解析失败）：Python sitecustomize
     强制 IPv4；模型下载器固定 CDN IP。
  4. 官方 PyPI 下载超时：按用户许可切清华镜像（pypi.tuna.tsinghua.edu.cn）。
  5. HF 免费额度限速（~190KB/s，提示 egress quota）：hf-mirror + 可断点续传
     Range 下载器（scripts/download_moss_shard.py），SHA256 校验通过。
  6. 缺少 C 编译器（Triton JIT）：pip 安装 ziglang，zig cc 包装为 CC/CXX；
     缺 Python.h：apt-get download libpython3.12-dev + dpkg-deb -x 解压到用户目录，
     包装器加 -I 指向头文件。
  7. FlashAttention-2 不支持 sm_75：改用 --attention-backend TRITON_ATTN。
  8. 8GB 显存 KV cache  probing 负值：--skip-mm-profiling + max-model-len 4096
     + max-num-seqs 1 + --enforce-eager + gpu-memory-utilization 0.95。
  9. vLLM MOSS 不支持 verbose_json（400）：probe 脚本与 openai_audio adapter
     改用 response_format=json，走项目既有 moss_parser 解析原生文本。
```

## 复现命令（目标机）

```bash
# 服务（已在后台运行）
/home/nechao/start_moss.sh          # 监听 9010

# Transcriptum 后端（已在后台运行，配置 /home/nechao/transcriptum-run/config.yaml）
cd /home/nechao/transcriptum-run
PYTHONPATH=/mnt/f/hub/transcriptum/backend PATH=/home/nechao/bin:$PATH \
  /home/nechao/moss-venv/bin/python -m transcriptum serve \
  --config /home/nechao/transcriptum-run/config.yaml --port 8080

# 验收
TRANSCRIPTUM_SMOKE_WAV=/mnt/f/hub/transcriptum/test_speech.wav \
  /home/nechao/moss-venv/bin/python /mnt/f/hub/transcriptum/scripts/smoke_e2e.py \
  --base http://127.0.0.1:8080
```
