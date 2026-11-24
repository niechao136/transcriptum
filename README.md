# Transcriptum · 会议与评估记录系统

按《会议与评估记录系统-设计文档.md》实现的本地试点版本。

**核心约束**：音频与文本都不出内网。ASR / Diarizer / LLM 全部通过可替换适配器接入本机或内网服务。

---

## 🐳 Docker 部署（推荐）

单镜像同时提供 API、Worker 和前端页面（同源，免 CORS），以非 root 运行，内置 ffmpeg：

```bash
docker build -t transcriptum:0.1.0 .
cp .env.example .env            # 按需改模型端点与白名单
docker compose up -d
# 打开 http://127.0.0.1:8080  ，健康检查 /api/health
```

- 数据落在命名卷 `transcriptum-data`（SQLite + 音频）；
- 默认 mock 适配器，无需模型即可跑通全流程；
- 接真实模型、API/Worker 拆分、运维命令见 **[Docker 部署指南](docs/Docker部署指南.md)**。

---

## 本地开发（不用 Docker）

```bash
# 1. 后端依赖
pip install fastapi "uvicorn[standard]" pydantic pytest httpx python-multipart python-docx openpyxl

# 2. 初始化并启动（worker 与 API 同进程）
cd backend
python -m transcriptum --config ../config.yaml --init-db
python -m transcriptum --config ../config.yaml --port 8080

# 3. 前端
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173，已代理 /api 到 8080
```

默认配置用 **mock 适配器**，无需模型即可跑通全流程，用于验证上层逻辑。

### 一条命令验收

```bash
# 后端起着的情况下
python scripts/smoke_e2e.py --base http://127.0.0.1:8080
```

输出 33 项检查，覆盖：健康检查 → 上传 → 转写 → 逐句校对 → 说话人确认 →
评估解析 → 引用强制 → 导出 → 审计。

---

## 接真实模型

### ASR（MOSS-Transcribe-Diarize）

在 `config.yaml` 里改：

```yaml
asr:
  adapter: moss
  base_url: http://127.0.0.1:9000
  model: MOSS-Transcribe-Diarize
  custom_http:
    submit_path: /v1/jobs
    poll_path: /v1/jobs/{job_id}
    fetch_path: /v1/jobs/{job_id}/result
    capabilities:
      has_word_ts: false      # 按实际服务能力填，不要虚报
      has_confidence: false
      has_speaker: true
```

**能力声明必须诚实**。`has_word_ts: false` 时 UI 会明确提示「跳播精确到句，diff 为字符级」，
不会假装能定位到字。

如果部署的 ASR 接口形状不同（同步返回、字段名不同），只需新增一个适配器类并注册：

```python
# transcriptum/adapters/registry.py
ASR_ADAPTERS["my_asr"] = MyAsrAdapter
```

### LLM

```yaml
llm:
  adapter: openai_compatible
  base_url: http://127.0.0.1:8000/v1
  model: Qwen2.5-7B-Instruct
  json_mode: true
```

`json_mode: false` 时会退回「提示词约束 + 严格校验 + 重试」，不模糊解析。

---

## 架构

```
backend/transcriptum/
├── adapters/        ASR / Diarizer / LLM 契约与实现（编译期注册，无动态加载）
│   ├── base.py      三个 Protocol + Capabilities
│   ├── moss_parser.py  MOSS 全文 → 句子时间轴
│   ├── http_asr.py     三段式作业 + OpenAI audio/transcriptions
│   ├── http_llm.py     OpenAI compatible chat
│   ├── mock.py         无模型跑通全链路
│   └── registry.py     适配器注册表
├── align.py         §7.5 时间重叠说话人对齐（核心难点）
├── proofread.py     逐句保守校对：受保护 token、拒改写、改动幅度上限
├── assessment.py    框架定义、引用强制、类型校验
├── worker.py         SQLite 队列表 + 单进程轮询 + 阶段状态机
├── service.py        业务服务（录音/句子/评估/保留期）
├── api.py            HTTP 接口 + 音频 Range 播放
├── exporters.py      docx / xlsx / md 导出（含脱敏与「AI 未确认」标注）
└── schema.sql        SQLite schema（asr_text 不可变由触发器保证）

frontend/src/
├── ProofreadDesk.tsx  校对台：逐句跳播、键盘全流程、diff 高亮
├── AssessmentView.tsx 评估页：引用可见可跳转、缺引用不可提交
├── RecordingList.tsx  上传与导出
└── diff.ts            字符级 diff（基于结构化 ai_edits）
```

---

## 设计文档要点的落地方式

| 文档要求 | 实现 |
| --- | --- |
| §3.2 `asr_text` 不可变 | SQLite 触发器 `trg_segment_asr_immutable`；API 层额外显式拒绝（422），不静默忽略 |
| §7 适配器可替换、能力显式声明 | `Capabilities` 四元组，UI 与 pipeline 都读它降级 |
| §7.5 重叠对齐三种失败情形 | 歧义→标待确认；句内切换→按时间分界切分；无重叠→最近块+待确认 |
| §4.3 保守校对不许编 | 受保护 token 不得改；模型改写整句不采纳；净长度变化 >30% 拒绝；失败保留原文 |
| §6 引用强制 | 无 citations 的字段 `missing_citation`，`can_submit` 阻塞，接口返回 422 |
| §10 审计与脱敏 | `audit_log` 记录改前改后；导出默认脱敏手机号/身份证/邮箱；音频保留期可清理 |
| §9 Range 播放 | `_range_response` 支持 206 与 Content-Range，浏览器可 seek |

---

## 验证状态

```
pytest backend/tests        → 82 passed
python scripts/smoke_e2e.py → 33 passed, 0 failed
npx tsc -b                  → 0 error
npx vite build              → build ok
```

### 尚未验证的部分（重要）

**MOSS 在 RTX 2060 Super 上的实际性能与输出形态没有实测。**
当前开发机无 NVIDIA GPU（仅 AMD Radeon 625 + Intel UHD），MOSS+vLLM 无法在此启动。

因此：
- MOSS 输出解析器按官方 README 的 `[时间][说话人]` 格式实现，并有单元测试；
- 但真实模型的输出是否严格符合该格式、是否有额外的换行/标点差异，**需要在目标机上跑一次确认**；
- `config.yaml` 里 `custom_http.*_path` 三处路径需要按实际的 vLLM/SGLang 服务调整；
- 建议先在那台机器上跑 `scripts/probe_moss.py`（见下）确认接口形状，再切 `adapter: moss`。
