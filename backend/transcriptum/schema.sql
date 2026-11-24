-- 会议记录 / 评估记录系统 —— SQLite schema (试点期)
-- 设计依据：设计文档 §3 核心数据模型（句子时间轴）、§6 评估、§10 合规

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 3.1 recording — 录音
CREATE TABLE IF NOT EXISTS recording (
    id              TEXT PRIMARY KEY,             -- uuid
    title           TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('meeting', 'assessment')),
    source          TEXT NOT NULL CHECK (source IN ('upload', 'live')),
    audio_path      TEXT,                         -- 归一化后 16k mono WAV
    original_path   TEXT,                         -- 原始上传文件
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'uploaded'
                    CHECK (status IN ('uploaded','transcribing','transcribed',
                                      'proofreading','proofread','parsing','done','failed')),
    error_msg       TEXT,
    created_by      TEXT NOT NULL DEFAULT 'local',
    created_at      TEXT NOT NULL,
    -- 能力快照：ASR/Diarizer/LLM 的能力决定 UI 降级（§7.1）
    asr_adapter     TEXT,
    asr_capabilities TEXT,                        -- json
    diarizer_adapter TEXT,
    llm_adapter     TEXT,
    -- 告警（重复/幻觉检测命中则非空，§4.1）
    warnings        TEXT,
    consent_at      TEXT                          -- 录音前同意提示的回执（§10）
);

CREATE INDEX IF NOT EXISTS idx_recording_status ON recording(status);

-- 3.3 speaker_cluster — 录音内声纹簇
CREATE TABLE IF NOT EXISTS speaker_cluster (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id    TEXT NOT NULL REFERENCES recording(id) ON DELETE CASCADE,
    label           TEXT NOT NULL,                -- A/B/C 或 S01
    person_id       INTEGER REFERENCES person(id) ON DELETE SET NULL,
    avg_embedding   TEXT,                         -- 声纹向量路径（v2）
    needs_review    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (recording_id, label)
);

CREATE INDEX IF NOT EXISTS idx_cluster_recording ON speaker_cluster(recording_id);

-- 3.3 person — 跨录音人员
CREATE TABLE IF NOT EXISTS person (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    role            TEXT,
    voiceprint_path TEXT,
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL
);

-- 3.2 segment — 句子时间轴 ★ 核心表
CREATE TABLE IF NOT EXISTS segment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id    TEXT NOT NULL REFERENCES recording(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,
    speaker_id      INTEGER REFERENCES speaker_cluster(id) ON DELETE SET NULL,
    asr_text        TEXT NOT NULL,                -- 写入后不可变（触发器保护）
    asr_confidence  REAL,
    words           TEXT,                         -- jsonb 预留
    ai_text         TEXT,
    ai_edits        TEXT,                         -- json [{type,from,to,reason,confidence}]
    final_text      TEXT,
    review_status   TEXT NOT NULL DEFAULT 'pending'
                    CHECK (review_status IN ('pending','accepted','rejected','edited')),
    needs_review_speaker INTEGER NOT NULL DEFAULT 0,
    low_confidence  INTEGER NOT NULL DEFAULT 0,
    reviewed_by     TEXT,
    reviewed_at     TEXT,
    UNIQUE (recording_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_segment_recording_seq ON segment(recording_id, seq);

-- asr_text 不可变：任何 UPDATE 尝试改它都会被拒绝（§3.2 可复原性保证）
CREATE TRIGGER IF NOT EXISTS trg_segment_asr_immutable
BEFORE UPDATE OF asr_text ON segment
FOR EACH ROW
WHEN OLD.asr_text IS NOT NEW.asr_text
BEGIN
    SELECT RAISE(ABORT, 'asr_text is immutable');
END;

-- 3.4 framework / assessment
CREATE TABLE IF NOT EXISTS framework (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT NOT NULL,
    name            TEXT NOT NULL,
    builtin         INTEGER NOT NULL DEFAULT 0,
    fields          TEXT NOT NULL,                -- json
    version         INTEGER NOT NULL DEFAULT 1,
    created_by      TEXT NOT NULL DEFAULT 'local',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE (code, version)
);

CREATE TABLE IF NOT EXISTS assessment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id    TEXT NOT NULL REFERENCES recording(id) ON DELETE CASCADE,
    framework_id    INTEGER NOT NULL REFERENCES framework(id),
    -- 框架版本化：历史评估不因框架修改而失效（§6）
    framework_version INTEGER NOT NULL,
    framework_snapshot TEXT NOT NULL,             -- 解析时的字段定义快照
    status          TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft','submitted')),
    created_by      TEXT NOT NULL DEFAULT 'local',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assessment_recording ON assessment(recording_id);

CREATE TABLE IF NOT EXISTS assessment_field_value (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id   INTEGER NOT NULL REFERENCES assessment(id) ON DELETE CASCADE,
    field_key       TEXT NOT NULL,
    content         TEXT,
    citations       TEXT NOT NULL DEFAULT '[]',   -- json [segment_id, ...]
    confidence      REAL,
    source          TEXT NOT NULL DEFAULT 'ai' CHECK (source IN ('ai','human_edited')),
    status          TEXT NOT NULL DEFAULT 'ok'
                    CHECK (status IN ('ok','not_extracted','missing_citation')),
    UNIQUE (assessment_id, field_key)
);

-- 术语表（§4.2 热词 / 正确写法）
CREATE TABLE IF NOT EXISTS glossary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    term            TEXT NOT NULL,                -- 正确写法
    variants        TEXT NOT NULL DEFAULT '[]',   -- json 常见错误写法
    scope           TEXT,                         -- 科室等分组（§12.3）
    enabled         INTEGER NOT NULL DEFAULT 1,
    UNIQUE (term, scope)
);

-- 队列：SQLite job 表 + 单进程轮询（§4.3）
CREATE TABLE IF NOT EXISTS job (
    id              TEXT PRIMARY KEY,
    recording_id    TEXT NOT NULL REFERENCES recording(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('transcribe','proofread','assessment')),
    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued','running','done','failed','retry')),
    attempts        INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL DEFAULT 3,
    next_run_at     TEXT,
    error_msg       TEXT,
    -- 审计：model_name / provider / prompt_version（§7.1 安全边界）
    model_name      TEXT,
    provider        TEXT,
    prompt_version  TEXT,
    trace_id        TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_job_status ON job(status, created_at);

-- 审计日志（§10：谁在何时看了/改了哪句）
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    entity          TEXT NOT NULL,
    entity_id       TEXT,
    recording_id    TEXT,
    before          TEXT,
    after           TEXT,
    at              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_recording ON audit_log(recording_id, at);
