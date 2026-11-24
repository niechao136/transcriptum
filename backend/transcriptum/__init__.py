"""Transcriptum —— 会议与评估记录系统（本地试点）。

分层：
- adapters/  : ASR / Diarizer / LLM 的可替换契约
- config.py  : 配置 + 数据边界校验
- audio.py   : 音频归一化
- align.py   : 时间戳对齐（§7.5）
- proofread.py / assessment.py : 保守校对 / 引用强制
- worker.py  : 队列 + 阶段状态机
- service.py : 业务服务
- exporters.py : 导出
- api.py     : HTTP 接口
"""

__version__ = "0.1.0"
