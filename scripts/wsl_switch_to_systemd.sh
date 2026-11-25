#!/bin/bash
# 把手动 nohup 启动的 MOSS / Transcriptum 交给 systemd 用户服务管理。
# 单独脚本运行，避免命令行动态匹配把自身 shell 杀掉。
set -u

echo '===== [1] 停止手动进程 ====='
for pat in '/home/nechao/moss-venv/bin/vllm serve' '/home/nechao/moss-venv/bin/python -m transcriptum serve'; do
  for pid in $(pgrep -f "$pat" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    # 排除本脚本自身及其父 shell
    [ "$pid" = "$PPID" ] && continue
    echo "kill $pid ($pat)"
    kill "$pid" 2>/dev/null || true
  done
done
sleep 4

echo '===== [2] 剩余相关进程 ====='
ps -eo pid,cmd | grep -E 'bin/vllm serve|m transcriptum serve' | grep -v grep | grep -v wsl_switch || echo 'none'

echo '===== [3] enable 单元 ====='
systemctl --user enable moss-serve.service transcriptum.service

echo '===== [4] 启动 moss-serve ====='
systemctl --user start moss-serve.service
sleep 2
systemctl --user is-active moss-serve.service
