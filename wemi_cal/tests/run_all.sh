#!/usr/bin/env bash
# T0–T8 全量验收：Python 测试 + JS lockstep 测试 + 无模拟器端到端自检
set -euo pipefail
cd "$(dirname "$0")/../.."

# 固定用 WemiEnv 环境，避免误用 base 解释器
PY="${WEMI_CAL_PYTHON:-/opt/miniconda3/envs/WemiEnv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "找不到 WemiEnv 解释器：$PY（可用 WEMI_CAL_PYTHON 覆盖）" >&2
  exit 1
fi

echo "=== 1/3 Python 测试（协议/特征/网络/replay/learner/状态机/checkpoint/评测/端到端）==="
"$PY" -m pytest wemi_cal/tests -q

echo "=== 2/3 JS lockstep 环境核心 ==="
node wemi_cal/tests/js/test_envcore.js

echo "=== 3/3 无模拟器端到端自检（不是能力实验）==="
"$PY" -m wemi_cal.train --smoke --smoke-steps 600

echo
echo "全部通过。注意：这不包含微信模拟器联调，也不构成能力实验结论。"
