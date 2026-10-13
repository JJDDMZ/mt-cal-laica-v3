#!/usr/bin/env bash
# 真机单阶段训练驱动（技术方案 §4.1/§9）
#
# 关键顺序约定（踩过的坑）：
#   1. 先 `cli open` 把项目编译起来 —— 客户端启动时就会去连我们的端口；
#   2. **再**起 Python 桥接 —— 顺序反了的话桥接会先超时退出，客户端连不上；
#   3. 客户端已内置指数退避重连，所以即使时序有偏差也能自愈。
#
# 用法：
#   ./wemi_cal/run_stage.sh Flappybird 50000 runs_real/c-strict-f [额外参数...]
set -euo pipefail
cd "$(dirname "$0")/.."

GAME="${1:?用法: run_stage.sh <Flappybird|Timberman> <steps> <run-root> [额外参数]}"
STEPS="${2:?缺少 steps}"
RUN_ROOT="${3:?缺少 run-root}"
shift 3
EXTRA=("$@")

PY="${WEMI_CAL_PYTHON:-/opt/miniconda3/envs/WemiEnv/bin/python}"
CLI="/Applications/wechatwebdevtools.app/Contents/MacOS/cli"
case "$GAME" in
  Flappybird) PORT=17911 ;;
  Timberman)  PORT=17912 ;;
  *) echo "未知游戏: $GAME" >&2; exit 1 ;;
esac
PROJ="wechat_games_cal/$GAME"

mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/stage_${GAME}.log"

echo "=== 1/3 打开项目 $GAME（先让它编译并开始尝试连接） ==="
"$CLI" close --project "$PWD/$PROJ" >/dev/null 2>&1 || true
"$CLI" open  --project "$PWD/$PROJ" 2>&1 | grep -E "^✔|error" || true
sleep 8   # 留出编译时间；客户端的重连会兜住这段时间

echo "=== 2/3 起桥接并训练（日志: $LOG） ==="
nohup "$PY" -m wemi_cal.run_real --game "$GAME" --port "$PORT" --steps "$STEPS" \
    --run-root "$RUN_ROOT" --no-devtools-open "${EXTRA[@]}" > "$LOG" 2>&1 &
BRIDGE_PID=$!
echo "bridge pid=$BRIDGE_PID"

echo "=== 3/3 等待结果（Ctrl-C 可中断；中断不会丢进度，会落 INTERRUPTED checkpoint） ==="
set +e
wait "$BRIDGE_PID"
CODE=$?
set -e
tail -40 "$LOG"
exit "$CODE"
