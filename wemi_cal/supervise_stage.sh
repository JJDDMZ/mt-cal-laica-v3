#!/usr/bin/env bash
# 真机训练守护脚本：解决"客户端/服务端时序"问题（踩坑记录见下）
#
# 踩过的坑（每一条都真实发生过）：
#   1. 先起桥接再开项目 → 项目编译期间桥接已等待超时退出 → 客户端连不上。
#   2. 客户端本身**会**重连（实测每 10s 重试一次），所以只要桥接一直在，
#      时序偏差能自愈；真正致命的是桥接先死。
#   3. `cli open` 同一个项目不一定触发重新编译 → 必须先 close 再 open。
#
# 因此本脚本：先起桥接并让它**长时间等待**，再开项目；随后全程托管。
#
# 用法：./wemi_cal/supervise_stage.sh <Flappybird|Timberman> <steps> <run-root> [额外参数...]
set -uo pipefail
cd "$(dirname "$0")/.."

GAME="${1:?用法: supervise_stage.sh <游戏> <steps> <run-root> [额外参数]}"
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
PROJ="$PWD/wechat_games_cal/$GAME"
LOG="$RUN_ROOT/stage_${GAME}.log"
mkdir -p "$RUN_ROOT"

echo "=== 清理旧进程 ==="
pkill -f "wemi_cal.run_real" 2>/dev/null || true
sleep 1

echo "=== 1/4 起桥接（等待窗口 30 分钟，避免编译期间超时退出） ==="
nohup "$PY" -m wemi_cal.run_real --game "$GAME" --port "$PORT" --steps "$STEPS" \
    --run-root "$RUN_ROOT" --no-devtools-open --reconnect-timeout 1800 \
    "${EXTRA[@]}" > "$LOG" 2>&1 &
BRIDGE_PID=$!
echo "bridge pid=$BRIDGE_PID  log=$LOG"
sleep 3

echo "=== 2/4 关闭并重新打开项目（强制重新编译） ==="
"$CLI" close --project "$PROJ" >/dev/null 2>&1 || true
sleep 2
"$CLI" open --project "$PROJ" 2>&1 | grep -E "^✔|error" || true

echo "=== 3/4 等待客户端连接（最多 180s；客户端有重连，会自动接上） ==="
for i in $(seq 1 60); do
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "桥接进程已退出，训练结束或失败。"
    break
  fi
  if grep -q "客户端已连接" "$LOG" 2>/dev/null; then
    echo "已连接（第 ${i} 次检查）"
    break
  fi
  sleep 3
done

echo "=== 4/4 等待训练结束 ==="
wait "$BRIDGE_PID"
CODE=$?
echo "--- 训练输出尾部 ---"
tail -30 "$LOG"
exit "$CODE"
