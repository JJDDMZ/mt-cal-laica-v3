# 复现步骤

本项目先分别验证两个单任务基线，再从新模型执行严格顺序训练。真实实验需手动切换微信客户端；本文命令不会自动操作开发者工具。

## 1. 环境与运行时

历史实验环境为 macOS ARM64、Python 3.10.21、PyTorch 2.14.0、NumPy 1.26.4。微信开发者工具版本为 2.02.2608070，客户端基础库为 3.17.3。其他平台需另行核验依赖和实时延迟。

```bash
git clone https://github.com/JJDDMZ/mt-cal-laica-v3.git
cd mt-cal-laica-v3
conda create -n mt-cal-laica python=3.10.21 -y
conda activate mt-cal-laica
python -m pip install -r requirements.txt
export PYTHONDONTWRITEBYTECODE=1

git clone https://github.com/Ada9io/WemiEnv.git external/WemiEnv
git -C external/WemiEnv checkout 5970399a0e8f1c29ed2da5b453606a323136882d
python scripts/prepare_runtime.py --upstream external/WemiEnv
```

`prepare_runtime.py` 核对固定版本与 ZIP/源码哈希，并生成 `WemiEnv/`、`wechat_games_rt/` 和测试使用的 `wechat_games_cal/`。它拒绝覆盖已有输出目录。项目配置使用游客 AppID 和基础库 3.17.3，游戏逻辑按历史适配准备；配置身份与原机器的 AppID 不同，会写入新的运行记录。

使用微信开发者工具打开生成的项目。如果游客模式不适用于当前工具，可在生成的项目配置中使用自己有权限的 AppID，并在复现记录中注明。不要修改已有实验目录。

## 2. 离线核验与自动化测试

```bash
python scripts/verify_results.py
python scripts/test_prepare_runtime.py
python scripts/test_experiment.py
python -m pytest wemi_cal/tests -q -p no:cacheprovider
node wemi_cal/tests/js/test_envcore.js
WEMI_CAL_PYTHON="$(command -v python)" node wemi_cal/tests/js/test_runtime_bridge.js
```

这些测试使用 fake 环境。历史结果核验复算公开的 100 局记录，不产生新的真实游戏结果。

## 3. 单任务基线

以下每条命令在上一条正常结束后手动执行。先启动 Python，看到服务等待连接后，再打开对应的 `wechat_games_rt/Flappybird` 或 `wechat_games_rt/Timberman`。新进程只编译一次客户端，已连接后保持项目运行。

```bash
python scripts/experiment.py baseline-f --seed 0
python scripts/experiment.py eval-baseline-f --seed 0
```

随后切换到 Timberman：

```bash
python scripts/experiment.py baseline-t --seed 0
python scripts/experiment.py eval-baseline-t --seed 0
```

每个基线从新模型训练 50,000 步，再做 50 局 greedy 评测。两个基线 checkpoint 只用于比较，不参与主实验初始化。检查 F 至少 40/50 局得分不低于 10，T 至少 40/50 局得分不低于 50。

## 4. 严格顺序主实验

切回 Flappybird，先从新模型训练，再检查门槛：

```bash
python scripts/experiment.py strict-f --seed 0
python scripts/experiment.py gate-f --seed 0
```

F 门槛通过后，使用同一个 F checkpoint 恢复训练 Timberman：

```bash
python scripts/experiment.py strict-t --seed 0
```

两个阶段各交互 50,000 步。T 阶段每批固定 F 历史 64 + T 在线 64，初始离线更新数应为 0。结束后得到同目录的 `stage_Timberman.pt`。包装器只使用 v3 正式入口；可加 `--dry-run` 检查对应的完整命令。

输出根目录为 `runs_v3/reproductions/`。包装器对已经存在的新训练目录执行保护检查，不覆盖历史产物。

## 5. 同实例双游戏评测

```bash
python scripts/experiment.py dual --seed 0
```

评测程序只加载一次最终 checkpoint。先打开 F 客户端，等 F 的 50 局全部完成；出现 T 评测块提示后，保留 Python 进程，再切到 T 客户端。等待连接上限为 900 s。

最终两游戏均须至少 40/50 局达到各自绝对门槛，均分须达到同 seed 单任务基线的 80%。同时检查 `optimizer_steps=0`，参数、model state 和 RNG 检查均未变化。

F 的完整 50 局通常约需 85～90 分钟。真实游戏按自身时钟推进，延迟影响复现。长评测时可用独立终端或 screen，集中在预计完成时间和游戏切换点检查日志。

## 6. 直接使用已发布模型

下载历史 seed-0 模型并验证文件：

```bash
mkdir -p models
gh release download v3-seed0 --repo JJDDMZ/mt-cal-laica-v3 \
  --pattern stage_Timberman.pt --dir models
python scripts/verify_results.py --checkpoint models/stage_Timberman.pt
```

也可以在 [Releases 页面](https://github.com/JJDDMZ/mt-cal-laica-v3/releases/tag/v3-seed0)下载。使用该模型真实重评时，另建输出目录，并运行：

```bash
python -u -m wemi_cal.evaluate_rt_dual \
  --checkpoint models/stage_Timberman.pt --steps 50000 --episodes 50 --seed 0 \
  --port 7911 --run-root runs_v3/reproductions/published-model-eval \
  --wemi-env-root WemiEnv \
  --st-flappy-report results/seed0/baseline-Flappybird.json \
  --st-timber-report results/seed0/baseline-Timberman.json
```

游戏切换方式与第 5 步相同。哈希一致证明文件身份一致，重新评测才检验新机器上的实际表现。

## 7. 短演示：每个游戏 2 局

查看游戏画面时，使用同一个最终模型分别演示两款游戏。以下命令各加载一次相同 checkpoint，不更新模型，也不作为 50+50 局同实例验收结果。

先运行 F 命令，等待服务监听 7911，再在微信开发者工具打开 `wechat_games_rt/Flappybird`。新进程编译一次客户端，连接后游戏会自动运行；无需点击或按键。F 的 2 局预计约 3～4 分钟。

```bash
python -u -m wemi_cal.run_rt --game Flappybird \
  --evaluate models/stage_Timberman.pt --episodes 2 \
  --steps 50000 --seed 0 --schedule strict --port 7911 \
  --connect-timeout 900 --wemi-env-root WemiEnv \
  --run-root "runs_v3/demos/flappy-$(date +%Y%m%d-%H%M%S)"
```

F 结束后关闭 F 项目窗口，再运行 T 命令。等待监听提示后打开 `wechat_games_rt/Timberman`，连接方式相同。

```bash
python -u -m wemi_cal.run_rt --game Timberman \
  --evaluate models/stage_Timberman.pt --episodes 2 \
  --steps 50000 --seed 0 --schedule strict --port 7911 \
  --connect-timeout 900 --wemi-env-root WemiEnv \
  --run-root "runs_v3/demos/timber-$(date +%Y%m%d-%H%M%S)"
```

`--evaluate` 指定只做评测；`--steps 50000` 用于匹配原训练配置，不会重新训练 50,000 步。输出目录按时间新建，保留已有演示记录。中途停止使用终端的 Ctrl+C，再关闭游戏窗口。
