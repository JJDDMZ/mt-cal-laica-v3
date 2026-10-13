## 概述

这次我以 [WemiEnv](https://github.com/Ada9io/WemiEnv) 为交互平台，使用 [Cal-LAICA](https://github.com/bailei951/Cal-LAICA-Experiments) 的方法，尝试让同一个模型先学 Flappybird，再学 Timberman。

实验结束后，我用同一个 checkpoint[^1]，只加载一次模型，分别评测两个游戏，检查学完新游戏后是否还保留旧游戏的能力。

*我想验证的是“一个模型连续学会两个游戏”，两个单任务模型只作为比较基线。*

## 实现

```text
微信小游戏 → 29 维观测 → 共享 β → 潜动作 e → φ 解码动作
     ↑                                      ↓
     └───────────── 执行动作 ─────────────────┘

交互经验 → 历史与当前回放 → 校准 Q 值、更新共享模型
```

| 部分 | 项目中的做法 |
| --- | --- |
| 观测 | 用结构化数据表示游戏状态，包含游戏标识 |
| 模型 | 共享 β、ψ、φ 和双 Q，按当前游戏屏蔽非法动作 |
| Q 校准 | 用轨迹参考回报约束价值估计 |
| OORB | 训练 T 时，每批取 F 历史 64 条 + T 在线 64 条 |
| 恢复训练 | 连同优化器、回放和随机数状态一起恢复；初始离线训练只做一次 |

论文研究的是迷宫中新增动作，我把这套方法用于两个不同小游戏。固定 64/64 回放和动作概率校正是项目中的适配；初始数据来自随机采集。

## 实验

我先分别训练两个单任务基线，各 50,000 步。主实验再从新模型开始，按以下顺序训练：

```text
F 50,000 步 → F 门槛评测 → 从同一 checkpoint 恢复 T 50,000 步
                                              ↓
                            最终 checkpoint：同实例评测 F50 + T50
```

2026 年 10 月 6 日，seed 0[^2] 的独立 greedy[^3] 评测结果如下，每组均为 50 局：

| 游戏 | 单任务均分 | 最终均分 | 达标局数 |
| --- | ---: | ---: | ---: |
| Flappybird | 56.50 | 56.62 | 50/50 |
| Timberman | 674.36 | 831.92 | 50/50 |

![单任务与最终模型](figures/baseline-vs-final.png)

图：单任务基线与最终模型的比较。来源：本项目评测 JSON；误差线为局间标准差。

训练 T 前，F 的均分为 56.38。本次未观察到 F 能力下降。最终 100 局均达到每局 1000 步上限，评测期间模型参数未变化。

验收要求：F 至少 40/50 局得分不低于 10，T 至少 40/50 局得分不低于 50；两游戏最终均分还须达到各自单任务基线的 80%。

## 复现步骤

仓库：[JJDDMZ/mt-cal-laica-v3](https://github.com/JJDDMZ/mt-cal-laica-v3)。完整命令说明见仓库中的 `docs/reproduction.md`。

### 环境准备

这次的环境是 macOS ARM64、Python 3.10、微信开发者工具 2.02.2608070，客户端基础库为 3.17.3。环境准备命令如下：

```bash
git clone https://github.com/JJDDMZ/mt-cal-laica-v3.git
cd mt-cal-laica-v3
conda create -n mt-cal-laica python=3.10.21 -y
conda activate mt-cal-laica
python -m pip install -r requirements.txt

git clone https://github.com/Ada9io/WemiEnv.git external/WemiEnv
git -C external/WemiEnv checkout 5970399a0e8f1c29ed2da5b453606a323136882d
python scripts/prepare_runtime.py --upstream external/WemiEnv
```

运行时脚本生成所需的平台和游戏目录，拒绝覆盖已有目录。

### 核验已有结果

100 局原始记录已保存在仓库中，统计核验命令如下：

```bash
python scripts/verify_results.py
```

这一步复算已有记录，不运行游戏。最终 checkpoint 的下载和哈希核验命令如下：

```bash
mkdir -p models
gh release download v3-seed0 --repo JJDDMZ/mt-cal-laica-v3 \
  --pattern stage_Timberman.pt --dir models
python scripts/verify_results.py --checkpoint models/stage_Timberman.pt
```

### 看模型玩游戏

看游戏画面时，我用最终模型做短演示，每款游戏 2 局。两次启动都加载同一个 checkpoint，不更新模型；正式验收仍是前面记录的同实例 50+50 局评测。

在仓库目录和已准备的 Conda 环境中，模型路径沿用前面的下载位置：

```bash
DEMO_CHECKPOINT="models/stage_Timberman.pt"
```

F 的演示命令：

```bash
python -u -m wemi_cal.run_rt --game Flappybird \
  --evaluate "$DEMO_CHECKPOINT" --episodes 2 \
  --steps 50000 --seed 0 --schedule strict --port 7911 \
  --connect-timeout 900 --wemi-env-root WemiEnv \
  --run-root "runs_v3/demos/F-$(date +%Y%m%d-%H%M%S)"
```

连接顺序：Python 提示“等待客户端”后，开发者工具打开 `wechat_games_rt/Flappybird`，编译一次。已经自动编译并连接时，不再重复编译。模型会自动控制小鸟，2 局约需 3～4 分钟。

F 结束后关闭项目窗口，再运行 T 命令：

```bash
python -u -m wemi_cal.run_rt --game Timberman \
  --evaluate "$DEMO_CHECKPOINT" --episodes 2 \
  --steps 50000 --seed 0 --schedule strict --port 7911 \
  --connect-timeout 900 --wemi-env-root WemiEnv \
  --run-root "runs_v3/demos/T-$(date +%Y%m%d-%H%M%S)"
```

等待客户端时打开 `wechat_games_rt/Timberman`，连接后模型会自动选择左右砍树。两个游戏都无需手动控制。

`--evaluate` 表示只评测；`--steps 50000` 匹配原训练配置，不会重新训练。演示结果按时间写入新目录，中途停止用 Ctrl+C。

### 重新训练

实验按阶段手动运行，每阶段结束后再开始下一阶段。连接顺序是先启动 Python，等待 7911 服务，再打开对应的 `wechat_games_rt/` 游戏；每个新进程只编译一次客户端。

单任务基线的训练和评测命令：

```bash
# 打开 Flappybird，训练后为新的评测进程重新连接一次
python scripts/experiment.py baseline-f --seed 0
python scripts/experiment.py eval-baseline-f --seed 0

# 切换到 Timberman
python scripts/experiment.py baseline-t --seed 0
python scripts/experiment.py eval-baseline-t --seed 0
```

主实验命令：

```bash
# 切回 Flappybird
python scripts/experiment.py strict-f --seed 0
python scripts/experiment.py gate-f --seed 0

# F 门槛通过后，切换到 Timberman
python scripts/experiment.py strict-t --seed 0
```

复现输出目录为 `runs_v3/reproductions/`。`--dry-run` 只打印命令，不启动实验。

### 最终评测

```bash
python scripts/experiment.py dual --seed 0
```

最终评测先连接 F 客户端，完成 50 局后再切换到 T。切换期间保留同一个 Python 进程，保证始终使用同一个模型。F 的完整 50 局约需 85～90 分钟，切换时的连接等待上限为 900 s。

已发布模型的直接评测命令记录在 `docs/reproduction.md`，重评结果写入新的输出目录。

## 后续工作

这次完成了 seed-0 的双游戏验证。接下来我想验证更多 seed，再通过消融比较校准与回放的作用，之后尝试第三个游戏。目前还不能说明跨 seed 的稳定性，也不能说明任意新游戏的泛化能力。

[^1]: Checkpoint 保存模型及训练状态，用于恢复训练或评测。这里使用 format 3，包含优化器、回放和随机数状态。

[^2]: Seed 是控制训练随机初始化和采样的编号。固定 seed 便于比较实验；跨 seed 验证需要重新训练多个模型。

[^3]: Greedy 评测使用确定性动作选择，区别于训练时带探索的行为。
