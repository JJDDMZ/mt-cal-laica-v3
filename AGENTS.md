# 项目约定

正式方案为 MT-Cal-LAICA-v3。算法源码与实验工具位于 `wemi_cal/` 和 `scripts/`。中文说明沿用短段落、具体术语、数据来源和必要复现命令；可用时使用 chinese-technical-writing skill。

正式入口为 `python -m wemi_cal.run_rt` 和 `python -m wemi_cal.evaluate_rt_dual`。使用 realtime 游戏、7911 端口、format-3 checkpoint。先验证单任务基线，再进行 Flappybird 50k → 同 checkpoint 恢复 Timberman 50k。T 阶段每批 F 历史 64 + T 在线 64；initial offline 只执行一次。最终同实例、一次加载、两游戏各 50 局 greedy 评测。

已有证据在 `results/seed0/`。它们是历史实验结果，不能把本地测试通过当成新的真实游戏实验。新实验只写入忽略的 `runs_v3/reproductions/`，启动长实验须有用户明确指示。

上游源码和游戏由 `scripts/prepare_runtime.py` 生成到被 Git 忽略的目录。不修改其他已有 WemiEnv 仓库，不覆盖旧实验。`wechat_games_cal/` 只用于 fake 协议测试，能力实验只用 `wechat_games_rt/`。不执行旧 `--evaluate-multi`、PPO 或 lockstep 能力实验路线。
