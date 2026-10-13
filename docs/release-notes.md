MT-Cal-LAICA-v3 的 seed-0 最终 checkpoint。模型从 Flappybird 50,000 步训练连续恢复到 Timberman 50,000 步，保留模型、优化器、回放和 RNG 状态，checkpoint 格式为 3。

2026 年 10 月 6 日，同一模型实例一次加载完成两个游戏各 50 局 greedy 评测：Flappybird 均分 56.62，Timberman 均分 831.92，均为 50/50 局达标；每局上限 1000 步。

文件 SHA-256：`143d08ec214888b9e99370514ba3ed872b3ea692b39835420df9820f6b4f61ae`。

该文件来自原始成功实验，未重写或合并。公开记录见 `results/seed0/`，复现步骤见 `docs/reproduction.md`。跨 seed 稳定性、第三个游戏与消融仍待验证。
