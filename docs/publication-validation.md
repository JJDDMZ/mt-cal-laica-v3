# 发布验证记录

2026 年 10 月 13 日，在独立发布副本中验证。没有启动真实训练或重新评测；历史研究结果仍来自 2026 年 10 月 6 日的正式实验。

| 检查 | 结果 |
| --- | --- |
| `python -m pytest wemi_cal/tests -q -p no:cacheprovider` | 334 项通过，35 项 warning |
| `node wemi_cal/tests/js/test_envcore.js` | 46 项检查，0 失败 |
| `node wemi_cal/tests/js/test_runtime_bridge.js` | 两游戏 fake 客户端协议检查通过 |
| `python scripts/test_prepare_runtime.py` | 6 项通过 |
| `python scripts/test_experiment.py` | 13 项通过 |
| `python scripts/verify_results.py --checkpoint …` | 完整 100 局统计、评测不变性记录和原始 checkpoint 哈希一致 |

验证环境：macOS ARM64、Python 3.10.21、PyTorch 2.14.0、Node.js 24.16.0。新增工具使用标准库；协议测试使用 fake 客户端。

运行时由固定上游提交准备，已有文件不会被覆盖。上游下载内容、生成运行时、个人路径和大体积训练产物不进入源码仓库。算法源码沿用成功实验的实现；新增发布工具只负责准备文件、包装命令和核验结果。

这些检查确认了发布材料的完整性和自动化行为。使用不同客户端或硬件重新获得相同分数，仍需按复现指南进行真实实验。
