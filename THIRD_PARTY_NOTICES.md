# 来源与许可

本仓库的算法实现、复现工具和项目说明使用 MIT License。算法依据 Cal-LAICA 论文及其动作表示、价值校准和回放思想实现，并针对 WemiEnv 的实时小游戏进行了适配。

| 来源 | 用途 | 获取方式 |
| --- | --- | --- |
| [Ada9io/WemiEnv](https://github.com/Ada9io/WemiEnv) | Python 平台、小游戏源代码与图片资源 | 使用者下载固定提交，复现脚本在本地准备运行时 |
| [Cal-LAICA-Experiments](https://github.com/bailei951/Cal-LAICA-Experiments) | 算法语义参考 | 链接上游，本仓库提供自己的适配实现 |
| PyTorch、NumPy、Gym、Gymnasium、PyYAML、websocket-server | 运行依赖 | 通过包管理器安装，许可归各上游项目 |

截至本次发布，WemiEnv 上游固定提交中没有 LICENSE 文件。本仓库不重新分发其完整源码、小游戏资源或论文 PDF。`prepare_runtime.py` 从使用者已获取的上游文件生成本地运行时，保留来源与必要适配；生成目录由 Git 忽略，其内容不适用本仓库的 MIT 许可声明。

公开结果来自本项目 seed-0 实验。路径字段已转换为相对路径，分数和验收数据保留。转换说明与原始文件哈希见 [source-records.json](results/seed0/source-records.json)。完整原始日志和本地归档保留在实验机器上。
