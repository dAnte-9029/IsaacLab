# 阶段 1 性能结果发布范围

这里发布跨配置 JSON/CSV 汇总、校验结果、硬件/实际配置、调用树计时和原 runner 的运行摘要。完整报告见 [中文报告](../../unified_rl_performance_phase1_zh.md)。

原始产物保留在测量机器的同一路径 `/home/zn/IsaacLab/.worktrees/unified-rl-performance-phase1/docs/results/unified_rl_performance_phase1/`，未纳入 Git：

- `*/run/evidence/`：SQLite 正文及相关文件。
- `*/resources.jsonl`、`*/console.log`、`*/run/*.jsonl`：逐秒资源和原始运行日志。
- `*/run/assets/`、`*/run/source_snapshot/`：生成资产及完整运行源码快照。
- 其他运行期缓存和辅助输出。

这些文件没有被删除或修改；性能 checkpoint 已在 benchmark 完成时按原计划删除。硬件/计时摘要可直接读取；重新运行汇总脚本需要本地完整 `resources.jsonl`。`benchmark_source.py` 保存对应运行的插桩版本，正式可复用入口在 `scripts/flapping_px4/`。

测量时基底为 `5406ab6c527364ef985e9fc2af83472d590d6709`，运行摘要中的 dirty 状态与路径按原样保留；其中已删除 checkpoint 的路径是历史记录，不代表可恢复权重。两个 excluded 目录仅作诊断说明，不参与性能比较。
