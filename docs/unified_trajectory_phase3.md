# Unified Trajectory 第三阶段：训练前修复

主工作树 `.worktrees/unified-trajectory-online-v1`，起点 `8ff48866`，开始时干净。
独立 L/I 接线树 `.worktrees/unified-online-li-smoke` 基于物理分支 `4303c1ad`；来源树的
未跟踪论文未触碰。没有 commit/push、PPO、气动/网络/奖励或轨迹数学定义修改。
原始 88 条 v2 离线记录仍只读。旧机制结果 `results/unified_trajectory_phase3/mechanism.json`
保留为历史，不能代表下述实际抽样结果。

## 实现与版本归属

`trajectory_feedback.py` 改为 SQLite WAL/FULL 持久化。episode 元数据、逐步控制轨迹、
终止摘要、唯一条件成功见证分别存于 `episodes/steps/terminals/witnesses` 表；应用仅 INSERT，
不覆盖或删除旧记录。摘要通过 `trace_ref` 定位完整动作证据，不复制轨迹到反馈或能力窗口。
每步数据流为**命令 → decimation 个物理步 → 实际执行器、关节和机体状态**。
原始非有限数据保留为明确的 `nan/inf` 标记，不替换为零；任何必要字段缺失或非有限均
产生 `invalid_data`，不能认证成功。`completed/failed/timeout/manual_reset/invalid_data`
语义分开；人工 reset 也有摘要，但不进入成功率统计。

每个 episode 的每个动作区间保存实际 `policy_version` 和 `execution_mode`。
runner 在提交新版本动作前调用：

```python
env.set_policy_version("ppo_update_0042", evaluation_mode="training")
# 独立冻结 checkpoint 评估时显式使用 evaluation_mode="frozen"。
# 未知版本传 None；不能用 episode 编号或步数猜版本。
```

版本在 `_pre_physics_step` 捕获，reset 标签仅作背景。跨版本 episode 不截断；
已知多版本标为 `mixed`，含未标注动作标为 `unknown`。成功仍可形成**该完整控制执行**
的条件见证，但不会归给任何冻结 checkpoint。即使宣称 frozen 的 episode 实际跨版本，
其 `frozen_checkpoint` 仍为空。完整动作版本顺序在步骤表中可查。

`PolicyCapability` 将训练与冻结评估分开：训练窗口默认最多 2,048 条终止摘要；
最多保存最近 2 个冻结版本、每版 2,048 条；查询邻域最多取最近 32 条。
这些窗口只保存几何向量、结果和合同作用域，不保存控制轨迹。
训练中的混合/未知执行可影响探索统计，不能影响 checkpoint 成功率和遗忘。
遗忘只比较独立冻结评估；单纯更新 runner 版本、或同一 checkpoint 内采样成功率波动，
都不会制造遗忘信号。区域历史冻结表现基线单独保留，不依赖训练版本标签。

`OnlineSampler` 使用默认 64 个连续轨迹区域代表和有界能力窗口，在每 8 条有效终止反馈后
增量更新一次采样状态。新权重以 0.20 凸混合进入旧权重；reset/sample 不读历史摘要，
不重建能力对象。区域是连续几何邻域，不是 C1–C4 任务 ID。
首次构造/重启时**仅流式读一次终止摘要**恢复状态，不读取逐步轨迹；启动恢复仍是 O(N)，
本轮未实现启动快照。常态采样/反馈不再随全部历史数量线性增长。

历史成功永久保留在磁盘，窗口淘汰不会删除见证。保持分支的 20% 使用磁盘索引均匀抽取
历史唯一条件见证，其余使用有界区域 anchor 和遗忘权重；因此未留在内存 bank 的成功
仍有复习机会。SQLite 索引避免扫描全部成功记录。实例 ID、步骤主键和唯一见证身份
阻止重启重放制造重复认证；重启恢复的采样权重也通过一致性测试。

模拟与真实证据分别写 `simulation_feedback.sqlite3` 和 `rollout_feedback.sqlite3`。
模拟查询明确返回 NOT PHYSICAL EVIDENCE。数据库是单写者接口，步骤与摘要逻辑追加，
不是承诺 SQLite 文件字节只追加。若目录有旧版 JSONL，入口明确拒绝并保留原文件，
要求显式迁移/重新评价；旧版被净化过的执行器轨迹不能静默升级为新版成功证据。

## 实际采样机制

三分支仍为 base / retention / exploration = **40% / 25% / 35%**，base 配置下限 20%。
base 原样调用 q0；retention 对完整成功系数施加连续幅值扰动。
exploration 使用 65% 区域连续扰动、35% 新 q0 候选加权抽样，使失败降权、学习进展和
冻结评估的遗忘能够落实到实际抽样，而不只改变探针分数。所有结果仍经过原几何筛选；
不合法 kernel 或无可用保持 anchor 时明确回退 base，来源记录包含 fallback。
权重更新平滑约束的是权重向量，不把区域代表变化解释为全连续分布的严格 TV 定理。

真实 88 条离线记录提供同 backend 几何优先参考，不给 60 m 新任务授予物理标签；
不同 backend、扭转或合同不能继承认证。距离是对称的，不假设各特征越小越容易。
本次机制实验半径 0.3，其余使用 OnlineConfig 初值；这些都是工程假设，不是最优值。

调用真实 `sample_batch()` → `sample()`，四阶段使用同一组 seed **10000–11023**，
每阶段 **1,024 条轨迹**，全部反馈为构造数据。类别可重叠，数量不是类别概率之和。
困难区域预先选为初始 bank 中反向/耦合需求最高的 8 个代表的最近邻区域。
简单阈值为 K≤0.00125、G≤0.0075；持续为最大有效段≥30 m；复杂反向至少一个通道
有显著反号；三维耦合有效段≥6 m。未知表示距已反馈轨迹几何距离>0.3，
不是“没有物理成功证明”的同义词。

| 实际轨迹数 / 1024 | 初始 | 困难失败后 | 重新成功后 | 冻结 B 退化后 |
|---|---:|---:|---:|---:|
| 简单 | 100 | 102 | 118 | 88 |
| 持续机动 | 738 | 727 | 759 | 788 |
| 复杂反向 | 216 | 216 | 388 | 427 |
| 三维耦合 | 485 | 481 | 608 | 640 |
| 预定义困难区域 | 104 | 100 | 310 | 368 |
| 能力未知区域 | 1024 | 984 | 642 | 624 |

困难区域实际频率为 **10.16% → 9.77% → 30.27% → 35.94%**。
失败降权的实际效应较小，仅减少 4/1024；这里只验证该固定种子机制方向，不作统计优势或
RL 学习性能结论。复习/遗忘响应则在实际样本中出现明显覆盖增加。

各阶段 base 固定抽到 419 次、exploration kernel 233 次、候选探索 113 次；
retention 的 259 次分支抽签从初始全部回退，转为成功后 258 次保持+1 次回退，再到
退化后 259 次保持。额外定向测试强制不兼容航程，覆盖探索/保持两种几何回退。
基础 40% q0 始终不变；默认合法起点下精确直线的理论基础机会至少 2.5%。
未知区域没有永久排除，简单任务在退化后仍有 88 条。

**三种量严格分开**：`distribution()` 是固定探针诊断；上表是实际生成任务的经验频率；
`OfflineEvidence/FeedbackStore.query()` 才是按完整条件/合同查询控制成功见证。
上表全部是模拟机制结果，不是物理可飞概率或飞行成功。
特征歧义仍存在：零坡度时全部曲率反号，特征向量可相同而左右机动顺序相反；
邻域只服务采样，完整轨迹身份不合并。

## 规模基准

CPU、无 PhysX/PPO；每个规模单次采样 20 个种子，256 条批量重复 3 次取中位数；
反馈用 32 条单步模拟终止，包含轨迹写入、FULL 提交及必要的统计更新。
每个规模的反馈计时会追加 32 条，因此最终累计数分别为 32/288/4128。

| 计时前累计反馈 | 单次采样中位 ms | 单次 p95 ms | 256 条批量中位 ms | 反馈中位 ms | 反馈 p95 ms | 训练窗口 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.38 | 38.71 | 1198.58 | 6.45 | 16.41 | 32 |
| 256 | 1.41 | 22.71 | 1155.82 | 6.06 | 24.22 | 288 |
| 4096 | 1.43 | 23.52 | 1391.43 | 5.75 | 41.70 | 2048 |

反馈 p95 包含每 8 条一次的有界窗口更新，饱和前仍随窗口填满而增加；不是恒定零成本。
最后永久唯一成功见证 1,378 条，远多于 bank 的 64 个代表，仍可从磁盘抽取。
256 env 同时 reset 约需 1.2–1.4 s，不能把单次中位数视为全环境吞吐。
历史重放主瓶颈已消除，但几何构建、同步持久化、启动恢复和长期磁盘容量仍有成本。

## 真实 L/I 接线

独立树只移植统一轨迹任务/原有采样器、缓存、在线模块和 smoke 脚本。
L/I 原文件仅缺少第一阶段采样/筛选接口；补齐既有实现后无实质冲突，未更改其已有
Trajectory 数学函数、气动物理或整个 RL 分支。复用来源树已编译 native constraint。

L3D、I3D 各一次：**4 env，在线 32 步 + 同环境关闭在线接口 32 步**，480 Hz physics /
60 Hz control，扭转 35°、10 sections、无风，固定真实动作。验证实际 `_control_aero_model`
和 backend 配置、预瞄、有限机体/关节/执行器状态、部分 env reset 隔离、版本 A→B、
真实 timeout/manual reset 记录、动作轨迹与重启读取。未注入成功状态或完成整条任务。
每个 backend 9 条终止记录：4 timeout、5 manual_reset；4 mixed、5 single，全部无
frozen_checkpoint，也无新增 completed 认证。离线条件不匹配时没有继承成功。

| backend | 关闭在线步中位 ms | 在线步中位 ms | 增量 | 部分 reset ms |
|---|---:|---:|---:|---:|
| L3D | 35.72 | 42.35 | +6.63 ms / 18.6% | 12.81 |
| I3D | 40.44 | 46.87 | +6.42 ms / 15.9% | 12.84 |

时间排除前 3 步；在线/关闭顺序运行，任务 reset 序列不同，不是严格配对性能实验。
这是短程接线开销，不能外推 256 env 或长期 PPO。Kit 有已有文件监视配额警告，但两次
均生成 `ONLINE_LI_SMOKE_PASSED` 并正常结束，未降级为 DeLaurier/L3D 替代 I3D。

## 验证、复现与停止边界

**30 项定向检查通过**：版本混合/未知、冻结统计隔离、同 checkpoint 波动、六种必要
控制字段非有限、追加/重启/重复、部分 reset、旧 JSONL 拒绝静默升级、真实 sample 分支/
回退，以及原数学和缓存测试。实际抽样实验与 L/I smoke 的结果独立于这些单元测试。

紧凑结果位于 [pretraining_fix](results/unified_trajectory_phase3/pretraining_fix/)。
完整 synthetic SQLite 位于主树 `logs/unified_phase3_fix_validation_v1/`；真实步骤证据位于
L/I 接线树 `logs/unified_online_smoke_{L3D,I3D}_v1/evidence/`，没有混写。

主树中激活环境后：

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
./isaaclab.sh -p -m pytest tests/test_online_trajectory.py tests/test_unified_trajectory.py tests/test_unified_trajectory_cache.py -q
./isaaclab.sh -p scripts/flapping_px4/validate_online_trajectory.py \
  --rules ../control-oriented-3d-wing-aero/logs/unified_trajectory_phase2_v1/screening_rules_v2.json \
  --output logs/unified_phase3_fix_repro
```

在已接入模块的 `.worktrees/unified-online-li-smoke` 中，同样激活环境并重设 PYTHONPATH，
分别运行（输出目录必须不存在）：

```bash
./isaaclab.sh -p scripts/flapping_px4/smoke_online_trajectory_li.py --backend L3D \
  --plant-assets ../control-oriented-3d-wing-aero \
  --rules ../control-oriented-3d-wing-aero/logs/unified_trajectory_phase2_v1/screening_rules_v2.json \
  --output logs/unified_online_L3D_repro
# 将 backend 与输出目录改为 I3D，复现第二次 smoke。
```

结论：常态在线路径已消除明显的全历史重放；冻结策略遗忘不受混合版本训练 episode
污染；实际抽样体现探索与保持。**已具备等预算固定/在线自适应 PPO 对照的机制与真实
L/I 接口条件**，不等于已经验证 RL 收益或长期训练吞吐。runner 仍须显式提交逐动作版本，
并安排独立冻结评估；比较时固定 plant/reset/合同/预算，定义清楚评估成本如何计入。
本轮未实现新的 PPO runner，也未启动训练，验证完成后停止。
