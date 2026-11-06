# Unified Trajectory RL 性能优化：阶段 2

2026-11-07。独立工作树 `.worktrees/unified-rl-performance-phase2`，分支 `perf/unified-rl-performance-phase2`，基于 `84232239e6429b0d8abebeb23aa969853c15dd9b`。原阶段 1、原 PPO 工作区及原始数据未改动；未 commit/push。

**实测 Fixed 完整 PPO iteration：L3D 1.319×，I3D 1.155×。** 默认 7200 周期上限、同一未训练冻结策略的 development 评价，全进程分别节省 513.33 s / 284.25 s。动力学、终止、奖励、PPO 更新及完整逐步证据语义保持；评价计算成本按真实执行量记账。阶段 3 首选 I3D 气动执行的数值等价优化。

## 实现与合同

- `evaluate()` 在终止记录已生成、落盘且终止 mask 核对之后检查全任务结束，立即退出。development/final 共用此循环；未完成则仍拒绝，单条任务 120 s / max_steps=7199 合同不变。提前结束没有补造 episode 或成功见证。
- `reserved_step_budget` 是每环境最大控制周期；`executed_environment_steps` 是执行周期乘槽位数；`active_episode_steps` 为独立重放所得 episode 步数之和；`padding_steps=executed-active`；`early_stopped` 表示全部结束且执行周期小于上限。旧 `environment_steps` 保留为实际执行量的别名，绝不填预留预算。
- 接收端先独立重放 SQLite，再验证 `executed = 任务数 × 最后终止周期`、active、padding、上限及 early_stopped。Fixed/Adaptive 共用最大预算、固定任务、初态和判据；实际成本允许随策略变化。
- checkpoint 状态中的 `eval_env_steps` 累加实际执行；另存 `eval_reserved_env_steps`、`eval_padding_steps`，保留 `eval_active_steps`。pending/evaluated 恢复通过同一接收端；重复接收拒绝，接收失败回滚 sampler。没有放宽跨源码训练 checkpoint 的原合同校验；旧格式评价 receipt 不冒充新记账格式。
- `_interval_controls()` 一次读取各批 Tensor 并转换为列，再按环境组织完整控制见证。原始动作、实际命令、滤波器/相位驱动内部状态、频率、关节、机体、参考和投影字段全部保留，包括长期不变字段。状态标量与 episode step 同样批量读取。
- `RolloutEvaluator.observe_intervals()` 接收原始数据而非上游认证布尔标志，每条 interval 只执行一次 `valid_interval()`；环境共用结果。严格离线重放仍走 `observe()`，从 SQLite 内容独立重算。`terminal_outcome()` 判定次序与规则未改；Python float 有限性调用改为 `math.isfinite`，整数/NumPy 标量及空容器规则保留。没有进一步引入双套 Tensor 终止规则。
- SQLite 表、字段、逐步 JSON、WAL/FULL、每控制步 flush、终止时持久化及 witness 写入规则未改。本轮没有改数据库事务架构。

## 有界真实 PPO 性能

全部为 CPU-native PhysX、256 env × 48 控制步、seed=0、Torch intra-op=1，原 39D/4D、480/60 Hz、35°扭转、每翼10站、L3D/I3D、5 epochs × 4 minibatches。每个后端/采样组合只做一次完整 PPO iteration，每组20次 Adam step（四组共80次）；没有正式学习或多种子训练。原始基线直接使用阶段 1 产物（L3D Fixed 使用有效计时目录 `L3D_fixed_n256_t1_v2`）。

| 后端/采样 | rollout s 基线→优化 | update s 基线→优化 | iteration s 基线→优化 | PPO transitions/s 基线→优化 | iteration 加速 |
|---|---:|---:|---:|---:|---:|
| L3D_fixed | 10.524 → 7.724 | 0.917 → 0.950 | 11.441 → 8.674 | 1074.0 → 1416.7 | 1.319× |
| L3D_adaptive | 10.575 → 7.709 | 0.856 → 0.823 | 11.432 → 8.532 | 1074.9 → 1440.2 | 1.340× |
| I3D_fixed | 15.419 → 13.183 | 0.937 → 0.976 | 16.356 → 14.159 | 751.3 → 867.9 | 1.155× |
| I3D_adaptive | 15.780 → 12.502 | 0.909 → 0.908 | 16.688 → 13.410 | 736.3 → 916.4 | 1.244× |

**完整命令进程还包含 Kit 启动、初始化、停止时在途 episode 的 manual_reset、保存加载验证和关闭**，其速度不能用 iteration 吞吐替代：

| 后端/采样 | 全进程 s 基线→优化 | 全进程加速 |
|---|---:|---:|
| L3D_fixed | 25.82 → 22.41 | 1.152× |
| L3D_adaptive | 27.84 → 23.67 | 1.176× |
| I3D_fixed | 30.13 → 27.69 | 1.088× |
| I3D_adaptive | 32.41 → 28.84 | 1.124× |

以上为真实单次端到端有界训练调用，不是稳态长训练统计。第一次 iteration 不触发第25次更新的评价；下节评价独立实测。没有把训练与评价分项拼接后的理论摊销称为正式训练端到端实测。机器非独占、单次运行无置信区间；L3D 长预算基线在中断前运行，其余续跑时间可从原日志核对。

| `_get_dones()` / iteration | 基线 s | 优化 s | 消除时间 | 降幅 |
|---|---:|---:|---:|---:|
| L3D_fixed | 5.8415 | 3.0788 | 2.7627 s | 47.3% |
| I3D_fixed | 5.5153 | 2.8234 | 2.6919 s | 48.8% |

I3D 同等级细分计时对照：两处完整有效性递归合计 **2.0895 → 0.5701 s**（减少72.7%，调用次数24,576→12,288）；逐环境见证转换 **0.6710 → 0.2772 s**（减少58.7%，顶层调用12,288→48）。原普通计时的逐行转换为0.8065 s，不能与另一轮细分计时混成同一次分解。L3D 首个优化运行未单列新增批量转换包装器，因此不虚构其独立转换耗时。

优化后 `_get_dones()` 仍需完整验证、JSON构造、统计和持久化，约2.8–3.1 s，不能视为全部可移除。I3D Fixed 的气动运动学/计算/坐标变换仍为 **8.232 s / 14.159 s iteration（58.1%）**；其中 horseshoe_influence 3.904 s、cond 1.086 s、运动学采样1.685 s。SQL execute/commit 未重写。

## 默认长预算冻结评价：实际墙钟与记账

使用历史同一 development 未训练快照的本地副本，**只将副本 contract.eval_step_budget 从256改为7200**，策略、任务、初态与原始终止合同不变；原快照未改。两版本都使用该副本，完整严格重放通过。通过 harness 分别导入阶段1/阶段2实现，物理资产与 physical_v2 身份仍从原 PPO checkout 读取；没有绕过身份校验。

| 后端 | 指标 | 基线 | 优化后 |
|---|---|---:|---:|
| L3D | 评价循环+严格验证 s | 516.231 | 5.192 |
| L3D | 全进程墙钟 s | 526.346 | 13.014 |
| L3D | 每环境最大预算 | 7200 | 7200 |
| L3D | 真实环境步数 | 36000 | 630 |
| L3D | 有效episode步数 | 626 | 626 |
| L3D | padding步数 | 35374 | 4 |
| I3D | 评价循环+严格验证 s | 287.624 | 4.430 |
| I3D | 全进程墙钟 s | 296.010 | 11.756 |
| I3D | 每环境最大预算 | 7200 | 7200 |
| I3D | 真实环境步数 | 36000 | 475 |
| I3D | 有效episode步数 | 471 | 471 |
| I3D | padding步数 | 35529 | 4 |

全进程 L3D 节省 513.33 s（40.45×），I3D 节省 284.25 s（25.18×）。两组 optimized 的 early_stopped=true；最后终止周期仍为126/95，均为5个真实失败终止，无成功见证。剩余4个padding是任务异步结束造成的槽位计算，未隐去。

这些是默认7200上限早期失败工况的实测，不是阶段1按256周期外推的估算。L3D长预算padding成本明显高于先前短预算线性外推；本轮不以此推断另一后端或未来策略的耗时，也未额外重跑长预算以挑选数字。

## 数值、证据与恢复验证

- 四个真实 PPO 组合：初始及最终 policy digest、20次 Adam计数、loss、平均reward完全一致；SQLite每个episode的轨迹/初态/合同、全部原始控制与物理状态、终止统计与顺序逐字段一致。只排除运行UUID、policy版本字符串及checkout路径等明确来源标识；physical_sources及实际配置核对保留。
- 确定性动作 `0.35*sin(arange(256*4)*0.013 + step*0.17)`，每后端独立baseline/optimized/profile重放48步。保存原始NPZ：动作[48,256,4]；气动力/力矩、关节与机体状态[384,256,35]；观测/奖励/done/timeout/投影[48,256,46]。力及子步状态在施力后、积分前采样，控制周期后状态另由SQLite及interval数组覆盖。
- 两后端三个数组优化前后及计时开关间均逐值相等，最大绝对差0；实际plant身份精确一致，完整SQLite重放证据同样一致。没有通过容差或终止阈值放宽掩盖差异。这里证明与原实现数值一致，不构成真实飞行物理真实性验证。

| SQLite / Fixed | 基线 bytes | 优化 bytes | step数 两者 | terminal数 两者 |
|---|---:|---:|---:|---:|
| L3D 训练 | 106475520 | 106479616 | 12288 | 291 |
| L3D development | 3596288 | 3596288 | 626 | 5 |
| I3D 训练 | 100044800 | 100052992 | 12288 | 257 |
| I3D development | 2949120 | 2949120 | 471 | 5 |

训练terminal数包含干净停止时256个manual_reset：L3D另35个失败、I3D另1个失败。SQLite文件大小相差少量页不代表缺失内容；未对具体页布局差异做归因，内容与行数一致性由逐记录比较确认。

- 定向测试最终 **132 passed, 1 skipped**：非有限原始输入、失败优先于成功、timeout一致性、完整无效interval、严格冻结重放/篡改拒绝、预算耗尽拒绝、development/final提前退出、actual/reserved/padding篡改拒绝、Fixed/Adaptive pending→evaluated checkpoint落盘恢复与重复接收拒绝。skip为新worktree缺历史快照；另从原历史路径独立运行真实SQLite前缀只读恢复测试 **1 passed**。
- checkpoint恢复采用定向orchestration及真实torch.save/load往返，SQLite前缀另用实际历史文件验证；没有额外启动训练resume或正式学习。
- 最初沙箱Kit/CUDA启动崩溃的目录不计入性能；未产生PPO更新。一次跨checkout联合pytest因同名测试模块导入串用而失败，拆分进程后通过。日志保留，未用失败产物充当有效对照。

## 插桩开销与边界

主性能使用阶段1同类perf_counter调用树；优化后启用额外细分计时，原始吞吐不做人为扣减。以下为独立、相同动作/相同状态捕获的真实PhysX重放，包含必要NPZ采样成本但不包含PPO；不能把它等同于PPO全迭代精确计时误差：

| 后端 | 优化无细分计时 env.step s | 优化开细分计时 s | 观测差异 |
|---|---:|---:|---:|
| L3D | 6.6566 | 6.8207 | +2.46% |
| I3D | 12.2292 | 12.0876 | -1.16% |

两次均保存原始数组且结果完全一致。单次差异含运行噪声，不是置信区间。没有继续增加PPO重复、线程搜索或气动微优化。

## 复现与产物

[完整机器可读汇总](results/unified_rl_performance_phase2/summary.json)、[原始结果目录](results/unified_rl_performance_phase2/)、[定向测试日志](results/unified_rl_performance_phase2/tests_final.log)。各benchmark运行保留SQLite、参数/身份、源码快照、计时/资源与日志；物理重放保存SQLite、身份、replay.npz及日志。性能PPO权重已按原harness保存加载验证后删除；冻结smoke副本明确不可用于正式resume。

前提：原phase1、原PPO和control-oriented-3d-wing-aero资产工作树仍位于同一父目录，使用已配置env_isaaclab。以下一键入口要求新输出目录，会顺序执行定向测试、每组一次有界PPO、两版本7200上限development评价、确定性物理重放及汇总，绝不启动正式训练：

```bash
cd /home/zn/IsaacLab/.worktrees/unified-rl-performance-phase2
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
./isaaclab.sh -p scripts/flapping_px4/run_unified_phase2.py \
  --output docs/results/unified_rl_performance_phase2_repeat
```

仅重新汇总已有结果：

```bash
./isaaclab.sh -p scripts/flapping_px4/summarize_unified_phase2.py \
  docs/results/unified_rl_performance_phase2
```

入口检查completion.json而非只信wrapper退出码；总汇总passed还要求原始物理/PPO/SQLite一致性。原始首次L3D Fixed运行尚未单列批量转换计时；复现入口使用最终细分包装器，这一插桩差异如实保留。

## 阶段 3 判断

优先 **I3D气动执行效率**：horseshoe_influence、矩阵条件数及运动学采样已实测占据主要成本，可在不改公式、站数与数值保护的前提下研究重复构造/分配与线性代数实现。该方向边界明确，比现在直接重构多进程PPO更便于做严格数值对照。

CPU多进程对L3D及总体吞吐仍值得后续小规模可行性验证，但IPC、worker负载、原生PhysX启动、证据分片与统一on-policy采样尚未测量，不能给出多worker加速承诺。本轮没有实现多进程、GPU迁移、物理降精度、气动公式改动或正式学习。
