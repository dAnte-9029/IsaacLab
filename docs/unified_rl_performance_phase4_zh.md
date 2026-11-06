# Unified RL 阶段 4：多进程 CPU-native PhysX

2026-11-07。独立工作树 `.worktrees/unified-rl-performance-phase4`，参考提交 `eff6ccf5bdbcc4c79f55aeeae436da8cadf93fa3`。所有权重和数据仅用于 benchmark；没有正式训练或 distributed optimizer。测量阶段结束时未 commit/push，随后按用户要求发布代码、报告及精简结果；原始大体积产物保留本地，范围见结果目录 README。

**VERIFIED：真实 CPU-native PhysX 多进程可以提高有界总吞吐。** I3D 1/2/4 个独立 256-env 进程的训练段总吞吐分别为 894 / 1684 / 3088 transitions/s。已测最高总吞吐是四进程；没有测试更多进程，也没有证明长期最优。默认调度与双进程均分 affinity 的差异不足以支持调度修改。

## 实测吞吐：增加总环境数

| 后端/配置 | 总 env | worker transitions/s | 总 transitions/s | 总吞吐/单进程（效率） | 最慢训练 s | 全部实验墙钟 s | 端到端 transitions/s |
|---|---:|---|---:|---:|---:|---:|---:|
| I3D 1×256 | 256 | 894 | 893.8 | 1.000× (100.0%) | 27.495 | 44.042 | 558.0 |
| I3D 2×256 | 512 | 838 / 847 | 1684.2 | 1.884× (94.2%) | 29.341 | 45.450 | 1081.4 |
| I3D 4×256 | 1024 | 755 / 771 / 785 / 777 | 3088.4 | 3.455× (86.4%) | 32.564 | 52.257 | 1881.2 |
| L3D 1×256 | 256 | 1420 | 1419.7 | 1.000× (100.0%) | 17.310 | 32.446 | 757.5 |
| L3D 2×256 | 512 | 1330 / 1377 | 2707.2 | 1.907× (95.3%) | 18.475 | 34.440 | 1427.2 |

“总预算/最慢训练时间”对应 I3D 双/四进程分别为 1675.2 / 3018.8 transitions/s，略低于简单 rates 相加。四进程开始到最后退出的墙钟已包含慢 worker；单进程/双进程/四进程总 transition 预算不同。L3D 双进程同样接近翻倍，本数据不支持 I3D 比 L3D 更易扩展的强结论。

## 实测吞吐：总环境数固定 256

| 配置 | worker transitions/s | 总 transitions/s | 最慢 rollout 合计 s | 最慢训练 s | 全部墙钟 s | 训练段加速 | 端到端加速 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1×256 | 894 | 893.8 | 25.801 | 27.495 | 44.042 | 1.000× | 1.000× |
| 2×128 | 730 / 748 | 1478.1 | 15.915 | 16.823 | 29.134 | 1.634× | 1.512× |
| 4×64 | 540 / 545 / 538 / 538 | 2162.0 | 10.869 | 11.418 | 25.938 | 2.408× | 1.698× |

三项均为 24,576 transitions，但每个 worker 独立更新较小 batch，且不同 seed/local env ID 形成不同任务序列。这里只验证执行能力，不是同一全局 PPO 更新。4×64 比 2×128 的完整墙钟收益小于训练段收益，因为每个 Kit 的固定开销无法按环境数缩小。

## 独立实验与 affinity

相同 seed 身份的两个 Fixed 实验：单独运行合计 **86.394 s**，并发 **45.450 s**，实测 **1.901×**。Fixed(seed=0)+Adaptive(seed=1)：顺序合计 **88.594 s**，并发 **47.847 s**，实测 **1.852×**。顺序时间为分别实测的完整进程墙钟之和，不是把一个 seed 的时间乘二；未包含人工发起两条命令之间的等待。该结论只覆盖各两轮有界实验，不证明长训练到同等学习质量所需时间。

双进程拓扑均分：worker 0=`0,1,4,5,8,9,12,13,16,18,20,22,24,26,28,30`；worker 1=`2,3,6,7,10,11,14,15,17,19,21,23,25,27,29,31`。每个 worker 为 4 P-core（8 SMT 线程）+8 E-core，物理核和逻辑 CPU 均不重叠；按 sysfs CORE 分组，未按连续 CPU 范围猜测。E-core L2 cluster 仍可能跨 worker 共享。

默认调度：1684.2 transitions/s、45.450 s；均分 affinity：1692.8 transitions/s、46.253 s。吞吐差 0.51%，不具备统计显著性证据；无需扩大线程/亲和性搜索，推荐默认调度。

## CPU、内存、启动和存储

Intel i9-14900K，8 P-core×2 SMT +16 E-core，32 逻辑 CPU 全在线、单 NUMA。初始 load average 0.09/0.07/0.17，允许 CPU 0–31。Torch intra-op=1、inter-op=24；Kit 设置 `/persistent/physics/numThreads=8`、`/physics/physxDispatcher=true`、tasking threadCount=32；这是真实设置读取值，不能把全部后台线程归为 PhysX solver。PL1/PL2 读取均为 253 W；未修改调度器、功率限制或全局设置。没有读取到可归属 worker 的 package 能耗，因此不报告每 transition 焦耳数。

| 配置 | 各 worker 采样峰值 RSS MiB | 各 worker 峰值线程数 | 全进程 CPU ms/transition 下界 | Kit 启动 s | build_env s | SQLite commit 合计 s | 数据库合计 MiB |
|---|---|---|---:|---|---|---:|---:|
| i1 | 3804 | 194 | 2.393 | 4.18 | 4.49 | 3.508 | 168.2 |
| i2 | 3804 / 3820 | 194 / 195 | 2.468 | 3.98 / 4.01 | 5.18 / 5.18 | 7.046 | 327.8 |
| i4 | 3846 / 3809 / 3797 / 3801 | 194 / 194 / 195 / 195 | 2.595 | 5.68 / 5.67 / 5.60 / 5.68 | 6.85 / 6.83 / 6.83 / 6.85 | 14.710 | 641.5 |
| b2 | 3724 / 3719 | 195 / 194 | 3.259 | 4.37 / 4.29 | 3.45 / 3.49 | 3.986 | 162.3 |
| b4 | 3658 / 3660 / 3661 / 3669 | 194 / 194 / 194 / 194 | 5.234 | 5.69 / 5.72 / 5.70 / 5.69 | 4.47 / 4.55 / 4.61 / 4.59 | 5.199 | 160.3 |
| l1 | 3799 | 194 | 1.920 | 3.58 | 4.46 | 3.362 | 166.1 |
| l2 | 3783 / 3797 | 194 / 194 | 1.949 | 4.16 / 4.22 | 5.17 / 5.20 | 6.832 | 325.5 |
| aff2 | 3802 / 3796 | 179 / 179 | 2.303 | 4.16 / 4.19 | 5.06 / 5.12 | 7.193 | 327.8 |

四进程独立 RSS 峰值之和约 14.89 GiB（各自峰值不一定同时出现），远低于约 125 GiB 内存；共享映射按每个进程 RSS 重复计入。**CPU coretemp 最高 100.0°C，达到传感器 critical 阈值；单进程也出现这一读数。** 仅有温度/频率快照，未获得本次任务前后的 throttle counter 差值，不能确定降频持续时间，也不能把四进程吞吐损失全部归因于线程争用。长期扩展需把热限制纳入预算，本阶段不追加热负载测试。每核利用率和频率逐秒原始值见 resources.jsonl，均值见 summary.json。
以各配置 w0 的整机采样为准，i1/i2/i4 的 100°C 样本分别为 9/39、12/41、7/47；对应 P-core 逻辑 CPU 平均频率 3355 / 3498 / 3640 MHz，整机平均忙碌量 1.57 / 3.06 / 5.43 个逻辑核当量。均值包含空闲核与启动，不能当作执行线程频率；原始每核数据保留。[温度/频率摘要](results/unified_rl_performance_phase4/thermal_summary.json)。

初始 GPU 进程清单无其他训练任务；四 Kit 快照中每实例在 GPU0/GPU1 约 454/336 MiB，合计设备用量约 2157/1589 MiB（含桌面）。CPU PhysX 仍会由 Kit 创建图形/CUDA上下文，显存使用不代表动力学迁移到 GPU。驱动把每张 4090 的总显存报告为 49140 MiB，本报告沿用实际读数，不按标准零售容量推断。

完整墙钟减最慢训练时间，i1/i2/i4 分别为 16.547 / 16.109 / 19.694 s，包含启动、环境构建、快照/保存、关闭和监控开销。summary 中另列 close_wrapper_poll_residual_s：parent 全进程时间减 child 关闭前时间，只是关闭+wrapper/import+轮询尾部的近似残差，不能当成精确 Kit shutdown 时间。
Kit 仍记录 change-watch errno=28 的文件监视资源不足提示，单进程基线也存在；磁盘实查剩余约 277 GiB，所以不是数据盘空间耗尽。实例与验收均完成，未修改系统 inotify 限额，但不能据此保证更多实例可继续成功启动。
SQLite 各 worker 独占，完整性检查与 evidence 重算全部通过。数据库字节数和 block I/O 都有记录，但 page cache、WAL、snapshot/copy 会使物理写入量不同于最终数据库大小。当前没有数据库锁竞争或磁盘瓶颈证据；相同 256 总环境拆成四 Kit 时 CPU 成本上升明显，说明初始化/重复进程工作和小 batch 效率比 IPC 更值得关注。长期 evidence 线性增长、缓存压力及多小时温度未验证。

## 数值与证据验收

**VERIFIED：** 单独 I3D replay 与同时运行的两个 I3D replay，动作 `[48,256,4]`、384 个物理子步的力/矩及机体/关节状态 `[384,256,35]`、控制间隔的 obs/reward/done/timeout/参考误差 `[48,256,46]` 均逐字节一致，最大差 0；对应 SQLite 控制和终止证据一致。阶段 3 影子力学对照同样通过。全部 26 个 worker 数据库独立重算，steps 总数 503,808，witness 总数 0；manual_reset 未算成功。未发生跨 shard episode UUID 重复，所有 steps 数与预算一致。
相同 seed 的 Fixed 单/双进程，以及 Adaptive 单独/配对运行，初始/最终 policy 和两轮全部 loss/reward 精确一致；这支持有界并发不改变各独立实验，不代表跨 worker 合成一个 PPO 等价。物理源身份在各 worker 间完全相同，且 git diff 未修改物理、训练、奖励或采样源码。新 harness 语法检查、实际拓扑分区检查和真实运行完成；没有运行完整测试集。

## IPC 实测及收益模型

实际载荷：策略广播 1,140,679 bytes；两个 rollout shard 合计 2,893,722 bytes，含 39D obs、4D action、raw/bootstrapped reward、done/timeout、log-prob/value、mu/sigma、最后 obs/value、returns/advantages。6 次传输，首轮单列，其余 5 轮中位数：

| IPC 指标 | 实测 |
|---|---:|
| 策略序列化 | 0.117 ms |
| 广播至两 worker 确认（含反序列化和校验） | 0.504 ms |
| 真实 rollout 接收与反序列化 | 0.685 ms |
| 其中 learner 反序列化 | 0.146 ms |
| learner 拼接 | 0.139 ms |
| 两 worker 准备时刻差 | 0.127 ms |
| 总通信+重组 | 1.390 ms |
| 首轮总通信+重组 | 3.511 ms |
| 两个通信进程冷启动/读入 | 679.437 ms |

worker pickle 序列化中位数分别为 0.149 / 0.410 ms；learner 轮末最大采样 RSS 482.0 MiB（包括 Python/Torch 基础内存、源数据、收到的 shards 和重组副本），两个通信 worker 约 477.6 / 477.6 MiB。数据已预载为 CPU ndarray，未把磁盘读入算作每轮通信；未测与 PhysX 同时竞争的传输成本、活跃 worker barrier 或故障恢复。重组逐项精确相等，错版本/错轮次/错 worker 均拒绝。

以固定全局 256 env 为例，实测两轮单进程 rollout 25.801 s，2×128 最慢 worker rollout 15.915 s；沿用原 256-env learner 的两轮 update 1.693 s，加两轮 IPC 0.0028 s，得到持久 worker 的**理想成本模型** 17.610 s，对照实际单进程训练 27.495 s，即 1.561× 潜力。通信约占一次双 worker 数据生成 0.017%。这不是实际同步 PPO 加速比：各 worker 在性能矩阵用了独立策略，尚未测统一策略、真实 barrier、全局 storage/GAE 和 Adaptive tick 协调。两 worker 两轮 rollout 合计时间差 0.437 s 只是负载不均的观测，不能当成逐轮 barrier 等待实测。

## 下一阶段决策

1. **现在采用独立实验并发。** Fixed/Adaptive 各自保持完整实验身份，已实测减少获得两份短程结果的墙钟；两组实验就用两个默认调度进程。有四份独立实验时，四进程是已测吞吐最高配置，但已有 100°C 短时读数，不据此推荐四进程长期运行。
2. **同步 PPO 值得做下一阶段小范围 Fixed 原型。** 保持全局 256 env，优先 2×128 持久 PhysX worker + 一个 CPU learner；通信成本没有吞噬实测采样潜力。不要默认上四 worker、独立更新后平均参数或扩大 batch。Adaptive 暂时只用独立实验并发，先解决逐 tick 事件顺序与证据存储接口再谈等价合并。
3. **主要限制是每 Kit 固定启动/内存成本、CPU/小 batch 效率及可能的热限制，其次是同步时最慢 worker 与 Adaptive 协调；本次 IPC、SQLite 不是主瓶颈。** 长期采样可摊薄启动，但本阶段未测长稳态，不能外推学习收敛时间。
4. **不建议立即介入双 RTX 4090 learner。** 原 CPU update 只占单进程训练约 6.2%；即使完全消除这一部分，单进程理论上限也仅 1.066×。单 GPU learner 可在以后 global update 真正成为瓶颈时单独测，双 GPU 梯度同步没有本次实测支持；不能把 Kit 已分配显存当成 GPU learner 加速证据。

每项仅一组两轮测量；seed0/seed1 独立基准提供近邻交叉参照，不构成多次统计实验。未观察失败/超时，没有反复试错；首次 wrapper 因 TERM=dumb 在启动前退出，设置 TERM=xterm 后才开始测量，不计为 PhysX 启动失败。本阶段到有界测量、IPC 与报告为止。

## 测量合同

沿用阶段 3 的 `benchmark_unified_performance.py` 与实际训练入口，仅扩展外部启动、seed、两轮预算、资源记录及可选真实 storage 导出。每 worker：CPU-native PhysX、480 Hz 物理/60 Hz 控制、48 steps × 2 rollout/update、Torch intra-op=1、39D/4D、原 256×3 actor/critic、5 epochs × 4 minibatches。保持气动、约束、URDF、执行器、奖励、终止、任务采样及 PPO 源码不变。独立 worker 的 seed 为 0、1、2、3，输出和 SQLite 独占；不共享策略或优化器。不同规模比较不宣称相同学习轨迹。

性能矩阵使用相同 instrumentation，未开启额外气动 detail。IPC source 单独导出第一轮真实数据，导出耗时不计入性能矩阵。重放沿用阶段 3 的确定性动作与影子对照，也不用于计时结论。单项失败/超时立即停止，保留日志，不自动重试。各训练 checkpoint 经过原 runner 保存/加载验证后删除，IPC 文件保留权重但没有正式恢复所需完整合同，禁止用作恢复。

**分母：** worker transitions/s = 本地 transitions / (rollout + PPO update)；总吞吐为 worker rates 之和，另列总预算 / 最慢训练时间，避免异步启动造成误读。端到端吞吐 = 总 transitions / 从启动全部 wrapper 到所有进程退出的实际墙钟。两轮 rollout 包含证据记录和 returns，最终证据 flush、checkpoint 保存/加载及关闭在端到端分母内。启动成本不能只看 Kit 初始化，还包括环境构建和 Python/import 等固定开销。

CPU 时间取 `/proc` 进程树累计 user+system 的最后可读样本，不累加 children 字段以避免重复计数；约 1 秒采样，退出前的尾部可能遗漏，因此 CPU/transition 是全进程成本下界，不能视为纯物理求解成本。每核利用率、频率、温度、磁盘统计和 GPU 总量是整机观测；进程树 CPU/RSS/I/O 可归属 worker，不能把整机磁盘或 GPU 指标全归因给某个 worker。1 秒父进程轮询使退出时间有约 1 秒量化误差，外层另有 0.1 秒轮询。

## 同步 PPO 协议与算法边界

建议的协议为 `Prepare(round, version, policy_state, plant_contract, global_env_ids, RNG_streams)` → 全部 worker 确认版本 → `Collect(48)` → 各 worker 返回 `Rollout(round, version, worker_id, local_sequence, tensors, evidence_cursor)` → learner 校验版本、数量、形状、终止与分片完整性 → barrier → 一个 learner 统一 returns/advantages/PPO update → 发布下一版。超时、缺片、重复片或版本不符均中止本轮，不能默默丢 worker 或混入下一版数据；不能平均独立 PPO 更新后的参数。

载荷至少保留 `[T,N,39]` obs、`[T,N,4]` raw policy action、raw reward、terminated、timeout、old log-prob、value、Gaussian mu/sigma、最后 obs/value，以及 episode/环境/版本定位信息。SQLite 保留环境实际执行的控制证据。PPO raw action 与环境 applied action 分开；不能用裁剪后的动作重算原 log-prob。真实导出还包含原 storage 的 bootstrapped reward、returns 和 advantages，后两项仅用于检查，不能把各 worker 已标准化的 advantage 直接拼起来。

当前 RSL-RL 的 timeout bootstrap 是 `reward += gamma * V_current * timeout`，随后以 `done | timeout` 截断 GAE；必须保持这个已存在的具体行为，不能悄悄改为另一种 terminal-state bootstrap。沿每个环境的时间轴递推 GAE，再对全局 `T × N_global` advantage 做一次 mean/std 标准化（当前 `normalize_advantage_per_mini_batch=False`）。只在 env 轴合并 storage，避免连接两个 worker 的时间轴。全局固定 256 env 时，仍是 48×256 batch、5×4 次 Adam；总环境增至 512/1024 会改变 global batch 和 minibatch 尺寸，不能再声称原 256-env 数学合同等价。

当前 `RolloutStorage.mini_batch_generator` 一次产生全局 randperm，再用于各 epoch；learner 应持有唯一 minibatch RNG。worker 的任务 RNG 与动作采样 RNG 需分离、按全局 env ID 分配并可恢复。仅给不同 worker 相同 seed 会重复任务/动作；不同 seed 保持分布却不保证与单进程逐位相同。网络的 actor/critic observation normalization 当前关闭；如果未来开启，normalizer 状态也必须由全局合同规定，不能独立漂移。optimizer、scheduler、global update/transition count 只属于 learner。

**Adaptive 是额外的语义约束。** 当前 `OnlineSampler.feedback` 要求原始 journal 记录可回查，且 `sequence` 严格递增；`_pending`、region 更新、capability、witness 计数和冻结评价 receipt 都受顺序影响。不能直接合并各 shard 的本地 seq，也不能把数据库拼接后假称等价。应保留原始 `(worker, local_seq)`，由唯一协调器按控制 tick、全局 env ID、局部事件序号确定全局顺序；在同一 tick 的终止处理完成后，再按原 reset 顺序分配新任务。要与原同一 global-env 实验保持反馈时效，可能需要每个控制 tick 的反馈/reset 协调，成本未测。仅在 rollout barrier 合并会延迟 Adaptive 更新，是协议变化，未获本阶段授权。跨策略 episode 仍记录每步版本，不能变成冻结能力 witness；独立 frozen evaluation 仍独立执行和去重。

本阶段 IPC 原型用两个持久 spawn 进程，通过 Pipe/pickle protocol 5 广播实际策略参数，再传回同一真实 48×256 rollout 的两个 128-env shard，逐项验证重组后数据相等并拒绝错误版本/轮次。它测量真实尺寸和真实数值的传输/序列化/重组成本，**没有在这两个通信进程中运行 PhysX**，也没有统一 PPO 更新；不能当成 live synchronous PPO 已验证。通信进程启动和常驻内存单列，不与 Kit worker 重复相加；未来 IPC endpoint 应位于已有 Kit worker 内。

## 复现与产物

```bash
cd /home/zn/IsaacLab/.worktrees/unified-rl-performance-phase4
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
./isaaclab.sh -p scripts/flapping_px4/benchmark_unified_phase4.py \
  --output docs/results/unified_rl_performance_phase4_repeat
```

新输出目录完整执行有界矩阵、实际载荷 IPC、确定性物理重放和 SQLite 重算验收。`--case i1/i2/i4/b2/b4/l1/l2/aff2/seed1/adaptive1/mixed2/ipc_source/replay1/replay2` 可单独运行；`--continue-completed` 只跳过成功项，遇到失败项不会重试。双进程吞吐增长低于 1.15 时，一键流程跳过增加总环境数的四进程配置。wrapper 的退出码不可靠，因此所有子项显式检查 completion 与最终 passed。

原始结果位于 [results/unified_rl_performance_phase4](results/unified_rl_performance_phase4/)，包含每 worker 的参数、runtime、真实 plant 身份、training.jsonl、SQLite、资源时间序列、计时、日志和 completion；[summary.json](results/unified_rl_performance_phase4/summary.json) 为只读重算结果，[ipc.json](results/unified_rl_performance_phase4/ipc.json) 为通信成本。物理重放保留 NPZ；不存在自动选优 checkpoint 或正式恢复产物。
