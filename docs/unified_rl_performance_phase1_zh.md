# Unified Trajectory PPO 性能定标与瓶颈归因（阶段 1）

2026-11-06。**下一轮首先改 `UnifiedTrajectoryEnv._get_dones()` 的逐 env 控制见证构建与重复有效性检查，保持完整证据和终止语义。** 这段共享链路在 L3D/I3D 都占约 5 秒/iteration；它是 L3D 的首要瓶颈，也是 I3D 的第二大瓶颈。I3D 自身第一大项是气动力计算，后续应针对其已测热点做数值等价优化。当前数据不支持先重构 PhysX 内部或将小型 PPO 单独迁到双 GPU。

本轮仅 profiling 和有界 benchmark；没有实施上述优化，没有正式学习实验，没有 commit/push。原 worktree 保持干净。

## 范围与可复用产物

- 原工作区：`/home/zn/IsaacLab/.worktrees/unified-trajectory-ppo-v1`，起止 HEAD 均为 `5406ab6c527364ef985e9fc2af83472d590d6709`。
- 独立 detached worktree：`/home/zn/IsaacLab/.worktrees/unified-rl-performance-phase1`，同一 HEAD。仅新增 benchmark、汇总脚本、此报告和结果，既有源码零修改。
- [一键入口](../scripts/flapping_px4/benchmark_unified_performance.py)、[汇总入口](../scripts/flapping_px4/summarize_unified_performance.py)、[JSON](results/unified_rl_performance_phase1/summary.json)、[CSV](results/unified_rl_performance_phase1/summary.csv)、[结果目录](results/unified_rl_performance_phase1/)。每个目录保存参数、硬件、实际物理身份、版本、逐秒资源、原日志、细分计时和原始 SQLite。
- 每次训练调用严格 `iterations=1, seed=0`，默认 `256 env × 48 steps`，原 PPO 的 5 epochs × 4 minibatches，共 20 次 Adam step。保留原日志、反馈与评价设置：`eval_interval=25, save_interval=25, eval_step_budget=7200`；第一次更新本来就不触发评价。48 步是 0.8 s 仿真时间，不是长期稳态吞吐。
- 11 个有效一次更新测试（四组基准、两组计时关闭对照、四组 CPU 扩展、一组 I3D 细分）以及两次已有快照 development 评价。另有一个计时未导出的诊断 iteration 和一次身份拒绝的评价初始化，均列入 `excluded`，没有用于比较。未重跑 final 集合。
- 原 runner 保存/加载校验完成后，benchmark 删除**本次目录内**的性能测量权重及 latest 指针，仅保留 checkpoint 时间/大小。没有可用于正式 resume 的新 checkpoint。

实际 plant 为 `measured_wing_multibody + native_holonomic_drive`，CPU PhysX，480 Hz 物理、60 Hz 控制、decimation=8，L3D/I3D 原 35° 扭转、每翼 10 站、原候选初态、39D 观测/4D 动作。几何、气动力/施力、URDF、质量惯量、native 约束、执行器、任务、PPO 数学定义和 Fixed/Adaptive 合同均不变。每轮 `runtime.json`/`run/contract.json` 保存实际配置与原有 physical_v2 身份。

## 方法与测量边界

通过独立进程内的方法包装器使用 `perf_counter`；不向每个物理子步打印。`timings.json/csv` 对完整调用路径保存调用数、累计 inclusive/exclusive 秒、中位数、P95、exclusive 占整个已计时根节点的比例。父项包含子项，**不能把两者再次相加**；exclusive 只减去已插桩的子调用，不能解释成“纯 Python 时间”。本报告占比以 iteration 为分母，和 CSV 的全生命周期分母不同。

`sim.step` 包含 PhysX/native 求解及 Isaac 接口开销；内部约束求解、积分、内部关节更新没有可靠单独测量，标为未分解。`_apply_action` 剩余项还含执行器命令、其他翼/尾/机体载荷与缓冲操作，不能全部叫执行器成本。Articulation 状态存在惰性读取，其成本可落在使用者（例如气动运动学）里。SQLite commit 包含 FULL 持久化等待，未用 syscall 跟踪把磁盘同步与 SQLite 内部工作拆开；INSERT 时间不包含已单列的 JSON 编码。

计时使用主线程调用树；后台 PhysX/Kit 线程以整个同步调用的墙钟计入，不把各线程 CPU 时间加成墙钟。资源每约 1 s 采样，含两张卡、每核系统占用、目标进程/线程累计 CPU、RSS、I/O 和数据库/WAL 大小。端到端计时含 wrapper/进程启动和退出，结束检测有最多约一个采样周期的量化误差；硬件快照在此计时之前。当前资源数据覆盖全生命周期，不把短 PPO 峰值误当持续占用。

## 硬件与运行环境（实测）

| 项目 | 观测 |
|---|---|
| CPU | Intel Core i9-14900K，1 socket，24 个物理核心、32 个逻辑 CPU |
| 混合核心 | 8 个带 SMT 的核心：CPU 0–15，每对为同一物理核；16 个单线程核心：CPU 16–31。最高频率 5.7/6.0 GHz 与 4.4 GHz，最低 0.8 GHz；快照另保存当前 MHz |
| NUMA | 只有 node0，CPU 0–31，全部内存属于 node0。`numactl` 未安装，使用 lscpu 与 `numa_sysfs.json`，未安装新依赖 |
| 内存 | 134,717,734,912 bytes（125.47 GiB），初始可用约 99 GiB；已有 swap 使用，不能据此声称本 benchmark 发生换页 |
| GPU | 两张设备报告 NVIDIA GeForce RTX 4090，**各报告 49,140 MiB**，照录实际值；bus 01:00.0 / 02:00.0；PCIe 支持 Gen4，快照空闲时 Gen1、当前 x8（设备最大 x16） |
| 磁盘 | SQLite 实际落在新 worktree 的 `run/evidence/rollout_feedback.sqlite3`，ext4 `/dev/nvme0n1p2`，Samsung 990 PRO 2 TB NVMe，约 288 GiB 可用；`/data` 是 10 TB 机械盘，未用于这些 SQLite |
| 软件 | env_isaaclab；PyTorch 2.7.0+cu128、CUDA runtime 12.8、driver 580.173.02（nvidia-smi 报 CUDA 支持 13.0）、Isaac Sim 5.1.0.0、IsaacLab 0.48.6、omni.physx 107.3.26、RSL-RL 3.1.2、TensorDict 0.9.0、NumPy 1.26.0 |
| 线程配置 | 基准 PyTorch intra-op=1；interop=24（runtime 记录为准）；affinity=0–31。`/physics/physxDispatcher=true`，`/persistent/physics/numThreads=8`，Carb tasking=32。三者不是物理核心数，也不是实际总线程数 |

PhysX 插件 UI 与自带测试确认 `SETTING_NUM_THREADS` 是模拟线程控制入口；IsaacLab 设置 physxDispatcher。没有修改持久设置或运行 PhysX 线程 sweep：`sim.step` 只有约 4–6% iteration，消除它的理论收益已经很小。单 NUMA 无跨节点假设需要验证，未额外做 NUMA affinity 搜索；入口仍支持 `--affinity`。物理核心间性能不同，因此未来分 worker 应按拓扑划分，不能把 32 个逻辑 CPU 当成 32 个同速 worker。

Kit 启动有 `change watch ... ENOSPC` 日志，磁盘空间仍充足；未修改 inotify 或其他全局设置。真实运行、更新、保存加载和评价均通过。此机器不是完全独占，单次差异不能视为稳定加速比。

## 当前配置：一次真实 PPO iteration

| 后端/采样 | rollout s | update s | iteration s | 环境 transitions/s | PPO整体 transitions/s | 完整进程 s |
|---|---:|---:|---:|---:|---:|---:|
| L3D / fixed | 10.524 | 0.917 | 11.441 | 1185.3 | 1074.0 | 25.82 |
| L3D / adaptive | 10.575 | 0.856 | 11.432 | 1179.5 | 1074.9 | 27.84 |
| I3D / fixed | 15.419 | 0.937 | 16.356 | 801.3 | 751.3 | 30.13 |
| I3D / adaptive | 15.780 | 0.909 | 16.688 | 783.0 | 736.3 | 32.41 |

环境吞吐 = transitions / env.step 总时间；PPO整体吞吐 = transitions / (rollout+update)。仅 update 的样本等效吞吐也在 JSON 中，但不能当作端到端采样速度。Fixed/Adaptive 此处是计算成本比较，不是学习效果比较；第一次 iteration 尚无按第25次更新触发的冻结反馈。

下表是 Fixed 的**互斥 iteration 分解**；同名更细的子项另述，不重复相加。

| 项目 | L3D s / 占iteration | I3D s / 占iteration |
|---|---:|---:|
| PhysX/Isaac sim.step（384次） | 0.629 / 5.5% | 0.699 / 4.3% |
| L/I 气动运动学、计算与坐标变换 | 2.496 / 21.8% | 7.875 / 48.1% |
| 其余 apply_action（含执行器/其他载荷） | 0.779 / 6.8% | 0.798 / 4.9% |
| scene 写入及缓冲更新 | 0.360 / 3.1% | 0.385 / 2.4% |
| 终止、控制见证、SQLite（含参考查询） | 5.841 / 51.1% | 5.515 / 33.7% |
| rollout 内 reset | 0.188 / 1.6% | 0.008 / 0.0% |
| 其他环境逻辑 | 0.072 / 0.6% | 0.054 / 0.3% |
| rollout 中环境外工作（含actor/returns/日志） | 0.157 / 1.4% | 0.084 / 0.5% |
| PPO update | 0.917 / 8.0% | 0.937 / 5.7% |

L3D 的终止/证据链占主导；I3D 的气动链更大，二者瓶颈并不相同。不能因为原生 PhysX 在 CPU 上，就认定物理求解器是主瓶颈。

细分证据：

| 子项（与父项嵌套） | L3D s | I3D s |
|---|---:|---:|
| 气动 evaluate 公式 | 0.582 | 5.980 |
| 逐env控制记录 tensor→CPU/Python | 0.821 | 0.806 |
| RolloutEvaluator.observe（含其子项） | 2.937 | 2.615 |
| rollout SQLite execute | 0.060 | 0.056 |
| rollout SQLite commit（含终止提交） | 0.761 | 0.558 |
| 步骤 JSON 编码 | 0.432 | 0.403 |
| 终止/证据链扣除参考与SQL后的其余时间 | 4.961 | 4.843 |

I3D 另一次细分运行 16.556 s：马蹄涡影响计算 3.747 s、`torch.linalg.cond` 1.046 s、`solve` 0.214 s，均在 evaluate 内；`LinkKinematics3D.sample` 1.644 s，在 evaluate 外、气动链内。两次 `valid_interval` 调用位置各 12,288 次，分别 0.962 s（环境）和 1.128 s（记录器）；这是两个实际检查点的累计成本，不意味着可以删掉有效性合同。保留参数、奇异性/残差校验和所有力矩参考点，再讨论等价实现。

| Fixed 的单次调用统计 | 次数 | L3D median/P95 ms | I3D median/P95 ms |
|---|---:|---:|---:|
| sim.step | 384 | 1.566 / 2.197 | 1.783 / 2.292 |
| 气动链 | 384 | 6.494 / 6.915 | 20.891 / 22.176 |
| 终止/见证 | 48 | 118.979 / 140.541 | 110.887 / 134.098 |
| actor+critic act | 48 | 1.439 / 1.601 | 1.432 / 1.526 |
| backward | 20 | 27.237 / 27.728 | 27.507 / 28.761 |

PPO 每轮 act 共约 0.070/0.069 s，returns 0.0018/0.0016 s，process_env_step 约 0.0049/0.0048 s，TensorDict 构造约 0.0019/0.0020 s。20 次 backward 共 0.546/0.556 s，Adam step 共 0.021/0.020 s；其余更新含前向、loss、minibatch 整理和内存操作，未把张量分配/复制单独全部拆开。每轮只有一次 update/returns，不能把它的 median=P95 解读成分布稳定。

初始 reset 约 1.172/1.233 s，其中缓存 update_rows 0.115/0.127 s；剩余含轨迹生成、初态/证据构造和父类 reset，未单独分离成“纯采样时间”。B 样条投影/参考查询在 rollout 共约 0.060/0.058 s，已批量化，不是当前首选热点。

干净停止会 abort 256 个在途 episode：L/I 分别额外约 2.406/2.643 s，包含各 terminal 的 FULL 提交，**不在 iteration 时间内**。初始化、退出的成本解释了进程墙钟比 iteration 长；不能把这项一次性成本摊给每个正式 iteration。checkpoint 保存调用约 0.041 s、3.50 MB，随后原 runner 验证加载并由 harness 删除。JSONL 文件写入（collect_logs 内外的 append_json）只有数十毫秒量级；collect_logs 的几何分类等计算不等于文件写入。没有成功 episode，因此“成功 witness 插入”的独立成本没有样本，不能外推。

## 计时扰动与 CPU 扩展

L/I 关闭细分计时的 iteration 为 11.048/16.321 s；开启为 11.441/16.356 s，观测差异 +3.56%/+0.22%。对照仍保留资源监测及原 runner 计时，故测的是**额外细分插桩**的扰动，不是所有监测的总扰动。两后端开关计时的初始/最终 policy digest 和 loss 完全一致；这支持插桩未改变这两次实际更新的结果，不是跨所有配置的数值等价证明。没有重复采样足以给出置信区间。

L3D、Fixed、48 步、seed0，以下均开启同等级计时：

| env | Torch线程 | iteration s | env transitions/s | PPO整体 transitions/s | update s | 单transition环境成本 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 6.663 | 1009.3 | 922.2 | 0.464 | 0.991 |
| 256 | 1 | 11.441 | 1185.3 | 1074.0 | 0.917 | 0.844 |
| 512 | 1 | 21.186 | 1283.4 | 1160.0 | 1.731 | 0.779 |
| 256 | 4 | 10.924 | 1171.6 | 1124.9 | 0.304 | 0.854 |
| 256 | 8 | 11.726 | 1078.9 | 1047.9 | 0.208 | 0.927 |

128→512 是环境数4倍，PPO整体吞吐仅约1.26倍、纯环境吞吐约1.27倍。512 能摊薄部分固定开销，但没有接近线性扩展。4 线程单次整体约快4.7%，主要受益于 PPO update；8 线程 update 更快但环境更慢，整体反而慢2.5%。因此不能把“更多 Torch 工作线程”当作环境 Python 多核并行。没有测 I3D 的线程 sweep，也不能把 L3D 的最佳单次配置直接推广给 I3D。

## 资源利用与 GPU 收益边界

四组基准全进程平均 CPU 约139–146%（100%=一个逻辑 CPU），并非139%的整台机器；峰值约550–562%，主要包含启动等短峰。约193–194个实际进程线程存在，但主线程累计 CPU 为19–26 s；其余排名靠前的单个持续线程通常不足1 s。线程数多不等于并行计算充分。4/8 Torch线程的全进程平均 CPU 增至约219%/346%，8线程墙钟并未改善。逐核记录能观察线程迁移，不能仅从每核平均值认定某一个固定核心饱和。退出前线程表会漏掉已结束线程，故总进程 CPU 用 cpu_times，不用现存线程求和替代。

四组基准采样 RSS 峰值约3.70–3.72 GiB，相对于可用内存没有容量压力。进程累计写入约237–266 MB，持久 SQLite 约100 MB；read_bytes=0 的窗口反映页缓存命中，不代表代码没有读文件。Fixed L/I 的全进程平均实际写入分别约10.78/8.37 MiB/s，1秒采样窗口峰值约28.15/26.76 MiB/s；不是磁盘能力压测。磁盘吞吐、WAL增长和系统每核数据保存在 resources.jsonl。没有内存带宽计数器证据，**内存带宽瓶颈未判定**；单 NUMA 排除本机跨节点访问问题，但不排除缓存/内存代价。实际 SQL execute+commit 远小于 Python证据链，数据不支持优先更换磁盘或仅修 INSERT。

两 GPU 在初始调查时0%；四组运行全生命周期平均约0.8–2.3%，峰值4–8%，峰值显存约607 MiB和424–427 MiB（包括系统图形/Kit）。它们基本未承担训练计算；PPO 和环境张量确实在 CPU。不能把 Kit 的图形负载称为 PPO GPU 利用率。

只把 PPO update 理想化为零耗时，L3D 可消除 0.917 s（8.02%），速度上界 **1.087×**；I3D 为 0.937 s（5.73%），上界 **1.061×**。连 act/returns/process_env_step 一并归零也仅约1.095×/1.066×；没有计入CPU↔GPU传输、同步、启动开销，实际收益可能更小甚至为负。双GPU不能突破串行环境部分的上界。潜在空间主要是未来多CPU worker供给集中 learner，或I3D气动张量计算的异构执行；后者每秒480次状态/力传输和同步是否划算尚未测量，本轮不声称其加速比。

## 冻结评价：有效步骤、padding 与总成本

直接复用历史22个任务的验收 summaries，保存于 [historical_evaluation.json](results/unified_rl_performance_phase1/historical_evaluation.json)，未打开原 SQLite。历史 L/I development 的有效步数626/471、最后终止周期126/95；final 为748/565、最后终止125/95。历史总padding为3222 env-step，但只有3202是在**全部任务结束后**，另20为槽位提前结束至最后任务结束之间的padding，不能全部通过break省掉。

为了把padding墙钟与验证时间拆开，本轮只重放两个已有未训练development快照，保持原256周期预算和完整终止合同；新输出在独立目录，原源码只读。严格物理身份含URDF绝对路径，所以首次跨worktree直接加载被拒绝；最终通过 `--evaluation-root` 从原checkout构造同一身份，没有改快照或绕过检查。policy、终止步数和SQLite严格验证均通过，PPO/optimizer/backward禁止保护仍有效。

| 后端 | 有效env-step | 总env-step | 全结束后多余周期 / env-step | 多余环境时间 s | 有效周期环境时间 s | SQLite验证 s | 评价循环 s | Kit启动 / 建环境 s | 端到端 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L3D | 626 | 1280 | 130 / 650 | 4.354 | 5.193 | 0.164 | 9.785 | 3.650 / 2.098 | 17.042 |
| I3D | 471 | 1280 | 161 / 805 | 6.282 | 4.438 | 0.124 | 10.919 | 3.692 / 2.192 | 18.089 |

“有效周期环境时间”指仍有至少一个有效任务的整批env.step，含少量其他已结束槽位的成本，并非逐环境物理时间求和。全结束后的两项是实测可移除环境时间（4.35/6.28 s），占评价循环约44.5%/57.5%；没有把推理/循环管理的其余时间也虚报成已量得的padding。

正式默认预留7200周期×5=36000 env-step；本轮没有运行7200周期，不能将256周期的总耗时直接称为默认评价实测。若**仅作成本估算**，假定未训练快照的终止周期和已测padding每周期成本保持不变，默认预算结束后的padding为7074/7105周期，约237/277 s。它不是新策略评价的预测，也不是已实现加速。任务120 s资源限额、`max_steps=7199`始终未改。

可以在所有任务完成终止且记录落盘后退出环境循环，不缩短任何单任务允许时长。但 `receive_evaluation()` 当前强制 `environment_steps == 5*budget`，同时维护 `eval_env_steps`；必须把reserved预算、实际执行步数、有效episode步数和padding分开，保留严格完整批次验证、pending事务与Fixed/Adaptive一致性。不能只加一个break而继续伪报已执行36000步。本阶段没有修改评价循环或记账。

包含评价的成本应写为 `T_total = T_train_start + ΣT_iteration + Σ(T_eval_start + T_eval_loop + T_validate + T_eval_close + T_receipt) + T_train_close`。本轮基准1不触发评价；独立256步评价的进程成本实测17.04/18.09 s。按每25次更新摊销这份**smoke评价成本**约0.68/0.72 s/iteration（不含正式接收端再次验证）；这是成本拼接，不是已跑的25轮训练，也不是默认7200步流程。正式默认包含评价的整体吞吐目前不能直接给出实测数，以上分项与预算给出了可复用计算依据。

## 下一阶段优先级与同步多worker可行性

| 方向 | 已测可针对的成本 | 判断与收益上界 |
|---|---|---|
| 逐env Python批量化 | L/I 终止证据链扣除参考与SQL后约4.96/4.84 s；包括逐行转换、两处有效性递归、记录构造/JSON等 | **首选共享模块**。一次性提取张量块并批量检查、保留逐步完整见证；整个上述范围全消除的宽松上界为1.77×/1.42×，实际只能消除其中一部分 |
| SQLite缓冲/批量写 | rollout execute+commit约0.82/0.61 s | 较低优先级；全部SQL成本归零的上界1.077×/1.039×。原本已每控制步事务，不是每条step提交；terminal另提交。可评估批量execute/终止批次，不能把放松FULL持久化当等价加速 |
| I3D气动力实现 | 马蹄涡影响3.75 s、条件数1.05 s、运动学1.64 s | **I3D专属的重要后续项**。保留L/I公式、站数、条件数与残差保护，优先避免重复张量构造/不必要分配。只消除马蹄涡函数的宽松上界约1.29×，实际未知，不能删掉物理/稳定性检查 |
| 多核CPU与CPU/GPU分工 | 单进程主线程重，Torch8线程并未提速；PPO本身<1 s | 多进程环境并行比单独GPU learner更有空间，但IPC、复制、worker负载与SQLite写入竞争未测；不宣称2×/4×。PhysX内部不是当前首先改的模块 |
| 评价全结束后退出 | 256周期评价已测可省4.35/6.28 s环境时间 | 同时规划记账修复；默认7200早失败时可能极其重要，但现有数据只支持上述实测和明确条件下的预算估算 |

现代码**有可复用PPO构件，没有现成的多CPU worker同步采样架构**。runner直接绑定一个Kit/PhysX环境、一个在线采样器、一个单写者FeedbackStore、一个本地rollout buffer；固定 `device='cpu'`。RSL-RL PPO已有multi_gpu_cfg/梯度all_reduce，不等于本项目已支持多进程CPU采样。

最大架构障碍是把**环境/证据/采样器所有权和统一on-policy版本边界**从串行runner中拆开：每worker独立Kit和SQLite分片、全局env编号与确定性seed/reset序列，统一广播同一policy版本；收齐所有worker rollout后才能按原定义执行一次PPO更新。应保持总batch、48步、4个minibatch和5 epochs不变，避免“每worker各做一次update”改变数学合同；全局advantage归一化、采样顺序、优化器状态也要一致。中央管理Adaptive状态和冻结反馈的恰好一次接收，合并证据不能共享一个连接或丢失完整控制见证。IPC性能与公平分配尚未验证。

这些是下一阶段建议，本轮没有批量化、改变SQL事务、迁移GPU、增加worker或更改评价记账。

## 验证与复现

[实际运行校验记录](results/unified_rl_performance_phase1/verification.json)保存13个有效运行及两组插桩一致性结果。计时器已做嵌套exclusive核算、异常清理、关闭模式和JSON/CSV导出检查。实际基准全部成功更新一次、20次Adam step，开启计时的训练均记录384次sim.step；开关计时的两组policy/loss逐项一致。评价记录626/471个有效控制步、完整5个终止，原严格批次验证通过。细分统计的inclusive/exclusive和返回数据未替代任何真实物理运行。`git diff --check`及两个新增脚本语法检查通过。没有全套测试、长训练、多种子或额外物理精度消融。

```bash
cd /home/zn/IsaacLab/.worktrees/unified-rl-performance-phase1
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
# 输出目录必须不存在；每次只一次更新，默认600秒进程组硬超时。
./isaaclab.sh -p scripts/flapping_px4/benchmark_unified_performance.py \
  --backend L3D --sampling fixed --num-envs 256 --threads 1 --rollout-steps 48 \
  --output docs/results/unified_rl_performance_phase1/repeat_L3D_fixed
# 可选：--backend I3D / --sampling adaptive / --threads 4 / --num-envs 512
#       --affinity 0,2,4,6,8,10,12,14 / --no-profile / --detail
./isaaclab.sh -p scripts/flapping_px4/summarize_unified_performance.py \
  docs/results/unified_rl_performance_phase1
# 原冻结smoke快照，仅development 256周期，严格验证、不更新参数：
./isaaclab.sh -p scripts/flapping_px4/benchmark_unified_performance.py \
  --backend L3D --evaluation-root ../unified-trajectory-ppo-v1 \
  --evaluate-checkpoint ../unified-trajectory-ppo-v1/logs/unified_physx_acceptance_20261106_v1/L3D_development/evaluation_smoke_only.pt \
  --output docs/results/unified_rl_performance_phase1/repeat_L3D_eval
```

成功必须检查该目录 `completion.json: passed=true` 和 `run/status.json: completed`；仓库shell wrapper可能掩盖Python退出码，不能只看shell状态。`timings.csv`提供每项次数/中位数/P95/累计与嵌套路径，`summary.csv`提供跨配置比较。失败/诊断目录不覆盖，汇总会排除。所有性能权重仅用于本次测量、验证后删除，输出不能作为学习成果或Fixed/Adaptive科研结论。
