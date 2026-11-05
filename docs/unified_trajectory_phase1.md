# Unified Trajectory RL：第一阶段

工作树：`.worktrees/native-multibody-rl`。独立任务 ID 为
`Isaac-FlappingBot-UnifiedTrajectory-Direct-v0`。初版已提交为 `c64ad098`；
下述遗留问题修复未自动 commit/push，未训练 PPO，未注册 PPO runner。
旧 C1–C4、气动 plant、网络与奖励定义均未修改。

## 数学接口与查询缓存

`flapping_bot.path_tracking.unified_trajectory` 提供 `TrajectorySpec`、`Trajectory`、
`generate_batch`、`sample_trajectory`、`screen`。输入为完整三次夹持 B 样条节点向量、
两组控制系数、总弧长、世界位置、初始航向及种子；内部节点严格递增。

`Trajectory.query(s)` 返回 position、tangent、normal、up、heading、kappa、gamma、
kappa_prime、gamma_prime、tangent_prime 和实际三维 curvature。
`kappa=dψ/ds` 不等于 Frenet 曲率；后者为
`sqrt((cos(gamma)*kappa)^2 + gamma_prime^2)`。
normal 为水平左法向 `[-sinψ,cosψ,0]`，up 为 tangent × normal；直线处仍有定义。
世界坐标为 z-up，s 为三维弧长，所有接口都没有参考速度函数。

解析接口保持不变：航向用样条反导数，位置在包含节点的最大 0.2 m 网格上
作 8 点 Gauss 积分，末段查询重新积分，几何空间导数解析计算。

新增 `unified_trajectory_cache.TrajectoryCache`：reset 时预计算位置、未绕回航向、
kappa/gamma 及其空间导数；policy step 用批量 Torch 三次 Hermite 插值，
不再逐 env 调用 SciPy 或重复积分。采用 float64 缓存，进入环境观测时转为状态 dtype。
默认缓存间距 0.2 m，可通过 `reference_cache_spacing_m` 配置；同批环境共享长度。
局部投影仍只搜索上一进度 ±（实际位移 + 0.5 m），包含被窗口裁剪的首尾段，
只 gather 局部候选段而非全路径距离矩阵。距离预瞄与世界坐标语义保持不变。
部分 reset 仅更新指定行；非有限位置保持旧进度并交给安全终止处理。

## 默认采样分布与起始一致性

统一系数空间内直接采样，不选择任务类别或旧课程 ID。当前默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| length_m / knot_spacing_m | 60 / 10 m | 总弧长 / 机动区间节点间隔 |
| initial_straight_m | 5 m | 精确水平直飞入口长度 |
| kappa_amplitude / gamma_amplitude | 0.025 rad/m / 0.15 rad | 系数幅值 |
| axis_active_probability | 0.75 | 每个通道启用概率；未启用则全零 |
| nonzero_density | 0.8 | 系数刷新时非零概率，零值精确保留 |
| coefficient_change_probability | 0.35 | 相邻自由系数刷新概率，否则保持，形成平台 |
| transition_strength | 0.8 | 非零刷新时向新值移动的比例，小值提供缓慢变化 |
| coupling | 0.5 | 横纵创新值耦合，范围 [-1,1]，稀疏/保持掩码独立 |
| required_reversals | 空 | 默认不强制任何反向 |

每个通道先抽启用标志，再生成连续幅值、精确零值和保持段。长平台、单向、
反向及耦合机动均来自此过程。双通道同时未启用的先验概率为 6.25%，
加上零系数事件，保证精确直飞具有非零概率；自由系数多次刷新并变号也有非零概率。
这里的概率是固定采样分布参数，不是任务 ID 混合或自动难度调度。
`transition_strength=0` 保持入口之后的首个自由系数；置零刷新不经渐近衰减，
因此 `nonzero_density` 控制的是精确零系数而非仅接近零。

入口末端成为实际节点，默认前四个系数固定为零，保证整个 [0, initial_straight_m]
上 kappa/gamma 及前两阶导数均为零，之后从三次基函数平滑进入机动。
允许入口长度为零，此时前三系数为零，仍满足起点值及前两阶导数为零。
参考起点和航向取实际 reset 状态，因此与现有水平初始速度方向一致。
8 m/s 仍只是启动条件，不是速度跟踪目标；入口一致性不等于验证了稳定配平。

可显式要求 kappa、gamma 或双反向。指定通道按“启用、非零、刷新”条件生成提议，
避免先大量抽取不可能反向的零/保持通道；最终仍检查实际曲线在航程 15%–85%
内有幅值超过采样幅值 8% 的正负两侧。零平台起点不再误算为反向。
不可能的入口/机动长度组合提前拒绝，其他筛选最多 256 次，失败明确抛出原因。
种子由总 seed、env ID、reset 次数派生，完整参数保存在 `env.task_records`、
完整解析轨迹保存在 `env.paths`，可通过 `TrajectorySpec(**record)` 回放。
新采样分布改变了 seed 到新任务的映射；历史任务按其完整参数回放不变。

## 几何限制与环境定义

筛选上限保持 K=0.06 rad/m、G=0.4 rad、D_K=0.04 rad/m²、D_G=0.2 rad/m；
高度相对 env 原点为 1–60 m，非局部弧长间隔 8 m、近交距离 0.5 m。
检查分段多项式极值、连续性、保守高度界及 KD-tree 非局部近交。
这些仍是工程示例值，不是旧 C4 包线或 L/I plant 可飞性结论。

保留 CPU-native measured-wing PhysX / DeLaurier、480 Hz physics / 60 Hz policy、
扑频/方向舵/左右尾舵四通道动作。旧纵向/空间 stage 为 None。
39 维观测和 2/5/10 m 距离预瞄不变；奖励仍为新增最大已达航程加权奖励、
路径误差与动作变化代价、完成奖励和安全失败惩罚，无目标速度。
终点 0.5 m 容差与独立 120 s 资源超时不变；超时不是成功。
安全检查仍包含高度、路径偏离、姿态、角速度和非有限状态。

## 本轮验证结果

18 项定向测试通过：原有数学/筛选/回放测试，加上可配置平滑入口、采样控制与覆盖、
缓存解析误差/空间导数、反向附近局部投影与预瞄、部分缓存 reset 和禁止热路径调用 SciPy。

默认配置固定种子 0–255，共 256 条，全部一次采样通过。以下是实际曲线统计，
标签允许重叠，不用于生成任务或调度：

| 几何特征 | 数量 / 256 | 判据 |
|---|---:|---|
| 精确直飞 | 29 | 两通道全零 |
| 直飞/近直飞 | 31 | 两通道峰值均不超过各自采样幅值 5% |
| 单向机动 | 125 | 有效幅值超过 8%，两通道均不出现有效正负两侧 |
| 持续机动 | 79 | 至少一个通道非零平台 ≥10 m；归一化导数 <1e-6 /m |
| 反向过渡 | 56 | 任务内部至少一个通道满足反向判据 |
| 双反向 | 5 | 两通道均满足反向判据 |
| 三维耦合 | 103 | 至少 10% 的机动段同时存在有效横纵几何量 |

CPU 基准：256 env、24 Torch threads、一次预热后 8 次计时取中位数，
投影窗口 0.7 m、三点距离预瞄，进度覆盖 15–45 m：

| 项目 | 实测 |
|---|---:|
| 修改前分布的轨迹生成 | 0.544 s / 256 |
| 修改前逐 env 查询 | 96.18 ms / 256 / step |
| 新默认分布轨迹生成 | 0.368 s / 256 |
| 新缓存构建 | 0.134 s / 256 |
| 同一批新路径，旧查询实现 | 93.42 ms / 256 / step |
| 同一批新路径，批量缓存查询 | 0.848 ms / 256 / step |

同路径查询约提速 110 倍。这是参考查询微基准，**不包含 PhysX 步进或 PPO**；
256 条生成/缓存计时也不包含机器人 reset、环境世界位置平移重建和 Kit 启动。
默认配置下，每条路径 257 个离网格点与解析接口比较，最大位置误差
3.61e-8 m、切线分量误差 6.96e-9；kappa/gamma 及其一阶导数误差 ≤2.43e-15。
配置更短节点间隔或更粗缓存间距时应重新检查插值误差，不能外推这些数值。
完整数据：[fixes_benchmark.json](results/unified_trajectory_phase1/fixes_benchmark.json)。

一次 4 env × 12 step（0.2 s）CPU-native smoke 通过，随后已停止仿真。
4 env 初次 reset（不含 Kit/环境构造）16.57 ms，参考切线与实际初速度单位向量
最大分量差 5.97e-8。部分/全部 reset、有限状态/观测/奖励、距离预瞄均通过。
单独生成显式双反向轨迹，在航向反向 s=48.41144 m 的前/中/后注入状态，
检查局部投影、预瞄、奖励公式和未完成判定；kappa 依次约为
+0.001947、0、-0.001782 rad/m。端点完成、低高度安全失败与超时断言仍通过。
**注入状态只是接线检查，不是真实闭环飞行成功。**
详见 [fixes_runtime_summary.json](results/unified_trajectory_phase1/fixes_runtime_summary.json)。
系统仍有文件监视器配额警告，但本轮环境正常构造、运行、保存和退出。

## 复现与后续边界

在本工作树激活环境，输出路径须不存在：

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
export PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
./isaaclab.sh -p -m pytest tests/test_unified_trajectory.py tests/test_unified_trajectory_cache.py -q
./isaaclab.sh -p scripts/flapping_px4/benchmark_unified_trajectory.py --output /tmp/unified_benchmark_repro.json
./isaaclab.sh -p scripts/flapping_px4/run_unified_trajectory_smoke.py \
  --num-envs 4 --steps 12 --seed 7 --output /tmp/unified_fixes_smoke_repro
```

wrapper 可能吞掉 Python 失败退出码，须确认 `UNIFIED_SMOKE_PASSED` 与 JSON `passed=true`。
初版[代表性轨迹图](results/unified_trajectory_phase1/trajectories.png)、完整任务参数及
`runtime_summary.json` 保留为历史初版结果；它们不是修复后默认分布的覆盖统计。

本轮已具备进入第二阶段**选定 L/I plant 的可飞性验证**所需的任务生成、查询与接线条件。
尚未验证物理包线、完整航程闭环跟踪、稳定入口配平或 256 env 全环境吞吐。
后续仍需确定几何上限、初始状态范围、预瞄和安全阈值；本轮未开展这些研究，
未运行 PPO、自动课程或旧 C4 性能对比。
