# Unified RL 阶段 3：I3D 气动执行优化

2026-11-07。工作树 `.worktrees/unified-rl-performance-phase3`，分支 `perf/unified-rl-performance-phase3`，起点 `bab086dd631aac28eab46122ab9a933bf7661471`。未 commit/push；原工作树、历史 checkpoint、阶段 1/2 产物未改动。

**实测 I3D Fixed 完整 iteration 14.159 → 12.571 s，1.126×；Adaptive 13.410 → 12.634 s，1.061×。** 最大收益来自马蹄涡端点向量和模长复用。真实 PhysX 重放和三组 PPO 数值/证据对照通过。这里的 iteration 是一次 rollout + 原 PPO update，不是长训练性能或收敛结果。

## 分析与最终实现

完整阅读 `control_aero_3d.py`、`control_aero_kinematics.py`、`straight_flight_env.py::_compute_control_3d_wrench()`、`ullt_geometry.py`、`ullt_twist.py` 及 `ULLTGeometry` 派生几何定义。保留原实现的实际物理定义：直马蹄涡、线性附着环量、局部分离载荷、relative-only added mass；本阶段没有引入新的非定常尾迹状态或改变原有模型。

1. **马蹄涡：** 原函数在 bound 项已构造 `ra/rb` 并求模，两个 wake ray 又各做一次同样的相减和模长。现在每次调用仅构造一对端点向量、一对模长，在 bound/ray 中复用。向量减法、叉乘、求和、除法及最终 `bound + ray(b) - ray(a)` 运算次序保留；不消除任何诱导项。`keepdim` 改为视图增加末维，不改变模长归约轴。
2. **I3D 运动学：** 同一次 `sample` 内复用世界系 3/4 弦点、变换后的扭转速度与角速度，避免重复的属性计算和 `einsum`。不跨子步保存姿态、相位、速度、变形几何。左右翼顺序、actual-link COM 速度、关节符号、附加质量输入及载荷施加参考点均未改。L3D 保留原计算分支；共享函数增加的分支判断仍可能有微小开销。
3. **条件数/求解不改：** 每子步 256 个 20×20 float32 系统，仍调用默认 2-范数 `torch.linalg.cond`，通过检查后用 `torch.linalg.solve`，原 residual 检查保留。cond 使用奇异值信息，原 solve 使用 LU；用 SVD 重建解会改变算术并增加工作量，没有证据说明能等价提速。没有引入替代矩阵指标、额外分解或更宽松阈值。

固定拓扑的 20×20 eye/mask 很小；本轮没有为其增加跨调用缓存。扭转几何含相位依赖，不能缓存为静态翼型。未改站数、35°扭转、dtype、clamp、奇异性阈值、1e6 条件数阈值、残差容差、480/60 Hz PhysX、多体/执行器、39D/4D、奖励、采样或 PPO 定义。未采用 torch.compile/C++，没有新增编译、部署或首次编译成本。

## 真实 PPO 性能

配置与阶段 2 相同：CPU-native PhysX、256 env × 48 控制周期、8 子步/周期、Torch intra-op=1、seed=0、原网络与 5 epochs × 4 minibatches。最终每组一次 iteration；此前仅马蹄涡改造额外做了一次 I3D Fixed 归因运行。总计四次有界 PPO iteration，没有正式训练或评价预算搜索。

| I3D Fixed 指标 | 阶段 2 | 阶段 3 |
|---|---:|---:|
| 气动链 s | 8.2323 | 6.9972 |
| horseshoe influence s | 3.9038 | 3.0552 |
| 运动学采样 s | 1.6845 | 1.5609 |
| 条件数 s | 1.0856 | 0.9546 |
| rollout s | 13.1827 | 11.7403 |
| PPO update s | 0.9761 | 0.8303 |
| iteration s | 14.1588 | 12.5706 |
| transitions/s | 867.87 | 977.52 |
| 全进程 s | 27.6937 | 25.7914 |

| 后端/采样 | rollout s 前→后 | update s 前→后 | iteration s 前→后 | iteration 加速 | 全进程 s 前→后 |
|---|---:|---:|---:|---:|---:|
| I3D Fixed | 13.183→11.740 | 0.976→0.830 | 14.159→12.571 | 1.126× | 27.694→25.791 |
| I3D Adaptive | 12.502→11.795 | 0.908→0.838 | 13.410→12.634 | 1.061× | 28.839→27.621 |
| L3D Fixed | 7.724→7.997 | 0.950→0.909 | 8.674→8.906 | 0.974× | 22.414→23.438 |

I3D Fixed 墙钟下降 11.22%，Adaptive 下降 5.79%。L3D 单次 iteration 增加 2.68%，没有观察到大幅退化，但单次数据不能证明不存在小幅退化或给出统计显著性。

**不能把全部计时差归因于优化。** 未修改的条件数、PPO update 也变快，说明机器负载/运行噪声存在。阶段 2 数字来自原始产物，未以当前机器重跑基线或挑选最佳轮次。仅马蹄涡候选的真实 Fixed iteration 为 13.3928 s、horseshoe 为 3.3553 s；最终的进一步下降不全是运动学改造收益。

为分离局部贡献，在相同 256 环境输入、同进程单线程上交替原/新函数，6 块 × 30 调用，不进行参数搜索：

| 核函数 | float32 原→新 ms/调用（块下中位数） | 加速 | float64 加速 |
|---|---:|---:|---:|
| horseshoe | 13.430→10.913 | 1.231× | 1.086× |
| sample | 5.211→5.025 | 1.037× | 1.032× |

核函数计时使用规定姿态/相位分布，不能代替真实 PPO 吞吐。首个计时 float32 调用 horseshoe 原/新为 16.716/12.060 ms，sample 为 5.486/5.059 ms；输入准备已调用 sample，因此此项不是冷启动测量；没有 JIT 编译或长期摊销门槛。这组数据支持马蹄涡是最大贡献项，运动学只带来较小收益。

## 数值与证据验收

验收标准为**零容差**，未放宽任何阈值。

- 每后端独立新旧真实 PhysX 重放：256 环境、相同 seed/初态、48 个控制周期、384 个子步。动作沿用阶段 2：`0.35*sin(arange(256*4)*0.013 + step*0.17)`。
- I3D 每子步以当前真实状态调用阶段 2 参考气动/运动学，比较影响矩阵、bound、诱导速度、环量、alpha、分离权重、每站 force/moment、condition/residual，以及两翼完整几何、full/relative motion。27 个字段各比较 384 次，全部逐值相等，逐项最大绝对/相对误差为 0，全部误差分位数为 0。正常重放无异常分支触发。
- 独立进程新旧 NPZ 的动作 `[48,256,4]`、气动力/力矩及关节/机体状态 `[384,256,35]`、观测/奖励/done/timeout/参考误差 `[48,256,46]` **字节一致**。子步采样在施力后、积分前；控制周期后的状态另由 interval/SQLite 覆盖。
- L3D 同样完成 384 子步运动学/载荷同状态比较和独立进程完整重放，全部一致。
- I3D Fixed/Adaptive、L3D Fixed 的初始/最终 policy digest、20 次 Adam step、全部 loss 与 mean reward 精确一致。没有宣称长期学习收敛或多 seed 统计等价。
- 三组 PPO 的每条 SQLite 控制数据、初态、轨迹、终止合同、终止序列/统计与阶段 2 一致。再从 SQLite 原始 interval 经 `RolloutEvaluator.observe()` 独立重算全部终止结果；manual_reset 仅在记录末尾按原合同重建。两后端新旧动作重放数据库同样独立验证。

| SQLite 数据 | steps | episodes/terminals | success witnesses |
|---|---:|---:|---:|
| I3D Fixed（各版本） | 12288 | 257 | 0 |
| I3D Adaptive（各版本） | 12288 | 257 | 0 |
| L3D Fixed（各版本） | 12288 | 291 | 0 |
| 各后端确定性动作重放（各版本） | 12288 | 256 | 0 |

见证表为空的事实保留，不将 manual_reset 解释为成功。比较排除 run UUID/policy-version 字符串及绝对路径等运行来源；物理源代码身份在下节单独严格核查。

定向测试 **43 passed**：原有 11 项气动测试及新增 32 项，覆盖 float32/float64、零速/低速/门限附近/反流、self principal value、bound/wake 奇异性、构造的 1e6 条件数附近及非有限系统、残差拒绝、active circulation 下未定义 wake。边界测试给原/新实现相同几何或受控线性系统，原 condition/solve/residual 判断均实际执行；残差拒绝测试故意注入错误求解结果，确认原拒绝分支仍触发。核函数工具另验证两种 dtype 下的实际翼几何/运动学精确一致。

## 资源、身份与限制

| 后端/采样 | 峰值 RSS MiB 前→后 | 全进程采样平均 CPU% 前→后 | Kit 启动 s 前→后 | build_env s 前→后 |
|---|---:|---:|---:|---:|
| I3D Fixed | 3810.15→3792.45 | 126.35→125.11 | 3.755→3.730 | 4.471→4.179 |
| I3D Adaptive | 3801.67→3797.89 | 128.43→124.73 | 3.767→3.786 | 4.649→4.610 |
| L3D Fixed | 3785.75→3786.34 | 128.78→130.50 | 3.692→4.065 | 4.434→4.987 |

CPU% 是进程树按约 1 s 采样，100% 对应一个逻辑核，包含启动及关闭；不是 rollout 专用利用率或整机占用率。当前三组采样峰值约 505–523%，启动/后台原生线程可使用多核，即使 Torch intra-op=1。全进程包含初始化、完整证据落盘、checkpoint save/load 检验和关闭；没有把成本移到漏计阶段。

没有新增持久缓存，因此无新增 reset/相位更新时的缓存失效问题。临时量仅在单次调用存活；两个 float32 模长数组共约 0.78 MiB，替代后续重复求模，而非随训练步数积累。实测进程峰值内存没有明显增加，但本阶段未做长时间泄漏测试。算法仍产生 `B×20×20×3` 广播临时张量；不能声称已达到底层效率极限。

`physical_v2` 来源清单保留两个修改文件的真实新指纹；汇总逐项验证两版 source_snapshot 的实际字节与各自指纹一致，差异集合严格限定为 `control_aero_3d.py` 和 `control_aero_kinematics.py`，其余配置和物理来源一致。可将本次实现称为 **phase3-exact-reuse-v1**，这是经上述有界验证的新计算版本，不是旧代码身份。未改 checkpoint 兼容校验，也未用历史 checkpoint 冒充新版本。

最初沙箱内 Kit 在 cuInit 崩溃，未发生 PPO 更新；该失败目录保留，不计性能，遗留崩溃进程已清理。成功测试通过沙箱外驱动访问运行，仍为 CPU PhysX。Kit 还记录了 inotify watch 资源不足警告，仿真和验收均完成；这不是磁盘容量耗尽。没有更换 Conda/Python/Isaac Sim 或安装依赖。

## 复现与交付

[最终机器可读汇总](results/unified_rl_performance_phase3/final/summary.json)、[定向测试日志](results/unified_rl_performance_phase3/tests.log)、[交替核函数计时](results/unified_rl_performance_phase3/kernels.json)。原始目录包含 NPZ、SQLite、源码身份/快照、计时调用树、CPU/内存记录、日志和 completion 状态。内部影子对照增加计算量，因此 **replay 的 env_step_s 不作为性能优化结果**。

前提：阶段 2 参考工作树及 control-oriented-3d-wing-aero 的原资产/规则文件仍在相同父目录。一次入口执行定向测试、三组各一次 PPO、新旧重放、核函数计时和严格汇总；必须使用新的输出目录：

```bash
cd /home/zn/IsaacLab/.worktrees/unified-rl-performance-phase3
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
./isaaclab.sh -p scripts/flapping_px4/run_unified_phase3.py \
  --output docs/results/unified_rl_performance_phase3_repeat
```

只重算已有数据：

```bash
./isaaclab.sh -p scripts/flapping_px4/summarize_unified_phase3.py \
  docs/results/unified_rl_performance_phase3/final
```

wrapper 不可靠地传播子 Python 退出码，所以入口显式检查测试状态文件、各运行 completion 及最终 summary.passed。最终报告数据来自 `final/`；首次仅马蹄涡候选及先行影子重放单列保留，没有混为最终结果。

## 下一步判断

**VERIFIED：** 在本次有界负载中保持原数值结果与异常语义，Fixed/Adaptive 分别实测 1.126×/1.061×。**SUPPORTED：** 最大局部收益来自重复端点向量/模长消除；小规模运动学复用收益有限。**UNRESOLVED：** 长训练稳态吞吐、内存长期行为与多进程 scaling。

剩余 Fixed 气动链约 6.997 s，占 iteration 55.7%；horseshoe 3.055 s、运动学 1.561 s、cond 0.955 s，另有 `_get_dones` 2.632 s。继续单进程要获得明显收益，可能需要更大范围的融合/native kernel，并重新验证浮点运算及拒绝边界；本轮没有证据支持这种复杂度一定有收益。

建议下一阶段**测试少量多进程 CPU PhysX worker 的可行性**，先测独立仿真吞吐、IPC 和证据写入成本，再决定是否整合 PPO。它是尚未验证的方向，不能预报加速倍数。本阶段到此停止：没有开发多进程/分布式 PPO、迁移 GPU PhysX、启动正式训练或自动进入下一阶段。
