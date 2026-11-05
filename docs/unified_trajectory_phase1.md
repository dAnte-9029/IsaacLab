# Unified Trajectory RL：第一阶段

工作树：`.worktrees/native-multibody-rl`。新增独立任务
`Isaac-FlappingBot-UnifiedTrajectory-Direct-v0`，未注册 PPO runner，未训练或提交。
旧 C1–C4 的实现与配置不变；只在项目任务注册文件新增一个 ID。

## 数学接口

`flapping_bot.path_tracking.unified_trajectory` 提供 `TrajectorySpec`、`Trajectory`、
`generate_batch`、`sample_trajectory`、`screen`。输入是三次 B 样条完整夹持节点向量、
两组控制系数、总弧长、世界位置、初始航向及随机种子。内部节点要求严格递增，
保证曲率参数和坡度 C2 连续；重复系数可形成持续恒定机动。

`query(s)` 返回 position、tangent、normal、up、heading、kappa、gamma、
kappa_prime、gamma_prime、tangent_prime 和实际三维 curvature。
`kappa=dψ/ds` 不等同于 Frenet 曲率；后者为
`sqrt((cos(gamma)*kappa)^2 + gamma_prime^2)`。
normal 是水平左法向 `[-sinψ,cosψ,0]`，up 是 tangent × normal，
在直线处也有定义，并非要求非零曲率的 Frenet 主法向。

航向使用样条反导数精确计算；位置在包含所有节点的最大 0.2 m 网格上作
8 点 Gauss 积分，查询时对末段重新积分，避免线性位置插值与解析切线不一致。
空间导数均解析计算。CPU reset 批量接口支持不同节点和长度，每条路径独立计算；
当前未做大批量吞吐优化。所有接口都没有参考速度函数。

## 直接连续采样与筛选

默认航程 60 m、节点间隔 10 m，系数幅值分别为 0.025 rad/m、0.15 rad。
`transition_strength∈[0,1]` 将常值系数与独立随机系数混合；
`coupling∈[-1,1]` 控制横纵系数的同向/反向耦合。没有任务 ID 或类别抽样。
`required_reversals` 可为空、仅 kappa、仅 gamma 或两者；默认两者。
零过渡强度且无强制转换时可采样恒定机动；显式零系数表示精确直飞。

要求的转换必须在完成航程 15%–85%（默认 9–51 m）内出现，且函数在该区间
正负幅值均超过对应采样幅值的 8%，不是仅检查控制系数。
最多拒绝重采样 256 次，失败抛出原因，不静默退回简单任务。
每个 env 的任务种子由总 seed、env ID、该 env reset 次数派生。
完整节点、系数、位置、航向和种子保存在 `env.task_records`，完整轨迹保存在
`env.paths`；运行脚本将首批和末批参数写入 JSON，可用 `TrajectorySpec(**record)` 重建。

| 筛选参数 | 默认值 | 单位 |
|---|---:|---|
| K | 0.06 | rad/m |
| G | 0.4 | rad |
| D_K | 0.04 | rad/m² |
| D_G | 0.2 | rad/m |
| 相对 env 原点高度 | 1–60 | m |
| 非局部近交距离 | 0.5 | m |
| 非局部弧长间隔 | 8 | m |

四项几何上限检查分段多项式极值，不仅检查网格。高度采用单位切线的 Lipschitz
界作保守裕量；非局部近交使用 KD-tree 和网格误差膨胀距离，可能保守拒绝临近路径。
这些是工程示例值，不是旧 C4 包线，也不是 L/I plant 的可飞性结论。

## 环境接入

继承现有 CPU-native measured-wing PhysX 的场景、执行器和力调用，保持
480 Hz physics / 60 Hz policy、扑频/方向舵/左右尾舵四通道接口。
本工作树现有 plant 是 DeLaurier；没有修改或引入 L/I 气动模型。
禁用旧 PureRL 观测/奖励，旧纵向/空间 stage 均为 None。

39 维独立观测包含机体系速度、角速度、重力、路径误差、参考切线、
2/5/10 m 距离预瞄、四个几何量、翼相位、动作、实际扑频、实际尾舵角和剩余航程。
预瞄距离可配置，观测维度随预瞄点数调整。路径投影限定在上一进度附近，
搜索窗口由实际位移加 0.5 m 裕量决定，避免跳到远处相邻分支。

奖励为新的最大已达弧长增量乘 `exp(-error²/2²)`、按时间积分的路径误差代价、
动作变化代价、完成奖励与安全失败惩罚。回退再前进不会重复赚取同一段进度。
不奖励停留生存、不设置目标速度。8 m/s 仅为可配置初始条件。
完成要求到达路径末端 0.5 m 内且误差 ≤0.5 m；跌落、过高、过大偏离、
倾斜、角速度超限或非有限状态触发安全终止。120 s 仅为独立资源超时，
超时单独标记 truncation，不等于完成，无 140 m/20 s 规则。
有限超时仍会限制一次 rollout 可行驶的距离；本阶段没有证明任意策略能在超时内完成。

## 验证及复现

定向数学测试：10 passed（含 float32 reset 位置 JSON 导出与回放）。覆盖直线/恒曲率螺旋解析解、位置/切线/空间导数、
节点连续性、局部正交标架、七种代表机动、四个上限拒绝、高度/非局部近交、
重现性、局部投影与端点预瞄、复杂转换位于任务内部。
图见 [代表性轨迹](results/unified_trajectory_phase1/trajectories.png)，同目录含 SVG 和完整参数 JSON。

在该工作树运行（输出目录须不存在）：

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
export PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
./isaaclab.sh -p scripts/flapping_px4/run_unified_trajectory_smoke.py \
  --num-envs 4 --steps 12 --seed 7 --output /tmp/unified_trajectory_smoke_repro
```

wrapper 可能吞掉 Python 失败退出码，须确认输出 `UNIFIED_SMOKE_PASSED` 且
`summary.json` 中 `passed=true`。运行包含部分/全部 reset、真实短步进，以及
注入端点/低高度/超时状态的任务接线检查。注入完成不是物理飞行完成证据。

后续需基于选定 L/I plant 确定 K/G/D_K/D_G、持续与过渡机动可飞性、
初始状态范围、预瞄距离、偏离/姿态/角速度安全阈值以及奖励权重。
本阶段不包含 PPO、自动课程、长时间航迹跟踪或旧 C4 性能比较。

实际运行结果：最终 4 env × 12 policy steps（0.2 s）通过，输出
`UNIFIED_SMOKE_PASSED`，进度、状态、39 维观测与奖励均有限。
首个环境中途单独 reset，其余环境任务和进度未被覆盖；端点完成、低高度安全失败、
全部 reset 后资源超时的接线断言全部通过。
结果见 [runtime_summary.json](results/unified_trajectory_phase1/runtime_summary.json)。
首次沙箱内启动崩溃；沙箱外短运行暴露了 float32 JSON 导出问题，修复并通过导出/回放测试后，
最终短运行正常保存结果并退出。所有运行都使用固定动作，无 PPO 或长时间仿真。
系统仍报告文件监视器配额警告，但最终环境构造和验证通过。
