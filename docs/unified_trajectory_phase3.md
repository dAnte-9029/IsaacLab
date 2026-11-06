# Unified Trajectory 第三阶段：在线采样机制

独立开发位置：`.worktrees/unified-trajectory-online-v1`，分支
`feat/unified-trajectory-online-v1`，基于 native HEAD `d343f74d`。
离线来源 worktree HEAD `4303c1ad`。开始时两者均无已跟踪文件修改，
未跟踪 PID、eval 和论文文件未触碰。未自动提交或推送。

**VERIFIED（机制层面）**：25 项定向检查通过。使用真实 88 条 v2 记录初始化，
后续反馈全部是确定性人工构造的 rollout 测试数据，不是物理仿真成功。
没有训练 PPO、启动 PhysX、改动气动模型、奖励或网络。

## 模块与信任边界

- `path_tracking/trajectory_evidence.py`：只读原始 JSON 快照，返回副本；保留完整原始
  物理条件、轨迹、评价合同、控制见证。精确匹配才返回 VERIFIED FEASIBLE；
  同任务同物理条件但合同不同返回 RELATED EVIDENCE / TARGET UNVERIFIED。
  NOT FOUND 不排除任务。不同 L/I、DeLaurier、扭转配置不能继承认证。
- `trajectory_feedback.py`：`RolloutEvaluator.begin/observe` 绑定完整任务、初始状态、
  plant、合同、policy version 和实例 ID。逐控制步记录命令/执行器见证，拒绝缺步、
  重复终止和直接 status 标签。只有终止评价器可以写 journal；安全失败优先于完成，
  完成优先于超时。未完成的手动 reset 丢弃未终止 episode，不伪造失败。
  `FeedbackStore.query` 只按完整任务、plant、初始状态和合同查成功；后续失败仍追加。
- `trajectory_capability.py`：有限对称邻域内的当前版本完成/失败/超时数、成功率、
  学习进展及历史掌握标记。当前窗口 32 条；不同版本不混算当前成功率。
  初始状态完整保留，能力统计是同配置 reset 分布的边际估计，未对初始状态再平滑建模。
  它不是物理可飞概率，也不能提升邻居的证据标签。
- `online_trajectory.py`：独立连续混合采样器，不引用任何 C1–C4 ID 或自适应逻辑。
- `trajectory_evidence_geometry.py`：仅从第二阶段移植峰值/持续距离等几何提取函数；
  没有移植允许直接追加标签的旧 API，也没有移植逐分量“更小即更容易”的启发式。

真实入口写 `rollout_feedback.jsonl`；测试显式使用 `simulated=True`，只能写
`simulation_feedback.jsonl`，查询返回 SIMULATED SUCCESS / NOT PHYSICAL EVIDENCE。
采样器拒绝未经本 journal 评价、篡改或重复的反馈。原始 v2 永远只读；没有迁移脚本
写入在线目录的路径，重读/重启不会覆盖记录。journal 使用单写者、追加和 fsync；
不支持多进程同时写同一文件。进程中 API 边界不等于防恶意 Python/文件篡改的安全系统。

## 采样定义与工程初值

固定混合权重为 base / retention / exploration = **0.40 / 0.25 / 0.35**。
配置强制 base 下限 0.20、retention 和 exploration 正权重。

base 完全调用原始连续采样器。exploration 每次从 q0 生成 16 条新候选，按离线几何
邻近参考、未知度、正学习进展和遗忘状态加权抽取。几何距离对称，包含曲率、坡度、
两项变化率、航程、横/纵持续距离、净高度、高度跨度、耦合距离与积分。
离线参考只使用**相同 backend** 的成功记录；这些仅是参考，不要求与目标合同等价，
也绝不授予目标可飞标签。默认 native DeLaurier 不会获得 L/I 成功先验加成。
60 m 与历史 130–200 m 明确保留为不同任务。L/I 物理实现没有合并进此 worktree。

retention 使用历史成功的完整系数形状，给曲率/坡度各自施加 [0.9,1.1] 连续幅值扰动；
仍过原几何筛选，扰动不合法则尝试原 anchor，再不合法回退 q0 并记录来源。
只保持当前共享航程/合同/plant 的 anchor；环境批缓存仍采用固定共享航程。
独立采样器可通过 SamplingConfig 配置其他航程，但此次没有引入逐 env 可变缓存长度。

每 8 条终止反馈更新一次；新条件分布以 0.20 凸混合并入旧分布。版本改名本身不改变
采样分布；每次批更新的版本由该批最后一个终止记录确定，旧批版本冻结。
未来 runner 应在版本边界排空在途 episode，避免同一批混合旧/新版本影响优先级；
每条记录自己的能力版本仍准确。固定候选集合上每次总变差变化不超过
`(0.25+0.35)*0.20=0.12`。实际 retention 采用历史分布凸混合抽样。

探索原始分数下限 0.15，重复失败无进展只降低权重；保留未知度项。
邻域默认半径 1.0；几何尺度顺序见 `OnlineConfig.scales`。
四情景测试使用半径 **0.08** 以隔离三个探针，不宣称这是训练最优值。
上述所有数值均为可配置工程假设。

覆盖保证：q0 始终占 40%；默认两个通道都关闭的先验概率为 6.25%，因此在默认
可接受起始高度下，精确直线至少有 **2.5% 总体基础采样机会**，不依赖训练成功率。
retention 分支固定占 25%；首个成功批后，其历史 anchor 分布至少分到该分支的 20%，
即总体 5%，后续平滑增加（前提是 anchor 在当前几何条件仍合法）。
没有历史成功时该分支回退 base。未知/困难区域只要在 q0 支持内，永远不会被永久排除。

## 四个模拟情景结果

[完整数值](results/unified_trajectory_phase3/mechanism.json)。下表是固定三个探针
（简单/困难/未知）上的**条件采样质量诊断**，不是连续空间某条精确轨迹的抽中概率。
`distribution()` 将探针当成候选支持；实际 retention 还包含连续扰动核和几何回退，
因此这些数值也不等于完整空间的区域积分或实际 rollout 频率。

| 情景 | 简单 | 困难 | 未知 |
|---|---:|---:|---:|
| 初始 | 35.96% | 31.10% | 32.94% |
| 困难连续失败 32 次 | 37.47% | 28.24% | 34.29% |
| 随后困难成功 8 次 | 34.99% | 33.03% | 31.99% |
| A 掌握后 B 失败 32 次 | 28.57% | 45.57% | 25.86% |
| 未知区域再失败 64 次 | 29.42% | 50.10% | 20.48% |
| 简单成功后持续注入复杂任务进展 | 38.29% | 41.21% | 20.50% |

A 在困难邻域的当前成功率先达 100%，B 的为 0%；32 条历史**模拟**成功见证仍存在。
新成功立即追加见证，采样优先级在下一批更新修正，不会被旧失败永久封锁。
探针 base 每项下限为 13.33%；简单任务即使曾成功仍持续复习。
测试还验证三种采样分支实际返回合法完整 TrajectorySpec 和来源，部分 reset 不改变
其他 env 的任务、缓存或在途反馈；默认关闭与原种子采样一致。

## 环境接入与验证边界

`UnifiedTrajectoryEnvCfg.online_sampling=False` 为默认。启用时必须提供
`online_rules_path`、`online_feedback_directory`、`online_policy_version`；
`online_sampler_config` 可传 OnlineConfig。完整 plant 配置和实际初始 root/joint/phase/
frequency/wind/actuator 状态由环境采集，合同由当前终止配置构造，不能套用 phase2 合同。
任务仍在 `task_records`，来源另存 `task_sampling_sources`；partial reset 只绑定对应行。
训练端更新 `env.online_sampler.policy_version`，之后新 episode 使用新版本。

本轮接口测试用张量 plant fixture 执行**实际环境 reset 与 done 方法**，不启动 Isaac Sim。
数值生成器和缓存原有测试同时通过。没有声称 PhysX 闭环 smoke 或真实新飞行成功。

反例：坡度全零时，将一条反向轨迹的所有曲率系数乘 -1，其上述特征向量完全相同，
但先左后右与先右后左互换。邻域会混合这两类经验；完整轨迹身份仍不同，不能相互认证。
没有为掩盖这一局限任意增加特征。

接口足以接下一轮固定/自适应等预算 PPO 对照：同一任务表示、默认固定分布、可选
自适应 reset、版本化终止反馈均存在。**这只是机制与接口就绪**；仍需选定共同 plant、
reset 分布、合同、预算及版本更新边界。未实现 runner，也未验证长训练吞吐。
当前实现保留并重放历史批计算分布，数据量增长时成本增加；本轮没有宣称可直接承受
大规模长期训练吞吐。L/I 对照还需在有相应物理实现的工作树接入这些独立模块。

复现（在独立 worktree 中，要求已有第二阶段数据；无需 PhysX）：

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
./isaaclab.sh -p -m pytest tests/test_online_trajectory.py tests/test_unified_trajectory.py tests/test_unified_trajectory_cache.py -q -s
```

`UNIFIED_RULES` 可覆盖原 v2 路径；`UNIFIED_REPORT` 可指定不存在的新 JSON 输出路径。
wrapper 可能吞退出码，应确认 pytest 输出 `25 passed`。测试临时 journal 留在 pytest 临时目录，
不向真实证据目录写入。
