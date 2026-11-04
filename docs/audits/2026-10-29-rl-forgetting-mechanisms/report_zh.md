# 扑翼 RL 遗忘机制与下一阶段路线选择

实际分析日期：2026-11-04。保留任务指定的目录日期。本报告只讨论历史 DeLaurier CPU-native plant。

**结论：优先研究“功能保持约束的快慢分离策略”，首先改变策略更新的约束对象，而非继续串行课程 PPO，或立即引入 MoE/GRU。** 已有证据最支持：在当前采样状态上局部下降的 PPO 更新，仍可累计改写已经成功的控制函数；旧任务采样配额、局部 KL、参数更新范数和软 action-MSE 都不等于功能保持。C4 的直接闭环故障已定位到协调尾舵策略；C2c 还涉及扑频请求、尾舵协调及失稳后的限幅，不能统一归结为一个坏舵面。

下一阶段唯一优先验证：从 D-50 做一次短程、同起点、同训练分布、同训练用成功行为锚点的比较，**软行为损失 vs 显式限制尾舵函数漂移**。不在本轮执行。若约束确实保持行为却仍失去闭环能力，才能否定“锚点上功能漂移足以解释失败”的机制，转向覆盖不足或可组合模块的必要性；不能先长期训练多个新架构再挑最好者。

## 1. 范围、实物与证据分级

- 实际工作区：`/home/zn/IsaacLab/.worktrees/native-multibody-rl`，分支 `feat/native-multibody-rl`，HEAD `a7b3e08e0c1e2271ba5131b9a5837af29d82f8e2`。与用户给出的远端历史 SHA 一致；本轮没有联网查询 Git 远端，也没有发布。
- 根工作区仍为 `feat/ullt-state-space-wing-aero @ 69d316e4`。历史工作区原有 PID、`eval/` 等未跟踪产物保留。没有修改历史训练、气动、reward、执行器或 frozen gate。
- **无训练、无 optimizer step、无权重更新、无封存 holdout/test 读取。** 用现有 actor 权重构建冻结推理对象；optimizer 只读取状态计数。使用 `env_isaaclab` 和 `./isaaclab.sh -p`。
- 新增有效物理数据只有一个 `+12°/heading 0/phase 0` 工况下 teacher 与 task-aware-75 的冻结回放。teacher 初次完成但 Kit 关闭前未保存输入，修正保存时机后重录一次；candidate 运行一次。C4 没有新增 PhysX 实验。
- 沙箱内最初 Kit 初始化崩溃，日志有 file-watch `errno=28`；磁盘仍有约 300 GB 空间，不能据此声称磁盘已满。原生运行环境完成补录。初始参数传递错误及日志都留在新目录，没有用失败运行冒充结果。

标记含义：**VERIFIED** 为当前读取的实物、严格推理重构或明确冻结干预支持的窄结论；**SUPPORTED** 为多项证据一致但仍有替代解释；**HYPOTHESIS** 为尚未隔离验证的机制；**UNRESOLVED** 为缺少足够数据。历史晋级仍以当时合同为准，本轮不新增晋级。

新增数据位于 [logs/rl_forgetting_mechanisms_v1](../../../logs/rl_forgetting_mechanisms_v1/)，最终离线结果是 [final/](../../../logs/rl_forgetting_mechanisms_v1/final/)，早期 `offline_v1/v2` 是开发过程产物，不作为最终数值版本。

## 2. checkpoint 与能力不是同一张表

当前实际严格加载 19 个登记 checkpoint，另加载 9 月 10 日原始 C4 的 25/50 两个 checkpoint，共 21 个；没有缺失文件被假定存在。见 [inventory.json](../../../logs/rl_forgetting_mechanisms_v1/final/inventory.json) 和 [legacy_channels.json](../../../logs/rl_forgetting_mechanisms_v1/final/legacy_channels.json)。

| 策略 | 身份与已有能力证据 | 当前网络/输入 |
| --- | --- | --- |
| C1 `2026-08-07…seed2/model_1300` | 历史稳定 C1 authority baseline；不能据此填充其未测 C3/C4 能力 | shared ELU `[256,128]`，世界航向 |
| C2c `2026-08-12…resume500_to550/model_550` | 正式晋级，525/550 相邻证据；纵向 authority | shared ELU `[256,128]`，世界航向 |
| task-aware C3a `2026-08-18…/model_25/50/75/99` | research-only；C3a 均 96/96，`+12°` 为 9/4/4/7 of 16 | shared ELU `[256,128]`，世界航向 |
| bounded C3a `model_475` | research-only；C3a 96/96，C2c `+12°` 2/16，C1 tail-limit 54.54% | shared ELU `[256,128]` |
| distill / large actor `2026-08-26…/model_100` | research-only；`+12°` 分别 6/16、4/16 | shared ELU；大 actor `[512,256]` |
| split C3a `2026-08-29_10-07-47…/model_200` | **正式晋级**，175/200 相邻通过 C3a+C1+C2c；200 的 `+12°` 16/16 | `PureRLSplitActor`，世界航向 |
| heading-canonical C3b `2026-08-31_17-32-50…/model_75` | **正式晋级**，50/75 相邻通过；C3b 176/176，并保留 C1/C2c/C3a | split，route-heading canonical |
| C3c continuation `2026-08-31_19-52-40…/model_100` | research-only；C3c 80/96、C3b 116/176，且 C1 gate 失败 | split，canonical |
| full-C3 `2026-09-01…/model_175` | 五套全通过一次；**无相邻 passing pair，research-only** | split，canonical |
| C4 original 25/50，pilot D 0/25/50/75/100 | 均非正式晋级；D50 C4 96/96、旧套件 4/5；D75 C4 0/96、旧套件 5/5 | split，canonical；C4 preview 1.2 s |

正式 C3b 晋级实物是 `2026-08-31_17-32-50…/eval_authority_50_100_c3b_v3_heading_canonical/promotion_result*.json`，`promoted=true`；full-C3 实物是 `eval/c3_full_joint_from_c2c_seed0_301iter_frozen_v1/model_175/checkpoint_summary.json`。C3a 依据 [晋级 audit](../2026-08-29-pure-rl-c3a-split-actor-promotion.md)。其余数字由指定 handoff、bounded audit 和现存 evaluation 汇总交叉核对，不能把上述各阶段分数拼成“所有策略在同一全任务矩阵上已测”。

**VERIFIED：接口检查是必要条件。** 同为 555 维，不等于相同 observation 合同：30×14 sensor history、30×4 applied-action history、5×3 body-relative preview；固定物理量缩放，无 empirical actor normalizer。早期 quaternion 在世界系，后期相对固定 route heading；旧任务 preview 0.6 s，C4 1.2 s。split actor 的 frequency 分支实际用一周期平均、相位固定输入，tail 分支用原始相位历史。加载时严格区分 shared 与 split，按 checkpoint 张量尺寸重建，未把 split 当 shared MLP。

历史 plant 是 measured multibody、CPU PhysX、480 Hz physics/60 Hz policy、DeLaurier wing loads、现有五面尾翼、无风。动作始终为 `[frequency,rudder,left_elevon,right_elevon]`。旧接口文档中某些通用控制器的 `[freq,rudder,pitch,roll]` 表述不适用于 PureRL；以 `straight_flight_env.py` 的 PureRL allocation 和 checkpoint contract 为准。

## 3. A1：强爬升退化发生在哪里

### 3.1 新增冻结数据真正补上了什么

原 bounded/task-aware frozen 汇总没有逐步 555 维输入。已有 longitudinal CSV 保留四路动作和阶段，但缺少完整 sensor history、机翼相位等，不能据此精确重建 actor observation；本轮没有这样重建。

沿用历史 worker `scripts/flapping_rl/compare_pure_rl_longitudinal_checkpoints.py`，只加推理 hook：`+12°`、heading=0、flap phase=0、entry=17.5 m、slope=25 m、时限12 s、seed0，原 C2c task 和原 termination。补录文件：

- [teacher trace](../../../logs/rl_forgetting_mechanisms_v1/c2_capture_v2/baseline_trace.npz)：719 次推理，11.983 s timeout，进入 recovery，最大观测 tilt 28.95°。
- [task-aware-75 trace](../../../logs/rl_forgetting_mechanisms_v1/c2_capture/candidate_trace.npz)：414 次推理，6.9 s terminated，progress=42.171 m，尚未到 42.5 m recovery 边界；末次输入 tilt=72.84°，7 个输入超过60°。
- 四路记录动作重构最大误差分别 `8.35e-7`、`9.54e-7`。task-aware-0 在两批输入上与 C2c teacher 完全一致，排除这一比较中的初始化/错加载解释。

**VERIFIED：这个退化个例在恢复之前已失败。** 不能只说“recovery 控制器坏了”。该 worker 保存 `terminated`，不保存 terminal cause，终止后姿态被自动 reset，CSV 为 NaN；因此本次只直接证明接近 tilt 限制后终止，不能将最后一步 cause 冒充已直接记录的 tilt。历史完整 grid 仍明确记录大量 `+12°` tilt failure。

### 3.2 完全相同输入上的动作变化

下表是 task-aware-75 减 C2c teacher 的**归一化原始 actor 均值**，不是两条不同状态轨迹的动作相减。`sym=(L+R)/2`，`diff=(L−R)/2`。阶段按动作前 progress 划分；recovery 为到达42.5 m后的约2 s，post-recovery 为之后，仅为分析分箱，不修改任何 frozen phase/gate。

| 固定输入来源/阶段 | N | frequency | rudder | L | R | sym | diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher entry | 131 | −0.1320 | −0.1969 | −0.1764 | −0.0950 | −0.1357 | −0.0407 |
| teacher 正常 climb | 239 | −0.4236 | −0.0135 | −0.1417 | −0.1392 | −0.1404 | −0.0012 |
| teacher recovery | 121 | −0.3610 | −0.0102 | −0.1538 | −0.1546 | −0.1542 | +0.0004 |
| teacher post-recovery | 228 | −0.2889 | −0.1511 | −0.2257 | −0.1009 | −0.1633 | −0.0624 |
| failed actor 的 climb | 278 | −0.4753 | +0.0312 | −0.1318 | −0.1522 | −0.1420 | +0.0102 |
| failed actor 的 tilt≥60° | 7 | −0.3895 | +0.1085 | −0.0159 | −0.2317 | −0.1238 | +0.1079 |

来源：[c2.json](../../../logs/rl_forgetting_mechanisms_v1/final/c2.json)。RMS、各 checkpoint 和覆盖统计也在其中。

**SUPPORTED：正常爬升/恢复阶段主要表现为较低扑频请求和对称尾舵偏移，entry/post-recovery 还伴随方向舵及差动改变。** 未做 C2 通道替换，不能把其一认定为独立故障原因。

这里限幅会改变机制解释：在 failed actor 的最后0.5 s，原始 frequency 差为−0.477，但两者请求均已超过上限，`clip[-1,1]` 后 frequency 差为0；tilt≥60°的7帧，frequency/rudder 的 clipped 差均为0，sym/diff 只剩 `+0.0026/+0.0185`。这说明临界状态上原始动作大差异，未必对应不同可执行命令。teacher 也可能在这些偏离状态要求相同饱和动作；不能据“调用 teacher”就宣称能救回。

闭环自身状态又产生另一种现象：candidate 的 slope 实际平均扑频 **4.636 Hz**，teacher 为 **4.515 Hz**，尽管在 teacher 的同一输入上 candidate 请求更低。candidate 末步实际扑频约5 Hz、方向舵/左舵已限幅、横向误差−2.176 m，teacher 则完成 recovery。这是状态偏离后反馈补偿与饱和的证据，不能用自身轨迹的平均扑频反驳同状态漂移。

已有 joint-from-C1 与 split 的 `+12°` CSV 也按 entry/slope/recovery 提取了四路与 sym/diff，保存在 `c2.json/historical_csv`；它们属于不同状态轨迹的描述统计，**没有被当成同状态或 C2 tilt 因果证据**。本轮没有补齐整个112-case逐步输入集，不能泛化单个新回放的相位机制到全部历史失败。

## 4. A2：C4 尾舵内部的行为机制

### 4.1 D50→D75 的新增定位

复用全部192条开发轨迹，共230,208个 observation。重构各自记录动作误差≤`1.20e-7`。D50状态上的 D75−D50：

| 指标 | frequency | rudder | L | R | sym | diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 均值 | +0.0294 | −0.1291 | −0.1993 | −0.1507 | −0.1750 | −0.0243 |
| RMS | 0.0704 | 0.1352 | 0.2068 | 0.1534 | 0.1773 | 0.0416 |
| clipped 后均值 | +0.0294 | −0.1291 | −0.1988 | −0.1504 | −0.1746 | −0.0242 |

这里的 tail 差异基本没有被命令限幅消除，和 A1 末期饱和不同。

| D50 固定状态分组 | rudder 均值变化 | sym 均值变化 | diff 均值变化 |
| --- | ---: | ---: | ---: |
| 正曲率 >0.002 rad/m | −0.1291 | −0.1751 | −0.0226 |
| 负曲率 <−0.002 rad/m | −0.1294 | −0.1732 | −0.0258 |
| 坡度 >+1° | −0.1320 | −0.1715 | −0.0284 |
| 坡度 <−1° | −0.1264 | −0.1773 | −0.0197 |
| 坡度变化率 >1°/s | −0.1285 | −0.1742 | −0.0239 |

**VERIFIED：这是跨任务阶段持续同号的方向舵/对称尾舵偏移，不是只在曲率或坡度符号改变时出现的尖峰。** 首帧同状态已出现 rudder/sym/diff=`−0.1585/−0.2258/−0.0118`，发生在两条闭环轨迹尚未分叉时。后续 D75 状态上的 sym RMS 增至0.3237、rudder RMS增至0.3027，说明漂移与状态分布偏离共同放大差别。

从各自闭环实际舵角看，D50→D75 的 rudder 平均为 `−0.0351→−0.0734 rad`，sym 为 `−0.0932→−0.1059 rad`，diff 为 `−0.0348→−0.0394 rad`；这些是**不同状态下的实际执行量**，不可与同输入原始动作均值直接等同。tail-limit fraction 仅 `0.414%→0.666%`，不支持“普遍舵面饱和造成 C4 崩塌”。

![C4行为漂移与跟踪结果](../../../logs/rl_forgetting_mechanisms_v1/final/c4_mechanisms.png)

**SUPPORTED：协调尾舵控制的持续偏置是最直接的行为描述。** 对称尾舵与纵向控制、方向舵/差动与横侧向协调有关，但在当前五面尾翼、速度/姿态依赖气动和多体 plant 中不是完全解耦坐标。没有用归一化命令的正负直接推断世界系升降/偏航效果。

### 4.2 复用干预：有闭环证据，但不是三个独立技能

9月22日 [既有诊断](../2026-09-22-c4-retention-diagnosis.md) 已证明：D75 frequency + D50 tail =96/96，D50 frequency + D75 tail =0/96；本轮没有重跑。这是 **VERIFIED 的尾舵整体因果定位**，不是对某一舵面的定位。

进一步找到并读取9月11日的**另一对** original C4 checkpoint 25/50 的现存通道替换结果。所有输入均为同一 development.json、1.2 s preview；不是 D50/D75，也不是封存测试。八种组合各抽查 `development_0` 的整段输入/动作，按实际两个 actor 重构，最大误差≤`5.97e-8`。

| original C4 的控制组合 | 成功数/96 | 横向均误差 m | 垂向均误差 m |
| --- | ---: | ---: | ---: |
| 25全部 | 96 | 0.1951 | 0.0929 |
| 50全部 | 5 | 0.6003 | 0.1847 |
| 50 frequency + 25全部尾舵 | 96 | 0.2048 | 0.0994 |
| 25 frequency + 50全部尾舵 | 6 | 0.5799 | 0.1810 |
| 50，只替换旧 rudder | 95 | 0.3274 | 0.1278 |
| 50，只替换旧 left | 96 | 0.2024 | 0.1319 |
| 50，只替换旧 right | 0 | 0.9609 | 0.1352 |
| 50，同时替换旧 left+right | 95 | 0.3895 | 0.0996 |

来源：`logs/continuous_curve/20260911_c4_channel_swap/` 与 `20260911_c4_tail_channels/` 的 launch/summary/trace；提取见 [legacy_channels.json](../../../logs/rl_forgetting_mechanisms_v1/final/legacy_channels.json)。

**VERIFIED：在这对 original checkpoint 上，左舵替换足以恢复，右舵替换反而有害；多个替换方案能恢复绝大多数能力。** 左舵替换同时改 sym 与 diff，所以不是“已经识别纵向模块”或“左舵就是唯一根因”。整个尾舵替换优于某些单通道替换，反映配合关系；95/96也不能按96/96处理。

**UNRESOLVED：D50/D75 中单独 rudder、sym、diff 的闭环因果份额。** 不把9月11日的左舵结论迁移到 D75。已有证据已足以判断“尾舵整体保护有价值、独立舵面专家尚无依据”，本轮不再扩展 PhysX 通道实验。

## 5. H1、H2、H3 的竞争解释

### H1：共享参数的梯度干扰——存在，但不足以定因

重新统计 task-aware 原始 `actor_gradient_conflict.csv` 的97次更新：strong-climb vs C3a 的 cosine 为负比例 **42.27%**，平均 **+0.0175**；strong-climb post-update KL均值 **0.02726**，C3a为 **0.01473**。因此任务权重/样本平衡没有使各技能承受相同函数变化。

新增更有区分力的事实：**97/97 次 actual-update · strong-climb loss-gradient 都为负**，均值 **−0.09666**。即使旧能力随后退化，实际步长对当次采样 strong-climb surrogate 仍是局部下降方向。D的55/60/65/70/75五次 probe 同样对 C4 为负，均值 **−0.02407**；同时 C1-C4 5/5、C3b-C4 3/5出现负 cosine，尾舵子空间也有冲突。

**SUPPORTED：共享表示会产生任务间竞争；UNRESOLVED：投影掉负 cosine 是否能防止失败。** 这些日志反对“新任务梯度单纯把旧任务 loss 推高，所以 PCGrad 必然解决”的简单故事。采样状态、advantage和有限步长都可能使 surrogate 改善与闭环可靠性分离；负方向导数也不是实际有限步长目标必然下降的数学保证。

### H2：局部更新累计漂移——当前最强机制

对**固定 teacher 回放的719个输入**，C2 teacher到 task-aware 25/50/75/99 的四维 Gaussian KL为 **0.552/0.968/2.105/2.596**；相邻已保存 checkpoint KL为 **0.552/0.489/0.582/0.447**。这里“相邻”是25个左右更新间隔，不能叫逐 update KL，也不能把 KL 相加。固定 failed-75 输入上的 teacher-relative KL为 **0.602/1.116/2.406/2.508**，结论方向一致。

对固定 D50 开发输入：

| 比较 | Gaussian KL |
| --- | ---: |
| D0→D25 | 0.663 |
| D25→D50 | 4.282 |
| D50→D75 | 1.939 |
| D75→D100 | 0.231 |
| D0→D75 | 16.403 |

这是原始、限幅前的对角 Gaussian KL，包含均值和标准差；不与物理动作 KL 混用。D50与D0行为差异很大却成功，也说明**远离旧参数/旧teacher本身不是失败判据**；关键是哪些状态上的哪些功能被改变，以及闭环结果。

参数方面，D50→D75 的 tail 三层 weight 相对L2变化为 **4.12%/6.21%/7.40%**，frequency为 **3.50%/3.74%/2.19%**；tail 最后一层 bias 只变 **0.38%**。持续动作偏移并非仅改了输出 bias，涉及整个尾舵表示/映射。参数范数不具有跨层同等功能含义，不能据此计算“尾舵承担多少百分比因果”。

optimizer 实物：D50→D75 Adam step统一从2592到4192，19个状态条目；task-aware 25/50/75/99 为992/2592/4192/5728，13个条目。没有看到中途重置的证据。既有 warm-start 使用 fixed LR `5e-5`，配置 `desired_kl=0.01`不代表执行了硬KL限制。D55–75实测 C4 post-update KL均值0.02011。

critic/reward复用9月22日诊断：C4 critic每次更新后相对当前target的MSE下降，但不同probe的 before MSE从0.718到11.383变化，不能比较为同一个固定回归问题；C4 total/path reward也下降，并不存在已有证据证明“奖励越来越好而任务越来越坏”。课程统计约25次更新才获得一批完整episode反馈，且混合多个策略版本；但固定采样C臂也崩溃，延迟不是唯一解释。

**SUPPORTED：累积函数漂移最符合目前证据。UNRESOLVED：它由 value误差、advantage偏差、任务trade-off、有限状态保护或优化器哪一因素主导。** 没有 neuron utility、特征秩变化或可塑性恢复干预，本报告不诊断 loss of plasticity，也不把 UPGD 当已验证药方。

### H3：保护覆盖不足——有缺口，也有“覆盖了仍没保住”

早期 distillation 是在 **student 当前 on-policy C1/C2c observation** 上监督冻结teacher均值；没有固定成功轨迹memory，未保护critic/std。本次failed actor甚至没到recovery，而teacher正常访问349个recovery/post-recovery输入。**SUPPORTED：此机制存在可具体展示的状态访问缺口。** 但补录的是无distillation task-aware-75，不能将它作为8月26日distillation运行的同一条失败轨迹；该实验中是否发生同样覆盖塌缩仍是HYPOTHESIS。

9月22日 memory 实物含六家族各4096条，共24,576条；只来自成功episode，未注入recovery扰动；每个样本无episode/phase/time字段。因此无法证明保留了连续的恢复序列，也不能精确统计“多少条是官方 recovery phase”。strict full-C3 teacher重新推理与存储target一致；D0也一致。memory不是D50的函数快照。full-C3 teacher的成功C4训练轨迹确实进入memory，故不能称它为“不懂C4的错误teacher”。

覆盖度采用16个物理特征：最新roll/pitch、body ground velocity、p/q/r、实际扑频、近远preview的y/z、0.483 s内角速度变化；不使用555维重复历史的原始欧氏距离。roll/pitch容差10°，u为2 m/s，v/w为1 m/s，角速度及其变化为1 rad/s，频率0.5 Hz，近preview0.5 m、远preview1 m；最近邻用最大标准化坐标差。表中1×表示所有特征在上述容差内，0.5×/2×作敏感性检查；这些是**诊断邻域，非训练/晋级阈值**。

| 查询状态 | 0.5×邻域覆盖 | 1×覆盖 | 2×覆盖 |
| --- | ---: | ---: | ---: |
| D50开发状态 vs C4 memory | 57.18% | 100% | 100% |
| D50坡度过渡 | 54.39% | 100% | 100% |
| D75开发状态 vs C4 memory | 1.15% | 40.03% | 100% |
| D75坡度过渡 | 0.018% | 39.17% | 100% |
| D75首次单点误差>0.5m前0.5s | 0% | 52.08% | 100% |
| C2 teacher正常climb vs C2c memory | 30.96% | 73.22% | 100% |
| C2 teacher recovery约2s | 16.53% | 85.12% | 100% |
| failed C2c最后0.5s | 0% | 0% | 0% |
| failed C2c tilt≥60° | 0% | 0% | 0% |

C4覆盖查询为各条件确定性每10帧抽样；C2使用全部帧。C4的0.5m crossing是时序诊断事件，**不是原有均值误差成功gate**，D50成功轨迹也可出现瞬时crossing。特征忽略yaw、完整相位历史和action-history，近邻覆盖并不保证策略等价；数值对容差敏感，不将其解释为完整状态分布概率。

更强的实物事实：**全部memory中没有一条 |pitch|>45°，也没有一条 |roll|>25°；C2c family同样如此。** failed C2c临界输入显然缺乏邻域。注意这份memory收集于9月，从未用于8月task-aware运行；比较只说明后来的memory方案对同类危险状态的覆盖能力，不能倒推它导致早期失败。

另一方面，在memory自己的精确输入上，D50/D75的 C4 sym RMS对teacher分别 **0.2083/0.2495**，rudder **0.2590/0.2406**，diff **0.0191/0.0557**。方向舵对teacher略靠近并未阻止崩塌。既有辅助损失确实运行5792次，family action-MSE为D50旧任务0.00297–0.00432/C4 0.04580，D75旧任务0.00594–0.00806/C4 0.05335。

**VERIFIED：软memory在已存输入上也允许显著漂移；SUPPORTED：失败后的状态偏离进一步扩大覆盖缺口。** 因而“只补更多恢复样本”不充分，“memory完全没用”也不正确：C/D臂在50时达到96/96，A/B只有5/67；但75时四臂都0/96。单seed、训练分布与轨迹反馈不同，不能据此估计通用方法效果。

## 6. 为什么历史方法部分有效、部分失败

| 方法 | 解决了什么 | 没解决什么/替代解释 |
| --- | --- | --- |
| exact rehearsal | 修正早期C3直飞rehearsal不等同C2纵向恢复合同 | 已漂移optimizer续训后 `+12°` 仍由8/16恶化至4/16；正确采样不保证函数保持 |
| bounded warm start | 首次actor displacement从0.38454降到0.00808，消除entry-only cold-start冲击 | 到475仍能转弯，499降60/96；启动保护并非持续保护 |
| task-aware PPO | task advantage/权重/phase配额更明确，C1和C3a得以保留 | strong-climb KL更大，且97次更新对本地surrogate下降仍发生遗忘 |
| actor distillation | 某次 `+12°`改善至11/16，说明旧行为监督有价值 | on-policy覆盖、只保护均值、软损失；与task-aware组合及扩容仍不通过全部gate |
| 扩大MLP容量 | 大actor model100的C1 tail-limit从11.44%改善到3.03% | `+12°`从6/16降4/16；不支持“单纯容量不足” |
| joint任务训练 | full-C3-175同时通过五套；避免C3c训练集不包含完整C3b的结构性遗忘 | 无相邻passing pair，hard-slice continuation仍168/176；C4本来也是六家族joint，仍崩溃 |
| split actor | 频率分支不直接追逐翼拍相位，tail仍可用快相位；正常C2爬升的高幅扑频翻转消失，C3a晋级 | 同时有独立trunk与输入时间尺度变化，不能把收益全归因于“更多模块”；尾舵仍可漂移 |
| heading canonical | C3b达到正式相邻晋级；显式表示不变量比弱yaw一致性loss更有效的证据 | 无风/固定模型下的观测对称性收益，不是GRU或动力学辨识收益 |
| behavior memory | C4中期成绩改善，有真实auxiliary调用 | 旧teacher、成功轨迹覆盖、软均值约束均有限；不能保护任意恢复状态和未来成功函数 |

full-C3 hard-slice历史还存在异步single-reset sampler只覆盖首个template/右转的实现偏差，已在[9月3日audit](../2026-09-03-pure-rl-c3joint-hard-slice-continuation.md)说明。不能把已修正代码追认成旧实验实际覆盖，也不能据该次168/176否定所有joint training。

## 7. 策略可组合性：哪些是真的，哪些还是上界

**VERIFIED：存在可复用能力，但证据强弱不同。** C3b正式策略同时保留C1/C2c/C3a，因此不是所有阶段都需要独立专家。full-C3-175五套通过进一步反对“单策略原则上无法容纳这些技能”。

同一历史合同下D50/D75具有**套件gate互补性**：

| checkpoint | C1 | C2c | C3a | C3b | C3c | C4 |
| --- | --- | --- | --- | --- | --- | --- |
| D50 | PASS | PASS | PASS | FAIL | PASS | 96/96 |
| D75 | PASS | PASS | PASS | PASS | PASS | 0/96 |

D50的C3b逐case success其实176/176，失败的是聚合横向均误差0.505992m略超0.5m；D75为0.494234m。故这里不是“D75获得大量D50不会的机动”，而是很窄的精度gate互补。事后选D50负责C4、D75负责旧任务可拼出 **6/6 suite oracle上界**，没有单个checkpoint通过六套，也没有连续任务切换器的闭环成绩。不能将suite gate上界写成所有轨迹逐状态oracle。

跨frequency/tail模块替换已有两次不同C4训练运行的闭环恢复证据，是比单纯同状态动作差更强的**有限组合证据**。但它保持整块tail函数；没有证明稳定/纵向/横向/机动协调能独立拆解。单左/右替换的不对称结果尤其要求保留耦合。

无需训练的状态gate诊断：以可观测参考坡度符号选择D50或D75，沿现存D50轨迹共有334次符号切换；切换边界两actor同状态rudder/sym/diff差RMS为 **0.1347/0.1778/0.0413**。在D75轨迹为317次、**0.3078/0.3290/0.0532**。该规则事先选为诊断规则，没有按分数调参，但**没有任务优势支持应在坡度正负时选择不同专家**；两类坡度上漂移方向近乎相同。动作差不会因状态边界而自动连续。

实际部署还需：共同坐标/preview语义、相同实时history而非各自幻想历史、切换或混合后的执行器限幅、coupled tail稳定性。当前MLP/split没有隐式recurrent hidden state，但30帧输入history包含实际过去动作，混合后会进入新分布。未来若改GRU还需定义两专家隐状态的更新与reset。简单平均动作并不平均稳定性。

**结论：保留成功技能有实际研究价值；按任务/舵面自动组合的可部署性仍 UNRESOLVED。** 当前证据不足以把通用MoE、CompoNet或Soft Modularization排在功能保持机制之前。

## 8. 面向 L-3D/I-3D 的路线筛选

| 方向 | 本机证据的支持 | 主要失败方式 | 本轮优先级 |
| --- | --- | --- | --- |
| 连续联合任务分布 | 串行C3c并非C3b超集；joint full-C3成功容纳五类 | C4已joint仍崩溃；联合分布不能约束函数漂移 | 必要训练范式，但不是本轮唯一新机制 |
| 可组合模块化控制 | 快慢分离晋级；冻结跨时标组合恢复C4 | 门控错配、切换瞬态、尾舵非独立、冻结旧plant错误规律 | 有价值的后备结构假设，尚不支持舵面MoE优先 |
| 功能保持约束更新 | 同状态累计漂移、soft memory失守、整块tail替换恢复 | 锚点不足或过时、约束抑制合理适应、KL与闭环误差不等价 | **唯一主路线** |
| 动力学上下文/历史自适应 | 555维已有约0.5s历史，快慢表示与heading canonical有收益 | 旧固定plant证据无法证明隐藏动力学可辨识；上下文可能把风与动作/轨迹相关性混淆 | 面向新plant值得研究，但当前无优先证据 |

建议的策略形态是**行为锚定的快慢双通路**：慢扑频保留一周期响应表示，快尾舵保持可联合输出三舵面的协调函数；对已经验证的正常/恢复/过渡训练状态明确保护其功能。可以用函数约束而非冻结所有权重实现；当前并不要求trunk必须是MLP。是否需要可学习历史编码、稳定基底加有界机动残差或新专家，应由“约束在不损害新能力的前提下是否可行”的实验决定，不能先将这些部件一起加上。

这条路线优于继续串行PPO的理由不是代码改动少，而是它直接针对已干预确认的对象——**成功控制函数的保持**；串行课程+rehearsal只规定看什么任务，常规PPO只约束当前policy附近的surrogate，二者都未承诺保留成功tail函数。它也比立即拆舵面专家更符合已有耦合证据。

L-3D/I-3D将改变三维局部流动和横侧向控制规律；本报告**不迁移旧权重、不推断旧技能在新plant有效**。未来可迁移的是快慢表示、函数保持与可辨识上下文的研究问题。新plant上的抗风、持续三维航迹与结构不确定性仍需要它自己的控制响应/可观测性证据；旧零风数据不能排名GRU与非递归历史编码，也不能证明RMA式latent会识别气动误差。

### 文献候选只提供机制，不提供本机效果

检索核对正式论文/作者实现；未安装、训练或运行其代码。

| 方法 | 原论文机制与本任务的关系 | 筛选意见 |
| --- | --- | --- |
| CompoNet，ICML2024 | growable modular policy、复用并保护历史模块；[论文](https://proceedings.mlr.press/v235/malagon24a.html)、[官方代码](https://github.com/mikelma/componet) | 对技能保存有启发，但本机仅有跨frequency/tail复用和狭窄gate互补；连续状态门控、增长成本及新plant兼容性未证实 |
| Soft Modularization，NeurIPS2020 | 学习routing组合模块路径，缓解多任务表示竞争；[论文](https://papers.neurips.cc/paper_files/paper/2020/hash/32cfdce9631d8c7906e8e9d6e68b514b-Abstract.html)、[作者项目及代码入口](https://rchalyang.github.io/SoftModule/) | 可以表达尾舵协调，但软routing和共享模块仍会漂移；它本身不是旧函数保持保证 |
| UPGD，ICLR2024 | 依据utility保护有用参数、扰动低utility参数；[论文](https://arxiv.org/abs/2404.00781)、[官方实现](https://github.com/mohmdelsayed/upgd) | 与防遗忘相关，但当前没有loss-of-plasticity实证；参数utility保护不等于关键飞行状态的函数约束 |
| RMA，RSS2021 | base policy结合由历史推断的adaptation context；[作者项目](https://ashish-kmr.github.io/rma-legged-robots/)、[链接的实现](https://github.com/antonilo/rl_locomotion) | 面向风与模型不确定性有针对性；首先需证明历史响应能区分所需控制规律，不能由旧固定plant退化反推其必要性 |
| LEGION，NMI2025 | 非参数知识空间、语言任务推断与知识保存/组合，使用SAC/replay；[论文](https://www.nature.com/articles/s42256-025-00983-2)、[官方代码](https://github.com/Ghiara/LEGION) | 不是直接替换PPO actor的现成方案；语言任务边界与连续飞行参考不对应，不能用机器人任务成功率证明扑翼60Hz协调可行 |

### 唯一后续辨别实验（本轮未执行）

以**历史D50**为同一起点，固定旧plant/reward/任务比例/optimizer初始化/更新次数（如25次）/seed预算，做一次短程两臂比较：

1. 两臂使用**完全相同、另行采集的 training-only D50成功行为锚点**及旧技能恢复/过渡锚点，避免把本报告开发评估输入喂入训练。两臂有相同成功reference，避免将“换teacher”和“换更新机制”混为一项结果。
2. 对照臂为软actor行为损失；实验臂为关键分层状态上的显式协调尾舵函数漂移约束。提前声明可执行动作/原始Gaussian约束的定义、预算、拒绝更新及optimizer回退语义。不能按哪组过gate再调阈值。
3. 唯一核心判别：约束是否确实控制了漂移，且阻止已知C4退化，同时保留历史五套能力并允许新任务改善。仍用原有全部gate，不因D50起点C3b=0.50599而放宽。

若函数漂移受控且闭环保持，支持先改学习约束而非泛化MoE；若约束只能通过停止有效学习满足，支持共享表示/目标间不可兼得，才有更强理由增加受保护模块；若锚点保持而评估仍崩溃，优先检查分布/上下文覆盖，而不能直接宣布模块化获胜。该实验只选择机制，不证明跨seed普适性或新plant效果；本轮停止在这一选择处。

## 9. 实现、核对与未完成项

新增工具：[rl_forgetting_mechanisms.py](../../../source/flapping_bot/flapping_bot/analysis/rl_forgetting_mechanisms.py)、[forgetting_capture.py](../../../source/flapping_bot/flapping_bot/analysis/forgetting_capture.py)。没有修改历史worker、环境、PPO或模型实现。

定向测试：[test_rl_forgetting_mechanisms.py](../../../tests/test_rl_forgetting_mechanisms.py)，验证动作坐标可逆性、Gaussian KL方向与四通道求和、物理特征单位及yaw/quaternion double-cover不变性，**3 passed**。实际artifact核对包括21个actor严格加载、memory teacher重构、192条C4逐输入重构、两条C2补录重构，以及8种旧干预各一条轨迹重构。没有运行完整测试套件或新训练。

离线复现（需选择不存在的新输出目录）：

```bash
cd /home/zn/IsaacLab/.worktrees/native-multibody-rl
export TERM=xterm
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
export MPLCONFIGDIR=/tmp/rl-forgetting-mpl
./isaaclab.sh -p -m flapping_bot.analysis.rl_forgetting_mechanisms \
  --output logs/rl_forgetting_mechanisms_v1/NEW_OFFLINE_OUTPUT
./isaaclab.sh -p -m pytest -q tests/test_rl_forgetting_mechanisms.py
git diff --check
```

补录使用同一环境下的 `-m flapping_bot.analysis.forgetting_capture --worker`；所有工况与数据目录在§3列明，CSV旁summary保存checkpoint、工况、seed，日志保存实际runtime结果。该工具运行冻结historical worker而不是训练入口；本报告复现离线分析不需要再次执行PhysX。

仍未解决：完整C2c112-case的同状态阶段泛化、D50/D75单尾舵坐标的因果分解、早期distillation实际训练轨迹中的recovery覆盖、memory精确episode/phase标签、可观测gate在闭环中的技能组合、新plant抗风/模型适应。它们均没有用其他任务的成功替代。

本轮没有commit/push/merge/rebase，没有清理已有文件；研究结论停在路线选择。
