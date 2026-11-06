# Unified Trajectory PPO 训练前审查（2026-11-06）

结论：**CONDITIONAL GO**。代码允许开展预先声明为探索性的 Fixed/Adaptive 对照；尚不能把该对照解释为可飞性筛选或防遗忘机制已得到验证。若正式实验要求回答三个机制各自的因果贡献，则 **NO-GO（研究识别性不足）**。本轮没有运行 PPO、参数更新或 PhysX 评价，不改变 L/I plant，不提交代码。

起点 HEAD `43239b9d977aa5354afc1db9085229f0c0506311`。继续时仅有上轮 checkpoint 部分修改。历史 `logs/unified_ppo_smoke/L3D_adaptive_v2/model_000002.pt` 和 `_final.pt` 保留。历史 smoke 的全部安全失败来自既有集成报告，未在本轮重现。阅读范围包括 PROJECT_STATE、集成合同、runner、任务环境、轨迹、采样器、能力统计及 SQLite 前缀恢复。

## 已解决的代码问题

1. **BLOCKER / 已修复：周期 checkpoint 丢反馈。** 更新后保存 pending 模型供子进程评价；验证完整开发批次，接收反馈，强制刷新采样分布，然后写 `_evaluated.pt`，latest 指向它。模型与 latest 分别使用临时文件原子替换。Adam、随机状态、reset 序列、sampler、receipts、训练计数和 SQLite terminal 截止序号仍保留。
2. **BLOCKER / 已修复：中断隐式跳过评价。** pending 恢复在任何新训练动作前重新运行该冻结模型的开发评价，并在新目录保存完整恢复点。评价前、过程中、接收后但保存前中断，都从旧 pending 状态重放；已完成 checkpoint 不再重放。重放会增加真实计算消耗，不能把累计已提交预算当作包含失败尝试的总成本。旧目录及未完成评价不复用、不覆盖。恢复是新物理 episode，不是 PhysX 位级续接。
3. **BLOCKER / 已修复：没有独立最终集合。** development 为原 seed 401–405 的五条 60 m 轨迹。final 为 1401–1406 的六条 80 m 轨迹，依次直飞、持续转弯、持续爬升、横向反向、纵向反向、三维耦合。长度不同也避免直飞几何实际重复。任务、初态和判据在同 backend 两模式相同；80 m 是预声明的航程外推，不能与开发成功率混为一谈。保存完整 spline 参数、任务日志、初态证据和实际结果。只允许 `--evaluate --eval-suite final`，不创建训练采样器；反馈接收器拒绝 final。默认仍 development。
4. **BLOCKER / 已修复：Git HEAD 作为物理身份。** physical_v2 包含物理 Python 文件、执行器所在 straight_flight_env、实际外部 native extension 源码/配置/二进制、URDF 内容、翼几何 CSV、实际配置、初态、physics/control dt。SHA/dirty 单独写 source_provenance，物理源文件另存快照。文档和训练日志代码不进入物理身份。整个 physics 目录保守纳入，未启用模型代码变化也可能要求新身份，这是可接受的过度失配，不是漏检。
5. **已修复：反馈延迟与重复。** 五条评价不必等待原八条反馈阈值，完成批次后主动更新；同 checkpoint、轨迹、初态、plant、contract 的语义 receipt 去重，不能靠重新生成 episode ID 增加统计样本。接收器检查 suite、完整任务参数、版本、步数、记录数及 single/frozen 身份。
6. **已修复：非有限动作终止不一致。** 环境 failed 同时检查原始动作有限性，与证据 evaluator 一致，避免证据已终止而环境继续。

旧 checkpoint 不取消身份检查：旧身份只记录三个源文件，不能证明 native、其余物理依赖与当前一致，明确拒绝迁移。保留旧 final 文件与 optimizer 数据，但它们不能直接用于新版恢复/评价。需要历史依赖快照证据才能另行迁移，不能仅凭 Git SHA 推断 dirty 树内容。

## A. 研究假设与实际机制

**VERIFIED（代码）**：Fixed 是 q0；Adaptive 是基础 .4、保持 .25、探索 .35 的混合。探索分数包含访问次数倒数、正成功率变化、遗忘和离线相似性，不包含低成功率惩罚或硬可飞性排除。保持围绕已成功 anchor/历史 witness 抖动，失败时回退 q0；未知区域有基础采样和探索下限。

因此 Adaptive 更接近“成功邻域保持＋访问不足/进步区域探索”，不能直接描述成减少难以完成任务。基础项保证覆盖，也会持续支出在不可完成区域；保持可能提高容易任务比例，探索可能抵消它。只有观察来源分布、每来源有效航程/失败率/transition 成本，才能知道净收益。

离线证据只从同 backend 的 VERIFIED FEASIBLE 几何特征形成正的高斯 bonus，没有把它转成当前初态和 60 m 任务的成功认证。长度尺度为 150 m，持续长度尺度 60 m，距离由多个分量叠加；不能仅凭航程不同断言它完全无效，也没有证据证明它有意义。**UNRESOLVED**：本轮未计算实际 prior 分布及与均匀分布差异。最有区分力的验证是固定当前 proposal bank，报告最近离线距离、bonus 范围与实际抽样偏移，无需训练。

**RESEARCH RISK / 高**：两组只能识别整套 Adaptive 相对 q0 的净效果，无法分别识别离线先验、探索和保持贡献。单靠分布变化无法约束 PPO 对未访问状态的函数漂移；保持采样也不等于保持性能。若要求机制归因，应先明确研究问题或另行授权最小机制对照，本轮不自动增加实验组。

## B. 任务几何与控制时间

Spline 对 kappa=d heading/ds 与 gamma 建模，弧长积分得到连续三维曲线；几何限幅、局部投影及 high-water 防止跨分支跳跃与反复刷进度。连续几何不等于动态可飞。持续/反向在生成参数中存在；默认 60 m 加初始直飞段会压缩持续机动与恢复空间。

初速约 9.25–9.66 m/s 时，2/5/10 m 预瞄约为 .21/.53/1.06 s；60 m 名义通过时间约 6.2–6.5 s。与约 3–3.74 Hz 翼拍相比，最远预瞄只覆盖约 3–4 个周期。**RESEARCH RISK**：可能不足以提前建立持续爬升/转弯和反向恢复，但没有实际闭环延迟证据支持立即改预瞄。最有效证据是首次稳定策略的预瞄范围与姿态/轨迹响应延迟，而不是盲目调参数。

共同初始水平速度与初始水平切线匹配，机体鼻上仰来自有限时域候选，不能称严格配平。相位为零，翼/尾状态相同；初始直段不能保证在随机动作下稳定。final 80 m 增加持续/恢复空间，也明确改变评价航程。

完成需要路径终点附近且误差≤.5 m，安全失败优先完成、完成优先 timeout。120 s 是资源上限；停滞可避免大失败惩罚，但没有速度跟踪或最低前进约束。其学习后果属于奖励风险，不自动改变任务目标。

## C. PPO 目标、观测和动作

39D 包含机体线/角速度、重力、路径误差/切线、2/5/10 m 预瞄、四个几何量、翼拍相位、动作、实际频率、三尾舵位置、剩余航程。没有实际翼关节误差/速度、尾舵速度和部分执行器内部状态。native 强约束使名义翼位置可从相位推断，但不能证明整个动力学对该 MLP 完全可观测；若迟滞/执行器记忆主导，可能需要状态补充或递归策略，而非仅调网络宽度。

Actor/critic 为三层 256 ELU，四通道联合输出，初始 std=1。频率 governor 与快速尾舵有不同时间尺度；独立高方差探索可迅速破坏候选初态。历史五条评估全失败只支持“当前未学会”，不能证明 PPO 不可学习，也不能证明课程有收益。

奖励是 tracking 加权的新航程、按 dt 的负路径误差、动作变化惩罚、完成 +10、安全失败 -5。high-water 阻止往返重复领奖；贴近路径静止的即时奖励约为零，对比高风险前进可能成为保守局部解。频率与舵面用相同归一化动作差惩罚并不代表相同物理成本。gamma=.99 在 60 Hz 对应约 1.66 s 折扣时间尺度，终点奖励远于多数初始视野；幸有密集进度，但早期失稳可能截断前飞学习信号。

**RESEARCH RISK / 高**：两组可能共同受制于基础稳定飞行不足，届时课程对照无解释力。最有区分力的是经另行授权后，检查基础直飞的存活时间、有效航程和动作饱和，先分辨稳定控制瓶颈与任务采样瓶颈。若是结构瓶颈，应考虑稳定行为初始化/分层动作或有记忆的控制表示；本轮不擅自更换。

## D. 证据、能力与遗忘

物理 witness 由完整 episode 的精确 trajectory/plant/initial/contract 身份查询；策略能力是邻域经验估计，不是物理认证。mixed episode 不进入冻结版本统计。新的语义 receipt 阻止同一冻结任务重复记账；最终集合不反馈。

**RESEARCH RISK**：PolicyCapability 的邻域上下文只包含 plant/contract，没有 initial；开发五环境与训练多环境的 scene/index/世界坐标不同，当前共同无风初态具有平移等价假设。不能把邻域成功率称为对任意初态的认证。未来增加初态扰动必须先扩展能力上下文。精确 witness query 本身保留 initial，因此未发现它直接错误认证其他初态。

训练任务来源、几何分类、完整轨迹及 episode 航程有记录；每 iteration 的来源计数是新 reset 次数，不是每来源 transitions。有效 completion_rate 也不是有效 transitions。要声称“减少无效训练”，必须离线关联任务/step 证据按来源统计实际 transitions 与进度，不能仅用 reset 分布。

## E. 公平性和计算预算

同 backend 保持 plant、候选初态、网络、PPO 参数、四动作权限、奖励、终止、任务域、seed 和初始权重一致；Adaptive 的开发反馈是预期机制。两组评价五任务×相同固定步数，提前结束后 padding。训练 trans 与开发步数分开记录；final 是六任务×预算，完全独立输出。相同 seed 不意味着采样变化后 episode 配对，故对照对象是整个训练系统。

默认 256×48×1000 = 12,288,000 训练 transitions；40 次开发评价×5×7200 = 1,440,000 环境步；一次 final = 43,200 环境步。总计算还包括 padding、子进程启动、数据库、保存及中断重放；不能用训练 transitions/s 代表端到端成本。一个 seed 不能支持总体统计优势，正式结论必须限定该运行；不自动增加种子。

## F. 运行与复现

SQLite 前缀只复制 seq≤cutoff 的已终止 episode/steps/witness，不把在途 episode 变成成功。完整恢复仍依赖源 evidence 文件；单独 .pt 不是便携归档。恢复前校验网络/训练合同和物理/评价身份；新增 final 参数和任务环境代码身份进入训练合同。最新文件更新原子，但不宣称在断电且文件系统未落盘时具有数据库级跨文件事务保证。

物理源码/URDF/CSV 内容变化会失配；保守文件集会误拒绝无物理影响的同文件编辑。源码溯源与兼容性分开，文档与日志提交不会再因 HEAD/dirty 改变 plant。外部 native 二进制纳入身份，避免只检查仓库 C++ 却运行另一份插件。环境/Isaac Sim 版本固定仍是运行前提。

磁盘主要随每 transition 的 SQLite step 增长，checkpoint 不复制全库。能力窗口有界，但 receipt/evaluations 随评价次数增长；默认数十次可接受。历史单批 12,288 transitions 约 55–61 MiB 只能做容量预警，不能当长期精确外推。长期运行前应按目标 transitions 预留几十 GB 并监测，不能靠删 evidence 节省空间。

## 验证与停止边界

验证结果：原 35 项 + 新增 10 项，共 **45 项通过**（分批执行）；语法检查与 `git diff --check` 通过。

原 35 项数学、缓存、在线证据、合同测试通过；新增测试覆盖 final 可重复/几何合法/反向覆盖、文档不改物理身份而物理代码会改、旧身份拒绝、临时 checkpoint 保存 sampler/optimizer/SQLite 截止、final 反馈拒绝、评价中断顺序，以及历史真实 SQLite 的只读前缀恢复。编排函数隔离执行，未构造 PPO 或启动 PhysX。原测试中模拟证据仍不能进入真实反馈路径。

没有做新 runner 的真实 PhysX 验证，因此运行时接线可信度来自原实现与定向测试，不能称新版本端到端已实测。没有新增训练结果，也未证明 Adaptive 优势。

## 命令

保留 integration 文档的正式训练参数与命令；本轮不执行。新版首次训练必须新建输出，不从旧 physical_v1 smoke 恢复。后续 resume 使用最新 `_evaluated.pt` / `_final.pt`；pending `.pt` 会先重放开发评价。

```bash
# 已激活 env_isaaclab，设置 PYTHONPATH 后；CHECKPOINT 是新版真实训练产物
./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py --evaluate \
  --eval-suite final --backend L3D --sampling adaptive --checkpoint "$CHECKPOINT" \
  --eval-step-budget 7200 --output logs/unified_ppo/final_L3D_adaptive_s0
# Fixed 换 sampling 与对应 checkpoint，其他参数不变；每次使用新 output。
./isaaclab.sh -p -m pytest tests/test_unified_pretraining.py tests/test_unified_ppo_contract.py \
  tests/test_online_trajectory.py tests/test_unified_trajectory.py tests/test_unified_trajectory_cache.py -q
```

优先待决策事项：①研究结论限定整套采样净效果，还是要求三个机制因果归因；②基础稳定飞行不足时是否允许改变策略初始化/控制架构；③正式运行前量化离线 prior 的实际作用。这些是研究决策，不应伪装成已经由测试解决的代码问题。
