# PureRL C3a 课程学习能力遗忘 Handoff

## 1. 本文档的用途

这份 handoff 面向接手 `feat/native-multibody-rl` 的新 agent。目标是让接手者不依赖聊天记录，也能理解：

- 仓库中扑翼飞行项目与上游 Isaac Lab 的边界；
- 当前仿真、任务、训练和评估代码如何串联；
- C1、C2、C3 课程的定义及晋级规则；
- 我们要解决的能力遗忘问题；
- 已经完成的实验、有效结论和失败路线；
- 容易导致错误结论的实现和评估陷阱；
- 下一步最小、可判定的实验是什么。

本文档从 2026-08-26 的状态开始，并已追加到 2026-08-29 的 C3a split-actor 正式晋级结果。已有架构文档、ADR 和 audit 仍是各自主题的详细事实源；本文档只做导航、整合和接续，不替代它们。当前结论优先看第 29 节和对应 promotion audit。

## 2. 当前工作区状态

- 仓库根目录：`/home/zn/IsaacLab`
- 当前工作树：`/home/zn/IsaacLab/.worktrees/native-multibody-rl`
- 分支：`feat/native-multibody-rl`
- 当前提交：`04ef4c12 feat(rl): add controlled C3a retention experiments`
- 当前远端：`origin/feat/native-multibody-rl` 与本地提交一致
- 当前工作树包含尚未提交的 split-actor 实现、评估与文档变更；必须先读 `git status`，不要覆盖或拆散这些改动
- 本地存在未跟踪的 `*.pid`、`eval/` 和实验结果；它们属于运行产物，不要删除、提交或据其文件名推断实验成功

`docs/PROJECT_STATE.md` 仍是项目总状态入口，并已更新到 C3a split-actor 晋级和 zero-shot C3b 下一步。实验细节以本文档第 29 节和 promotion audit 为准。

## 3. 接手后的必读顺序

在规划或修改代码前，按以下顺序读取：

1. 工作树内全部 `AGENTS.md`；必须用 `rg --files -g AGENTS.md` 重新发现，不能只读仓库根规则。
2. `docs/PROJECT_STATE.md`。
3. 本文档。
4. 下列架构文档：
   - `docs/architecture/current_simulation_call_chain.md`
   - `docs/architecture/coordinate_frames_and_units.md`
   - `docs/architecture/wing_tail_controller_interfaces.md`
5. 课程合同和最新方法 ADR：
   - `docs/decisions/ADR-2026-08-10-pure-rl-curriculum-domain-contract.md`
   - `docs/decisions/ADR-2026-08-12-pure-rl-maneuver-envelope-v2.md`
   - `docs/decisions/ADR-2026-08-13-pure-rl-c3a-exact-c2c-rehearsal.md`
   - `docs/decisions/ADR-2026-08-13-pure-rl-c3a-actor-distillation-experiment.md`
   - `docs/decisions/ADR-2026-08-13-pure-rl-c3a-retention-aware-sampling.md`
   - `docs/decisions/ADR-2026-08-14-pure-rl-c3a-bounded-warm-start.md`
   - `docs/decisions/ADR-2026-08-29-pure-rl-split-frequency-actor.md`
6. 已完成实验 audit：
   - `docs/audits/2026-08-13-pure-rl-c3a-c2c-rehearsal.md`
   - `docs/audits/2026-08-13-pure-rl-c3a-actor-distillation.md`
   - `docs/audits/2026-08-14-pure-rl-c3a-retention-aware-sampling.md`
   - `docs/audits/2026-08-18-pure-rl-c3a-bounded-warm-start.md`
   - `docs/audits/2026-08-29-pure-rl-c3a-split-actor-promotion.md`
7. 最后运行 `git status --short --branch` 和 `git log -8 --oneline --decorate`，确认没有比本文档更新的代码或实验记录。

历史 handoff 可用于理解形成过程，但不能覆盖当前结论：

- `docs/handoffs/2026-08-05-native-multibody-plant.md`
- `docs/handoffs/2026-08-10-pure-rl-curriculum2.md`
- `docs/handoffs/2026-08-11-pure-rl-curriculum3.md`

## 4. 仓库架构和责任边界

### 4.1 上游框架与项目代码

这是 Isaac Lab 的 fork。默认把 `source/isaaclab/`、`source/isaaclab_tasks/`、`source/isaaclab_assets/` 和共享应用视为上游维护代码。扑翼项目主要位于：

```text
source/flapping_bot/
├── flapping_bot/
│   ├── config/          # 项目配置和参数入口
│   ├── direct/          # DirectRLEnv、任务配置、观测、reward、termination、reset
│   ├── path_tracking/   # C1/C2/C3 mission、路径采样和路径查询
│   ├── physics/         # 扑翼/尾翼气动、原生多体机构、外力计算
│   ├── px4_like/        # 控制器、估计器、RSL-RL 项目扩展、训练诊断
│   └── ...              # 项目资产、场景和辅助模块
├── setup.py
└── config/extension.toml

scripts/flapping_rl/     # PureRL 训练、评估、晋级和 checkpoint 选择入口
scripts/flapping_px4/    # PX4 风格控制/仿真脚本
tests/                   # 项目合同、单元和 CPU-native runtime gates
docs/                    # PROJECT_STATE、架构、ADR、audit、handoff
```

仓库规则冻结的依赖方向是：

```text
config -> physics/control/path -> environment -> scripts
```

不要让 `physics/` 依赖 reward、课程或 PPO；也不要为了让 RL 通过而修改气动物理、质量属性或控制器增益。

### 4.2 当前仿真调用链

PureRL 的主要运行链如下：

```text
scripts/flapping_rl/train_and_watch.py
    -> Gym task ID / source/flapping_bot/flapping_bot/direct/__init__.py
    -> C1/C2/C3 EnvCfg / direct/flapping_bot/straight_flight_env.py
    -> mission/path sampler and query / path_tracking/
    -> four direct policy actions
    -> native wing trajectory constraint + DeLaurier wing loads + tail loads
    -> CPU PhysX multibody integration
    -> 555-value actor observation, reward, termination and reset
    -> RSL-RL PPO
    -> project adapters / px4_like/rl_training_utils.py
```

正式环境实现是：

```text
source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py
```

旧的 `flapping_env.py` 已弃用，不要在旧环境中实现修复。

### 4.3 冻结的 plant 和接口合同

当前训练/晋级 authority 必须保持：

- measured multibody wing plant；
- project-local native `FlappingWingTrajectoryJoint`；
- CPU PhysX；该原生 constraint 与 direct-GPU PhysX 不兼容；
- 物理步长 `1/480 s`，decimation `8`，即策略频率 `60 Hz`；
- `scene.replicate_physics=False`；
- DeLaurier strip-integrated wing loads 和现有五面尾翼模型；
- 最终 wrench 以 base-link local 语义施加；
- 四维 action：扑翼频率、方向舵、左升降副翼、右升降副翼；
- actor observation 为固定的 555 个归一化值；
- C1-C3 authority 中无风；
- 当前 task-aware 实验没有启用 actor/critic empirical observation normalizer。

坐标系、符号和作用点不能凭记忆推断，必须查 `coordinate_frames_and_units.md` 和接口文档。任何 frame、单位、reference point 或 batch/device/dtype 的隐式改变都可能制造看似“课程遗忘”的假象。

## 5. 课程与晋级合同

### 5.1 课程定义

| 课程 | 要学习的能力 | 当前状态 |
| --- | --- | --- |
| C1 | 水平直飞和基本维稳 | 已有晋级 authority |
| C2a/C2b/C2c | 水平、爬升、下降及恢复；C2c 是 4--12 度强纵向包线 | C2c 已晋级 |
| C3a | 独立水平转弯 | split actor `model_200.pt` 已由相邻 175/200 全套通过证据正式晋级 |
| C3b | 顺序动作事件 | 已实现合同；下一步为晋级 C3a 的 zero-shot 评估，尚未训练 |
| C3c | 转弯与爬升/下降耦合 | 已实现合同，未授权训练 |

C3a 不是从 C2c optimizer 继续训练，而是从正式晋级 C2c checkpoint 进行 **weights-only** 初始化，使用全新的 optimizer 和 iteration 计数。

正式 C2c source 是：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-12_17-12-48_pure_rl_c2c_seed0_resume500_to550/model_550.pt
```

相邻 `model_525.pt` / `model_550.pt` 是 C2c 已晋级证据，也是历史 C3a 实验的纵向 authority source。C3a 现已由 split actor 的相邻 `model_175.pt` / `model_200.pt` 全套通过证据晋级；后续 C3b 必须使用该 run 的 `model_200.pt` 作为 weights-only source，并按 split actor 类重建 policy，不能当作标准 shared actor 加载。

### 5.2 C3a 当前训练分布

当前 C3a rollout 使用累积课程：

```text
C1 level rehearsal             15%
authoritative C2c rehearsal    35%
C3a turn                       50%
```

C2c rehearsal 内部重点分配 climb/descent `2:1`，并可将强爬升概率设为 `0.5`。C2c rehearsal 使用独立的 authoritative C2 longitudinal path，不再用“看起来相似”的 C3 vertical geometry 代替。

这点很关键：早期 C3 rehearsal 的确没有精确覆盖 C2c；该问题已经修复。当前问题不是“C2c 数据完全消失”，而是混合 on-policy 优化仍然会破坏强爬升功能。

### 5.3 冻结评估与晋级

| Suite | 固定案例数 | 关键 hard gate |
| --- | ---: | --- |
| C1 retention | 16 | 全部生存；tail-limit fraction 不高于 10% |
| C2c v2 | 112 | `0, +/-4, +/-8, +/-12`；survival 至少 95%；climb success 至少 90%；无不允许的 tilt failure |
| C3a v2 | 96 | 当前阶段 survival/event/overall 和空间误差、roll 等完整 gate |

`+/-15 deg` 只用于 C2c 诊断，不属于晋级 grid。

默认 256 env、每 rollout 48 steps 时：

- 第一次晋级证据从 iteration 50 开始；
- 每 25 iteration 一次；
- 需要两个相邻、sample-equivalent checkpoint 同时通过当前课程和所有旧课程 retention。

因此 100 次训练实际保存的最终模型通常是 `model_99.pt`。`75 -> 99` 不是冻结的 25-iteration 相邻证据。需要合法的 `model_100.pt` 时，应运行 **101 iterations**。

训练 telemetry、在线 scheduler success、watcher 输出或单个 suite 通过，都不能代替三套 fresh-process frozen evaluation。

## 6. 我们真正要解决的问题

目标不是让 C3a 单独达到高 reward，而是找到同一个 actor，使其同时保留：

```text
C1 level stability
+ C2c climb/descent/recovery, especially +12 deg climb
+ C3a turning
```

目前最精确的诊断是：

1. 早期存在一个独立的冷 optimizer 启动冲击，已经由 bounded warm start 修复。
2. 修复首步冲击后，长期 shared-policy PPO drift 仍然存在。
3. C2c 遗忘高度集中在正强爬升和 recovery 附近，常见失败是 `+12 deg` tilt termination；不是 C2c 所有能力同时消失。
4. C3a 与 C2c strong-climb actor gradient 间歇性冲突，但 task sample balance 本身不足以保证 optimization balance。
5. 当前没有证据表明需要修改 plant、reward 或评估包线。

## 7. 当前代码中与遗忘实验直接相关的部分

### `straight_flight_env.py`

主要类和配置：

- `FlappingBotStraightFlightDeLaurierMeasuredPureRLC2cEnvCfg`
- `FlappingBotStraightFlightDeLaurierMeasuredPureRLC3aEnvCfg`
- C1/C2c/C3a task-family 与 phase 标签
- exact C2c rehearsal path/query
- adaptive sampling 配置
- actor distillation mask
- task-aware PPO 配置

当前 C3a 配置显式启用：

```text
pure_rl_warm_start_guard_enabled = True
pure_rl_actor_gradient_probe_enabled = True
pure_rl_task_aware_ppo_enabled = True
```

### `rl_training_utils.py`

这里包含项目本地训练扩展：

- `PpoWarmStartGuard`
- `ActorGradientConflictProbe`
- `TaskAwarePpoAdapter`
- actor-only source policy distillation wiring
- CSV 日志和 runner 集成

`TaskAwarePpoAdapter` 当前做了：

- 按 C1/C2c/C3a 分别处理 advantage 分布；
- 对 actor loss 使用精确的 `0.15/0.35/0.50` task 权重；
- 对 strong-climb 和 active-turn phase 做分层采样/诊断；
- critic 仍使用普通 mixed PPO；
- 输出 `task_aware_ppo.csv`；
- 保留 task/phase gradient norm、cosine、KL 和样本统计。

它是实验性项目扩展，不等于 C3a 已晋级。

### 训练和评估入口

关键脚本：

- `scripts/flapping_rl/train_and_watch.py`
- `scripts/flapping_rl/watch_and_eval.py`
- `scripts/flapping_rl/pure_rl_longitudinal_eval.py`
- `scripts/flapping_rl/pure_rl_spatial_eval.py`
- `scripts/flapping_rl/build_pure_rl_retention_matrix.py`
- `scripts/flapping_rl/checkpoint_selection.py`

注意：仓库根 `AGENTS.md` 的默认可修改范围只列出 `scripts/flapping_px4/`，没有自动授权修改 `scripts/flapping_rl/`。如下一方案确实需要改 `scripts/flapping_rl/`，必须先向用户说明具体文件和理由并取得明确批准。读取和运行这些脚本不等于获准修改。

## 8. 已尝试的实验和结论

下面按因果顺序记录。数字均来自 frozen fresh-process evaluation，除非明确标为在线诊断。

### 8.1 第一版 C3a：能转弯，但遗忘强爬升

- C3a turn grid 可以通过。
- C2c overall survival `92.86%`，climb success `83.33%`。
- `+12 deg` 只有 `8/16`，失败全部为 tilt termination。
- `-12` 到 `+8 deg` 各 slice 仍为 `16/16`。

结论：不是 C2c 整体消失，而是正强爬升局部功能被破坏。

### 8.2 修复为 exact authoritative C2c rehearsal

发现并修复：原 C3 “straight rehearsal” 没有精确重现 C2c 纵向路径/恢复合同。C3 现在维护独立 `_pure_rl_c2c_rehearsal_path` 并复用 authoritative C2 sampler/query。

随后从已有 C3 optimizer 做 50-iteration continuation：

- C2c survival 降至 `85.71%`；
- climb 降至 `66.67%`；
- `+12 deg` 降至 `4/16`；
- C3a 仍能通过。

结论：exact rehearsal 必要，但从已经漂移的 C3 optimizer 继续训练不是恢复旧能力的 authority 路径。

### 8.3 actor-only distillation，系数 0.05

从 promoted C2c `model_550.pt` weights-only 重新开始，冻结 source actor；只在当前 on-policy C1/C2c observation 上约束 actor action mean，C3a row、critic 和 action std 不约束。

iteration 99：

- C3a `96/96`；
- C2c survival `95.54%`；
- C2c climb `89.58%`；
- `+12 deg` `11/16`；
- C1 tail-limit fraction `11.64%`。

结论：distillation 明显有效且未阻止 C3a 学习，但仍差 C2c climb gate 和 C1 tail gate，因此不能晋级。

### 8.4 adaptive sampling + distillation 0.05，200 iterations

scheduler 根据在线 episode 信号动态调整 C1/C2c/C3a 和 strong-climb 概率。最终分配约为：

```text
C1/C2c/C3a = 0.2413/0.4494/0.3093
strong climb = 0.5757
```

最佳 iteration 175：

- C3a `96/96`；
- C2c survival `93.75%`；
- climb `85.42%`；
- `+12 deg` `9/16`。

iteration 199 又退化。

结论：把近一半 env 分给 C2c 仍不能通过，说明“增加旧任务 env”不是充分条件。在线 strong-climb success 还存在 metric mismatch：到达 recovery 可能被在线逻辑计为成功，但 frozen grid 会继续捕捉 post-recovery tilt。

### 8.5 gradient probe 暴露首个 PPO update 冲击

最初 weights-only run 的首个 rollout 同步处于 entry-only 状态，没有 active strong-climb/turn transition。新 Adam optimizer 与 adaptive LR 的首个 PPO update 造成 actor parameter displacement：

```text
0.38454
```

held-out C1/C2c 泛化随即崩坏，之后才出现间歇性 strong-climb/C3a gradient conflict。

结论：首次大更新是一个单独的启动问题，不能把它与长期遗忘混为一谈。

### 8.6 bounded warm start 修复启动冲击，但长期仍漂移

已实现：

- 前 3 个 rollout 只收集、不更新；
- 接下来 10 次 update 只用 1 epoch；
- LR 从 `1e-5` 线性升至 `5e-5`；
- 后续恢复原 epoch 数，固定 `5e-5`；
- warm-up 阶段 actor update norm 超过 `0.10` 时 fail closed。

smoke 中首个真实 update displacement 降到：

```text
0.00808
```

早期 guard 曾错误地在 warm-up 结束后仍执行 `0.10` 限制，训练于 iteration 288 因 `0.100094 > 0.100000` 停止；该 bug 已修为只约束 warm-up。随后续跑至 iteration 499。

500-iteration 结果：

- C3a 到 iteration 475 均为 `96/96`；iteration 499 退化为 `60/96`；
- C1 在 450/475 虽生存 `16/16`，tail-limit fraction 为 `61.53%/54.54%`；
- C2c 在 450/475 survival `85.71%/87.50%`；climb `66.67%/70.83%`；`+12 deg` `0/16` 和 `2/16`。

结论：启动冲击已修复，长期 shared-policy PPO drift 未修复；另一次 500-iteration 调参没有价值。

### 8.7 task-aware PPO，最新已提交实验

运行目录：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-18_10-41-16_pure_rl_c3a_task_aware_seed0_100iter
```

设置：task-wise advantage、actor 精确 `15/35/50` 权重、phase-stratified mini-batch、mixed critic、bounded warm start；没有额外 distillation。

首个 update displacement 为 `0.0123`。Frozen evaluation：

| Iteration | C3a | C1 tail-limit | C2c survival | C2c climb | `+12 deg` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 96/96 | 5.91% | 93.75% | 85.42% | 9/16 |
| 50 | 96/96 | 2.77% | 89.29% | 75.00% | 4/16 |
| 75 | 96/96 | 7.59% | 89.29% | 75.00% | 4/16 |
| 99 | 96/96 | 8.29% | 91.96% | 81.25% | 7/16 |

诊断日志显示：

- strong-climb 与 C3a actor gradient cosine 为负的 update 约占 `42.3%`；
- strong-climb per-update KL 均值约 `0.0273`；
- 其他 task/phase 约 `0.015`。

结论：task-aware loss 保住了 C1 和 C3a，却没有单独保住 C2c strong-climb。gradient conflict 确实存在，但不是每次都冲突；仅做 task/mini-batch balance 仍不够。当前没有可晋级 checkpoint。

本地 frozen-suite 汇总位于：

```text
logs/rsl_rl/flapping_bot_straight_flight/2026-08-18_c3a_task_aware_25_99_c3a_eval/eval/summary.csv
logs/rsl_rl/flapping_bot_straight_flight/2026-08-18_c3a_task_aware_25_99_c1_eval/eval/summary.csv
logs/rsl_rl/flapping_bot_straight_flight/2026-08-18_c3a_task_aware_25_99_c2c_eval/eval/summary.csv
```

这些是本地生成证据，不应被提交到 Git。

## 9. 已排除或暂不支持的解释

- **“C3 rehearsal 完全没有 C2”**：早期成立，现已由 exact authoritative C2c rehearsal 修复。
- **“只要增加 C2 env 就能解决”**：adaptive run 已把 C2c 提到约 45%，仍失败。
- **“只是首次 optimizer shock”**：bounded warm start 修复了首次大更新，但 500 iterations 仍长期漂移。
- **“是 observation normalizer 漂移”**：最新 task-aware run 的 actor/critic empirical normalizer 没有启用。
- **“C3a 学不会”**：多个实验均能达到 `96/96`；核心 gate 是 retention。
- **“修改 reward、plant 或放宽 frozen grid 即可”**：这会改变问题和晋级 authority，不是允许的修复。
- **“继续坏 optimizer 能恢复”**：已有 exact-rehearsal continuation 反而恶化。

## 10. 当前最可能的机制

### 10.1 Sample balance 不等于 optimization balance

即使 env 比例是 `15/35/50`，不同任务的 advantage scale、episode/termination 分布、active phase 数量和 gradient norm 仍不同。普通 mixed PPO 不保证最终 actor gradient 也是 `15/35/50`。

task-aware PPO 已修正一部分 loss/mini-batch 不平衡，但 strong-climb 仍显示更高 KL 和频繁负 cosine，说明局部功能仍会被 C3a update 改写。

### 10.2 当前 on-policy distillation 有 state coverage collapse 风险

现有 distillation 只在 student 当前访问到的 C1/C2c observation 上保护 action mean。student 一旦在强爬升早期 tilt，成功 teacher 会访问的后半段和 recovery/post-recovery 状态就不再进入 student rollout，真正关键的状态反而失去保护。

### 10.3 PPO 的局部 trust region 不保护长期 teacher-relative drift

PPO clipping 或当前 rollout 的 KL 只限制相邻 update。很多小 update 可以累计远离 promoted C2c teacher。因此必须区分：

- current-policy-relative PPO KL；
- promoted-teacher-relative C1/C2/strong-climb KL。

后者目前还不是硬约束。

## 11. 推荐下一步计划

不要直接实现完整 GEM、PCGrad、EWC 或 residual network。先用最小实验确认 task-aware PPO 与已表现有效的 distillation 是否互补。

### Phase A：101-iteration task-aware + distillation 0.05

保持下列条件不变：

- 同一 promoted C2c `model_550.pt`；
- weights-only 和 fresh optimizer；
- CPU-native、256 env、16 mini-batches；
- task mix `15/35/50`；
- strong climb probability `0.5`；
- bounded warm start；
- task-aware PPO；
- seed 0；
- frozen grids、reward、plant 和 action/observation contract 不变。

唯一相对最新 task-aware run 的变量是：

```text
actor distillation coefficient = 0.05
```

运行 101 iterations 并保存每 25 iterations，确保得到 `model_25/50/75/100.pt`。按 fresh process 分别评估 C3a、C1、C2c。判定规则：

- 若存在合法相邻 pair（例如 50/75 或 75/100）三套都通过，停止扩展方法并走正式 promotion helper；
- 若 C1/C3a 通过而 C2c strong climb 仍失败，进入 Phase B；
- 若 C3a 被 distillation 阻止，则不要增大系数，先检查 loss/gradient 日志。

建议命令：

```bash
cd /home/zn/IsaacLab/.worktrees/native-multibody-rl
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

SOURCE_RUN="2026-08-12_17-12-48_pure_rl_c2c_seed0_resume500_to550"
SOURCE_CHECKPOINT="/home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/${SOURCE_RUN}/model_550.pt"

TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
  --run-name pure_rl_c3a_task_aware_distill005_seed0_101iter \
  --native-cpu \
  --num-envs 256 \
  --agent-num-mini-batches 16 \
  --max-iterations 101 \
  --save-interval 25 \
  --train-only \
  --seed 0 \
  --load_weights_only \
  --load_run "$SOURCE_RUN" \
  --checkpoint model_550.pt \
  --source-stage c2c \
  --source-checkpoint-path "$SOURCE_CHECKPOINT" \
  --c2c-strong-climb-probability 0.5 \
  --actor-distillation-coefficient 0.05 \
  --headless
```

长训练若按用户要求后台启动，必须记录 PID、stdout/stderr 路径和 resolved run directory；确认进程存活、日志完成环境构建且产生首个正常 update 后即可停止监看。不要把“进程启动”写成“实验通过”。

### Phase B：先做 fixed teacher anchor 的离线判别

只有 Phase A 仍失败时才进行。

1. 用 promoted C1/C2 teacher 在 **训练分布** 上生成成功 episode observation anchors。
2. C2 anchors 覆盖 `-12,-8,-4,0,+4,+8,+12 deg`，并对 `+8/+12` 增加 entry、pitch establishment、maximum climb、minimum airspeed、recovery entry、recovery、post-recovery 状态。
3. 保存 raw observation 和 teacher Gaussian policy target；不要保存旧 PPO advantage/return 用于 replay。
4. 先离线计算 teacher-relative KL，比较已知好 source、task-aware 25、50、75、99 和已知坏的 late checkpoint。
5. 只有 strong/recovery anchor KL 能稳定区分好坏模型时，才把 anchor 约束接入训练。

重要防泄漏规则：anchor 来自训练分布的独立 teacher rollout，不能从 frozen promotion/evaluation grid 抽取，否则会污染 held-out gate。

### Phase C：actor-only strong-climb GEM-style constraint

若 Phase B 的 KL 判别成立：

- 先只对 strong-climb/recovery anchor 建立 retention gradient；
- 保护完整 Gaussian actor（mean 和 std），critic 保持普通 mixed PPO；
- 以非对称约束阻止 C3 update 一阶增加 old-skill retention loss；
- 先跑 unit test 和短 smoke，再跑 25-iteration pilot，最后才跑 101 iterations；
- 仅在 C1 或 general C2 gate 仍失败时增加对应 constraint，避免一次实现三个约束。

该方法比 PCGrad 更贴合当前目标：C3 是要学习的新任务，C1/C2 是不允许退化的旧任务，二者不是对称目标。

### Phase D：最后才考虑 residual actor

如果 fixed anchors + actor-only gradient constraint 仍不能同时通过三套 frozen grid，再考虑：

```text
frozen C2 base policy + C3 residual action
```

这属于结构性改变，需要单独设计审查和 ADR。EWC、Progressive Network、每课程一个独立 policy 目前都不是首选。

## 12. 验证和完成标准

### 12.1 修改代码前

- 确认当前 checkpoint lineage 和 source 文件真实存在；
- 确认分支/commit，没有覆盖用户改动；
- 为 proposed files、假设、测试和停止条件给用户一个简短计划；
- 设计敏感或扩展到未授权文件时先等批准。

### 12.2 最低代码验证

当前 task-aware 提交的 focused test 是：

```bash
cd /home/zn/IsaacLab/.worktrees/native-multibody-rl
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_actor_gradient_conflict_probe.py \
  tests/test_pure_rl_spatial_env_contract.py
```

当前提交结果：`23 passed`。

每次修改至少还应执行：

```bash
git diff --check
git status --short --branch
```

涉及 environment runtime contract 时，应补 fresh-process CPU-native targeted runtime gate；不能用 direct-GPU 或 mock-only 结果替代。

### 12.3 实验完成的定义

必须分别报告：

1. 训练进程是否正常退出；
2. checkpoint lineage 是否正确；
3. C3a frozen suite；
4. C1 frozen retention；
5. C2c signed-slope/termination frozen retention；
6. 是否存在合法相邻 checkpoint pair；
7. promotion helper 是否接受。

缺任一项都不能写“C3a 已解决”或“已晋级”。

## 13. 容易踩的坑

1. **必须使用 CPU-native authority。** native wing constraint 不支持 direct-GPU PhysX；GPU 结果不能替代晋级证据。
2. **每个 evaluator 使用 fresh process。** 同一进程关闭后再构造第二个完整 environment 可能在 scene creation 阶段挂住。
3. **设置 `TERM=xterm`。** 否则 Isaac/终端初始化可能出现 tabs/terminfo 相关错误。
4. **训练时用 `--train-only`。** 不要让 concurrent evaluator 与训练争用 CPU/Isaac Sim；authority evaluation 事后串行运行。
5. **wrapper 的退出状态要核实。** 某些 `isaaclab.sh -p` 路径可能没有可靠传播 child exit status；检查 traceback、结果 JSON 和 gate 字段，不要只看 shell 最后一行。
6. **`model_99` 不是 `model_100`。** 需要 25-iteration 相邻证据时运行 101 iterations。
7. **weights-only 不是 resume。** Sequential C3a 从 C2c source actor 复制权重；显式
   `c3a_joint_from_c1_v1` 路线从已选 C1 source actor 复制权重。两者都必须创建 fresh optimizer，且不得同时使用 `--resume`。
8. **不要 replay stale PPO transitions。** PPO 仍需 current-policy on-policy rollout；anchor memory 只用于 teacher behavior/KL，不用于旧 PPO surrogate。
9. **在线 success 不是 frozen success。** 当前 scheduler 可能漏掉 post-recovery tilt；不能根据在线曲线宣布 retention。
10. **不要从 frozen grid 采 anchor。** 这会泄漏 promotion test 状态。
11. **不要改 plant/reward/grid 来消除失败。** 这会破坏基线和问题定义。
12. **不要把 generated artifacts 提交。** 包括 checkpoints、logs、videos、portable cache、PID、临时 eval 和大型二进制。
13. **不要清理陌生 untracked files。** 它们可能是用户或其他实验产物。
14. **不要开始 C3b。** 在 C3a 与 C1/C2c 有完整相邻通过证据前，C3b/C3c checkpoint 没有合法 lineage。
15. **当前只有 seed 0。** 即使晋级，也不能据此声称跨 seed 鲁棒、真实飞行有效或解决了一般 continual RL 问题。
16. **normalizer 结论只适用于当前配置。** 若未来启用 empirical normalization，必须把 normalizer state 纳入 teacher/source 和 retention audit。
17. **warm-start iteration 与 optimizer step 不再一一对应。** 前 3 个 rollout 无 optimizer step，读日志时不能把 iteration 当更新次数。
18. **现有 distillation 只保护 mean/current states。** 在实现新方法前不要误写成已经保护完整 Gaussian 和固定成功轨迹。

## 14. 允许和禁止的修改范围

### 默认允许

- `source/flapping_bot/` 内项目代码；
- 项目相关 `tests/`；
- `docs/`；
- `scripts/flapping_px4/`。

### 需额外明确批准

- `scripts/flapping_rl/` 中任何具体修改；
- 上游 `source/isaaclab/`、`source/isaaclab_tasks/`、`source/isaaclab_assets/`；
- 共享 Isaac Lab 应用/API；
- reward、plant、质量属性、控制器增益、action/observation contract 或 frozen evaluation grid 的改变；
- 新的网络结构或 residual policy 架构。

2026-08-27 用户已明确批准为 `c3a_joint_from_c1_v1` 修改
`scripts/flapping_rl/train_and_watch.py`、对应测试和文档，并在测试通过后启动训练；该批准不扩展到
environment、upstream、reward、plant、frozen grid 或其他算法修改。

随后用户基于同轨迹频率动作诊断明确要求执行 requested-frequency smoothness 方案，因此批准范围扩展到
`pure_rl_reward.py`、正式 `straight_flight_env.py`、对应 launcher/tests 和一份新 ADR。该批准只覆盖
默认关闭的 governor 前频率请求差分惩罚及其单变量 joint route；不覆盖 plant、555维观测、governor、
任务采样、网络、PPO schedule 或 frozen grid 的改变。

平方差分实验完成后，用户进一步批准直接在 reward 中惩罚快速跳变，并要求修改、验证后启动训练。
本次范围只增加同一 governor 前频率请求差分项的 L1 total-variation 模式、显式 launcher route、测试和
决策文档；原 joint 与平方差分 route 保持可复现，仍不覆盖 plant、观测、governor、网络或 frozen grid。

所有实现都应优先放入 `source/flapping_bot/` 的适当层，保持 batch 维度、device、dtype 和已有 baseline 选择路径。

## 15. 尚未解决的决定

- task-aware + distillation 0.05 已完成 seed-0、101-iteration 实验并确认不足以跨过完整 retention gate；结果见第 19 节。
- 方法 2 的大 actor 实验也已完成 seed-0、101-iteration 训练和冻结评估；它改善了 C1 retention，
  但没有恢复 C2c strong-climb，结果见第 21 节。
- 用户已批准以 C1 稳定飞行 checkpoint 初始化、联合学习 C1/C2c/C3a 的路线作为当前主实验；
  冻结定义见第 22 节和 `ADR-2026-08-27-pure-rl-c3a-joint-from-c1.md`。
- fixed teacher anchors 的最小状态数量和 phase 配额；应先由离线 KL 区分能力决定，而不是先写大型 buffer 框架。
- actor Gaussian std 是否对当前遗忘有实质贡献；现有实验没有保护 std。
- GEM-style 约束的具体数值容差、投影实现和 per-task/per-phase 粒度；Phase B 前不应过早冻结设计。
- 完整通过后是否需要额外 seeds 才允许进入 C3b；当前晋级合同主要冻结相邻 checkpoint 和旧课程 retention，但科研结论需要更强的 robustness 计划。

## 16. Exact next task

等待已启动的 201-iteration、seed-0
`c3a_joint_from_c1_requested_frequency_total_variation_v1` 正常结束，不要并发运行 evaluator。
训练完成后对
`model_50/75/100/125/150/175/200.pt` 分别运行 fresh CPU-native C3a、C1 和 C2c frozen suites。
然后重复固定 `+12 deg` 同轨迹响应比较，检查 requested-frequency sign flip、phase harmonic、实际频率和
高度响应。没有三套均通过的合法相邻 pair 时不得晋级或开始 C3b。

## 17. Suggested skills

- `right-size-coding`：限制 continual-learning 实现规模，优先完成可证伪的小实验，避免直接引入完整框架。
- `get-available-resources`：仅在准备启动新的 CPU-native 长训练前检查 CPU、内存和磁盘，避免与残留 Isaac Sim 进程争用。
- `checkpointed-subagent-supervision`：仅在把长训练或多套 fresh-process evaluation 委派给子 agent 时使用，并要求 checkpoint、日志和 frozen-suite 证据回传。

## 18. 给接手 agent 的一句话

不要把问题重新简化成“多给一些 C2 样本”。Sequential C2c-to-C3a 的多种 retention 方法和 wider actor
均未恢复 `+12 deg` strong climb。当前主实验从同一 lineage 的稳定 C1 source 开始，让 C2c 和 C3a
共同形成表示，以区分 sequential path-dependent overwrite 与 joint optimization interference。

## 19. 2026-08-26 Phase A 实际结果

训练目录：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-26_19-28-45_pure_rl_c3a_task_aware_distill005_seed0_101iter
```

lineage 和设置：promoted C2c `model_550.pt` weights-only、fresh optimizer、seed 0、101 iterations、256 env、16 mini-batches、task-aware PPO、bounded warm start、strong-climb `0.5`、actor distillation `0.05`、CPU-native、train-only。冻结评估由 `evaluate_pure_rl_c3a_checkpoint.py` 对 `model_75.pt` 和 `model_100.pt` 逐 checkpoint、逐 suite 使用 fresh process 完成。最终汇总：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-26_19-28-45_pure_rl_c3a_task_aware_distill005_seed0_101iter/
eval_authority_75_100/checkpoint_evaluation.json
```

| Iteration | C3a | C1 tail-limit | C1 gate | C2c survival | C2c climb | `+12 deg` | `+12` tilt | C2c gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 75 | `96/96` | `8.11%` | pass | `92.86%` | `83.33%` | `8/16` | 8 | fail |
| 100 | `96/96` | `11.44%` | fail | `91.07%` | `79.17%` | `6/16` | 10 | fail |

观察事实：两者均完整通过 C3a；iteration 75 保留 C1 但未达到 C2c survival/climb gate；iteration 100 的 C1 tail-limit 和 C2c 均失败。没有三套均通过的合法相邻 checkpoint pair，Phase A 不可晋级。结果继续把遗忘定位在正强爬升，且从 75 到 100 仍在退化。该结果不证明网络容量是根因，只证明当前 Phase A 组合不足。

## 20. 用户批准的方法 2：actor capacity

代码核对显示正式 straight-flight policy 的实际结构是：

```text
actor  = [256, 128]
critic = [256, 128]
```

因此方法 2 采用同深度双宽 actor `[512,256]`，而不是基于先前推测的 `[512,256,128]`；critic 保持 `[256,128]`，避免同时改变 value-function capacity。默认注册配置不变，大 actor 仅由显式 `--c3a-large-actor` 启用。

跨尺寸 warm start 采用受限 Net2Wider：每个旧 hidden unit 复制一次，下一层出边按 `0.55/0.45` 拆分，两个分支之和严格等于旧权重。这样 iteration 0 的 actor action mean 与 promoted C2c source 数值等价，同时两个复制分支的反向梯度不同，可在训练中分化。critic、`log_std` 和其他同形状 state 原样加载；只接受两层 actor 的精确二倍扩容，其他 key、深度、非 actor shape 或非二倍尺寸不匹配均 fail closed。

方法 2 不改变：plant、reward、action/555-observation contract、task mix、strong-climb quota、PPO schedule、distillation coefficient、frozen grid 或晋级门槛。正式训练和评估结果见下一节。

## 21. 2026-08-26 方法 2 actor-capacity 实际结果

训练目录：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-26_20-38-23_pure_rl_c3a_taskaware_distill005_largeactor512x256_seed0_101iter
```

保存配置和 `curriculum_source.json` 确认：promoted C2c `model_550.pt` weights-only、fresh optimizer、
seed 0、101 iterations、256 env、16 mini-batches、task-aware PPO、bounded warm start、strong-climb
`0.5`、actor distillation `0.05`、CPU-native；actor 为 `[512,256]`，critic 保持 `[256,128]`。
训练生成 `model_0/25/50/75/100.pt` 并正常完成 iteration 100。冻结评估结果位于：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-26_20-38-23_pure_rl_c3a_taskaware_distill005_largeactor512x256_seed0_101iter/
eval_authority_75_100/checkpoint_evaluation.json
```

| Iteration | C3a | C1 tail-limit | C1 gate | C2c survival | C2c climb | `+12 deg` | `+12` tilt | C2c gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 75 | `96/96` | `1.76%` | pass | `92.86%` | `83.33%` | `8/16` | 8 | fail |
| 100 | `96/96` | `3.03%` | pass | `89.29%` | `75.00%` | `4/16` | 12 | fail |

观察事实：两个 checkpoint 均通过 C3a 和 C1，但都未通过 C2c，因此没有三套均通过的合法相邻 pair。
与 Phase A 小 actor 相比，iteration 75 的 C2c aggregate 和 `+12 deg` 结果相同；iteration 100 的 C1
tail-limit 从 `11.44%` 改善到 `3.03%`，但 C2c survival 从 `91.07%` 降到 `89.29%`、climb 从
`79.17%` 降到 `75.00%`，`+12 deg` 从 `6/16` 降到 `4/16`。因此本次单 seed 实验支持的结论是：
增加 actor capacity 改善了 C1 retention，但没有解决 C2c strong-climb 遗忘；方法 2 不可晋级。

## 22. 2026-08-27 主实验：C1 初始化的 C1+C2c+C3a joint training

### 22.1 冻结问题和单变量

该实验不是随机初始化，也不是 C2c-to-C3a sequential fine-tuning。它从已选稳定 C1 checkpoint 出发，
在现有 C3a mixed environment 中同时学习 authoritative C2c 和 C3a：

```text
C1 stable-flight model_1300.pt
        -> C1 15% + C2c 35% + C3a 50% joint optimization
```

与之前 task-aware sequential 实验相比，唯一核心变量是 source initialization 从 promoted C2c
`model_550.pt` 改为已选 C1 `model_1300.pt`。正式 source 为：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-07_05-05-44_curriculum1_overnight_seed2/model_1300.pt
```

现有 seed-0 C2a-to-C2c lineage 也从该 checkpoint 开始，因此 joint/sequential 比较共享同一 C1 起点。

### 22.2 冻结训练合同

```text
route                         c3a_joint_from_c1_v1
task                          measured CPU-native C3a
initialization                C1 model_1300.pt weights-only
optimizer / iteration         fresh / start at 0
seed                          0
environments                  256
steps per environment         48
mini-batches                  16
launcher iterations           201, to materialize model_200.pt
save interval                 25
C1/C2c/C3a                    0.15 / 0.35 / 0.50
C2c level/climb/descent       0 / 2/3 / 1/3
strong climb within climb     0.5
task-aware PPO                enabled
bounded warm start            enabled
gradient probe                enabled
actor / critic                [256,128] / [256,128]
distillation                  disabled
adaptive sampling             disabled
recycle on recovery           disabled
wider actor                   disabled
PCGrad/GEM/EWC/residual        disabled
watcher                       disabled during training
```

运行命令：

```bash
cd /home/zn/IsaacLab/.worktrees/native-multibody-rl
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

SOURCE_RUN="2026-08-07_05-05-44_curriculum1_overnight_seed2"
SOURCE_CHECKPOINT="$PWD/logs/rsl_rl/flapping_bot_straight_flight/${SOURCE_RUN}/model_1300.pt"

TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0 \
  --run-name pure_rl_c3a_joint_from_c1_seed0_201iter \
  --c3a-joint-from-c1 \
  --load_run "$SOURCE_RUN" \
  --checkpoint model_1300.pt \
  --source-stage c1_straight \
  --source-checkpoint-path "$SOURCE_CHECKPOINT" \
  --headless
```

### 22.3 冻结评估合同

训练结束后按 fresh process 评估：

```text
model_50.pt
model_75.pt
model_100.pt
model_125.pt
model_150.pt
model_175.pt
model_200.pt
```

每个 checkpoint 必须完成 96-case C3a v2、16-case C1 和 112-case C2c v2。需要两个间隔 25 iterations
的相邻 checkpoint 三套同时通过，并由现有 promotion helper 接受。单 seed 通过只支持该初始化和优化轨迹
存在兼容 actor，不构成跨 seed、sim-to-real 或一般 continual-learning 结论。

## 23. 2026-08-27 启动记录

测试和静态检查完成后，正式训练已于 2026-08-27 08:00 Asia/Shanghai 后台启动：

```text
launcher PID: 1105494
PID file:     /home/zn/IsaacLab/.worktrees/native-multibody-rl/c3a_joint_from_c1_training.pid
stdout log:   /home/zn/IsaacLab/.worktrees/native-multibody-rl/c3a_joint_from_c1_training.log
run directory:
/home/zn/IsaacLab/.worktrees/native-multibody-rl/logs/rsl_rl/flapping_bot_straight_flight/
2026-08-27_08-00-35_pure_rl_c3a_joint_from_c1_seed0_201iter
```

启动后已确认：

- launcher、parent Python 和 RSL-RL child 都在独立 session 中存活；
- 实际 child command 为 CPU、256 env、201 iterations、16 mini-batches、save interval 25；
- actor/critic 均显式为 `[256,128]`；strong-climb 为 `0.5`；
- `curriculum_source.json` 记录 `source_stage=c1_straight`、精确 seed-2 C1 `model_1300.pt` 和
  `curriculum_route=c3a_joint_from_c1_v1`；
- 保存的 env 配置记录 distillation `0.0`、adaptive sampling false、task-aware PPO、gradient probe 和
  bounded warm start enabled；
- iterations 0--2 正确跳过 optimizer update；iteration 3 首个真实 update 的 actor displacement 为
  `0.01793 < 0.10`，并已包含 1336 个 active strong-climb transition 和 5893 个 active C3a transition；
- iteration 4/5 继续运行，未见 traceback；此证据只说明启动正常，不是训练完成或 promotion 结果。

本轮代码验证命令：

```bash
TERM=xterm ./isaaclab.sh -p -m pytest -q \
  tests/test_train_and_watch.py \
  tests/test_watch_and_eval.py \
  tests/test_evaluate_pure_rl_c3a_checkpoint.py \
  tests/test_actor_policy_distillation.py \
  tests/test_actor_gradient_conflict_probe.py \
  tests/test_pure_rl_spatial_env_contract.py
```

结果：`122 passed`。`git diff --check` 通过。训练尚未完成，冻结评估尚未运行。

## 24. 2026-08-27 joint频率请求振荡与单变量修复

`c3a_joint_from_c1_v1` 已完成训练及七个checkpoint的fresh-process三套评估。iteration 75--200均通过
C3a和C1，但所有checkpoint都未通过C2c。iteration 175的C2c survival/climb/descent/recovery均为
`100%`，失败来自路径精度：mean absolute height error `0.6609 m > 0.50 m`，p95 absolute height error
`1.9065 m > 1.50 m`。

固定heading 0、reset phase 0、`+12 deg` C2c轨迹对比：joint `model_175.pt`在active slope内产生27次
requested-frequency正负翻转，对应约`3.49 Hz`，与平均实际翼拍`3.43 Hz`一致；重建相位的一阶谐波
解释约76%的请求变化。promoted C2c `model_550.pt`在相同轨迹保持正请求。joint的90%垂直速度响应
延迟为`0.567 s`，baseline为`0.25 s`。证据位于：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-27_08-00-35_pure_rl_c3a_joint_from_c1_seed0_201iter/
response_compare_model550_vs_model175_slope12_h0_p0/
```

已接受默认关闭的修复：在clipped policy frequency request进入2 Hz/s governor之前，新增

```text
(0.5 * (request[t] - request[t-1])) ** 2
```

惩罚。默认weight `0.0`保持旧任务；新route
`c3a_joint_from_c1_requested_frequency_smoothness_v1`固定weight `0.05`，其余与原joint recipe完全相同。
实现、替代方案和验证合同见
`docs/decisions/ADR-2026-08-27-pure-rl-requested-frequency-smoothness.md`。

实现后的纯Tensor reward、launcher和C3合同测试共`178 passed`，`py_compile`与`git diff --check`通过。
完整6环境C3 Isaac runtime gate持续计算10分钟但未结束，已终止且不计为通过；正式训练fresh-process启动
已补足集成证据：

```text
resolved run:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-27_13-53-31_pure_rl_c3a_joint_reqfreqsmooth005_seed0_201iter/

launcher PID file: c3a_joint_reqfreqsmooth005_training.pid
launcher PID:      1449399
stdout/stderr:     c3a_joint_reqfreqsmooth005_training.log
```

`curriculum_source.json`确认`source_stage=c1_straight`、来源为promoted C1 `model_1300.pt`，route为
`c3a_joint_from_c1_requested_frequency_smoothness_v1`。保存后的`params/env.yaml`确认新惩罚weight为`0.05`、
strong-climb probability为`0.5`、task-aware PPO开启、distillation为`0.0`且adaptive sampling关闭；
`params/agent.yaml`确认CPU、16 minibatches和actor/critic `[256, 128]`。训练已产生`model_0.pt`并完成至少
98,304 timesteps；warm-start actor update norm在已检查更新中为`0.0117--0.0256 < 0.10`，日志中已出现
requested-frequency penalty/contribution遥测且未见traceback。此证据只证明训练按指定合同正常启动，不代表
训练完成或通过冻结评估。

## 25. 2026-08-27 平方差分结果与 L1 total-variation 实验

`c3a_joint_from_c1_requested_frequency_smoothness_v1` 的七个 checkpoint 已完成三套 frozen evaluation。
C1 全部通过；C3a 在 iteration 50/75/150/200 通过；C2c 全部未通过。最接近 gate 的 `model_200.pt`
通过 survival、climb、descent、recovery 和 cross-track，但 mean/p95 absolute height error 分别为
`0.54884 m > 0.50 m` 和 `1.57906 m > 1.50 m`。

固定 heading 0、reset phase 0、`+12 deg` 对比中，相对未正则 joint `model_200.pt`，平方差分
`model_200.pt` 将 clipped request RMS delta 从 `0.3632` 降至 `0.3247`，raw saturation 从 `65.29%`
降至 `58.59%`，垂直响应延迟从 `0.8667 s` 降至 `0.7667 s`，全轨迹高度 MAE 从 `1.558 m` 降至
`1.156 m`；但 sign flips 仅从29降至28。promoted C2c在同轨迹为0次flip。因此平方项改善幅值和精度，
却没有消除高频反转。

下一单变量实验改用同一 normalized delta 的 L1 total variation：

```text
abs(0.5 * (request[t] - request[t-1]))
```

显式 route `c3a_joint_from_c1_requested_frequency_total_variation_v1` 固定 weight `0.10`，其余完全复用
C1初始化 joint recipe。平方项和默认 baseline 均不改变。实现与验证合同见
`docs/decisions/ADR-2026-08-27-pure-rl-requested-frequency-total-variation.md`。

实现后的 reward/launcher 单元测试、C1/C2/C3 contracts 共 `183 passed`，`py_compile` 和
`git diff --check` 通过。资源检查时系统有32个可用逻辑核、约80 GB可用内存、约1.0 TB可用磁盘，
CPU pressure为0且没有残留的本项目训练或评估进程。训练已按 train-only fresh-process 启动：

```text
resolved run:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-27_21-21-54_pure_rl_c3a_joint_reqfreqtv010_seed0_201iter/

launcher PID file: c3a_joint_reqfreqtv010_training.pid
launcher PID:      1929498
training PID:      1929539
stdout/stderr:     c3a_joint_reqfreqtv010_training.log
```

`curriculum_source.json`确认route为
`c3a_joint_from_c1_requested_frequency_total_variation_v1`，source为promoted C1 `model_1300.pt`且
`source_stage=c1_straight`。保存后的配置确认CPU、256 environments、16 minibatches、actor/critic
`[256,128]`、strong-climb probability `0.5`、task-aware PPO开启、distillation `0.0`、adaptive sampling
关闭，以及 requested-frequency delta mode `absolute`、weight `0.10`。`model_0.pt`已生成；前三个
rollout-only iteration没有optimizer step，第一个PPO update的actor displacement norm为`0.0178 < 0.10`，
日志已出现L1 penalty/contribution遥测且未见traceback。此证据只证明按冻结合同正常启动，不代表训练完成
或任何checkpoint通过frozen suites。

## 26. 2026-08-28 governor-gap 单变量实验

L1 total-variation run 的固定 `+12 deg` 诊断仍显示 model 150/175/200 各有 24--27 次
requested-frequency sign reversal。保存配置确认 `act_lpf_tau_s=0.0`、`act_rate_limit_per_s=0.0`，当前只有
对称 `2 Hz/s` 的 physical frequency governor 生效。因此本实验不移除 governor，而直接惩罚 governor
无法执行的请求分量：

```text
abs(clipped_requested_frequency_action - governor_applied_frequency_action)
```

新 reward 默认 weight `0.0`；显式 route
`c3a_joint_from_c1_requested_applied_frequency_gap_v1` 固定 weight `0.05`，并关闭原 request-delta 项。
其余完全复用 C1 初始化 joint recipe，尤其保持 cubic flap-frequency penalty `0.04`、governor、任务分布、
网络、PPO schedule 和 frozen gates 不变。该隔离设计先回答 governor-interface mismatch 能否消除 flip；
若 flip 消失但实际频率仍低且 C2c 失败，再另做 flap-frequency penalty ablation。

实现与验证合同见
`docs/decisions/ADR-2026-08-28-pure-rl-frequency-governor-gap.md`。训练启动记录应在 targeted tests、
contract tests、`py_compile`、`git diff --check` 和资源/残留进程检查通过后补充。

### 26.1 实现、验证与启动记录

实现新增默认关闭的 `requested_applied_frequency_action_gap_penalty_weight`、环境 telemetry、显式 launcher
flag 和独立 provenance route；旧 joint、平方差分和 L1-TV 路线保持可复现。验证结果：

```text
reward + launcher focused tests: 104 passed
reward/launcher/watch/eval/distillation/gradient/spatial selected tests: 166 passed
py_compile: passed
git diff --check: passed
```

一个额外扩展合同集合在 pytest collection 时因为未启动 Isaac App 而缺少 `carb`，未计为通过；正式
fresh-process 启动已覆盖环境接线。资源检查记录 32 个逻辑核、约 110 GB 可用内存和约 1.04 TB 可用磁盘，
启动前没有本项目并发训练或评估进程。

训练于 2026-08-28 15:56 Asia/Shanghai 在持久 tmux session `c3a_gap_20260828` 中启动：

```text
launcher PID file: c3a_joint_reqappliedgap005_training.pid
launcher PID:      2606817
training PID:      2606849
stdout/stderr:     c3a_joint_reqappliedgap005_training.log
run directory:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-28_15-56-43_pure_rl_c3a_joint_reqappliedgap005_seed0_201iter/
```

`curriculum_source.json` 确认 route 为
`c3a_joint_from_c1_requested_applied_frequency_gap_v1`，source 为选定 C1 `model_1300.pt` 且
`source_stage=c1_straight`。保存配置确认 CPU、256 env、201 iterations、16 mini-batches、save interval
25、actor/critic `[256,128]`、strong-climb probability `0.5`、distillation `0.0`、adaptive sampling false、
gap weight `0.05`、request-delta weight `0.0`、flap penalty `0.04`，以及 governor 保持对称 `2 Hz/s`。
`model_0.pt` 和首轮 gap penalty/contribution telemetry 已生成，未见 traceback。

Isaac 启动时因系统 inotify watch 配额出现大量 `errno=28` change-watch 告警；磁盘空间充足，且这些告警
没有阻止环境构建、配置保存、`model_0.pt` 或训练迭代。训练已交由 tmux 后台继续，本记录只证明正确
启动，不代表完成或 frozen-suite 结果。

## 27. 2026-08-29 governor-gap 结果与 iteration 225 续训

governor-gap run 的七个 checkpoint 已完成三套 frozen evaluation。`model_200.pt` 是唯一三套同时通过的
checkpoint：C3a survival/event/success 均为 `100%`，C1 tail-limit fraction 为 `0.86%`，C2c survival
为 `98.21%`、climb `100%`、descent `95.83%`、recovery `100%`，mean/p95 absolute height error 为
`0.49187/1.41311 m`。结果位于：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-28_15-56-43_pure_rl_c3a_joint_reqappliedgap005_seed0_201iter/
eval_authority_50_200_reqappliedgap005/checkpoint_evaluation.json
```

由于只有一个 all-suite pass，没有合法相邻 pair，不能据此晋级。随后从该 checkpoint 以原 optimizer
续训到 `model_225.pt`。225 的 C1 和 C2c 均通过，C2c 进一步达到 112/112 survival 和所有 signed-slope
slice 100% success；但 C3a 出现一个 roll-limit termination，survival `95/96 = 98.958%`，因此 C3a
gate 失败。结果位于：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_09-15-55_pure_rl_c3a_joint_reqappliedgap005_seed0_resume200_to225/
eval_authority_225_reqappliedgap005/checkpoint_evaluation.json
```

该结果说明 iteration 200 后继续训练并非单调改善：纵向精度提高的同时，转弯 robustness 在一个 frozen
case 上回退。新的结构实验因此从最后一个三套同时通过的原始 `model_200.pt` 做 weights-only 初始化，
而不从 225 开始。

## 28. 2026-08-29 独立 frequency/tail actor 实验

用户批准从根源隔离 frequency 的 phase 信息路径。实现不是只拆最后一层，而是：

```text
full phase-aware 555 observation -> independent [256,128] tail trunk -> actions 1:4
cycle-averaged phase-fixed 555 observation -> independent [256,128] frequency trunk -> action 0
```

frequency slow observation 用最新实际频率选择 `clamp(round(60/f), 12, 30)` 个 history step，对 sensor 和
governor-applied action 求均值，四元数重新归一化，把 phase 固定为 `(sin,cos)=(0,1)`，然后重复均值以
保持 555 维；15 维 path preview 不变。本轮仍以 60 Hz 计算 frequency request，保留 2 Hz/s governor，
不加入 cycle-level sample-and-hold。完整决策见
`docs/decisions/ADR-2026-08-29-pure-rl-split-frequency-actor.md`。

严格 warm start 将标准 actor 两层 hidden weights 同时复制给两个 trunk，output row 0 给 frequency，
rows 1:4 给 tail；critic 和 `log_std` 原样复制，optimizer 为 fresh。显式 route：

```text
c3a_split_frequency_actor_v1
source stage/checkpoint  c3a_joint / governor-gap model_200.pt
iterations              201
seed/env/minibatches     0 / 256 / 16
task mix                 15/35/50
strong climb             0.5
gap reward               0.05
actor trunks/critic      [256,128] + [256,128] / [256,128]
distillation/adaptive    off / off
governor                 symmetric 2 Hz/s
```

Pure Tensor、warm-start、launcher、watch/evaluator、gradient、distillation、observation 和 C3 contract
定向测试共 `153 passed`；`py_compile` 和 `git diff --check` 通过。16-env、4-iteration fresh CPU-native smoke 正常
注册 `PureRLSplitActorCritic`，从共享 `model_200.pt` 完成严格映射，前三轮只 rollout，iteration 3 首个
PPO update 的 actor displacement 为 `0.0121 < 0.10`，有 active strong-climb/C3a samples，无 traceback。
smoke 只证明运行接线，不是性能或消除 flip 的证据。正式训练启动记录见下一小节。

### 28.1 正式训练启动记录

训练于 2026-08-29 10:07 Asia/Shanghai 以脱离终端的 session 启动：

```text
launcher PID file: c3a_split_frequency_actor_training.pid
launcher PID:      3222196
stdout/stderr:     c3a_split_frequency_actor_training.log
run directory:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/
```

启动核验确认 launcher 的 PPID 为 1、session ID 为自身 PID，因此关闭当前终端不会终止训练。
`curriculum_source.json` 确认 route 为 `c3a_split_frequency_actor_v1`、policy class 为
`PureRLSplitActorCritic`，source stage 为 `c3a_joint`，并严格指向上述三套 frozen suite 同时通过的原始
governor-gap `model_200.pt`。运行时配置确认 CPU、256 env、201 iterations、16 mini-batches、save
interval 25、strong-climb probability `0.5`、gap reward `0.05`，以及 actor 两个独立 `[256,128]`
trunk 和 `[256,128]` critic。

日志确认共享 actor 到独立 frequency/tail trunk 的严格 warm-start 已执行，warm-start guard 前三轮只
rollout，iteration 3 首个 PPO update 正常完成，actor update norm 为 `0.0211 < 0.10`；`model_0.pt`
和配置/provenance 文件均已保存，未见 traceback。训练现已交由后台继续；本记录只证明配置正确且首个
可训练更新成功，不代表训练完成、frequency flip 已消除或任何 checkpoint 通过 frozen suites。

## 29. 2026-08-29 split actor C3a 正式晋级

正式 run：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/
```

iteration 50--200 的七个 checkpoint 已在独立 fresh CPU-native 进程中完成 C3a、C1 和 C2c frozen
evaluation。`model_175.pt` 与 `model_200.pt` 构成合法相邻全套通过 pair；标准 promotion helper 返回
`promoted=true`、`checkpoint=model_200.pt`、`forgetting=false`。因此：

```text
C3a promoted checkpoint:
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/model_200.pt
policy class: PureRLSplitActorCritic
load mode: weights-only
```

关键冻结结果：

| Checkpoint | C3a survival / success | C1 success / tail-limit | C2c survival / climb / recovery | `+12 deg` | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `model_175.pt` | 1.000 / 1.000 | 1.000 / 0.08658 | 0.96429 / 0.91667 / 0.96429 | 12/16 | PASS |
| `model_200.pt` | 1.000 / 1.000 | 1.000 / 0.06267 | 1.000 / 1.000 / 1.000 | 16/16 | PASS |

完整证据和标准晋级结果分别位于：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/
eval_authority_50_200_split_actor/checkpoint_evaluation.json

logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/
eval_authority_50_200_split_actor/promotion_result_175_200.json
```

固定 `+12 deg` deterministic actor-mean 轨迹上，`model_200.pt` 在 active climb 中没有大幅 frequency
request 符号翻转，mean absolute step delta 为 `0.01346`；这一结论只覆盖已检查的固定轨迹，不等同于
多 seed 或实飞鲁棒性证明。正式解释和完整限制见
`docs/audits/2026-08-29-pure-rl-c3a-split-actor-promotion.md`。

后续课程继续采用累积、fail-closed 晋级：C3b 即使自身 PASS，只要 C1、C2c 或 C3a 任一 frozen suite
FAIL 或缺失证据，就不能晋级；C3c 同理必须同时保留 C1、C2c、C3a 和 C3b。下一步只做该
`model_200.pt` 的 zero-shot C3b grid evaluation，再根据失败模式选择 C3b 训练 recipe；尚未启动 C3b
训练。

## 30. 2026-08-29 C3b v3 合同与首轮 joint training

C3b v2 的 loiter template 在 20 s episode 内结构性不可达：确定性 entry 后 final event 为 `277.5 m`，
需要平均 `13.875 m/s`，高于 `7 m/s` command 和 `12 m/s` path design speed。其他模板 final event 约为
`125--133 m`。因此不能把 v2 的 16/16 loiter failure 归因于 actor。

已接受 `pure_rl_spatial_c3b_v3`：只把两段 loiter 总长度从 `260 m` 缩到 `110 m`，确定性 final event
变为 `127.5 m`；20 s episode、176-case template/slope/turn/heading/phase grid 和所有 gate 均不变。每次
spatial evaluation 现在还写入 compact per-case JSON，保留 case 输入、termination/event completion 和
误差摘要，不复制完整 step trace。决策见
`docs/decisions/ADR-2026-08-29-pure-rl-c3b-v3-joint-training.md`。

promoted C3a split `model_200.pt` 的 corrected zero-shot v3 结果：

```text
overall survival / event completion / success: 0.81818 / 0.79545 / 0.79545
mean horizontal / vertical error:             0.29894 / 0.25152 m
p95 horizontal / vertical error:              0.73217 / 1.01995 m
template success 3/4/5/6/7/8/9:              12/16, 16/16, 16/32, 32/32,
                                                 16/32, 32/32, 16/16
promotion gate:                                FAIL
```

结果位于：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_10-07-47_pure_rl_c3a_split_frequency_actor_seed0_201iter/
eval_zero_shot_c3b_v3_split_actor_model200/model_200.json
```

v3 loiter 16/16 通过，证明旧失败来自 horizon；当前主要缺口是含 climb 的顺序组合，template 5 和 7
各 16/32，且所有 32 个 termination 都集中在这两类。第一轮训练冻结为显式 route
`c3b_split_frequency_actor_v1`：promoted C3a weights-only、fresh optimizer、split actor、bounded warm
start、task-aware PPO `15/20/15/50`、C2c strong climb `0.5`、gap reward `0.05`、seed 0、256 env、16
mini-batches、251 iterations、save interval 25；distillation/adaptive/PCGrad/GEM 和新 reward 全部关闭。

定向 Pure Tensor/contract/launcher/evaluator tests 首轮 `204 passed`；随后 runtime 启动发现 C3b seed-0
的 256-row random batch 在一次 replacement 后仍可能包含低于 `0.05 m` 的 descent path，环境按合同
fail closed，未生成 run。sampler 已改为只对 invalid rows 做至多 16 次确定性 rejection resampling；新增
seed-0、256-row 回归后相关集合 `155 passed`，`py_compile` 和 `git diff --check` 通过。失败启动日志保留为
`c3b_split_frequency_actor_failed_start.log`，不能当作训练证据。

正式训练于 2026-08-29 14:30 Asia/Shanghai 脱离终端启动：

```text
launcher PID file: c3b_split_frequency_actor_training.pid
launcher PID:      3928520
training PID:      3928573
stdout/stderr:     c3b_split_frequency_actor_training.log
run directory:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_14-30-44_pure_rl_c3b_split_frequency_actor_seed0_251iter/
```

launcher 的 PPID 为 1 且 session ID 为自身 PID，关闭终端不会终止训练。`curriculum_source.json` 确认
route、split policy class、`source_stage=c3a` 和 promoted `model_200.pt` 精确 lineage；保存配置确认
CPU、256 env、251 iterations、16 mini-batches、save interval 25、task weights `0.15/0.20/0.15/0.50`、
strong climb `0.5`、gap reward `0.05`、distillation/gradient probe off、bounded warm start on。`model_0.pt`
已生成，iteration 4 首个 trainable warm-start update 的 actor norm 为 `0.0192 < 0.10`，无 traceback。
训练已交由后台继续，本记录不声称完成或任何 frozen suite PASS。

### 30.1 C3b v1 phase coverage 中止与 v2 修正

上述 v1 run 未完成。在约 iteration 59、`724,992` timesteps 时，当前 rollout 的 active
`c2c_strong_phase` 只有 4 samples，低于诊断门槛 16，`TaskAwarePpoAdapter` fail closed。最后保存的
checkpoint 是 `model_50.pt`。这不是 NaN、PhysX failure 或 task group 缺失；C2c task normalization
本身把普通与 strong rows 合并，phase count 不参与 PPO loss。

最小修正只在 C3b config 把 `pure_rl_task_aware_ppo_minimum_phase_samples` 设为 0：phase count 继续写入
`task_aware_ppo.csv`，但不再因单个 rollout 的偶发低计数终止；四个 task group 的 minimum samples
仍为 32，C3a 原 phase minimum 仍为 16。route 因训练语义变化升级为
`c3b_split_frequency_actor_v2`，并从 promoted C3a 重新开始，不续用 v1 的 partial optimizer。

定向 adapter/env/launcher/path tests 为 `156 passed`，route 更新后的直接集合为 `101 passed`，
`py_compile` 和 `git diff --check` 通过。v2 于 2026-08-29 15:01 Asia/Shanghai 脱离终端启动：

```text
launcher PID file: c3b_split_frequency_actor_v2_training.pid
launcher PID:      4026770
stdout/stderr:     c3b_split_frequency_actor_v2_training.log
run directory:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_15-01-17_pure_rl_c3b_split_frequency_actor_v2_seed0_251iter/
```

`curriculum_source.json` 确认 route 为 v2、source 为 promoted C3a split `model_200.pt`；保存 env config
确认 phase minimum 0、task minimum 32、四任务权重 `0.15/0.20/0.15/0.50`。首个 trainable update 的
actor norm 为 `0.0192 < 0.10`。只有越过旧 v1 的 iteration-59 failure point 才能证明 runtime 修正已
覆盖原故障。v2 在 iteration 59 再次观测到完全相同的 `c2c_strong_phase=4`，正常完成该次更新并进入
iteration 60；下一 rollout 计数回升到 46，且没有 traceback。因此 runtime 修正已覆盖原故障。训练
仍在后台继续，此记录不代表训练完成或 frozen suite PASS。

## 31. 2026-08-30 C3b fixed-mixture 结果与 adaptive-ability 实验

C3b v2 已完成，并在独立 fresh CPU-native 进程中评估 iterations 50--250 的九个 checkpoint。没有
checkpoint 同时通过 C3b v3、C3a、C2c 和 C1。`model_100.pt` 是最佳受控起点：C1、C2c、C3a 全部
通过，C3b overall success 为 `0.94318`；十个失败全部为 roll-limit termination，集中在含 `+12 deg`
climb 的 templates 5/7 和特定航向。iteration 175 以后 C2c retention 也开始失败。

完整结果：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-29_15-01-17_pure_rl_c3b_split_frequency_actor_v2_seed0_251iter/
eval_authority_50_250_c3b_v3/checkpoint_evaluation.json
```

下一实验不直接进入 C3c，也不改变固定 `15/20/15/50` task-aware PPO 目标。新增显式 route
`c3b_adaptive_sampling_from_model100_v1`，从 C3b `model_100.pt` weights-only、fresh optimizer 开始，
训练 101 iterations 并保存 50/75/100。reset sampler 保持 simple/C3b 总量 `50/50`，只做三类有界
调整：simple 半区内 C1/C2c/C3a 重分配、C3b templates 5/7 占比、templates 5/7 内 `10--12 deg`
strong-climb 占比。EMA、最低 episode 数、概率下限和单次变化上限全部保留。完整决策见
`docs/decisions/ADR-2026-08-30-pure-rl-c3b-adaptive-ability-sampling.md`。

该实验的训练 telemetry 只能证明 scheduler 工作；最终仍必须对 50/75/100 跑 C3b v3、C3a、C2c、
C1 frozen suites，并要求相邻 all-suite passing pair。C3c 在此之前保持未授权训练。

正式训练于 2026-08-30 10:15 Asia/Shanghai 脱离终端启动：

```text
launcher PID file: c3b_adaptive_sampling_training.pid
launcher PID:      1097162
training PID:      1097207
stdout/stderr:     c3b_adaptive_sampling_training.log
run directory:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-30_10-15-00_pure_rl_c3b_adaptive_sampling_from100_seed0_101iter/
```

launcher 的 PPID 为 1 且 session ID 为自身 PID，关闭终端不会终止训练。`curriculum_source.json` 确认
route 为 `c3b_adaptive_sampling_from_model100_v1`、policy 为 `PureRLSplitActorCritic`、source stage 为
`c3b`，并精确指向 v2 `model_100.pt`。`model_0.pt` 已生成，iteration 0 完成且无 traceback；初始 reset
概率为 C1/C2c/C3a/C3b `0.15/0.20/0.15/0.50`，weak-template 概率为 `2/7`，weak strong-climb 概率为
`0.25`，adaptive update count 为 0，符合基线等价起点。训练已交由后台继续，本记录不声称完成、
scheduler 已产生收益或任何 frozen suite PASS。

## 32. 2026-08-31 C3b paired global-yaw consistency 实验

heading-0/heading-180 matched transition 诊断显示：相同 body-relative C3b template 在不同 global yaw
下可以从相同初始 path error 产生不同 actor action，并在 heading 180 集中触发 templates 5/7 roll-limit
失败。因此本轮不改 sampler 或 observation contract，先做单变量因果实验：对每个 PPO actor observation
随机采样一个 `[-pi, pi]` global-yaw delta，将 30 帧 world quaternion 全部左乘同一个 yaw rotation，
其余 body-frame state、action history 和 path preview 保持不变，并以原 mean action 为一致性目标。
auxiliary actor MSE coefficient 为 `0.05`；critic 和原始 PPO batch 不做 augmentation。完整决策见
`docs/decisions/ADR-2026-08-31-pure-rl-c3b-yaw-consistency.md`。

纯函数、PPO hook、launcher、spatial env 和 split actor 定向测试共 `109 passed`，`git diff --check`
通过。16-env、4-iteration fresh CPU-native smoke 完成，iteration 3 首个真实 update 的 symmetry loss
为 `0.0269`，actor displacement 为 `0.0121 < 0.10`，无 traceback 或 non-finite value。该 smoke 只证明
运行接线，不是性能证据。

正式实验从与 adaptive baseline 相同的 fixed-mixture C3b v2 `model_100.pt` weights-only、fresh optimizer
启动；adaptive sampler、split actor、task-aware PPO `15/20/15/50`、reward、seed 0、256 env、16
mini-batches、101 iterations 和 save interval 25 均不变。2026-08-31 13:49 Asia/Shanghai 脱离终端启动：

```text
launcher PID file: c3b_yaw_consistency_training.pid
launcher PID:      2201303
training PID:      2201368
stdout/stderr:     c3b_yaw_consistency_training.log
run directory:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-31_13-49-30_pure_rl_c3b_yawconsistency005_from100_seed0_101iter/
```

launcher 的 PPID 为 1、session ID 为自身 PID。`curriculum_source.json` 确认 route 为
`c3b_adaptive_sampling_yaw_consistency_from_model100_v1`、source stage 为 `c3b`、policy 为
`PureRLSplitActorCritic`，并精确指向 v2 `model_100.pt`。保存配置确认 consistency coefficient `0.05`、
adaptive sampling 开启、101 iterations 和 16 mini-batches。`model_0.pt` 已生成且 iteration 0 无
traceback；在 frozen C1/C2c/C3a/C3b evaluation 完成前不声称有收益或可 promotion。

### 32.1 训练与 frozen evaluation 结果

训练正常完成，return code 为 0，`model_50/75/100.pt` 均存在。随后用
`evaluate_pure_rl_c3a_checkpoint.py --spatial-stage c3b --pure-rl-split-frequency-actor` 对三个 checkpoint
逐 checkpoint、逐 suite 启动 12 个 fresh CPU-native Isaac 进程。权威汇总：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-31_13-49-30_pure_rl_c3b_yawconsistency005_from100_seed0_101iter/
eval_authority_50_100_c3b_v3_yaw_consistency/checkpoint_evaluation.json
```

与同 source、同 adaptive 配置但无 consistency 的 baseline 比较：

```text
iteration   baseline C3b   yaw-consistency C3b   roll-limit   C3a   C1    C2c
50          0.94318        0.92045               8 -> 14      PASS  PASS  PASS
75          0.90909        0.89773              16 -> 18      PASS  PASS  PASS
100         0.93750        0.95455              11 -> 6       PASS  PASS  PASS
```

三点都没有通过 C3b gate，因此没有 all-suite passing checkpoint，更没有相邻 passing pair，不能 promotion。
model 100 是唯一显示正收益的点：C3b 从 165/176 提升到 168/176，baseline 的 11 个失败中恢复 3 个且
没有新增失败；恢复的是 template 5 的 heading 270 两个 phase，以及 heading 90 一个 phase。剩余 8 个
失败全部集中在 heading 180、`+12 deg`、templates 5/7；其中 6 个仍是 roll-limit，另外两个存活到
20 s 但没有完成所有事件。C3a 和 C1 在三个 checkpoint 都完整通过。C2c 虽均过 gate，但 baseline
三个点的 survival/climb success 都是 1.0，而 consistency 的 50/75 点为 `0.96429/0.91667`，100 点为
`0.99107/0.97917`，说明 auxiliary loss 不是无代价改善。

### 32.2 matched heading trace 与结论边界

为直接检查机制，对 baseline model 100 和 yaw-consistency model 100 分别运行 template 5/7、heading
0/180、`+12 deg`、turn sign -1、phase 0 的 fresh-process matched trace：

```text
eval/c3b_transition_heading_diagnostic_adaptive100_v1/comparison.json
eval/c3b_transition_heading_diagnostic_yawconsistency100_v1/comparison.json
```

两者的 heading-0 control 均完成，heading-180 template 5 均约 13 s roll-limit，template 7 均约 6.4 s
在 turn onset 前 roll-limit。reset 首帧四动作的 heading-180 minus heading-0 RMS 差异没有下降：baseline
为 `0.234636`，consistency 为 `0.249338`。正式训练的 mean symmetry loss 到 iteration 100 仍为
`0.0193`，没有趋近于零。

因此本实验的受支持结论是：paired global-yaw consistency 0.05 在单 seed 的 late checkpoint 恢复了
3/176 C3b frozen cases并减少 roll-limit，但收益非单调，C2c 略退化，而且没有消除 h180 action/state
分叉或通过 C3b gate。不能把改善归因表述为已学会严格 yaw invariance；单 seed 也不能排除优化随机性。
若继续此方向，下一步应先针对 quaternion double-cover / heading-180 boundary 检查 observation
canonicalization 或显式 `q`/`-q` consistency，而不是直接提高 coefficient；提高 coefficient 可能进一步
放大早期 C3b 和 C2c 退化。

## 33. 2026-08-31 route-heading-canonical observation 实验

用户批准从 observation 表示层消除无关的 global-yaw 自由度。新增默认关闭的
`pure_rl_heading_canonical_observation`：送入 30 帧 policy history 的姿态改为

```text
q_route_body = q_z(-route_heading) * q_world_body
```

其中 `route_heading` 是 episode 初始路径切向航向，不是飞机当前 yaw，因此相对航向误差仍被保留。
规范化后统一选择非负 scalar quaternion sign，避免 heading 180 附近的 `q/-q` 双覆盖分叉。世界四元数
仍用于物理、风速、reward、termination 和 body-frame path preview；维度保持 555。完整决策见
`docs/decisions/ADR-2026-08-31-pure-rl-heading-canonical-observation.md`。

该 route 仅改变 observation，保持 adaptive sampler、split actor、task-aware PPO `15/20/15/50`、
governor-gap reward、bounded warm start、source、seed 和训练预算不变；paired yaw-consistency coefficient
显式为 `0.0`。训练和 frozen evaluator 都必须显式开启 canonical flag。

定向 observation/env/launcher/watch/evaluator/split-actor 测试共 `169 passed`，`py_compile` 和
`git diff --check` 通过。16-env、4-iteration fresh CPU-native smoke 生成 `model_3.pt`，保存配置确认
canonical observation 开启、yaw consistency 为 0；iteration 3 首个真实 PPO update 的 actor
displacement 为 `0.0122 < 0.10`，无 traceback 或 non-finite value。

正式训练于 2026-08-31 17:32 Asia/Shanghai 脱离终端启动：

```text
launcher PID file: c3b_heading_canonical_training.pid
launcher PID:      2811024
stdout/stderr:     c3b_heading_canonical_training.log
run directory:
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-31_17-32-50_pure_rl_c3b_headingcanonical_from100_seed0_101iter/
```

launcher 的 PPID 为 1、session ID 为自身。`curriculum_source.json` 确认 route 为
`c3b_adaptive_sampling_heading_canonical_observation_from_model100_v1`、policy 为
`PureRLSplitActorCritic`、source stage 为 C3b，并精确指向 fixed-mixture v2 `model_100.pt`。保存配置确认
CPU、256 env、101 iterations、16 minibatches、canonical observation true、yaw consistency 0、adaptive
sampling true 和 governor-gap weight 0.05。iteration 3 首个正式 update 的 actor norm 为
`0.0218 < 0.10`，iteration 4 继续正常运行。训练现已交由后台继续；本记录不声称训练完成、frozen suite
通过或 C3b promotion。

## 34. 2026-09-01 完整 C3 joint training 决策

route-heading-canonical C3b run 的 `model_75.pt` 已由标准 helper 正式晋级。该 checkpoint 的 fresh
C3c v2 zero-shot 为 `88/96`；失败只出现在 high-severity right-turn + climb。随后从该 checkpoint
进行的 101-iteration C3c continuation 没有提高这一结果：iteration 50/75/100 的 C3c 分别为
`88/96`、`80/96`、`80/96`，C3b 则从 `168/176` 下降到 `152/176` 和 `116/176`，iteration 100
还因 tail-limit fraction `0.1335 > 0.10` 失去 C1 gate。没有 checkpoint 可 promotion。

代码检查确认旧 C3c sampler 的 `15% earlier spatial` 不是完整 C3a+C3b rehearsal：它只包含 isolated
C3a turn 和两个基础 C3b multi-turn templates，不包含完整 turn/vertical transitions 与 loiter。因此
C3c 不是 C3b 的训练超集。

用户批准停止把 C3a/C3b/C3c 当作依次 fine-tune 的策略阶段。新增显式
`--c3-full-joint-from-c2c` route，从 promoted C2c `model_550.pt` weights-only、fresh optimizer 开始，
使用最终 `PureRLSplitActorCritic`、route-heading-canonical observation 和固定五组 task-aware PPO：

```text
C1 / C2c / C3a / C3b / C3c = 0.15 / 0.20 / 0.15 / 0.25 / 0.25
```

C3b bucket 覆盖全部七个 sequential templates，C3c bucket 独立覆盖 coupled paths。adaptive sampling、
distillation、yaw-consistency、EWC、新 head 和 reward 修改均关闭。正式判断仍要求相邻 checkpoint
同时通过 fresh C3c/C3b/C3a/C2c/C1 suites。完整决策见
`docs/decisions/ADR-2026-09-01-pure-rl-full-c3-joint-training.md`。
