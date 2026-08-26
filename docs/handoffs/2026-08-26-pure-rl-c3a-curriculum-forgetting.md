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

本文档记录的是 2026-08-26 的状态。已有架构文档、ADR 和 audit 仍是各自主题的详细事实源；本文档只做导航、整合和接续，不替代它们。

## 2. 当前工作区状态

- 仓库根目录：`/home/zn/IsaacLab`
- 当前工作树：`/home/zn/IsaacLab/.worktrees/native-multibody-rl`
- 分支：`feat/native-multibody-rl`
- 当前提交：`196343dc feat(rl): add task-aware C3a PPO diagnostics`
- 当前远端：`origin/feat/native-multibody-rl` 与本地提交一致
- 已跟踪文件在写本文档前是干净的
- 本地存在未跟踪的 `*.pid`、`eval/` 和实验结果；它们属于运行产物，不要删除、提交或据其文件名推断实验成功

`docs/PROJECT_STATE.md` 仍是项目总状态入口，但其中“Exact next task”要求评估 bounded-warm-start 的 100/200/300/400 checkpoint。该工作已经完成，因此那一小节已经过期。最新 C3a 遗忘结论和下一步以本文档为准，直到 `PROJECT_STATE.md` 被显式更新。

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
6. 已完成实验 audit：
   - `docs/audits/2026-08-13-pure-rl-c3a-c2c-rehearsal.md`
   - `docs/audits/2026-08-13-pure-rl-c3a-actor-distillation.md`
   - `docs/audits/2026-08-14-pure-rl-c3a-retention-aware-sampling.md`
   - `docs/audits/2026-08-18-pure-rl-c3a-bounded-warm-start.md`
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
| C3a | 独立水平转弯 | 已实现、能学会，但未通过 C1/C2c retention |
| C3b | 顺序动作事件 | 已实现合同，未授权训练 |
| C3c | 转弯与爬升/下降耦合 | 已实现合同，未授权训练 |

C3a 不是从 C2c optimizer 继续训练，而是从正式晋级 C2c checkpoint 进行 **weights-only** 初始化，使用全新的 optimizer 和 iteration 计数。

正式 C2c source 是：

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-12_17-12-48_pure_rl_c2c_seed0_resume500_to550/model_550.pt
```

相邻 `model_525.pt` / `model_550.pt` 是已晋级证据。当前不要改用某个 C3a checkpoint 作为新的 source，因为尚无 C3a checkpoint 通过完整晋级合同。

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
7. **weights-only 不是 resume。** C3a 必须从 C2c source actor 复制权重并创建 fresh optimizer；不要同时使用 `--resume`。
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

所有实现都应优先放入 `source/flapping_bot/` 的适当层，保持 batch 维度、device、dtype 和已有 baseline 选择路径。

## 15. 尚未解决的决定

- task-aware + distillation 0.05 已完成 seed-0、101-iteration 实验并确认不足以跨过完整 retention gate；结果见第 19 节。
- 方法 2 的大 actor 实验也已完成 seed-0、101-iteration 训练和冻结评估；它改善了 C1 retention，
  但没有恢复 C2c strong-climb，结果见第 21 节。
- fixed teacher anchors 的最小状态数量和 phase 配额；应先由离线 KL 区分能力决定，而不是先写大型 buffer 框架。
- actor Gaussian std 是否对当前遗忘有实质贡献；现有实验没有保护 std。
- GEM-style 约束的具体数值容差、投影实现和 per-task/per-phase 粒度；Phase B 前不应过早冻结设计。
- 完整通过后是否需要额外 seeds 才允许进入 C3b；当前晋级合同主要冻结相邻 checkpoint 和旧课程 retention，但科研结论需要更强的 robustness 计划。

## 16. Exact next task

方法 2 已完成且没有产生可晋级 checkpoint。不得开始 C3b。下一项代码实验需要用户显式批准：按第 11 节
Phase B 先生成独立训练分布上的 fixed teacher anchors，并离线检验 strong-climb/recovery teacher-relative KL
能否区分 promoted source、已知中期 checkpoint 和已知退化 checkpoint。在该判别成立前，不实现 GEM-style
约束或 residual actor。

## 17. Suggested skills

- `right-size-coding`：限制 continual-learning 实现规模，优先完成可证伪的小实验，避免直接引入完整框架。
- `get-available-resources`：仅在准备启动新的 CPU-native 长训练前检查 CPU、内存和磁盘，避免与残留 Isaac Sim 进程争用。
- `checkpointed-subagent-supervision`：仅在把长训练或多套 fresh-process evaluation 委派给子 agent 时使用，并要求 checkpoint、日志和 frozen-suite 证据回传。

## 18. 给接手 agent 的一句话

不要把问题重新简化成“多给一些 C2 样本”。当前证据是：C2c 已经 exact rehearsal，C3a 能稳定学会，首步 optimizer shock 已修复，但 `+12 deg` strong-climb/recovery 功能仍被 shared actor 的长期优化改写。task-aware PPO 与 actor distillation `0.05` 的组合已确认不足；用户选择先做单变量 actor-capacity 实验，再决定是否进入 fixed teacher anchor / hard retention constraint。

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
