# Unified Trajectory PPO：真实 PhysX 冻结评价验收

2026-11-06，结论：**GO**。L3D/I3D × development/final 四个完整批次均通过真实 CPU-native PhysX 运行及严格 SQLite 重放验证，可以进入首次受控 PPO 基础学习实验。本轮未启动训练或参数更新；此结论不表示策略已学会飞行，也不构成 Fixed/Adaptive 性能比较。

## 代码、物理身份与冻结快照

起点和当前 HEAD 均为 `51ba5a1e075636f16b823949de6fb328def62a92`，开始时工作区干净。实际运行包含本轮**未提交**的 smoke 入口、隔离标记及验收检查；各运行的 `source_snapshot/` 保存相关源码，快照来源明确记录 `workspace_dirty=true`。未 commit/push。

实际配置为 `measured_wing_multibody`、`native_holonomic_drive`，CPU PhysX 480 Hz、控制 60 Hz；分别启用 L3D/I3D，35° 扭转、每翼 10 站，沿用各后端原有限时域候选初态。`check_initial()` 核对后端、39D/4D 接口、初速、四元数、相位、扑频及尾舵关节。物理身份为 `physical_v2`，包含 34 个物理/native 依赖文件及 URDF、翼几何、实际配置；完整身份分别在两个 development 目录的 `snapshot_contract.json`，final 与对应快照逐项匹配。终止合同为 `unified_interval_v3`。

本工作区现有 21 个 checkpoint 均缺少新版合同，未修改或复用其权重。通过显式 `--evaluate --smoke-evaluation --create-smoke-snapshot`，从实际环境 reset 读取身份和初态，以 seed 0 初始化原 `[256,256,256]` ELU ActorCritic，保存 L/I 各一份 `evaluation_smoke_only.pt`。没有构造 PPO、optimizer 或训练 rollout，没有 loss/backward/step。

快照使用独立 schema `unified_evaluation_smoke_only_v1`，带 `smoke_fixture=True`、`training_updates=0`，不含 optimizer；绑定两个完整任务集合、固定采样标签、256 步评价预算和场景布局。正式训练（包括 warm start/resume）及普通评价拒绝它。每个真实 episode 同样带 smoke 标记，普通批次验证、反馈接收器和 `OnlineSampler.feedback_evaluation()` 均拒绝此类证据；仅显式 smoke 验证接受。

## 四组实际结果

所有输出位于 [本轮独立目录](../logs/unified_physx_acceptance_20261106_v1/)，汇总见 [只读复核结果](../logs/unified_physx_acceptance_20261106_v1/acceptance_verification.json)。每组均只有一次真实运行，无需 PhysX 复验。

| 后端 / 集合 | 完成 | failed | invalid_data | timeout | 有效控制步 | 总环境步 | 评价循环 s | 启动至评价结束 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L3D development | 0 | 5 | 0 | 0 | 626 | 1280 | 10.00 | 16.50 |
| I3D development | 0 | 5 | 0 | 0 | 471 | 1280 | 10.86 | 17.28 |
| L3D final | 0 | 6 | 0 | 0 | 748 | 1536 | 10.29 | 16.76 |
| I3D final | 0 | 6 | 0 | 0 | 565 | 1536 | 11.49 | 17.86 |

最后一列包含 Kit 启动、建环境和评价，不含最终 Kit 关闭；不是独占机器性能基准。按槽位排列的 episode 步数分别为 `[125,125,125,125,126]`、`[94,94,94,94,95]`、`[125,125,124,125,124,125]`、`[94,94,94,94,94,95]`。

256 步/环境预算在快照生成时预先写入合同。**120 s 资源超时没有缩短**，原 `max_steps=7199` 保持。22 条任务均在预算内安全失败，所以四批评价完整；余下 3222 步为 padding，不登记新 episode。全部 2410 条有效控制记录与 terminal 的 steps 总和相等，没有重复成功、遗漏或额外任务。

四组运行及退出后的只读复核均通过 `validate_evaluation_batch()`：完整世界坐标 trajectory、初态、版本、plant、终止合同、单一冻结策略执行及逐步重放结果一致。每个 terminal 同时核对环境 `failed/invalid_data/completed/timeout` 标志和返回的 done/timeout。记录包含裁剪前动作、执行命令、实际执行器/滤波器/频率状态、关节位置/速度、根状态、相位、路径投影和参考误差；采样时点为每个 60 Hz 控制区间的 decimated physics 之后。

四组参数摘要前后相同，并与对应快照初始化摘要一致；`requires_grad=False`，推理使用 inference mode。smoke 进程对 PPO 构造、Adam/SGD step 和 backward 安装拒绝调用保护，调用记录均为空，optimizer 更新次数为 0。四组始终 `online_sampler is None`，没有调用训练反馈接收；final 的 schema/证据隔离在退出后的真实数据库复核中再次验证。

## 真实布局与缓存

下表来自四组实际 `env.scene.env_origins`，不是用预期函数生成的 fixture。L/I 相同，与 `evaluation_origin()` **数值和顺序完全一致**，无需修正 GridCloner 合同。

| 槽位 | development 实测 = 预期 (m) | final 实测 = 预期 (m) |
|---|---|---|
| 0 | (10,-5,0) | (10,-5,0) |
| 1 | (10,5,0) | (10,5,0) |
| 2 | (0,-5,0) | (0,-5,0) |
| 3 | (0,5,0) | (0,5,0) |
| 4 | (-10,-5,0) | (-10,-5,0) |
| 5 | 无此环境 | (-10,5,0) |

逐槽位世界路径等于对应完整预声明轨迹加实际场景平移，初始高度为 20 m。development 缓存 60 m / 301 点，final 缓存 80 m / 401 点，间距均为 0.2 m。每条实际缓存取 17 个位置，与独立解析轨迹比较，最大分量误差分别为 `4.835e-9 m`、`2.843e-14 m`；实际缓存中点局部投影检查误差为 0。final 任务保留 seed 1401–1406 的完整参数，未用名称或 seed 替代几何匹配。

## 定向测试与原始文件保护

97 项不同的定向测试通过（首批 96 项；补充 timeout 后 smoke/隔离 5 项复测通过）。涵盖非有限原始动作、执行动作、关节/执行器状态、相位、根状态、真实缓存投影输入和参考误差、重复终止、timeout，以及 smoke 加载/反馈拒绝、pending/evaluated 恢复。没有向真实 PhysX 注入 NaN/Inf。语法检查和 `git diff --check` 通过。

全部历史 checkpoint、SQLite 正文及 WAL、离线 88 条证据保持原大小和修改时间；本轮没有对它们执行数据库写入。**保护检查发现并修复一处旧测试问题**：首次回归中 `test_actual_journal_prefix_is_read_only` 使用 `mode=ro` 连接历史库，SQLite 更新了 `logs/unified_ppo_smoke/L3D_adaptive_v2/evidence_000002_final.sqlite3-shm` 的共享内存读标记/修改时间。该辅助文件不是持久化证据正文，但本轮不能声称所有历史辅助文件都完全未触碰。没有尝试回写它。测试现已先把静态历史数据库及 WAL 复制到临时目录，再打开 SQLite；一次针对性复验通过，并确认包括原 SHM 在内的源文件元数据不再变化。其余 76 个受保护文件元数据未变化。

未发现真实评价链路 BLOCKER，未改变气动、奖励、策略架构、PPO 算法或采样权重。新增实现仅涉及 smoke 快照生成/隔离、实际布局/缓存/终止核对、来源与耗时记录，以及上述测试保护修复。尚未验证学会飞行或长期训练稳定性；本次 GO 只放行首次受控基础学习实验，不能用这些 smoke 快照恢复正式训练。

## 复现命令

在本工作区激活原环境，使用全新输出目录；先完成两个 development，再运行两个 final。以下只运行冻结评价。

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
OUT=logs/unified_physx_acceptance_repeat  # 必须尚不存在
for B in L3D I3D; do
  ./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py \
    --evaluate --smoke-evaluation --create-smoke-snapshot \
    --backend "$B" --sampling fixed --seed 0 --threads 1 --eval-step-budget 256 \
    --output "$OUT/${B}_development"
done
for B in L3D I3D; do
  ./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py \
    --evaluate --smoke-evaluation --eval-suite final \
    --backend "$B" --sampling fixed --seed 0 --threads 1 --eval-step-budget 256 \
    --checkpoint "$OUT/${B}_development/evaluation_smoke_only.pt" \
    --output "$OUT/${B}_final"
done
```

以每个 `status.json` 的 completed 和 `summary.json` 的 passed 为准；仓库 shell wrapper 的返回码不能单独作为成功依据。实际运行完整参数在各目录 `launch.json`，真实布局在 `scene_layout.json`，不可变终止和逐步证据在各自 `evidence/rollout_feedback.sqlite3`。验收后停止，没有启动任何后续训练。
