> 2026-11-06 更新：训练前修复、physical_v2 兼容性与独立 final 集合见 [中文审查](unified_trajectory_pretraining_review_zh.md)。下文 smoke/旧 checkpoint 是历史记录；新版明确拒绝未经验证的旧物理身份迁移。

# Unified Trajectory L/I PPO 集成

开发树 `.worktrees/unified-trajectory-ppo-v1`，分支 `feat/unified-trajectory-ppo-v1`，
基于已验证接线提交 **9b4ee3e8**。来源 online **13c352e2**、L/I **4303c1ad**，
开始时三个来源均无已跟踪文件修改，原有未跟踪论文保留。当前集成尚未 commit/push；
各运行记录真实 HEAD、dirty 状态、物理源文件/URDF 身份和 `source_snapshot/`，
不能把基底 SHA 说成已提交了本轮实现。

**结论：L3D/I3D × Fixed/Adaptive 均已接通；PPO 更新、保存、真实续跑和冻结评估通过。**
未启动正式长训练，未修改气动、扑翼执行器、B 样条数学、39D 观测或奖励公式。

## 共同训练合同与初始化

注册 ID：`Isaac-FlappingBot-UnifiedTrajectory-{L3D,I3D}-Direct-v0`。
配置在 `unified_li_cfg.py`，runner 实际通过 gym 注册入口构造环境。
480 Hz CPU-native PhysX / 60 Hz policy，四通道直接执行器动作，39D 观测，2/5/10 m
预瞄，默认 60 m、120 s 资源超时，原成功/安全判据不变，无参考速度跟踪。

两后端从原 `periodic_trim_lateral_v1/{L3D_late,I3D_refined}/search.json` 的 best 读取依据，
数值固化并有逐项相等测试；**有限时域候选，不是严格配平**：

| backend | 初速 m/s | 鼻上仰 ° | 扑频 Hz | 对称尾舵 ° | 方向舵 ° | 差动尾舵 ° |
|---|---:|---:|---:|---:|---:|---:|
| L3D | 9.656337 | 13.971708 | 3.043206 | 0.873482 | 1.631019 | -2.205493 |
| I3D | 9.250000 | 8.555605 | 3.743225 | -3.576212 | 0.246065 | -1.996931 |

共同初始翼拍相位 0、垂直速度/角速度 0、高度 20 m、无风、扭转 35°、10 站。
实际 root 速度、四元数鼻上仰符号、频率、相位和三尾舵关节位置均有运行时核对。
没有沿用 DeLaurier 的 8 m/s，也没有调整 L/I 参数来使它们一致。

直接复用已安装 RSL-RL `ActorCritic` / `PPO`，没有复制或改写优化算法：
actor/critic 都是标准 `[256,256,256]` ELU MLP，39→4 动作，初始 std=1；Adam LR=3e-4，
5 epochs × 4 minibatches，clip=.2，gamma=.99，lambda=.95，entropy=.01，固定 LR。
没有旧 555D 权重、C1–C4 任务或自适应课程回调。数值是共同工程初值，不宣称最优。

Fixed 只从 q0 取任务；Adaptive 使用既有三分支。二者都启用相同真实 SQLite 反馈，
避免只给 Adaptive 记日志造成不同观察合同。共同系数幅值、节点/航程及几何约束不变；
保持/探索扰动越出共同幅值盒时回退 q0。实际任务参数、来源、几何分类写入 `tasks.jsonl`。
四个 smoke 的初始网络参数完全相同；同 backend 的 plant/终止合同逐项相等。

## 版本、冻结评估与预算

runner 在每批新权重生成动作前显式 `set_policy_version()`；版本来自实际 PPO update，
不由 episode 或环境步数猜测。跨更新 episode 保留动作区间版本并标 mixed。
训练记录不能计入冻结 checkpoint 成功率；独立评估只产生 single/frozen 记录。

评估为独立 Kit/PhysX 子进程：训练暂停更新和步进，但保留在途 episode；评估完成后继续。
固定 5 条完整 60 m B 样条，覆盖简单、持续机动、先右后左、先左后右、三维耦合；
seed 401–405。左右按 kappa 符号命名（403 先右、404 先左），显示名已校正；
历史原始日志不改写，resume 比较完整轨迹参数而不是显示名。
评估始终固定任务，不能调用自适应采样；每次重置初始条件相同，不更新参数/optimizer。
这是一组会反馈给采样器的**开发评估集合**，不是封存的泛化 holdout。

`--eval-step-budget` 指每个评估 env 的固定实际步数，默认 7200，覆盖原 120 s 合同。
任务提前终止后只继续填充步数，不登记额外 episode 或成功。预算耗尽但任务尚未终止，
整个评估报错，不伪造超时/成功。报告区分 `environment_steps / active_episode_steps / padding_steps`。
因此两模式每次都花 `5 × budget` 环境步，训练 transitions 单独计数。

正式默认每 25 PPO iteration 评估一次，Adaptive 只从真实冻结日志更新冻结能力/遗忘。
评估日志独立存在 `evaluation_N/evidence/`；采样器保存已接收实例 ID，恢复后不会重复计数。

## 证据与恢复

固定历史 scope 已删除。plant 身份记录当前代码 HEAD、实际 backend、完整共同配置、
物理源文件指纹、URDF 路径/内容、候选初值、sim/control dt；扭转/站数和执行器限制均在配置中。
输出目录/采样开关/批大小不伪装成不同气动模型；实际 scene 数量、行号、位置与完整初态
单独进入 episode 初始条件。不同物理/评价合同禁止继承认证；离线 88 条记录仍只读，
仅作明确的同 backend 几何先验。

checkpoint 保存模型、Adam、RNG、更新/transition 计数、reset 序列、完整增量采样状态、
冻结接收凭据、实验合同以及追加式 SQLite 路径/截止序号。
**保留对应 evidence 目录；仅复制 .pt 不足以完整恢复。** resume 必须使用兼容实现和相同
backend/采样/超参数/seed/预算合同，写入新的 output，恢复后重新开始物理 episode；
这不是位级物理轨迹续接。在途未完成 episode 不升级为成功。

初版 smoke 使用全量 SQLite 备份；实测存储增长后，最终保存路径改为日志位置引用，
避免每次 checkpoint 重复复制全部历史。resume 按截止序号流式复制已终止的 metadata、
steps、terminal、witness 到新库；之后的原日志记录不会进入旧 checkpoint。
此数据路径用真实日志验证：源 9 条恢复前 5 条及对应 64 个动作区间，排除后续 4 条，
源不变；模拟数据库不能导入真实库。该最后修复未追加 PPO/PhysX 运行。
历史 smoke 的全备份 checkpoint 仍可由当前 loader 恢复。

主要日志：`launch.json`、`contract.json`、`training.jsonl`、`episodes.jsonl`、`tasks.jsonl`、
`evaluations.jsonl`、`status.json`、`latest_checkpoint.txt`、`model_*.pt`、SQLite 和 source_snapshot。
单写者 SQLite，完整步骤证据在磁盘，内存能力统计仍有界；不做成功标签捷径。

## 短程实测

[机器可读结果](results/unified_trajectory_ppo/smoke_summary.json)。最终四组合均为 4 env ×
8 步 × 2 PPO iteration，各 64 transitions / 40 次 Adam minibatch step。

| backend / mode | 训练 transitions/s | 冻结实际步数 | 有效评估步数 | 结果 |
|---|---:|---:|---:|---|
| L3D Fixed | 89.5 | 1280 | 235 | 5 安全失败 |
| L3D Adaptive | 90.8 | 1280 | 235 | 5 安全失败 |
| I3D Fixed | 82.2 | 1280 | 895 | 5 安全失败 |
| I3D Adaptive | 81.2 | 1280 | 895 | 5 安全失败 |

每组均有 4 个真实 mixed 训练 episode；全部不属于冻结 checkpoint。
冻结权重前后相同、optimizer 更新次数 0，没有整条轨迹成功认证。
L3D Adaptive 真实 resume 再运行 1 次 PPO iteration：Adam **40→60**、transitions **64→96**，
加载权重与原最终权重相同；旧 5 条终止记录保留，新库 9 条、无重复 ID，5 个冻结接收凭据保留。

在预算补严前执行的 v1 接线日志也保留，不作最终等预算结果；每次运行最多 2 iteration，
没有将这些 smoke 当正式训练或策略性能实验。

256 env 容量检查：每个 backend 仅 Adaptive × 48 步 × 1 iteration，无冻结评估。

| backend | transitions | rollout s | PPO update s | transitions/s | 峰值 RSS | SQLite 大小 |
|---|---:|---:|---:|---:|---:|---:|
| L3D | 12288 | 7.28 | 0.96 | 1492 | 3.71 GiB | 61.4 MiB |
| I3D | 12288 | 11.83 | 0.90 | 965 | 3.71 GiB | 55.6 MiB |

L3D/I3D rollout 中分别有 35/1 次安全失败，均真实终止并重采样。
环境 step 累计 7.12/11.74 s，已含物理、任务和记录成本；不能全部归因于气动计算。
整次运行 SQLite flush 累计 2.11/1.88 s，含退出手动终止和保存，不能直接当作 rollout 占比。
没有发现 256 env 内存阻塞；CPU 环境步进和持久化开销可见，长期磁盘增长仍需预算。
单次 journal 大小含启动/退出元数据，粗略只能提示长期运行需要几十 GB 量级空间，
不能把这一批数据线性当成精确容量预测。checkpoint 已不再产生二次增长的全量备份。

**35 项定向测试通过**，包括原数学/缓存/在线证据测试、候选初值、固定评估集合、
共同采样幅值、采样状态恢复和模拟证据隔离。真实前缀恢复另行通过。
环境/预瞄/部分 reset/动作/奖励/终止、gym 注册及实际 L/I 后端由上述真实运行验证。
Kit 有已有监视器配额警告，但所有交付运行正常保存并退出。

## 命令（本轮没有执行下面的正式训练）

在新开发树：

```bash
cd /home/zn/IsaacLab/.worktrees/unified-trajectory-ppo-v1
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
B=L3D                         # 换为 I3D 使用另一后端
COMMON=(--backend "$B" --seed 0 --num-envs 256 --rollout-steps 48 --iterations 1000 --eval-interval 25 --eval-step-budget 7200 --save-interval 25)
./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py "${COMMON[@]}" --sampling fixed --output "logs/unified_ppo/${B}_fixed_s0" > "/tmp/${B}_fixed_s0.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py "${COMMON[@]}" --sampling adaptive --output "logs/unified_ppo/${B}_adaptive_s0" > "/tmp/${B}_adaptive_s0.log" 2>&1
```

两条按顺序执行，不能复用已存在 output；若改参数，两组共同修改。
默认读取相邻 control-oriented 工作树的只读 v2 文件及已编译 native extension；
可用 `--rules/--plant-assets` 显式指定。普通依赖环境不另安装。

独立冻结评估（不更新策略，实际完整 episode；已交付 smoke checkpoint 可直接使用）：

```bash
./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py --evaluate --backend L3D --sampling adaptive \
  --checkpoint logs/unified_ppo_smoke/L3D_adaptive_v2/model_000002_final.pt \
  --eval-step-budget 7200 --output logs/unified_ppo/eval_L3D_example
```

正式 resume：沿用原 COMMON（可更改额外 iterations），加
`--checkpoint /absolute/model_N.pt --resume --output NEW_DIRECTORY`，保持原 sampling。
当前 smoke 续跑示例参数须保持 `num-envs=4, rollout-steps=8, eval-interval=2,
 eval-step-budget=256, save-interval=2`。不要用 smoke checkpoint 冒充正式 256 env 合同恢复。

第二终端实时查看：

```bash
tail -F /tmp/L3D_fixed_s0.log
tail -F logs/unified_ppo/L3D_fixed_s0/training.jsonl logs/unified_ppo/L3D_fixed_s0/evaluations.jsonl
cat logs/unified_ppo/L3D_fixed_s0/status.json logs/unified_ppo/L3D_fixed_s0/latest_checkpoint.txt
```

复现短程：将正式命令改为 `--num-envs 4 --rollout-steps 8 --iterations 2 --eval-interval 2
 --eval-step-budget 256 --save-interval 2 --smoke-check`，仍使用新的 output。
定向测试命令：

```bash
./isaaclab.sh -p -m pytest tests/test_unified_ppo_contract.py tests/test_online_trajectory.py tests/test_unified_trajectory.py tests/test_unified_trajectory_cache.py -q
```

例：1000 iteration = 每组 **12,288,000 训练 transitions**；40 次默认冻结评估另计
**1,440,000 环境步**。按上述单批实测估算，L3D 训练约 2.29 h、I3D 约 3.54 h；
固定预算评估分别另约 2.83/3.37 h，再加进程启动/保存，合计约 **5.3 h / 7.1 h 每组**。
同 backend 两模式可暂用此数量级预算；没有实测 256 env Fixed，不能宣称两模式性能等价。
这些是短程外推，reset 频率、学习后轨迹长度、SQL 增长和机器负载都可能改变实际耗时。

机制与真实接口足以开展同预算对照；不表示已证明自适应优于固定采样、已学会飞行，
也不是长期稳定性保证。本轮在必要短程验证后停止，没有正式训练或自动发布。
