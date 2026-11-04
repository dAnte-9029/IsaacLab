# D-50 显式尾舵行为约束与软锚定

两组各 25 次更新和四次六套冻结评估全部完成。**H 将 C4 development 保持在
96/96，而 S 从 96/96 降至 0/96；但 H 拒绝了 88% 的尾舵候选步，且两组最终
都未通过 C3b/C3c。** 本设置支持显式功能约束的 C4 防退化效果，尚未证明兼顾
持续尾舵可塑性与全部旧任务保持。

## 研究合同与实现

仅检验同一 D-50 source、同一训练锚点和相同新数据预算下，H 是否比 S 更能
保持已获得的控制行为，同时允许继续学习。工作区为 `.worktrees/native-multibody-rl`，
分支 `feat/native-multibody-rl`，起点 HEAD `79a172457bc9ab0c85720e167dbf8915e6009b82`。
保持原 CPU-native measured-wing plant、480 Hz physics / 60 Hz policy、
六任务族、奖励和终止条件；没有新气动模型、MoE/GRU、任务扩展或系数搜索。

训练前固定设置见 [合同](../plans/2026-11-04-d50-behavioral-anchoring.md)。
两组均为 seed 0、256 environments、48 steps、16 minibatches、4 PPO epochs、
固定学习率 5e-5、25 次追加 iteration，即每组 307,200 条新 on-policy transitions。
actor、critic、log_std、完整 19 项 Adam 状态、guard/task 进度均从历史
`logs/continual_c4/20260922_pilot/D/training/model_50.pt` 恢复。
文件名 50 对应已完成 51 轮；本轮 +12/+25 的绝对文件名为 model_62/model_75。
它们与历史 D-75 分属不同输出目录，不是同一策略。

两组使用原 `training_only` memory 的全部 24,576 条输入，六族各 4,096。
**所有目标重新由冻结 D-50 生成，旧 full-C3 teacher loss 关闭。**
555 维固定归一化、heading-canonical quaternion、sin/cos phase、动作历史和
split frequency actor 均保持原定义；无 empirical normalization。
约束只覆盖未裁剪 deterministic mean 的 rudder/left/right 三维；frequency
仅作漂移诊断，不冻结其网络。没有新增或虚构 recovery 标签，也没有将 development
或 sealed holdout observations 用于训练。

S 使用 lambda=1、六族等权的完整锚点尾舵 MSE，与 PPO 梯度相加后沿用 global
gradient clipping。H 没有该惩罚项，而是在每次实际 Adam 候选步后检查六个
完整任务族的 MSE≤0.0004（RMS≤0.02）；最多试到 alpha=1/1024。
部分接受保留候选 moments/step，等价于该尾舵梯度使用 alpha×lr 的单个 Adam 步；
拒绝则同时恢复尾舵参数、moments 和 step。critic、frequency、log_std 始终保留
普通 PPO 更新。每轮结束再次检查完整锚点；保证仅适用于这些锚点的分组平均距离，
不是逐样本最大误差或闭环稳定性保证。

## 恢复边界

实际初始化检查确认两组 policy/optimizer/curriculum 状态相同，锚点初始漂移全为零。
不过复用的 `RslRlVecEnvWrapper.__init__` 会在 curriculum load 之前先 reset。
**首批路径是共同 seed-0 初始化采样，随后才恢复 D-50 课程统计、概率与私有 RNG。**
此顺序在 S 完成后查明，并对 H 原样保留；不能把它写成已通过的训练前完整运行状态
恢复检查。两组属于相同初始化规则下、完整 optimizer 恢复的受控续训，
不是完整 D-50 仿真/采样轨迹续跑。策略变化后 on-policy 状态轨迹不再逐状态配对。

## 冻结评估

起点复用历史 D-50 正式结果。+12/+25 使用原独立进程入口，执行 C4 development
96-case 和 C1/C2c/C3a/C3b/C3c 原冻结套件。sealed holdout 未打开，门槛未改变。
D-50 的 C3b 均横向误差为 0.505992 m，略超 0.5 m；它本来就没有通过全部旧任务。

| checkpoint | C4 成功数 | C4 横/垂均误差 m | C1/C2c/C3a | C3b gate | C3c 成功数 / gate |
| --- | ---: | ---: | --- | --- | --- |
| D-50 起点 | 96/96 | 0.271 / 0.255 | 均通过 | 未通过 | 96/96，通过 |
| S +12 | 21/96 | 0.521 / 0.459 | 均通过 | 通过 | 88/96，未通过 |
| H +12 | 96/96 | 0.412 / 0.241 | 均通过 | 未通过 | 88/96，未通过 |
| S +25 | 0/96 | 0.850 / 0.641 | 均通过 | 未通过 | 84/96，未通过 |
| H +25 | 96/96 | 0.257 / 0.314 | 均通过 | 未通过 | 84/96，未通过 |

四个新检查点的 C4 存活率、完成率均为 100%；因此存活并不能代表航迹能力保持。
H +25 横向均误差略低于起点，但垂向均误差更高，不能写成所有指标均改善。
最终两组 C3b 都为 168/176，template-5 成功率 75%，并非起点的轻微均误差问题：
其均横向误差实际为 S 0.49135 m、H 0.46059 m，均低于 0.5 m，仍被原 slice gate 拒绝。
H +12 也出现相同 C3b 完成缺口；S +12 则通过。这不是旧任务保持的全面优势。

## 约束、有效更新量与成本

独立加载最终 checkpoint 重算全锚点，H 六族全部满足约束；25 轮检查也无违规。
下表为最终 normalized unclipped tail RMS，分母始终是各族样本数乘 3。

| 任务族 | S | H |
| --- | ---: | ---: |
| C1 | 0.03030 | 0.01215 |
| C2c | 0.03353 | 0.01481 |
| C3a | 0.03908 | 0.01379 |
| C3b | 0.04443 | 0.01682 |
| C3c | 0.04078 | 0.01714 |
| C4 | 0.11926 | 0.02000 |

全程最大分组 RMS 为 S 0.13140、H 0.01999980。C4 最终三舵 RMS
`[rudder,left,right]` 为 S `[0.14084,0.11654,0.09619]`、
H `[0.02530,0.02002,0.01262]`；sym/diff 分别为 S `0.10567/0.01584`、
H `0.01571/0.00578`。分组 RMS 上限不意味着每个通道均小于 0.02。

| 更新/计算指标 | S | H |
| --- | ---: | ---: |
| 新 on-policy transitions | 307,200 | 307,200 |
| Adam 候选步 | 1,600 | 1,600 |
| 接受尾舵步 | 1,600 | 192（12%） |
| 其中回溯后接受 | 0 | 80（全部候选的 5%） |
| 拒绝尾舵步 | 0 | 1,408（88%） |
| sum(alpha) / 1,600 | 100% | 7.494% |
| 尾舵参数相对 D-50 的净 L2 位移 | 0.77116 | 0.08829 |
| 扑频参数净 L2 位移 | 0.53983 | 0.67092 |
| critic 参数净 L2 位移 | 1.73364 | 2.25231 |
| 训练进程 wall time（含启动/保存） | 365.19 s | 1,290.30 s |
| rollout collection 累计时间 | 97.24 s | 98.89 s |
| PPO update 累计时间 | 256.64 s | 1,179.56 s |
| 锚定额外计算 | 229.98 s（软损失/诊断） | 1,146.65 s（约束）+5.94 s（诊断） |

H 接受的尾舵参数步长累计为 0.42136，其自身未缩放候选步长累计为 13.42385，
比例 **3.14%**；这是 H 内部的长度加权比值，不是与 S 的逐梯度配对。
`sum(alpha)` 也不是训练数据利用率或闭环改善率。H 最后 6 轮尾舵步全部拒绝。
最终 optimizer 实物核对：尾舵各参数 Adam step 增量为 192，其他参数为 1,600；
S 全部为 1,600。没有把拒绝步偷偷计为接受步，也没有回退 critic/std。

PPO 并未整体停止：H 扑频、critic、std 仍更新。C4 锚点的扑频输出最终 RMS
漂移为 S 0.17702、H 0.09485；旧任务族则为 S 0.206–0.224、H 0.159–0.172。
这些是归一化请求均值，不等同于 governor 后实际频率。仅保护尾舵不保证整个
策略或未覆盖状态上的行为不变；本轮没有额外模块替换实验来归因 C3b/C3c 失败。

C4 训练 total reward 的前/后 5 轮均值为 S `0.9091→0.8207`、H `0.9125→0.8984`；
对应水平误差为 `0.221→0.783 m`、`0.222→0.502 m`，垂直误差为
`0.231→0.633 m`、`0.209→0.215 m`。PPO value loss 后 5 轮均值为
S 1.0250、H 0.1715，surrogate 分别为 0.00890、0.000149。
这些量来自不断变化的 on-policy 状态/回报目标，不能当作同状态配对或 critic 正确性证明。
逐轮全部损失、反馈、三舵/sym/diff/扑频漂移和每个候选步的 alpha/耗时均保存在日志。

## 对唯一研究问题的判断

**SUPPORTED：** 在本次共同锚点、共同初始化规则、相同数据预算和单 seed 下，
直接限制尾舵函数漂移确实比 lambda=1 的软锚定更能防止 C4 航迹能力崩塌。
这项差异不能归因于换了 teacher，因为两者都使用同一 D-50 reference。

**联合目标尚未充分成立：** H 的保持伴随很强的尾舵更新压缩，且 C3b/C3c 未保持。
critic/扑频继续优化、H 中间到最终 C4 横向误差下降，说明整个 PPO 没有被冻结；
但不能据此宣称持续尾舵可塑性、新能力获得或全 C1–C4 保持已解决。
也不能从这一个阈值和回溯实现推断所有硬约束方法都会失去可塑性，或证明共享表示必然冲突。
本轮到此停止，不追加 seed、更新预算或阈值搜索。

[四面板对比图](../../logs/continual_c4/20261104_behavioral_anchoring/comparison.png)
（[SVG](../../logs/continual_c4/20261104_behavioral_anchoring/comparison.svg)）区分约束、
有效步长与闭环结果；右下 C3c 的 S/H 曲线完全重合。
[完整统计](../../logs/continual_c4/20261104_behavioral_anchoring/analysis.json)保存精确数值。

## 验证、工件与复现

新定向测试覆盖 Adam 接受/回溯/拒绝、任务族不能互相掩盖、tail-only 辅助梯度、
无约束触发时与安装版 PPO 的参数/optimizer 一致、anchor/reference resume，
以及评估中断后的已完成套件复用。加上受影响 C4 入口测试，共 11 项通过。
实际 +12 checkpoint 核对：两组 reference/anchors 均完整；S 的所有参数 Adam
计数增加 768，H 尾舵增加实际接受的 177、其余参数增加 768，符合拒绝回退语义。

旧评估入口要求 `model_<iteration>.pt` 名称，初次旧任务评估因此暂停。
通过同内容硬链接适配名称后复用已完成 C4 结果并继续；没有额外 S 训练或重复采样。
新 pipeline 保留失败日志，拒绝自动重启已有不完整训练。

主要实现：

- `px4_like/behavioral_anchoring.py`：完整锚点损失、真实输出约束和 PPO 更新。
- `analysis/c4_training.py` 与 `train_continuous_curve.py`：受限 opt-in 接入、恢复、保存。
- `analysis/behavioral_anchoring_experiment.py`：固定 source/锚点与两组实验编排。
- `analysis/behavioral_anchoring_results.py`：独立端点核对、统计和图。
- `tests/test_behavioral_anchoring.py`：新逻辑的定向测试。

所有输出位于 `logs/continual_c4/20261104_behavioral_anchoring/`：`contract.json`、
`anchors.pt`、两组 `initialization.json`、`anchoring.jsonl`、TensorBoard、
checkpoint、curriculum/episode 日志和四次六套评估。
最终 pipeline 状态为 `completed`，进程 exit code 0；独立统计程序再次核对全部
端点 reference、约束与 Adam 计数后正常退出。`git diff --check` 通过。

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
export PYTHONPATH="$PWD/source/flapping_bot:$PYTHONPATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/d50-anchoring-mpl
# 离线分析，不启动训练或评估
./isaaclab.sh -p -m flapping_bot.analysis.behavioral_anchoring_results \
  --output logs/continual_c4/20261104_behavioral_anchoring
./isaaclab.sh -p -m pytest -q tests/test_behavioral_anchoring.py tests/test_c4_training.py
```

历史文件、用户已有未跟踪内容均保留；未 commit、push 或修改 Git 历史。
结论仅限单 seed、短程、旧固定 plant。锚点缺少危险/recovery 状态，D-50 自身有旧任务
误差；本轮不能证明全 C1–C4 问题已解决，更不能外推到新气动模型或真实飞行。
