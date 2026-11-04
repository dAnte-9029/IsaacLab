# D-50 尾舵行为锚定：训练前固定合同

本轮仅比较 S/H，seed 0，各 25 次 PPO iteration、307,200 条新 on-policy
transitions。历史 worktree `feat/native-multibody-rl`，起点 HEAD `79a17245`。
源为 `logs/continual_c4/20260922_pilot/D/training/model_50.pt`：文件名 50 对应
`completed_iterations=51`。恢复 actor、critic、log_std、19 项 Adam 状态、
固定学习率 5e-5、4 epochs、guard/task iteration、torch RNG、课程概率/历史/私有 RNG。
不恢复 PhysX episode；两组从一致新 episode 开始，策略分歧后轨迹不再配对。
256 CPU-native environments、48 steps、16 minibatches、原六族 PPO 权重与奖励均不变。
两组串行运行，OMP/MKL 各 1 线程；不以资源快照作为性能比较。

## 固定锚点与超参数

复用 `logs/continual_c4/20260922_memory/memory.pt` 的 training_only observations，
六族各 4096，共 24576；不收集新样本，不读 development/holdout observations。
冻结 D-50 split actor，重算全部未裁剪 deterministic action means。
保存 observations、family IDs、目标和完整 reference actor；每个新 checkpoint
也嵌入该锚点对象及 mode/update，resume 不依赖另外重算 teacher。
原 full-C3 memory loss 关闭；原 collection provenance 仅描述 observations 的来源。

输入仍为 30×14 sensor + 30×4 applied action + 5×3 preview = 555。
sensor 为 heading-canonical wxyz、body ground velocity、p/q/r、forward air speed、
actual frequency、sin/cos phase；沿用既有固定尺度与 safety clipping，没有经验归一化。
tail 使用完整输入；frequency 使用既有一周期平均、phase=(0,1) 的独立支路。
输出顺序为 frequency/rudder/left/right；距离仅对后三维求均方。
缺少 episode/phase 标签，不宣称存在独立 recovery 锚点组或完整危险状态覆盖。

S：lambda=1.0、w_k=1/6，每个 PPO minibatch 对完整锚点计算精确均方辅助损失。
系数继承历史 memory coefficient 的量级；分母为 3 而非历史 4 个通道。
full-bank 梯度和 PPO 梯度相加后共同执行既有 global gradient clipping。
不搜索系数，不根据本轮评估调整。

H：各族 epsilon_k=0.0004，即 RMS≤0.02 归一化动作单位，约 [-1,1]
全跨度的 1%，显著小于历史 D50→D75 的约 0.1–0.3 尾舵漂移。
这是保守功能容差，不是物理舵角或已验证的闭环安全界限。
每次 Adam minibatch step 检查全部六族真实输出；候选失败后依次尝试
alpha=1/2,…,1/1024，均失败则拒绝尾舵步。
部分接受保留候选 Adam moments/step，等价于该梯度在 alpha×lr 下的单次
Adam 尾舵步（无 weight decay）；拒绝精确恢复尾舵参数和 moments/step。
critic、frequency、log_std 均保留正常候选步，不回退、不冻结。
约束是有限固定锚点上各族平均 MSE，不保证每条样本最大误差，更不保证任意闭环状态。

## 验证与判别

新定向测试验证：真实 Adam 接受/回溯/拒绝、任务族不可被平均掩盖、仅 tail
辅助梯度、无约束触发时与安装版 PPO 参数/optimizer 完全一致、reference/state resume。
运行时断言实际恢复的 policy/optimizer/curriculum 与源相等；每 iteration 再查全锚点。
记录各族 tail、三舵、sym/diff、frequency RMS，PPO 损失、训练 reward/轨迹、
接受/回溯/拒绝、alpha 总量、参数变化、约束开销与 wall time。

起点复用历史 D50 正式六套评估；追加更新 12/25 在独立进程执行原 C4 development
及 C1/C2c/C3a/C3b/C3c 冻结评估。绝不打开 sealed holdout。
D50 的 C3b 横向均误差 0.505992 m 已超 0.5 m，因此保持它不等于解决 C1–C4。
比较完整闭环结果与实际接受更新量；若 H 接近冻结，不能宣称保持了继续学习能力。
单 seed、短程、旧 plant 下的结果不推广至新气动模型或真实飞行。

输出固定在 `logs/continual_c4/20261104_behavioral_anchoring/`，拒绝覆盖历史结果。

## 执行时查明的共同初始化边界

S 完成后核对 wrapper reset 顺序发现：`RslRlVecEnvWrapper.__init__` 先调用
`env.reset()`，现有 `train_c4` 随后才 load curriculum/torch RNG。因此首批
路径使用 seed-0 新建课程的采样状态；D-50 的概率/统计/私有 RNG 随后恢复，
作用于后续 reset。首批路径并非从已恢复的 D-50 curriculum 抽取。
本轮保持两个方案相同的既有顺序，不改变 H、不追加 S；这是配对的部分运行状态
恢复，不能写成完整 D-50 仿真/采样轨迹续跑。Adam、actor/critic/std 和固定调度
确实完整恢复，仍可检验这一起始条件下的 S/H 差异。该边界在 S 完成后发现，
不能追称为已通过的训练前恢复检查。
