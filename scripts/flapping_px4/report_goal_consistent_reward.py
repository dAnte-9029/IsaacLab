"""Generate the Chinese report from independently checked experiment artifacts."""
import os,json,math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
base=Path(os.environ.get('GOAL_OUTPUT_ROOT',root/'logs/unified_goal_consistent_reward_v1'))
def read(path):return json.loads(path.read_text())
data=read(base/'endpoint_comparison.json');rows=data['results'];pairs=data['paired_windows']
diag={g:read(base/f'{g}_diagnostics.json') for g in ['G','R']}
train={g:read(base/g/'summary.json') for g in ['G','R']}
rec=read(base/'recovery_check.json')
f=lambda x:'—' if x is None else f'{x:.3f}'
vec=lambda x:'/'.join(f(v) for v in x)
text=['# Unified Trajectory RL 阶段 6：真实终点一致的学习目标', '',
'基准 `experiment/unified-trim-centered-pilot-v1` / `a3a9a497f4f837f962a7031b7017d555c3bb90d8`。独立分支 `experiment/unified-goal-consistent-reward-v1`；实验完成时未 commit/push，后续发布由用户另行授权。', '',
'## 结果与决策', '']
last={g:[r for r in rows if r['group']==g and r['update']==25] for g in ['G','R','B']}
rate={g:sum(r['strict_completed'] for r in rr) for g,rr in last.items()}
text += [f"**VERIFIED：Update 25 严格成功 G={rate['G']}/5，R={rate['R']}/5，历史 B 按真实终点复核={rate['B']}/5（旧协议为 1/5）。**"]
p25=[p for p in pairs if p['update']==25]
agg=lambda g,key:np.sqrt(np.mean([np.array(p[g][key])**2 for p in p25],axis=0))
text += [f"共同前 125 步 H/V/3D RMSE：G={vec(agg('G','prefix125'))} m；R={vec(agg('R','prefix125'))} m。逐任务 G/R 最长共同存活窗口：G={vec(agg('G','common'))} m；R={vec(agg('R','common'))} m。",'']
if rate['R']>rate['G'] and np.all(agg('R','common')[:2]<=agg('G','common')[:2]):
 decision='SUPPORTED：R 提高严格到达率，且未见明显跟踪退化；可保留候选目标进入后续更长训练验证。本轮到此停止，单 seed/五条 development 不证明泛化。'
elif np.mean([r['min_goal_distance_m'] for r in last['R']])<np.mean([r['min_goal_distance_m'] for r in last['G']]) and np.any(agg('R','common')[:2]>agg('G','common')[:2]):
 decision='SUPPORTED：R 的最小终点距离下降，但共同窗口路径误差出现恶化，提示终点引导与跟踪存在冲突。下一步优先研究任务目标协调，不建议直接增加奖励权重或直接扩大训练预算。'
else:
 decision='SUPPORTED：本次固定预算尚不支持 R 改善严格终点控制，不建议直接长训或继续堆积惩罚项。下一步应研究基础稳定、路径跟踪、终点接近三个时间尺度的目标组织，可考虑分层或多目标 RL；本轮不实施下一阶段。'
text += [decision,'','**局部收益与退化须同时保留：** R25 四条非 sustained 轨迹的共同窗口水平、垂直误差均下降；sustained 的共同窗口 H/V 从 G 的 1.515/0.727 m 增至 R 的 2.419/1.405 m，最小终点距离从 0.923 m 增至 4.776 m。不能用平均最小终点距离改善掩盖这一退化。R 的平均最小终点距离在 U15 为 4.162 m，U25 回升至 4.993 m，也未形成持续改善。','','**下一阶段建议（HYPOTHESIS）：** 两组全部保存点均未掌握严格终点控制。保留严格成功定义；R 暂不作为可直接长训的已验证目标。优先研究基础稳定、路径跟踪和终点接近三个时间尺度的协调，包括分层或多目标组织；不新增惩罚、不调权重，不在本轮执行。这里只识别到任务间权衡，尚不能因果归因于 goal 或 recovery 单项。','','该判断基于冻结策略实际轨迹，非 reward 总值。各阶段变化与完整分项见下文；未观察到的终止后行为不作推断。','','## 终止、奖励与合同','','旧条件：`s >= L-0.5 && e <= 0.5`。新条件增加 `||x-p(L)|| <= 0.5`，三个条件同时成立；失败优先于完成，完成优先于 timeout。8 m 路径误差、高度、姿态、角速度和 120 s 时间限制不变。无需终点悬停或零速度。', '',
'终点在 reset 按实际世界坐标轨迹 `Trajectory.query(L)` 以 float64 保存；控制周期前复制机体位置并计算前势函数。部分 reset 只更新选中行。SQLite evaluator 独立从原轨迹重建终点、从原始 `root_state` 计算距离，不信任已记录距离或完成标志。旧 terminal v1 按旧语义只读重放；新 terminal v2 使用证据 evaluator v4。', '',
'令 `T=exp(-e²/4)`、`Δh=max(0,high_water_after-high_water_before)`，`Δa²=mean((a-a_prev)²)`：', '',
'```text\nr_G = Δh*T + (T-1)*dt - 0.01*Δa² + 10*completed - 5*(failed|invalid)\nr_rec = -0.5*dt*(sqrt(e²+2²)-2)/2\nPsi(d) = -d/(d+5)\nr_goal = 0.99*Psi_successor - Psi_before\nr_R = r_G + r_rec + r_goal\n```', '',
'`dt=1/60 s`，距离单位 m。成功/失败/invalid 的吸收后继势函数为 0；timeout 保留物理后继势函数。非有限路径误差不计算无界 recovery 惩罚，置零该不可定义分项，保留 invalid 终止及原失败惩罚；不修复或认证非有限证据。没有额外横向、姿态、速度或尾舵惩罚。', '',
'现有 RSL-RL 对 timeout 加 `gamma*V(s_t)`，不是 `V(s_{t+1,terminal})`；GAE 的 done mask 切断 episode。本轮不修改 PPO/GAE/optimizer，**不宣称 PB shaping 的策略不变性已得到证明**。有限正常 episode 的吸收势函数处理符合折扣望远镜求和；recovery 本身有意改变目标，因此也不能宣称整个 R 保持原最优策略。负势函数下静止也可能产生正 shaping；不能把单步正项解释为朝终点运动。', '',
'新任务身份：`unified_li_ppo_v2`、`unified_terminal_v2`、`frozen_full_batch_v4`，variant 为 `strict_goal` / `recovery_goal`。新 checkpoint 拒绝旧 schema；评价匹配 reward variant，resume 匹配完整合同。G/R 初始 Actor、Critic、可学习 std=0.15、Torch/NumPy/Python RNG 与实际 reset 观测一致。', '',
'奖励权重是预先固定的候选假设，未调参。B 全部 307,200 步只读量级检查：recovery 中位数 −0.001496、最小 −0.026803；非吸收 goal 中位数 +0.009253，范围 −0.014780 到 +0.023265；吸收 goal 在 [0,1]。e=0/2/4/8 m 时 recovery=0/−0.003452/−0.010301/−0.026026。', '',
'## 实施和验证', '',
'- `unified_trajectory_env.py`：固定终点缓存、前状态势函数、严格完成条件、G/R reward、每更新分项遥测。\n- `trajectory_feedback.py`：旧/新合同分派，独立原始几何终点验证及成功 witness。\n- `unified_goal_reward.py`：固定参数、势函数与 recovery 公式。\n- `unified_ppo_contract.py` / `unified_ppo_training.py`：schema、冻结评价身份及 reward 合同；PPO 数学未改。\n- `scripts/flapping_px4/*goal*`：定向验证、运行、恢复、只读分析和报告。核心实现与测试 diff 在实验目录 `implementation.diff`。', '',
'**VERIFIED**：定向测试先 41 项通过，补充后目标专项 19 项通过（合计覆盖 44 个不同测试）；包含边界 0.5 m、投影近但终点远、终点近但进度不足、安全优先、timeout、非有限、方向/场景平移、原始终点 witness 与重放、确定性、公式、旧 checkpoint 拒绝及实际 PPO timeout 接口。', '',
'**VERIFIED**：使用历史 B 首轮原动作，真实 CPU-native PhysX 重放 48×256=12,288 步，原始机体/关节、执行器内部状态、动作、投影及参考量全部精确一致；G reward 与旧公式逐元素精确一致。另 1×256 步检查部分 reset 后势函数不泄漏以及周期前状态保存；无 PPO 更新。物理源码未修改，原 39D/4D、MLP、480/60 Hz、trim-local、60 m B 样条分布及无风设置不变。', '',
'旧 B25 的五条轨迹完整只读重放通过；sustained 按原轨迹 float64 终点独立重算为 0.620188758 m（历史约 0.620187 m 来自 float32 preview 终点，差约 2 μm，不影响拒绝结论），新规则在旧终止步返回“继续”，旧结果未改写。新 12 批冻结评价均重新验证 SQLite 完整记录、原始终点、终止 mask、冻结策略身份及零 optimizer 更新。另对训练成功记录和 witness 逐条用原轨迹/原始终止状态重算：G 的 1 条、R 的 7 条均满足严格终点；这些是随机训练 episode，不能替代冻结成绩。此为数值/仿真控制证据，不是气动力物理真实性或实飞验证。', '',
'## 训练预算与资源', '',
'G/R 均 L3D / Fixed、seed 0、256 env、48 rollout、25 updates、每 5 updates 保存；原学习率 3e-4、5 epochs、4 minibatches、gamma .99、GAE lambda .95、[256,256,256] ELU Actor/Critic。每组 307,200 transitions、500 Adam steps。两组顺序执行，训练和评价不相互并发；未执行 I3D、Adaptive、多 seed、final 留出评价或额外更新。既有 runner 仍将预声明 final 几何元数据写入合同并作一致性比较；没有运行 final 任务或读取其评价成绩。', '',
'|组|训练耗时 s|含正常 U25 评价 loop s|训练 transitions/s|进程峰值 RSS MiB|', '|---|---:|---:|---:|---:|']
for g,s in train.items():text.append(f"|{g}|{s['train_seconds']:.2f}|{s['loop_wall_s']:.2f}|{s['measured_training_transitions_per_s']:.1f}|{s['peak_rss_mib']:.0f}|")
text += ['','启动检查未见其他训练进程，GPU 仅桌面显示占用；主计算为 CPU。G 的只读分析曾与 R 的早期训练并行；未控制整个共享主机负载，耗时不能解释为严格性能基准。队列最初 nohup 未留下存活进程、未创建训练输出；改用 tmux 后各组仅执行一次正式训练。', '',
'## 12 次 development 冻结评价', '',
'每批五条相同轨迹。U25 正常评价保存在训练状态；U0/5/10/15/20 为独立进程和 SQLite，未回写 Fixed 训练。下表 RMSE 为五任务前 125 步的均方后开根；Min goal 为逐任务最小终点距离的均值，不能代替成功率。', '',
'|组|更新|严格完成|前125 H/V/3D RMSE m|平均 Min goal m|', '|---|---:|---:|---|---:|']
for g in ['G','R']:
 for u in [0,5,10,15,20,25]:
  rr=[r for r in rows if r['group']==g and r['update']==u];v=np.sqrt(np.mean(np.array([r['prefix125_rmse_h_v_3d'] for r in rr])**2,axis=0))
  text.append(f"|{g}|{u}|{sum(r['strict_completed'] for r in rr)}/5|{vec(v)}|{np.mean([r['min_goal_distance_m'] for r in rr]):.3f}|")
text += ['','## Update 25 逐条行为', '',
'5/2/1/0.5 为首次进入对应真实终点球的秒数，— 表示本次存活期间未观察到。每条轨迹的全部更新、完整 RMSE、共同最长窗口、近终点误差、投影饱和/越过终点平面与失败原因均见 `endpoint_comparison.json`。越过终点平面仅作几何描述，不作为成功。', '',
'|组/任务|终止 s|最大 s m|最小/最终 dg m|进入5/2/1/0.5 s|最近点 H/V误差 m|投影饱和/越过终点平面|结局|', '|---|---:|---:|---|---|---|---|---|']
for g in ['G','R','B']:
 for r in sorted(last[g],key=lambda x:x['slot']):
  entries=[r['first_entry'][str(v)] for v in [5.,2.,1.,.5]]
  why='+'.join(r['failure_reasons']) or ('严格完成' if r['strict_completed'] else '旧协议早停' if r['outcome']=='completed' else r['outcome'])
  text.append(f"|{g}/{r['name']}|{r['time_s']:.3f}|{r['max_progress_m']:.3f}|{r['min_goal_distance_m']:.3f}/{r['final_goal_distance_m']:.3f}|{'/'.join(f(e['time_s']) if e else '—' for e in entries)}|{vec(r['error_at_closest_goal_h_v_m'])}|{r['projection_saturated']}/{r['crossed_endpoint_plane']}|{why}|")
text += ['','|任务 slot|G/R 共同步数|G H/V/3D RMSE m|R H/V/3D RMSE m|与 B 共同步数|', '|---|---:|---|---|---:|']
for p in p25:
 text.append(f"|{p['slot']}|{p['common_steps']}|{vec(p['G']['common'])}|{vec(p['R']['common'])}|{p['common_with_B_steps']}|")
text += ['','![冻结 U25 实际轨迹误差](../logs/unified_goal_consistent_reward_v1/frozen_update25.png)', '',
'![冻结 U25 水平路径和实际机体轨迹](../logs/unified_goal_consistent_reward_v1/frozen_update25_xy.png)', '',
'## Reward、动作与策略解释指标', '',
'下表各列只用于组内机制解释，**不按 reward 总值跨组排序**。G 的 recovery/goal 实际权重为零；训练遥测同时计算的候选项不代表它们进入 G 的优化目标。分项由原始 SQLite 独立重建，episode 终止以各自合同为准。', '',
'|组|更新|base/rec/goal 均值|高斯 std（4D）|value/surrogate loss|entropy|', '|---|---:|---|---|---|---:|']
for g in ['G','R']:
 for r in diag[g]['training_reports']:
  if r['update'] not in [5,10,15,20,25]:continue
  parts=diag[g]['training']['per_update'][str(r['update'])]['reward_parts_mean'];loss=r['losses']
  text.append(f"|{g}|{r['update']}|{vec([sum(parts[:5]),*parts[5:]])}|{vec(r['policy_std'])}|{vec([loss['value_function'],loss['surrogate']])}|{loss['entropy']:.4f}|")
text += ['','|组|更新窗|raw 动作均值|裁剪比例 %（4D）|实际尾关节均值 °（rudder/left/right）|', '|---|---|---|---|---|']
for g in ['G','R']:
 for k,v in diag[g]['training']['groups'].items():
  j=int(k);text.append(f"|{g}|{j*5+1}–{j*5+5}|{vec(v['raw_command']['mean'])}|{vec(np.array(v['clip_fraction'])*100)}|{vec(v['joint_deg']['mean'])}|")
text += ['','|组|Actor 权重更新|下一轮访问状态上的 Actor 均值|均值距 trim L2|', '|---|---:|---|---:|']
for g in ['G','R']:
 for u,v in diag[g]['training']['actor_means'].items():text.append(f"|{g}|{u}|{vec(v['mean_stats']['mean'])}|{v['distance_trim']['mean']:.4f}|")
text += ['','Actor 均值为 checkpoint 对下一轮已访问状态的重建，不是同一状态分布上的因果比较。raw、clipped、applied 和实际关节分开保存；所有 raw→clipped 及尾舵 applied 映射检查通过。近零动作裁剪不等于跟踪成功，std/熵/loss 也不能替代严格终点指标。', '',
f"独立 reward 重建与训练均值最大差：G={diag['G']['reward_reconstruction_max_error']:.3g}，R={diag['R']['reward_reconstruction_max_error']:.3g}。训练最后 5 个更新中，R 的实际 rudder 均值约 2.56°，G 约 4.94°；这与水平轨迹变化同时出现，但不是尾舵偏置导致变化的独立因果证明。训练 episode 可跨策略版本，只作为解释指标。R 同时加入 recovery 和 goal，当前 G/R 对照不能分别识别两项的独立贡献；没有增加消融实验。", '',
'## 恢复、复现与停止', '',
'G/R 正式 `model_000025_final.pt` 的策略、完整 Adam 状态精确加载；Torch/NumPy/Python RNG 重新抽样一致；SQLite 只读源按 checkpoint terminal 序号 fork 并通过 integrity check。所有保存点初始化合同一致、两组实际物理配置和终止合同一致，学习合同唯一差异为 reward variant。恢复验证没有额外 PPO 更新。现有 resume 重建物理 episode，不声称连续恢复中途 PhysX 状态。', '',
'```bash\ncd /home/zn/IsaacLab/.worktrees/unified-goal-consistent-reward-v1\n# 全新输出目录；已存在的训练/评价目录会拒绝覆盖\nGOAL_OUTPUT_ROOT="$PWD/logs/unified_goal_consistent_reward_v1_reproduction" bash scripts/flapping_px4/reproduce_goal_consistent_reward.sh\n# 监看\ntail -f logs/unified_goal_consistent_reward_v1/{G,R}_console.log\ncat logs/unified_goal_consistent_reward_v1/{G,R}/status.json\ncat logs/unified_goal_consistent_reward_v1/{G,R}/latest_checkpoint.txt\n```', '',
'正式输出 `logs/unified_goal_consistent_reward_v1/`：G/R 下为 checkpoint、optimizer/RNG、训练 SQLite 及 U25 评价；`evaluation_{G,R}_000000/000005/000010/000015/000020` 为其余冻结评价。`endpoint_comparison.json`、`G_diagnostics.json`、`R_diagnostics.json`、`recovery_check.json`、`training_witness_check.json`、`pretraining_gate.json` 和 `historical_replay_check.json` 为对应独立检查。', '',
'历史 A/M/B checkpoint、SQLite 未改写；不自动 commit/push，不扩大训练预算，不进入下一阶段。']
(base/'unified_rl_goal_consistent_reward_zh.md').write_text('\n'.join(text)+'\n')
if base.resolve()==(root/'logs/unified_goal_consistent_reward_v1').resolve():
 (root/'docs/unified_rl_goal_consistent_reward_zh.md').write_text('\n'.join(text)+'\n')
print(decision)
