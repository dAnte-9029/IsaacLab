"""Generate comparable frozen metrics and Chinese evidence tables; never train or evaluate."""
import os,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[2];base=Path(os.environ.get('TRIM_OUTPUT_ROOT',root/'logs/unified_trim_centered_pilot_v1'))
data={g:json.loads((base/f'{g}_diagnostics.json').read_text()) for g in 'AMB'}
names=['simple','sustained','reverse_right_first','reverse_left_first','coupled_3d']
updates=[0,5,10,15,20,25]
endpoint_audit=json.loads((base/'endpoint_audit.json').read_text())
endpoints={(r['group'],r['update'],r['slot']):r for r in endpoint_audit['results']}
lines=['# Unified Trajectory RL：配平中心 PPO 初始化实验','', '基准 `9e0ba64884d2bb81066adc0e3a4e269becf0179b`；L3D / Fixed / seed 0。', '', '## 可比性与执行边界','', 'A 复用历史 25 updates，仅额外执行一次 48-step rollout/PPO 复现核对。初始 digest、12,288 条逐步物理与动作证据（仅剔除运行版本号）、reward、loss、参数变化量及结束事件完全一致。M/B 各 256×48×25=307,200 transitions，合计新增正式训练 614,400；验证 rollout 单独记录，未重训 A。', '', '网络、39D/4D、PPO 数学定义、reward、任务、物理与执行器未变。初始化只修正 actor.6.bias；B 额外将可学习 std 初始化为 0.15。真实 M/B reset 观测、Actor/Critic 权重和 bias 偏移逐 tensor 配对检查。Update 0 是各组真实训练策略保存，不在评价环境重算。', '', 'A 的物理合同包含旧绝对资源路径，第一次跨 worktree 补评被拒绝且未执行控制步；失败目录保留。A 补评改用原入口，输出仍在新目录。两工作区物理合同的差异仅资源路径，物理源码/URDF 指纹一致，且首轮逐步数值完全一致。没有放宽物理校验。', '', '训练配置：CPU-native PhysX 480 Hz、control 60 Hz；256 env、48 rollout steps、25 updates、5 epochs×4 minibatches、lr 3e-4、threads=1；save=5、正常 eval=25、development 上限7200 steps/env。未执行 final 六条任务。', '', '## 初始化合同','']
for g in ['M','B']:
 c=json.loads((base/g/'initialization_contract.json').read_text())
 lines += [f'**{g}**：mode `{c["mode"]}`，trim `{c["trim"]}`；std `{c["initial_std"]}`。',f'偏移 `{c["bias_offset"]}`；调整前均值 `{c["before_mean"]}`；调整后 `{c["after_mean"]}`。',f'初始 policy digest `{c["initial_policy_digest"]}`。','']
lines += ['trim 由实际候选、运行频率范围、URDF 限位及 softness 计算，并对实际 reset action 核对。它是有限时域前飞候选，不代表严格稳定周期配平。全部配置在各组 `contract.json`、checkpoint 与 `source_snapshot/`；初始化前后完整参数及观测在 `initialization.pt`。','','## 冻结评价总表','','存活栏是停止前有效时长，成功完成会提前截断，不能将其视作失败时间。航程为路径投影高水位，不是路程积分。H/V 是逐任务 RMSE 的均值；全程指标受存活长度影响，另列共同窗口。','', '|组|Update|存活均值 s|中位 s|航程 m|H RMSE m|V RMSE m|协议完成|严格终点完成|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
summary={};common=min(125,min(len(t['errors']) for d in data.values() for rows in d['details'].values() for t in rows))
for g,d in data.items():
 summary[g]={}
 for u in updates:
  rows=d['frozen'][str(u)]['terminal_rows']; vals={k:float(np.mean([r[k] for r in rows])) for k in ['seconds','distance','h','v']};vals['median']=float(np.median([r['seconds'] for r in rows]));vals['completed']=sum(r['outcome']=='completed' for r in rows);vals['strict_completed']=sum(endpoints[g,u,i]['strict_completed'] for i in range(5));summary[g][u]=vals
  lines.append(f'|{g}|{u}|{vals["seconds"]:.3f}|{vals["median"]:.3f}|{vals["distance"]:.3f}|{vals["h"]:.3f}|{vals["v"]:.3f}|{vals["completed"]}/5|{vals["strict_completed"]}/5|')
lines += ['',f'## 共同前 {common} steps 与逐任务证据','',f'共同前缀在全部 18 个策略、五条任务之间取 min(125, 最短有效轨迹)。任何不足125步的策略单列：`{[(g,u,t["slot"],len(t["errors"])) for g,d in data.items() for u,rs in d["details"].items() for t in rs if len(t["errors"])<125]}`。','', '|组|Update|任务|存活 s|航程 m|全程 H/V m|共同窗 H/V m|首个约束|末端投影时刻 s|首次末端3D误差 m|停止时真实终点误差 m|越过|协议完成|严格完成|','|---|---:|---|---:|---:|---|---|---|---:|---:|---:|---|---|---|']
for g,d in data.items():
 for u in updates:
  es={r['id']:r for r in d['frozen'][str(u)]['terminal_rows']}
  for t in sorted(d['details'][str(u)],key=lambda t:t['slot']):
   ep=endpoints[g,u,t['slot']]
   e=es[t['episode']];xyz=np.array(t['errors'][:common]);h=float(np.sqrt(np.mean(np.sum(xyz[:,:2]**2,axis=1))));v=float(np.sqrt(np.mean(xyz[:,2]**2)));t['common_h']=h;t['common_v']=v
   endtime=f'{t["first_end_step"]/60:.3f}' if t['first_end_step'] else '—';enderr=f'{t["first_end_error"]:.3f}' if t['first_end_error'] is not None else '—'
   lines.append(f'|{g}|{u}|{names[t["slot"]]}|{e["seconds"]:.3f}|{e["distance"]:.3f}|{e["h"]:.3f}/{e["v"]:.3f}|{h:.3f}/{v:.3f}|{"+".join(e["triggers"]) or e["outcome"]}|{endtime}|{enderr}|{ep["terminal_endpoint_error"]:.3f}|{t["overshoot"]}|{e["outcome"]=="completed"}|{ep["strict_completed"]}|')
lines += ['', '投影达到60 m不自动完成。原协议实际要求 progress≥59.5 m 且到当前投影路径点误差≤0.5 m，并非严格终点球半径0.5 m；本报告另用实际固定终点三维距离审计。B25 sustained协议完成，但真实终点误差0.620187 m；全部90条冻结记录均未观测到真实终点≤0.5 m。协议早停后的行为未知。“越过”按达到末端投影后终止时沿切线超越判定。','','## 训练动作、裁剪与 governor','','向量顺序 frequency/rudder/left/right；分布 std 是跨状态样本散布，不是策略高斯参数。实际尾关节角与 applied 命令分别保留。','', '|组|更新窗|raw 均值|raw std|clip %|raw距trim L2|applied距trim L2|请求/governor/实际 Hz|governor限速 %|实际关节均值 °|','|---|---|---|---|---|---:|---:|---|---:|---|']
def vec(x):return '['+', '.join(f'{a:.3f}' for a in x)+']'
for g,d in data.items():
 for window,v in sorted(d['training']['groups'].items(),key=lambda item:int(item[0])):
  w=int(window);lines.append(f'|{g}|{5*w+1}–{5*w+5}|{vec(v["raw_command"]["mean"])}|{vec(v["raw_command"]["std"])}|{vec(np.array(v["clip_fraction"])*100)}|{v["raw_trim_distance"]["mean"]:.3f}|{v["applied_trim_distance"]["mean"]:.3f}|{vec(v["freq"]["mean"])}|{v["governor_limited_fraction"]*100:.2f}|{vec(v["joint_deg"]["mean"])}|')
lines += ['', '首轮 rollout（任何 PPO 更新前）的动作分布：', '', '|组|raw 均值|raw std|裁剪 %|raw距trim L2|applied距trim L2|', '|---|---|---|---|---:|---:|']
for g,d in data.items():
 v=d['training']['per_update']['1']['actions']
 lines.append(f'|{g}|{vec(v["raw"]["mean"])}|{vec(v["raw"]["std"])}|{vec(np.array(v["clip_fraction"])*100)}|{v["raw_trim_distance"]:.3f}|{v["applied_trim_distance"]:.3f}|')
lines += ['', '完整 raw/clipped/applied 的均值、std、P10/P90，尾舵命令与实际偏角、每更新 reward 分项和存活统计，见 `{A,M,B}_diagnostics.json`。频率请求经 governor 和相位驱动过滤，不能用请求幅度替代实际扑频。','','|组|Update|高斯 std|训练 reward¹|value loss|surrogate loss|entropy|','|---|---:|---|---:|---:|---:|---:|']
for g,d in data.items():
 for u in updates:
  r=d['training_reports'][u-1] if u else None
  lines.append(f'|{g}|{u}|{vec(d["checkpoint_std"][str(u)])}|'+('|'.join(f'{x:.6f}' for x in [r['reward_mean'],r['losses']['value_function'],r['losses']['surrogate'],r['losses']['entropy']]) if r else '—|—|—|—')+'|')
lines += ['', '训练 episode 按结束所在窗口汇总，排除 manual_reset 删失；不是冻结单策略成绩。', '', '|组|窗口|结束数|存活均值 s|航程均值 m|H/V RMSE m|rollout reward均值|', '|---|---|---:|---:|---:|---|---:|']
for g,d in data.items():
 for w,v in sorted(d['training']['groups'].items(),key=lambda item:int(item[0])):
  i=int(w);e=v['terminated_episode_stats'];rr=np.mean([r['reward_mean'] for r in d['training_reports'][5*i:5*i+5]])
  lines.append(f'|{g}|{5*i+1}–{5*i+5}|{v["terminated_count"]}|{e["seconds"]["mean"]:.3f}|{e["distance"]["mean"]:.3f}|{e["h"]["mean"]:.3f}/{e["v"]["mean"]:.3f}|{rr:.5f}|')
lines += ['', 'Actor 均值来自匹配 checkpoint 的下一轮 rollout 重建观测，排除 episode 第一步；这是有权重的更新点子样本，不代表整窗全量。', '', '|组|权重 Update|样本数|Actor均值|均值到trim L2|', '|---|---:|---:|---|---:|']
for g,d in data.items():
 for u,v in sorted(d['training']['actor_means'].items(),key=lambda item:int(item[0])):
  lines.append(f'|{g}|{u}|{v["n"]}|{vec(v["mean_stats"]["mean"])}|{v["distance_trim"]["mean"]:.3f}|')
lines += ['', '¹ 对应单个更新的 rollout reward，非冻结性能。训练 episode 可跨多个策略版本，不能当作单 checkpoint 成功。','','## 同状态 Actor 漂移','','每组先在自身 Update 10 已访问的非终止后观测上对比后期权重；另用 M/B 两组状态的固定并集，对两组作完全相同输入的比较。它识别函数漂移，不单独证明物理因果。','', '|组|权重差|观测数|均值变化|RMS变化|','|---|---|---:|---|---|']
for g,d in data.items():
 for u,v in d['same_state_drift'].items():lines.append(f'|{g}|{u}−10|{v["n"]}|{vec(v["mean"])}|{vec(v["rms"])}|')
def actor(x,sd):
 for i in [0,2,4,6]:
  x=F.linear(x,sd[f'actor.{i}.weight'],sd[f'actor.{i}.bias'])
  if i<6:x=F.elu(x)
 return x
pool=torch.cat([torch.load(base/f'{g}_update10_visited_observations.pt',weights_only=True) for g in ['M','B']]);paired={}
with torch.no_grad():
 for g in ['M','B']:
  sd10=torch.load(base/g/'model_000010.pt',weights_only=False)['policy'];paired[g]={}
  for u in [15,20,25]:
   delta=actor(pool,torch.load(base/g/f'model_{u:06d}.pt',weights_only=False)['policy'])-actor(pool,sd10)
   v=dict(n=len(pool),mean=delta.mean(0).tolist(),rms=delta.square().mean(0).sqrt().tolist());paired[g][u]=v
   lines.append(f'|{g}（共同状态池）|{u}−10|{len(pool)}|{vec(v["mean"])}|{vec(v["rms"])}|')
(base/'comparison.json').write_text(json.dumps(dict(summary=summary,common_steps=common,details={g:d['details'] for g,d in data.items()},paired_drift=paired),indent=2))
lines += ['', '## 恢复与复现','', '`recovery_check.json` 检查完整策略、Adam 状态精确加载、Torch/NumPy/Python RNG 重放、SQLite prefix 恢复及所有 checkpoint 初始化合同一致。未额外执行 PPO update。现有 resume 会新建物理 episode；不声称精确延续中途 PhysX 状态。', '', '可续训 checkpoint（本轮不继续）：', '', f'- `{base}/M/model_000025_final.pt`', f'- `{base}/B/model_000025_final.pt`', '', '正常 Update25 评价回写训练状态；其余评价独立进程、独立 SQLite，不修改 sampler/RNG。逐个评价 `summary.json` 提供严格全量 replay、terminal mask 与 policy unchanged 校验。', '', '一键复现（必须使用全新目录，已有目录拒绝覆盖）：', '', '```bash', 'TRIM_OUTPUT_ROOT="$PWD/logs/unified_trim_centered_pilot_v1_reproduction" bash scripts/flapping_px4/reproduce_trim_centered_pilot.sh', '```', '', '监看：', '', '```bash', 'tail -f logs/unified_trim_centered_pilot_v1/{M,B}_console.log', 'cat logs/unified_trim_centered_pilot_v1/{M,B}/status.json', 'cat logs/unified_trim_centered_pilot_v1/{M,B}/latest_checkpoint.txt', '```', '', '单 seed、五条 development 任务，无 noise-only 组：A→M 是均值初始化干预，M→B 是均值居中条件下初始 std 的增量，A→B 是组合效果；不能完整识别独立效应和交互作用，不能宣称泛化。']
(base/'report_tables.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
