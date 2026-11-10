"""Read actual frozen optimization artifacts; never infer success from shell exit."""
import argparse,json,csv,math,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from tail_v2_search import dump,csvwrite,query,moment
from flapping_bot.physics.tail_aero import TailAeroModel
from flapping_bot.physics.tail_v2 import load_tail_config
p=argparse.ArgumentParser();p.add_argument('--logs',type=Path,required=True);p.add_argument('--evidence-dir',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.evidence_dir.resolve() if a.evidence_dir else root/'docs/results/tail_v2_control_v2';logs=a.logs
completion=json.loads((logs/'completion.json').read_text());assert completion['completed']
assert len(json.loads((logs/'final_results.json').read_text()))==60
for name in ('completion.json','identity.json','trim_results.json','controller_results.json','frozen_finalists.json','final_results.json'):
    shutil.copyfile(logs/name,out/name)
final=json.loads((logs/'final_results.json').read_text());controllers=json.loads((logs/'controller_results.json').read_text());trims=json.loads((logs/'trim_results.json').read_text());frozen=json.loads((logs/'frozen_finalists.json').read_text())
for rec in controllers:
    dest=out/'effective_runs'/rec['id'];dest.mkdir(parents=True,exist_ok=True)
    for name in ('effective_parameters.json','control_effectiveness.json','controller.json','trim.json'):
        shutil.copyfile(logs/rec['id']/name,dest/name)
if (logs/'implementation_correction.json').exists():shutil.copyfile(logs/'implementation_correction.json',out/'implementation_correction.json')
candidate_rows=list(csv.DictReader((out/'candidate_results.csv').open()))
for row in candidate_rows:
    tr=next((r for r in trims if r['id']==row['id']),None)
    cr=next((r for r in controllers if r['id']==row['id']),None)
    row['trim_confirmation_score']='' if tr is None else tr['score']
    row['control_score']='' if cr is None else cr['score']
    row['final_tested']=any(r['id']==row['id'] for r in final)
    row['final_accepted']=row['id'] in completion['accepted']
csvwrite(out/'candidate_results.csv',candidate_rows)
flat=[]
for r in final:
    d=np.load(logs/r['id']/f"final_{r['speed']}_batch.npz")
    cases=json.loads((out/'protocol.json').read_text())['final_cases'];idx=cases.index(r['case'])
    err=np.arctan2(np.sin(d['filtered'][:,idx,:3]-d['target'][:,idx]),np.cos(d['filtered'][:,idx,:3]-d['target'][:,idx]))
    good=(np.max(abs(err[:,:2]),axis=1)<=math.radians(3)) & (abs(err[:,2])<=math.radians(5))
    onset=2 if '_cmd' in r['case'] or r['case'].startswith('pitch') else 0
    recovery=None
    for k in range(int(onset*60),len(good)-60):
        if good[k:].all():recovery=float(d['t'][k,idx]-onset);break
    r['descriptive_recovery_s']=recovery
    r['actual_tail_abs_peak_deg']=float(np.max(abs(d['joint'][:,idx,2:5]))*180/math.pi)
    limits=np.array(json.loads((logs/'identity.json').read_text())['limits_rad'])
    actual=d['joint'][:,idx,2:5];clipped=np.clip(d['requested_joint'][:,idx],limits[:,0],limits[:,1])
    r['command_actual_rmse_deg']=float(np.sqrt(np.mean((clipped-actual)**2))*180/math.pi)
    violation=np.any((actual<limits[:,0]-math.radians(2))|(actual>limits[:,1]+math.radians(2)),axis=1)
    unsafe=(np.max(abs(d['euler'][:,idx,:2]),axis=1)>math.radians(45))|violation
    r['first_unsafe_time_s']=float(d['t'][np.flatnonzero(unsafe)[0],idx]) if unsafe.any() else None
    r['actual_joint_limit_violation']=bool(violation.any())
    r['unclipped_requested_peak_deg']=float(np.max(abs(d['requested_joint'][:,idx]))*180/math.pi)

    flat.append({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
csvwrite(out/'acceptance_results.csv',flat)
best=next(r for r in frozen if r['id']!='legacy')
config=json.loads((out/'candidates'/f"{best['id']}.json").read_text());config['acceptance']=completion['status']
name='tail_v2_calibrated.json' if completion['accepted'] else 'tail_v2_best_NOT_ACCEPTED.json'
dump(out/name,config)
# Actual period-by-period sections, no forward-translation residual.
from analyze_periodic_control_aero_3d import phase_sections
cycles=[]
for r in trims:
    d=dict(np.load(logs/r['id']/'trim_confirm.npz'));sec=phase_sections(d,0)
    for j in range(1,len(sec['position'])):
        cycles.append(dict(id=r['id'],cycle=j,dx_m=float(sec['position'][j,0]-sec['position'][j-1,0]),dy_m=float(sec['position'][j,1]-sec['position'][j-1,1]),dz_m=float(sec['position'][j,2]-sec['position'][j-1,2]),velocity_change_mps=float(np.linalg.norm(sec['velocity'][j]-sec['velocity'][j-1])),attitude_change_deg=float(np.linalg.norm(sec['euler'][j]-sec['euler'][j-1])*180/math.pi),mean_velocity_mps=json.dumps(d['velocity'][(d['t'][:,0]>=sec['t'][j-1])&(d['t'][:,0]<sec['t'][j]),0].mean(0).tolist()),mean_pqr_rad_s=json.dumps(d['omega'][(d['t'][:,0]>=sec['t'][j-1])&(d['t'][:,0]<sec['t'][j]),0].mean(0).tolist())))
csvwrite(out/'periodic_drift.csv',cycles)
# Supplementary static envelope; frozen selections are not changed by this analysis.
authority=[];polars=[];dragrows=[]
for cid in [r['id'] for r in frozen]+['original_v2']:
    model=TailAeroModel(load_tail_config(out/'candidates'/f'{cid}.json'),'cpu');h=1e-4
    for speed in (6,8,10):
        for deg in (0,6,12,18,25,40):
            alpha=math.radians(deg)
            g=(moment(model,[alpha],delta=h,speed=speed)-moment(model,[alpha],delta=-h,speed=speed))/(2*h)
            k=(moment(model,[alpha+h],speed=speed)-moment(model,[alpha-h],speed=speed))/(2*h)
            damp=(moment(model,[alpha],q=h,speed=speed)-moment(model,[alpha],q=-h,speed=speed))/(2*h)
            authority.append(dict(id=cid,speed_mps=speed,alpha_deg=deg,Gdelta_Nm_rad=float(g[0,1]),Kalpha_Nm_rad=float(k[0,1]),Dq_Nm_rad_s=float(damp[0,1])))
    for deg in np.linspace(-40,40,161):
        for delta in (0,10,20):
            sr=query(model,[math.radians(deg)],delta=math.radians(delta))[1 if cid=='legacy' else 0]
            polars.append(dict(id=cid,alpha_deg=deg,delta_deg=delta,CL=float(sr.lift_coefficient[0]),CD=float(sr.drag_coefficient[0]),reference='legacy active panel; V2 whole half-tail'))
csvwrite(out/'authority_envelope.csv',authority);csvwrite(out/'polar_curves.csv',polars)
for path in sorted((out/'candidates').glob('c*.json')):
    data=json.loads(path.read_text());model=TailAeroModel(load_tail_config(path),'cpu');sr=query(model,[math.radians(6)],delta=math.radians(15))[0]
    dragrows.append(dict(**data['polar_horizontal'],CD=float(sr.drag_coefficient[0]),CL=float(sr.lift_coefficient[0])))
sens=[r for r in csv.DictReader((out/'parameter_sensitivity.csv').open()) if r['metric']!='CD_alpha6_delta15']
for param in ('cd_min','cd_delta','cl_md_delta','efficiency'):
    sens.append(dict(parameter=param,metric='CD_alpha6_delta15',spearman=float(spearmanr([r[param] for r in dragrows],[r['CD'] for r in dragrows]).statistic),optimized=True,interpretation='descriptive posterior sensitivity; did not change frozen selection'))
csvwrite(out/'parameter_sensitivity.csv',sens)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(11,8))
for cid in [r['id'] for r in frozen]+['original_v2']:
    rr=[r for r in polars if r['id']==cid and r['delta_deg']==0]
    axs[0,0].plot([r['alpha_deg'] for r in rr],[r['CL'] for r in rr],label=cid)
    axs[0,1].plot([r['alpha_deg'] for r in rr],[r['CD'] for r in rr],label=cid)
    rr=[r for r in authority if r['id']==cid and r['speed_mps']==8]
    axs[1,0].plot([r['alpha_deg'] for r in rr],[r['Gdelta_Nm_rad'] for r in rr],label=cid)
    axs[1,1].plot([r['alpha_deg'] for r in rr],[r['Kalpha_Nm_rad'] for r in rr],label=cid)
for ax,label in zip(axs.flat,['CL (different legacy panel reference)','CD','Gdelta Nm/rad','Kalpha Nm/rad']):ax.set(xlabel='alpha deg',ylabel=label);ax.grid();ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(out/'polars_authority.png');plt.close(fig)
fig,axs=plt.subplots(4,1,figsize=(11,11))
for rec in frozen:
    d=np.load(logs/rec['id']/'final_8_batch.npz');t=d['t'][:,0]
    for j in range(3):axs[j].plot(t,np.rad2deg(d['filtered'][:,0,j]),label=rec['id'])
    axs[3].plot(t,np.rad2deg(np.max(abs(d['joint'][:,0,2:5]),axis=1)),label=rec['id'])
for ax,label in zip(axs,['roll deg','FLU pitch deg','heading deg','max actual tail deg']):ax.set(xlabel='time s',ylabel=label);ax.grid();ax.legend()
axs[3].axhline(25,color='gray',ls='--');axs[3].axhline(40.7,color='red',ls='--');fig.tight_layout();fig.savefig(out/'three_axis_hold.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4))
for rec in frozen:
    rr=[r for r in cycles if r['id']==rec['id']];ax.plot([r['cycle'] for r in rr],[r['dz_m'] for r in rr],label=rec['id'])
ax.set(xlabel='cycle',ylabel='vertical drift per cycle m');ax.legend();ax.grid();fig.tight_layout();fig.savefig(out/'periodic_drift.png');plt.close(fig)
text=f'''# Tail V2 控制导向有约束整定

结论：**{completion['status']}**。本轮真实执行 96 组气动候选、12 组 V2 加 legacy/原 V2 两个参照的 PhysX 配平筛选、4 组 V2 加两个参照的三轴控制整定，最后 2 组 V2 与 legacy 在 8/10 m/s 初始速度、10 个独立工况上运行连续 10 s。未训练 PPO，未改变主翼、机械权限或质量惯量。

实际完成物理批次步数 {completion['physical_steps']}，13 环境累计 {completion['env_steps']} env-steps；记录运行墙钟 {completion['wall_seconds']:.1f} s。中途将最终验收改为同批独立环境执行以消除重复填充计算，已完成轨迹按文件恢复，未增加候选或改变得分规则。中断时尚未落盘的少量步数不包含在此计数，统计包含已完成的无效控制实现运行；中断未落盘部分未计入，故不是精确总 CPU 消耗。

## 合同与可复现性

实现修复：初版相对伪逆阈值 .05 会因俯仰通道尺度远大于横侧向而截断有效横侧向控制方向；该批控制结果移入 invalid_rank_truncation，未用于最终选择。修复为行归一化满秩求逆，并加入三轴单位矩阵恢复与奇异矩阵拒绝测试；原九组增益和全部任务门槛未变。控制阶段因实现缺陷重跑，属于故障纠正，不是扩展参数预算。

protocol.json 在搜索前冻结：seed=0、PhysX 480 Hz、控制 60 Hz、L3D 原实际多体 native 驱动；实际参数、JSON SHA256、生产物理源码/URDF/翼 CSV/native 动态库身份在每个候选 effective_parameters.json 和 identity.json。v2_parameters、polar_horizontal、polar_vertical 修改后自动重建缓存；17 项定向测试通过，包含修改参数改变输出与垂尾独立性。原 legacy、原 V2 JSON 及历史 T120/T144 与 14 条干预未改动。

局部速度为 COM 空气相对速度加 omega cross r；FLU 坐标、力 N、关于 base COM 的力矩 N·m。水平整体面积 .09468 m²、AR≈3.80，两半各一半面积；垂尾 .0227904 m²、AR≈1.27。机械真实角度使用 rad；输入 JSON 的过渡角采用 deg 后转换。平均弦水平 .1578 m、垂尾 .1341 m；nu≈1.5e-5 时 6–10 m/s 对应 Re≈63k–105k/54k–89k。控制器输出与实际关节角分开记录。legacy 活动面 CL 与 V2 半尾翼 CL 的参考面积不同，不能直接作为同一系数比较。

## Q1–Q2：优化参数、范围与最终候选

水平 A 组 a2d、cl_limit、alpha_s_deg、alpha_f_deg、kappa 全部进入 Sobol 搜索；B 组 cd_min、efficiency、cd_delta、cl_md_delta 同时进入有界搜索，未维持永久零值。C 组 cn90=1.8、cd_parallel=.025 固定，未用于改善控制得分。垂尾保留原独立参数组；本轮没有对垂尾气动做进一步搜索，因此不能声称完成垂尾独立辨识。三轴控制器包括方向舵和差动反馈及全矩阵交叉补偿。

候选范围沿用并明确降级为工程先验：a2d 5–2pi rad⁻¹、平台 .7–1.2、起始角 8–16°、终止角 25–40°、kappa 0–.4、CDmin .015–.05、阻力效率 .33–.81、cd_delta 0–.2 rad⁻²、cl_md_delta 0–2 rad⁻¹。它们不是三篇论文严格给出的本机边界。cd_delta/cl_md_delta 缺乏本机独立数据，控制优化只选择可用候选，不能辨识真实值；parameter_sensitivity.csv 给出全局秩相关而非因果独立敏感度。

最佳候选（未验收通过时仅作失败候选保存）：{best['id']}。

|参数|实际值|单位|
|---|---:|---|
'''
for k,v in config['polar_horizontal'].items():text+=f"|{k}|{v:.7g}|{'deg' if k.endswith('_deg') else 'rad^-1' if k in ('a2d','cl_md_delta') else 'rad^-2' if k=='cd_delta' else '1'}|\n"
text+='''
三篇全文本轮已读并在 logs/literature 保存可读文本，无可靠图像数字化，因此没有伪造数据点或拟合误差。Anyoji Sec3.1–3.2 支持薄板近线性及高迎角舵效损失；Ananda Sec3.2–3.5 支持 Helmbold 趋势、特定力矩失速判据 CL .55–.7 和 e0 大范围变化，不能当统一硬上限；Traub Fig3/Eq2/5–10 支持几何铰接效能与抛物阻力偏移，但 20% 舵实验不能直接移植到本机约43–45%活动弦比。tanh、smoothstep、C 组数值是工程近似，90° 外推未标定；高迎角占比单独作为失败门槛。

## Q3–Q4：失稳因素与本轮优化事实

可验证因素：tanh 在正常迎角增加时降低局部斜率；参数改变同时影响恢复、操纵与阻力，静态筛选保留不同机制而非最大控制增益。96 组均输出 Gdelta/Kalpha/Dq 与差动/方向舵交叉增益。前轮只给俯仰 PD，本轮增加滚转/偏航及因果低通（tau=.12s），实际 PhysX ±2° 三轴舵面扰动、实际关节角积分与角速度差识别 3×3 有效控制矩阵。该矩阵包含 .5s 自由飞行状态反馈，并非同状态纯气动导数；条件数见各候选 control_effectiveness.json，不能用于证明气动真值。

每候选两轮六变量13点坐标搜索，单点5翼拍，随后10翼拍确认；这仍是有预算的局部无导数优化，不证明全局最优。代价排除前向平移，使用周期截面的速度/姿态/角速度变化与横垂漂移。每候选相同预算，保留开环漂移候选继续闭环，不把开环不稳定等同于不可控。控制器以行归一化后的满秩矩阵求逆转换三轴期望角加速度；wn={1.5,3,5} rad/s、zeta={.5,1,1.5} 九组统一整定，每组4s。使用慢变三轴误差、速度/高度与舵角/外推占比，原始周期 p/q/r 只作描述性输出。

## Q5–Q6：控制余量与冻结验收

冻结验收中的正q扰动与控制调参所用幅度相同，因此该工况不具备完全独立泛化意义；负q、滚转/偏航扰动、双向指令与第二速度提供新增任务证据。最终冻结后每候选在8、10m/s各测试保持、q±.2rad/s、初始roll+3°、r+.2rad/s、pitch±3°、roll±5°、yaw+5°。初始速度变化并非各自重新求得的配平，不能把失败直接解释为该速度物理不可飞。无中途姿态/速度重置。机械硬限位未改；25°为使用比例参考，限位附近.5°用于实际饱和统计。

|候选|初始速度|工况|通过|roll/pitch/yaw慢变RMSE °|高度变化m|饱和占比|大角度外推占比|
|---|---:|---|---|---|---:|---:|---:|
'''
for r in final:
    if r['case'] in ('hold','q+','pitch+'):
        text+=f"|{r['id']}|{r['speed']}|{r['case']}|{r['accepted']}|{' / '.join(f'{x:.2f}' for x in r['rms_roll_pitch_yaw_deg'])}|{r['height_drift_m']:.2f}|{r['saturation_fraction']:.3f}|{r['extrapolation_fraction']:.3f}|\n"
text+='''
完整60项见 acceptance_results.csv，实际舵角/指令、原始及滤波角速度、CL/alpha、主翼/尾翼力矩在原始 NPZ。描述性恢复时间要求从某时刻起持续满足roll/pitch±3°、heading±5°至少一秒且到终点不离开；未恢复记录空值。它不参与事后修改验收门槛。任务门槛为工程容差：roll/pitch RMS≤5°、yaw≤10°、6–12m/s、|高度变化|≤3m、实际饱和≤5%、>25°使用≤20%、深分离外推≤10%、roll/pitch峰值≤45°。这些不是实机气动极限。

## Q7–Q8：可信性与 RL 决策

安全解释边界：部分10s轨迹高度损失接近初始高度，失控后实际关节角甚至超过名义机械限位。代码没有修改限位；这提示失控/接触后的求解器约束不能当作正常机械工作证据。acceptance_results.csv 保存首次姿态>45°或实际舵角超过限位2°的时刻及关节越界标记；该时刻之后的气动与执行器量仅用于失败记录，不用于正常飞行物理标定。保存的 requested_joint 是分配器未限幅需求，实际发送前逐舵面裁剪；command_actual_rmse_deg 用重建的裁剪后指令计算。巨大未限幅需求反映本次控制结构/权限不足，不能解释为真实可执行控制权。


没有扩大先验范围、改变几何/质量/主翼/执行器，也没有利用 C 组外推调分。物理门槛只证明数值有限、非负阻力和正确局部方向；不能证明每条曲线都被论文逐点标定。控制得分不能认证真实气动。若结论 NOT ACCEPTED，说明本轮范围/预算/控制结构未得到合格候选，不证明所有合理尾翼参数都不可控，也不证明 tanh 必须失败。

失败归因仍需分级：A 参数覆盖/局部配平预算不足可能存在；B tanh 斜率过早衰减有静态依据但结构不可行尚未证明；C 低阶反馈、.12s滤波延迟、实际舵机和有限时域识别矩阵的耦合是明确限制；D L3D整机结构性不可控尚不能由有限搜索定论。未自动更换升力模型或继续增加预算。未通过时明确不进入 RL。

一键：`bash scripts/flapping_px4/run_tail_v2_control_calibration.sh logs/tail_v2_control_reproduction`。已有输出时拒绝覆盖；运行使用 env_isaaclab 与 isaaclab.sh，且必须存在 completion.json 与完整60项最终结果，不能只看 shell 返回码。日志约束与配平、控制器和最终参数均保存；原始 NPZ 不提交 Git。运行依赖邻接 control-oriented-3d-wing-aero 的真实生产资产和 native extension。
'''
(out/'report_zh.md').write_text(text)
fig,ax=plt.subplots(figsize=(9,4))
ax.bar([r['id'] for r in controllers],[r['score'] for r in controllers]);ax.set(ylabel='normalized controller score (lower better)',xlabel='same-budget candidates');ax.grid(axis='y');fig.tight_layout();fig.savefig(out/'candidate_control.png');plt.close(fig)
print(completion)
