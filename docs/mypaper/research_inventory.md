# Duty-Cycle / SCCPFM 研究资料清单（分支：`DeLaurier_qsm_modulation`）

> 目的：把**方法**、**数据**、**结果产物**、**已有草稿**与**关键代码位置**整理成一份索引，方便后续论文写作/复现实验。

## 0) 当前代码版本（用于可复现性）

- Git branch：`DeLaurier_qsm_modulation`
- Git commit：`5aa65cca1a8f0930ab9a60d966986c6e8ef9ca58`
- commit message：`feat: add duty-cycle modulation study scripts, paper draft, and results`

## 1) 研究问题与实验类型（Research Types）

### 1.1 研究问题（可写进论文导言/方法摘要）

- 控制输入：周期占比（downstroke ratio）`δ ∈ (0,1)`，表示下扑时间占一个周期 `T` 的比例。
- 调制方式：**时间重参数化**（SCCPFM / phase-warp），保持几何幅值不变，通过改变上下半拍的“时间尺度”来改变瞬时角速度/加速度分配。
- 关注输出：周期平均 `\bar{F}_x`（流向力/推阻力）、`\bar{F}_z`（升力）、`\bar{P}_{in}`（输入功率）以及效率 `\bar{η} = \bar{F}_x U / \bar{P}_{in}`。

### 1.2 实验类型（你已有的数据/脚本覆盖了哪些）

1. **单次风洞仿真（baseline / sanity /画波形）**
   - 固定 `(f, U, pitch)` 与模型参数，输出 `windtunnel_v50.csv` 以及（可选）绘图。
2. **Duty-cycle 扫描（主实验）**
   - 固定 baseline 工况，扫描 `δ ∈ [0.3, 0.7]`，输出 `Fx(δ), Fz(δ), Pin(δ), η(δ)`，并可做 Pareto。
3. **Trim-like 网格扫描（找“模型内自洽”的基准点）**
   - 扫 `(f, pitch, U)`，筛选 `|Fx|≈0` 且 `Fz≈mg` 的点作为比较基准（强调“模型内部 trim”，不等于实机 trim）。
4. **分离模型敏感性（sep on/off、stall 阈值扫描）**
   - 通过调整 `α_stall` 的范围来观察“经常分离→难以 trim”的问题，并记录 sep_ratio。
5. **扭转（twist on/off）消融**
   - prescribed twist（`qd_scaled`）对 `δ→(Fx,Fz,Pin)` 的敏感性影响，用于解释“结构/被动扭转的重要性”。

## 2) 方法要点（Methods）

### 2.1 运动学与调制（关键：避免“相位偏置”）

- 基线无调制与 `phase_warp, δ=0.5` 使用同一族波形（余弦 `q=amp*cos(·)`），避免仅相位不同导致的均值偏差。
- `phase_warp`：通过 `source/flapping_bot/flapping_bot/physics/phase_warp.py` 把一个 `2π` 周期重映射成：
  - 下扑用时 `δT`，上扑用时 `(1-δ)T`
  - 对应瞬时频率：`f_down = f_eff/(2δ)`，`f_up = f_eff/(2(1-δ))`

### 2.2 气动模型（DeLaurier 1993 strip theory）

- 主要实现：`source/flapping_bot/flapping_bot/physics/qsm_delaurier1993.py`
- 输出力在“Wang 共转坐标系”中定义（脚本会再转到 link/world）。
- 参数集中在 `DeLaurierParams`，包括：
  - `alpha0_rad, eta_s, cd_cf, alpha_stall_{min,max}, xi, c_mac, nu, cd_f`
- 可开关分离：`enable_separation`
  - attached：`dN = dN_c + dN_a`，且保留 `dF_x = dT_s - dD_camber - dD_f`
  - separated：只保留法向（并加入横流阻力近似），`dF_x` 置 0

### 2.3 坐标系/力方向约定（写论文时要固定说法）

- world frame：默认 `+x` 向前、`+z` 向上（IsaacSim 常用约定）；数据里 `F_world_x_T` 就是你关心的流向力。
- 风速方向通过 `--flow_dir_world` 指定；默认 `(-1,0,0)` 表示来流沿 `-x`（所以 `+x` 是“向前/推力方向”）。
- `scripts/isaac_wind_tunnel_flappingbot_v50.py` 内部流程：
  1) 计算 `F_c`（Wang 共转坐标系）
  2) `Wang -> link` 旋转矩阵映射
  3) `link -> world` 用翼 link 的 `quat` 映射
  4) 左右翼求和得到 `F_world_*_T`，并用翼位置对 base 做力矩合成

### 2.4 周期平均（写结果时建议统一口径）

- 代码里均值统一用“时间积分/时间平均”（避免简单算术平均受 dt/缺样影响）。
- 论文草稿采用：`dt=1/240 s`、`f=3.8 Hz`、`steps=1200`（5 s ≈ 19 个周期）来保证整数周期窗口。

## 3) 数据与结果产物（Results / Data）

### 3.0 结果目录速查（本地数据在这里）

- `outputs_DeLaurier/runs/`
  - 手工单次运行的归档（`run_000X/` 与若干带参数名的 `run_*` 文件夹），每个里面通常有 `windtunnel_v50.csv` 和一些绘图产物。
- `outputs_DeLaurier/modulation_scan_*`
  - duty-cycle 扫描结果（按不同开关/工况拆分输出目录）。
- `outputs_DeLaurier/trim_scan_*`
  - trim 网格扫描结果（含 `trim_scan_summary.csv / ranked.csv / candidates.csv`）。
- `outputs_DeLaurier/stall_scan_*`
  - stall 阈值敏感性扫描（每个组合一个 `windtunnel_v50.csv`）。

### 3.1 关键输入数据（几何）

- 机翼几何（条带离散输入）：`outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv`
  - 字段：`x_mid_m`（展向中点）、`c_m`（弦长）、`dhat`（俯仰轴在弦向的位置比例，脚本里用 `dhat*c` 画 pitch axis）。
  - 快速可视化：`scripts/plot_wing_geom.py`（输出默认 `outputs_DeLaurier/wing_geom_plot.svg`）。

### 3.2 单次仿真输出（最基础的“原始数据”）

- 每次运行都会写一个 `windtunnel_v50.csv`，核心列：
  - 运动学：`wing_q, wing_qd, wing_qdd`（以及 R 侧对称列）
  - 总力：`F_world_x_T, F_world_z_T`
  - 功率：`power_in_T`
  - DeLaurier 分项：`del_Nc_* del_Na_* del_Fx_* del_sep_ratio_*`
  - AoA 统计：`aoa_mean/max` 与 `aoa_frac_*`（用于“分离占比”替代统计）

### 3.3 Duty-cycle 扫描（主结果）

代表性 baseline（草稿中使用）：
- `f=3.8 Hz, U=7.0 m/s, pitch=13°`
- `sep_off`（`--no-delaurier-enable-separation`）
- `twist_on`（`--twist-mode prescribed --twist-prescribed qd_scaled --twist-eta-max-deg 10`）

主要结果文件夹（已生成）：
- `outputs_DeLaurier/modulation_scan_trim_f3p8_U7_pitch13_steps1200/`
  - `modulation_scan.csv`：每个 `δ` 的周期平均指标
  - `mod_*.svg`：`Fx/Fz/Pin/η/sep_ratio` 等随 `δ` 变化的曲线（带刻度）
  - `pareto.csv`、`pareto_fx_pin.svg`：Pareto 前沿（见 §3.5）
- `modulation_scan_summary.txt`：对 `δ=0.5` 的相对增益摘要（注意：若基准值接近 0，相对增益会非常大）

同类的“开关消融”结果目录（用于写消融/讨论）：
- `outputs_DeLaurier/modulation_scan_sep_off_twist_on/`：sep off + twist on（含汇总与 `mod_*.svg`）
- `outputs_DeLaurier/modulation_scan_sep_on_twist_on/`：sep on + twist on（含汇总与 `mod_*.svg`）
- `outputs_DeLaurier/modulation_scan_sep_on_twist_off/`：sep on + twist off（含汇总与 `mod_*.svg`）
- `outputs_DeLaurier/modulation_scan_sep_off_twist_off/`：sep off + twist off（当前目录仅有逐点 CSV 子目录，未生成汇总文件时可重新跑一次 sweep）

草稿里引用的数值（可直接对齐 `outputs_DeLaurier/modulation_scan_trim_f3p8_U7_pitch13_steps1200/modulation_scan.csv`）：
- `δ=0.30`: `\bar{F}_x≈0.655 N`, `\bar{F}_z≈10.126 N`, `\bar{P}_{in}≈55.63 W`, `\bar{η}≈0.082`
- `δ=0.50`: `\bar{F}_x≈0.095 N`, `\bar{F}_z≈9.209 N`, `\bar{P}_{in}≈42.56 W`, `\bar{η}≈0.016`
- `δ=0.70`: `\bar{F}_x≈0.588 N`, `\bar{F}_z≈8.736 N`, `\bar{P}_{in}≈50.50 W`, `\bar{η}≈0.081`

### 3.4 Trim 扫描（基准点选择材料）

- `scripts/trim_scan.py` 的输出（已生成示例）：
  - `outputs_DeLaurier/trim_scan_sep_off_twist_on/trim_scan_summary.csv`
  - `outputs_DeLaurier/trim_scan_sep_off_twist_on/trim_scan_ranked.csv`
  - `outputs_DeLaurier/trim_scan_sep_off_twist_on/trim_scan_candidates.csv`
- 其中 `candidates` 的筛选逻辑：`|Fx| <= tol` 且 `|Fz - Fz_target| <= tol`（目标可用 `trim-mass-kg` 转 `mg`）。
- 开启分离模型（`sep_on`）时，常见现象是候选点很少甚至为空（见：`outputs_DeLaurier/trim_scan_sep_on_twist_on/trim_scan_candidates.csv` 只有表头）。

### 3.5 Pareto 前沿（用于“多目标权衡”叙述）

- 生成脚本：`scripts/pareto_frontier.py`
  - 目标：`max Fx, max Fz, min Pin`
  - 输出：`pareto.csv`（非支配点集合）+ 可选 `pareto_fx_pin.svg`（Fx–Pin 投影 + frontier 折线）
- 注意：`pareto_fx_pin.svg` 只是一种二维投影；真实 Pareto 是三目标。

### 3.6 stall/separation 敏感性数据（用于“局限性/讨论”）

已有数据目录：
- `outputs_DeLaurier/stall_scan_U6p9_pitch15/`：不同 `α_stall_max/min` 组合下的单次仿真 CSV（每个子目录一个 `windtunnel_v50.csv`）。
- `outputs_DeLaurier/trim_scan_sep_on_twist_on_near_all/`：筛选后 “Fx≈0 且 Fz≥阈值” 的点（并记录 `mean_sep_ratio` 很小的现象）。

## 4) 已有论文草稿（Drafts）

主草稿目录：`docs/mypaper/`

- `main.tex` / `main_prism.tex`：主入口
- `00_abstract.tex`：摘要
- `01_introduction.tex`：引言（问题背景/贡献点）
- `02_related_work.tex`：相关工作
- `03_duty_cycle_modulation.tex`：调制定义与直观解释（δ 改变了什么）
- `04_aero_sim_model.tex`：气动仿真模型描述（DeLaurier + 约定）
- `05_trim_protocol.tex`：trim-like 选择与平均方式
- `06_results.tex`：主结果（δ sweep + 表格）
- `07_pareto.tex`：Pareto 定义与解释
- `08_hardware_flight.tex`：实机部分（可写正/反两种结论，提供 insight）
- `09_limitations.tex`：局限性（sep/stall 不确定、结构滞后、带宽等）
- `10_conclusion.tex`：结论
- `11_reproducibility.tex`：复现实验命令（非常实用）
- `refs.bib`：参考文献
- `plan.md`：写作/实验开展路线图（更偏“项目计划”）

## 5) 主要代码位置与作用（Code Map）

### 5.1 入口脚本（你最常用的命令行）

- `scripts/isaac_wind_tunnel_flappingbot_v50.py`
  - IsaacSim 风洞/台架仿真主脚本
  - 负责：运动学（cos + phase_warp/phase_ff）、扭转（off/prescribed）、DeLaurier 参数、坐标变换、CSV 输出、可视化 debug（draw frames/forces）。
- `scripts/flapping_modulation_scan.py`
  - duty-cycle 扫描器：循环调用风洞脚本、读 CSV、做时间平均、写汇总 CSV/相对 CSV/摘要 txt、画 `mod_*.svg`。
- `scripts/trim_scan.py`
  - `(f,pitch,U)` 网格扫描 + trim 过滤；支持多 GPU worker（每个 worker 一个 `subprocess.run`）。
- `scripts/pareto_frontier.py`
  - 从 sweep CSV 提取三目标 Pareto 非支配点，并输出 `pareto.csv` 和可选 SVG 图。

### 5.2 物理/模型实现（论文方法部分的“源代码依据”）

- `source/flapping_bot/flapping_bot/physics/qsm_delaurier1993.py`
  - DeLaurier(1993) strip-theory 的主要数学实现（含 sep 开关与分项输出）。
- `source/flapping_bot/flapping_bot/physics/phase_warp.py`
  - SCCPFM（phase-warp）重参数化；提供 `cos_kinematics()` 返回 `q, qd, qdd, qddd`。
- `source/flapping_bot/flapping_bot/physics/qsm_wang2016.py` / `qsm.py` / `virtual_twist.py`
  - Wang2016 QSM 与虚拟扭转（本分支研究主线以 DeLaurier 为主，但这些文件用于对照/扩展）。

### 5.3 绘图与检查工具

- `scripts/plot_wing_geom.py`：几何 SVG（planform + pitch axis）
- `scripts/plot_windtunnel_forces_svg.py`：无 matplotlib 的 SVG 曲线（带刻度/网格）：`Fx/Fy/Fz + flap q + tip twist`
- `scripts/plot_windtunnel_wrench.py`：matplotlib PNG（可画 wang/link/world frame 的力矩）
- `scripts/plot_thrust_lift.py`：简化的 `Fx/Fz` 曲线图（历史工具）

## 6) 快速复现实验命令（直接抄即可）

（同 `docs/mypaper/11_reproducibility.tex`）

### 6.1 baseline 单次运行
```bash
./isaaclab.sh -p scripts/isaac_wind_tunnel_flappingbot_v50.py \
  --aero-model delaurier1993 \
  --f_hz 3.8 --airspeed 7.0 --body_pitch_deg 13 \
  --steps 1200 --print-every 0 --headless --livestream 0 --force-exit \
  --no-delaurier-enable-separation \
  --twist-mode prescribed --twist-prescribed qd_scaled \
  --twist-f-ref-hz 4.0 --twist-eta-max-deg 10 --twist-eta-limit-deg 10
```

### 6.2 δ 扫描
```bash
./isaaclab.sh -p scripts/flapping_modulation_scan.py \
  --out-dir outputs_DeLaurier/modulation_scan_trim_f3p8_U7_pitch13_steps1200 \
  --f-hz 3.8 --U 7.0 --body-pitch-deg 13 \
  --steps 1200 --delta 0.3:0.7:0.02 \
  --extra-args "--no-delaurier-enable-separation \
  --twist-mode prescribed --twist-prescribed qd_scaled \
  --twist-f-ref-hz 4.0 --twist-eta-max-deg 10 --twist-eta-limit-deg 10"
```

### 6.3 Pareto
```bash
./isaaclab.sh -p scripts/pareto_frontier.py \
  --in-csv outputs_DeLaurier/modulation_scan_trim_f3p8_U7_pitch13_steps1200/modulation_scan.csv \
  --min-fz 9.0 \
  --out-csv outputs_DeLaurier/modulation_scan_trim_f3p8_U7_pitch13_steps1200/pareto.csv \
  --out-svg outputs_DeLaurier/modulation_scan_trim_f3p8_U7_pitch13_steps1200/pareto_fx_pin.svg
```

## 7) 写作建议（把材料“变成论文”）

- 正文主线建议以 `sep_off + twist_on` 为主（可复现、能找到 trim-like 基准、趋势清晰）。
- `sep_on` 作为“敏感性/局限性”讨论：stall 阈值一变，trim 是否存在就可能变化；这恰好是论文的 insight 点。
- 图表最小集合（RA-L 风格）：
  - `waveform_compare_delta0p5.svg`（或自行再做 δ=0.3/0.7 的波形示意）
  - `mod_Tbar.svg, mod_Lbar.svg, mod_Pinbar.svg, mod_eta.svg`
  - `pareto_fx_pin.svg`
