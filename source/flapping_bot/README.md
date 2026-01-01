# Flapping Bot — Wang2016 Quasi-Steady Aerodynamics

本目录包含扑翼相关扩展代码（环境、物理模型、脚本）。其中 `flapping_bot/physics/qsm_wang2016.py` 实现了论文：

- Wang, Goosen, van Keulen (JFM 2016) “A predictive quasi-steady model of aerodynamic loads on flapping wings”

## 坐标系与输入输出

- **惯性系**：`(x_i, y_i, z_i)`；`x_i-y_i` 为 stroke plane，`z_i` 垂直于 stroke plane。
- **共转系**：`(x_c, y_c, z_c)`（论文 §2.1）：
  - `x_c`：俯仰轴（spanwise）
  - `z_c`：翼面内、与 `x_c` 垂直（chordwise）
  - `y_c`：翼面法向（升力/合力方向）

核心 API：

- `flapping_bot.physics.qsm_wang2016.compute_aero_wrench(...) -> (F_c, tau_c)`
  - 返回共转系下的合力 `F_c=[0,F_yc,0]` 与合矩 `tau_c=[tau_xc,0,tau_zc]`
  - 四项载荷可通过 `wing_geom` 中的 `enable_translation/enable_rotation/enable_coupling/enable_added_mass` 开关

## 几何与 BEM 离散

`WingGeometry.from_input({...})` 支持：

- `R`：翼展（积分上限）
- `N`：spanwise strip 数（默认 20）
- `c` / `chord_func(xc)`：弦长分布
- `dhat` / `dhat_func(xc)`：归一化前缘到俯仰轴偏置 `d̂`
- `eta_shape` / `eta_shape_func(xc)`：可选，分布式扭转形函数 `g(x)∈[0,1]`，用于近似 `η(x,t)=g(x)·η_tip(t)`（刚翼假设的扩展）
- `aspect_ratio`：可选“有效展弦比”覆盖（论文 §3.1 用 `A_eff=2.83`，与几何 `R/c̄` 不同）
- `gyradius_rhat2`：可选；缺省按附录 A (A3) 离散积分计算

## Wagner 效应

`include_wagner=True` 会按论文 (2.26) 对**循环载荷**（平移/旋转/耦合项）施加乘子。

- TODO：Wagner 的 `t*` 需要历史（行进半弦数），当前需在 `wing_geom["wagner_t_star"]` 提供；否则默认乘子为 1。

## 脚本

- 纯 Python 复现（不依赖 IsaacSim）：`scripts/reproduce_wang2016_sweep_pitch.py`
- IsaacLab 可视化/外力 demo：`scripts/isaac_demo_task.py`（对 `wing` link 施加外力/力矩并记录 CSV）
- IsaacLab “风洞台架” demo：`scripts/isaac_wind_tunnel_rig.py`（双翼同步扑动 + 前飞速度 `v_forward`，并记录 CSV）
- IsaacLab “风洞台架（v50 扑翼机）” demo：`scripts/isaac_wind_tunnel_flappingbot_v50.py`（对 `flap_robot_v50.urdf` 的左右翼 link 施加 Wang2016 QSM 外力，并记录 CSV）

## 柔性翼（虚拟被动扭转）

Wang2016 未包含结构动力学。若 URDF 是刚性的但真实机翼会被动扭转，可用一个“虚拟外段扭转”自由度 `η_tip(t)`：

- 设 `d̂(x)=0`（俯仰轴在前缘 LE）时，对应“绕前缘转”的近似。
- 用 `η(x,t)=g(x)·η_tip(t)`（`g(x)` 可用 `eta_shape_piecewise(x0, x1)`：内段 0，外段渐变到 1）。
- 用外段（例如 `x∈[0.42, 0.65]`）的气动矩 `τ_x` 驱动 `η_tip(t)` 的 1-DOF 扭转模型：
  - `step_virtual_twist(...)` 提供 `quasi_static` 与 `dynamic` 两种模式；`k/c/I` 参数需要你用实验标定（TODO）。
