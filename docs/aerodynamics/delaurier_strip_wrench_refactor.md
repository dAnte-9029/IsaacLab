# DeLaurier Strip Wrench Refactor

日期：2026-07-13

## 范围与状态

已实现 DeLaurier attached-flow strip load 暴露与 strip-integrated wing wrench。本记录描述当前代码；未实现 corrected force 或 corrected distribution。

## Frozen baseline: `delaurier-strip-wrench-v1`

本 tag 冻结用于后续 corrected distribution 的 DeLaurier prior 和 wing-moment 链路。它是可复现的数值仿真基线，不表示气动模型已经完成 real-flight 或 sim-to-real validation。默认边界由 `FlappingBotStraightFlightEnvCfg` 与实际调用路径共同定义：

- separation disabled：`delaurier_enable_separation=False`；
- prescribed twist active（historical `v1` behavior）：切换引入前，环境始终使用 `qd`-scaled virtual twist proxy；
- pitching axis at leading edge：`build_wing_geometry_from_csv(..., dhat=0.0)`；
- aerodynamic-centre coefficient：`DeLaurierParams.c_mac=0.0`；
- apparent-mass free couple enabled：`delaurier_include_apparent_mass_moment=True`；
- strip-integrated wing moment enabled：`wing_moment_mode="strip_integrated"`；
- induced drag disabled：`delaurier_induced_drag_efficiency=0.0`。

`dM_ac` 的完整计算、开关和 power-sign test path 都保留，但在该冻结默认值下因 `c_mac=0.0` 而数值为零。任何后续 corrected distribution 实验应记录其 parent 为此 tag，并分别报告 prior force、correction distribution、corrected resultant force 和最终 wrench 的差异；不得把这些层次的失败重新归因于已冻结的 original moment chain，而不先给出新的直接证据。

## Current post-v1 baseline candidate: dynamic twist disabled

日期：2026-07-14。`delaurier-strip-wrench-v1` 保持不可变，因而仍准确记录切换前始终启用的 `qd`-scaled twist proxy。当前分支以 `dynamic_twist_mode` 取代 boolean 开关；三个互斥模式是 `disabled`、`delaurier_linear_spanwise` 和 `legacy_qd_scaled_proxy`。默认配置为：

```text
dynamic_twist_mode = "disabled"
dynamic_twist_tip_amplitude_deg = 0.0
```

因此默认环境仍以 `theta=theta_bar`（沿翼展广播）调用 DeLaurier，并令 `thetad=0`、`thetadd=0`。`dM_a` 的代码路径与独立 moment inclusion switch 保留，但在默认输入下数值为零；`c_mac=0.0` 使 `dM_ac` 同样为零。

当前待用户明确冻结为 successor tag 的可解释性 baseline 边界为：

- separation disabled：`delaurier_enable_separation=False`；
- prescribed dynamic twist disabled：`dynamic_twist_mode="disabled"`、`dynamic_twist_tip_amplitude_deg=0.0`；
- pitching axis at leading edge：`d_hat=0`；
- aerodynamic-centre coefficient：`c_mac=0`；
- `dM_a` code path retained，但在 zero twist-rate 输入下数值为零；
- strip-integrated force-arm moment enabled：`wing_moment_mode="strip_integrated"`；
- induced drag disabled：`delaurier_induced_drag_efficiency=0`。

这不改变 DeLaurier force formula、strip force integration、free-couple 定义或 legacy moment mode。

## Dynamic twist definition

`compute_delaurier_dynamic_twist()` 实现 DeLaurier numerical example 的 prescribed linear-spanwise kinematics：

```text
delta_theta = -theta_tip * (y/R) * sin(phi_D)
delta_theta_dot = -theta_tip * (y/R) * cos(phi_D) * phi_D_dot
delta_theta_ddot = theta_tip * (y/R)
                    * (sin(phi_D) * phi_D_dot^2 - cos(phi_D) * phi_D_ddot)
theta = theta_bar + delta_theta
```

这不是被动柔性翼或气动弹性求解。`dynamic_twist_tip_amplitude_deg` 是理论几何翼尖 `y=R` 的最大动态扭转幅值，配置入口使用 degree，进入 physics helper 前转换为 rad。与原文线性斜率的关系是 `beta_0=theta_tip/R`。

环境优先使用 `WingGeometry.R`。helper 在没有显式 `R` 时使用 `max(y_i+0.5*strip_width_i)`；最后一个 strip center 的幅值通常小于 `theta_tip`，不会被强制归一化到 1。

## Dynamic-twist phase mapping

当前工程 phase 以正方向递增，且 wing command 为：

```text
q = Gamma * cos(current_phase)
h = -q*y = -Gamma*y*cos(current_phase)
```

它与原文 `h=-Gamma*y*cos(phi_D)` 直接一致，所以当前配置与 `resolve_delaurier_phase()` 明确采用：

```text
phi_D = +current_phase + 0
phi_D_dot = +2*pi*f
phi_D_ddot = 0
```

`phase=0, pi/2, pi, 3pi/2` 依次对应当前定义的 top stroke、mid downstroke、bottom stroke、mid upstroke。dynamic twist 与 plunge 相差 90 degree。环境当前把 frequency 视为一个 physics step 内恒定，因此 phase acceleration 为零；helper 已支持未来传入非零值。

左右翼使用同一个 scalar phase、`q/qd/qdd` 和 `theta` sign。相同 local twist distribution 进入两个 Wang frame；右翼 reflection 继续由 `transform_wang_wrench_to_link()` 的 polar/axial 规则处理，不对右翼 `theta` 增加经验性负号。

## Legacy qd-scaled proxy

`legacy_qd_scaled_proxy` 保留冻结历史的 full-span-uniform proxy：其 tip-like scalar 由 clamped `qd/qd_ref` 缩放，且没有 `y/R` 分布。它不是 DeLaurier numerical-example dynamic twist，默认不会进入运行路径，仅用于显式历史 A/B。mode 是单值枚举，因此 legacy 与新模型不能同时启用。

## Affected DeLaurier physics

环境只生成一次 `DeLaurierTwistKinematics`，随后统一把 `theta`、`theta_dot`、`theta_ddot` 传给 `compute_delaurier_strip_loads()`。它们直接或通过 `alpha/alpha_dot/alpha_prime` 影响 `dN_c`、`dN_a`、`dT_s`、`dM_a`、input power、`alpha`、`alpha_prime`、`alpha_le` 以及 separation diagnostics。`delaurier_store_strip_diagnostics=True` 时，最近一步 kinematics 可从 `_debug_last_delaurier_twist_kinematics` 读取；默认不打印或累计。

## Dynamic-twist validation

2026-07-14 实际执行 non-Isaac targeted suite：

```bash
PYTHONDONTWRITEBYTECODE=1 ./isaaclab.sh -p -m pytest -p no:cacheprovider -q \
  tests/test_delaurier_dynamic_twist.py \
  tests/test_delaurier_strip_wrench.py \
  tests/test_delaurier_convention_contract.py \
  tests/test_delaurier_log_export_phase_contract.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_straight_flight_env_reset_contract.py \
  tests/test_startup_phase.py \
  tests/test_flapping_joint_contracts.py \
  tests/test_qsm.py
```

最终结果：`124 passed in 2.38s`。float64 独立数值检查得到：disabled 与 zero-tip regression 最大误差 `0`；span linearity `0`；theoretical-tip semantics `5.55e-17`；special phase points `2.76e-16`；一阶/二阶 centered-finite-difference 误差分别为 `6.20e-9`、`7.29e-8`；non-zero phase-acceleration term `2.22e-16`；mean-pitch addition `2.78e-17`；engineering-to-DeLaurier phase mapping `0`；`dM_a` 直接公式比较 `0`。选定的非偶然消零 case 中，dynamic twist 使 integrated force、moment 和 strip power 的最大变化分别为 `1.086 N`、`0.570 N m` 和 `0.652 W`；这些是测试 fixture 的数值敏感性，不是模型准确性或真实飞行验证。

同日两次尝试运行 `tests/test_delaurier_isaac_wrench_reference.py`，Isaac Sim 都在 pytest 收集前因 `DerivedDataCache` exclusive-lock error 和后续 `cuInit` crash 退出；备份遗留 lock directories 后重试仍相同。因此本次没有把 Isaac regression 写成通过。该 test 在上一阶段的已记录结果仍为通过，但尚未在本次 dynamic-twist diff 上重新验证。

## 修改前

`compute_aero_wrench_delaurier1993()` 在 `qsm_delaurier1993.py` 中计算 strip quantities，但仅返回 Wang frame 的整翼合力，且返回的 moment 为零。`FlappingBotStraightFlightEnv._compute_wing_delaurier_wrench()` 将该合力放到由 `compute_area_weighted_quarter_chord_link_points()` 给出的固定等效 quarter-chord 点，再相对 `root_com_pos_w` 计算 `r x F`。

该路径仍作为 `wing_moment_mode="legacy_fixed_quarter_chord"` 保留，用于 A/B 比较，并保留 separation 的旧 aggregate-force 行为。

## 当前 strip-load 接口

`DeLaurierStripLoads` 由 `compute_delaurier_strip_loads()` 返回。所有 raw component 都是 Wang co-rotating frame 的 `(B, N_strip)` tensor，单位分别为 N 或 N m，且已包含 strip width：

- `dN_c`：circulatory normal force；
- `dN_a`：apparent-mass normal force；
- `dT_s`：leading-edge suction；
- `dD_camber`：camber drag magnitude；
- `dD_f`：skin-friction drag magnitude；
- `dM_ac`：aerodynamic-centre free pitching couple；
- `dM_a`：apparent-mass free pitching couple。

它还暴露 `span`、`chord`、`strip_width`、`d_hat` 及可选的 angle/separation diagnostics。合成属性为：

```text
normal_force_total = dN_c + dN_a
chordwise_force_total = dT_s - dD_camber - dD_f
free_pitching_moment_total = dM_ac + dM_a
```

为兼容旧 API，数据对象同时保留 separation-selected aggregate resultant；它仅由 `compute_aero_wrench_delaurier1993()` 使用。新的 strip wrench 不使用或分配 separation force。默认环境配置 `delaurier_enable_separation=False`。

## 坐标、作用点与积分

Wang frame 是右手系：`x` 为由 wing root 指向翼尖的 span/pitching-axis，`y` 为翼面 normal，`z` 沿 chord 指向 leading edge。正 `theta`/`thetad` 是围绕 Wang `+x` 的右手转动；正 `dM_ac`/`dM_a` 是同一 `+x` 方向的 axial couple。`d_hat` 是从 LE 到 pitching axis 的无量纲距离。当前环境调用 `build_wing_geometry_from_csv(..., dhat=0.0)`，即 pitching axis 在 LE；该参数仍通过 geometry 保留，不在公式中硬编码。

`integrate_delaurier_strip_wrench()` 的 moment reference 是 wing-root pitching-axis origin。normal component 的 strip lever arms 是：

```text
r_Nc = [span, 0, (d_hat - 0.25) * chord]
r_Na = [span, 0, (d_hat - 0.50) * chord]
```

chordwise components 沿 Wang `+z`/`-z` 作用，默认在 strip pitching-axis point `r_T=[span,0,0]` 计算。沿 chord 移动该力不会改变 `r_T x dF_T`，因为位移与 force 平行；但 spanwise lever arm 保留，因此 chordwise force 仍产生 roll/yaw-related moment。

`dM_ac` 和 `dM_a` 不被转换为 application point。它们是 Wang `+x` 方向的独立 axial free couples；加入 moment 后与现有输入功率约定一致：`P_moment = -(dM_ac + dM_a) * thetad`。两个 contribution 分别通过 `delaurier_include_aerodynamic_center_moment` 和 `delaurier_include_apparent_mass_moment` 控制。环境的 `DeLaurierParams.c_mac` 仍为 `0.0`，故默认 `dM_ac` 数值为零但路径存在；`dM_a` 默认计入。

每翼积分为：

```text
F_wing = sum_i dF_i
M_wing,O = sum_i(r_Nc x dF_Nc + r_Na x dF_Na + r_T x dF_T)
            + sum_i(dM_ac + dM_a)
```

在 `wing_moment_mode="strip_integrated"` 中，环境把这个 wrench 从 Wang frame 转到 wing link、再转到 world，并平移到 base COM：

```text
M_G = M_O + (p_wing_origin - p_base_com) x F
```

不会再把已经带有 strip moment 的合力放到 quarter-chord 等效点。

## 左右翼映射

Wang-to-left-link 是 proper transform；Wang-to-right-link 是 span reflection。force 是 polar vector，使用 `A F`。moment 是 axial vector，使用 `det(A) A M`。这与先在物理 frame 中逐项计算 `r x F` 再映射等价，并使对称的 free pitching couple 在左右翼 link/body frame 中同向。link-to-world quaternion 是 proper rotation，因此 force 和 moment 之后使用同一 quaternion rotation。

## API 与 diagnostics

新增：

- `DeLaurierStripLoads`；
- `DeLaurierStripWrench`；
- `compute_delaurier_strip_loads()`；
- `integrate_delaurier_strip_wrench()`；
- `transform_wang_wrench_to_link()`；
- `translate_wrench_moment()`。

`DeLaurierStripWrench` 提供 normal/chordwise resultant force，以及 `dN_c`、`dN_a`、`dT_s`、`dD_camber`、`dD_f`、`dM_ac`、`dM_a` 各自的 wing-origin moment。默认不打印。设定 `delaurier_store_strip_diagnostics=True` 后，环境在 `unwrapped._debug_last_delaurier_strip_loads` 和 `unwrapped._debug_last_delaurier_strip_wrench` 暴露最后一个 physics step 的 tensors。

## 验证

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ./isaaclab.sh -p -m pytest -p no:cacheprovider -q \
  tests/test_delaurier_strip_wrench.py \
  tests/test_delaurier_convention_contract.py \
  tests/test_wing_equivalent_ac.py
```

结果：`10 passed`。`test_delaurier_strip_wrench.py` 检查：关闭 separation 时 strip and legacy total force 的误差不超过 `1e-12`（float64）；named moment component sum 的误差不超过 `1e-12`；chordwise force 在 `0c/0.25c/0.5c` 的 moment 差异不超过 `1e-12`；free-couple switch 的差异精确对应相应 component；镜像静态构造的 lateral force、roll 和 yaw residual 均不超过 `1e-12`。同一 float64 fixture 的显式数值检查得到 force 最大绝对误差 `0`、moment component conservation 最大绝对误差 `0`、chordwise application-point invariance 最大绝对误差 `0`、lateral/roll/yaw symmetry residual `0`。

该验证是 pure-model/unit level，不构成 closed-loop flight、物理模型准确性或 sim-to-real 验证。

## Free-couple sign validation

2026-07-13 新增 `tests/test_delaurier_strip_wrench.py` 的 free-couple power-sign tests，并实际运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ./isaaclab.sh -p -m pytest -p no:cacheprovider -q \
  tests/test_delaurier_strip_wrench.py
```

结果：`25 passed in 0.79s`。测试对 `dM_ac`、`dM_a` 和两者合成项分别覆盖正 moment/正 `thetad`、正 moment/负 `thetad`、负 moment/正 `thetad`，左右翼均覆盖。Wang `+x` 是 spanwise twist axis；按照环境的 Wang-to-link 定义，left/right 的正 axial twist axis 都是 link `+y`。这是因为右翼的 polar span axis 是 link `-y`，但 moment 和 angular velocity 都是 axial vectors，均使用 `det(A) A`。

对所有构造值，测试确认：

```text
M_free,link · omega_twist,link = (dM_ac + dM_a) * thetad
P_input,moment = -M_free,link · omega_twist,link
```

在 float64 fixture 中 left/right、各 component 和 reflection dot-product invariance 的最大绝对误差均为 `0`（容差 `1e-12`）。本次没有修改任何 free-couple 符号或生产代码。

## Isaac wrench reference validation

2026-07-13 新增 headless Isaac Sim integration test：`tests/test_delaurier_isaac_wrench_reference.py`。它以项目 `FlappingBotCfg` 初始化真实 articulation，关闭 gravity，显式读取 `body_pos_w` 的 wing link origin、`body_link_pos_w` 的 base-link origin 和 `root_com_pos_w` 的 base COM。测试先在 world frame 手算：

```text
M_G = M_free + (p_wing_origin - p_base_COM) x F
```

再将该 complete wrench 表达到 base-link local frame，并调用与环境相同的 `Articulation.set_external_force_and_torque(..., body_ids=[base_link], is_global=False)`。Isaac Lab API 在 `is_global=False` 时接收 body link local force/torque；未给 `positions`，因此 force 施加在该 body 的 COM，传入 torque 是关于该 COM 的 free couple。测试直接检查 API input buffers，因而不依赖由动力学响应反推 wrench。

实际运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ./isaaclab.sh -p -m pytest -p no:cacheprovider -q -s \
  tests/test_delaurier_isaac_wrench_reference.py
```

结果：`1 passed in 5.38s`，`dt=0.01 s`。Case A 在左翼测试两个不共线 force direction；Case B 在 left/right 分别测试 zero-force free couple；Case C 在 left/right 分别测试 force-plus-couple。每个 case 的 API buffer `expected == actual`，force 和 moment 输入最大绝对误差均为 `0`（float32 buffer）。例如：

- Case A force Wang `[0,0,80] N`：base-link target force `[80, 2.49e-7, 2.72e-7] N`，target moment `[1.52e-8, -0.122833, -4.367042] N m`；
- Case B free couple Wang `[25,0,0] N m`：left/right base-COM target moment 的主要 `+y` 分量都是 `24.995300 N m`，而镜像几何使其 `z` 分量分别为 `-0.484745` 和 `+0.484745 N m`；
- Case C left/right force-plus-couple：target force 的主要 `+x` 分量均为 `45 N`；target moment 分别为约 `[0,17.927523,-2.805477]` 和 `[0,17.927523,2.932559] N m`。

测试还检查每个 case 的 one-step root-COM linear/angular velocity response 与施加 force/moment 同向。该 reference articulation 仍有可动 wing/tail links，因此内部 joint reaction 会使 root COM 的单步响应不能直接等同为单刚体的 `m a=F` 或 `I_G alpha=M_G`；测试没有把这种多刚体响应伪报为单刚体残差。输入闭合误差为 `0`，而动力学层的已验证量是力/力矩方向的正投影。该测试确认 strip-integrated target 不包含 legacy quarter-chord 的第二个 `r x F` 项。

## 后续 corrected distribution 边界

未来 correction 只能以 strip-level aggregate residual（例如 `delta_dN`、`delta_dFx`）进入，不能声称能唯一辨识它属于 `dN_c`、`dN_a`、suction 或任一 drag component。本次未添加 correction、网络、训练接口或任务级调参。

## 仍存假设

- `theta_a` 的环境 FRD 注释与项目其他 FLU 文字仍有既有冲突；本次未修改该 force-input 约定。
- `d_hat=0.0` 当前由环境显式传入；CSV 中的 `dhat` 列尚未接入 active geometry。
- wing-root origin、base COM 和 articulation local wrench reference 的对应关系沿用既有环境实现，尚无 Isaac Sim one-force reference test。
- induced drag 仍在 wing wrench 形成之后仅修改 force，沿用旧行为，未为该附加 force 新增 moment closure。
