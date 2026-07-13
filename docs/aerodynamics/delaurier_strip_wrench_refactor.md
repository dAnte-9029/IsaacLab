# DeLaurier Strip Wrench Refactor

日期：2026-07-13

## 范围与状态

已实现 DeLaurier attached-flow strip load 暴露与 strip-integrated wing wrench。本记录描述当前代码；未实现 corrected force 或 corrected distribution。

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

## 后续 corrected distribution 边界

未来 correction 只能以 strip-level aggregate residual（例如 `delta_dN`、`delta_dFx`）进入，不能声称能唯一辨识它属于 `dN_c`、`dN_a`、suction 或任一 drag component。本次未添加 correction、网络、训练接口或任务级调参。

## 仍存假设

- `theta_a` 的环境 FRD 注释与项目其他 FLU 文字仍有既有冲突；本次未修改该 force-input 约定。
- `d_hat=0.0` 当前由环境显式传入；CSV 中的 `dhat` 列尚未接入 active geometry。
- wing-root origin、base COM 和 articulation local wrench reference 的对应关系沿用既有环境实现，尚无 Isaac Sim one-force reference test。
- induced drag 仍在 wing wrench 形成之后仅修改 force，沿用旧行为，未为该附加 force 新增 moment closure。
