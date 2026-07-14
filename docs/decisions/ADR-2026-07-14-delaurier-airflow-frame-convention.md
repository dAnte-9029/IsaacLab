# ADR: DeLaurier airflow frame convention

日期：2026-07-14

## Context

项目 articulation/body 数据采用 FLU：`+x` forward、`+y` left、`+z` up。DeLaurier 环境入口曾直接计算 `theta_a=atan2(v_air_b.z,v_air_b.x)`，同时把同一个未转换的 `v_air_b` 注释为 FRD。该写法把 Isaac FLU 的 `+z up` 当成 DeLaurier 的 `+z down`。同一物理来流因此会因入口标签不同而得到相反 `theta_a`。

另一个直接相关的 pose 审计发现：URDF 左右翼 joint axis 都是 joint-frame `+x`，而实际 mesh 展向分别为 link `+y` 和 link `-y`。相同 URDF joint coordinate 会使两翼高度反向，不能形成关于 body `x-z` 平面的物理镜像。

## Decision

DeLaurier physics 边界使用显式 section convention：

```text
x_D: forward
y_D: right
z_D: down
theta_a = atan2(w_D, u_D)
```

Isaac body FLU polar velocity通过：

```text
v_D = diag(1, -1, -1) v_FLU
```

转换后再计算 `theta_a`。实现由 `body_air_velocity_to_delaurier_section_velocity()` 和 `compute_delaurier_axis_incidence()` 负责；环境不再包含无 frame 声明的裸 `atan2(vz,vx)`。

工程 flap scalar `q=Gamma*cos(phi)` 定义为物理镜像 coordinate。它映射到 URDF joint space 时 left 使用 `+q`、right 使用 `-q`；joint velocity 同样镜像。该映射只修正 articulation pose，不改变 DeLaurier 的 `h=-q*y`、dynamic twist 公式或左右翼相同的 local scalar load input。

## Alternatives considered

1. 全部 DeLaurier 公式改写为 FLU。拒绝：需要重新推导 normal、pitch 和原论文 section 符号，扩大物理变更面。
2. 保持环境裸 `atan2`，仅修改注释。拒绝：数值仍把 FLU `z` 当成 FRD `z`。
3. 在 `theta_a` 前临时乘 `-1`。拒绝：虽然纵向二维情况下数值等价，但会继续隐藏完整 polar-vector frame conversion。
4. 右翼 aerodynamic scalar 单独乘 `-1`。拒绝：错误位于 URDF joint coordinate 映射；DeLaurier/Wang local scalar 及 axial reflection 已有独立验证。

## Consequences

- 对 FLU 输入，旧式 `atan2(vz_FLU,u)` 改为 `atan2(-vz_FLU,u)`。例如 `[8,0,-1] m/s` 从 `-7.125 deg` 修正为 `+7.125 deg`。
- 同一物理 velocity 用 FLU 或 FRD 表示时，section velocity、`theta_a`、strip force 和 moment 一致。
- 左右翼 articulation target/velocity 使用相反 joint sign；真实 link pose 恢复镜像。该项会改变此前同号 wing-joint baseline 的多刚体 pose 与惯性响应，需要后续 mission baseline 重新评估。
- `dynamic_twist_phase_direction=1`、`dynamic_twist_phase_offset_deg=0` 不变；DeLaurier dynamic twist 核心公式不变。

## Assumptions

- `root_lin_vel_b` 和由 root quaternion 转入 body 的 wind 使用项目/Isaac base FLU convention。
- DeLaurier 的 `theta_a` 使用 vehicle velocity relative to air，而不是指向来流源的风矢量。
- forward speed 仍使用正 `u_D` 并保留历史低速 clamp；本 ADR 不定义倒飞模型。
- prescribed dynamic twist 仍是 kinematic input，不是被动气动弹性解。

## Validation requirements

- `tests/test_delaurier_airflow_frame_convention.py` 必须覆盖水平、上下分量、FLU/FRD 等价、force/moment 传播和左右对称。
- `tests/test_delaurier_isaac_phase_pose_contract.py` 必须用真实 PhysX link pose 覆盖四个关键 phase、镜像 chord/span、axial twist axis 和运动方向。
- `tests/test_delaurier_isaac_wrench_reference.py` 必须继续通过，确认 pose convention 修改没有破坏 base-COM wrench contract。

2026-07-14 实际结果：non-Isaac 指定集合 `77 passed`；两个 Isaac test 在独立临时 `XDG_CACHE_HOME` 下组合运行 `2 passed in 8.16s`。真实 pose polar/axial mirror 最大绝对误差为 `9.537e-7`（tolerance `2e-5`），articulation wrench input force/moment 最大绝对误差均为 `0`。

## Reconsideration triggers

- URDF joint axis、wing mesh local axes或 base body frame 改变；
- DeLaurier 接口改为 wind-direction vector 而不是 vehicle air-relative velocity；
- 支持 reverse flow、large sideslip section projection 或 passive aeroelastic twist；
- 实测 joint encoder convention 证明当前 engineering `q` 定义与硬件相反。
