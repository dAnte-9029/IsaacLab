# Domain Randomization From System Identification Results

本文记录从 `/home/zn/flap-system-identification` 和
`/home/zn/paper/AeroConf_effective_aero` 中提炼出的 domain randomization 建议。
定位是给当前 IsaacLab flapping RL 环境提供实现优先级，而不是证明某个气动原因。

## 证据来源

主要参考：

- `/home/zn/flap-system-identification/docs/results/2026-05-26-deployable-wrench-correction-v2.md`
- `/home/zn/flap-system-identification/docs/results/2026-05-26-component-residual-attribution-diagnostic.md`
- `/home/zn/flap-system-identification/docs/results/2026-05-27-phase-structured-wrench-correction.md`
- `/home/zn/flap-system-identification/docs/results/2026-05-25-residual-guided-delaurier-greybox-force-arm.md`
- `/home/zn/paper/AeroConf_effective_aero/sections/03_method.tex`
- `/home/zn/paper/AeroConf_effective_aero/sections/05_results.tex`
- `/home/zn/paper/AeroConf_effective_aero/research_notes/force_moment_arm_diagnostic_note.md`

核心读法：

- 这些结果是 real-flight log effective-wrench prediction，不是 closed-loop simulator validation。
- DeLaurier / grey-box prior 的误差不是纯随机噪声，而是有 wingbeat phase、flapping frequency、
  AoA / dynamic pressure、lateral-directional variables 和 body-rate 结构。
- 当前最适合补的随机化不是 contact friction，而是 aerodynamic, actuator, sensor, phase, and moment-structure randomization。

## 实测质量属性补充

2026-06-02 的实机测量表 `/home/zn/质量 重心位置 惯量.xlsx` 给出了整机质量、CG 和惯量矩阵。
用户确认的测量坐标约定是：

- 机体测量坐标系：原点在 URDF 原点，坐标轴为 FRD。
- 机翼测量坐标系：机翼水平放置，原点在右翼 link 点，坐标轴为 FRD。

从表格原始称重数据重算得到：

| Quantity | Value | Notes |
| --- | ---: | --- |
| body mass | `0.78261 kg` | 四点称重 10 次平均，std 约 `0.00048 kg` |
| body CG x | `-0.13103 m` | 相对机体测量/URDF 原点，std 约 `1.49 mm` |
| body CG y | `0.00625 m` | 相对机体测量/URDF 原点，std 约 `0.13 mm` |
| right wing mass | `0.06077 kg` | 三点称重 3 次平均，std 约 `0.00006 kg` |
| right wing local CG x | `-0.06040 m` | 相对右翼 link 点 FRD，std 约 `0.81 mm` |
| right wing local CG y | `0.29394 m` | 相对右翼 link 点 FRD，std 约 `0.96 mm` |
| whole-aircraft mass | `0.90415 kg` | body + mirrored left/right wings |
| whole-aircraft CG in URDF FRD | `[-0.12154, 0.00541, -0.01298] m` | 与 sysid metadata source 字段一致 |
| whole-aircraft inertia diag | `[0.02329, 0.02573, 0.04270] kg*m^2` | 关于 neutral-wing whole-aircraft CG，当前使用对角近似 |

注意：`/home/zn/flap-system-identification/artifacts/20260602_measured_massprops_metadata_snapshot/aircraft_metadata.yaml`
当前同时记录了 `cg_b_m = [-0.12154, 0.00541, -0.04298] m`，并标注为 relative to `imu_origin`。
其 `source` 字段说明 whole-aircraft CG in U_FRD 是 `[-0.12154, 0.00541, -0.01298] m`，随后减去了近似
IMU offset `[0, 0, 0.030] m`。因此在 IsaacLab 中不要直接混用这两个值：

- 如果 Isaac / URDF 的 `base_link` 原点就是测量原点，应使用 `[-0.12154, 0.00541, -0.01298] m`。
- 如果某个日志标签或状态估计模块以 IMU 为 body reference，则使用 `[-0.12154, 0.00541, -0.04298] m`。

当前 `straight_flight_env.py` 只支持 `base_body_com_override_x_m`，并且默认值是 `-0.10 m`；整机质量可以通过
`total_mass_kg_override` 整体缩放，但还没有直接设置三维 COM 和 measured inertia diagonal 的配置入口。
因此 measured mass properties 还没有被完整用进 RL 仿真。

## 最高优先级随机化

### 1. DeLaurier / Aerodynamic Parameter Randomization

系统辨识中的物理校准参数可以直接转成 domain randomization 参数：

- `wing_normal_force_scale`
- `wing_chordwise_force_scale`
- `delaurier_theta_w_deg`
- `twist_eta_max_deg`
- `delaurier_induced_drag_efficiency`
- `fuselage_drag_cda`
- `tail_lift_scale`
- `phase_delay_s`

当前 repo 已有相关配置入口或校准边界：

- `source/flapping_bot/flapping_bot/analysis/delaurier_gain_calibration.py`
- `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`

建议初始范围：

| Parameter | Initial DR range | Notes |
| --- | ---: | --- |
| `wing_normal_force_scale` | `0.8 - 1.2` | 后续可按校准边界扩大 |
| `wing_chordwise_force_scale` | `0.7 - 1.3` | 影响 thrust / drag balance |
| `tail_lift_scale` | `0.8 - 1.2` | 先做整体 tail effectiveness |
| `delaurier_theta_w_deg` | `-3 deg - +3 deg` | effective AoA / incidence bias |
| `twist_eta_max_deg` | nominal `+- 3 deg` | 扭转幅值不确定性 |
| `delaurier_induced_drag_efficiency` | `0.6 - 1.1` | 有限翼诱导阻力项 |
| `fuselage_drag_cda` | `0.5x - 1.5x nominal` | 机身阻力不确定性 |
| `phase_delay_s` | `-0.02 - +0.02 s` | 相位/执行器/日志对齐误差 |

不要一开始使用全边界训练。先用窄范围训练稳定策略，再通过 curriculum 扩大。

### 2. Phase-Structured Force / Moment Randomization

论文结果显示主力通道有强 wingbeat-synchronous residual：

- `fx_b` phase peak-to-peak residual 约 `15 N`
- `fz_b` phase peak-to-peak residual 约 `28.8 N`
- `fx_b` 和 `fz_b` 的 residual energy 主要集中在 wingbeat fundamental / second harmonic

建议新增相位调制项，而不是只做常数 force scale：

```text
force_scale(phi)
  = 1
  + a1 sin(phi) + b1 cos(phi)
  + a2 sin(2 phi) + b2 cos(2 phi)
```

建议初始实现：

- per-env reset 采样 `a1, b1, a2, b2`
- 分别作用到 wing force 的 normal / chordwise components，或者直接作用到 body-frame `fx_b/fz_b`
- 初始幅值用小范围，例如 `+-5%` 到 `+-10%`
- 后续根据 replay / rollout 稳定性扩大

同时补：

- phase zero offset randomization
- left/right wing phase asymmetry
- commanded-vs-realized phase delay
- phase jitter

### 3. AoA / Frequency / Dynamic-Pressure Condition Randomization

结果显示 residual 随 AoA、flapping frequency 和 dynamic pressure 改变。高频 bin 下
`fx_b/fz_b` prior residual 更大。因此需要加入工况相关随机化：

- effective AoA bias
- stall angle / separation threshold perturbation
- frequency-dependent lift / thrust scale
- twist amplitude uncertainty
- air-density / dynamic-pressure scale
- flapping-frequency actuator response error

建议实现形式：

```text
force_scale = base_scale
            * (1 + k_alpha * alpha)
            * (1 + k_freq * (f - f0))
            * (1 + k_q * normalized_q_dyn)
```

其中 `k_alpha/k_freq/k_q` 先用小范围 per-env 采样。这样比单纯扩大全局 force scale 更贴近系统辨识结果。

### 4. Lateral-Directional / Tail Randomization

`fy_b` 是当前最弱 force channel。残差 attribution 显示它和以下变量有弱但稳定关系：

- `body_rate_r`
- `q_dyn * body_rate_r`
- `servo_rudder`
- `q_dyn * servo_rudder`
- `elevon_diff_proxy`

这不证明误差由 rudder 单独导致，只说明 lateral-directional model mismatch 仍存在。

建议补：

- rudder effectiveness scale
- vertical-tail side-force scale
- elevon differential effectiveness scale
- elevon sum / diff mixing uncertainty
- rudder bias / neutral offset
- beta proxy / sideslip estimate noise
- lateral wind / crosswind estimate error

初始范围建议：

| Parameter | Initial DR range |
| --- | ---: |
| rudder effectiveness | `0.7 - 1.3` |
| elevon differential effectiveness | `0.7 - 1.3` |
| tail q scale | `0.8 - 1.2` |
| rudder neutral bias | `+-2 deg` |
| elevon neutral bias | `+-2 deg` |
| beta proxy noise | `+-2 deg` equivalent |

### 5. Rate-Dependent Moment / Damping Randomization

`mx_b` residual 最清楚地和 roll rate `p` / `q_dyn * p` 相关。`my_b/mz_b` 也有 body-rate 结构，
但证据弱于 `mx_b`。

建议补：

- roll damping coefficient randomization
- pitch damping coefficient randomization
- yaw damping coefficient randomization
- `q_dyn * p/q/r` moment terms
- phase-dependent roll/yaw moment perturbation

建议形式：

```text
tau_rate_b =
  [
    -C_p * q_dyn * p,
    -C_q * q_dyn * q,
    -C_r * q_dyn * r,
  ]
```

其中 `C_p/C_q/C_r` per-env reset 随机化。`C_p` 优先级最高。

## 第二优先级随机化

### 6. Dynamic Moment Arm / Free Moment Randomization

force-moment arm diagnostic 的结论：

- per-sample minimum-norm arm 可以解释大部分 moment energy，但它使用了真实 moment，只是 diagnostic upper bound。
- fixed global arm 对 held-out logs 解释力很弱。
- 更合理的结构是 `M = r_hat(x) x F_hat + tau_free_hat(x)`。

因此不建议只随机一个固定 aerodynamic center。建议补：

- wing/tail application point perturbation，初始 `+-1-3 mm`
- phase-dependent center-of-pressure shift
- airspeed / AoA dependent center-of-pressure shift
- small free-moment residual, especially `mx_b` and `mz_b`

实现边界：

- dynamic arm 先作为 bounded perturbation，不要无限制 neural correction。
- free moment 先用小幅 additive torque，配合 rate damping 随机化。

### 7. Sensor / Estimator Randomization

系统辨识模型依赖的关键可部署特征包括：

- wingbeat phase
- flapping frequency
- airspeed
- dynamic pressure
- AoA / beta proxy
- body rates
- servo commands

部署时这些都不是 perfect truth，因此 RL 训练需要估计误差随机化：

- airspeed scale / bias / noise / dropout
- phase encoder zero offset
- phase delay / jitter
- gyro bias / noise
- attitude estimator delay
- velocity / altitude delay
- wind estimate error
- beta / sideslip proxy error

当前 repo 已有 `truth / estimated` state source 和 synthetic IMU 相关基础，可以沿这个方向扩展。

### 8. Mass / Inertia / COM Randomization

质量、惯量、COM 仍然需要补。它们不是当前 wrench residual 的主要解释，但这组实测数据给出了比旧
`mass_props.json` 更可信的 nominal model，应先作为仿真名义值，再围绕它做小范围随机化。
moment 结果显示大误差更多来自 distributed aerodynamic load / dynamic center of pressure / free moment，
不是简单调 COM 就能解决。

建议用途：

- sim-to-real robustness
- battery / payload / assembly variation
- inertial response uncertainty

建议初始范围：

| Parameter | Initial DR range |
| --- | ---: |
| total mass | nominal `0.90415 kg`, first `+-1-2%`, later `+-3-5%` |
| inertia diagonal | nominal `[0.02329, 0.02573, 0.04270] kg*m^2`, first `+-5%`, later `+-10%` |
| base/whole-aircraft COM x | nominal `-0.12154 m`, first `+-5 mm`, later `+-10 mm` |
| base/whole-aircraft COM y | nominal `0.00541 m`, first `+-3-5 mm` |
| base/whole-aircraft COM z | nominal `-0.01298 m` relative URDF origin, first `+-5 mm`, later `+-10 mm` |
| off-diagonal inertia | keep zero initially; later sample bounded products if needed |

实现注意：

- 对当前仿真结构，若继续使用 prescribed / kinematic wing motion 并把 appendage masses 降到很小，最稳妥的做法是把
  whole-aircraft effective mass, COM, and inertia 放到 `base_link` 上，而不是把机翼真实惯量完整分布到运动 link。
- 如果之后要把左右翼真实 mass properties 分布到 link，应使用右翼局部 FRD 测量值，并对左翼做镜像；同时需要
  parallel-axis transform，不能只复制质量和惯量对角项。
- current sysid metadata 中的 IMU-origin CG 适合日志标签/状态估计链路；Isaac PhysX `set_coms` 应使用 body/link
  local frame 下的 COM，优先使用 URDF-origin CG，除非模型的 base link 原点明确等于 IMU origin。
| appendage mass scale | 小范围，且保持现有 appendage mass decoupling 设计 |

当前 repo 的 mass / inertia / COM 主要还是 fixed override，需要 per-reset randomization 和 `_mass_total`
同步更新。

## 不建议优先做

- Contact friction randomization：对空中 path tracking 贡献很小，除非研究起降或地面碰撞。
- 全参数大范围随机化：容易让 residual PPO 训练变保守、不收敛。
- 直接把 supervised neural residual 当作闭环真值模型：当前结果是 log-based wrench prediction，
  不是 long-horizon rollout validation。
- 把 `fy_b` residual 归因到单一 rudder 参数：现有证据只支持弱相关和 lateral-directional mismatch。

## 建议实现顺序

### Step 1: `aero_param_dr`

实现 per-env / per-reset aerodynamic parameter randomization：

- wing normal / chord force scale
- tail lift scale
- theta / AoA bias
- twist amplitude
- induced / fuselage drag
- phase delay

这是最直接由系统辨识结果支持的一组。

### Step 2: `phase_harmonic_dr`

实现 phase-dependent force / moment modulation：

- first harmonic
- second harmonic
- phase offset
- left/right asymmetry

这对应当前最明显的 force residual 结构。

### Step 3: `actuator_sensor_dr`

实现部署相关随机化：

- servo LPF / rate / delay
- flap frequency response delay
- phase encoder offset / jitter
- airspeed noise / dropout
- IMU / estimator delay

### Step 4: `moment_structure_dr`

实现 rate-dependent moment 和 dynamic arm / free moment：

- `q_dyn * p/q/r` damping terms
- application point perturbation
- phase-dependent center-of-pressure shift
- small free torque residual

### Step 5: `mass_inertia_com_dr`

补通用 rigid-body robustness：

- mass
- inertia
- base COM
- appendage mass/inertia consistency

这一步重要，但不应取代气动/相位/执行器随机化。

## 论文和实验表述边界

可以说：

> System-identification results motivate aerodynamic, phase, rate, and sensor/actuator randomization because the real-flight residual is structured by wingbeat phase, flight condition, and body-rate variables.

不要说：

> The residual proves that a specific rudder coefficient or center of pressure is wrong.

更稳的中文表述：

> 这些结果说明当前模拟器 prior 的误差具有可重复结构，因此 domain randomization 应优先覆盖相位同步气动误差、频率/AoA/动压相关误差、横航向尾翼不确定性、角速度阻尼力矩以及传感器/执行器延迟。质量、惯量和 COM 随机化仍应补充，但它们更像通用鲁棒性项，而不是当前 effective-wrench residual 的主解释。
