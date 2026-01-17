# Wang et al. (JFM 2016) QSM：论文公式到代码的逐条对应

本文档把论文 **Wang et al., 2016, “A predictive quasi-steady model of aerodynamic loads on flapping wings”** 的主要公式，
与当前仓库中的 Wang2016 QSM 实现逐条对应起来，方便你对照检查“每一项力/力矩是怎么来的、最后怎么变换到 world/base_link 的”。

核心代码文件：
- `source/flapping_bot/flapping_bot/physics/qsm_wang2016.py`
- `scripts/isaac_wind_tunnel_flappingbot_v50.py`

---

## 1) 坐标系与符号（论文 §2.1）

### 1.1 论文的 co-rotating 坐标系（c-frame）
论文定义的 co-rotating 坐标系为 `(x_c, y_c, z_c)`，并约定：
- `x_c`：沿 **pitching axis**（翼的展向/铰轴方向）
- `z_c`：在翼平面内的 **chordwise**（弦向）
- `y_c`：翼面法向（normal to the wing surface）

代码对应：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:12`（模块 docstring）。

### 1.2 论文的欧拉角与角速度/角加速度（eq. (2.1)–(2.3)）
- 旋转矩阵 `R_phi, R_theta, R_eta`：eq. (2.1a–c)
  - 代码：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:266`
- `ω_c`：eq. (2.2)
- `α_c = \dot{ω}_c`：eq. (2.3)
  - 代码：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:646`（函数 `_wang2016_omega_alpha`）

> 备注：在 wind-tunnel rig 脚本里我们经常直接驱动 `ω_c/α_c`（避免把完整的 3-DOF 欧拉角链条塞进 1-DOF 台架），但核心 QSM 仍然以 `ω_c/α_c` 为入口实现论文四项加载。

---

## 2) 几何参数：需要哪些、代码里怎么表示

论文几何参数（刚性翼）核心是：
- `R`：半展长（积分上限）
- `c(x_c)`：弦长分布
- `d̂(x_c)`：**LE→pitch axis** 的无量纲偏置（用弦长归一化）
  - `d̂ = 0`：pitch axis 在前缘（LE）
  - 论文里一般要求 `0 ≤ d̂ < 0.5`（避免 pitch axis 超过弦中线）

代码表示：`WingGeometry`
- 构造入口：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:492`
- 内部离散：把 `[0, R]` 分成 `N` 个 spanwise strip，用中点法积分
  - `x_mid[i]`：第 i 个 strip 的 `x_c` 中点
  - `dx[i]`：strip 宽度
  - `c[i] = c(x_mid[i])`
  - `d_hat[i] = d̂(x_mid[i])`
  - 代码：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:510`

wind-tunnel rig（v50）脚本如何喂几何：
- 如果提供 `--wing-geom-csv`（来自你拟合的后缘/弦长分布），会在脚本里插值成 `c(x)`，并从 `x_mid` 推断 `R`，再计算 `S、c̄、AR_geom`
  - 代码：`scripts/isaac_wind_tunnel_flappingbot_v50.py:392`
- 否则（可选）从 URDF+STL 估算翼平面面积，再结合 `AR` 推回 `R、c̄`
  - 代码：`scripts/isaac_wind_tunnel_flappingbot_v50.py:260`

---

## 2.5) 关于“被动扭转”实现方式（扩展，不属于 Wang2016）

Wang2016 本身假设刚性翼（一个全翼统一的 η(t)）。本仓库为了贴近你的“外翼段薄膜被动扭转”，在
`scripts/isaac_wind_tunnel_flappingbot_v50.py` 里加了一个 **virtual twist** 扩展，仅作用在外翼段 `[--twist-x0,--twist-x1]`：

- `--twist-mode quasi_static`：每步解静力平衡 `η = η0 + sign*τ_x(η)/k`（需要少量迭代，较慢但稳）
- `--twist-mode dynamic`：积分二阶扭转振子 `Iη̈ + cη̇ + k(η-η0)=τ_x`（有相位滞后，可能更容易产生净推力）
- `--twist-mode prescribed`：最简单/最快，直接给定 `η_tip(t)`（open-loop），用于把“结构耦合”从 Wang QSM 中剥离出来做对比

对应代码入口：
- twist 更新：`scripts/isaac_wind_tunnel_flappingbot_v50.py:661`
- 二阶系统：`source/flapping_bot/flapping_bot/physics/virtual_twist.py:81`

---

## 3) 从运动学到局部速度/迎角（eq. (2.4)–(2.7)）

### 3.1 平移速度与加速度（eq. (2.4)–(2.5)）
论文对 pitching axis 上 `r=[x_c,0,0]^T` 的点：
- `v_c = ω_c × r = x_c [0, ω_zc, -ω_yc]^T`（eq. (2.4)）
- `a_c = α_c × r + ω_c × v_c`（eq. (2.5)）

代码实现位置：
- 速度 `v_c`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:141`
  - `v_y = x * ω_zc`
  - `v_z = -x * ω_yc`
  - 这里我们**允许叠加 forward-flight 速度**（见下一条）
- 加速度里只取 added-mass 需要的 `a_yc`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:223`
  - `a_yc = x_c (α_zc + ω_xc ω_yc)`（对应 eq. (2.5) 的 y 分量）

### 3.2 Forward-flight 速度叠加（论文 eq. (2.5) 下方的 note）
论文说明：前飞速度需要从 inertial frame 变到 c-frame 后，加到 `v_c` 里。

代码实现：
- QSM 入口 `compute_aero_wrench_from_omega_alpha(..., v_forward_c=...)` 会把 `v_forward_c` 以向量形式加到每个 strip 的 `v_c` 上：
  - `source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:120`
- wind-tunnel rig 中把“风洞气流速度”转换成“等效前飞速度”并变换到 c-frame：
  - `v_air_w = airspeed * flow_dir_world`
  - `v_forward_w = -v_air_w`（风洞里机体不动、空气在动）
  - 再做 `world -> link -> wang(c)`：`scripts/isaac_wind_tunnel_flappingbot_v50.py:488` 与 `scripts/isaac_wind_tunnel_flappingbot_v50.py:650`

### 3.3 迎角 α̃（eq. (2.7)）
论文：`α̃ = arccos(|v_zc| / |v_c|)`（eq. (2.7)）

代码：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:147`
- `aoa = acos(|v_z| / sqrt(v2))`
- `v2 = v_x^2 + v_y^2 + v_z^2`（包含 forward-flight 的 `v_x`）

---

## 4) 四个加载分量：逐项对应论文公式

实现入口：`compute_aero_wrench_from_omega_alpha()`  
代码：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:73`

> 重要结构：本实现最终只返回 `F_c=[0, Fy, 0]` 与 `τ_c=[τ_x, 0, τ_z]`（c-frame），对应论文“合力沿 `y_c`、力矩只考虑 `x_c` 与 `z_c`”的建模假设。

### 4.1 Translation-induced load（§2.2.1，eq. (2.11)–(2.14)）
论文：
- 合力 `F_yc^trans`：eq. (2.11)
- `d̂_cp^trans`：eq. (2.12)
- `τ_xc^trans`：eq. (2.13)（根据 `ω_yc` 的正负做 LE/TE 切换）
- `τ_zc^trans`：eq. (2.14)

代码：
- 系数与 CP：
  - `A_lift_max`：eq. (2.8) → `source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:156`
  - `C_Fy^trans`：eq. (2.10) → `source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:162`
  - `d_hat_cp_trans`：eq. (2.12) → `source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:165`
- strip 积分（用 `Σ dFy` 近似 `∫ … dxc`）：
  - `dFy = -sgn(v_y) * 0.5 * ρ * |v|^2 * C_Fy * c * dx` → 对应 eq. (2.11) 的 integrand
  - `Fy_trans = Σ dFy`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:175`
  - `τ_z_trans = Σ x * dFy`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:177`（对应 eq. (2.14)）
  - `τ_x_trans = Σ dFy * (k - d_hat) * c`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:180`（对应 eq. (2.13)）
    - `k` 是 CP 的 chordwise 位置（0..1），并在 `ω_yc>0` 时做 `k = 1 - d̂_cp`（相当于“把 TE 当作 LE”）

### 4.2 Rotation-induced load（§2.2.2，eq. (2.15)–(2.19)）
论文：
- `F_yc^rot`：eq. (2.15)
- `τ_xc^rot`：eq. (2.16)
- `τ_zc^rot`：eq. (2.17)
- `C_D^rot`：eq. (2.18)
- `d̂_cp^rot`：eq. (2.19)

代码（对 chordwise 的 `∫ z|z| dz`、`∫ |z|^3 dz` 做了解析积分，得到多项式项 `poly3/poly4`）：
- `C_D^rot`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:157`
- `Fy_rot, τx_rot, τz_rot`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:190`
- `d̂_cp^rot` 的函数实现（目前主要用于对照/工具函数）：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:332`

### 4.3 Coupling load（§2.2.3，eq. (2.21)–(2.23)）
论文给出分段形式（`ω_yc ≤ 0` 与 `ω_yc > 0`）：
- `F_yc^coup`：eq. (2.21)
- `τ_xc^coup`：eq. (2.22)
- `τ_zc^coup`：eq. (2.23)

代码：
- `base_coup = π ρ ω_x ω_y`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:204`
- 对应 eq. (2.21) 里“(3/4 − d̂) + 1/4 = (1 − d̂)”与“(d̂ − 1/4) + 1/4 = d̂”这两种分段：
  - `main = where(wy>0, d_hat-0.25, 0.75-d_hat)`
  - `Fy_coup = Σ base_coup * (main+0.25) * c^2 * x * dx`
  - `τ_z_coup = Σ base_coup * (main+0.25) * c^2 * x^2 * dx`
  - 代码：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:205`
- `τ_xc^coup` 的两项（quarter-chord + Coriolis）用 `coef_sum` 合并实现：
  - `source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:209`

### 4.4 Added-mass load（§2.2.4，eq. (2.24)–(2.25)）
论文：
- added-mass 系数矩阵 `M`：eq. (2.24)
- ` [F_yc^am, τ_xc^am]^T = -∫ M [a_yc, α_xc]^T dxc`：eq. (2.25)

代码：
- `a_yc` 与 `α_xc`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:223`
- `M` 的各项：
  - `m22, m24(=m42), m44`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:233`
- spanwise 积分实现：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:238`

### 4.5 Wagner multiplier（§2.2.5，eq. (2.26)）
论文：对 circulatory loads 乘以 `Φ(t*)`（eq. (2.26)）。

代码：
- `WingGeometry.wagner_multiplier()`：`source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:626`
- 当前限制：如果没有提供 `wagner_t_star`（即 `t*`），默认返回 1.0，不起作用（TODO）。

### 4.6 合成总载荷与输出格式
代码：
- circulatory：`Fy_trans + Fy_rot + Fy_coup`（乘 `wagner`）
- added-mass：单独相加
- 最终输出（c-frame）：
  - `F_c = [0, Fy, 0]`
  - `τ_c = [τ_x, 0, τ_z]`
  - `source/flapping_bot/flapping_bot/physics/qsm_wang2016.py:261`

---

## 5) 从 c-frame 到 link/world：脚本里的坐标变换

### 5.1 `--wang-axes`：定义“Wang 轴在 wing link 坐标里怎么取”
wind-tunnel rig 使用 `--wang-axes` 来告诉脚本：
> “`x_c/y_c/z_c` 分别等于 wing link 的哪根轴（允许符号翻转）”

构造矩阵 `A`（行向量是 Wang 基向量在 link 坐标中的表达）：
- `v_c = A * v_link`
- `v_link = A^T * v_c`（因为 A 是纯置换/翻转矩阵）

代码：
- `scripts/isaac_wind_tunnel_flappingbot_v50.py:311`

左右翼镜像：
为了处理 URDF 里左右翼 link frame 常见的镜像关系，脚本默认会对**右翼**的 `x_c` 做符号翻转（可关闭）：
- `scripts/isaac_wind_tunnel_flappingbot_v50.py:457`

### 5.2 world ↔ link ↔ c 的力/力矩变换
脚本做了三步：
1) QSM 计算得到 `F_c, τ_c`（c-frame）
2) c → link：`F_link = A^T F_c`，`τ_link = A^T τ_c`
3) link → world：用 wing link 的 world quaternion 旋转向量

代码：
- c→link：`scripts/isaac_wind_tunnel_flappingbot_v50.py:747`
- link→world：`scripts/isaac_wind_tunnel_flappingbot_v50.py:749`

---

## 6) 输出 CSV 中的各列到底是什么

wind-tunnel rig 输出：
- `F_wang_*_{L,R}` / `tau_wang_*_{L,R}`：**c-frame**（论文定义的 Wang co-rotating frame）
- `F_link_*_{L,R}` / `tau_link_*_{L,R}`：URDF 的 **wing link frame**
- `F_world_*_{L,R}` / `tau_world_*_{L,R}`：Isaac Sim 的 **world frame**

总和（仅 world）：
- `F_world_*_T`：左右翼合力（简单相加）
- `tau_world_*_T`：左右翼“纯力矩”相加（不含力臂）
- `tau_world_*_T_about_base`：关于 `base_link` 的合力矩（包含 `r×F`）
  - `τ_about_base = τ_L + τ_R + (r_L×F_L) + (r_R×F_R)`
  - 代码：`scripts/isaac_wind_tunnel_flappingbot_v50.py:755`

---

## 7) 如何从 world 总力得到 lift / drag / thrust（推荐）

关键点：Wang2016 模型在 c-frame 只输出 `F_yc`，但把它旋转到 world 后会自然产生 forward/vertical 分量。

在 wind-tunnel rig 默认设置 `--flow_dir_world -1 0 0` 下：
- 空气“朝 world -X 方向流动”，等效前飞方向是 `+X`
- `thrust ≈ F_world_x_T`（正值 = 推力，负值 = 阻力）
- `lift ≈ F_world_z_T`（正值 = 向上）
- `side ≈ F_world_y_T`（正值 = 向 world +Y）

如果你把风向设成任意方向，最稳妥的做法是用向量投影：
- `e_forward = normalize(-v_air_w)`
- `thrust = F_world_T · e_forward`
- `drag = -thrust`

---

## 8) “对称但某个分量很毛躁”的常见原因

当理论上某个总分量应当接近 0（例如对称扑动时 `F_world_y_T≈0`），数值上你会看到：
- 左右翼各自产生较大的 `±` 分量
- 相加后出现 `1e-6~1e-4` 量级的残差，并随时间抖动

这通常是“近似相消 + 浮点误差 + r×F 的差分放大”的表现，不是模型本身真的在产生侧向力。
