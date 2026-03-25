# Path 纵向补偿设计

**背景**

新尾翼模型切换后，`straight baseline` 已经明显改善，但 `path teacher` 在 `level_turn / level_loiter / multi_segment` 里仍然稳定掉高。现有日志表明，问题首先出在纵向能量管理，而不是横向导航：

- `tecs_pitch_sp_deg` 在大量时间里顶到 `-25°`
- `tecs_throttle_sp` 在大量时间里顶到 `1.0`
- `action_elevon_pitch` 也会顶死，但饱和比例低于 TECS 外环

这说明当前控制器不是“完全不会飞”，而是纵向外环对转弯附加负载的认知太弱，等高度已经开始掉了，才一起把 pitch 和 throttle 顶满。

同时，参考 `/home/zn/PX4-Autopilot` 可见，PX4 固定翼并不是只靠调 TECS 增益解决转弯掉高问题。它在纵向链路里显式引入了：

- `load_factor = 1 / cos(bank)`
- `roll-to-throttle compensation`

并把这部分附加需求直接加到 TECS 的总能量率需求中。

**目标**

在不破坏当前 `straight baseline` 默认行为的前提下，给 `path teacher` 增加一条 PX4 风格的“转弯纵向补偿”链路，让控制器在进入较大 bank 角的路径段时更早补能量，降低 `level_turn / level_loiter / multi_segment` 的掉高和早终止概率。

第一阶段只做最小有效改动：

- 不整体重写 TECS
- 不搬运 PX4 全套 performance model
- 不先动 inner pitch loop

**现状诊断**

当前实现的问题不在横向导航。横向部分已经采用了类似 PX4 `DirectionalGuidance + AirspeedDirectionController` 的结构，路径几何跟踪本身并不差。

主要缺口在于：

1. `path controller` 没有把“当前正在转弯”这一事实反馈给 TECS。
2. `TECS` 只根据高度误差和空速误差做反应式调节，没有额外的 bank 负载补偿。
3. 结果是转弯初期先掉能量，之后外环再追，导致 pitch/throttle 常年顶死。

**备选方案**

**方案 A：继续只调共享 TECS 增益和限位**

- 做法：继续增大 `max_pitch_up_deg`、`tecs_max_climb_rate_mps`、`capture_extra_climb_rate_mps`，必要时加强 inner pitch loop。
- 优点：改动最少。
- 缺点：`straight` 和 `path` 共享一套默认值，容易把现在已经还行的直飞再搞坏；而且这仍然是“掉高以后再追”。

**方案 B：在 path controller 中引入 PX4 风格的转弯纵向补偿**

- 做法：计算 bank 对应的 load factor，并把它作为 TECS 额外总能量需求的一部分。
- 优点：直接针对转弯掉高；可以只作用于 `path teacher`；与 PX4 实机成功经验一致。
- 缺点：需要扩展 TECS 接口并新增少量 path 专用参数。

**方案 C：一次性把 PX4 performance model / TECS 参考模型整套迁入**

- 做法：继续补齐 bank-aware 最小空速、完整 reference model、更多 performance constraints。
- 优点：理论上最完整。
- 缺点：改动面太大，当前没有必要；难以区分到底是哪一层起效。

**推荐方案**

选择 **方案 B**，并拆成两个阶段：

1. 第一阶段先做 `load_factor + roll-to-throttle compensation`
2. 第二阶段如有必要，再加 `bank-aware tas_min / speed floor`

这样可以先验证最关键的 PX4 设计是否足以解决当前问题。

**第一阶段设计**

第一阶段只增加一条非常明确的数据流：

`path geometry -> roll_sp / roll -> load_factor -> TECS total-energy bias -> throttle / pitch_sp`

具体做法：

1. 在 `PX4LikeTECS.update()` 中增加可选运行时输入：
   - `load_factor`
   - `load_factor_correction`

2. 仿照 PX4，把额外能量率补偿写成：
   - `ste_rate_sp += load_factor_correction * (load_factor - 1.0)`

3. 在 `PX4LikePathTrackingController` 中计算 bank 对应的负载因子：
   - 初版用 `load_factor = 1 / cos(bank)`
   - `bank` 优先取 `roll_sp`
   - 这样补偿是前馈的，而不是等机体已经掉高后再追

4. 新增少量 path 专用参数：
   - `use_tecs_load_factor_compensation`
   - `tecs_roll_throttle_compensation`
   - `tecs_load_factor_clamp_max`

5. 默认只让 `path controller` 开启这条补偿链
   - `straight controller` 保持默认零补偿
   - 这样不改变现有直飞默认行为

**为什么先用 `roll_sp`**

初版优先用 `roll_sp`，原因是它能在转弯需求出现时更早开始补能量，符合 PX4 这类“转弯诱导阻力前馈”的目的。

如果后续发现：

- `roll_sp` 前馈过猛
- 或者实际滚转建立明显跟不上

再考虑改为：

- `max(abs(roll), abs(roll_sp))`
- 或 `roll_sp` 的低通版本

但第一阶段不把这一层做复杂。

**为什么暂时不先改 inner pitch loop**

当前日志显示，首先顶死的是外环：

- `tecs_pitch_sp`
- `tecs_throttle_sp`

而不是舵面内环先顶死。

所以这一步如果先加大 `pitch_kp / inner_pitch_ki / inner_elevon_pitch_rate_limit_per_s`，更大的风险是把外环能量不足伪装成“舵面不够强”，然后引入新的振荡。

结论是：

- 第一阶段先补能量管理认知
- 第二阶段才看 inner loop 是否仍偏弱

**与 PX4 原版的对应关系**

第一阶段借鉴以下 PX4 设计，而不是整套照搬：

- `FwLateralLongitudinalControl` 每周期计算 `1 / cos(bank)` 并传给 TECS
- `FW_T_RLL2THR` 用于在转弯时补油门
- TECS 将这部分补偿加入总能量率需求

我们保留现有简化版 TECS 架构，只借鉴这条最关键的纵向耦合路径。

**非目标**

第一阶段明确不做：

- 全量迁移 PX4 performance model
- 按 load factor 动态提升最小安全空速
- 共享修改 `straight controller` 默认增益
- 调整尾翼几何或尾翼气动系数
- 调整 RL 策略

**验证标准**

第一阶段成功的标志不是“所有路径都一次性完美飞过”，而是：

1. `level_turn / level_loiter / multi_segment` 中：
   - `tecs_throttle_sp=1.0` 的占比下降
   - `tecs_pitch_sp=-25°` 的占比下降
   - 高度误差增长变慢

2. `straight baseline` 默认行为不变

3. 代码层面新增回归测试：
   - 更高 load factor 会提高 TECS 的 throttle 需求
   - path controller 在相同高度/空速条件下，较大 bank 需求会触发更高的 TECS throttle 需求
