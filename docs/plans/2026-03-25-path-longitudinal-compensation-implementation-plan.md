# Path 纵向补偿第一阶段实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `path teacher` 增加一条 PX4 风格的转弯纵向补偿链路，在不改变 `straight baseline` 默认行为的前提下，降低转弯和盘旋时的掉高。

**Architecture:** 保持现有简化 TECS 结构不变，只扩展一个可选运行时接口：`load_factor + load_factor_correction`。`path controller` 负责根据 bank 角计算 load factor，并将补偿项送入 TECS 的总能量率需求；`straight controller` 默认继续以零补偿运行。

**Tech Stack:** Python 3.11, PyTorch, IsaacLab PX4-like controllers, pytest

---

### Task 1: 锁定 TECS 负载因子补偿合同

**Files:**
- Modify: `tests/test_tecs.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/tecs.py`

**Step 1: 写 failing test，证明更高 load factor 会提高 throttle 需求**

在 `tests/test_tecs.py` 里新增一个用例：

- 两组输入高度、空速、目标高度、目标空速完全相同
- 唯一差别是：
  - Case A: `load_factor=1.0`
  - Case B: `load_factor=1.3`
- 同时传入正的 `load_factor_correction`
- 断言：
  - `throttle_sp_B > throttle_sp_A`
  - `diag["tecs_load_factor"]` 被记录
  - `diag["tecs_load_factor_energy_bias"]` 为正

**Step 2: 跑测试，确认先红**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_tecs.py -q
```

Expected:
- 新测试失败
- 失败原因是 `PX4LikeTECS.update()` 还不接受这些新输入或未提供对应诊断量

**Step 3: 最小实现 TECS 运行时补偿接口**

在 `source/flapping_bot/flapping_bot/px4_like/tecs.py` 中：

- 给 `PX4LikeTECS.update()` 增加可选参数：
  - `load_factor: float | Tensor | None = None`
  - `load_factor_correction: float | Tensor | None = None`
- 采用 batch-safe tensor 形式处理
- 默认值保持中性：
  - `load_factor=1.0`
  - `load_factor_correction=0.0`
- 在 `ste_rate_sp` 上追加：

```python
ste_rate_load_factor_bias = load_factor_correction * (load_factor - 1.0)
ste_rate_sp = ste_rate_sp + ste_rate_load_factor_bias
```

- 将诊断量写入 `diag`：
  - `tecs_load_factor`
  - `tecs_load_factor_correction`
  - `tecs_load_factor_energy_bias`

**Step 4: 跑测试，确认转绿**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_tecs.py -q
```

Expected:
- 全绿

**Step 5: Commit**

```bash
git add tests/test_tecs.py source/flapping_bot/flapping_bot/px4_like/tecs.py
git commit -m "feat: add TECS load-factor compensation hook"
```

### Task 2: 锁定 path controller 的 bank-aware 补偿合同

**Files:**
- Modify: `tests/test_px4_path_tracking_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py`

**Step 1: 写 failing test，证明更大转弯 bank 需求会提高 TECS throttle**

在 `tests/test_px4_path_tracking_controller.py` 里新增一个用例：

- 构造两个 path query，位置、高度、空速相同
- 唯一差别是曲率：
  - Case A: `curvature_m_inv = 0.0`
  - Case B: `curvature_m_inv > 0.0`
- controller config 开启补偿并设置非零 `tecs_roll_throttle_compensation`
- 断言：
  - Case B 的 `diag["tecs_load_factor"] > 1.0`
  - Case B 的 `diag["tecs_throttle_sp"] >= Case A`
  - Case B 的 `diag["tecs_load_factor_energy_bias"] >= 0.0`

**Step 2: 跑测试，确认先红**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_px4_path_tracking_controller.py -q
```

Expected:
- 新测试失败
- path controller 还没有把 bank/load factor 传入 TECS

**Step 3: 给 path controller 增加 path-only 补偿参数**

在 `source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py` 中：

- 将 `PX4LikePathTrackingControllerCfg` 改成 dataclass
- 增加字段：
  - `use_tecs_load_factor_compensation: bool = True`
  - `tecs_roll_throttle_compensation: float = 0.0`
  - `tecs_load_factor_clamp_max: float = 2.0`
  - `tecs_load_factor_use_roll_sp: bool = True`

默认设计要求：

- `path controller` 默认可开启此能力
- 若补偿系数为 `0.0`，行为仍保持中性

**Step 4: 在 path controller 中计算 load factor 并传给 TECS**

在 `compute_actions_from_query()` 中：

- 在算出 `roll_sp` 之后，根据配置选择：
  - 优先 `roll_sp`
  - 必要时可切到实测 `roll`
- 计算：

```python
bank_for_load = roll_sp if self.cfg.tecs_load_factor_use_roll_sp else roll
load_factor = 1.0 / torch.clamp(torch.cos(bank_for_load), min=1.0e-3)
load_factor = torch.clamp(load_factor, min=1.0, max=self.cfg.tecs_load_factor_clamp_max)
```

- 调用 `self._tecs.update(...)` 时传入：
  - `load_factor=load_factor`
  - `load_factor_correction=float(self.cfg.tecs_roll_throttle_compensation)`，仅当开关启用

**Step 5: 跑测试，确认转绿**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_px4_path_tracking_controller.py tests/test_tecs.py -q
```

Expected:
- 全绿

**Step 6: Commit**

```bash
git add tests/test_px4_path_tracking_controller.py tests/test_tecs.py \
        source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py \
        source/flapping_bot/flapping_bot/px4_like/tecs.py
git commit -m "feat: add bank-aware path TECS compensation"
```

### Task 3: 跑 focused verification，确认默认 straight 不变

**Files:**
- Modify: none

**Step 1: 跑相关纯 Python 回归测试**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_tecs.py \
  tests/test_px4_path_tracking_controller.py \
  tests/test_px4_like_guidance.py \
  tests/test_rl_teacher_guidance.py \
  tests/test_teacher_schedule.py \
  tests/test_fly_straight_line_contract.py -q
```

Expected:
- 全绿

**Step 2: 跑一条短 straight，确认默认 baseline 没被悄悄改行为**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0 \
  --num_envs 1 --steps 120 \
  --out_dir logs/flapping_px4/straight_line_bank_comp_short \
  --headless
```

Expected:
- 正常结束
- `summary.json` 中 baseline 默认参数不出现新的非零 bank compensation 痕迹

**Step 3: 跑最小 path 验证，先看 `level_turn`**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_turn --num_envs 1 --steps 2400 --auto_extend_steps \
  --out_dir logs/flapping_px4/path_tracking_bank_comp_check \
  --headless
```

重点检查：

- `tecs_throttle_sp` 是否比旧结果更少顶在 `1.0`
- `tecs_pitch_sp_deg` 是否比旧结果更少顶在 `-25`
- `final_progress_ratio` 是否不低于旧结果
- `mean_abs_height_error_post_warmup_m` 是否下降

**Step 4: 仅当 `level_turn` 改善后，再跑 `level_loiter` 和 `multi_segment`**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_loiter --num_envs 1 --steps 2400 --auto_extend_steps \
  --out_dir logs/flapping_px4/path_tracking_bank_comp_check \
  --headless
```

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase multi_segment --num_envs 1 --steps 2400 --auto_extend_steps \
  --out_dir logs/flapping_px4/path_tracking_bank_comp_check \
  --headless
```

Expected:
- 至少一个指标方向改善：
  - `final_progress_ratio`
  - `steps_completed`
  - `mean_abs_height_error_post_warmup_m`
  - `throttle/pitch saturation ratio`

### Task 4: 决定是否进入第二阶段

**Files:**
- Modify: none in this task

**Step 1: 用 rollout 结果做分流**

如果出现下面结果：

- `tecs_throttle_sp` 顶死比例明显下降
- 但 `action_elevon_pitch` 仍显著顶死

结论：
- 第二阶段优先改 inner pitch loop

如果出现下面结果：

- `tecs_throttle_sp` 仍长期顶死
- 且 `load_factor` 主要出现在较大 bank 路段

结论：
- 第二阶段优先加 `bank-aware tas_min / speed floor`

**Step 2: 记录结果，不在本计划内继续扩 scope**

本计划只完成第一阶段，不在同一提交里继续叠加第二阶段改动。
