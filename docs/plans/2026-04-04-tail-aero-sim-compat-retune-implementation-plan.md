# Tail Aero 仿真兼容校正实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在保留当前最新尾翼几何的前提下，为尾翼气动模型增加少量“整机仿真兼容”校正参数，缓解 `turn/loiter` 中的持续掉高问题，并避免再次把整机闭环配平打坏。

**Architecture:** 不回退五表面尾翼几何，也不先堆更多 TECS 外环补偿。只在 `tail_aero.py` 与 `straight_flight_env.py` 中加入一层低维等效气动校正：固定平尾配平偏置、固定面效能、活动面效能、活动面独立失速限幅、水平尾翼动压缩放。所有新增项必须默认可关闭、可诊断、可单独 sweep，并通过 focused rollout 逐步验证。

**Tech Stack:** Python 3.11, PyTorch, IsaacLab direct env configs, pytest, rollout CSV/summary diagnostics

---

### Task 1: 先锁定“兼容校正层”的配置合同

**Files:**
- Modify: `tests/test_tail_aero.py`
- Modify: `tests/test_straight_flight_env_reset_contract.py`
- Modify: `source/flapping_bot/flapping_bot/physics/tail_aero.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`

**Step 1: 写 failing tests，锁定新增参数存在且默认中性**

在 `tests/test_tail_aero.py` 里新增用例，断言 `TailAeroCfg` 暴露这些字段：

- `horizontal_tail_incidence_bias_deg`
- `fixed_horizontal_effectiveness`
- `elevon_effectiveness`
- `elevon_alpha_limit_deg`
- `horizontal_tail_q_scale`

默认值要求：

- `horizontal_tail_incidence_bias_deg == 0.0`
- `fixed_horizontal_effectiveness == 1.0`
- `elevon_effectiveness == 1.0`
- `elevon_alpha_limit_deg` 与当前活动面限幅一致
- `horizontal_tail_q_scale == 1.0`

在 `tests/test_straight_flight_env_reset_contract.py` 或已有 env contract 测试里新增用例，断言：

- `FlappingBotStraightFlight...EnvCfg` 暴露同名配置透传入口
- 默认值仍为中性，不应悄悄改变当前环境行为

**Step 2: 跑测试，确认先红**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/test_tail_aero.py \
  tests/test_straight_flight_env_reset_contract.py -q
```

Expected:
- 新测试失败
- 原因是配置字段和 env 透传接口尚不存在

**Step 3: 最小实现配置结构**

在 `source/flapping_bot/flapping_bot/physics/tail_aero.py` 中：

- 给 `TailAeroCfg` 增加上述 5 个字段
- 明确注释：
  - 几何不变
  - 这些量是整机兼容校正，不是几何参数

在 `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py` 中：

- 暴露同名 env cfg 字段
- 默认全部保持中性
- 保证旧任务不因默认值改变而发生额外行为变化

**Step 4: 跑测试，确认转绿**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/test_tail_aero.py \
  tests/test_straight_flight_env_reset_contract.py -q
```

Expected:
- 全绿

### Task 2: 锁定“固定面配平 + 活动面控制权”这两条校正路径

**Files:**
- Modify: `tests/test_tail_aero.py`
- Modify: `source/flapping_bot/flapping_bot/physics/tail_aero.py`

**Step 1: 写 failing tests，分别验证 3 类行为**

新增 3 组测试：

1. `horizontal_tail_incidence_bias_deg` 改变时：
- 零舵偏也会改变总俯仰力矩符号或大小

2. `elevon_effectiveness` 提高时：
- 同一对称 elevon 偏角下，总俯仰力矩绝对值变大

3. `elevon_alpha_limit_deg` 提高时：
- 大舵偏（如 `-30 deg` 与 `-41 deg`）不再过早饱和到几乎相同的俯仰矩

**Step 2: 跑测试，确认先红**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/test_tail_aero.py -q
```

Expected:
- 新测试失败
- 当前模型还没有这些校正逻辑

**Step 3: 最小实现 3 条校正链**

在 `source/flapping_bot/flapping_bot/physics/tail_aero.py` 中：

- 对固定水平尾翼：
  - 在有效攻角中叠加 `horizontal_tail_incidence_bias_deg`
  - 仅对固定水平尾翼生效

- 对活动 elevon：
  - 用 `elevon_effectiveness` 缩放活动面 `cl_alpha`
  - 用独立的 `elevon_alpha_limit_deg` 代替共享尾翼限幅

- 对固定水平尾翼和 elevon：
  - 在动态压 `q` 上乘 `horizontal_tail_q_scale`
  - 只对水平尾翼家族生效，不影响垂尾/方向舵

实现要求：

- 不修改已有几何中心、面积、力臂、铰链位置
- 不把这些校正硬编码到每个 surface geometry 里
- 优先在计算路径中统一处理，避免几何和效能耦死

**Step 4: 跑测试，确认转绿**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/test_tail_aero.py -q
```

Expected:
- 全绿

### Task 3: 把校正参数真正接到环境和任务配置里

**Files:**
- Modify: `tests/test_path_tracking_env_contract.py`
- Modify: `tests/test_fly_path_mission.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py`

**Step 1: 写 failing tests，锁定透传合同**

新增测试，确保：

- `straight env` 能把 5 个 tail 校正参数传入 `TailAeroCfg`
- `path env` 同样能透传
- `fly_path_mission.py` 支持 CLI 覆盖这些参数
- summary.json 里会记录本次 rollout 用到的 tail 校正参数

**Step 2: 跑测试，确认先红**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/test_path_tracking_env_contract.py \
  tests/test_fly_path_mission.py -q
```

Expected:
- 新测试失败

**Step 3: 最小实现透传**

在 env / rollout 侧实现：

- `straight_flight_env.py`：从 env cfg 构造 `TailAeroCfg`
- `path_tracking_env.py`：teacher/path task 共享这层配置
- `fly_path_mission.py`：加入 CLI 入口与 summary 记录

约束：

- 默认值必须全部中性
- 不在这一步更改 path teacher 默认控制器参数

**Step 4: 跑测试，确认转绿**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/test_path_tracking_env_contract.py \
  tests/test_fly_path_mission.py -q
```

Expected:
- 全绿

### Task 4: 先做最小静态/半静态诊断，不直接上大 sweep

**Files:**
- Modify: `scripts/flapping_px4/fly_path_mission.py`
- Modify: `tests/test_fly_path_mission.py`

**Step 1: 写 failing tests，锁定新增诊断输出**

新增测试，要求 rollout CSV 至少能写出这些量：

- `tecs_pitch_sp_deg`
- `action_elevon_pitch`
- `pitch_deg`
- `tecs_tas`
- `tecs_tas_sp`

若已有则无需新增字段；若缺失则补齐。

同时新增与 tail 校正有关的 summary 字段：

- `tail_horizontal_tail_incidence_bias_deg`
- `tail_fixed_horizontal_effectiveness`
- `tail_elevon_effectiveness`
- `tail_elevon_alpha_limit_deg`
- `tail_horizontal_tail_q_scale`

**Step 2: 跑测试，确认先红（如需要）**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/test_fly_path_mission.py -q
```

Expected:
- 若字段未暴露则失败
- 若当前已足够则此任务只补 summary 合同

**Step 3: 最小实现**

- 只补诊断，不改控制器
- 保证后续 sweep 可以清楚区分“空速不够”与“俯仰力矩不足”

**Step 4: 跑测试，确认转绿**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest tests/test_fly_path_mission.py -q
```

Expected:
- 全绿

### Task 5: 分三阶段做小范围参数验证，避免一次改炸

**Files:**
- Modify: none

**Step 1: 先做配平阶段，只扫固定面 bias**

Run（示意）:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight --num_envs 1 --steps 1600 --auto_extend_steps \
  --device cuda:1 --headless \
  --tail_horizontal_tail_incidence_bias_deg <bias>
```

只扫很小范围，例如：

- `-4`
- `-2`
- `0`
- `+2`

验收标准：

- straight 中不再需要长期接近满负 `action_elevon_pitch`
- 高度误差不恶化

如果 straight 就明显变差，立即停止，不进入下一阶段。

**Step 2: 再做控制权阶段，只扫活动面效能与活动面限幅**

Run（示意）:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_turn --num_envs 1 --steps 2400 --auto_extend_steps \
  --device cuda:1 --headless \
  --tail_elevon_effectiveness <k> \
  --tail_elevon_alpha_limit_deg <deg>
```

建议先小范围：

- `elevon_effectiveness`: `1.0`, `1.15`, `1.30`
- `elevon_alpha_limit_deg`: `25`, `30`, `35`

验收标准：

- `tecs_pitch_sp` 打满后，实际 `pitch` 能更快接近
- `level_turn` 高度误差下降
- 横向 tracking 不显著恶化

**Step 3: 最后才调静稳定/尾流等效项**

Run（示意）:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_loiter --num_envs 1 --steps 2400 --auto_extend_steps \
  --device cuda:1 --headless \
  --tail_fixed_horizontal_effectiveness <k_fixed> \
  --tail_horizontal_tail_q_scale <q_scale>
```

建议只在前两阶段找到候选值后再扫：

- `fixed_horizontal_effectiveness`: `0.85`, `1.0`, `1.15`
- `horizontal_tail_q_scale`: `0.9`, `1.0`, `1.1`

验收标准：

- `loiter progress` 提升
- `mean_abs_height_error_post_warmup_m` 下降
- 不出现明显新振荡

### Task 6: Focused verification 与默认值落地

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: related tests only if default changes

**Step 1: 跑 focused pytest**

Run:

```bash
/home/zn/anaconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/test_tail_aero.py \
  tests/test_tecs.py \
  tests/test_px4_path_tracking_controller.py \
  tests/test_px4_like_guidance.py \
  tests/test_rl_teacher_guidance.py \
  tests/test_teacher_schedule.py \
  tests/test_fly_straight_line_contract.py \
  tests/test_straight_flight_env_reset_contract.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_fly_path_mission.py -q
```

Expected:
- 全绿

**Step 2: 只在 rollout 证据足够后才改默认值**

默认值修改条件：

- straight 不恶化
- `level_turn` 高度误差有稳定改善
- `level_loiter` 的 `progress` 与高度误差同时改善

若只改善 progress、不改善高度误差：

- 不改默认值
- 保留 CLI 可配入口

**Step 3: Review diff scope**

Run:

```bash
git diff -- \
  docs/plans/2026-04-04-tail-aero-sim-compat-retune-implementation-plan.md \
  source/flapping_bot/flapping_bot/physics/tail_aero.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  scripts/flapping_px4/fly_path_mission.py \
  tests/test_tail_aero.py \
  tests/test_straight_flight_env_reset_contract.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_fly_path_mission.py
```

Expected:
- diff 只覆盖计划、tail aero 兼容层、env 透传、rollout 诊断和测试

### Task 7: Commit 策略

**Files:**
- Modify: none

**Step 1: 分阶段提交，避免再次混成一个大补丁**

建议最少拆成 3 个 commit：

1. `test/feat: add tail aero compatibility config hooks`
2. `feat: add tail aero compatibility scaling layer`
3. `feat: expose tail aero retune controls in path rollout`

**Step 2: 不提交无关文件**

不要 add：

- 用户未跟踪的旧计划文件
- 用户本地图片/工具脚本
- rollout 产生的资产 hash 变更

