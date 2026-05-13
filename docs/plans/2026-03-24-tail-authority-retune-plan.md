# Tail Authority Retune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让新尾翼模型在 `baseline / teacher / RL` 共用同一套接近物理极限的舵面权重，并把当前“纵向掉高、俯仰与频率双饱和”的问题压回到可接受范围。

**Architecture:** 先只改环境侧可用 elevon 权限，把 `elevon_max_deg` 放宽到接近 URDF 物理限位，再用短程 headless rollout 验证饱和是否缓解。若仍旧明显掉高，则优先加静态 trim（`tail_elevator_bias_deg` / `reset_elevon_pitch_deg`），最后才动 PX4-like 纵向控制默认参数；横向参数保持后置，避免把“尾翼权重不足”和“控制器过弱”混在一起。

**Tech Stack:** Python 3.11, IsaacLab direct envs, PX4-like controllers, AST/text unit tests, headless rollout scripts

---

### Task 1: 锁定新的舵面权限合同

**Files:**
- Modify: `tests/test_flapping_asset_cfg.py`
- Modify: `tests/test_straight_flight_env_reset_contract.py`
- Modify: `tests/test_path_tracking_env_contract.py`
- Check: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf`
- Check: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Check: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`

**Step 1: 写 failing contract tests**

在 `tests/test_flapping_asset_cfg.py` 里加一个轻量 XML 解析测试，确认 `left_tail/right_tail/rudder` 的 URDF 物理限位大于 `41 deg`。

在 `tests/test_straight_flight_env_reset_contract.py` 里加一个 AST 测试，确认 `FlappingBotStraightFlightEnvCfg.elevon_max_deg` 是接近物理极限的值，而不是旧的 `25.0`。

可选地在 `tests/test_path_tracking_env_contract.py` 里加一个 AST 测试，确认 path teacher 初始化时仍然用 `self.cfg.elevon_max_deg` 对 `initial_elevon_pitch_action` 做归一化，不会偷偷写死旧范围。

示例断言：

```python
assert tail_joint_limit_deg >= 41.0
assert env_elevon_limit_deg >= 40.5
```

**Step 2: 运行测试，确认先红**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_flapping_asset_cfg.py \
  tests/test_straight_flight_env_reset_contract.py \
  tests/test_path_tracking_env_contract.py -q
```

Expected:
- 新加的 `elevon_max_deg` 合同测试失败
- 失败原因是当前环境默认值仍然是 `25.0`

**Step 3: 记录“为什么现在必须改”**

把当前已知行为问题作为人工验收基线保留在 notes 里：
- `straight`: `mean_abs_track_error_m ≈ 0.44`, 平均高度明显低于 `10 m`
- `path`: `level_turn / level_loiter / multi_segment` 都在约 `2.7~2.9 s` 因高度误差接近 `-5 m` 提前终止

**Step 4: Commit**

```bash
git add tests/test_flapping_asset_cfg.py tests/test_straight_flight_env_reset_contract.py tests/test_path_tracking_env_contract.py
git commit -m "test: lock tail authority contracts"
```

### Task 2: 把 elevon 可用范围放宽到接近物理极限

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Verify: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf`

**Step 1: 最小实现**

把 `FlappingBotStraightFlightEnvCfg.elevon_max_deg` 从 `25.0` 提到 `41.0`，并更新注释说明：
- 目标是“接近 URDF 物理限位”
- 实际仍会和 `joint_limit_softness=0.98` 的软限位取交集，所以最终有效上限会略低于 `41 deg`

这一 task 只改权限，不同时改 trim / gain。

**Step 2: 跑 Task 1 的测试，确认转绿**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_flapping_asset_cfg.py \
  tests/test_straight_flight_env_reset_contract.py \
  tests/test_path_tracking_env_contract.py -q
```

Expected:
- 全绿

**Step 3: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
        tests/test_flapping_asset_cfg.py \
        tests/test_straight_flight_env_reset_contract.py \
        tests/test_path_tracking_env_contract.py
git commit -m "feat: widen elevon authority near physical limit"
```

### Task 3: 先只验证“放宽权限”能不能解掉双饱和

**Files:**
- Run only: `scripts/flapping_px4/fly_straight_line.py`
- Run only: `scripts/flapping_px4/fly_path_mission.py`
- Inspect: `logs/flapping_px4/straight_line_tailcheck_current/.../summary.json`
- Inspect: `logs/flapping_px4/path_tracking_tailcheck_current/.../summary.json`

**Step 1: 跑 straight baseline**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0 \
  --num_envs 1 --steps 1500 \
  --out_dir logs/flapping_px4/straight_line_tailcheck_current \
  --headless
```

重点看：
- `mean_abs_track_error_m`
- `p95_abs_track_error_m`
- `z` / `mean_height_m`
- `action_elevon_pitch` 是否还长期为 `-1.0`
- `action_freq` 是否还长期接近 `1.0`

**Step 2: 跑 path teacher 三个 phase**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_turn --num_envs 1 --steps 2400 --auto_extend_steps \
  --out_dir logs/flapping_px4/path_tracking_tailcheck_current --headless
```

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_loiter --num_envs 1 --steps 2400 --auto_extend_steps \
  --out_dir logs/flapping_px4/path_tracking_tailcheck_current --headless
```

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase multi_segment --num_envs 1 --steps 2400 --auto_extend_steps \
  --out_dir logs/flapping_px4/path_tracking_tailcheck_current --headless
```

重点看：
- `completed_path`
- `failure_kind`
- `failure_step`
- `final_progress_ratio`
- `max_abs_height_error_m`
- `action_elevon_pitch` 是否仍长时间打满

**Step 3: 设 gate，决定是否继续**

如果满足下面两条，就先不继续动 trim / controller：
- `straight` 不再明显掉高，且 `action_elevon_pitch` 不再长期钉死 `-1.0`
- `path` 不再因为 `terminate_height_error_m` 提前终止

否则进入 Task 4。

**Step 4: Commit（仅当“只放宽权限”已经足够好时）**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py
git commit -m "tune: validate wider elevon authority rollout behavior"
```

### Task 4: 优先加静态 trim，而不是先改控制律

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Re-run: `scripts/flapping_px4/fly_straight_line.py`
- Re-run: `scripts/flapping_px4/fly_path_mission.py`

**Step 1: 先加 aerodynamic trim bias**

优先尝试 `tail_elevator_bias_deg`，不要一上来改很多 controller gains。

原因：
- 它直接改变尾翼空气动力零位
- 不依赖 controller 积分项长期顶住一个常值误差
- 更贴合“新尾翼几何换了，名义配平也变了”这个物理事实

符号约定沿用当前代码的现有习惯：
- 更 nose-up / 更保高度 的 elevon 方向应该是更负的 pitch command
- 所以 bias 也优先沿负方向扫

建议扫描顺序：
- `tail_elevator_bias_deg = -4.0`
- 若仍明显掉高，再试 `-6.0`
- 必要时再试 `-8.0`

**Step 2: 只在需要时联动 reset trim**

如果仅有 `tail_elevator_bias_deg` 还不够，再小步调整：
- `reset_elevon_pitch_deg`

这一步只负责让 reset 后更快贴近新配平点，不承担长期稳态修正的主要责任。

**Step 3: 每改一次只跑最小验证**

每次只改一个 trim 参数，然后重复：

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py \
  --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0 \
  --num_envs 1 --steps 1500 \
  --out_dir logs/flapping_px4/straight_line_tailcheck_current \
  --headless
```

只有当 straight 明显改善后，才继续跑三个 path phase。

**Step 4: 停止条件**

出现下面任一组合即可先停 trim 扫描：
- `straight` 平均高度明显回到目标附近，且 `action_elevon_pitch` 不再整段打满
- `path` 三个 phase 不再因高度误差先终止

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py
git commit -m "tune: add static tail trim for latest tail geometry"
```

### Task 5: 只有 trim 还不够时，才动纵向 controller 默认参数

**Files:**
- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py`
- Modify: `scripts/flapping_px4/fly_straight_line.py`
- Optional: `scripts/flapping_px4/fly_loiter.py`
- Re-test: `tests/test_px4_path_tracking_controller.py`
- Re-test: `tests/test_px4_like_guidance.py`
- Re-test: `tests/test_rl_teacher_guidance.py`
- Re-test: `tests/test_teacher_schedule.py`

**Step 1: 保持“纵向优先，横向后置”**

第一轮只考虑这些参数：
- `pitch_kp`
- `pitch_kd`
- `inner_pitch_ki`
- `tecs_pitch_speed_weight`
- `tecs_altitude_error_gain`
- `tecs_max_climb_rate_mps`

先不要动：
- `roll_kp`
- `roll_kd`
- `yaw_kp`
- `yaw_kd`

除非 path 验证表明横向已经成了新的主瓶颈。

**Step 2: 让 baseline 与 teacher 默认值保持一致**

因为：
- `path teacher` 直接吃 env 内构造的 controller 默认参数
- `fly_straight_line.py` 又有一套 CLI 默认值

所以只改 controller 默认类还不够；需要同步 `fly_straight_line.py` 的 parser 默认值，否则你手工跑 baseline 时还是旧参数。

**Step 3: 一次只改一个纵向旋钮**

建议顺序：
1. `pitch_kp`
2. `pitch_kd`
3. `inner_pitch_ki`
4. `tecs_altitude_error_gain`
5. `tecs_pitch_speed_weight`
6. `tecs_max_climb_rate_mps`

每次调整后先跑 `straight`，通过后再跑三条 `path`。

**Step 4: 跑回归测试**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_px4_path_tracking_controller.py \
  tests/test_px4_like_guidance.py \
  tests/test_rl_teacher_guidance.py \
  tests/test_teacher_schedule.py -q
```

Expected:
- 全绿

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py \
        source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py \
        scripts/flapping_px4/fly_straight_line.py \
        scripts/flapping_px4/fly_loiter.py
git commit -m "tune: retune longitudinal defaults for new tail"
```

### Task 6: 最终行为验收与收口

**Files:**
- Run only: `scripts/flapping_px4/fly_straight_line.py`
- Run only: `scripts/flapping_px4/fly_path_mission.py`
- Run only: `tests/test_tail_aero.py`
- Run only: `tests/test_flapping_joint_contracts.py`
- Run only: `tests/test_flapping_task_registration.py`
- Run only: `tests/test_straight_flight_env_reset_contract.py`

**Step 1: 跑最终 targeted pytest**

Run:

```bash
./isaaclab.sh -p -m pytest \
  tests/test_flapping_asset_cfg.py \
  tests/test_tail_aero.py \
  tests/test_flapping_joint_contracts.py \
  tests/test_flapping_task_registration.py \
  tests/test_straight_flight_env_reset_contract.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_px4_path_tracking_controller.py \
  tests/test_px4_like_guidance.py \
  tests/test_rl_teacher_guidance.py \
  tests/test_teacher_schedule.py -q
```

Expected:
- 全绿

**Step 2: 跑最终 headless 行为验收**

至少重复：
- 1 次 `straight`
- 3 个 `path` phase

验收目标：
- `straight` 不 reset，明显减轻掉高，横向误差优于当前基线
- `level_turn / level_loiter / multi_segment` 不再在 `~3 s` 内因高度误差先终止
- `action_elevon_pitch` 与 `action_freq` 不再同时长期顶格

**Step 3: 整理结论**

输出一个简明表格，记录：
- 最终使用的 `elevon_max_deg`
- 最终使用的 `tail_elevator_bias_deg`
- 是否调整 `reset_elevon_pitch_deg`
- 是否调整 `pitch_kp/pitch_kd/inner_pitch_ki/TECS`
- 四个 rollout 的关键指标

**Step 4: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
        source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py \
        source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py \
        scripts/flapping_px4/fly_straight_line.py \
        scripts/flapping_px4/fly_loiter.py \
        tests/test_flapping_asset_cfg.py \
        tests/test_straight_flight_env_reset_contract.py \
        tests/test_path_tracking_env_contract.py
git commit -m "tune: stabilize controllers for latest tail geometry"
```
