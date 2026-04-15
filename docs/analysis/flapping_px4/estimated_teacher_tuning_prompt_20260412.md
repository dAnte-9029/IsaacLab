你现在是一个“最小改动、controller 优先、evidence 优先”的 coding agent，在本地仓库 `/home/zn/IsaacLab` 中工作。

你的任务不是继续做 IMU plumbing，也不是做 RL 调参，而是基于当前已经完成的 dual-teacher / IMU contract，专门调优 **estimated teacher controller**，同时保持 **truth teacher controller** 作为固定法医基线，不做主调。

## 绝对前提

1. 当前工作基线 commit 是 `17e3a40c`
2. 机翼气动后端固定使用 `DeLaurier`
3. 不要把主要精力放到 reward / observation / actor / student-teacher 训练机制
4. 不要把问题归咎于 RL
5. truth teacher 保持固定，estimated teacher 单独调
6. 优先做“同一套 controller 代码 + 两套参数档”，不要轻易复制两套 controller 实现

## 当前已确认状态

### 1. dual-teacher contract 已经显式化

现在环境和脚本支持：

- `teacher_state_source = truth | estimated`
- `policy_state_source = truth | estimated`
- `imu_source = synthetic | isaacsim`

关键文件：

- `source/flapping_bot/flapping_bot/direct/flapping_bot/state_source_contract.py`
- `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`

### 2. 非 RL controller-only 脚本已支持 estimated teacher

关键脚本：

- `scripts/flapping_px4/fly_straight_line.py`
- `scripts/flapping_px4/fly_loiter.py`

### 3. live Isaac IMU 路径已经真正跑通

不是名义选择，而是真绑定 Isaac Lab IMU。

关键文件：

- `source/flapping_bot/flapping_bot/px4_like/imu_provider.py`
- `source/flapping_bot/flapping_bot/px4_like/isaacsim_imu_adapter.py`
- `source/flapping_bot/flapping_bot/px4_like/state_estimation.py`

### 4. RL 入口默认不再把 path-tracking 评估 silently 指向 truth teacher

关键文件：

- `scripts/flapping_rl/train_and_watch.py`
- `scripts/flapping_rl/eval_suites.py`
- `scripts/flapping_rl/path_tracking_eval_common.py`

当前 RL 默认策略：

- `teacher_state_source=estimated`
- `policy_state_source=estimated`
- `imu_source=synthetic`

truth suites 仍保留给 forensic baseline。

## 当前证据

### straight-line smoke

truth:

- `logs/flapping_px4/straight_line/20260412_165617/summary.json`
- mean abs track error = `0.058 m`

estimated + synthetic:

- `logs/flapping_px4/straight_line/20260412_165649/summary.json`
- mean abs track error = `0.253 m`
- mean est pos xy err = `0.916 m`
- mean est vel xyz err = `0.557 m/s`
- mean est yaw err = `0.752 deg`

estimated + isaacsim_live:

- `logs/flapping_px4/straight_line/20260412_165322/summary.json`
- mean abs track error = `3.372 m`
- mean est pos xy err = `1.100 m`
- mean est vel xyz err = `0.643 m/s`
- mean est yaw err = `1.741 deg`

结论：

- estimated path 是通的
- 但 live IMU 直飞明显比 synthetic 差
- 这更像 tuning / phase 问题，不像 gross sign 或 frame error

### loiter smoke

truth:

- `logs/flapping_px4/loiter/20260412_165724/summary.json`
- mean abs track error = `0.244 m`
- mean abs height error = `2.608 m`

estimated + synthetic:

- `logs/flapping_px4/loiter/20260412_165759/summary.json`
- mean abs track error = `0.839 m`
- mean abs height error = `2.790 m`
- mean est pos xy err = `1.056 m`
- mean est vel xyz err = `1.042 m/s`
- mean est yaw err = `1.540 deg`

estimated + isaacsim_live:

- `logs/flapping_px4/loiter/20260412_165420/summary.json`
- mean abs track error = `0.618 m`
- mean abs height error = `1.491 m`
- mean est pos xy err = `1.175 m`
- mean est vel xyz err = `0.926 m/s`
- mean est yaw err = `0.966 deg`

结论：

- live IMU 在 loiter 上并没有崩
- 剩余问题仍然更像 estimated controller tuning，而不是 IMU 接口错误

## 你的核心任务

请把主要精力放在 **estimated teacher controller** 的整定上，目标是：

1. `truth teacher` 基本不动，保留为法医基线
2. `estimated + synthetic` 明显改善
3. `estimated + isaacsim_live` 至少不要恶化，并尽量接近 `estimated + synthetic`

## 推荐排查/改动顺序

1. 先读 controller 与 estimator 关键文件
2. 明确哪些参数应该只对 estimated teacher 生效
3. 先改最小参数面，不要大重构
4. 优先在 `estimated + synthetic` 上做闭环
5. 最后拿 `estimated + isaacsim_live` 做 transfer check

## 优先阅读文件

- `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- `source/flapping_bot/flapping_bot/px4_like/loiter_controller.py`
- `source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py`
- `source/flapping_bot/flapping_bot/px4_like/tecs.py`
- `source/flapping_bot/flapping_bot/px4_like/state_estimation.py`
- `scripts/flapping_px4/fly_straight_line.py`
- `scripts/flapping_px4/fly_loiter.py`

## 推荐调优方向

重点不是改 truth controller，而是给 estimated teacher 单独的：

- 更合适的带宽
- 更合适的阻尼
- 对估计高度率 / 空速 / yaw / wind 误差更稳健的 TECS 参数
- 必要时更强的内环平滑 / rate limit / cycle-mean 使用方式

但要求：

- 不要用大范围经验调参
- 不要顺手重写 controller 架构
- 不要先碰 RL

## 验证要求

至少跑这些：

```bash
python -m py_compile source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/loiter_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py
python -m py_compile source/flapping_bot/flapping_bot/px4_like/tecs.py
./isaaclab.sh -p -m pytest tests/test_train_and_watch.py tests/test_eval_suites.py tests/test_fly_controller_state_source_contract.py tests/test_imu_provider.py tests/test_sensor_state_estimator.py tests/test_rl_teacher_guidance.py tests/test_path_tracking_env_contract.py -q
```

然后至少做这些 runtime 对照：

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source truth --teacher_state_source truth --imu_source synthetic --steps 1200 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source estimated --teacher_state_source estimated --imu_source synthetic --steps 1200 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py --state_source estimated --teacher_state_source estimated --imu_source isaacsim --steps 1200 --headless

./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source truth --teacher_state_source truth --imu_source synthetic --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source estimated --teacher_state_source estimated --imu_source synthetic --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --state_source estimated --teacher_state_source estimated --imu_source isaacsim --steps 1200 --min_loiter_turns 0.5 --metrics_warmup_s 2.0 --headless
```

## 你最终必须回答

1. 你是否保持了 truth teacher 基本不动
2. 你给 estimated teacher 增加/修改了哪些独立参数面
3. `estimated + synthetic` 相比修改前改善了多少
4. `estimated + isaacsim_live` 是否也随之改善，还是暴露出额外 estimator 问题
5. 现在是否建议把 estimated teacher 作为后续 RL 的默认 realism baseline

## 输出格式

1. `Changes`
- 改了什么
- 为什么只改 estimated teacher 或 estimated 参数档

2. `Verification`
- 实际运行了哪些命令
- 结果如何

3. `Evidence`
- straight / loiter 的 truth、estimated+synthetic、estimated+isaacsim_live 对照
- 至少给出关键指标前后变化

4. `Decision`
- estimated teacher 是否已经足够稳定
- truth teacher 是否保持为法医基线

5. `Residual Risk`
- 剩余 1~3 个风险点
