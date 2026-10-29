# 证据索引（本轮只读核验）

路径均为本机绝对路径；研究产物不是新的物理真值。报告以 E 编号回溯来源。

| ID | 来源 | 本轮用途 |
|---|---|---|
| E1 | `/home/zn/flap-system-identification/.worktrees/control-identifiability-audit-v1/experiments/control_identifiability_audit_v1/report_zh.md`，HEAD `1745345` | 161航次控制覆盖、独立通道匹配缺口、硬件UUID/固件边界 |
| E2 | `/home/zn/flap-system-identification/.worktrees/closed-loop-control-informativity-v1/experiments/closed_loop_control_informativity_v1/report_zh.md`，HEAD `4fa56ea` | 121航次主采样组、反馈来源、条件剩余、负时间对照、配置与航次依赖 |
| E3 | `/home/zn/flap-system-identification/.worktrees/phase-coupled-mpc-v1/experiments/phase_coupled_mpc_v1/report_zh.md`、同目录 `config.json`，HEAD `ab1896e` | 机械相位、频率、Hall可用性、候选电机预测、质量惯量未闭合 |
| E4 | `/home/zn/flap-system-identification/docs/analysis/results/control_dynamics_realdata_v1/report.md`、`baseline/protocol.json` | 41/17航次历史研究；预测与控制效能分离，局部回归/GRU符号冲突、参考IV失败 |
| E5 | `/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1/configs/data/velocity_rate_dataset_registry.yaml`、`configs/data/velocity_rate_multistep_v1.yaml`；`/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1/{dataset_manifest.json,flights.csv,flight_provenance.json}` | 本轮注册数据、划分、原日志路径、分组、质量与坐标合同 |
| E6 | `/home/zn/flap-system-identification/configs/data/trajectory_dataset_registry.yaml`、`dataset/trajectory_v3_september_expanded/manifest.json`、`configs/data/canonical_dataset_registry.yaml` | 区分9月历史代理模型与4月canonical；不加载两者样本 |
| E7 | `/home/zn/IsaacLab/docs/audits/2026-10-28-crosswind-aero.md`；`source/flapping_bot/flapping_bot/analysis/crosswind_aero.py` | 当前实现侧风机制、数值收敛与成本；模型输出不是实测 |
| E8 | `/home/zn/IsaacLab/docs/audits/2026-10-27-ullt-flight-comparison.md`；`docs/handoffs/2026-10-27-ullt-stage1.md`；`docs/PROJECT_STATE.md` | 4月独立历史比较、高迎角边界、自由体耦合与CPU/GPU边界 |
| E9 | `/home/zn/PX4-Autopilot/src/modules/ekf2/EKF/aid_sources/sideslip/sideslip_fusion.cpp`、`msg/versioned/AirspeedValidated.msg` | 零侧滑伪观测与空速来源语义；另用 `git show c9af571575af33d5403fbb479256848797c801e2:<上述路径>`核对9月6日对应版本，不将当前HEAD覆盖到所有固件 |
| N1 | `results/state_per_flight.csv`、`conditions_per_flight.csv`、`joint_occupancy.csv`、`cruise_windows.csv` | 新增三维联合支持、日期/UUID/固件分层、稳定运动候选组合 |
| N2 | `results/air_wind_per_flight.csv`、`state_per_flight.csv`、`read_scope.json` | 新增原始ULog空速来源、风覆盖、候选侧滑与假设风误差敏感性 |
| N4 | `results/air_motion_joint.csv`、`air_motion_joint_coverage.csv` | 传感器空速/p/r/频率联合支持与另列的候选侧滑五维支持 |
| N3 | `results/response_per_flight.csv` | 新增非重叠原生状态端点增量；观察性包络，不拟合任何动力学参数 |

主工作区：IsaacLab `feat/ullt-state-space-wing-aero@f207c93d`；已有未跟踪PDF、`.ua/`、HTML保留。本轮仅新增此独立研究目录。flap主仓库 `reference-dynamics-data-v1@84a9d1b` 有既存修改/删除/大量未跟踪产物；三个指定worktree的已跟踪文件检查无变化。QgcLogs为`main`，已跟踪文件检查无变化；没有改变分支。未提交、推送、合并、清理。

本轮曾阅读共享 `kinematic_consistency.json`（其中含历史test汇总）；不引用、不重算或用于决策其中的test统计。没有打开test Parquet、test ULog或测试标签，也没有计算其哈希。报告的新增量全部来自显式train/validation。
