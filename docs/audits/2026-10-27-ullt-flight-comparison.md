# ULLT / DeLaurier / 日志整机载荷：首轮实飞比较

日期：2026-10-27。分支 `feat/ullt-state-space-wing-aero`，HEAD `a7b3e08e0c1e`，本轮实现未提交。
用户授权直接比较两个机翼模型与日志整机重建载荷，暂不增加整机气动模型；
[分析合同](../decisions/ADR-2026-10-27-ullt-flight-wing-versus-aircraft.md)记录全部假设与选择规则。

## 主要结果

当前无扭转、未标定的 ULLT-4 **没有整体优于冻结 DeLaurier**。
在本轮全部 25 次飞行的已选窗口内，ULLT 的 `Fx/Fz/My` RMSE 均高于 DeLaurier。
周期均值和周期内峰峰值误差也更大；`Fz` 第一谐波的相位误差更小，但第二谐波没有改善。
这不是对论文方法的一般否定，也不是纯机翼误差结论；比较对象是整机重建载荷。

主报告：[交互式图表索引](../../logs/ullt_flight/20261027_screen_report_v2/report.html)。
所有图可单独打开/导出，包括每次飞行的六轴相位曲线、时间曲线、周期均值、跨飞行误差和变频过渡。

五次 validation 飞行等权汇总，以下均为与整机日志重建值的差：

| 量 | DeLaurier RMSE | ULLT-4 RMSE | DeLaurier 周期均值 RMSE | ULLT 周期均值 RMSE | DeLaurier 峰峰值 MAE | ULLT 峰峰值 MAE |
|---|---:|---:|---:|---:|---:|---:|
| Fx (N) | 3.982 | 9.572 | 1.081 | 5.710 | 2.716 | 12.332 |
| Fz (N) | 7.466 | 13.693 | 4.057 | 8.704 | 9.754 | 25.206 |
| My (N m) | 1.426 | 2.267 | 0.760 | 1.129 | 1.188 | 3.603 |

Validation 第一谐波峰相位绝对误差，DeLaurier → ULLT：Fx **27.38° → 28.31°**，
Fz **13.07° → 8.78°**，My **68.29° → 73.48°**。这里是可辨识谐波的周期相位误差，
不是将整条波形自由平移后的最优相关滞后。

20 次 train 飞行的等权 RMSE，DeLaurier → ULLT：Fx **3.893 → 9.278 N**，
Fz **8.623 → 14.861 N**，My **1.502 → 2.299 N m**。没有用这些结果拟合参数。
按飞行 bootstrap 的配对 RMSE 差区间在这三轴均不跨零，方向支持 DeLaurier 更接近日志；
只有 5 次 validation 飞行，不能据此声称覆盖其他日期或飞行条件。

25 个按输入选出的过渡事件中，validation 的等权过渡 RMSE 为：

| 量 | DeLaurier | ULLT-4 |
|---|---:|---:|
| Fx (N) | 3.164 | 8.195 |
| Fz (N) | 7.602 | 13.402 |
| My (N m) | 1.311 | 1.977 |

## 数据与覆盖

- 数据：`canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3`。
- DeLaurier：`delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4`，物理来源 `3b5d4ec`；原冻结预测不修改。
- 读取 388,289 条 train/validation 行，来自 25 次四月飞行；未读取封存 test。
- 每次飞行两个不重叠窗口，共 50 个；每窗 12 s，其中前 2 s 预热，后 10 s 计分。
- 50,050 条计分行，2,027 个完整周期，25 个满足阈值的变频事件。没有积分拒绝。
- 328,230 行在所选窗口外；10,000 行用于预热；9 行属于过短连续段。排除记录随原始行一起保存。
- 不是全航程或九、十月日志验证。输入选窗不查看任何模型误差，规则和每个窗口起止时刻均已保存。

主标签保持已注册版本的固定中立翼惯量重建；没有将已有三体 candidate 标签替换进去。
坐标为 FRD，力矩统一关于相对 IMU `[-0.12154,0.00541,-0.04298] m` 的中立构型整机重心。
两种模型均仅输出机翼载荷；未扣除尾翼、机身、干扰或估计器误差。

## 数值和运动学诊断

采用 `ULLTConfig(model="four_state")`，10 面元/翼、`Clα=2π`、`Cd=0`、`Kam=1`、无扭转。
DeLaurier 的 80 条带、原阻力和其他截面参数保留。因此这轮不是只有尾迹状态不同的严格消融。
ULLT 包含机体角运动和频率导数；冻结 DeLaurier 的对应运动学近似也保持原样，不能把差异全部归于 ULLT 状态方程。

当前每条记录的最大局部迎角的 P50/P95/最大值为 **45.34° / 66.91° / 75.53°**。
薄板附着流假设的物理适用性存疑；误差来源尚未唯一分解，不能仅凭此统计确定是失速导致。
记录相位差分与 `2πf` 的 RMS 差为 **1.572 rad/s**；本轮保留各测量通道，不调相位或频率去拟合载荷。
附加质量使用日志加速度，整机重建也用加速度，因此结果只支持离线载荷一致性，不是独立动力学预测证据。

在第一条按输入选定的飞行 `log_0_2026-4-14-10-21-28` 的 16.46–20.46 s 中，
只对后 2 s 计分的敏感性结果如下（差值相对 10 面元/翼、10 ms、零尾迹基准）：

| 变化 | Fx 相对 RMS 差 | Fz 相对 RMS 差 | My 相对 RMS 差 |
|---|---:|---:|---:|
| 外层步长 10 → 5 ms，内部仍作 RK4 子步 | 2.65e-9 | 6.17e-9 | 4.90e-9 |
| 每翼面元 10 → 20 | 5.149% | 2.120% | 5.107% |
| 零尾迹 → 稳态尾迹初始化，均预热 2 s | 0.00634% | 0.00540% | 0.00552% |

该窗口的面元加密绝对 RMS 差为 Fx 0.580 N、Fz 0.546 N、My 0.108 N m。
空间精度尚不能称为完全收敛；这一有限检查未显示足以解释主要偏差的时间积分/初始化误差。
未进行所有工况的加密或原始信号采样率敏感性验证。

## 实现与验证

新增项目内模块：

- `analysis/ullt_flight_replay.py`：registry/哈希/逐键校验、连续段与输入选窗、三维运动学、尾迹重放、载荷分项。
- `analysis/ullt_flight_report.py`：完整周期、谐波、过渡、飞行等权指标和 HTML/PNG/CSV 报告。
- `analysis/ullt_flight_sensitivity.py`：独立的有限数值敏感性检查。
- `scripts/flapping_px4/{compare,report,check}_ullt_flight_*.py`：运行、重绘与敏感性 CLI；具体命令见下方。
- `tests/test_ullt_flight_replay.py`：运动学、断段、幅相、输入选窗独立性和原始周期边界测试。

针对性回归 **90 passed, 2 skipped**，包括原 ULLT、DeLaurier strip wrench、动态扭转、legacy 扭转和 multibody wing coupling。
跳过的是 CUDA 一致性测试；没有运行 GPU 仿真或训练。新增文件基础 flake8 通过。
另有独立统计复核与来源校验，见最终报告目录的 `verification.json`。

首个诊断版按窗口内周期行数校验边界，最终复核改为与原始 `(log_id,segment_id,cycle_id)` 全部行数比较，
额外去掉两个截断周期（2029 → 2027）。重绘只更新统计/图表，没有重算或改写原重放载荷。
第一版保留为历史产物；最终统计使用 `screen_report_v2`，不是 `screen_v1` 内的旧报告。

## 产物与复现

- 原始重放及全部行排除原因：`logs/ullt_flight/20261027_screen_v1/replay.parquet`。
- 冻结输入/代码/几何来源：同目录 `manifest.json`、`replay_diagnostics.json`。
- **最终图表和指标**：`logs/ullt_flight/20261027_screen_report_v2/`，含 `report.html`、`summary.json`、
  `per_flight.csv`、`cycles.csv`、`phase_bins.csv`、`transitions.csv`、`cycle_rejections.csv`、来源与独立复核 JSON。
- 敏感性：`logs/ullt_flight/20261027_sensitivity_v1/summary.json`、`traces.npz`。

所有输出目录拒绝覆盖；复现时将下列 `*_new` 替换为未使用的路径。

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
MPLCONFIGDIR=/tmp/ullt-mpl-cache PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
python scripts/flapping_px4/compare_ullt_flight_logs.py \
  --identification-root /home/zn/flap-system-identification \
  --wing-csv outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv \
  --wing-urdf source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf \
  --output-dir logs/ullt_flight/replay_new

MPLCONFIGDIR=/tmp/ullt-mpl-cache python scripts/flapping_px4/report_ullt_flight_logs.py \
  --replay-dir logs/ullt_flight/20261027_screen_v1 \
  --output-dir logs/ullt_flight/report_new

python scripts/flapping_px4/check_ullt_flight_sensitivity.py \
  --replay-dir logs/ullt_flight/20261027_screen_v1 \
  --wing-csv outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv \
  --wing-urdf source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf \
  --output-dir logs/ullt_flight/sensitivity_new
```

需要可靠退出状态的自动检查使用已激活环境的 Python；也可使用 `./isaaclab.sh -p`，但需检查输出结果。
后续可显式 `--full-runs` 扩展到全部有效连续段；`--smoke` 只执行每段前四秒，不适合作为完整结论。

## 后续工作和修改边界

优先检查布翼被动俯仰/扭转、大迎角截面闭合及整机重建误差，再考虑参数修正；
既有[改进清单](../plans/2026-10-27-ullt-model-improvements.md)继续有效。
这次结果不支持把 ULLT 提升为默认模型；也不能据此把具体参数认定为真实物理原因。
尚未开展尾翼/机身建模、三体标签敏感性、九/十月数据接入、全航程重放或独立论文曲线复现。
后续新增分析继续限定在项目分析/脚本/测试/文档范围；保留原始日志、注册数据、冻结预测与训练配置。
本轮无待用户补充才能交付的事项；没有提交或推送。
