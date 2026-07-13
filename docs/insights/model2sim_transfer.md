• 结论：现在还不适合直接把论文模型塞进 IsaacLab 后开始 RL。应先补齐“可部署模型 artifact → 坐标/相位/先验一
  致性 → 仿真闭环验证”三层，否则 RL 很可能在错误动力学上学出看似有效的策略。

  ## 当前模型的真实范围

  论文最终模型是：

  - DeLaurier 纵向力先验；
  - 只修正机体系 FRD 下的 (F_x,F_z)；
  - 输入为相位、扑翼频率、俯仰角速度 (q) 及其相位交叉项；
  - 每个通道采用特征相关 gain + bias，共 44 个系数；
  - 不修正 (F_y,M_x,M_y,M_z)。

  公式和输入见 /home/zn/paper/AeroConf_effective_aero/sections/03_method.tex:97。六折结果为：

  - (F_x)：4.100 N → 1.131 N；
  - (F_z)：6.488 N → 1.843 N；
  - (R^2)：0.934 / 0.953。

  但论文明确说明这只是离线日志预测，还没有进入动态仿真或闭环控制，/home/zn/paper/AeroConf_effective_aero/
  sections/05_results.tex:103。

  ## 目前必须先解决的三个问题

  1. 缺少真正的部署 artifact

  /home/zn/flap-system-identification/scripts/train_fx_fz_structured_correction.py:286 只保存指标、预测和
  manifest，没有保存最终 44 个系数。Nested CV 也只有每折结果，没有“用全部 29 条日志重新拟合”的最终模型。

  必须增加最终模型导出，至少包含：

  - feature 顺序；
  - (\gamma_x,\eta_x,\gamma_z,\eta_z)；
  - ridge 参数；
  - DeLaurier 参数；
  - 相位定义和坐标定义；
  - 训练包络；
  - 数据、代码 commit 和模型版本；
  - NumPy/PyTorch 推理参考输出。

  2. 论文输出是整机有效力，不是单纯机翼力

  论文标签是整机有效 (F_x,F_z)，而先验基本是机翼 DeLaurier。残差会吸收尾翼、机身阻力和其他未建模效应。

  Isaac 当前则计算：

  wing + tail + parasite drag

  然后统一施加到 base，source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py:1283。

  因此：

  - 把修正模型当作“机翼修正”，再加尾翼，会重复计算部分尾翼/机身效应；
  - 把修正结果直接当整机纵向力，又会丢失尾翼动作对纵向力的实时影响。

  推荐做法是重新导出 Isaac 的完整名义纵向先验：

  DeLaurier wing + tail aero + fuselage drag

  用它重新构造残差并拟合相同的 compact gain-bias 模型。这比直接搬论文系数更适合 RL，也保持尾翼控制导数。

  3. 相位和坐标目前没有完全闭合

  当前存在几个明显风险：

  - 系统辨识使用 (q=A\sin\psi)，零相位为中立位置开始上扑；
  - Isaac 环境使用 (q=A\cos\phi)，零相位为最大偏转，source/flapping_bot/flapping_bot/direct/flapping_bot/
    straight_flight_env.py:1196；

  - 两者应显式转换为 (\psi=\phi+\pi/2)，不能直接共用 phase；
  - 论文/日志采用 FRD，Isaac 通常采用 FLU，需要显式执行：
    (F_{\mathrm{FRD}}=[F_x,-F_y,-F_z]{\mathrm{FLU}})，并确认 (q{\mathrm{FRD}}=-q_{\mathrm{FLU}})；

  - 当前迎角符号还有未提交修改，仅用源码字符串测试不能证明物理约定正确；
  - 论文选出的 DeLaurier 参数与 Isaac 默认值明显不同：论文使用 alpha0=4°、cd_f=0、stall=18°、cd_cf=1.2、
    xi=1、开启分离流，而 Isaac 当前默认仍是旧参数，source/flapping_bot/flapping_bot/direct/flapping_bot/
    straight_flight_env.py:293。

  ## 推荐实施顺序

  ### 1. 冻结数据和模型定义

  先确定唯一数据版本、相位定义、质量/CG/惯量和先验参数。三个仓库当前都有未提交改动，暂时不能把论文数字、最
  新 pipeline 和 Isaac 配置视为同一冻结版本。

  特别要确认：

  - 系统辨识 CG 是 IMU 原点 FRD 下 [-0.12154, 0.00541, -0.04298]；
  - Isaac 当前写入 base_link COM 的是 [-0.12154, 0.00541, -0.01298]，source/flapping_bot/flapping_bot/
    direct/flapping_bot/straight_flight_env.py:383；

  - 两者只有在明确的 IMU→base_link 变换下才能同时成立。

  ### 2. 生成最终部署模型

  在 nested CV 完成模型族和超参数选择后：

  1. 用全部 29 条日志重新拟合；
  2. 导出固定推理 artifact；
  3. 保存每折模型，用于估计模型不确定性；
  4. 验证导出模型和训练脚本预测误差小于 1e-5 N。

  ### 3. 建立 Isaac 侧独立修正模块

  建议新增：

  source/flapping_bot/flapping_bot/physics/longitudinal_force_correction.py

  职责只包括：

  - artifact 加载和校验；
  - FLU↔FRD 转换；
  - phase 转换；
  - feature 构造；
  - gain-bias 推理；
  - 包络检测、NaN fallback 和 correction blend；
  - debug 输出。

  不要把公式直接堆进 _apply_action()。

  ### 4. 做离线 parity 测试

  必须先通过：

  - 四个典型相位点的相位/扑翼方向测试；
  - FRD/FLU 力和角速度符号测试；
  - Isaac DeLaurier 与日志 prior exporter 的逐样本一致性；
  - Python reference 与 Isaac Torch 实现一致性；
  - 论文 held-out prediction 指标复现；
  - 当前物理步长 240 Hz 下无跳变、无 NaN。

  ### 5. 做仿真验证，再碰 RL

  依次进行：

  1. 固定状态力扫描：相位、频率、空速、(q) 网格；
  2. 无风固定动作开环：检查周期平均升力、推力、能量和力峰值；
  3. 0.25–1 s 日志控制短时 replay：比较未修正/修正轨迹误差；
  4. 配平扫描：确认 7–8 m/s 附近不持续加速、失高或发散；
  5. PX4 teacher 闭环直线飞行；
  6. 随机初态与轻微风扰动。

  只有这些通过，才能说明模型从“日志预测”进入了“动态仿真可用”。

  ### 6. RL 训练顺序

  建议保持现有 task ID，另加 Corrected 变体做 A/B：

  1. measured nominal、无风直线飞行；
  2. teacher-guided 或 BC 初始化；
  3. straight-flight pure RL；
  4. 加入小范围质量、CG、惯量、风和执行器时延随机化；
  5. 对 correction strength 和六折模型进行 episode-level randomization；
  6. primitive turn；
  7. loiter；
  8. 完整 path tracking。

  当前扫描到的 path-tracking 日志仍没有新的明确成功证据。最终继续使用现有 gate：

  - completion rate ≥ 0.80；
  - final progress ≥ 0.85；
  - termination rate ≤ 0.20。

  推荐路线是“重新基于 Isaac 完整纵向先验拟合部署残差”，而不是直接把论文系数当机翼修正项。下一步应先完成部署
  artifact 和 phase/frame/prior parity 设计，得到你确认后再写详细实施计划或开始修改代码。




    ## 如何判断 robot 是否可信

  飞行包线相近是必要条件，但远远不够。控制器可以把错误模型“调”到相似包线，所以至少需要四层验证：

  1. 力学层
     比较真实日志与仿真的周期平均力、相位波形、峰值和频谱。

  2. 局部动态层
     从真实日志状态初始化，输入相同舵量和扑翼频率，比较 0.25、0.5、1 s 内的速度、角速度和姿态变化。

  3. 控制响应层
     比较扑翼频率、升降舵、方向舵阶跃响应的方向、增益、时间常数和通道耦合。

  4. 包线与闭环层
     比较配平空速、迎角、扑翼频率、爬升/下降能力、转弯半径、最大可控滚转角，以及相同 PX4 控制器下的误差和控
     制量分布。