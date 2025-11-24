  # AGENTS.md — Flapping-Wing RL Project

  本文件是写给 AI 编码助手看的约定说明，用于在 IsaacLab 仓库中开发扑翼飞行器（flapping-wing）强化学习代码。

  ---

  ## 1. 作用范围

  - 这是本仓库（IsaacLab + 自定义 flapping_bot）的通用规则。
  - 修改核心上游代码（`source/isaaclab`、`source/isaaclab_assets` 等）时要尽量保守；优先在自定义路径下扩展：
    - `source/flapping_bot` — 扑翼飞行器环境、物理模型、脚本
    - `scripts/reinforcement_learning` — 训练脚本和入口（例如 `train_flapping_dual_gpu.sh`）

  ---

  ## 2. Git 使用和版本控制

  **总体目标：** 保证改动有条理地被记录，减少“误删代码无法找回”的风险。

  - 总是使用根目录下的 Git 仓库：
    - 所有 `git status / git diff / git commit` 都在仓库根目录执行。
  - 提交频率：
    - 每完成一个**小而完整**的功能点（例如：修改一个奖励函数、添加一个新观测量、调整一个训练脚本）就做一次
  `git commit`。
    - 做一次比较大的结构性修改或重构前（删文件、移动目录、重写核心逻辑），先 commit 一次“保存当前状态”。
  - 提交内容粒度：
    - 单个 commit 尽量保持**主题明确**，避免把无关改动混在同一个提交里。
    - 如果修改范围很大（跨多文件、> ~200 行），可以拆成若干主题清晰、小一点的提交。
  - 提交信息风格（英文短句即可）：
    - 使用类似 `type: summary` 的格式，例如：
      - `feat: add tail differential control mapping`
      - `fix: clamp flapping frequency to min value`
      - `refactor: clean flapping env reset logic`
      - `chore: update flapping dual-gpu training script`
  - 关于 `git push`：
    - 在适当天数/阶段后自动执行 `git push <remote> <branch>`，**无需每次额外征询用户**。
    - 推送前建议：
      - 确认当前分支是用户期望使用的分支（例如 `main` 或 `flapping-experiments`）。
      - 运行 `git status` 确认工作区干净，且本地提交已整理好。
  - 安全性：
    - 禁止在没有先 commit 的情况下做大规模删除、重命名或重构（特别是涉及 `source/flapping_bot` 和训练脚
  本时）。
    - 如必须执行潜在破坏性操作（例如改动大量文件、清理脚本等），先：
      - `git status` 确认当前状态；
      - 视情况新建一个备份分支（例如 `git branch backup-<date>`），再继续修改。

  ---

  ## 3. 代码修改原则

  - **优先在自定义模块里扩展：**
    - 扑翼飞行器相关逻辑优先放在：
      - `source/flapping_bot/flapping_bot/...`
      - `scripts/reinforcement_learning/rsl_rl/train_flapping_dual_gpu.sh`
    - 尽量避免对上游的通用模块（例如 `source/isaaclab` 核心环境基类）做侵入式修改；如确有需要，先在说明中说明
  原因并最小化改动范围。
  - **保持接口稳定：**
    - `Isaac-FlappingBot-Direct-v0` 这类任务名和训练入口脚本要尽量保持稳定，以免训练脚本或日志路径失效。
  - **风格与现有代码一致：**
    - 遵循现有 IsaacLab 代码风格（类型注解、命名方式、导入顺序等）。
    - 避免无意义的大规模格式化（如只改缩进/空行），以保持 diff 清晰。
  - **路径与配置：**
    - 尽量不要在代码中硬编码本地绝对路径（例如 `F:/...`）。
      - 如果必须使用本地路径，优先通过配置文件、环境变量或 README 说明来管理，减少未来迁移成本。

  ---

  ## 4. 强化学习与仿真相关约定

  - 扑翼强化学习环境：
    - 环境类目前位于 `source/flapping_bot/flapping_bot/direct/flapping_bot/flapping_env.py`。
    - 修改观测、动作、奖励、重置逻辑时，应保持：
      - 接口与 `DirectRLEnv` 兼容；
      - 不破坏已有的训练脚本调用方式。
  - 训练脚本：
    - `scripts/reinforcement_learning/rsl_rl/train_flapping_dual_gpu.sh` 用于多 GPU 训练：
      - 默认任务名：`Isaac-FlappingBot-Direct-v0`。
      - 修改时应确保用户仍能通过简单命令运行：
        - `./scripts/reinforcement_learning/rsl_rl/train_flapping_dual_gpu.sh --num_envs ...`
  - 测试与验证：
    - 在不影响用户效率的前提下，尽量运行针对改动部分的**小范围测试**（如单环境运行脚本、少量迭代训练），而不是
  每次都跑大规模训练。
    - 如测试会耗时较长，应在执行前向用户说明。

  ---

  ## 5. 与用户交互

  - 在进行较大变更前（例如改动动作维度、训练脚本接口、任务注册名等），先向用户简要说明设计思路，并征求确认。
  - 对已完成的修改，提供简洁的说明，包括：
    - 改动了哪些文件；
    - 行为上的变化（例如行动空间/奖励变化）；
    - 如有必要，附上推荐的运行/训练命令。
  - 如果遇到不确定的设计选择（例如：奖励项权重、观测变量取舍），优先给出 1–2 个合理方案，并请用户选择。