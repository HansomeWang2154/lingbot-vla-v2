# LingBot-VLA 2.0 RoboTwin clean 快速测评报告

## 报告定位

本报告记录 10k-step clean-only LoRA 模型在 RoboTwin `demo_clean` 环境中的
端到端快速测评。它可以作为**工程验证报告和候选模型初筛结果**，不能作为稳定性能、
比赛最终成绩或 randomized 泛化能力的结论：本轮每个任务只执行了一个 episode。

## 实验配置

| 项目 | 配置 |
|---|---|
| 完成日期 | 2026-09-23（运行日志时间） |
| 模型 | `robotwin_4090_lora_clean_c3447b1_10k` |
| 初始化基座 | 官方 RoboTwin 50k HF checkpoint |
| 微调 | clean-only LoRA，10,000 optimizer steps |
| LoRA 作用域 | `action_expert`，并训练动作输入/输出与时间投影模块 |
| 仿真器 | RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1` |
| 环境 | `demo_clean` |
| 任务数 | 50 |
| Episode | 每任务 1，共 50 |
| 推理精度 | BF16；FP32 关闭 |
| 编译 | `torch.compile` 关闭 |
| 并发 | 1 张 RTX 4090，1 个推理/仿真槽 |
| 视频 | 关闭 |

部署时使用 clean 训练清单对应的归一化统计、55 维动作空间和合并后的 10k 权重。
模型严格加载成功；被过滤的 84 个张量仅属于配置中已禁用的辅助对齐/蒸馏分支，
不包含动作策略权重。

## 结果摘要

| 指标 | 结果 |
|---|---:|
| 完整执行 | 50/50 |
| 跳过/基础设施失败 | 0 |
| 成功 | 21 |
| 失败 | 29 |
| 快速成功率 | **42.0%** |
| 仿真任务总耗时 | 5,387 s（约 89 min 47 s，不含首次模型加载） |
| 平均每任务 | 107.6 s |
| 中位每任务 | 97.5 s |
| 最短/最长 | 65 s / 231 s |
| 成功任务平均耗时 | 77.8 s |
| 失败任务平均耗时 | 129.2 s |

若暂时把 50 个异质任务近似为独立 Bernoulli 样本，42.0% 的 95% Wilson 区间约为
29.4%–55.8%。该区间仅用于说明单 episode 结果的不确定性；任务并非同分布样本，
不能把它视为严格的模型置信区间。

成功任务：

`stack_bowls_three`、`open_microwave`、`beat_block_hammer`、`grab_roller`、
`handover_mic`、`move_can_pot`、`move_playingcard_away`、
`place_container_plate`、`place_empty_cup`、`place_mouse_pad`、
`place_object_scale`、`place_phone_stand`、`open_laptop`、
`pick_diverse_bottles`、`place_a2b_right`、`place_bread_basket`、
`place_bread_skillet`、`rotate_qrcode`、`shake_bottle_horizontally`、
`shake_bottle`、`turn_switch`。

失败任务：

`lift_pot`、`hanging_mug`、`scan_object`、`handover_block`、`click_bell`、
`put_object_cabinet`、`stack_blocks_three`、`place_shoe`、`adjust_bottle`、
`blocks_ranking_rgb`、`blocks_ranking_size`、`click_alarmclock`、
`dump_bin_bigbin`、`move_pillbottle_pad`、`place_cans_plasticbox`、
`place_dual_shoes`、`place_fan`、`place_object_basket`、
`place_object_stand`、`move_stapler_pad`、`pick_dual_bottles`、
`place_a2b_left`、`place_burger_fries`、`place_can_basket`、
`press_stapler`、`stack_blocks_two`、`stack_bowls_two`、`stamp_seal`、
`put_bottles_dustbin`。

## 结果解释

### 为什么测评慢

模型单次生成动作通常约需 1 秒，但一个 episode 包含最多 400 个物理控制步、
多路 RGB-D 渲染、双臂运动/碰撞规划和场景初始化。当前 50 个任务在一个槽中串行运行，
所以主要瓶颈是仿真而非神经网络推理。失败任务通常会跑满时间上限；成功任务可以提前
结束，因此失败任务平均比成功任务多耗时约 51 秒。

### 42.0% 能说明什么

本轮证明了模型加载、WebSocket 推理、RoboTwin 仿真、任务调度和结果汇总能够完整闭环，
并给出了一个低成本的候选模型初筛值。它尚不能区分以下来源：

- 单 seed 的随机性；
- BF16 相对官方 FP32 设置的数值影响；
- 原始 50k 基座本身的能力上限；
- 10k LoRA 对不同任务的改善或退化；
- 动作 chunk 长度、闭环频率和任务时限的影响；
- 某些任务在 clean 训练数据中的覆盖度或数据质量问题。

因此，当前结果不能支持“10k LoRA 优于基座”的结论，也不能外推到 randomized 环境。

## 改进与验证路线

### P0：先建立可信对照

1. 使用完全相同的 RoboTwin 提交、BF16、任务顺序和 seed，对官方 50k 基座执行
   `50 tasks × 1 episode`。这是判断 LoRA 是否真正增益的最低成本对照。
2. 对 1k、5k、10k checkpoint 做同配置快速筛选，避免只按训练 loss 选择最终权重。
3. 从成功与失败任务中选取固定的诊断集，各执行 10 episodes；诊断集用于定位，
   不能替代全任务最终评测。
4. 只对最有希望的模型运行 `50 tasks × 10 episodes`，再决定是否投入 100 episodes。

### P1：提高模型成功率

1. **checkpoint 选择**：比较基座、1k、5k、10k，检查 10k 是否已经过拟合或遗忘。
2. **任务均衡采样**：审计失败任务在 clean 训练集中的 episode、语言指令和动作帧占比；
   对低覆盖任务进行任务均衡采样，但不得使用 randomized/隐藏评测数据训练或调参。
3. **动作闭环消融**：在独立 clean 验证 seed 上比较 action chunk 25/50；较短 chunk
   可能改善长时序纠错，但会增加推理次数，最终评测配置必须固定。
4. **LoRA 范围消融**：当前主要适配 action expert。可比较仅动作专家与增加跨模态/投影
   adapter 的方案，同时严格监控 24 GB 显存、训练稳定性和灾难性遗忘。
5. **动作损失与归一化审计**：按关节、末端位姿和 gripper 维度检查误差，确认少数高方差
   维度没有主导 loss；为精细接触任务单独检查 gripper 与末端动作分布。
6. **失败阶段标注**：为诊断集开启视频，将失败分为感知/目标选择、接近、抓取、搬运、
   放置和终止判断，再决定补数据还是改训练目标。
7. **精度对照**：在显存不少于 32 GB 的 GPU 上运行一组 FP32 对照，量化 BF16 带来的
   差异，而不是凭经验归因。

### P2：缩短迭代时间

1. 快速筛选只运行固定诊断任务；正式模型再覆盖全部 50 个任务。
2. 保持推理服务常驻和 `--no_video`；当前流程已经采用这两项。
3. 单独基准 `torch.compile`。它只能减少动作推理时间，无法消除占主导的仿真耗时，
   且需要计入首次编译成本。
4. 若预算允许，使用多 GPU 按“一卡一个模型服务和仿真槽”并行。单张 24 GB 4090
   无法安全驻留两个约 15–18 GB 的模型副本。
5. 在不改变官方物理步数、相机和成功判据的前提下评估仿真 CPU、渲染和资产读取瓶颈；
   不要通过缩短任务时限来制造不可比较的加速结果。

## 建议的下一轮实验矩阵

| 阶段 | 模型 | 任务 × Episode | 目的 |
|---|---|---:|---|
| A | 官方 50k 基座 | 50 × 1 | 建立最低成本基线 |
| A | LoRA 1k、5k、10k | 各 50 × 1 | checkpoint 初筛 |
| B | 基座与最佳 LoRA | 10 × 10 | 固定诊断集、估计 seed 方差 |
| C | 最佳候选 | 50 × 10 | 形成可用于模型选择的 clean 报告 |
| D | 最终候选 | 按比赛要求 | 官方环境最终验收；不使用隐藏数据调参 |

在完成阶段 A 的基座对照前，优先级最高的工作不是继续增加训练步数，而是确认当前
42.0% 相对原始模型究竟是提升还是退化。
