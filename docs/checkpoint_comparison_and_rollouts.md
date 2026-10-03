# Checkpoint 对照与 rollout 再训练路线

## 当前执行范围

2026-10-03 核验：初筛全部完成，基座 38%、1k 30%、5k 40%、10k 42%。
详见 [初筛对照报告](evaluation_checkpoint_comparison_20260930.md)。
下一轮为基座与 10k 的固定 10-task × 10-episode 诊断；新的 seed block 1，
实际 seed/指令进入 episode 审计。这不是训练 rollout，也不作为正式全任务成绩。

```bash
# 进入项目并加载云容器环境变量。
cd /root/shared-nvme/lingbot-challenge
source .challenge.env

# 仅检查固定诊断计划；不启动测评、不写结果目录。
/root/shared-nvme/lingbot-assets/conda/envs/lingbotvla-comp/bin/python \
  scripts/challenge/compare_clean_checkpoints.py \
  --training-run /root/shared-nvme/lingbot-assets/outputs/robotwin_4090_lora_clean_c3447b1 \
  --output /root/shared-nvme/lingbot-assets/outputs/checkpoint_diagnostic_clean_20261003 \
  --tasks-file configs/vla/robotwin/diagnostic_clean_tasks_20261003.txt \
  --steps 0 10000 --episodes 10 --seed 1 --dry-run

# 查看这轮后台诊断进度；Ctrl+C 只停止查看。
tail -f /root/shared-nvme/lingbot-assets/logs/checkpoint_diagnostic_clean_20261003.log
```

2026-09-30：顺序执行官方 50k 基座、LoRA 1k、5k、10k 的 clean 初筛，每个模型
50 tasks × 1 episode。统一 RoboTwin 13c3c47、BF16、关闭 compile/video、chunk 50、
simulator seed 0、policy RNG seed 42，并显式复用 clean 10k 训练配置和归一化统计。
这是一项固定配置下的权重对照，不是官方 FP32 成绩复现。脚本不会启动训练或提交比赛。

- 云端结果：`/root/shared-nvme/lingbot-assets/outputs/checkpoint_comparison_clean_20260930`。
- 云端总日志：`/root/shared-nvme/lingbot-assets/logs/checkpoint_comparison_clean_20260930.log`。
- `comparison.json`：配置/代码/数据清单/归一化/adapter 校验值、阶段状态和逐任务结果。
- `summary.md`：全部阶段成功完成后生成。任务跳过、缺少回合或计数不一致会停止队列。
- 1k/5k 如尚未合并则离线合并；不覆盖已有目录，不重复保存冻结基座到训练 checkpoint。
  仅为部署新增完整合并权重。新执行须使用新输出目录，失败后保留现场，不自动覆盖续跑。

```bash
# 进入项目并加载容器路径变量。
cd /root/shared-nvme/lingbot-challenge
source .challenge.env

# 仅验证环境、checkpoint 和实验计划，不启动测评、不写结果目录。
/root/shared-nvme/lingbot-assets/conda/envs/lingbotvla-comp/bin/python \
  scripts/challenge/compare_clean_checkpoints.py \
  --training-run /root/shared-nvme/lingbot-assets/outputs/robotwin_4090_lora_clean_c3447b1 \
  --output /root/shared-nvme/lingbot-assets/outputs/checkpoint_comparison_clean_20260930 \
  --dry-run

# 查看当前后台队列日志；Ctrl+C 只停止查看，不会停止测评。
tail -f /root/shared-nvme/lingbot-assets/logs/checkpoint_comparison_clean_20260930.log

# 查看阶段状态和已完成的统计。
cat /root/shared-nvme/lingbot-assets/outputs/checkpoint_comparison_clean_20260930/comparison.json
```

单回合只能粗筛。后续固定 10 个诊断任务，至少对基座和候选各跑 10 个独立 seed，
再考虑最优候选 50 × 10。筛选用的 seed 不再作为独立最终成绩；如按失败结果选择任务，
该诊断集也不再是无偏的泛化估计。保存实际使用的 seed：专家规划失败可能导致跳过初始 seed。

## 能否用 rollout 数据再训练？

技术上可以，但现有评测客户端只保存结果/日志和可选视频，没有保存逐步 RGB、状态、
动作及对齐时间戳。历史 21 条成功结果不等于 21 条可直接训练的 trajectory。

### 先确认数据权限

当前项目继续遵守：追加训练仅使用获准 clean 数据；不使用 randomized、隐藏评测数据，
不提交比赛结果。自行生成的 clean rollout 是否属于比赛许可的训练数据，仍须以
主办方完整规则或明确答复确认；“能在仿真器采集”不代表“比赛允许训练”。
现已按用户授权开展独立的观察/诊断 rollout 采集，详见 [采集与回放](rollout_replays.md)；
尚未开展 rollout 训练，也未改动已有训练清单。

官方仓库的 post-training 示例同时列出 clean 和 randomized。我们的 **追加 LoRA 训练**
仅用 clean，并不能证明官方基座的所有训练来源均为 clean。官方基座能否作为预训练
权重使用，应按比赛预训练模型规定核对；不能凭示例推断已违规，也不能宣称来源全 clean。

### 首选：成功轨迹筛选 + 混合监督微调

1. 在获准、独立的 clean 采集 seed 上 rollout，记录成功判据、策略版本和数据来源。
2. 筛选真正完成任务、观测正常、动作合法、无明显碰撞/反复震荡的成功轨迹；分任务
   限额，避免只收集简单任务。成功筛选具有选择偏差，不保证提高困难任务的成功率。
3. 将数据转换为现有训练器认可的 LeRobot schema，验证相机、动作维度、帧率和索引。
4. 与原始 clean 示范混合。初始可试原始/rollout 约 80/20 的采样比例，并使用更小学习率、
   较短训练周期及独立验证。这只是实验起点，不能视为已验证的最优配比。
5. 首轮保持原有动作归一化和 LoRA 范围，以便隔离数据影响；审计新增数据越界情况。
   如必须改归一化，需只用训练数据重算，并把它作为独立实验、同步部署统计。

成功轨迹 SFT 是行为克隆/自训练，不是强化学习，不会因保存了 reward 就自动变成 RL。
不要将失败轨迹动作作为正例、用失败标签反向回归动作，或直接训练整批未筛选 rollout。

### 失败轨迹：专家纠正或 RL

- 专家纠正：保留模型偏离后的观测，获得该状态下专家的正确动作，训练纠错能力
  （DAgger 风格）。当前环境的初始场景专家规划不能直接当任意失败中间状态的纠正器；
  必须确认可从该状态重新规划并验证安全/可达性。
- RL：成功/失败轨迹都可进入专门的 RL 框架，但需奖励、策略概率或适配的 flow-policy
  优化算法、环境交互和独立验收。现有 SFT 脚本并不支持这件事。
  已有 LingBot-VLA 4B 的第三方 RL 实现不等同于 LingBot-VLA 2.0 6B 的即插即用支持；
  单张 24 GB 4090 上也应先验证 actor/rollout/训练共存的资源需求。

## rollout 采集接口要求

raw NPZ 与头部相机 MP4 的小样本采集已实现；下述训练数据转换及许可审计尚未完成。

每条 transition 至少保存：

- 任务、指令、episode ID、初始/实际 seed、step、时间戳；
- **执行动作前**的 head/left/right 三路原始 RGB，以及原始机器人状态；
- 实际交给仿真器的动作，chunk ID/offset，模型原始输出作为可选审计字段；
- success/reward/terminated/truncated，最终环境判据，而非仅模型判断；
- 模型和 adapter hash、仿真提交、task config、精度、chunk 长度、相机/动作 schema。

执行一段 action chunk 时必须逐步采集观测与实际动作，不能只保存 chunk 开头图像，
再重复配给整个 chunk。训练目标应使用当前 robotwin adapter 期望的环境动作语义，
不要把 55 维 padded/normalized 内部输出直接当机器人原始动作。动作维度及 gripper
约定以固定版本的机器人配置、客户端和 converter 为准，先做单 episode round-trip 检验。

不要在 WebSocket 推理服务里逐帧同步压缩视频造成吞吐损失；采集应在 sim 客户端，
采用有界写入队列/episode 缓冲，报告掉帧和磁盘用量，失败 episode 也保留用于分析。
raw trajectory 与转换后的训练数据目录分开，来源和划分进入运行清单。

## 来源与边界

- [官方 LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)：训练示例和官方推理设置。
- [RoboTwin 数据采集文档](https://robotwin-platform.github.io/doc/usage/collect-data.html)：
  提供专家轨迹采集流程；文档可能对应新版本，不能直接套用到本项目固定旧提交。
- [RLinf LingBot-VLA 4B 模型说明](https://huggingface.co/RLinf/RLinf-lingbot-vla-4b/blob/main/README.md)：
  只作技术参考，不代表本项目 2.0 的训练兼容性或比赛许可。
- [比赛入口](https://tianchi.aliyun.com/competition/entrance/532514/information)：
  登录后的完整规则/答复才是数据权限依据，目前尚未取得明确的 rollout 训练许可条款。
