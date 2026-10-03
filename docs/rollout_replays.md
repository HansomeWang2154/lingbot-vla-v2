# Clean rollout 与视频回放

## 当前计划

基座诊断已完成 10 tasks × 10 episodes，35/100 成功。随后发现模拟器仍使用旧客户端，
未写出 episodes.jsonl；队列在审计环节停止，10k 未执行。旧结果保留，不能当作完整
审计后的配对成绩。启动器现改为按内容同步客户端，先备份旧文件，再记录实际客户端 hash。

按用户要求，优先采集 LoRA 10k 的 clean 小样本：grab_roller、place_empty_cup、lift_pot，
各 2 episodes，共 6 组，seed block 20（初始 seed 2100000）。它们是观察/诊断用 rollout，
不是无偏评测集；不加入训练清单，不启动微调。新增仿真数据用于比赛训练的许可仍待确认。

运行目录：`/root/shared-nvme/lingbot-assets/outputs/rollout_preview_clean_20261003_v2`。
总日志：`/root/shared-nvme/lingbot-assets/logs/rollout_preview_clean_20261003_v2.log`。
首个不带 v2 的目录保留启动路径拼接失败日志，没有生成有效轨迹。已规范化输入目录，
并将客户端/采集器存在性检查提前到模型加载前。
所有资料保留在共享盘；GitHub 仅保存采集代码、测试和说明。

首次真实轨迹验收：grab_roller 的两回合均成功，79/108 个动作。逐步 NPZ、最终观测和
episode 元数据校验通过；MP4 均可解码，320×240，分别 80/109 视频帧（包含末尾成功帧），
8.0/10.9 秒。其余任务继续按队列采集；这不是对全部 6 组完成的声明。

```powershell
# 在你电脑的 PowerShell 中运行：下载已经验收的两组成功回放到当前文件夹。
# 密码通过交互提示输入，不会回显，不要写进命令。
scp -P 2233 'root@ackcs-00gjhnt6@ssh.bj8.bz1.paratera.com:/root/shared-nvme/lingbot-assets/outputs/rollout_preview_clean_20261003_v2/lora_10000/robotwin_4090_lora_clean_c3447b1_10k_demo_clean_20261003_215933/eval_results/grab_roller/episode*_success.mp4' .
```

## 保存内容

每个任务结果目录下：

- `episode0_success.mp4` / `episode0_failure.mp4` 等：头部相机 MP4。
- `episodes.jsonl`：真实场景 seed、指令、episode 和成功标记。
- `rollouts/episode_000_seed_2100000/episode.json`：来源、模型、仿真提交、动作语义、
  状态、帧数、success、binary success reward、terminated/truncated。
- `step_000000.npz` 等：执行动作前的 head/left/right 三路 uint8 RGB、原始机器人状态、
  交给环境的 qpos 目标动作、chunk/offset、环境 step 与采集 Unix 时间。
- `final_observation.npz`：最终观测，供最后一个 transition 对齐。

每个动作前刷新观测；不重复给整个 chunk 配同一张开头图像。写入线程最多缓存两组
待写帧，保证不静默丢帧，写入失败会报错；因此比不录像的测评慢。异常 episode 标记为
incomplete，不算有效 trajectory；已写文件保留，不自动删数据或覆盖同名 episode。

MP4 为 10 fps 的观察回放，不宣称与仿真物理时间或动作频率等速。RGB 原始数据有三路
视角；MP4 默认头部视角。成功与失败均保留，但不保证每个任务都产生成功回合。
启动时备份旧模拟器客户端是可恢复的代码同步，未改动任务、物理参数或成功判据。

## 查看与复用

```bash
# 查看 rollout 总进度；Ctrl+C 只停止查看。
tail -f /root/shared-nvme/lingbot-assets/logs/rollout_preview_clean_20261003_v2.log

# 列出已生成的成功/失败 MP4 回放。
find /root/shared-nvme/lingbot-assets/outputs/rollout_preview_clean_20261003_v2 \
  -type f -name '*.mp4'

# 查看运行清单；完成后会列出有效 rollout 的位置和动作帧数。
cat /root/shared-nvme/lingbot-assets/outputs/rollout_preview_clean_20261003_v2/comparison.json

# 用 Windows PowerShell 将回放下载到当前文件夹（以实际视频路径替换占位符）。
# 输入密码时不会回显；不要把密码写在命令里。
scp -P 2233 'root@ackcs-00gjhnt6@ssh.bj8.bz1.paratera.com:/实际路径/episode0_success.mp4' .
```

raw NPZ 并非可直接交给现有训练器的 LeRobot 数据。后续需动作/相机 schema 审计、
独立采集/验证划分、成功轨迹筛选和 converter round-trip，再在获得数据许可后开展训练。
