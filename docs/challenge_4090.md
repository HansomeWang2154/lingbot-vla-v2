# 单卡 RTX 4090 云容器运行手册

本文档用于在一台 **1×RTX 4090（24 GB）** 的 Linux 云容器上准备 LingBot-VLA 2.0 的竞赛微调与推理环境。脚本不会下载模型或数据；模型许可、比赛数据授权和下载来源需要参赛者自行确认。

> [!IMPORTANT]
> 已经在聊天、截图或工单中发送过的密码应视为泄露。先在云平台重置容器密码，再改用 SSH 密钥；不要把密码、私钥、Hugging Face token 或 W&B key 写入仓库、命令行参数、训练配置和日志。

## 1. 安全登录

在本机生成独立密钥，并通过云平台控制台添加公钥。首次连接时人工核对主机指纹：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/lingbot_challenge -C lingbot-challenge
ssh -i ~/.ssh/lingbot_challenge -p <PORT> <USER>@<HOST>
```

建议禁止 root 密码登录、限制安全组来源 IP，并为 GitHub 使用单独的 deploy key。不要在 shell 历史中使用 `https://TOKEN@github.com/...`。

## 2. 代码与持久盘布局

容器系统盘可能在实例释放后清空。先从云平台文档或 `df -hT` 确认持久共享盘挂载点，以下用 `/shared/lingbot-challenge` 举例；不要盲目照抄路径。

```bash
git clone <YOUR_GITHUB_REPOSITORY> ~/lingbot-vla2-challenge
cd ~/lingbot-vla2-challenge

bash scripts/challenge/bootstrap.sh \
  --shared-root /shared/lingbot-challenge \
  --min-free-gb 100

source .challenge.env
bash scripts/challenge/preflight.sh --phase host
```

初始化后，缓存、模型、数据、输出、日志、临时文件和 Conda 包均位于共享盘：

```text
/shared/lingbot-challenge/
├── cache/                 # Hugging Face / Torch / pip / Conda / pycache
├── conda/envs/
├── models/
├── data/
│   ├── train/
│   └── local_validation/
├── outputs/
├── logs/
├── tmp/
└── wandb/
```

`.challenge.env` 含本机绝对路径，权限为 `0600` 且已被 `.gitignore` 排除。每次新 shell 都先 `source .challenge.env`。

## 3. 安装固定版本环境

安装需要联网且耗时较长；它只安装依赖，不下载模型：

```bash
bash scripts/challenge/bootstrap.sh \
  --shared-root /shared/lingbot-challenge \
  --install-env

source .challenge.env
eval "$(conda shell.bash hook)"
conda activate lingbotvla
bash scripts/challenge/smoke.sh --phase host
```

目标版本为 Python 3.12、PyTorch 2.8.0、TorchVision 0.23.0、Transformers 4.57.3、FlashAttention 2.8.3。若有与 CUDA/PyTorch/Python ABI 完全匹配的本地 wheel，可增加 `--flash-attn-wheel /shared/...whl`。

## 4. 下载白名单与数据边界（不入 Git）

Hugging Face 下载白名单固定如下：

| 类型 | 仓库 | 允许范围 |
|---|---|---|
| 数据 | `TianxingChen/RoboTwin2.0` | **仅** `lerobot_dataset/RoboTwin_lerobot_v21.zip`（约 2.4 GB） |
| 模型 | `robbyant/lingbot-vla-v2-6b-robotwin` | 模型仓库 |
| 模型 | `Qwen/Qwen3-VL-4B-Instruct` | 模型仓库 |
| 模型 | `Ruicheng/moge-2-vitb-normal` | `model.pt` |

严禁对 `TianxingChen/RoboTwin2.0` 执行 `snapshot_download`、`git clone` 或不带精确文件名的 `hf download`；整个数据仓库约 **1.53 TB**。仓库提供的受限脚本把数据下载硬编码为单个文件：

```bash
source .challenge.env
eval "$(conda shell.bash hook)"
conda activate "$CHALLENGE_ENV_NAME"

# 只下载约 2.4 GB 的 v2.1 压缩包，不会抓取 1.53 TB snapshot
bash scripts/challenge/fetch_whitelist.sh dataset

# 只下载白名单模型；不会下载任何数据集
bash scripts/challenge/fetch_whitelist.sh models
```

下载完成后核对路径。比赛只允许 50 个 clean 任务用于训练，`randomized` 仅用于官方评测，不得放入训练目录、用于调参或生成统计量。不要直接调用 `unzip`；准备脚本会先校验固定 SHA-256、安全解压并审计 v2.1 来源，再为 LeRobot 0.4.2 离线转换出 v3.0 训练副本：

```bash
source .challenge.env
eval "$(conda shell.bash hook)"
conda activate "$CHALLENGE_ENV_NAME"
bash scripts/challenge/prepare_robotwin_data.sh
find "$CHALLENGE_TRAIN_DATA" -iname '*randomized*' -print
```

最后一条命令必须没有输出。脚本保留只读审计意义上的 `RoboTwin_lerobot_v21` 来源目录，在同一文件系统中以硬链接暂存副本运行官方转换器（`push_to_hub=false`），验证 `v3.0`、2500 episodes 和 2413 instruction tasks 后才原子发布 `RoboTwin_lerobot_v30`。`clean_training_data.txt` 只包含 v3.0 目录的一行；失败不会切换训练清单，重复执行会重新验证并复用合格的 v3.0 产物。不要从不受信任的压缩包直接以 root 身份覆盖系统目录；若比赛方更新文件，应先核对官方公告与校验值。

按 `.challenge.env` 中的路径放置：

```text
$CHALLENGE_MODEL_ROOT/
├── lingbot-vla-v2-6b-robotwin/ # 预训练/后训练模型仓库
├── Qwen3-VL-4B-Instruct/       # 至少包含 config.json
└── moge-2-vitb-normal/model.pt

$CHALLENGE_TRAIN_DATA/           # v2.1 审计来源 + 本地转换的 v3.0 clean 训练副本
$CHALLENGE_LOCAL_VAL_DATA/       # 自行从训练数据划分的本地验证集
```

不要把 randomized/隐藏测试集、评测标签、评分服务凭据挂进训练进程。脚本会拒绝训练目录中的 `*randomized*` 路径，也约定任何带 `.competition_eval` 标记的数据目录都禁止训练；若平台设置了 `COMPETITION_EVAL_DATA`，训练预检也会立即失败。官方线上评测数据应只由官方评测器读取。

数据准备完毕后：

```bash
source .challenge.env
eval "$(conda shell.bash hook)"
conda activate "$CHALLENGE_ENV_NAME"
bash scripts/challenge/compute_clean_norm_stats.sh
bash scripts/challenge/preflight.sh --phase train
bash scripts/challenge/smoke.sh --phase train
```

统计脚本会先重新验证固定 v2.1 来源的 50 个任务和 2500 条 clean 轨迹、v3.0 元数据及单行训练清单，再把结果写到
`$CHALLENGE_CLEAN_NORM_STATS`。默认拒绝覆盖已有文件；明确需要重算时使用 `--overwrite`。

## 5. 单卡 24 GB 微调边界

上游 `configs/vla/robotwin/robotwin.yaml` 是 32 卡示例，包含 `micro_batch_size: 32`、`global_batch_size: 1024` 和 FP32，不能直接用于 4090。竞赛配置至少应满足：

- 已在 24 GB RTX 4090 上实测 `micro_batch_size: 2`、
  `gradient_accumulation_steps: 2`，全局 batch 为 4；若其他 4090 型号或
  驱动环境出现 OOM，可回退到 `micro_batch_size: 1`、累积 4 次；
- 开启 `enable_gradient_checkpointing`；
- 使用 BF16，不使用 FP32 全参训练；当前单进程路径需设
  `enable_mixed_precision: false`，这样模型会直接按 BF16 加载。该开关设为
  `true` 会在并行包装前把完整模型物化为 FP32，24 GB 卡会在 LoRA 开始前即面临 OOM；
- 优先冻结视觉编码器并使用 LoRA/参数高效微调；
- 单卡不开 FSDP2 分片和 Distributed Muon；优化器优先 AdamW；
- 先用极少量样本跑 2–5 step，再启动完整训练；
- checkpoint、TensorBoard/W&B 和日志统一写到 `$CHALLENGE_OUTPUT_ROOT`。

最终数值必须以比赛规则和实际显存测试为准。若完整 6B 模型即使采用上述设置仍 OOM，应进一步缩短序列/动作块、减少视觉帧、关闭教师蒸馏分支，或改用量化 LoRA；不要通过读取评测集来选超参数。

示意命令（以项目实际生成的竞赛配置为准）：

```bash
source .challenge.env
eval "$(conda shell.bash hook)"
conda activate "$CHALLENGE_ENV_NAME"

CUDA_VISIBLE_DEVICES=0 bash train.sh \
  tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin_4090_lora.yaml \
  --model.model_path="$CHALLENGE_VLA_MODEL_DIR" \
  --model.tokenizer_path="$CHALLENGE_QWEN3_DIR" \
  --data.train_path="$CHALLENGE_TRAIN_LIST" \
  --data.norm_stats_file="$CHALLENGE_CLEAN_NORM_STATS" \
  --train.output_dir "$CHALLENGE_OUTPUT_ROOT/run_001"
```

`data.norm_stats_file` 必须指向仅由上述 clean 训练集重算得到的统计文件；不要直接复用来源范围不明或包含 randomized 数据的统计量。

启动前用 `nvidia-smi` 确认无残留进程。训练日志中记录 Git commit、配置副本、随机种子和数据版本，但不记录 token 或密码。
多进程数据加载时应让 `TMPDIR` 指向容器本机磁盘（例如
`/tmp/lingbotvla-train`），避免将 Python multiprocessing 临时目录放在
NFS 共享盘上产生 `.nfs*` 清理警告。

## 6. 推理预检

LoRA checkpoint 中的 `lora_adapter` 不是可独立部署的完整模型。先与训练时 `model.model_path` 指向的**同一份**原始 HF checkpoint 离线合并：

```bash
python tools/merge_lora_adapter.py \
  --base-model "$CHALLENGE_VLA_MODEL_DIR" \
  --adapter "$CHALLENGE_OUTPUT_ROOT/run_001/checkpoints/global_step_<N>/lora_adapter" \
  --output "$CHALLENGE_OUTPUT_ROOT/run_001/checkpoints/global_step_<N>/merged_hf_ckpt"
```

合并工具不会联网，也不会实例化 6B 模型；它按现有 safetensors 分片逐个合并，并校验 adapter 的每个参数都能匹配基座。输出的 `merged_hf_ckpt` 才是可供严格加载的完整权重目录：

```bash
export CHALLENGE_INFER_MODEL_DIR="$CHALLENGE_OUTPUT_ROOT/run_001/checkpoints/global_step_<N>/merged_hf_ckpt"
bash scripts/challenge/preflight.sh --phase infer
bash scripts/challenge/smoke.sh --phase infer
```

模拟器必须使用官方固定提交和独立 Python 3.10 环境，不要复用云机里已有的 `sim` 环境：

```bash
# 显式联网/安装步骤；每个动作都可单独重跑
bash scripts/challenge/setup_robotwin.sh --clone
bash scripts/challenge/setup_robotwin.sh --install-env
bash scripts/challenge/setup_robotwin.sh --download-assets

# 默认仅做只读验收：提交号、Python 依赖、CUDA、Vulkan、资产目录
bash scripts/challenge/setup_robotwin.sh --verify
```

固定的 RoboTwin 提交为 `13c3c47ff4312dd62484bcd51be034af55c062d1`；独立环境位于
`$CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX`。安装脚本不会删除或覆盖其他 conda 环境。
模拟器侧按该提交锁定 Python 3.10、Torch 2.4.1/CUDA 12.1、NumPy 1.26.4 和
cuRobo v0.7.8；容器还必须暴露 NVIDIA `graphics` 能力并通过 Vulkan 检查。
验收通过后，脚本会打印单任务、单回合、BF16 的 4090 smoke 命令。`--episodes 1`
仅验证端到端连通性；正式 clean/randomized 本地评测必须省略该参数，保持默认 100 回合。

4090 上先用 BF16 做管线验证。上游发布成绩采用 FP32 推理，约需 32 GB（还包含模拟器），因此单卡 24 GB 无法保证复现其数值；比赛若要求 FP32，应更换更大显存实例，而不是依赖 OOM 后的自动降级。

## 7. GitHub 维护边界

提交前执行：

```bash
git status --short
git ls-files | grep -E '\.(safetensors|ckpt|pth|pt|onnx|engine|log)$' && \
  echo 'ERROR: large/private artifact is tracked' && exit 1 || true
git diff --check
```

仓库只保存代码、脱敏配置、文档和小型统计文件。权重、原始数据、训练输出与日志保留在共享盘；大文件确需版本化时先确认比赛许可并使用单独的制品仓库或 Git LFS。向 GitHub 推送前再次检查 `git diff --cached`。

## 8. 常见故障

- **预检看到多张 GPU**：检查容器 GPU 配额和 `CUDA_VISIBLE_DEVICES`；本流程按恰好一张可见 4090 设计。
- **系统盘爆满**：确认已 `source .challenge.env`，并用 `python -c 'import os; print(os.environ["HF_HOME"])'` 核对路径。
- **FlashAttention 导入失败**：wheel 的 Python、Torch、CUDA 和 CXX11 ABI 必须一致；不要绕过版本检查。
- **训练立即 OOM**：先确认没有其他 GPU 进程，再降低 micro-batch/序列长度并启用梯度检查点。完整 FP32 模型不适合 24 GB 卡。
- **容器重启后找不到环境**：重新 `source .challenge.env`，初始化 Conda shell，再按环境名激活；环境本体在共享盘。
