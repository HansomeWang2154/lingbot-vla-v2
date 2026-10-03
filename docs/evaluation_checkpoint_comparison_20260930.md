# Clean checkpoint 对照报告

状态：已完成。核验日期：2026-10-03（北京时间）。
实际运行：2026-09-30 16:02:38—22:25:38（北京时间），总墙钟约 6 小时 23 分钟。

## 配置与完整性

RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1`；每模型 50 tasks × 1 episode，
共 200 回合，所有阶段 complete，均 50 done / 0 skipped，逐任务完整回合。
同一 clean 10k YAML、clean 动作归一化、BF16、关闭 compile/video、chunk 50，
simulator seed block 0、policy RNG seed 42。代码版本 bf12b11。

本实验是固定 clean 归一化下的权重对照，不是官方 FP32 设置或官方发布模型原生配置复现。
基座训练数据来源与比赛许可边界见 [rollout 路线](checkpoint_comparison_and_rollouts.md)。
完整云端清单的归档见 [逐任务数据及校验值](results/checkpoint_comparison_clean_20260930.json)。

## 总体结果

| 模型 | 成功/总回合 | 成功率 | 相对基座 | 仿真阶段耗时 |
|---|---:|---:|---:|---:|
| base_50k | 19/50 | 38.0% | — | 5540 s |
| lora_1000 | 15/50 | 30.0% | -8.0 pp | 5555 s |
| lora_5000 | 20/50 | 40.0% | 2.0 pp | 5396 s |
| lora_10000 | 21/50 | 42.0% | 4.0 pp | 5441 s |

结论：10k 是当前初筛候选，但仅比基座多成功 2 个任务（+4 个百分点）。
1k 低于基座；5k 与 10k 只差 1 个任务，不支持精确排名或稳定泛化增益的结论。
单任务仅一个场景，且模型沿用同一场景 seed，不能把 50 个不同任务当作某单一任务
的 50 次独立重复。旧客户端未保存指令审计；仅知道启动 seed 一致，不能完全核对
专家跳 seed 后的实际场景/语言是否一致。新一轮添加 episode 审计补齐这一限制。

## 基座与 10k 的差异

从失败转成功（7 个）：`beat_block_hammer`、`place_fan`、`place_mouse_pad`、`place_object_scale`、`place_phone_stand`、`pick_diverse_bottles`、`pick_dual_bottles`。

从成功转失败（5 个）：`blocks_ranking_rgb`、`dump_bin_bigbin`、`place_empty_cup`、`place_object_basket`、`place_object_stand`。

共同成功 14 个，共同失败 24 个。不能根据这种任务级变化直接宣称遗忘或过拟合；
需要多场景重复、视频或动作分布证据。

## 下一轮固定诊断集

基座 vs 10k，每模型 10 tasks × 10 episodes，共 200 回合；seed block 1（从 200000
开始，专家不可解场景会跳过）。使用同样 BF16/归一化/chunk 设置，不增加训练数据。
成功/失败、实际 seed、指令逐 episode 写入 episodes.jsonl；汇总时核对两模型场景和指令
一致，否则停止并报告配对无效。诊断集按已观察差异选择，具有选择偏差，不是正式全任务成绩。

| 分组 | 任务 |
|---|---|
| 改善 | beat_block_hammer、place_mouse_pad、pick_dual_bottles |
| 退化 | blocks_ranking_rgb、place_empty_cup、place_object_basket |
| 共同成功 | grab_roller、open_laptop |
| 共同失败 | lift_pot、stack_blocks_three |

诊断结果用于确定错误类型和下一步实验，不据此改 randomized/隐藏评测设置。
仍不采集训练 rollout；待确认比赛允许新增仿真数据后，另开采集 seed 与训练/验证划分。

## 全部任务初筛结果

1=成功，0=失败；每格仅 1 episode。

| Task | Base | 1k | 5k | 10k |
|---|---:|---:|---:|---:|
| lift_pot | 0 | 0 | 1 | 0 |
| hanging_mug | 0 | 0 | 0 | 0 |
| stack_bowls_three | 0 | 0 | 0 | 0 |
| scan_object | 0 | 0 | 0 | 0 |
| handover_block | 0 | 0 | 0 | 0 |
| click_bell | 0 | 0 | 0 | 0 |
| put_object_cabinet | 0 | 0 | 0 | 0 |
| open_microwave | 0 | 0 | 1 | 0 |
| stack_blocks_three | 0 | 0 | 0 | 0 |
| place_shoe | 0 | 0 | 0 | 0 |
| adjust_bottle | 0 | 0 | 0 | 0 |
| beat_block_hammer | 0 | 0 | 1 | 1 |
| blocks_ranking_rgb | 1 | 0 | 0 | 0 |
| blocks_ranking_size | 0 | 0 | 0 | 0 |
| click_alarmclock | 0 | 0 | 0 | 0 |
| dump_bin_bigbin | 1 | 0 | 0 | 0 |
| grab_roller | 1 | 1 | 1 | 1 |
| handover_mic | 1 | 0 | 1 | 1 |
| move_can_pot | 0 | 0 | 0 | 0 |
| move_pillbottle_pad | 0 | 0 | 0 | 0 |
| move_playingcard_away | 1 | 1 | 1 | 1 |
| place_cans_plasticbox | 0 | 0 | 0 | 0 |
| place_container_plate | 1 | 1 | 1 | 1 |
| place_dual_shoes | 0 | 1 | 0 | 0 |
| place_empty_cup | 1 | 0 | 1 | 0 |
| place_fan | 0 | 0 | 0 | 1 |
| place_mouse_pad | 0 | 1 | 1 | 1 |
| place_object_basket | 1 | 0 | 0 | 0 |
| place_object_scale | 0 | 1 | 1 | 1 |
| place_object_stand | 1 | 0 | 0 | 0 |
| place_phone_stand | 0 | 1 | 1 | 1 |
| move_stapler_pad | 0 | 0 | 0 | 0 |
| open_laptop | 1 | 1 | 1 | 1 |
| pick_diverse_bottles | 0 | 0 | 1 | 1 |
| pick_dual_bottles | 0 | 0 | 0 | 1 |
| place_a2b_left | 1 | 1 | 0 | 1 |
| place_a2b_right | 1 | 1 | 1 | 1 |
| place_bread_basket | 1 | 1 | 1 | 1 |
| place_bread_skillet | 1 | 1 | 0 | 1 |
| place_burger_fries | 1 | 0 | 0 | 1 |
| place_can_basket | 0 | 0 | 0 | 0 |
| press_stapler | 0 | 0 | 0 | 0 |
| rotate_qrcode | 0 | 0 | 1 | 0 |
| shake_bottle_horizontally | 1 | 0 | 1 | 1 |
| shake_bottle | 1 | 1 | 1 | 1 |
| stack_blocks_two | 0 | 0 | 0 | 0 |
| stack_bowls_two | 1 | 1 | 1 | 1 |
| stamp_seal | 0 | 0 | 0 | 0 |
| turn_switch | 1 | 1 | 1 | 1 |
| put_bottles_dustbin | 0 | 0 | 0 | 0 |
