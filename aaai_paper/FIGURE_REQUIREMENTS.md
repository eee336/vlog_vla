# VLOG-VLA 论文图片需求清单

本清单对应 `paper/` 中的占位图。最终优先使用矢量 PDF/SVG；截图和仿真帧使用
PNG。所有实验图必须保留生成脚本和原始 JSON/JSONL，不能从聊天文字手工抄数。

## 总体视觉规范

- 双栏通栏宽度：约 7.0--7.2 inch。
- 单栏宽度：约 3.3--3.45 inch。
- 字体最终打印尺寸不小于 7.5 pt。
- 冻结/预训练模块：灰色。
- 共享可训练模块：蓝色。
- embodiment-specific adapters：紫色。
- training-only posterior：橙色虚线。
- persistent controller state：绿色。
- 错误/负对照：红色轮廓，不大面积填红。
- option 颜色在所有图中保持一致；不要给 option 写“抓取”“放置”等语义名，
  除非有独立标注证据。

---

## F1：UniversalVLOG 总体结构图

**论文位置：** Method 开头，双栏通栏。
**目标文件：** `paper/figures/vlog_universal_architecture.pdf`
**建议尺寸：** 7.1 x 3.2 inch。

### 必须画出的内容

1. 输入：
   - 多视角或单视角 RGB；
   - language instruction；
   - native robot state；
   - embodiment ID。
2. 共享 Qwen3-VL backbone，输出 VL hidden tokens。
3. native adapter bank：
   - RoboCasa-GR1：58D state / 29D action；
   - LIBERO：7D state / 7D action；
   - 直接进入 shared DiT token space；
   - 明确禁止画成 LIBERO 7D → GR1 29D physical space。
4. shared GR00T-style flow-matching DiT。
5. 训练专用路径：
   - demonstrated future native actions；
   - native ActionEncoder；
   - posterior；
   - VQ codebook。
6. 部署路径：
   - current-context router；
   - per-environment semi-Markov controller；
   - active option。
7. option 注入：
   - explicit option token；
   - residual FiLM on noisy action tokens。
8. native ActionDecoder 输出 action chunk。
9. 用醒目标注写明：
   - graph / critic / learned termination：not used；
   - one shared Qwen + one shared DiT。

### 需要用户提供

- 最终实际训练配置的 resolved YAML。
- 如果最终 embodiment 不止 RoboCasa/LIBERO，请给出名称和 state/action 维度。
- 不需要提供照片；该图可以直接根据代码绘制。

---

## F2：Option-specific conditioning 与配对干预图

**论文位置：** Method / Option-Specific Action Conditioning，双栏通栏。
**目标文件：** `paper/figures/vlog_option_conditioning.pdf`
**建议尺寸：** 7.1 x 2.6 inch。

### 必须画出的内容

- Panel (a)：option code → bias-free option projection →
  bounded gamma/beta + option token。
- 标出：
  - final projections zero initialized；
  - fixed `rho=0.05`；
  - state/embodiment/bias bypass removed。
- Panel (b)：DiT token sequence：
  `[state, optional embodiment, option, future queries, noisy actions]`。
- Panel (c)：同一 observation、target、mask、flow time、noise 分叉成：
  - fusion-off base；
  - correct/oracle；
  - guaranteed wrong；
  - shuffled；
  - hard router。
- 每个分支标注相应 FM loss。

### 需要用户提供

- 不需要实验截图。
- 只需确认最终 `rho`、是否启用 `bound_conditioning_outputs`、是否对 base
  embodiment 插入 embodiment token。

---

## F3：Semi-Markov 持久化状态机

**论文位置：** Method / Persistent Controller，单栏。
**目标文件：** `paper/figures/vlog_persistent_controller.pdf`
**建议尺寸：** 3.4 x 2.7 inch。

### 必须画出的内容

- episode reset；
- initial router argmax；
- `age < d_min` 强制保持；
- `d_min <= age < d_max` 时由 router-logit hysteresis 决定；
- `age >= d_max` 重新选择，但允许选择回当前 option；
- switch 后 age 清零；
- 分开标记：
  - decision duration；
  - environment-step duration。

### 需要用户提供

- 最终 `d_min`、`d_max`、hysteresis。
- 评测时每次 policy query 实际执行多少 environment actions。

---

## F4：训练阶段与 Gate 流程图

**论文位置：** Method 结尾，双栏通栏。
**目标文件：** `paper/figures/vlog_training_gates.pdf`
**建议尺寸：** 7.1 x 2.4 inch。

### 必须画出的内容

横向阶段：

```text
U0-base → U0-native → U1-oracle → Gate O
        → U2-router → Gate R → U3-joint → paired rollout
```

每个阶段下方画：

- trainable modules；
- frozen modules；
- active losses；
- 产物：
  resolved config、checkpoint、gate JSON、episode JSONL、video、profiler。

明确区分：

- offline Gate O/R；
- simulator rollout evidence；
- engineering/unit tests。

### 需要用户提供

- 实际采用的训练阶段顺序。
- 每阶段真实步数、LR、batch、checkpoint 路径。
- 如果还未运行，可先只画结构，不填数值。

---

## F5：Gate O / Gate R 定量图

**论文位置：** Experiments / Offline Gates，双栏通栏。
**目标文件：** `paper/figures/vlog_gate_diagnostics.pdf`
**建议尺寸：** 7.1 x 3.1 inch。

### 建议面板

1. base / correct / wrong / shuffle / router paired FM loss。
2. wrong-minus-correct 与 shuffle-minus-correct relative gap，按 embodiment。
3. oracle/router option usage histogram。
4. router accuracy、router-oracle loss gap。
5. residual ratio、option token norm、velocity/action intervention magnitude。

### 必须提供的原始文件

- 每个 U1 candidate checkpoint 的 `evaluate_universal_gates.py` JSON。
- 每个 U2 candidate checkpoint 的相同 JSON。
- 至少分别包含 RoboCasa 与 LIBERO；不能只给全局平均。
- resolved threshold/config 文件。
- checkpoint selection 规则。

### 不接受

- 只有训练日志截图；
- 手工整理的均值；
- 没有 base/correct/wrong 共享 flow time/noise 的结果；
- 旧 Qwen-OFT Gate 或旧 Stage5 critic 指标。

---

## F6：配对 rollout + option timeline

**论文位置：** Experiments / Qualitative Evidence，双栏通栏。
**目标文件：** `paper/figures/vlog_rollout_evidence.pdf`
**建议尺寸：** 7.1 x 3.2 inch。

### 必须包含

- 同一 task、initial state、episode seed：
  - fusion-off base；
  - per-query option；
  - persistent VLOG。
- 每种方法 4--6 个对齐关键帧。
- option ID 色带。
- router candidate。
- decision age。
- switch reason。
- action-chunk query 边界。
- 最终 success/failure。
- 一个 VLOG recovery 和一个 VLOG-specific failure；若没有明确原因，写
  `unattributed`。

### 必须提供的原始文件

- 三种方法的原始 MP4。
- 每个 policy query 的 JSONL，至少包含：
  - task；
  - seed / initial state ID；
  - query index；
  - environment timestep；
  - active option；
  - router candidate；
  - option age；
  - switched；
  - switch reason；
  - native action chunk；
  - episode result。
- 视频与 JSONL 的 manifest。

---

## A1：完整 option timeline

**论文位置：** Appendix，双栏通栏。
**目标文件：** `paper/figures/vlog_full_option_timeline.pdf`

展示一个完整 episode，而不是只选几帧：

- observation frame strip；
- option/candidate；
- router confidence 与 logit margin；
- decision age；
- option-token norm；
- FiLM residual ratio；
- native action；
- episode reset 与 outcome。

所需原始文件与 F6 相同。

---

## A2：失败案例矩阵

**论文位置：** Appendix，双栏通栏。
**目标文件：** `paper/figures/vlog_failure_taxonomy.pdf`

建议类别：

- perception error；
- base action error；
- option collapse；
- wrong persistent option；
- oscillatory router；
- delayed reselection；
- execution failure；
- unattributed。

每行提供 fusion-off 与 VLOG 的 paired video/frame、option trace 和第一处可验证失败。

### 需要用户提供

- 预先定义的 failure sampling rule。
- paired episode videos。
- option/action timeline JSONL。
- 人工标注 CSV，包含标注者、类别、第一失败 timestep、置信度。

---

## 可选结果图

如果主结果完成且版面允许，可增加：

1. `success_delta_by_task.pdf`：VLOG 相对 fusion-off 的 per-task paired
   success difference 与置信区间。
2. `cross_embodiment_usage.pdf`：按 embodiment 的 option usage 与
   code--embodiment mutual information。
3. `performance_overhead.pdf`：success、latency、memory 的折中图。

这些图只有在对应机器可读结果齐全时再加，不预留主文版面。

---

## 请一次性给我的最小图片数据包

```text
1. final resolved YAML
2. U1 Gate JSON（所有候选 checkpoint，RoboCasa + LIBERO）
3. U2 Gate JSON（所有候选 checkpoint，RoboCasa + LIBERO）
4. main/ablation per-episode evaluation JSONL
5. option timeline JSONL
6. paired rollout MP4
7. evaluation manifest（task、seed、checkpoint、config、commit）
8. profiler JSON
```

拿到这些文件后，可以将 F1--F4 绘制为最终矢量结构图，将 F5/F6/A1/A2
替换为真实数据图，并自动填充论文表格。
