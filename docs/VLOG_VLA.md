# VLOG-VLA 设计与实验说明

本文档补充 [README.md](../README.md)，详述方法、六阶段课程、训练坑点与评测协议。

## 1. 方法概览

**VLOG-VLA**（Value-guided Latent Option Graphs for VLA）在冻结/轻量微调的视觉语言动作骨干（StarVLA / QwenOFT）之上，学习一组**离散潜在 option**：

| 模块 | 作用 |
|------|------|
| Posterior encoder + VQ codebook | 从轨迹片段发现 option 原型 |
| Option graph | 状态条件的 option 转移结构 |
| Option critic `Q(s,o)` | option 级价值（非 `Q(s,a)`） |
| Persistent router + termination β | 决定何时保持/切换 option |
| Zero-init / residual FiLM adapter | 将选中的 option 注入 hidden tokens |

推理时仍走原有 action head，输出与基座相同维度的 action chunk。

## 2. 六阶段课程（Stage 1–6）

与 StarVLA + `QwenOFTVLOG` 训练集成的课程：

| Stage | 名称 | 目标 | 典型步数（参考） |
|-------|------|------|------------------|
| 1 | preserve | 适配器保底（zero-init 时可能近似空转） | 短 |
| 2 | option discovery | VQ + distill + balance，发现 codebook | 较长 |
| 3 | graph | 转移图 / 稀疏边 | **需保护 codebook** |
| 4 | critic | 拟合 `Q(s,o)` | 中等 |
| 5 | router | action + router + critic | 中等 |
| 6 | full | 联合 + termination | 较长，定期存 ckpt |

训练入口示例：

- LIBERO：`examples/simBenchmarks/LIBERO/train_files/` 或 `examples/LIBERO/train_files/`
- RoboCasa GR1：`examples/simBenchmarks/Robocasa_tabletop/train_files/`

## 3. 关键实现约束（RoboCasa 踩坑总结）

### 3.1 Stage3 不可改写 codebook

早期实现中 `OptionGraphLayer` 对 `codebook.weight` 求梯度，RoboCasa Stage3 可把 `vq_loss` 从 ~0 拉到数百，option 空间塌缩。  
**修复**：graph 内对 option codes `detach`（见 `starVLA/model/vlog_vla/option_graph_layer.py`）。

### 3.2 Adapter 不可反向放大基座

错误形式近似 `h + α·((1+γ)h+β)`，α 为负时会削弱基座。  
**修复**：`h + α·(γ⊙h+β)`，`α = clamp(α, min=0)`。

### 3.3 与 OFT 基座的 state 协议一致

Qwen3-VL-OFT-Robocasa 基座训练/评测使用 **不向 VLM 注入 state token**。  
VLOG RoboCasa 建议 `include_state: false`，评测加 `--args.no_send_state`。

### 3.4 Early-stop 勿只盯 `action_loss`

Stage6 若仅用 `vlog/action_loss` 早停，易在 α 塌缩、critic 发散时停在坏 ckpt。  
建议：定期存盘 + 监控 `vq_loss` / `adapter_alpha` / NaN，必要时人工选 ckpt。

### 3.5 LR 配置键

勿传嵌套 CLI：`--trainer.learning_rate.vlog.critic`（会变成 `AccessTrackedConfig`，触发 LambdaLR `TypeError`）。用扁平的 `learning_rate.vlog`。

## 4. 指标含义

| 指标 | 含义 | 健康迹象 |
|------|------|----------|
| `vlog/vq_loss` | 向量量化误差 | Stage2/3 后应保持较小；突然飙升=码本被破坏 |
| `vlog/adapter_alpha` | option 注入强度 | ≥0；不应长期贴零或变负 |
| `vlog/mean_beta` | 终止概率均值 | 过早塌到 ~0 会导致几乎不切换 |
| `vlog/option_entropy` | option 分布熵 | 过高≈均匀；需结合 vq 一起看 |
| `vlog/action_loss` | 动作 L1 | 有限且正；单独下降≠评测一定更好 |

## 5. 评测

### LIBERO

```bash
bash examples/LIBERO/eval_files/run_vlog_policy_server.sh
bash examples/LIBERO/eval_files/eval_vlog_libero.sh
```

### RoboCasa GR1 tabletop（6 核心 PnP）

并行 suite（每任务独立 policy server）：

```bash
CKPT=/path/to/ckpt.pt \
RUN_ID=my_run \
N_PARALLEL=6 N_ENVS=1 FILL_WORKERS=1 \
EXTRA_EVAL_ARGS='--args.no_send_state' \
bash examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_eval_suite.sh
```

说明：

- VLOG persistent inference 要求 **batch size = 1**，故 `N_ENVS` 保持 1。
- 吞吐靠 **多任务 / episode 分片并行**（`N_PARALLEL` + `FILL_WORKERS`）。

## 6. 重训脚本（RoboCasa V2）

```bash
# 从健康 Stage2 起，安全 Stage3 →4→5→6
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_vlog_robocasa_retrain_v2.sh

# 若 Stage3 已完成，仅从 Stage4 续跑
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_vlog_robocasa_retrain_v2_from_s4.sh
```

推荐单卡大显存设置：`PER_DEVICE_BATCH_SIZE=16~24`，`USE_DEEPSPEED=0`。

## 7. 模块文件索引

```text
starVLA/model/vlog_vla/
  latent_option_codebook.py   VQ codebook
  option_graph_layer.py       转移图（codes detach）
  option_adapter.py           residual FiLM, α≥0
  option_critic.py            Q(s,o)
  persistent_option_router.py
  termination_head.py
  vlog_policy_wrapper.py      训练/推理封装
  qwen_oft_vlog.py            损失与 stage 课程
  losses.py                   CQL / transition 等

starVLA/model/framework/VLM4A/QwenOFTVLOG.py
  框架注册名 QwenOFTVLOG
```

## 8. 不入库内容

权重、数据集、训练日志与视频体积过大，已由 `.gitignore` 排除。需要自行准备：

- Qwen3-VL-4B-Instruct
- StarVLA / OFT 任务微调 ckpt（LIBERO 或 RoboCasa）
- 对应 LeRobot / GR00T Sim 数据
