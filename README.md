# VLOG-VLA

**Persistent Value-Guided Latent Option Graphs for Vision-Language-Action Models**

在 [StarVLA](https://github.com/starVLA/starVLA) / QwenOFT 视觉语言动作骨干之上，学习可持久化的**潜在 option 图**：离散 option、option 级价值 `Q(s,o)`、状态条件转移，以及零初始化残差 FiLM 适配器，在不改写原有 action head 接口的前提下提升长时程操作策略。

| 项 | 说明 |
|----|------|
| 仓库 | https://github.com/eee336/vlog_vla |
| 框架名 | `QwenOFTVLOG` |
| 主要基准 | LIBERO · RoboCasa GR1 Tabletop |
| 详细设计 | [docs/VLOG_VLA.md](docs/VLOG_VLA.md) |
| Benchmark 路径 | [docs/vlog_benchmark_setup.md](docs/vlog_benchmark_setup.md) |

---

## 目录

1. [方法一览](#方法一览)
2. [仓库结构](#仓库结构)
3. [环境安装](#环境安装)
4. [数据与权重（需自备）](#数据与权重需自备)
5. [训练：六阶段课程](#训练六阶段课程)
6. [评测](#评测)
7. [核心模块](#核心模块)
8. [稳定性修复说明](#稳定性修复说明)
9. [常见问题](#常见问题)
10. [引用与许可](#引用与许可)

---

## 方法一览

```text
image + language (+ optional state)
                 |
                 v
     StarVLA / QwenOFT backbone
                 |
                 v
           hidden tokens H
                 |
        +--------+--------+
        |                 |
        v                 v
  latent options     option critic Q(s,o)
  (VQ codebook)      + router / terminate beta
        |
        v
 residual FiLM adapter  (alpha >= 0)
                 |
                 v
           H' -> action head -> action chunk
```

要点：

- Option **不是**人工技能标签或语言子目标，而是从轨迹中经 VQ 学出的离散潜变量。
- Critic 是 **option 级** `Q(s,o)`，不是 `Q(s,a)`。
- 适配器对冻结骨干做轻量残差注入；推理仍输出与基座相同格式的动作块。
- Policy server 与官方 StarVLA 评测客户端兼容，并可透传 `vlog_info`（option 时间线）。

---

## 仓库结构

```text
starVLA/
  model/vlog_vla/          # VLOG 核心（codebook / graph / adapter / critic / losses）
  model/framework/         # QwenOFTVLOG 等框架封装
  training/                # train_starvla 训练入口
  dataloader/              # LeRobot 等数据管线
configs/vlog_vla/          # 分阶段 YAML / 消融
scripts/vlog_vla/          # 探针、分段训练与分析脚本
examples/
  LIBERO/                  # 精简 train/eval shell
  simBenchmarks/
    LIBERO/                # 完整 LIBERO 基准脚本
    Robocasa_tabletop/     # GR1 桌面 PnP 训练与并行评测
deployment/model_server/   # Websocket 策略服务
docs/                      # 设计说明与安装指南
tests/                     # VLOG 单元 / 集成测试
```

> **未入库**：预训练权重、数据集、训练 `results/`、评测 `outputs/`、本地 `playground/` 大文件。见 `.gitignore`。

---

## 环境安装

### 前提

- 已有可用的 CUDA + PyTorch（与 StarVLA 一致的版本）
- Conda 环境建议命名：`vlog_vla`
- 仿真评测另需对应环境（如 LIBERO / `robocasa`）

### 克隆与安装

```bash
git clone https://github.com/eee336/vlog_vla.git
cd vlog_vla

conda activate vlog_vla
pip install -r requirements.txt
pip install -e .
# 或开发依赖
pip install -e ".[dev]"
```

本仓库**不捆绑** `torch` 安装；请继续使用本机已验证可跑 StarVLA 的 CUDA 轮子。

### 本机路径

```bash
cp .vlog_vla.env.example .vlog_vla.env
# 编辑 .vlog_vla.env：CKPT / DATA_ROOT / VLM 路径等
```

检查基准依赖：

```bash
bash tools/setup_vlog_benchmarks.sh check
```

### 快速自测

```bash
pytest tests/test_vlog_vla_modules.py -q
```

---

## 数据与权重（需自备）

| 用途 | 示例路径（本地惯例，勿提交 Git） |
|------|----------------------------------|
| VLM 底模 | `playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| LIBERO OFT ckpt | `playground/Pretrained_models/.../steps_*_pytorch_model.pt` |
| RoboCasa OFT ckpt | `.../Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt` |
| LIBERO 数据 | LeRobot / 项目约定 JSONL |
| RoboCasa GR1 数据 | `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` + `fourier_gr1_unified_1000` |

仿真资产（RoboCasa Objectjaverse / textures 等）按官方文档放到 `robocasa` 包内 assets，**不要**提交到本仓库。

---

## 训练：六阶段课程

| Stage | `train_stage` | 目标 |
|------:|---------------|------|
| 1 | `stage1_preserve` | 适配器保底 |
| 2 | `stage2_option_discovery` | VQ + option 发现 |
| 3 | `stage3_graph` | 转移图（**勿破坏 codebook**） |
| 4 | `stage4_critic` | option critic |
| 5 | `stage5_router` | router + action |
| 6 | `stage6_full` | 联合 + termination |

### LIBERO（示例）

```bash
bash examples/LIBERO/train_files/run_vlog_libero_train.sh
# 或分 stage：run_vlog_libero_train_stage{1..6}.sh
```

配置参考：`examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml`。

### RoboCasa GR1（示例）

```bash
# 标准流水线
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_vlog_robocasa_stage_pipeline.sh

# 修复后的 V2 重训（从健康 Stage2 / Stage3 起，大 batch）
PER_DEVICE_BATCH_SIZE=24 \
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_vlog_robocasa_retrain_v2.sh
```

配置：`examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwen_oft_vlog_robocasa.yaml`  
推荐：`include_state: false`（对齐 OFT 基座）、单卡 `batch=16~24`。

更多坑点与指标含义见 [docs/VLOG_VLA.md](docs/VLOG_VLA.md)。

---

## 评测

### LIBERO

```bash
bash examples/LIBERO/eval_files/run_vlog_policy_server.sh
bash examples/LIBERO/eval_files/eval_vlog_libero.sh
```

### RoboCasa 六任务并行套件

```bash
CKPT=/abs/path/to/steps_XXXX_pytorch_model.pt \
RUN_ID=VLOG_ROBOCASA_EVAL \
OUTPUT_ROOT=outputs/robocasa_eval \
N_EPISODES=20 N_PARALLEL=6 N_ENVS=1 FILL_WORKERS=1 \
EXTRA_EVAL_ARGS='--args.no_send_state' \
bash examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_eval_suite.sh
```

说明：

- VLOG 持久 option 推理要求 **单环境 batch=1**；加速靠多 process / episode 分片。
- OFT 基线同样加 `--args.no_send_state`。

---

## 核心模块

| 文件 | 职责 |
|------|------|
| `latent_option_codebook.py` | VQ 码本 |
| `option_graph_layer.py` | 状态条件图（codes detach） |
| `option_adapter.py` | 残差 FiLM，`α≥0` |
| `option_critic.py` | `Q(s,o)` |
| `persistent_option_router.py` | 路由 |
| `termination_head.py` | β 终止 |
| `vlog_policy_wrapper.py` | 训练 / 推理封装 |
| `qwen_oft_vlog.py` | 分 stage 损失汇总 |
| `losses.py` | CQL / transition / sparse 等 |

框架注册：`starVLA/model/framework/VLM4A/QwenOFTVLOG.py` → `framework.name=QwenOFTVLOG`。

---

## 稳定性修复说明

相对早期流水线，本仓库已合入下列修复（详见设计文档）：

1. **Stage3 codebook 保护**：graph 前向对 option codes `detach`，避免 `vq_loss` 爆炸。  
2. **Adapter 公式**：`H' = H + α·(γ⊙H+β)`，α 截断非负。  
3. **CQL / NaN 防护**：loss 侧 clamp / `nan_to_num` / 连续 NaN 早停接口。  
4. **RoboCasa 与 OFT 对齐**：默认不向 VLM 拼 state；评测 `no_send_state`。  
5. **并行评测脚本**：单卡多 server，避免串行评测空转 GPU。  
6. **配置陷阱**：禁止使用破损的嵌套 LR CLI（如 `learning_rate.vlog.critic`）。

---

## 常见问题

**Q: `ModuleNotFoundError: starVLA`**  
确认在仓库根目录执行了 `pip install -e .`，或设置 `PYTHONPATH=$PWD`。

**Q: 仓库里没有 `.pt` 权重？**  
故意忽略。请从 StarVLA / HuggingFace / 自有训练目录准备。

**Q: GitHub 页面曾经是空的？**  
请拉取默认分支（`main` 或 `feature/vlog-vla-stage7-real-starvla`）。若网络不稳可用：

```bash
git clone --depth 1 https://github.com/eee336/vlog_vla.git
```

**Q: VLOG 评测低于 OFT 基线？**  
先检查 Stage3 `vq_loss`、`adapter_alpha` 是否健康，以及评测是否误开了 state。参考 [docs/VLOG_VLA.md](docs/VLOG_VLA.md) §3。

---

## 引用与许可

- 上游 StarVLA 请遵循其原仓库许可与引用要求。  
- 本仓库叠加代码见根目录 [LICENSE](LICENSE)。  
- 若需引用本工作，可参考 [CITATION.cff](CITATION.cff)（若已提供）。

---

## English Summary

VLOG-VLA adds a **latent option graph** on top of StarVLA/QwenOFT: VQ options, `Q(s,o)` critic, persistent routing, and a residual FiLM adapter. Train with a 6-stage curriculum; evaluate on LIBERO and RoboCasa GR1. Weights and datasets are **not** shipped—configure local paths via `.vlog_vla.env`. See [docs/VLOG_VLA.md](docs/VLOG_VLA.md) for design details, failure modes, and RoboCasa V2 retrain notes.
