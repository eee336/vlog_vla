# VLOG-VLA

面向 RoboCasa、LIBERO 与后续真机的统一 latent-option VLA。当前主线框架是
`QwenUniversalVLOG`：以官方 Qwen3-VL-GR00T RoboCasa policy 为可靠初始化，
保留一个共享 flow-matching DiT，并在它的原生 state/action 边界加入小型
embodiment adapters。Option 直接进入动作生成路径，而不是修改 VLM hidden 的
旁路信号。

| 项 | 当前主线 |
|---|---|
| Framework | `QwenUniversalVLOG` |
| VLM | Qwen3-VL-4B |
| Action expert | shared GR00T flow-matching DiT |
| Base embodiment | RoboCasa GR1：state 58D / action 29D / horizon 16 |
| Additional embodiment | LIBERO：state 7D / action 7D；真机按 native contract 注册 |
| Option | shared VQ codebook + posterior + router + explicit option token + action-token FiLM |
| Persistence | per-environment `d_min/d_max` + router hysteresis |
| 当前不开启 | graph、critic、RL/CQL、learned termination |

完整的真实代码路径审计见
[docs/QWEN_UNIVERSAL_VLOG_SOURCE_AUDIT.md](docs/QWEN_UNIVERSAL_VLOG_SOURCE_AUDIT.md)。

## 模型数据流

```text
image + language -> shared Qwen3-VL -> H_vl ----------------------┐
                                                                  |
native state(e)  -> StateAdapter[e] -> D_dit state tokens --------+
native action(e) -> ActionEncoder[e] -> noisy D_dit action tokens-+
                                                                  v
state + embodiment + option + future queries + noisy action -> shared DiT
                                                                  |
                                     ActionDecoder[e] <- D_dit output
                                                                  |
                                           native normalized velocity/action
```

RoboCasa base 的 `state_encoder`、`action_encoder`、DiT 和 `action_decoder` 保留
原始参数名。`fusion_enabled=false` 时，它仍使用原始
`[state, future_query, noisy_action]` 序列，以便与官方 GR00T checkpoint 做逐元素
parity。LIBERO 7D 动作直接进入共享 `D_dit`，不会先映射到 GR1 的 29D 物理坐标。

Option conditioner 产生：

1. 一个显式 option token；
2. 对 noisy action tokens 的 residual FiLM。

当前 repair-mode conditioner 的输出只能由 option code 产生；state、embodiment
和 projection bias 不能形成 code-independent adapter。RoboCasa base 也不再因为
开启 option 而额外插入常量 embodiment token。FiLM beta/token 输出有界，日志报告
的是乘过 `rho` 后实际加入 action token 的 residual 及其相对 action-token norm。

正确、base 与 wrong-option FM 比较复用同一 observation、target、action mask、
`t` 和 noise。Posterior 只在训练期看 future action；部署只使用 router 和持久化
controller。

## Checkpoint 原则

首选初始化：

```text
StarVLA/Qwen3-VL-GR00T-Robocasa-gr1/checkpoints/steps_90000_pytorch_model.pt
```

必须同时使用与该 checkpoint 匹配的 tokenizer、Qwen 和 action-head 配置。新
framework 会覆盖 trainer 的宽松 `strict=False` 行为：如果 Qwen embedding 或任一
GR00T 核心键缺失/shape mismatch，加载直接失败；只有新增的 UniversalVLOG 模块
允许随机初始化。

不能加载 QwenOFT、旧 QwenOFTVLOG 或 scratch QwenPI checkpoint。

## 训练阶段

| Stage | 可训练参数 | 晋级前必须验证 |
|---|---|---|
| `u0_base` | 无；只诊断/评测 | checkpoint strict load、fusion-off parity、normalization round-trip、RoboCasa Core-6 base |
| `u0_libero` | LIBERO native adapters + embodiment embedding | LIBERO base rollout；同时检查 RoboCasa 遗忘 |
| `u1_oracle` | aggregator、posterior、codebook、motion decoder、option conditioner | 分 embodiment Gate O；router 关闭 |
| `u2_router` | **仅 router** | 分 embodiment Gate R；conditioner 和 action expert 固定 |
| `u3_joint` | shared DiT、native adapters、VLOG、router；Qwen 冻结 | VLOG 对同一 unified base 非劣 |

`u0_base` 不是训练 stage，启动脚本会拒绝对全冻结模型执行 backward。

U2 的监督目标只是 posterior/codebook 产生的 oracle option ID。普通 U2 step
不会执行 DiT；每 `router_fm_diagnostics_every` 步才使用部署时的 hard routed
option 做一次 paired base/router FM 诊断。`repeated_diffusion_steps` 在 U2
自动按 1 处理，避免复制相同 router 标签。

## 安装与 CPU 测试

本仓库不捆绑 CUDA/PyTorch 轮子。先使用已验证可运行 StarVLA 的环境：

```bash
git clone https://github.com/eee336/vlog_vla.git
cd vlog_vla
pip install -r requirements.txt
pip install -e ".[dev]"

STARVLA_USE_DEEPSPEED=0 pytest -q
```

## U0 strict load 与 action parity

```bash
python scripts/vlog_vla/diagnose_universal_base.py \
  --config examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwen_universal_vlog_robocasa.yaml \
  --checkpoint /absolute/path/to/steps_90000_pytorch_model.pt \
  --device cuda
```

该命令只验证 checkpoint contract 和缓存输入下的 action-head parity，不会声称
完成了 GPU backward、normalization 审计或 simulator rollout。

## 训练命令

RoboCasa U1/U2/U3 使用同一配置和脚本，通过 stage 与 checkpoint 串接：

```bash
BASE_CKPT=/absolute/path/to/compatible_checkpoint.pt \
RUN_ID=qwen_universal_vlog_robocasa_u1 \
TRAIN_STAGE=u1_oracle \
BATCH_SIZE=8 GRAD_ACCUM=1 MAX_STEPS=30000 \
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_qwen_universal_vlog.sh
```

`GRAD_ACCUM` 现在会在创建 `Accelerator` 时生效，并与保存的 config 做一致性
检查。日志中的 `Total observation batch size` 才是独立 observation 数；
`repeated_diffusion_steps` 是每个 observation 的 FM noise draws，不能再次算作
独立 batch。

## Offline Gate O/R

U1/U2 晋级不得只看训练日志中的标量均值。对每个候选 checkpoint 使用固定
audit subset，复用完全相同的 observation、target、mask、FM `t` 和 noise，执行
base/correct/wrong/shuffle/hard-router paired intervention：

```bash
python scripts/vlog_vla/evaluate_universal_gates.py \
  --checkpoint /absolute/path/to/steps_30000_pytorch_model.pt \
  --output outputs/universal_gates/u1_steps_30000.json \
  --batch-size 16 --num-batches 32 --fm-repeats 2 \
  --device cuda
```

报告只具有 offline E2 证据级别，不替代 Core-6 rollout。旧的
`scripts/vlog_vla/analyze_options.py` 会生成合成占位序列，不能用于 UniversalVLOG
Gate 或论文证据。

### U1 conditioner repair

若 legacy U1 出现“correct 显著优于 base，但 wrong/shuffle 几乎等于 correct”，
不要降低 Gate 或直接蒸馏 router。这表明 conditioner 学成了公共 adapter。保留
posterior/codebook，显式重置 conditioner 后运行短 repair curriculum：

```bash
BASE_CKPT=/absolute/path/to/legacy_u1_checkpoint.pt \
RUN_ID=qwen_universal_vlog_robocasa_u1_option_repair \
TRAIN_STAGE=u1_oracle \
REINITIALIZE_MODULES=action_model.option_conditioner \
MAX_STEPS=10000 BATCH_SIZE=16 GRAD_ACCUM=1 \
SAVE_INTERVAL=2000 EVAL_INTERVAL=1000000 NUM_WARMUP_STEPS=500 \
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_qwen_universal_vlog.sh
```

Repair 配置使用 option-only conditioner、bounded FiLM、随机 wrong code、以 base
FM 为尺度的 5% relative counterfactual margin，以及 codebook cosine separation。
不要同时增加 `num_options` 或 `fusion_residual_scale`，否则无法判断修复来自哪里。

LIBERO U0-L：

```bash
BASE_CKPT=/absolute/path/to/robocasa_gr00t_or_universal_checkpoint.pt \
TRAIN_STAGE=u0_libero \
CONFIG_YAML=examples/LIBERO/train_files/starvla_qwen_universal_vlog_libero.yaml \
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_qwen_universal_vlog.sh
```

脚本要求显式设置 `BASE_CKPT`，不会读取 `.vlog_vla.env` 中可能残留的 OFT
checkpoint。训练数据必须提供连续 `state`、`embodiment_id`，以及可选的
`action_mask/episode_id/timestep`。

## 核心实现

| 文件 | 职责 |
|---|---|
| `starVLA/model/vlog_vla/qwen_universal_vlog.py` | framework、共享 action expert、VLOG core、paired FM losses、阶段冻结 |
| `starVLA/model/vlog_vla/universal_adapters.py` | native state/action ↔ shared DiT boundary |
| `starVLA/model/vlog_vla/option_conditioner.py` | option token + action-token FiLM |
| `starVLA/model/vlog_vla/semimarkov_controller.py` | 独立并行环境持久 option 状态 |
| `starVLA/model/framework/VLM4A/QwenUniversalVLOG.py` | StarVLA registry bridge |
| `tests/test_qwen_universal_vlog_modules.py` | parity、mask、adapter、option path、controller tests |

## 尚未验证

仓库代码和 CPU tests 的存在不等于实验完成。以下结果必须在训练机/benchmark
环境中生成并保存原始证据：

- 官方 4B/90k checkpoint 的 strict-load 报告；
- 单卡 GPU forward/backward/predict、梯度、显存和 step time；
- RoboCasa 与 LIBERO 实际 dataset statistics 的 normalize/unnormalize round-trip；
- RoboCasa Core-6、LIBERO rollout；
- Gate O、Gate R、wrong/shuffle intervention；
- 真机 rate limit、collision guard、watchdog 和急停验证。

## Legacy 实现

旧 `QwenOFTVLOG`、graph、critic、termination 和六阶段脚本仍保留，供失败分析与
历史 checkpoint 复查使用，但不再是统一 RoboCasa/LIBERO/真机模型主线，也不得与
`QwenUniversalVLOG` checkpoint 混用。

上游 StarVLA 代码遵循其原仓库许可；本仓库不提交模型权重、数据集、仿真资产或
训练输出。
