# VLOG-VLA Benchmark Setup

This checkout now contains the full StarVLA tree plus the VLOG overlay. The
benchmark paths are centralized in `.vlog_vla.env`.

## One-time config

```bash
cd /home/nlk/project/vlog_vla
cp .vlog_vla.env.example .vlog_vla.env
```

Edit `.vlog_vla.env` only when you want to move datasets, external simulator
checkouts, or downloaded checkpoints away from `playground/`.

## Local checks

```bash
bash tools/setup_vlog_benchmarks.sh check
```

Qwen3-VL is already present at:

```text
playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

## Download commands

Released checkpoints:

```bash
bash tools/setup_vlog_benchmarks.sh download-checkpoints
```

LIBERO LeRobot training data:

```bash
bash tools/setup_vlog_benchmarks.sh download-libero
```

Optional VLM co-training data:

```bash
bash tools/setup_vlog_benchmarks.sh download-vlm-data
```

External simulators:

```bash
bash tools/setup_vlog_benchmarks.sh robotwin-code
bash tools/setup_vlog_benchmarks.sh robocasa365-code
bash tools/setup_vlog_benchmarks.sh robocasa-tabletop-code
```

RoboCasa GR1 tabletop finetuning data is very large:

```bash
bash tools/setup_vlog_benchmarks.sh robocasa-tabletop-data
```

## Official references

- LIBERO: https://starvla.github.io/docs/zh-cn/benchmarks/libero/
- RoboTwin: https://starvla.github.io/docs/zh-cn/benchmarks/robotwin/
- RoboCasa: https://starvla.github.io/docs/zh-cn/benchmarks/robocasa/

The merged upstream examples remain under `examples/simBenchmarks/`.
