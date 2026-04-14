---
updated: 2026-04-14
tags: [changelog]
---

# 📝 变更日志

## 2026-04-07

- **[init]** 初始化项目知识库
- <!-- updated: 2026-04-07 20:56 --> **[runtime]** 将 `flash-attn`、`triton`、`kt-kernel` 下放为可选依赖，新增 CPU-only fallback，`example.py` 改为显式传入模型路径，新增 `tests/test_smoke_cpu.py` 覆盖无 PIM 路径。
- <!-- updated: 2026-04-07 21:18 --> **[qwen3]** 修复 `SimpleEngine` 使用错误 `head_dim` 预分配 KV cache 的问题，确认 `example.py` 可在 `/home/yangfu/models/Qwen--Qwen3-30B-A3B-Base` 上以 `--device cpu --max-new-tokens 1` 跑通。

## 2026-04-08

- <!-- updated: 2026-04-08 10:12 --> **[benchmarks]** 新增 [benchmark_inference.py](../../benchmarks/benchmark_inference.py) 和 [benchmarks/README.md](../../benchmarks/README.md)，可统一测 `cpu`、`cuda`、`cuda_cpu_offload` 三类推理 backend。
- <!-- updated: 2026-04-08 10:12 --> **[pim]** 新增 [pim_microbench](../../benchmarks/pim_microbench/) 下的 host/DPU microbenchmark、build 脚本和 run 脚本；simulator 模式已跑通，硬件模式当前卡在 `dpu_alloc_ranks`。
- <!-- updated: 2026-04-08 11:02 --> **[host-validation]** 用户宿主机已确认存在 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 和 `/dev/dpu_rank*`，并成功跑出真实 PIM 硬件 benchmark；另已修复 inference benchmark，使 `cuda` backend OOM 时不阻断后续 `cuda_cpu_offload`。

## 2026-04-09

- <!-- updated: 2026-04-09 00:00 --> **[qwen3-layout]** 为 `Qwen3` 增加 checkpoint 自适应 expert 布局检测，支持从真实 safetensor 键名自动切换到 unpacked `gate_proj` / `up_proj` / `down_proj`。
- <!-- updated: 2026-04-09 00:00 --> **[offload-fixes]** 修复 `cuda_cpu_offload` 链路中的 CPU fallback 内存翻倍问题和 attention mask dtype 问题，确认 `/home/yangfu/models/Qwen--Qwen3-30B-A3B-Base` 可在 `--offload-device-experts 2` 下真机跑通。
- <!-- updated: 2026-04-09 00:00 --> **[pim-metrics]** 将 PIM microbenchmark 指标改为明确的整数 workload 度量，新增 `kernel_workload`、`kernel_element_gops`、`kernel_int32_gops_estimate`，避免误读为浮点算力。
- <!-- updated: 2026-04-09 00:00 --> **[backend-abstraction]** 新增 `ExpertOffloadBackend`、`PIMMoEBackend` 和 `offload_backend` 选择逻辑，`HybridMoE` / `LLM` / benchmark 入口已支持 `pim_shadow` 主链路。

## 2026-04-14

- <!-- updated: 2026-04-14 11:30 --> **[versioning]** 将仓库版本提升到 `v0.2.0`，用于标记“CPU baseline + Qwen3 修复 + cuda_cpu_offload + pim_shadow + UPMEM benchmarks”这一阶段性里程碑。
- <!-- updated: 2026-04-14 11:30 --> **[knowledge-sync]** 同步更新知识库中的架构说明、当前焦点、路线图和日志，避免继续沿用 2026-04-08 之前的过期状态描述。
