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

## 2026-04-15

- <!-- updated: 2026-04-15 00:35 --> **[pim-runtime]** 新增 `pim_linear_runtime.py` 与 `pim_native/` 原生桥接代码，仓库现在具备最小真实 DPU 线性计算能力；已在真实硬件上用随机矩阵验证 DPU 结果与 CPU 对齐。
- <!-- updated: 2026-04-15 00:35 --> **[pim-backend]** `PIMMoEBackend` 新增实验性 `pim` 模式：expert 的线性投影可走真实 DPU，SiLU / gating 仍在 host 端执行，当前默认只对小 flattened batch 生效，其余情况自动回退 CPU。
- <!-- updated: 2026-04-15 15:10 --> **[pim-sharding]** 将原生 DPU linear host bridge 从单 DPU 扩展到多 rank / 多 DPU 行分片执行，新增 `runtime_dpu_count` 诊断，并让 `PIMLinearRuntime` 能按 `(profile, rank_count)` 复用不同运行时实例。
- <!-- updated: 2026-04-15 15:10 --> **[cuda-pim-validation]** 用真实 `/home/yangfu/models/Qwen--Qwen3-30B-A3B-Base` 跑通 `cuda_pim` 端到端链路；在 `--prompt Hi --offload-device-experts 2 --pim-rank-count 4 --pim-max-batch-tokens 1 --max-new-tokens 2` 下，确认所有层都发生了真实 DPU expert linear 调用并产出结果文件 `benchmarks/results/cuda_pim_2026-04-15.json`。
- <!-- updated: 2026-04-15 05:22 --> **[pim-fused]** 新增 `PIMExpertRuntime`、`dpu_expert_kernel.c` 和 `host_expert_bridge.c`，把 `gate/up/down + SiLU` 的完整 expert 子图接成实验性 fused DPU kernel，并让 `PIMMoEBackend` 支持 `pim_kernel_variant=fused`。
- <!-- updated: 2026-04-15 05:22 --> **[pim-fused-optimization]** 修正 fused expert kernel 的核心数据流，改为先在 WRAM 里计算完整 hidden 激活再复用到 down projection；真实 microbench 从原先二十到三十秒级降到约 `0.29s`，但仍显著慢于 `linear3` 的约 `0.03s`，因此当前默认策略保持 `linear`。
- <!-- updated: 2026-04-15 05:48 --> **[pim-fused-sharding]** 将 fused expert host bridge 从单纯 hidden 分片改成 `hidden_group x row_group` 二维分片；Qwen 级别单 expert microbench 进一步降到约 `0.26s`，并可看到 `expert_runtime_last_active_dpus=60`，但仍未优于 `linear3`。
- <!-- updated: 2026-04-15 06:04 --> **[dynamic-scheduler-skeleton]** 新增 `expert_runtime_state.py` 与 `scheduler/dynamic_expert_scheduler.py`，把系统目标从“静态 GPU expert mask”提升为“GPU/PIM 动态专家驻留”第一版骨架；`LLM`、`MixtralModel`、`HybridMoE` 已能携带驻留计划与调度器诊断，但真实迁移数据面尚未接入。
- <!-- updated: 2026-04-15 06:23 --> **[prefill-policy]** 为 `pim` backend 新增 `pim_prefill_policy` 和 `pim_prefill_token_threshold`，默认 prefill 走 CPU/GPU 路径，避免长 prompt 的大批量 token 直接压到 PIM。
- <!-- updated: 2026-04-15 06:23 --> **[dynamic-prefill-hook]** `HybridMoE` 现在会在 prefill 阶段基于路由结果更新调度器热度，并允许临时提升 GPU expert budget；当前只更新驻留状态与诊断，真实迁移数据面仍待实现。
- <!-- updated: 2026-04-15 06:31 --> **[migration-queue-semantics]** 调整动态调度语义：`HybridMoE` 现在只向 backend 排队 migration plan，不再在没有真实 GPU/PIM 数据面的前提下直接修改有效 `gpu_experts_mask`，避免控制面和执行面状态不一致。
