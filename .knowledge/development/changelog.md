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
- <!-- updated: 2026-04-15 06:40 --> **[migration-manager]** 新增 `expert_migration.py`，为 backend 提供每层 migration queue 与阶段历史记录；动态调度相关单测已补到 `tests/test_core.py`。
<!-- updated: 2026-04-15 06:58 -->

## 2026-04-15

- 新增 `nano_ktrans/layers/expert_mlp.py`，把 shared expert module 定义从 `mixtral.py` 抽离，供模型初始化和运行时 expert materialization 共用。
- `ExpertWeightLoader` 新增单 expert 加载接口，支持 decode 阶段按需从 safetensors 拉起单个 expert 权重。
- `HybridMoE` 新增最小 decode 迁移执行数据面：
  - drain 本层 migration queue
  - promotion 时动态构建 GPU expert 并注入 `gpu_experts`
  - demotion 时从 `gpu_experts` 移除并更新 mask
- 新增测试覆盖 decode 阶段 migration queue 被实际执行的路径。

<!-- updated: 2026-04-15 07:06 -->

- 新增 `nano_ktrans/kernels/expert_materialization.py`，提供单 expert 的 CPU staging cache、预取队列和基础诊断。
- `HybridMoE` 现在会在 `prefill` 阶段对候选 promotion expert 发起预取，并在 `decode` promotion 时优先命中 staging cache。
- 新增测试覆盖 prefill 阶段的 expert 预取路径。

<!-- updated: 2026-04-15 07:14 -->

- `HybridMoE` 的 decode migration 现在接入了 GPU budget 约束：若 promotion 时 GPU resident set 已满，会先按 hotness 驱逐冷 expert，再执行热点 expert promotion。
- 新增测试覆盖“为 promotion 驱逐冷 expert”的运行时路径。

<!-- updated: 2026-04-15 07:22 -->

- `HybridMoE` 现在会对 decode 阶段生成的 future promotions 也发起预取，不再仅限于 prefill 预热。
- decode migration 队列现按“当前活跃优先 + hotness 优先”排序，更接近真实热点 cache 的调度语义。
- 新增测试覆盖 decode 阶段的 future promotion 预取路径。

<!-- updated: 2026-04-15 07:30 -->

- decode promotion 队列进一步接入 `prefetch ready` 优先级，优先消费 staging cache 已就绪的专家。
- `HybridMoE` 诊断中新增 `decode_prefetch_hits` / `decode_prefetch_misses`，用于观察预热是否真正命中 decode promotion。

<!-- updated: 2026-04-15 07:37 -->

- `ExpertMaterializationManager.prefetch()` 现在会返回是否真的触发了新预取，避免把重复请求误记成有效预热。
- `HybridMoE` 诊断新增 `prefetch_enqueued`，用于区分“请求数”和“真正进入 staging cache 的次数”。

<!-- updated: 2026-04-15 07:44 -->

- `LayerExpertState` 新增 `last_access_step` 与 `last_residency_change_step`，scheduler 已开始维护这些 anti-thrashing 元数据。
- 当前 cooldown / idle-age 逻辑先以配置和诊断形式接入，默认值保持不改变现有行为。

<!-- updated: 2026-04-15 07:53 -->

- scheduler 新增 `prefill_collect_only`、`step_stride_prefill` 和 `step_stride_decode` 配置。
- `LLM`、`example.py`、`benchmark_inference.py` 已暴露这些入口，后续可以直接在真实 benchmark 中对比不同调度策略。

<!-- updated: 2026-04-15 08:01 -->

- decode migration 新增 `decode_require_prefetch_ready` 开关。
- 开启后，未完成 staging prefetch 的 promotion 会先 defer，而不是直接在 decode 关键路径上同步 materialize。
- 新增测试覆盖“defer until prefetch ready”的 decode migration 路径。

<!-- updated: 2026-04-15 08:08 -->

- `ExpertMigrationManager` 现在会按 expert 对 pending migration queue 去重。
- queue 诊断新增 `total_enqueued_ops`、`total_deduped_ops`、`total_drained_ops` 和 per-phase `deduped_plan_size`。

<!-- updated: 2026-04-15 08:16 -->

- scheduler 新增 `prefetch_candidate_budget_per_layer`，可按层从 offloaded experts 中挑选热点候选做预取。
- `HybridMoE` 新增 `prefetch_candidate_scans` 诊断，用于观察候选预取是否实际发生。

<!-- updated: 2026-04-15 10:58 -->

- scheduler 新增 profile 预设：`baseline`、`overlap_safe`、`eager`。
- `LLM`、`example.py`、`benchmark_inference.py` 现在都可直接选择 scheduler profile，而不必手工拼全部调度开关。
- benchmark 新增调度摘要输出，自动聚合：
  - `prefetch_requested / enqueued / materialized`
  - `decode_prefetch_hits / misses`
  - `runtime_evictions`
  - `runtime_deferred_for_prefetch`
  - migration queue 的 `enqueued / deduped / drained`
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `45 passed`。

<!-- updated: 2026-04-15 11:08 -->

- migration manager 新增 lifecycle 跟踪：`queued / prefetching / ready / deferred / applied`。
- `HybridMoE` 现在会在预取、ready 命中、defer 和 applied 路径上写回 lifecycle 状态，benchmark 摘要也会同步聚合这些指标。
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `46 passed`。

<!-- updated: 2026-04-15 11:20 -->

- `decode_require_prefetch_ready` 模式下，decode 入口现在只会消费“进入本层前已经 ready 的 promotion”。
- migration manager 新增 `take_layer()` / `peek_layer()`，让 decode 可以保留未 ready 的 pending op，而不是先 drain 再重排。
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `47 passed`。

<!-- updated: 2026-04-15 11:27 -->

- `ExpertMaterializationManager` 新增 `poll_ready()`，可把后台完成的 prefetch future 主动转成 staging cache 命中。
- `HybridMoE` 现在会在进入本层前先轮询 ready prefetch，再把 migration lifecycle 更新为 `ready`。
- benchmark 摘要新增 `prefetch_polled_ready`，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `48 passed`。

<!-- updated: 2026-04-15 11:34 -->

- migration manager 新增 `take_ready_layer()` 和 `total_ready_drains` 统计。
- decode ready-only 路径已改成直接消费 migration manager 的 ready 子集，而不是由 `HybridMoE` 手写过滤逻辑。

<!-- updated: 2026-04-15 11:39 -->

- `ExpertMaterializationManager` 新增 completion queue；后台 prefetch 完成后会先进入 queue，再由 `poll_ready()` 消费。
- benchmark 摘要新增 `prefetch_completion_events`，用于区分“future 已完成”和“前台已轮询并入 cache”。

<!-- updated: 2026-04-15 11:45 -->

- `HybridMoE.forward()` 不再每层自行轮询 ready prefetch。
- `SimpleEngine` 现在会在每次 `prefill` / `decode_step` 进入模型前统一调用 `MixtralModel.refresh_offload_state()`，将 ready 刷新上移到 token-step 级别。
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `49 passed`。

<!-- updated: 2026-04-15 11:50 -->

- `SimpleEngine` 新增统一 `_refresh_offload_state()` helper，full prefill、chunked prefill 和 decode 现在共用同一 refresh 入口。
- 新增对应测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `50 passed`。

<!-- updated: 2026-04-15 11:55 -->

- `MixtralModel` 新增 `offload_refresh_calls` 和 `offload_refresh_ready_total` 统计。
- `LLM.get_offload_diagnostics()` 与 benchmark 摘要现在会带出模型级 offload refresh 指标。

<!-- updated: 2026-04-15 11:59 -->

- `ExpertMaterializationManager` 新增 `has_pending_or_ready()`。
- `HybridMoE.refresh_offload_state()` 现在会在无 pending/ready prefetch 时直接短路返回，减少空轮询。

<!-- updated: 2026-04-15 12:05 -->

- benchmark 新增 `--scheduler-profile-sweep`，可在单次运行中依次比较多组 scheduler profile。
- `normalize_scheduler_profiles()` 会做 profile 归一化与去重，避免 sweep 配置重复。

## 2026-04-16

- <!-- updated: 2026-04-16 00:40 --> **[migration-pipeline-runtime]** 新增 `MigrationPipelineRuntime`，将 token-step 级 offload refresh 提升为最小流水线运行时；ready prefetch 轮询与 ready promotion 现在可在进入模型前统一推进，不再依赖层内 forward 临时收敛。
- <!-- updated: 2026-04-16 00:40 --> **[pipeline-diagnostics]** `MixtralModel.offload_refresh_diagnostics()` 与调度摘要新增 pipeline 指标，包括 `offload_pipeline_ticks`、`offload_pipeline_ready_applied_total`、`offload_pipeline_ready_deferred_total`，便于观察“ready 到 applied”是否开始形成流水线。
- <!-- updated: 2026-04-16 00:40 --> **[tests]** 新增 pipeline runtime 覆盖：模型级 refresh 已验证 phase-aware pipeline tick，`HybridMoE` 也新增 ready promotion 在 pipeline hook 中被提前应用的单测；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `52 passed`。
- <!-- updated: 2026-04-16 01:05 --> **[pipeline-priming]** `HybridMoE.advance_offload_pipeline()` 现在会在 decode 进入模型前主动检查 pending promotion，并统一推进 `queued -> prefetching/deferred`；层内 `decode_require_prefetch_ready` 路径不再重复承担这部分预取提交逻辑。
- <!-- updated: 2026-04-16 01:05 --> **[pipeline-counters]** pipeline runtime 新增 `offload_pipeline_prefetch_submitted_total` 统计，便于观察 token-step 级 runtime 是否真的在为后续 ready promotion 预热 pending experts；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `53 passed`。
- <!-- updated: 2026-04-16 01:30 --> **[resident-export]** `ExpertOffloadBackend` 新增 `export_expert_weights()` 接口，CPU/PIM backend 现在可直接导出 resident expert 权重；`HybridMoE` 的 prefetch 路径已优先尝试从 offload resident tier 直接 stage 到 materialization cache。
- <!-- updated: 2026-04-16 01:30 --> **[resident-staging]** `ExpertMaterializationManager` 新增 `stage_expert()` 和 `resident_stage_hits`，可以记录“从 resident tier 直接命中 staging cache”的次数，减少 decode promotion 对 checkpoint 扫描的依赖；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `55 passed`。
- <!-- updated: 2026-04-16 01:50 --> **[warm-expert-cache]** `HybridMoE` 新增 demotion 后的 warm expert cache：GPU 驱逐下来的 expert module 可暂存到 CPU 侧 warm cache，后续短时间 re-promotion 时可直接复用 module，减少重复构建成本。
- <!-- updated: 2026-04-16 01:50 --> **[warm-cache-diagnostics]** 新增 `warm_cache_hits / stores / evictions / size` 诊断，并补充单测覆盖 demote 后缓存与 re-promotion 命中；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `57 passed`。
- <!-- updated: 2026-04-16 02:10 --> **[ready-prebuild]** token-step pipeline 现在会在 decode 进入模型前，对已经 `READY` 但尚未 materialize 的 promotion 预先构建 expert module，并放入 warm cache；后续 `READY -> APPLIED` promotion 可直接命中 warm cache。
- <!-- updated: 2026-04-16 02:10 --> **[ready-prebuild-tests]** 新增单测覆盖 `READY` expert 在 pipeline hook 中被 prebuild，再由 promotion 直接命中 warm cache 的路径；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `58 passed`。
- <!-- updated: 2026-04-16 02:25 --> **[cpu-prebuild]** ready expert 的 prebuild 现在固定在 CPU 上进行，promotion 再执行单次 device transfer；这让 token-step pipeline 的 prebuild 更接近“后台准备对象，前台只做激活”。
- <!-- updated: 2026-04-16 02:25 --> **[warm-transfer-diagnostics]** 新增 `warm_cache_device_transfers` 统计，用于观察 warm cache 命中后实际发生了多少次 CPU->device 激活拷贝。
- <!-- updated: 2026-04-16 02:40 --> **[warmed-lifecycle]** migration lifecycle 新增 `warmed` 状态，专门表示“expert 数据已 ready 且模块已在 warm cache 中预构建”；这样可以把 `ready -> warmed -> applied` 与简单 `ready -> applied` 区分开来。
- <!-- updated: 2026-04-16 03:05 --> **[activation-stage]** migration lifecycle 进一步新增 `activated` 状态，表示 warm cache 中的 expert module 已完成 device transfer、尚未正式进入 GPU resident set；`HybridMoE` 的 token-step pipeline 现在会在 decode 前先推进 `warmed -> activated`，再由最终 promotion 完成 `activated -> applied`。
- <!-- updated: 2026-04-16 03:05 --> **[activation-diagnostics]** pipeline/runtime/scheduler 摘要新增 activation 相关指标，包括 `offload_pipeline_activation_ready_total`、`activation_submitted / ready / applied` 和 `migration_activated_events`，便于区分“模块已预构建”和“设备激活已完成”。
- <!-- updated: 2026-04-16 03:20 --> **[activated-cache]** `HybridMoE` 现在为已完成 device transfer 的 expert 引入独立 activated cache；decode promotion 会优先命中 activated cache，再退到 CPU warm cache 或冷路径，进一步压缩 `activated -> applied` 关键路径。
- <!-- updated: 2026-04-16 03:32 --> **[activated-cache-priority]** activated cache 现在会按 lifecycle 优先级与 hotness 做预算保留；decode 前只把最值得保留的 warmed experts 提升到 device-side activated cache，避免较冷 expert 抢占有限激活预算。
- <!-- updated: 2026-04-16 03:43 --> **[deferred-state-preservation]** migration queue 重新排入 `*_deferred` op 时，若 expert 已处于 `prefetching/ready/warmed/activated`，现在会保留该中间态，不再把 pipeline 进度重置成 `deferred`，避免已完成一半的 promotion 在控制面上“掉回队尾”。
- <!-- updated: 2026-04-16 03:55 --> **[requeue-diagnostics]** migration queue 现新增 `total_requeue_preserved_states`，用于统计 deferred/queued 重排时保留了多少个中间 lifecycle；scheduler 摘要也同步输出这一指标，便于衡量流水线是否真的在“只前进不回退”。
- <!-- updated: 2026-04-16 04:07 --> **[ready-queue-drain]** decode 的 ready promotion 路径已改成“peek + selective consume”：只在真正 `applied` 时把 op 从 pending queue 里移除，预算不足导致的未消费 ready op 会保留在原队列中等待下一步，而不再重复 enqueue 成 `*_deferred`。
- <!-- updated: 2026-04-16 04:18 --> **[strict-ready-only]** `decode_require_prefetch_ready` 语义进一步收紧：即使 resident tier 直接 stage 成功，decode prime 阶段也不会在同一步立刻把它标成 `ready` 并消费，而是等待下一次 refresh/pipeline tick，使“ready-only”真正意味着“前一阶段已完成”。
- <!-- updated: 2026-04-16 04:30 --> **[per-run-scheduler-summary]** inference benchmark 现在会在每次 generation 前重置 offload/runtime 计数器，并把单次 run 的 `scheduler_summary` 直接挂到 run 结果上；这样 profile 对比时不再被 warmup 或前序 decode 步的累计诊断污染。
- <!-- updated: 2026-04-16 04:42 --> **[prebuild-target-budget]** `HybridMoE` 的 ready prebuild 现在不再对所有 ready 候选一视同仁，而是按 lifecycle 优先级、hotness 和 decode 预算只保留更有价值的一批 prebuild target，避免较冷 expert 过早占用 warm cache 和构建开销。
- <!-- updated: 2026-04-16 04:55 --> **[promotion-source-breakdown]** pipeline 现在会统计每次 promotion 究竟来自 `activated cache`、`warm cache` 还是冷路径 build，并聚合为 `pipeline_prefetch_overlap_hits` 与 source breakdown；这让 benchmark 可以直接回答“有多少次 promotion 已经不是冷启动”。 
- <!-- updated: 2026-04-16 05:20 --> **[decode-queue-retention]** `HybridMoE._apply_queued_migrations()` 已从旧的 `drain/take_ready -> deferred requeue` 路径收敛到 `peek + selective consume`：只有真正 `applied` 的 promotion / demotion 才会从 migration queue 中移除，预算不足或仍未 ready 的 op 会原地保留在 pending queue 中，避免 layer-forward 和 token-step pipeline 重复搬运同一批 decode 迁移。
- <!-- updated: 2026-04-16 05:20 --> **[decode-ready-strictness]** layer-forward 现在在 `decode_require_prefetch_ready=true` 时只消费 lifecycle 已推进到 `ready/warmed/activated` 的 promotion；即便 materialization cache 已同步命中，也不会在同一个 forward 中越级把 expert 直接视为 `ready`，进一步统一 strict ready-only 语义。
- <!-- updated: 2026-04-16 05:20 --> **[tests]** 新增两条 decode 队列语义测试：一条验证“预算不足的 ready promotion 会继续留在 pending queue 中”，另一条验证“只移除已执行的 demotion，active expert 对应的 demotion op 会继续保留”；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `70 passed, 1 warning`。
- <!-- updated: 2026-04-16 05:40 --> **[pipeline-apply-batch]** ready promotion 现在先经过 `_select_ready_promotion_batch()` 做同层小批量截断；虽然底层 apply 仍逐 expert 执行，但 pipeline 已开始按批次统计 `pipeline_apply_batches` / `pipeline_apply_batch_experts`，为后续真正的 layer-batched apply 打基础。
- <!-- updated: 2026-04-16 05:40 --> **[batch-metrics]** `HybridMoE.diagnostics()`、`LLM.reset_offload_diagnostics()` 和 scheduler summary 现在都会输出 pipeline apply 批次指标，并补了对应单测；当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `71 passed, 1 warning`。
- <!-- updated: 2026-04-16 05:55 --> **[batch-eviction]** ready promotion 的同层批次现在会先统一计算需要腾出的 GPU slot，并通过 `_evict_for_promotion_batch()` 预先完成这批次的 eviction；这避免了 batch 内每个 expert 各自重复做一次 budget 检查。
- <!-- updated: 2026-04-16 05:55 --> **[batch-eviction-metrics]** 新增 `pipeline_apply_batch_evictions`，可直接观察一次 ready-apply 批次为了让位而预先执行了多少 GPU resident eviction；对应摘要和单测已补齐，当前 `tests/test_core.py + tests/test_pim_runtime.py` 仍为 `71 passed, 1 warning`。
- <!-- updated: 2026-04-16 06:10 --> **[runtime-batch-rollup]** `MigrationPipelineRuntime` 现在会汇总 token-step 级的 apply batch 指标，包括 `offload_pipeline_apply_batch_count_total`、`offload_pipeline_apply_batch_experts_total` 和 `offload_pipeline_apply_batch_evictions_total`，这样 benchmark 不只看层级局部状态，也能直接看每个 decode step 的批处理推进情况。
- <!-- updated: 2026-04-16 06:10 --> **[tests]** 新增 runtime 层的 batch 汇总测试，当前 `tests/test_core.py + tests/test_pim_runtime.py` 为 `72 passed, 1 warning`。
