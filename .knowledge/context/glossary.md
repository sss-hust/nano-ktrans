---
updated: 2026-04-21
tags: [domain, terminology]
---

# 📖 领域术语表

## MoE 推理相关

| 术语 | 定义 | 备注 |
|---|---|---|
| **MoE (Mixture-of-Experts)** | 将 FFN 拆成多个"专家子网络"，每个 token 只经过少数被 gate 选中的专家 | 例：Mixtral-8x7B 每 token 激活 2/8 |
| **Gate / Router** | 根据 hidden state 决定 token 路由到哪些 experts 的线性层 | 产物：logits → top-k ids + scores |
| **Top-k routing** | 每个 token 选 top-k 个最高分 expert，通常 k=2 | `topk_ids` + `topk_weights` |
| **Expert activation** | 某 expert 在当前 token/iteration 被 gate 选中的事件 | 二值信号（bincount）或连续信号（score） |
| **Hotness** | 每个 expert 被激活的频率/重要性 EMA | nano-ktrans 里是 `LayerExpertState.hotness` |
| **Prefill vs Decode** | prefill 一次处理整段 prompt（大 batch），decode 逐 token 生成（小 batch） | 两阶段的最优 offload 策略通常不同 |

## Expert Offloading 与 Cache

| 术语 | 定义 | 备注 |
|---|---|---|
| **Expert Offloading** | 把不在 GPU 上的 experts 放到 CPU/PIM/NVMe，按需加载 | nano-ktrans 的核心功能 |
| **Residency** | 某 expert 当前的物理位置：GPU / PIM / CPU | `ExpertResidency` enum |
| **Warm cache** | CPU 侧已 prebuild 的 expert module 缓存 | 命中时省去 checkpoint I/O |
| **Activated cache** | 已 `.to(device)` 完成但未正式进入 resident set 的 module | 命中时省去 device transfer |
| **Prepared tier** | warm + activated 两层合计的"准备好但未 apply"预算 | 由 `prepared_cache_budget` 控制 |
| **Resident set** | 实际在 GPU `gpu_experts` ModuleDict 中的 expert 集合 | 真正参与 forward 的那批 |
| **Migration op** | 一次 expert 位置变更请求 `src -> dst`，含生命周期状态机 | `ExpertMigrationOp` + `MigrationLifecycle` |
| **Migration lifecycle** | `queued → prefetching → ready → warmed → activated → applied` | 6 状态+`deferred` |

## 调度信号与控制面

| 术语 | 定义 | 来源 |
|---|---|---|
| **MRS (Minus Recent Score)** | HybriMoE 的 score-aware cache 公式 `S = α·TopP(s) + (1-α)·S`，把 router 概率直接注入 hotness EMA | HybriMoE (arXiv:2504.05897) §IV.C |
| **TopP(s)** | 每 token 只累加分数最高的 p 个 expert，低分被截断 | 典型 `p = 2 · top_k` |
| **Expert Map** | 一次 iteration 的"指纹"：prompt 嵌入 + 每层完整 gate 概率分布 | fMoE (arXiv:2502.05370) |
| **Expert Map Store** | 跨 iteration 积累的 LRU Expert Map 集合，支持语义/轨迹搜索 | nano-ktrans 实现：`utils/expert_map_store.py` |
| **Prompt anchor / semantic embedding** | 代表一次请求主题的向量；nano-ktrans 用 `embed_tokens(input_ids).mean()` | fMoE §5.1 论证：模型自身 embedding 已足够 |
| **Semantic search** | 冷启动层（`layer_idx < prefetch_distance`）用 prompt embedding 匹配历史 | 两阶段搜索第一段 |
| **Trajectory search** | 已有若干层 gate 观察后，用已观测分布匹配历史 | 两阶段搜索第二段 |
| **Cold promotion** | 未经过 warm/activated cache 直接从 checkpoint 构建的 expert promotion | `pipeline_promotion_source_cold` |
| **Cold promotion penalty** | 冷路径比例偏高时抬高 adaptive activation/prebuild limit 的信号 | prepared tier controller 信号 |
| **Throttle signal** | PIMoE 思路：DPU 利用率饱和时把 expert 切回 CPU | 尚未实现 |

## 诊断指标对应

| 论文术语 | nano-ktrans 指标 |
|---|---|
| EAM (Expert Activation Matrix) | `activation_freq[L, E]` + `LayerExpertState.hotness` |
| Expert hit rate | `decode_prefetch_hits / (hits + misses)` |
| Cold path ratio | `pipeline_promotion_source_cold / pipeline_promotion_source_*` |
| Prefetch redundancy | `prefetch_requested - prefetch_enqueued` |
| Batch apply size | `runtime_apply_batch_size_avg` |
| MRS observations | `hotness_mrs_observations / hotness_bincount_observations` |
| Semantic prefetch hits | `expert_map_semantic_prefetch` |
| Trajectory prefetch hits | `expert_map_trajectory_prefetch` |

## 硬件/工具

| 术语 | 定义 | 备注 |
|---|---|---|
| **UPMEM DPU** | 处理器-内存集成芯片；每个 DPU 在 DIMM 内部独立运行 kernel | 当前 nano-ktrans PIM backend 的目标硬件 |
| **PIM (Processing-in-Memory)** | 把计算搬到内存里，避免数据搬运瓶颈 | UPMEM / HBM-PIM / AiM 等 |
| **AMX / AVX-512** | Intel CPU 的矩阵/向量指令集 | `kt-kernel` CPU 加速依赖 |
| **GPTQ W4A32** | 权重 4-bit 量化、激活 32-bit 浮点的量化方案 | nano-ktrans 支持加载 GPTQ Int4 |
| **kernel_mode** | `PIMQuantizedRuntime` 的执行变体编号（1=full soft-float, 4=int8 fixed-point, 5=int16 runtime LUT 实验） | 当前稳定主路径是 mode 4 |
