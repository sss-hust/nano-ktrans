---
updated: 2026-04-21
tags: [pitfalls, debugging]
---

# ⚠️ 已知陷阱 & 踩坑记录

<!-- updated: 2026-04-07 20:56 -->

## 可选加速依赖不能再被当成硬依赖

- 现象：没有安装 `flash-attn` 或 `kt-kernel` 时，连导入 `nano_ktrans.layers` / `nano_ktrans.models` 都会失败。
- 根因：`attention.py`、`cpu_infer.py`、`cpu_moe.py` 之前在模块导入时直接强依赖这些包。
- 修复：改为运行时探测，可用则走加速路径，不可用则退化到纯 PyTorch attention 和 CPU fallback。


<!-- updated: 2026-04-07 21:18 -->

## Qwen3-MoE 不能假设 `head_dim = hidden_size / num_heads`

- 现象：真实 Qwen3 checkpoint prefill 时，`store_kvcache` 报 shape mismatch，缓存期望的 head dim 是 `64`，实际 K/V 是 `128`。
- 根因：`SimpleEngine` 之前按 `hidden_size // num_attention_heads` 计算 KV cache 形状，但 Qwen3 配置里有显式 `head_dim=128`。
- 修复：优先使用 `config.head_dim`，只有缺省时才回退到 `hidden_size // num_attention_heads`。

<!-- updated: 2026-04-08 10:12 -->

## 当前会话能看到 NVIDIA 内核态信息，但没有用户态设备节点

- 现象：`lsmod` 能看到 `nvidia` 模块，`/proc/driver/nvidia/gpus/0000:15:00.0/information` 也存在，但 `nvidia-smi` 失败，`torch.cuda.is_available()` 为 `False`，且 `/dev/nvidia*` 不存在。
- 影响：仓库里的 CUDA 路径无法在当前会话里做真实 benchmark，只能报告 `unavailable`。

<!-- updated: 2026-04-08 10:12 -->

## UPMEM SDK 可诊断 MCU，但不代表 rank 可分配

- 现象：`dpu-diag` 能列出大量 `dpu_rank` 的 MCU version，但 `dpu_alloc_ranks` 仍然返回 allocation error。
- 根因：当前会话里 `/dev/dpu_rank*` 设备节点不存在，硬件 rank 没有暴露给用户态分配器。
- 影响：可以运行 simulator 模式的 PIM microbenchmark 做功能验证，但不能把 simulator 数据当成真实硬件性能。

<!-- updated: 2026-04-08 10:56 -->

## 当前 Codex 执行会话的 `/dev` 是私有 tmpfs，不是宿主机原始 `/dev`

- 现象：`/proc/driver/nvidia/gpus/0000:15:00.0/information` 存在、`lsmod` 里也有 `nvidia` 模块，但会话内 `ls /dev` 只看到极少量基础节点，完全没有 `/dev/nvidia*` 和 `/dev/dpu_rank*`。
- 根因：当前执行环境把 `/dev` 单独挂载成了私有 tmpfs，屏蔽了宿主机真实设备节点。
- 影响：这个会话里无法直接做真实 CUDA benchmark，也无法分配真实 UPMEM rank；只能做 CPU benchmark 和 simulator 验证。

<!-- updated: 2026-04-08 11:02 -->

## Qwen3-30B-A3B 在 48GB GPU 上的“全专家纯 CUDA”路径会 OOM

- 现象：用户宿主机上有真实 `/dev/nvidia*` 节点，但 `benchmark_inference.py` 在 `backend=cuda` 时仍然报 `CUDA out of memory`。
- 背景：GPU 0 总显存约 `47.41 GiB`，而纯 CUDA 路径会把所有专家都保留在 GPU 上。
- 影响：这个模型在当前显存条件下应重点测试 `cuda_cpu_offload`，而不是坚持全专家常驻 GPU。

<!-- updated: 2026-04-09 00:00 -->

## 真实 Qwen3 checkpoint 的 expert 权重不是 packed `gate_up_proj`

- 现象：`cuda_cpu_offload` 之前报 `Weight key '...gate_up_proj.weight' not found`。
- 根因：当前这份 `Qwen3-30B-A3B-Base` safetensor 实际存的是分开的 `gate_proj` / `up_proj` / `down_proj`，而不是打包的 `gate_up_proj`。
- 修复：在 `LLM` 初始化时基于 checkpoint 键名自适应布局，必要时将 `qwen3_moe` 切换为 unpacked expert spec。

<!-- updated: 2026-04-09 00:00 -->

## CPU fallback 不能把专家权重复制成一大堆 `nn.Linear`

- 现象：`cuda_cpu_offload` 一度吃到 `118 GiB` 内存和满 swap，实际是在内存抖动。
- 根因：fallback 路径同时保留了堆叠权重张量，又额外为每个 CPU expert 复制了一套 `nn.Linear` 权重。
- 修复：保留单份堆叠权重，直接用 `F.linear` 做专家计算。

<!-- updated: 2026-04-09 00:00 -->

## 当前 offload 性能仍受限于纯 PyTorch CPU fallback

- 现象：`CPUMoEBackend` 每层都打印 `kt-kernel/AMX unavailable. Using PyTorch fallback.`。
- 根因：当前环境没有安装 `kt_kernel` / `kt_kernel_ext`，CPU 也只有 `AVX512`，没有 `AMX` 标志。
- 影响：`cuda_cpu_offload` 已能运行，但性能不是最终目标形态；若要进一步提速，需要接 `kt-kernel` 或真实 PIM backend。

<!-- updated: 2026-04-09 00:00 -->

## 当前 `pim_shadow` 是主链路集成，不是 DPU 数值执行

- 现状：`HybridMoE` 已支持选择 `pim_shadow` backend，并会在主推理链路里统计可见 PIM rank、offloaded token/expert pair 等信息。
- 语义：当前数值结果仍由 CPU fallback 保底，PIM 真实 DPU 计算仍停留在独立 microbenchmark。
- 影响：现在已经能做“推理主链路 + PIM 可见性/统计”联动，但还不能把它解释成“专家 MLP 已在 DPU 上执行”。

<!-- updated: 2026-04-15 00:35 -->

## Python `dpu.driver` 仍不够稳定，真实 PIM 主链路优先走 C host bridge

- 现象：在当前机器上，`dpu.driver.DpuSet(nr_ranks=1, profile='backend=hw')` 仍会报 `fetch_program: ERROR: cannot find file` 和 `DpuError b'system error'`。
- 影响：即使真实 `/dev/dpu_rank*` 可见，Python 原生驱动目前仍不适合作为推理主链路的核心桥接层。
- 规避：当前 repo 新增了 `pim_native/host_bridge.c` + `pim_linear_runtime.py` 方案，通过共享库和 C host bridge 来调用真实 DPU 线性 kernel。

<!-- updated: 2026-04-15 05:22 -->

## fused expert DPU kernel 不能按输出行重复重算 hidden

- 现象：fused expert 第一版虽然数值上可用，但单 expert microbench 需要二十到三十秒，远慢于三次 DPU linear 的几十毫秒。
- 根因：旧实现把 `hidden = silu(gate) * up` 的计算放在输出行循环内部，导致每个输出行都重复扫描 `gate/up` 权重和输入，算法复杂度被放大。
- 修复：改成先在 DPU WRAM 中计算完整 hidden 向量，再统一用于 `down_proj`；性能已从秒级降到亚秒级，但当前仍慢于 `linear3`，说明后续瓶颈已转到 `down_proj` 阶段的数据流设计。

<!-- updated: 2026-04-15 05:48 -->

## fused expert 仅按 hidden 分片会浪费大量 DPU

- 现象：即使 fused kernel 已经不再重复重算 hidden，`rank_count=4` 时 `expert_runtime_dpu_count` 很大，但真正参与单 expert 的 DPU 仍然偏少，收益有限。
- 根因：如果只沿 intermediate 维切分，每个 hidden shard 只对应一个输出全矩阵，`output_dim` 方向没有并行展开，很多 DPU 闲置。
- 修复：host bridge 改成 `hidden_group x row_group` 二维分片，按部分 hidden 和部分 output row 同时切块，再在 host 端做 partial sum 聚合。

<!-- updated: 2026-04-15 06:58 -->

## 动态调度不能只改驻留表，不改运行时 expert 模块

- 现象：如果只在 scheduler/residency plan 里把某个 expert 标成 `GPU`，但没有真的把对应 expert module 构建并注入 `HybridMoE.gpu_experts`，前向时这个 expert 仍然不会走 GPU 路径。
- 根因：当前推理执行依赖两套状态同时一致：
  - `gpu_experts_mask`
  - `gpu_experts` 中真实存在的模块对象
- 修复：当前最小可执行数据面已经改成 decode 阶段先 drain migration queue，再同步 materialize / demote GPU experts，并立即调用 backend 的 `update_gpu_expert_mask()`。

<!-- updated: 2026-04-21 20:00 -->

## MRS hotness 必须按 token 数归一化

- 现象：实现 HybriMoE MRS 公式 `S = α·TopP(s) + (1-α)·S` 时，第一版直接把 `torch.scatter_add_(..., router_scores)` 的结果当作新 observation；prefill 阶段一次 observe 看到 512 个 token，score_mass 被推到极大值，后续 decode 的 `(1-α)` 衰减完全盖不住，EMA 永远卡在高位。
- 根因：MRS 原论文的 `TopP(s)` 是"每次 iteration 的 top-p 分数"，不是"整段序列累加"；prefill 等价于一次性看到很多次 decode。
- 修复：`utils/expert_runtime_state.py::update_hotness` 在 MRS 分支里做 `score_mass = score_mass / token_count`，保证单次 observe 的贡献落在 `[0, 1]` 合理量级。
- 经验：所有把 router 概率灌进 EMA 的设计都要考虑 prefill 放大效应；`bincount` 模式同样存在这个问题，只是旧代码没修。

<!-- updated: 2026-04-21 20:00 -->

## `@lru_cache` 对 dict 参数会 TypeError，不是"自动忽略"

- 现象：`rotary_embedding.get_rope(rope_scaling=dict)` 在 Qwen3-30B 等使用 rope_scaling 的 checkpoint 上直接 `TypeError: unhashable type: 'dict'`。
- 根因：`@lru_cache` 把所有参数拼成缓存 key，dict 是 unhashable。
- 修复：拆成两层：`_validate_rope_scaling`（不缓存）+ `_build_rope(@lru_cache)`（只缓存 hashable 的 head_size/rotary_dim/max_position/base）。
- 经验：任何给 `@lru_cache` 函数加非 hashable 参数的 PR 必须拆分，**不要信"反正没人传 dict"**。

<!-- updated: 2026-04-21 20:00 -->

## Expert Map Store 的 prompt 锚点不能用 BOS token embedding

- 现象：Expert Map Store 第一版用 `hidden_states[0]`（第一个 token 的 embedding）作为 prompt 语义向量，但在真实多请求场景下，所有 prompt 的 BOS token 都是同一个，embedding 几乎完全一致，语义搜索退化成随机命中。
- 修复：改用 `embed_tokens(input_ids).mean(dim=(0, 1))`，跨所有 token 取平均，足以区分不同主题的 prompt，且不需要额外 forward 过一层 encoder。
- 经验：fMoE 论文 §5.1 已经论证过 "model 自身的 embedding layer 输出就足以做 expert routing 预测"，但要用 **token 维度的 mean** 而不是取首 token。

<!-- updated: 2026-04-21 20:00 -->

## `torch.bincount` 不支持浮点 weight，MRS 必须用 `scatter_add_`

- 现象：想把 router probability mass 累加进 `[num_experts]` 张量时，第一反应是 `bincount(ids, weights=scores)`；但 `torch.bincount` 的 `weights` 只支持 int tensor，给浮点会报 `RuntimeError: bincount only supports 1-d non-negative integral inputs`。
- 修复：改用 `score_mass.scatter_add_(0, top_ids.reshape(-1), top_values.reshape(-1))`，语义等价且原生支持浮点。
- 经验：涉及"按 index 累加 float"的场景统一用 `scatter_add_`；`bincount` 仅限二值/计数。

<!-- updated: 2026-04-21 20:00 -->

## 向后兼容：给 scheduler.observe 加新参数要保留双计数器

- 现象：`DynamicExpertScheduler.observe` 增加 `topk_weights` kwargs 时，必须考虑"MRS 开启但某些调用路径没传 weights"的场景（例如 profile 调用、测试调用）。
- 修复：`observe` 内部判断 `use_mrs = hotness_mrs_alpha is not None and topk_weights is not None`；两条路径分别累加 `hotness_mrs_observations / hotness_bincount_observations` 两个计数。
- 经验：**benchmark 如果看不到"新路径实际跑了多少次 vs 回退到旧路径多少次"，就没法判断新 feature 是否生效**。所有 feature flag 式改动都应该同时埋点新旧路径。

<!-- updated: 2026-04-21 20:00 -->

## `record_router_probs` 不要放进 `_pipeline_lock` 块里

- 现象：Expert Map Store 第一版在 `HybridMoE.forward` 的 `with self._pipeline_lock:` 块内部调用 `_record_router_probs`；`_pipeline_lock` 本意是保护 migration lifecycle / resident set / prepared tier 等共享状态，会被 background worker 持锁。放 record 进去等于无辜阻塞 background worker。
- 修复：`_record_router_probs` 只写 per-iteration 的 in-flight `ExpertMap`（由 `attach_expert_map` 挂到 `self._current_expert_map`，单线程），完全不触及共享状态，挪到 lock 外。
- 经验：**任何只改 per-request / per-iteration 私有对象的操作都不应持 shared pipeline lock**；否则锁粒度越来越粗、background worker 越来越难真正并行。
