---
updated: 2026-04-22
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

<!-- updated: 2026-04-22 10:50 -->

## 诊断方法里访问 `__init__` 才 set 的新字段必须用 `getattr` 容错

- 现象：`c816a9c`（P2 Expert Map Store）在 `LLM.__init__` 里新增 `self.expert_map_store` 字段。`tests/test_core.py::test_llm_get_offload_diagnostics_reports_prepared_budget_heuristic` 用 `LLM.__new__(LLM)` 绕过 `__init__` 以避免走完整的 `AutoConfig → safetensors` 加载，只手工 set 了部分字段，调用 `llm.get_offload_diagnostics()` 时抛 `AttributeError: 'LLM' object has no attribute 'expert_map_store'`。
- 根因：`get_offload_diagnostics()` 原来直接写 `self.expert_map_store.diagnostics()`，没给绕过 `__init__` 的测试路径留后路。
- 修复：改为 `getattr(self, "expert_map_store", None)`，与同文件 `reset_offload_diagnostics` 里其它字段的访问风格一致。
- 经验：凡是在 `LLM.__init__` 里才 set、且出现在对外 `get_*` / `reset_*` 诊断方法里的字段，**都要当作"可能不存在"来访问**（`getattr(self, "field", default)`），才能兼容：
  - `LLM.__new__(LLM)` 构造的单测
  - 继承/mixin 扩展类
  - 将来 feature flag 关闭时 `__init__` 里 set 成 None 的场景

<!-- updated: 2026-04-22 10:50 -->

## `ExpertWeightLoader` 在构造期扫描 safetensor 会破坏纯 GPU smoke test

- 现象：`tests/test_smoke_cpu.py::test_cpu_only_smoke_generation_path` 用默认 `MixtralForCausalLM(config, layer_gpu_expert_masks)` 构造（`weight_path=""`），失败在 `FileNotFoundError: No .safetensors files found in`。这个 smoke test 自 `047af5c` 加入后从未更新，但之后某次重构让 `HybridMoE` 无条件实例化 `ExpertMaterializationManager`，而 `ExpertWeightLoader.__init__` 又强制要求目录里至少有一个 `.safetensors`。
- 根因：smoke test 的合法用例是"纯 GPU 专家 + 随机初始化，根本不需要加载任何权重"，但构造期硬校验把这条路径直接堵死。本地单元测试逐组件构造时都手工传了合法 `weight_path`，所以没人发现。
- 修复：让 `ExpertWeightLoader.__init__` 在 `weight_path == ""` 时进入"空加载器"状态（`_files=[]`、`_key_to_file={}`、`_quantize_config={}`），不抛错；`load_*` 真的被调用时原有的 `KeyError("Weight key '...' not found")` 会自动抛出，诊断信息更具体。非空路径但缺 safetensor 时**仍然**抛 `FileNotFoundError`（避免掩盖 typo）。
- 经验：
  1. "既支持真实权重加载、又要支持随机初始化跑 forward"的类，其子组件在构造期**不应该**对 weight 文件存在性做强校验。
  2. 延迟到真正 load 调用时失败，错误信息（具体到缺失 key）反而比构造期的 `FileNotFoundError` 更有用。
  3. smoke test 要进 CI 必跑集，单元测试 + 逐组件测试无法暴露"整条类链路能否默认参数构造"的回归。

<!-- updated: 2026-04-22 10:50 -->

## AI 生成的测试必须本地实跑一次才能合入

- 现象：`04dfbda`（"Fix expert migration eviction"）引入的 `test_backend_notify_expert_evicted_called_on_demotion`，从一开始就**根本跑不起来**：
  - `from nano_ktrans.utils.context import InferenceContext` — `context.py` 里类名叫 `Context`，没有 `InferenceContext`
  - `HybridMoE(expert_hidden_size=..., expert_intermediate_size=..., expert_key_template=..., expert_proj_names=..., offload_backend_name=...)` — 真实 `__init__` 签名是 `hidden_size / moe_intermediate_size / offload_backend`，测试用的参数名完全对不上
  - `moe.update_residency_plan(...)` — `HybridMoE` 上根本没有这个方法
- 根因：PR 描述里写了 `Co-Authored-By: Claude Opus`，测试是 AI 生成但**没在本地跑过**。review 只看 diff，没跑 pytest。
- 修复：参照相邻真实测试 `test_hybrid_moe_applies_decode_migration_plan` 的模式重写：
  - 用真实存在的构造参数 `HybridMoE(num_experts=, top_k=, hidden_size=, moe_intermediate_size=, gpu_experts=ModuleDict(), gpu_experts_mask=, ...)`
  - 通过 `hybrid.offload_backend.queue_migration_plan([ExpertMigrationOp(GPU→PIM)])` 下发 demotion op
  - forward 时迁移管线会调用 `_demote_expert_from_gpu` → `notify_expert_evicted(expert_idx, 'gpu')`
- 经验：
  1. 任何 AI 生成的测试必须至少在本地 `pytest -k <new_test>` 跑通一次再提交
  2. 同 PR 里如果新加了 production code（如 `notify_expert_evicted`），应该**同时**跑全量 pytest 确认新测试和旧测试都绿
  3. `Co-Authored-By: Claude Opus` 不等于免责声明，PR 作者仍需对可运行性负责

<!-- updated: 2026-04-22 11:25 -->

## Layout 自适应要同时识别量化变体后缀（.weight + .qweight）

- 现象：`benchmark_inference.py --backend cuda_pim --model-path .../Qwen3-30B-A3B-GPTQ-Int4` 从入库起（2026-04-20 的 e2e JSON）就 abort 在 `Weight key 'model.layers.0.mlp.experts.0.gate_up_proj.weight' not found`；fp16 的 `Qwen3-30B-A3B-Base` 却可以正常跑。
- 根因：`models/config.py::adapt_config_to_checkpoint` 只探测 `.weight` 后缀 — 对 fp16/bf16 checkpoint 足够，但 GPTQ checkpoint 里**根本没有 `.weight` tensor**，只有 `.qweight / .scales / .qzeros / .g_idx`；两个探测分支都 miss → 配置保留默认的 packed spec → 下游 loader 去找 `gate_up_proj.weight` → 在 74000-key safetensor 里失败。
- 修复：`packed_keys` / `unpacked_keys` 同时包含 `.weight` 和 `.qweight` 两种后缀。单测 `test_qwen3_gptq_checkpoint_layout_adaptation` 锁定此路径。
- 经验：任何**以 `.weight` 为锚点的 checkpoint adapter**都会在 GPTQ / AWQ / QLoRA 等量化 checkpoint 上 silent-fail。新增 checkpoint 适配代码时，**必须同时覆盖 fp16 后缀（.weight）和至少一个量化后缀（.qweight / .qzeros）**，且要有单测同时覆盖两条分支。

<!-- updated: 2026-04-22 11:25 -->

## "自称 T-MAC" 的 kernel 要用静态审计或真机 breakdown 钉住

- 现象：`dpu_quantized_kernel.c::kernel_mode == 6` 的注释自称是 T-MAC bit-serial 实现，commit message 和 `2026-04-20` journal 也记作"P1: T-MAC bit-serial kernel 已实现"。但代码审查发现内循环还在做 `lut0_i16[q0]` + 7 条 `abs_x & 0xNN` 的 activation 侧 shift-add — 这是**朴素软件乘法器**，不是 T-MAC。
- 根因：真正的 T-MAC 要求"**weight 被完全编码进 LUT 索引**，内循环只做 `acc += T[bit][row][group][pack(x_bits)] << bit`，零乘法零分支"。当前 mode=6 的 LUT 仍以 weight nibble `q` 做索引，所以"消除乘法"其实没发生，反而在外层包了 7 条件分支。这一误解直接让 ADR-002 需要全新一轮 M-2 来落真实现。
- 修复：在 `tests/test_core.py::TestQuantizedKernelAudit` 里加 3 条静态断言固化当前"假 T-MAC"形状（`lut[q]` 查表 + `abs_x & 0xNN` 模式），作为 M-2 落 `kernel_mode=7` 时必须同步删除 / 替换的审计标记。
- 经验：**一个性能优化是否真的落地，不能靠 commit message 和代码注释判断**。需要任取其一：
  1. 静态代码审计（断言"不该出现的指令模式"真的不出现，如"内循环里不得有 `lut[q]`"）
  2. 真机 kernel breakdown（能定量看到 `launch_seconds` 下降到符合算法复杂度的水位）
  3. 与理论模型对照（DPU 无硬件乘法 → 真 T-MAC 的 cycle 数应接近 `MRAM_read_cycles × num_groups × output_dim`，不应随 activation bit-width 线性增长）

<!-- updated: 2026-04-22 16:40 -->

## bit-plane T-MAC 在 UPMEM DPU 上**比 int8 软件乘法慢**（负结果）

- 现象：`kernel_mode=7` 完全消除了 DPU 内循环的软件乘法（bit-plane bitmask + 条件加法，weight LUT 只查一次），数值与 mode=4 **bit-exact**。但 120-cell 真机 sweep（`benchmarks/results/pim_shape_sweep_M2_tmac.json`）显示 mode=7 **在 0/60 个 cell 跑赢 mode=4**（mode=7 peak 1.15× vs mode=4 peak 3.32×，mean 0.48× vs 1.45×）。
- 根因（决定性的算术）：
  - UPMEM DPU 的 `int8 × int16` 软件乘法 ≈ 10 cycles（SDK 编译器非常到位）
  - DPU **没有硬件 ctz/popcnt/SIMD**；`__builtin_ctz` 在 32-bit 半字上能被 lower 到 ~4-6 cycle 软件序列，但 7 个 bit-plane 即使仅 sparse 扫过一半 set bits 也要 ~200 次条件加 + 7 次 ctz ≈ 504 cycles
  - 64 次 int8×int16 软件乘法 ≈ 640 cycles — 和 mode=7 持平
  - 额外再加 mode=7 的 per-block 64B bit-plane DMA + 128 次 weight unpack（shift+mask），净成本反而高于 mode=4
- 经验：
  1. **T-MAC 论文的 2-5× 收益是在有 SIMD + 硬件 ctz 的 ARM/x86 上取得**，**不能平移到 UPMEM**。先估 cycle 再写 kernel。
  2. **"消除硬件不擅长的操作" ≠ "更快"**：要看替代方案的总 cycle 数。UPMEM 的乘法虽然是软件实现但已深度优化，不是真正的"不擅长"。
  3. **负结果也要完整落地并 benchmark**，不是只写个伪代码就下结论。这次 M-2 的 120 cell 真机数据 + bit-exact correctness 验证本身就是论文素材。
  4. **DPU kernel 新增大 stack 局部变量（>= ~512B）前务必先算 `STACK_SIZE_DEFAULT`**。本次 mode=7 加 `w0_block[64]+w1_block[64]+bp_cache[8]` ≈ 592B 直接把栈顶爆，全部 DPU `in fault`。修复是改用 `mem_alloc` 从 WRAM heap 分配。
- 参考：`ADR-002 §10`、`dpu_quantized_kernel.c::kernel_mode == 7` 的长注释块、`.codebuddy/dev_gate/M-2.toml` 的 KPI rationale。
