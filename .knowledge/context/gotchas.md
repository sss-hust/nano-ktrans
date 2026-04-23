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


<!-- updated: 2026-04-22 17:30 -->

## operator-only sweep 和 e2e decode 成本**不可同日而语**

- 现象：M-2 sweep 显示 PIM mode=4 `batch=1` 比 CPU grouped 快 2-3×；实装 cost-model 路由后 e2e decode 反而慢 13×（`decode_tps=0.228` vs `cpu=3.068`）。
- 根因：sweep 只测 "单算子 + weight 已 preload" 的稳态；e2e decode 每层每 step 都要付：
  1. GPU→CPU 同步拷贝（`hidden_states.to("cpu")`, `topk_ids.to("cpu", long)`）
  2. Python glue（`torch.where` + `index_add_`）
  3. per-step expert 切换的 `preload()` lookup + host pad copy
  4. 同步 `dpu_launch(DPU_SYNCHRONOUS)`，GPU attention 无法与 DPU 并行
- 合计 per-layer 实测 ~91ms（sweep 预测 ~2ms），~45× 加成全来自 orchestration，不是 kernel。
- 经验：**任何把 operator-only 数字外推到 e2e TPS 的路线图都要显式标注 overlap/async 假设**。M-3 cost model 对"决策对不对"有效，但"决策的收益落不落得下来"还要 M-4 async submit + GPU/PIM overlap 才能兑现。

<!-- updated: 2026-04-22 17:30 -->

## CPU baseline 如果输出 zeros，所有 "PIM vs CPU" TPS 对比都是假的

- 现象：`cuda_cpu_offload` 跑 GPTQ checkpoint 时 32-token decode 仅 4.17s（~7.7 tok/s）— 看起来 CPU 非常强；但 output_text 是完全乱码。
- 根因：`CPUMoEBackend.submit_forward` 在 `is_gptq and cpu_infer is None` 时写 `_fallback_output = torch.zeros_like(hidden_states)` 直接 return（M-1 为避免测试崩而加），decode 完全 no-op，TPS 只反映 GPU attention + logits + sampling，**不反映 MoE expert 计算**。
- 修复：`CPUMoEBackend._compute_expert_output_cpu_gptq(states, cpu_slot)` 调 `cpu_w4a32_matvec` 真算 CPU grouped W4A32；`submit_forward` 只有在子类已填 shape-matching `_fallback_output` 时才认为子类处理过，否则自己跑这条新路径。修后同样 prompt+32 token decode 跳到 10.43s（3.07 tok/s，**符合真实 W4A32 grouped 成本**）。
- 经验：
  1. **永远不要让 "fallback output = zeros" 看起来像正常执行路径**。至少 log 一个 warning 或标记 `computed=False` diagnostic。
  2. **任何 KPI 对比 baseline 要先单独验证 "baseline 真的在算"**：检查 output_text 合理性 + 检查 decode_seconds 是否与 model size 数量级一致（30B 模型 decode 不可能 < 1s/token）。
  3. dev_gate 现在有一条显式反回归规则：`decode_seconds >= 5.0`（32 token），防止 zeros-output 类 bug 混过 CI。


<!-- updated: 2026-04-22 18:10 -->

## PIMQuantizedRuntime 单 resident slot — 同 expert 的 3 个 projection 用不同 eid 会**互相覆盖**

- 现象：M-3 e2e diagnostics 聚合 `preload_hits=0, preload_misses=1,675,440`（128 万次不命中）。micro-bench 显示 `preload_hit+infer=2.20ms` vs `preload_miss+infer=3.63ms`，每次 miss 付 ~1.45ms host->DPU 权重传输。48 层 × 每层 8 expert × 每 expert 3 projection = 1152 次 miss/token，单 token 浪费 ~1.7s。
- 根因：`PIMQuantizedRuntime._resident_expert_id` 只追踪单个 expert_id。`PIMMoEBackend._run_expert_quantized_on_dpu` 为 gate / up / down 用 `base_eid ^ 0x1111.. / 0x2222.. / 0x3333..` 三个不同 eid，每次 preload 都把前一次挤出去 → 每个 infer 前都真实传输一次权重到 DPU。
- 修复（M-4.1）：DPU 内核对 output_dim 不敏感，host 端把 gate 和 up 的 W4A32 权重沿 row 轴 concat 成 `(2*output_dim, input_dim)` 一次性装入 MRAM，一次 DPU launch 同时算 gate + up，host 再 split。3 次 call → 2 次 call，DPU 调用 −33%，decode TPS +39%，数值 bit-exact。
- 经验：
  1. **"单 slot 常驻缓存" 在 MoE decode 下零命中率** — 因为 gate/up/down 三份权重在每个 expert 内都切一遍。任何 "最近一个 expert 常驻" 方案都毫无用处。
  2. **DPU 内核对 shape 的容忍度经常被 host 层低估**。具体到 UPMEM：`pim_quantized_kernel` 的 row loop 只要 output_dim 是偶数，对多大都不关心；把矩阵沿 output 轴 concat 是免费的性能优化。
  3. **proj level fuse 的下一步是 2-slot MRAM**（把 down 也常驻）— 需要改 DPU binary 的 MRAM 布局，留给 M-5。

<!-- updated: 2026-04-22 18:10 -->

## micro-bench 和 e2e 的 speedup **不能等价外推**

- 现象：M-4.1 的 fused gate+up micro-bench 显示 **2.98× 加速**（`7.57ms → 2.54ms/expert-pair`），但 e2e decode TPS 只拿到 **1.39× (+39%)**。
- 根因：e2e decode per-token = GPU attention + gate MLP + 48 × (MoE layer) + logit head。MoE layer 内部除了 gate+up fuse 省下的部分，还有 down projection call、Python glue、CPU-offloaded expert 的 CPU W4A32 forward、GPU expert forward、CUDA host sync 等非 fused 成本。fused 只砍掉其中一段。
- 经验：
  1. 任何 kernel/runtime 级 micro-bench 收益**至少打对折** 才对应 e2e 观测值。做 roadmap 估算时先算 "fused 那段在 e2e 里的 wall-clock 占比"，再把 micro-bench 数字乘这个占比。
  2. **观测端 to-token 延迟才是诚实的 KPI**。`decode_tokens_per_second` 是唯一不会骗人的数字；per-layer / per-call 的任何 timing 都要先用 e2e 数据验证它的占比。


<!-- updated: 2026-04-22 19:00 -->

## PIMQuantizedRuntime 的 0.96 ms/call preload overhead **是 DPU DMA，不是 Python**

- 背景：M-4 之后 preload hit ratio 仍然 0%。最直观的假设是 Python 层
  开销大（ctypes / torch padding / signature check 等），改成 C 侧批量
  就能赢。实测否决了这个假设。
- Micro-bench：
  - `_prepare_quantized_weights` (pure Python + torch copy) = **0.074 ms/call**
  - `pim_quantized_load_weights` ctypes (host→DPU weight DMA) = **0.96 ms/call**
  - `infer-only` (resident) = 2.31 ms/call
- 结论：preload miss 的成本 **95% 花在 DPU DMA 本身**，不是 Python。
  所以任何纯 Python 端的优化（批量 ctypes、padding cache、signature fast
  path）最多省 ~5% 左右。真正要省就得让权重**不再每 call 重传**，即 MRAM
  多 slot residency。
- 经验：优化前先测准具体热点在 Python 层还是 C/硬件层。M-5 dual
  runtime 就是在没测准这个分布前提下做的假设（以为 Python 开销大）；
  真机数据 6 行就排除了这个假设，避免再做类似的无用功。

<!-- updated: 2026-04-22 19:00 -->

## dual PIMQuantizedRuntime **单独不能降 MoE 跨-expert preload miss**

- 现象：M-5 把 `PIMQuantizedRuntime` 由单例拆成 gate_up + down 两个独立
  实例后，47/48 层成功分到不同 DPU rank pool，diagnostics 显示 dual
  landed，但 e2e decode_tps 0.309 vs M-4 的 0.317 —— 在 run-to-run 噪声
  范围内持平。`quantized_preload_misses_local = 23214 = total_dpu_calls`，
  **miss ratio 仍然 100%**。
- 根因：dual runtime 只避免**同一 expert 内** gate_up 桶和 down 桶的
  互相覆盖。但 MoE decode 的工作集是**每层 top_k 个不同 expert**
  (Qwen3 = 8)，每步对 runtime 来说都是 "new expert's bundle"，不管是
  1 个还是 2 个 runtime，MRAM 只能存 1 份权重 → 每 call 必 miss。
- 经验：
  1. "把单资源拆成两份" 的优化**只在工作集 ≤ 2** 时有效。MoE
     top_k ≥ 4 都需要 **N-slot** 而不是 **2-slot**。
  2. 任何 "更细粒度的 cache / residency" 设计要先估**工作集大小 vs
     slot 容量**的比例。≥ 10× 的时候考虑换策略而不是加 slot。
  3. 这次 M-5 的价值不在 perf，而在**排除**了 "dual runtime 就够了"
     这个假设，为 M-6 去改 DPU binary MRAM 布局提供了明确依据。
     publish 负结果也是结果。


<!-- updated: 2026-04-22 19:50 -->

## LRU slot hit 在 micro-bench 里 work，在 e2e 里 0% — **单例 runtime 被 N 层共享**

- 现象：`PIMQuantizedRuntime` M-6.1 加了 8-slot LRU 缓存，micro-bench 显示 4 expert round-robin 后反向 preload 4/4 hit（bit-exact），但真机 e2e (Qwen3 48 层 × 128 experts × top_k=8, 32 decode tokens) hit ratio = 0%。
- 根因：`PIMMoEBackend.__init__` 对每层调 `PIMQuantizedRuntime.get_shared(profile, rank_count)`，返回**同一个 process 级单例**。一次 forward pass 过 48 层，每层 top_k=8 expert，等效于对**同一个 8-slot LRU** 做 384 次 preload 请求；到下一个 decode step 同一层回来时，8 个 slot 已经全被下游层的 expert 覆盖。理论 hit 上限 = NUM_SLOTS / (num_layers × top_k) = 8/(48×8) = 0.2%。
- 修复（已 scope 到 M-7）：让 `get_shared` 的 key 包含 `layer_idx` 或 `layer_group_id`，把 48 层分成多组独立 runtime。或者每层独占一个 runtime（要求 DPU rank 资源足够）。
- 经验：
  1. **"cache 容量 vs 工作集"** 要按**实际并发访问模式**算，不是按单一 client 模型。一个 process 里多 consumer 共享一个 cache 时，工作集 = Σ per-consumer 工作集，不是 max。
  2. **micro-bench 和 e2e 的 hit ratio 可能差 3 个数量级**。不要用 micro-bench 的 hit ratio 外推到 e2e；至少要模拟 concurrent consumer 数量。
  3. M-6 是项目内第三个 "infra landed, null e2e" 的 milestone（继 M-2 kernel_mode=7、M-5 dual runtime）。这不丢人：排除假设 + ship 可用 infra 本身是路线图成果，但要诚实归类 + 不拿 "代码跑通了" 当 perf 胜利。

<!-- updated: 2026-04-22 19:50 -->

## DPU MRAM 多 slot 布局：`__host active_slot` + `dpu_push_xfer(offset_bytes)`

- DPU kernel C 如何用 runtime 动态 offset：在 kernel 里用 `__host uint32_t active_slot;` 暴露一个变量，算出 `const uint32_t slot_base = active_slot * CONST_PER_SLOT;` 然后所有 `mram_read((__mram_ptr void const *)(buffer + slot_base + other_offset), ...)` 就能根据 host 广播的值动态取位置。**编译期常量 `CONST_PER_SLOT` 必须把 MAX_* 总量除以 NUM_SLOTS 得出**，否则 slot offset 会超出 buffer 边界 → SILENT silent memory corruption。
- host 侧 `dpu_push_xfer(set, XFER_TO_DPU, "symbol", offset_bytes, size, DEFAULT)` 的 `offset_bytes` 参数完美对应 slot 偏移：`(size_t)slot_id * WORDS_PER_SLOT * sizeof(uint32_t)`。这样 host 只传输**当前 slot 需要的那一段**，不是每次重传全部。
- 三个 `__mram_noinit` buffer 都要按同一套 slot offset（`qweight_mram / scales_mram / lut_mram`）— 这里我 sed 批量替换时漏一个都会导致"slot 0 权重 + slot 1 scales"混着算，数值错了还难追。解决：写 unit test 覆盖 "LRU overflow 后 evicted slot 的数值不再使用"这类边缘情况。
- `dpu_broadcast_to(set, "active_slot", 0, &slot_id, sizeof(slot_id), ...)` 很便宜（一个 uint32），**每次 run 前做一次是正确的且不影响性能**（实测 < 0.05 ms / call）。不要觉得 broadcast 贵就跳过写它。


<!-- updated: 2026-04-22 20:50 -->

## 诊断 Python 隔离效果**之前**必须先审 C `.so` 的 `static` 全局变量

- 现象：M-5 dual runtime / M-6 multi-slot LRU / M-7 per-layer-group scoping 三次连续 null perf。每次诊断都指向"工作集 >> slot 容量"等数学 argue，但**真正的共同根因**只有一个 —— `nano_ktrans/kernels/pim_native/host_quantized_bridge.c` 的 ~20 个 `static` C 全局（g_set, g_initialized, g_input_dim, g_slot_loaded_mask, g_lut_i16_shards, ...）。Python 多个 `PIMQuantizedRuntime` 实例拿到**不同 Python 对象**，但 ctypes `self._lib` 都指向同一个 dlopen 的 `.so`，所以这些 `static` 只有一份。`pim_quantized_init()` 第二次见到 `g_initialized==true` 直接 return 0，第 2~N 个 Python runtime 被**静默 alias 到第一个的 DPU rank pool + 同一套 MRAM buffer**。
- 为什么 M-5/M-6 没 crash：所有调用是严格串行（一次完整 `load_weights + run`），全局 `g_input_dim` / `g_output_dim` 在下次被覆盖前刚好用完。M-7 speculative preload 打破了这个顺序（在同一"expert"里连续 load gate_up + load down 再 run），`g_output_dim` 被 down 的 load 写掉之后 gate_up 的 run 用错了 shape → `outputs = torch.empty(batch, wrong_shape)` → `munmap_chunk(): invalid pointer` heap corruption。
- 经验：
  1. **任何 "我要做 Python 层 N 个独立 state" 的方案，先 `grep '^static ' src.c`** 验证底层是不是真的允许 N 个。Python 对象多实例≠底层 C 多实例。
  2. Python 诊断字段（像 M-5 的 `quantized_runtime_down_distinct=47/48`）只反映 Python 层拿到**不同对象**，不保证底层物理资源**不同**。要验证真隔离，**切换 `g_set` 的 `dpu_get_nr_dpus()` 返回值看是否变化**，或者审底层函数是否 return early。
  3. **连续 4 个 null perf milestone 共享一个根因**应该早被察觉。下次如果 3 个 milestone 同维度失败，必须停下来审底层架构，而不是继续在 Python 层加新参数。
- 修复 scope：M-8 把 `host_quantized_bridge.c` 改成 handle-based（`pim_q_ctx_t*` 结构体存所有原 static state，API 第一个参数都是 `ctx`）。Python 端每 runtime 持 `c_void_p`。

<!-- updated: 2026-04-22 20:50 -->

## `pim_quantized_init()` 的 "`if (g_initialized) return 0`" 早退是**静默 bug**，不是 idempotency

- 这行代码本意：允许 `init` 幂等调用。但在 Python `get_shared()` 路由下，"第 2 次调 init" 发生在**想分配第二套 DPU rank 的时候**。早退意味着第二次调用者传的 `rank_count` / `profile` 参数被**无声丢弃**，第二个 runtime 获得的其实是第一个 runtime 的 DPU set。
- M-5 / M-6 / M-7 的 "isolation landed 47/48 layers" 诊断字段都建立在这个假设错上。
- 正确的 idempotency：要么 return ERROR（let caller 知道要先 shutdown），要么**真的按新参数再分配一份 context**。前者简单，后者需要 handle-based API。
- 提醒：任何 "第二次调用静默成功" 的系统级 init 函数都是**设计坑**。用 Python mock 测到 return 0 就以为成功，实际物理资源早就乱了。


<!-- updated: 2026-04-23 11:00 -->

## "profile" 参数在 Python cache key 和 UPMEM `dpu_alloc_ranks` 里**必须分开**

- 现象：M-7 第一次 crash 是 `munmap_chunk`，修了 handle 以后再跑 raw ctypes 真实 alloc 两个 runtime → UPMEM 报 "`dpu_alloc_ranks failed: invalid profile`"。M-5~M-7 一直用类似 `profile="|gate_up|g0"` 的 Python 逻辑键当作 UPMEM profile 传进去，之前没 crash 只是因为 `g_initialized==true` 早退让那个非法字符串**从没真正送到 UPMEM** 过。
- 修复（M-8）：`PIMQuantizedRuntime.get_shared(*, profile="", rank_count, instance_key="")`。`profile` 是真的传给 `dpu_alloc_ranks` 的 UPMEM profile 字符串（必须空或合法 UPMEM 字符串如 `"backend=hw"`）；`instance_key` 是 Python `_shared` dict 的键（可以是任意逻辑标签像 `"|gate_up|g7"`）。`instance_key=""` 时默认回退到 `profile` 保持向后兼容。
- 经验：**任何用作 "key" 的字符串如果同时流到另一个系统当作配置，就是一个时间炸弹**。在设计接口时如果两种用途有任何可能分叉（哪怕最初都传同一个值），最好从第一天就拆成两个参数。

<!-- updated: 2026-04-23 11:00 -->

## MoE 路由 temporal locality 可能**远低于直觉**：Qwen3 实测 hit ratio = 0.1%

- 现象：M-8 修完底层后，32 独立 runtime pool + 8-slot LRU + prefill-time 预热 96 个 hot expert，真机 32 token decode 测出 `preload_hits_local = 24 / misses = 23306 = 0.1%`。预期是 20-30%（ADR-002 §15.7）。
- 分析：每层在 prefill 统计出 2 个 hot expert 预热到 slot，32 decode step 之后只有 24 次再次激活，即这 96 个预热 expert 在 decode 阶段平均只被 reuse 了 24/96 = 0.25 次。相邻 decode step 之间的 top_k=8 集合几乎零重叠。
- 可能原因：
  1. Prompt 太短（14 token prefill）→ prefill 统计到的 hot 不代表 decode 分布；
  2. Qwen3-30B-A3B 的 routing 本来就追求分散（MoE 设计目标之一是 load balance）；
  3. decode 阶段 KV cache 累积使得 router 输入空间移动得很快。
- 后果：**slot-based LRU 无论多少 slot 都救不回** low locality。要真在 UPMEM 上赢 CPU 必须：
  (a) 放弃"靠 cache 命中降 DMA 次数"的思路，改成 mixed-precision expert 让单次 DMA 更便宜；
  (b) 或者 `dpu_launch(DPU_ASYNCHRONOUS)` 让 DMA 和 GPU 计算 overlap 掉。
- 经验：**在做 caching / prefetching 投资之前先用直方图量化 temporal locality**。`jaccard(active_experts(t), active_experts(t-1))` 是一行代码能加上的 diagnostic，值得在 M-1 就做。

<!-- updated: 2026-04-23 11:00 -->

## `ctx->handle` API 的 `shutdown()` 必须先置 `self._handle = NULL`，再调 C 端

- 现象：M-8 handle-based 实装里，如果直接 `self._lib.shutdown(self._handle)` 然后**不清零 self._handle**，第二次调 `shutdown()` 会把野指针再送给 C 端的 `free(ctx)` → double free → heap corruption。
- 修复：Python 端 `def shutdown(self): handle = self._handle; self._handle = ctypes.c_void_p(0); if handle: self._lib.shutdown(handle)`。C 端 `pim_quantized_shutdown(NULL)` 早退。
- 经验：**handle 语义的 API 任何时候都要在 free 前把 handle 置空**。这不是可选的防御性编程，这是正确性。
