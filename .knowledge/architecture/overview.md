---
created: 2026-04-07
updated: 2026-04-16
tags: [architecture]
---

# 系统架构总览

> `nano-ktrans` 当前的核心目标是把 Hybrid MoE 推理路径拆成可读、可验证、可替换的分层结构，并进一步演进为“主干层常驻 GPU、专家在 GPU/PIM 间动态迁移”的两级专家内存体系。

## 组件结构

- `nano_ktrans/llm.py`
  统一构建入口，负责加载 Hugging Face 配置、映射到通用 `GenericMoeConfig`，并把 offload backend 参数传给模型。
- `nano_ktrans/models/mixtral.py`
  主模型骨架。当前用于承载 Mixtral、Qwen2-MoE、Qwen3-MoE 这几类共享的 decoder/MoE 结构。
- `nano_ktrans/layers/hybrid_moe.py`
  Hybrid MoE 调度层。当前负责区分 hot experts 和 offloaded experts，并在 GPU 输出和 offload 输出之间合并结果；现已支持在 `decode` 阶段消费迁移计划并动态 materialize / demote GPU experts，是动态专家驻留与迁移的核心调度层。
- `nano_ktrans/kernels/offload_backend.py`
  offload 扩展点，定义 `ExpertOffloadBackend` 抽象和 backend 命名规范化逻辑；当前已承载迁移计划排队与分层 migration manager，后续还需要异步数据面接口。
- `nano_ktrans/kernels/weight_loader.py`
  现已支持按层按 expert 加载单个专家权重，供 decode 阶段的动态 GPU promotion 使用。
- `nano_ktrans/layers/expert_mlp.py`
  抽出的共享 expert 模块定义，供模型初始化和运行时 expert materialization 复用，避免 `HybridMoE` 与 `mixtral.py` 之间的循环依赖。
- `nano_ktrans/kernels/cpu_moe.py`
  当前唯一的数值正确 offload backend，实现 CPU fallback expert 计算。
- `nano_ktrans/kernels/pim_moe.py`
  当前的 PIM backend。`pim_shadow` 仍然只做统计，而 `pim` 已能把 expert 的线性投影送到真实 DPU 上执行；另有实验性 fused expert 路径，可把 `gate/up/down + SiLU` 整体送到 DPU。
- `nano_ktrans/kernels/pim_linear_runtime.py`
  Python 侧的真实 PIM 线性执行桥，负责构建并加载共享库，调用 UPMEM host bridge，并把 DPU 线性结果回传给 `PIMMoEBackend`。
- `nano_ktrans/kernels/pim_expert_runtime.py`
  Python 侧的 fused expert 执行桥，负责加载 expert host bridge，并将完整 expert MLP 子图提交给 DPU。
- `nano_ktrans/kernels/quantized_ops.py`
  量化算子级基线实现，当前提供 CPU W4A32 operator-only matvec 与 synthetic quantizer，用于脱离完整 decode 链路单独比较量化矩阵向量乘。
- `nano_ktrans/kernels/pim_quantized_runtime.py`
  Python 侧的 quantized matvec 执行桥，负责将 packed int4 qweight 与 scales 持久化加载到 PIM，再重复执行 operator-only matvec。
- `nano_ktrans/kernels/pim_native/`
  DPU 原生代码目录，当前同时包含 linear kernel、fused expert kernel、quantized matvec kernel、对应 host bridge 和 build 脚本。
- `benchmarks/benchmark_inference.py`
  统一 benchmark 入口，用于比较 `cpu`、`cuda`、`cuda_cpu_offload`、`cuda_pim_shadow`。
- `benchmarks/pim_microbench/`
  独立的 UPMEM host/DPU microbenchmark，用来测真实硬件上的传输与整数 kernel 指标。
- `benchmarks/benchmark_quant_matvec.py`
  独立 operator-only benchmark，支持 synthetic W4A32 或真实 GPTQ expert projection，直接比较 CPU 与 PIM 的单算子矩阵向量乘。

## 数据流

1. `LLM` 从 checkpoint 读取 Hugging Face 配置，并通过 `GenericMoeConfig` 适配到统一结构。
2. decoder 层中的 `HybridMoE` 在 GPU 上完成路由。
3. 命中的 hot experts 直接在设备侧执行。
4. 冷 expert 请求通过 `ExpertOffloadBackend` 派发。
5. `CPUMoEBackend`、`PIMMoEBackend(pim_shadow)` 或 `PIMMoEBackend(pim)` 处理 offloaded expert 请求。
6. offload 输出与 GPU 输出合并，再交还推理引擎继续 decode。

## 量化算子级实验路径

1. `WeightLoader` 从 GPTQ checkpoint 中读取某个 expert projection 的 `qweight / scales / g_idx`。
2. 当前最小支持路径假设：
   - `bits=4`
   - `sym=true`
   - `group_size=128`
   - 顺序 `g_idx`
3. CPU 基线通过 `GPTQLinearWeight.dequantize()` 还原权重，再执行 `F.linear`，作为 operator-only baseline。
4. PIM quantized runtime 先将 packed int4 权重与 scales 持久化加载到 DPU：
   - `pim_quantized_load_weights(...)`
   - 后续 repeated runs 只走 `pim_quantized_run(...)`
5. DPU quantized kernel 当前在 DPU 上执行：
   - int4 unpack
   - 逐 group scale 反量化
   - dequantized weight 与 float32 input 的矩阵向量乘
7. quantized runtime 现已暴露分项 profile：
   - `input_transfer_seconds_avg`
   - `launch_seconds_avg`
   - `output_transfer_seconds_avg`
   - `runtime_total_seconds_avg`
8. 当前真实 GPTQ operator-only 结果表明，W4A32 PIM 路径的大头时间集中在 `launch_seconds_avg`，说明主瓶颈是 DPU kernel 执行而不是 host 端输入/输出搬运。

6. 这条路径的目标不是先打通完整 MoE，而是先回答一个更直接的问题：
   - “在同一算子上，使用持久化驻留的 W4A32/GPTQ 权重时，PIM 是否可能比 CPU 更快？”

### 当前 decode 迁移执行链

1. `prefill` 或前几步 `decode` 先由 `DynamicExpertScheduler` 观察路由热度并生成迁移计划。
2. `HybridMoE` 将迁移计划排入 backend 的 `migration_manager`。
3. 到下一次 `decode` 进入本层时，`HybridMoE` 先 drain 本层队列。
4. 对 `PIM/CPU -> GPU` 的 promotion：
   - 使用 `ExpertWeightLoader.load_expert()` 从 checkpoint 读取单 expert 权重
   - 运行时构建 expert module
   - 注入 `gpu_experts` 并更新 `gpu_experts_mask`
5. 对 `GPU -> PIM/CPU` 的 demotion：
   - 从 `gpu_experts` 移除模块
   - 更新驻留表与 backend mask
6. 之后再进入本层正常的 GPU/offload 混合计算。

这一步仍是同步、逐 expert 的最小数据面，尚未包含真实 GPU<->PIM 异步传输，也没有与计算 overlap。

### 当前 prefill 预取链

1. `prefill` 阶段 scheduler 先根据路由热度生成候选 promotion plan。
2. `HybridMoE` 对其中 `dst=GPU` 的 expert 发起预取请求。
3. `ExpertMaterializationManager` 将单 expert 权重加载到 CPU staging cache。
4. 到 `decode` 真的需要 promotion 时：
   - 优先命中 staging cache
   - 再构建 GPU expert module
   - 避免在 decode 关键路径上重新扫 safetensors
5. 如果当前层 GPU resident experts 已达到 budget：
   - 先在非活跃 resident experts 里按 hotness 选择最冷的 victim
   - 执行 `GPU -> PIM/CPU` 的运行时 eviction
   - 再为新的热点 expert 执行 promotion
6. promotion 队列会按两级优先级排序：
   - staging cache 已就绪的 expert 优先
   - 其次是当前 step 已活跃的 expert
   - 同优先级内按 hotness 从高到低
7. 对没有在本步立刻 promotion 的候选 expert，系统仍会尽早发起预取，为后续 decode step 预热 staging cache。
8. 调度器当前还会记录每个 expert 的访问时间和最近一次驻留变化时间，为后续 anti-thrashing 策略提供元数据。
9. 调度器现在区分“逻辑步长”而不是简单依赖真实 token step：
   - prefill 可用更大的 `step_stride_prefill`
   - decode 保持更细的 `step_stride_decode`
10. 调度器也支持 `prefill_collect_only`：
   - prefill 只更新热度和预取候选
   - 不直接发出迁移计划
   这样更符合“prefill 负责探测和预热，decode 负责实际迁移”的目标。
11. decode 侧现在还支持更保守的 promotion 模式：
    - 若 `decode_require_prefetch_ready=true`
    - 则只消费 staging cache 已就绪的 promotion
    - 未就绪的 expert 先 defer，并继续预热
    这更接近真正的“迁移未完成时不要阻塞计算”的 overlap 语义。
12. migration manager 现在会按 expert 做队列去重：
    - 同一 layer / expert 的新 op 会覆盖旧 op
    - 避免同一 expert 在短时间内重复排队
    - 并记录原始提交量与去重量，便于后续评估控制面开销
13. scheduler 现在还支持“候选预取”路径：
    - 即使当前 step 没有立即可执行的 migration op
    - 也可以基于 offloaded experts 的 hotness，提前挑出一批候选 expert 做 staging prefetch
    - 为后续 decode promotion 提前铺垫数据面准备
14. scheduler 控制面现在还支持 profile 预设：
    - `baseline`: 保持当前默认行为
    - `overlap_safe`: prefill 只收集/预热，decode 只消费 prefetch-ready promotion
    - `eager`: 更积极地预热与迁移，便于观察控制面上限
15. benchmark 现在会在保留原始 `offload_diagnostics` 的同时，再输出一份调度摘要：
    - `prefetch_requested / enqueued / materialized`
    - `decode_prefetch_hits / misses`
    - `runtime_evictions`
    - `runtime_deferred_for_prefetch`
    - `migration_total_enqueued_ops / deduped_ops / drained_ops`
    这样可以直接比较不同调度策略的 overlap 相关效果，而不必人工汇总每层诊断。
16. migration manager 现在还会跟踪每个 expert 的 lifecycle：
    - `queued`
    - `prefetching`
    - `ready`
    - `deferred`
    - `applied`
    目前这仍主要服务于诊断和后续异步数据面设计，但已经把“迁移请求”和“迁移执行状态”区分开了。
17. `decode_require_prefetch_ready=true` 的执行语义也更严格了：
    - decode 入口先查看 pending migration queue
    - 只提取已经在进入本层前就处于 `ready` 的 promotion
    - 未 ready 的 promotion 保持在队列里，并被标成 `deferred`
    这样更接近真正的“只消费已完成迁移”的 overlap 路径，而不是“先同步预热，再同一步立即消费”。
18. `ExpertMaterializationManager` 现在支持 `poll_ready()`：
    - 后台 prefetch future 一旦完成，会被主动转成 CPU staging cache 命中
    - `HybridMoE` 在进入本层前会先轮询 ready 结果，再把对应 migration lifecycle 更新成 `ready`
    这让 `ready` 不再只依赖 decode 路径上的同步 `is_ready()` 检查，而开始具备更像后台 completion 的语义。
19. migration manager 现在还提供了 `take_ready_layer()`：
    - 直接按 lifecycle=`ready` 抽取当前层可消费的 promotion
    - 其余 pending op 留在队列中
    这样 decode 主路径不再自己实现 ready subset 过滤，控制语义也更集中。
20. materialization 侧的 ready 轮询也做了收敛：
    - 预取 future 完成时先进入 completion queue
    - `poll_ready()` 只消费 completion queue，而不是每次扫描所有 future
    这让后台 prefetch 的完成路径更接近真正的 completion event。
21. ready 轮询入口现在也从 `HybridMoE.forward()` 上移到了 engine/model 层：
    - `SimpleEngine.prefill()` / `decode_step()` 进入模型前先统一刷新一次
    - `MixtralModel.refresh_offload_state()` 再遍历各层 `HybridMoE`
    这样同一个 token step 只做一次全局 ready 刷新，而不是每层重复触发。
22. `SimpleEngine` 现在还抽出了统一 `_refresh_offload_state()` hook：
    - full prefill
    - chunked prefill
    - decode
    都通过同一入口触发 runtime ready 刷新，避免三处逻辑继续分叉。
23. `MixtralModel` 现在还会累计 offload refresh 诊断：
    - `offload_refresh_calls`
    - `offload_refresh_ready_total`
    这样 benchmark 可以直接观察“每个 token step 刷新了多少次、每次收敛了多少 ready expert”。
24. layer 级 refresh 现在也有空队列短路：
    - 如果某层既没有 pending future，也没有 completion queue 中的 ready item
    - `HybridMoE.refresh_offload_state()` 会直接返回
    这避免了在大多数 step 上做无意义的 Python 轮询。
25. benchmark 现在支持 scheduler profile sweep：
    - 单次运行可以同时比较 `baseline / overlap_safe / eager`
    - 每个结果都会带自己的 `scheduler_profile` 和摘要统计
    这样可以更直接地观察不同调度策略的端到端差异。
26. token-step 级 refresh 现在已经被收敛成 `MigrationPipelineRuntime`：
    - `SimpleEngine` 在 full prefill、chunked prefill、decode 前统一调用模型级 refresh hook
    - `MixtralModel.refresh_offload_state(phase=...)` 不再只是简单累加 ready 数，而是通过 runtime 统一推进每层的 offload pipeline
    - 当前 runtime 会在进入模型前完成两件事：
      - 将 materialization completion queue 中的 ready prefetch 转成 `READY`
      - 在 decode 阶段尽量把 `READY -> APPLIED` 的 promotion 提前完成
    - 这让“迁移准备”和“层内真正计算”进一步解耦，主线程进入 `HybridMoE.forward()` 时，更多热点 expert 已经提前进入 GPU resident set。
27. 当前 `MigrationPipelineRuntime` 仍然只是最小前台运行时：
    - 它依旧由 token-step hook 驱动，不是后台线程
    - 但系统已经从“每层 forward 内部边算边处理 ready”推进到“step 进入前统一推进流水线”
    - 这为后续真正引入后台 worker、事件队列和 GPU<->PIM resident 迁移执行器提供了清晰挂点。
28. pipeline runtime 现在还会在 decode 进入模型前主动 prime pending promotions：
    - 对 `dst=GPU` 的 pending op，若尚未 ready，会先统一提交 prefetch
    - 若启用了 `decode_require_prefetch_ready`，则会提前把这些 op 标成 `deferred`
    - 这样“排队中的热点专家”不必等到层内 forward 才第一次被预热，`queued -> prefetching/deferred` 已开始前移到 token-step runtime
29. 因此当前流水线已经粗略分成三段：
    - token-step runtime：prime pending promotion、poll completion queue、推进 ready promotion
    - layer forward：消费剩余未就绪/未提前应用的迁移，并执行真实 GPU/offload 混合计算
    - backend/offload tier：继续承担 CPU/PIM 数值计算
    这比早期“所有迁移控制都塞在 `HybridMoE.forward()` 里”更接近真正的流水线系统。
30. benchmark / example 路径现在也已经能显式接通 background offload worker：
    - `benchmark_inference.py` 和 `example.py` 新增了 `--enable-background-offload-worker`
    - benchmark 单次 generation 会在 prefill/decode 前启动 worker，结束后停止
    - 因而真实宿主机上的 `cuda_pim` benchmark 已经可以直接跑到后台 `prefetching -> ready -> warmed -> activated` 路径，而不再只停留在前台 refresh hook
31. 这意味着当前系统已经不只是“模型内部有 background worker 骨架”，而是：
    - `LLM.generate()` 可使用后台 worker
    - `benchmark_inference.py` 的真实测量路径也可使用后台 worker
    - 后续性能分析终于可以直接验证后台 worker 对真实 `cuda_pim` benchmark 的影响
30. resident tier 到 staging cache 的路径也开始收敛：
    - `ExpertOffloadBackend` 现在可以导出 resident expert 权重
    - `HybridMoE._request_prefetch()` 会优先尝试从 offload backend 直接拿 resident weights
    - `ExpertMaterializationManager.stage_expert()` 则把这些 resident weights 直接放入 CPU staging cache
    这意味着 GPU promotion 的预热路径，已经开始从“checkpoint -> staging”转向“resident tier -> staging”。
31. 当前这条 resident staging 路径的意义在于：
    - 对 CPU backend，它避免反复回 safetensors 扫描本来就常驻在 backend 内存里的专家
    - 对 PIM backend，它为后续真正的 “PIM resident -> GPU resident” 数据面留出了统一接口
    - 但现在它仍是同步 CPU staging，不是独立的 PIM->GPU 异步搬运
32. 运行时现在还新增了一个更短期的缓存层：warm expert cache。
    - 当某个 GPU resident expert 因 budget 或 eviction 被降回 offload tier 时
    - 其构建好的 expert module 不会立刻丢弃，而是先保留在 CPU 侧 warm cache
    - 若后续很快再次被 promote 回 GPU，则可直接复用该 module，而不必重新 build + load weights
33. 因此当前专家迁移链已经开始呈现三层缓存/驻留结构：
    - offload resident weights：CPU/PIM backend 内部保存的长期驻留专家权重
    - CPU staging / warm cache：为 promotion 准备的短期中间层
    - GPU resident experts：当前 token window 中的热点执行层
    这虽然还不是最终的异步分层存储系统，但已经更接近真正的缓存层次结构，而不是单纯“现用现建”。
34. warm cache 现在又前移了一步：
    - 当某个 promotion 已经进入 `READY`，但还没正式 `APPLIED`
    - token-step pipeline 会尝试先把对应 expert module build 出来并放进 warm cache
    - 这样真正执行 `READY -> APPLIED` 时，更多情况下只是在 warm cache 和 GPU resident set 之间搬运，而不是临场 build module
35. 这意味着当前 decode 前的 pipeline 已经在逐步承担三类工作：
    - `queued -> prefetching/deferred`
    - `prefetching -> ready`
    - `ready -> warm-prebuilt`
    主路径里剩下的更多是“消费这些已准备好的结果”，而不是从零开始搭建迁移对象。
36. `ready -> warm-prebuilt` 这一步现在也更贴近流水线语义：
    - prebuild 固定在 CPU 侧完成
    - promotion 时才把 warm module 激活到目标 device
    - 因此 pipeline hook 负责“对象准备”，而 decode promotion 更像“最后一跳激活”
    这比一开始直接在 GPU 上 build module 更适合后续接真正的异步拷贝和独立 stream。
37. 为了把这个阶段描述得更清楚，migration lifecycle 现在显式增加了 `warmed`：
    - `ready` 表示权重/数据已经可用
    - `warmed` 表示对应 expert module 也已经预构建并进入 warm cache
    - `applied` 则表示它已经真正进入 GPU resident set 并可被本步计算消费
    这样后续 benchmark 和诊断就能分辨系统瓶颈到底还卡在“数据到位”，还是已经推进到了“对象构建完成但尚未激活”。
38. 当前 token-step pipeline 又往前拆出了一层 `activated`：
    - `warmed` 表示 module 已在 CPU warm cache 里预构建
    - `activated` 表示该 module 已执行 device transfer，进入目标 device 上的 warm cache
    - `applied` 才表示它真正被插入 `gpu_experts` 并更新 resident set
39. `activated -> applied` 这条后半段流水线现在也开始分层：
    - `apply_candidate_queue`：保存已进入 `ACTIVATED`、可进入 resident commit 的候选 expert
    - `apply_commit_queue`：保存已被选中、等待 staged resident commit 的候选
    - `apply_commit_ready_cache`：保存 commit queue 中已经完成 source/module 解析、可被直接 resident commit 的 ready entry
    因此 resident commit 现在不再只是“从 activated cache 直接逐 expert 注入 resident set”，而是显式拆成 staged queue 和 ready cache。
40. 当前 background tick 对后半段流水线的语义是：
    - 可以在同一 tick 内把新 `ACTIVATED` expert 推进到 `apply_candidate_queue`
    - 再推进到 `apply_commit_queue`
    - 再完成 `apply_commit_ready_cache` 的 resolve
    - 但真正 resident commit 只消费 tick 开始前已存在的 staged commit 候选
    这样后台 enqueue/resolve 与 resident commit 的边界更清楚，避免 background tick 在同一轮里对同一 expert 同时 enqueue 和 commit。
41. resident commit 的最后一段现在也开始具备真正的 batch 语义：
    - `apply_commit_ready_cache` 中的 ready entry 会先按批量方式把 module 注入 `gpu_experts`
    - 并统一批量更新 `gpu_experts_mask`
    - 然后再逐 expert 写回 residency、history 和 migration lifecycle
    因此 `apply_commit_queue -> resident set` 已不再是纯逐 expert 注入，而是“batch module commit + per-expert metadata finalize”的两段式提交。
42. resident commit 的 staged queue 现在又往前拆出了一层 `apply_commit_batch_queue`：
    - `apply_commit_queue`：保存 staged commit 候选
    - `apply_commit_ready_cache`：保存已完成 source/module 解析的 ready entry
    - `apply_commit_batch_queue`：保存本轮真正可进入 resident commit 的 ready batch
    这意味着后半段现在已经演化成：
    `apply_candidate_queue -> apply_commit_queue -> apply_commit_ready_cache -> apply_commit_batch_queue -> resident set`
    后台 worker 可以先把 ready 的 staged commit 候选推进到 batch queue，resident commit 再消费这批 ready batch。
39. resident commit 现在也被拆成两段 staged queue：
    - `apply_candidate_queue`：承接所有已经进入 `ACTIVATED` 的候选 expert
    - `apply_commit_queue`：只承接被当前 batch policy 选中的 staged commit 候选
    - 之后才进入真正的 resident set commit
    这意味着后半段流水线现在已经从“activated 后立刻 opportunistic apply”推进成了“候选排队 -> staged commit -> resident 注入”的三段式结构。
40. 当前 background worker 与前台主路径的职责边界也因此更清晰：
    - background path 主要负责 `prefetching -> ready -> warmed -> activated -> apply_commit_queue enqueue`
    - foreground path 主要负责消费 `apply_commit_queue` 并做最终 resident commit
    - 这样后续再把 resident 注入继续收成真正的 per-layer batch commit 时，不必再从候选筛选阶段回退重构。
41. `apply_commit_queue` 现在也拥有独立 queue policy：
    - queue 有自己的 `limit`
    - 超预算时按 hotness 和 lifecycle 选择更冷的 staged commit victim
    - 并单独统计 `apply_commit_queue_evictions`
    这样后半段 resident commit 已经不再只是“复用 apply candidate queue 的 budget 信号”，而开始具备独立的 staged commit 缓冲层语义。
42. `apply_commit_queue` 现在还拥有独立 controller 信号：
    - `apply_commit_queue_pressure`
    - `apply_commit_queue_pressure_step`
    - `apply_commit_queue_pressure_ema`
    - `apply_commit_queue_budget_backoff`
    这些信号会与 candidate queue / prepared tier 一起作用于 activation、prebuild 和 prefetch 的 aggressiveness，因此系统现在已经不只感知“前半段 prepared tier 是否拥塞”，也开始感知“后半段 staged commit 是否拥塞”。
43. runtime 诊断现在也显式区分 background worker 对两段 queue 的推进量：
    - `offload_background_apply_queue_enqueued_total`
    - `offload_background_apply_commit_queue_enqueued_total`
    因而 benchmark/sweep 已能分辨后台线程究竟是在把 expert 推到 activated candidate queue，还是已经继续推进到了 staged commit queue。
39. prepared tier 当前不再只有静态预算：
    - `prepared_cache_limit` 表示配置上的 prepared 总预算
    - `effective_prepared_cache_limit` 表示运行时在当前压力下真正允许保留的 prepared expert 数
    - 当 prepared-cache 重平衡压力持续偏高且 activation stage bonus 偏低时，系统会临时收缩 effective prepared budget，优先减少低价值 warm/activated candidate 的保留
40. 因此 prepared tier 当前已经形成三类闭环信号：
    - `prepared_cache_rebalance_pressure`：prepared 层内部因为预算冲突而发生的回退压力
    - `prepared_cache_activation_stage_bonus`：更偏向保留 activated 还是 warm candidate
    - `cold_promotion_penalty`：真正 apply 时仍落到 cold path 的压力
    这三类信号开始共同影响 activation / prebuild aggressiveness 与 effective prepared budget，是后续演进为完整 per-layer controller 的基础。
41. 当前 effective prepared budget 又向前走了一步：
    - controller 不再只支持“收缩 1 个 slot”的单级退让，而是通过 `prepared_cache_budget_backoff` 表达多级收缩
    - `prepared_cache_rebalance_pressure` 越高，backoff 级别越高
    - `prepared_cache_activation_stage_bonus` 越高，说明 activated tier 越值得保留，会抵消部分 backoff
    - `cold_promotion_penalty` 越高，说明冷路径 promotion 仍多，会继续抵消 backoff，避免 prepared tier 被压得过小
    这使 prepared tier 已从“静态上限 + 局部 heuristic”进一步走向真正的小型闭环控制器。
42. 这套 controller 现在已不只影响 cache 容量：
    - `effective_prepared_cache_limit` 控制 prepared tier 最多保留多少 warm/activated candidate
    - `adaptive_activation_limit` 控制每步最多推进多少 warmed candidate 到 activated
    - `adaptive_prebuild_limit` 控制每步最多准备多少 ready candidate
    - 当 prepared budget backoff 提高时，后两者也会同步受到约束；但若 `cold_promotion_penalty` 偏高，又会重新放宽，避免为了节省 prepared slots 反而导致更多 cold promotion
    因此 prepared tier 当前已经形成“容量控制 + 候选推进 aggressiveness”一体化的最小闭环。
43. `prepared_cache_rebalance_pressure` 的语义也已调整：
    - 早期版本按静态 prepared budget 归一，更像“累计 eviction 数 / cache 大小”
    - 现在改为优先按 `pipeline_ticks` 归一，更接近“每步平均回退压力”
    - 这样在长 decode 运行中，controller 读到的是更稳定的 step-level 压力，而不是单纯随时间累计放大的总量
    这为后续把 controller 从静态 heuristic 推进到真正的滑动窗口或 EMA 反馈打下了基础。
46. 当前后台 worker 的后半段路径也开始显式化：
    - `ACTIVATED` expert 现在会先进入 `apply_candidate_queue`
    - background pipeline 负责 `activated -> apply queue enqueue`
    - resident commit 再由前台/后台从 apply queue 消费
    这让系统从“看到 ACTIVATED 就 opportunistic apply”推进成“显式 staged commit”的结构，更接近后续真正的 apply queue / batched resident commit。
47. 当前 apply queue 已经是独立于 activated cache 的一层控制面结构：
    - 有自己的 queue budget
    - 有基于 hotness 和 lifecycle 的 victim 选择
    - 有 `apply_queue_enqueued / committed / pruned / evictions` 诊断
    这让后台 prepared 阶段和前台 resident commit 之间开始有清晰的缓冲边界，而不再只是直接从 activated cache 命中就提交。
48. 当前 background pipeline 与 apply queue 的边界也进一步收紧：
    - background tick 负责把 `ACTIVATED` candidate 推入 apply queue
    - 但不会在同一个 tick 里立刻消费刚入队的 candidate
    - resident commit 仍由后续 staged commit 路径完成
    这样系统更接近“后台准备、前台提交”的分层执行语义，而不是单 tick 内再次退化成 opportunistic apply。
44. 当前 prepared pressure 已进一步拆成三类信号：
    - `prepared_cache_rebalance_pressure`：累计平均压力，反映长期拥塞
    - `prepared_cache_rebalance_pressure_step`：本步压力，反映瞬时抖动
    - `prepared_cache_rebalance_pressure_ema`：对 step 压力做平滑后的中期趋势
    prepared-tier controller 目前已把累计压力与 EMA 一起用于 budget backoff；这让 controller 开始具备最小的“趋势感知”能力，而不只是对单次抖动做反应。
45. 这些 controller 信号当前已经与 benchmark 生命周期对齐：
    - `reset_offload_diagnostics()` 会在每次单 run 前清零 prepared-tier 的 pressure/EMA 相关状态
    - 因此单次 benchmark 里的 summary/profile sweep 看到的是该次运行自身的 prepared pressure 轨迹，而不是跨 run 累积值
    这保证了后续基于 sweep 结果做 controller/profile 调优时，信号具备可比性。
    这样 `HybridMoE.advance_offload_pipeline()` 现在已经可以按 `queued -> prefetching -> ready -> warmed -> activated -> applied` 的顺序推进一次 promotion。
39. 这也让“前台 pipeline”里的工作边界更清晰了：
    - resident export / safetensors prefetch 负责准备权重
    - prebuild 负责准备 module 对象
    - activation 负责最后一次 CPU->device 拷贝
    - applied 负责真正切换执行路径
    当前它们仍然是 token-step hook 中的同步阶段，但系统已经具备把 activation 单独剥离成后台 worker/stream 的结构基础。
40. 为了减少 `activated -> applied` 的重复开销，当前运行时又加了一层 activated cache：
    - `warmed` expert 完成 device transfer 后不会立刻插入 `gpu_experts`
    - 而是先放进 device-side activated cache
    - promotion 真正发生时优先从 activated cache 命中，再更新 GPU resident set
    这样当前 decode promotion 的优先级已经逐渐变成：
    - activated cache hit
    - warm cache hit
    - resident staging hit
    - 冷路径 build/load
    它更接近一个分层缓存系统，而不是单次迁移动作。
41. activated cache 现在也开始承载预算与优先级语义：
    - cache 容量默认按 `decode_promote_k` 约束
    - pipeline 会根据 lifecycle 优先级（`activated > warmed > ready`）和 hotness 只保留更热的候选
    - 较冷的 activated expert 会被降回 CPU warm cache
    这让 decode 前的 device-side 准备不再是“谁先到谁上”，而是更接近真正的热点激活预算管理。
42. migration queue 的 deferred 重排语义也做了修正：
    - 早期实现里，已经 `ready/warmed/activated` 的 expert 若因 budget 或时机问题被重新排队，会被简单写回 `deferred`
    - 现在 queue 会保留这些 expert 已完成的中间 lifecycle，只把“尚未开始”的 op 记成 `queued/deferred`
    - 这保证了 pipeline 是“逐步向前推进”的，而不是遇到 defer 就把已完成的预热/激活进度丢掉
    对想把 PIM 路径真正做成流水线来说，这个语义修正很关键，因为它避免了控制面自身制造回退。
43. 为了让这件事可测，migration manager 现在还会统计 `requeue_preserved_states`：
    - 每次 deferred/queued 重排若保住了原有 lifecycle，会累加一次
    - 这样 benchmark 就能区分“队列被重排了很多次”与“虽然重排，但流水线进度没有丢”
    这对后续比较 `baseline / overlap_safe / eager` 很关键，因为真正好的 profile 不只是 defer 少，还应当让已准备好的 expert 不回退。
44. decode 的 ready promotion 现在也改成了“选择性消费”而不是“先全取出再回填”：
    - migration manager 里 ready 的 pending op 会先被 `peek`
    - 只有真正 `applied` 的 expert 才会从 queue 中移除
    - 因预算不足暂时没消费的 ready/warmed/activated expert 会继续留在原队列中
    这样 pipeline 就少了一次“取出 -> 再放回”的控制面抖动，更接近真正的流水线 buffer。
45. `decode_require_prefetch_ready` 的严格语义现在也覆盖 resident staging：
    - 即使某个 expert 的 resident weights 已能同步 stage 到 CPU cache
    - decode prime 阶段也不会在同一步把它直接视作 `ready`
    - 它必须先经过下一次 refresh/pipeline tick，才会真正进入 ready-only 消费路径
46. benchmark/profile sweep 的结果层现在也更像“调度决策面板”了：
    - `profile_sweep_summary` 除了保留原始 `profiles` 和按 decode TPS 选的 `best_by_decode_tokens_per_second`
    - 还会输出 `comparison_table`
    - 以及 `best_by_metric` / `metric_directions`
    - 可以直接按 `pipeline_promotion_non_cold_ratio`、`runtime_apply_batch_size_avg`、`runtime_deferred_for_prefetch` 这类 overlap 相关指标比较 profile
    这让后续宿主机真实 sweep 不再只是“哪个快”，而是能看清“为什么快/慢”。
47. ready promotion 的 batch apply 现在也能区分内部来源构成：
    - layer 级会记录 `pipeline_apply_batch_activated / warm / cold`
    - token-step runtime 也会累计 `offload_pipeline_apply_batch_*_total`
    - summary/profile sweep 会把这些比率透出
    这让系统不只知道“这一批应用了多少 expert”，还知道“这一批到底有多少已经是 activated 热路径，多少仍在走 cold path”。
48. cache 层次和 migration lifecycle 现在也重新对齐了：
    - 当 activated cache 因容量被挤出时，对应 expert 的 lifecycle 会从 `ACTIVATED` 回退到 `WARMED`
    - 当 warm cache 因容量被挤出时，对应 lifecycle 会从 `WARMED` 回退到 `READY`
    - 因此 cache 淘汰不再只是对象层面的变化，而会同步反映到控制面状态机
    这对后续做真正的后台 worker 很关键，因为系统终于能从 lifecycle 上准确知道 expert 还处于哪一层缓存。
49. 这些 cache 回退现在也开始变得可量化：
    - migration diagnostics 会分别累计 `activation_eviction_regressions` 与 `warm_eviction_regressions`
    - 前者表示 device-side activated candidate 被挤回 CPU warm 层
    - 后者表示 CPU warm candidate 被进一步打回 `READY`
    这意味着后续 profile sweep 不只可以看命中率，也可以看“系统为了维持 budget 到底回退了多少已准备好的 expert”。
50. 这些 regression 指标现在也已经进入 profile sweep 排名层：
    - `comparison_table` 会带出每个 profile 的 eviction regression 压力
    - `best_by_metric` 也能按更少的 regression 选出更稳的 profile
    这让 profile 对比从“只看快不快”继续推进到“快的同时是不是在靠频繁回退缓存状态硬撑”。
51. warm cache 的 eviction 策略现在也从“近似 FIFO”收紧成了“生命周期优先级 + hotness”：
    - 更冷、生命周期更低的 warm candidate 会先被回退到 `READY`
    - 更热、已经更接近 `APPLIED` 的 candidate 会尽量保留在 warm 层
    这让 warm cache 开始真正承担“promotion 二级候选池”的职责，而不是普通对象缓存。
46. benchmark 侧现在也开始按“单次 run”观察流水线：
    - 每次 generation 前会重置 HybridMoE 的 runtime/queue/cache 计数器
    - 每个 run 结果都带自己的 `scheduler_summary`
    - 这样 `baseline / overlap_safe / eager` 的 profile sweep 就可以直接按单次 run 比较 pipeline 行为，而不是读混在一起的累计诊断
    这对后续证明 PIM 路径是否真的开始压过 `cpu+gpu` 很关键，因为我们终于能把一次生成里的 promotion/prefetch/activation 代价独立量出来。
47. warm cache 的 prebuild 现在也引入了候选预算：
    - pipeline 不再对所有 `ready/warmed/activated` expert 都尝试预构建
    - 而是先按 lifecycle 优先级和 hotness 排序，再按 `decode_promote_k` 的倍数截断
    - 这样 warm cache 开始更像“promotion 的二级候选池”，而不是所有 ready expert 的公共堆积区
    这有助于把 CPU 侧 module 构建成本也压缩到更接近最终会进入 GPU 的那部分专家上。
48. 当前 pipeline 还开始显式区分 promotion source：
    - `activated`: 已完成 device transfer，只差 resident set 切换
    - `warm`: 命中 CPU warm cache，但仍需一次 device activation
    - `cold`: 仍需从 staging/checkpoint 构建 expert module
    系统现在会把这些 source 汇总成 per-run 指标，后面就可以直接比较“有多少 promotion 已经脱离冷启动”。这对评估 PIM 路径能否追上甚至超过 `cpu+gpu` 非常关键，因为它把真正的流水线收益量化出来了。
49. decode 层内 migration 消费最近也做了进一步收敛：
    - 旧路径会在 `HybridMoE.forward()` 里先 `drain/take_ready` 整层 queue
    - 再把本步没消费掉的 op 重新 enqueue 成 `*_deferred`
    - 当前改成 `peek + selective consume`
    - 只有真正 `applied` 的 promotion / demotion 才会从 queue 中移除
    - 因预算不足暂未消费的 `ready/warmed/activated` promotion 会继续保留在原 pending queue 中
    这样 layer-forward 和 token-step pipeline 不会再围绕同一批 decode migration 做多余的“取出 -> 再塞回”操作。
50. strict ready-only 语义也被进一步统一到了 layer-forward：
    - 若 `decode_require_prefetch_ready=true`
    - layer-forward 只接受 lifecycle 已推进到 `READY/WARMED/ACTIVATED` 的 promotion
    - 即便 materialization cache 当前步已经同步命中，也不会在同一个 forward 中越级把 expert 直接视为 `ready`
    这让 token-step pipeline 和 layer-forward 在“什么叫真正 ready”这个问题上保持一致，避免前台逻辑绕过既有流水线阶段。
51. 当前 ready promotion 又往前迈了一小步：
    - promotion candidate 现在会先按 hotness / lifecycle 排序
    - 然后经由 `_select_ready_promotion_batch()` 做同层批次截断
    - pipeline 诊断会记录 `pipeline_apply_batches` 和 `pipeline_apply_batch_experts`
    这还不是最终的“同层真正批量 apply”，但系统已经开始把 decode promotion 从逐 expert 思维转成“先选一批值得应用的 expert，再执行该批次”。
52. 这一批次语义又继续往前推进了一层：
    - pipeline 在真正 apply 这一批 ready experts 之前
    - 会先统一计算当前 batch 还缺多少个 GPU slot
    - 再通过 `_evict_for_promotion_batch()` 一次性完成这一批次需要的 eviction
    这样 batch 内的各个 expert 不再各自循环做一次 resident budget 检查，控制面更接近“先为本批次腾位，再消费本批次”。
53. token-step runtime 现在也开始汇总这批次推进信息：
    - 每次 tick 会带出本步 `apply_batch_count / apply_batch_experts / apply_batch_evictions`
    - 模型级 runtime 诊断则会累计成 `offload_pipeline_apply_batch_*_total`
    这样后续 benchmark 就不只是看 layer 本地的 ready/applied 计数，而能直接从 step 级视角判断“流水线这一轮到底像不像一个批处理系统”。
54. benchmark/profile sweep 现在也有了更贴近决策的自动汇总层：
    - 会把每个 profile 的 `decode_tokens_per_second`
    - `pipeline_prefetch_overlap_hits`
    - `promotion source breakdown`
    - `pipeline_apply_batch_*`
    - `runtime_deferred_for_prefetch`
    汇总成结构化摘要，并直接给出按 decode TPS 选出的当前最好 profile。
    这让后续调度实验不再只是看原始 JSON，而是能更快回答“哪组策略最接近让 PIM 路径追上甚至超过 `cpu+gpu`”。
55. 最近这个汇总层又继续往 step 级推进了一步：
    - `profile_sweep_summary` 现在也会带上 runtime 侧累计的 `offload_pipeline_apply_batch_*_total`
    - 因此一轮实验可以同时看到：
      - layer 本地的 apply batch 统计
      - token-step runtime 汇总的 apply batch totals
    这有助于判断“某个 profile 是真的让流水线更像 batch system”，还是只是把局部 layer 指标做得好看。
56. 为了让这些 step 级 totals 可信，runtime hook 现在返回的是“本次 tick 新增了多少 apply batch 指标”，而不是把 layer 上的累计计数每步重新上报一遍。
    这样 `MigrationPipelineRuntime` 聚合出来的 `offload_pipeline_apply_batch_*_total` 才真正代表 token-step 级的累计推进，而不是把同一层的历史计数重复累加。
57. activated cache 的淘汰策略也已经从简单插入顺序，推进到了“lifecycle 优先级 + hotness”：
    - device-side activated candidate 被逐出时，会优先挑选更冷、准备阶段更低的对象
    - 被逐出的 activated expert 会回退到 CPU warm cache
    - 同时 migration lifecycle 会从 `ACTIVATED` 降级为 `WARMED`
    这样 activated cache 开始具备和 warm cache 一致的热点保留语义，不再只是一个短暂的过渡列表。
58. warm cache 和 activated cache 现在还开始共享统一的 prepared-cache 总预算：
    - `expert_prepared_cache_size` 限制的是 `warm cache + activated cache` 两层合计可保留的 prepared experts 数量
    - activated cache 增长时，warm cache 的有效上限会动态收缩
    - 因而系统开始从“两个独立缓存”演化成“同一 prepared tier 的两段式表示”
    这使后续做统一 per-layer cache policy 或自适应 budget 调整更自然，不必再分别管理两套完全独立的容量逻辑。
59. prepared-cache 预算进一步从“共享上限”推进到了“统一重平衡”：
    - 当 `warm cache + activated cache` 的 prepared experts 总数超过预算时
    - 系统会把两层候选放到同一个 victim 选择逻辑里
    - 依据 hotness、lifecycle 和 cache stage 统一挑选应回退的对象
    这样 prepared tier 的控制语义开始更接近一个真正的分层缓存系统，而不是两个各自裁剪、偶尔共享上限的临时容器。
60. prepared-cache 预算现在也已经打通到用户入口和观测面：
    - `LLM`、`example.py`、`benchmark_inference.py` 都可以显式配置 per-layer prepared-cache budget
    - scheduler summary 会输出 `prepared_cache_limit / prepared_cache_size / effective_warm_cache_limit / prepared_cache_utilization`
    - 因此后续 benchmark 已可以直接比较不同 prepared-cache 预算对 overlap、cold-path 比例和 eviction regression 的影响
    这使 prepared tier 不再只是内部实现细节，而是成为可调、可测、可比较的系统参数。
61. profile sweep 现在也已经把 prepared-cache 指标纳入自动比较：
    - `profiles`
    - `comparison_table`
    - `best_by_metric`
    都会显式保留 prepared-cache utilization 与 budget 相关字段
    这样 prepared tier 不再只是 benchmark 的附加诊断，而是能直接参与 profile 排序和调度策略评估的一级信号。
62. prepared-cache 的重平衡压力也已进入观测面：
    - summary 和 profile sweep 会分别统计 warm side 被挤掉了多少、activated side 被挤掉了多少
    - 以及 activated candidate 被降回 warm、warm candidate 被降回 ready 的次数
    这样后续做自适应 prepared-cache policy 时，可以直接判断“预算不够”主要打在 warm tier 还是 activated tier，而不是只看到一个总的 utilization。
63. prepared-cache retention policy 现在还带一个最小的自适应 stage bonus：
    - activated tier 在 prepared-cache victim 选择时会带额外 stage bonus
    - 这个 bonus 会随着最近的 prepared-cache 重平衡方向做小幅调整
    - 因而系统开始具备“近期更容易牺牲哪一层，就稍微补偿哪一层”的最小动态倾向
    这还不是完整的 budget controller，但已经把 prepared tier 从静态 heuristic 推进到了弱自适应策略。
64. 这种弱自适应又进一步反馈到了 activation/prebuild 限额：
    - 当前系统会根据 prepared-cache pressure 和 stage bonus 推出 `adaptive_activation_limit`
    - 同时也会调整 `adaptive_prebuild_limit`
    - 当 prepared tier 已经吃满且 activated 偏置较低时，pipeline 会自动降低后续准备动作的激进程度
    这样 prepared-cache 不再只是事后淘汰，而开始对前向的 prebuild / activation 候选规模形成闭环约束。
65. prepared tier 现在还开始显式感知冷路径 promotion 压力：
    - 每次 batch apply 后都会统计本轮 `cold` promotion 比例
    - 若冷路径比例偏高，系统会提升 `cold_promotion_penalty`
    - 这个 penalty 会进一步抬高后续的 adaptive activation/prebuild limit
    也就是说，prepared tier 已不只是“被 cache 压力压缩”，也开始在“冷启动太多”时主动尝试扩大准备动作，形成一个最小的双向反馈回路。
66. prepared-tier controller 现在开始直接约束 prefetch aggressiveness：
    - pending promotion 的预取提交会经过 `adaptive_prefetch_pending_limit`
    - 热候选预取会经过 `adaptive_prefetch_candidate_budget`
    - 这两个预算同时受 `prepared_cache_budget_backoff`、rebalance step pressure 和 `cold_promotion_penalty` 影响
    这意味着 prepared tier 不再只调节 `prebuild/activation` 两段，而是开始把 prefetch 也纳入统一控制面，pipeline 的 `queued -> prefetching -> ready` 段现在也开始跟随 prepared pressure 和 cold-path 压力自适应变化。
67. prepared budget 的“静态基线”现在也已经进入观测面：
    - profile summary 会给出 `prepared_cache_budget_heuristic`
    - runtime diagnostics 会给出实际采用的 `prepared_cache_budget`
    - 因而 benchmark / profile sweep 后面可以直接对照“静态 prepared 预算是多少”与“controller 最终把 effective budget、prefetch/aggressiveness 调到了哪里”
    这让 prepared-tier controller 不再只是 runtime 黑盒，而开始具备可解释的 baseline-vs-controller 观测能力。
68. prepared budget heuristic 现在也开始显式受 profile 影响：
    - `baseline` 维持最小预算基线
    - `overlap_safe` 会额外抬高 prepared budget，给 strict ready-only decode 留出更多 warmed/activated 空间
    - `eager` 会进一步上调 prepared budget，配合更积极的 prefetch / prebuild / activation
    这让 profile 不再只是 scheduler 开关集合，而开始直接塑造 prepared tier 的初始容量形态，为后续做 profile-driven auto-tuning 打基础。
69. profile 现在也开始显式塑造 prepared-tier controller 的 aggressiveness：
    - `baseline` 使用最保守的 prepared-tier 推进力度
    - `overlap_safe` 会给 controller 一个中等 aggressiveness，主要帮助 strict ready-only 场景减少 cold promotion
    - `eager` 会进一步提高 prepared-tier 推进力度，使 activation / prebuild / prefetch 都更积极
    这意味着 profile 不再只决定“预算多大”，也开始决定“预算用得有多激进”，prepared tier 已开始具备 profile-driven 控制语义，而不是仅靠运行时局部 heuristic 自行漂移。
70. materialization 路径现在新增了一个真正的后台 resolve worker：
    - prefetch future 完成后不再等前台 `poll_ready()` 去执行 `future.result()`
    - 解析、取回张量和写入 staging cache 会在后台 resolver 线程完成
    - 前台 refresh/poll 只需要消费轻量 ready 通知
    这虽然还不是真正的后台 migration worker，但已经把 `prefetching -> ready` 路径上的重活从 decode 主路径上挪开了一部分。
71. promotion batch 现在已经显式拆成两段：
    - 第一段先统一 resolve 每个 expert 的来源：`activated / warm / cold`
    - 第二段再统一 apply 到 GPU resident set
    当前 resident 注入还不是底层真正 batched apply，但系统边界已经从“批量选目标、逐 expert 一边查来源一边 apply”推进到“batch resolve -> batch commit”的形态，这为后续把 `activated/warm -> resident set` 做成真正批处理提供了清晰接口。
72. materialization 解析完成后现在还可以直接触发后台 ready callback：
    - ready callback 会在 expert 完成解析并写入 staging cache 后立即触发
    - callback 可把对应 migration lifecycle 直接推进到 `READY`
    - 前台 token-step refresh 只需要保留兜底 drain 和汇总语义
    这让系统从“前台轮询 future 完成”进一步推进到了“后台完成即推状态”，虽然仍未形成独立 migration worker，但 `prefetching -> ready` 已不再完全依赖前台 hook。
73. 为了让后台 callback 安全进入 migration 控制面，`ExpertMigrationManager` 现在已经补上了内部锁：
    - queue/peek/take/mark_state/state_for/diagnostics` 都通过同一把 `RLock` 串行化
    - 这样 materialization 后台线程可以直接推进 `READY`
    - 同时不会和前台 token-step pipeline 的 `peek + selective consume` 语义互相踩状态
    这还不是完整后台 migration worker，但已经让 migration lifecycle 从“默认前台单线程结构”转成了“允许后台事件进入的线程安全结构”。
74. token-step runtime 现在又拆出了一条更轻的 background offload tick：
    - 每次真正执行主 refresh/advance 之前，会先单独推进一轮后台 ready callback
    - background tick 只负责消费后台 ready 事件，不负责完整的 pipeline apply
    - 主 refresh 仍负责 `READY -> WARMED/ACTIVATED/APPLIED`
    这让系统从“所有 offload 推进都挤在一个 refresh hook”进一步演进成了“背景事件推进 + 主流水线推进”两段式结构，虽然都还是 token-step 触发，但边界已经接近以后引入独立 migration worker 的形态。
75. background offload tick 现在也进入了 scheduler summary：
    - `offload_background_ticks`
    - `offload_pipeline_background_ready_callback_total`
    这样 benchmark 已经能单独观察“后台事件推进量”和“前台主流水线推进量”，两条路径终于可以分开对比，而不再都折叠进同一个 refresh 计数里。
76. runtime 级 background tick 指标现在也会被 `LLM.reset_offload_diagnostics()` 一起重置：
    - 单次 run 的 background tick 计数不会再混入前序 warmup
    - background ready callback 的推进量终于具备和 decode/apply 指标一致的 per-run 语义
    这让以后用 benchmark/profile sweep 比较“后台推进是否真的减少前台 stall”时，数据口径开始统一。
77. background offload tick 现在已经开始介入 prepared tier 的前半段推进：
    - 它不再只做 `prefetching -> ready`
    - decode 阶段也会在 background tick 中提前尝试 `ready -> warmed -> activated`
    - 主 refresh 则继续负责最终的 `activated -> applied`
    这样后台路径第一次真正覆盖到了 prepared tier，而不只是单纯的 ready callback；系统已经开始向“后台准备对象，前台只做最终 commit”这个方向靠拢。
78. 当前系统还新增了最小后台 offload worker：
    - `BackgroundOffloadWorker` 在独立线程中周期性调用模型级 `background_tick_offload_state()`
    - 后台线程当前主要推进：
      - background ready callback
      - `READY -> WARMED`
      - `WARMED -> ACTIVATED`
    - 前台 decode 主路径则继续负责：
      - `ACTIVATED -> APPLIED`
    这意味着当前系统已经从“纯前台 hook 推进”演化成了“后台准备 + 前台 commit”的雏形；虽然还不是真正的 GPU<->PIM 异步迁移执行器，但后台 worker 已经是独立对象，并具备独立计数、reset 和 shutdown 生命周期。
79. background worker 现在已经接入真实生成生命周期：
    - `SimpleEngine` 负责启动/停止后台 worker
    - `LLM.generate()` 在生成前启动 worker，在 `finally` 中停止并清理
    - 因此后台推进不再只是“模型里有个可选线程对象”，而是正式进入端到端 decode 路径
    这一步的意义是把后台 worker 从纯骨架推进成了真正会参与运行的 runtime 组件，为后续继续把 `READY -> WARMED/ACTIVATED` 甚至部分 apply 从前台主路径挪走打下基础。
80. background worker 现在默认显式启动：
    - `MixtralModel` 构造时只创建 worker 对象，不自动起线程
    - 真正进入生成路径时，再由 `SimpleEngine/LLM` 显式启动
    - 生成结束后立即停止
    这使后台迁移执行器的生命周期边界更清晰，也避免了“模型一构造就常驻后台线程”的隐式资源占用。
81. background worker 现在也进入了 summary/sweep 决策面：
    - summary 会显式输出 worker 的 `enabled / ticks / work_ticks / work_ratio`
    - profile sweep 也会把 `background_worker_work_ratio` 纳入比较
    这让系统终于可以直接回答一个关键问题：后台 worker 是否真的在产生有效工作，以及它和 decode 吞吐之间的关系。
82. prepared-tier 的“静态预算基线”和“运行时控制结果”现在也被明确区分：
    - `prepared_cache_budget_heuristic` 表示当前 profile 给出的静态 prepared budget 基线
    - `prepared_cache_budget / effective_prepared_cache_limit / adaptive_*` 则表示 runtime controller 实际如何调节 prepared tier
    这样 benchmark 和诊断不再只看到 controller 的结果，也能回溯“这组策略一开始给了 prepared tier 多大基线”，更利于分析 profile 本身和 controller 本身各自的贡献。
83. 当前 background worker 已和前台 refresh hook 做了去重：
    - 如果后台 worker 已运行，`SimpleEngine` 不再手动触发 `background_tick_offload_state()`
    - 前台路径只保留主 `refresh_offload_state()` 和后续 `advance_offload_pipeline()`
    这避免了同一个 token-step 里，后台线程和前台 hook 对 ready/warmed/activated 前半段做重复推进，是后台执行器真正接入运行路径后的一次必要收口。
84. background worker 的可观测面现在也更完整了：
    - `offload_background_work_items_total` 表示后台线程总共推进了多少 ready/warmed/activated/background-applied 工作项
    - `offload_background_activation_applied_total` 表示其中有多少已经推进到了后台 apply
    - `offload_background_work_items_avg` 则给出每个 background tick 的平均推进量
    因此 benchmark/profile sweep 现在不只是在看“后台线程有没有跑”，而是开始看“后台线程到底推进了多少有效迁移工作”。
85. 随着后台 worker 真正进入运行时，`HybridMoE` 现在增加了内部 pipeline 锁：
    - `refresh_offload_state()`
    - `background_tick_offload_state()`
    - `background_advance_offload_pipeline()`
    - `advance_offload_pipeline()`
    - decode 入口的 migration 应用与诊断快照
    都开始通过同一个 `RLock` 串行化访问 prepared-tier cache、migration lifecycle 和 resident set。
    这一步还没有把后台 apply 做成独立 commit queue，但它先把“后台线程已经真实在跑”的共享状态风险收住了，为后续继续拆 activation/apply queue 提供了安全边界。
86. 后半段 resident commit 现在开始反向影响前半段 prepared-tier controller：
    - `apply_candidate_queue` 已不再只是 staged commit buffer，而是会输出：
      - `apply_queue_pressure`
      - `apply_queue_pressure_step`
      - `apply_queue_pressure_ema`
      - `apply_queue_budget_backoff`
    - 当 apply queue 长期拥塞、或频繁出现 queue eviction 时，controller 会主动收缩：
      - `adaptive_activation_limit`
      - `adaptive_prebuild_limit`
      - `adaptive_prefetch_pending_limit`
      - `adaptive_prefetch_candidate_budget`
    这意味着系统开始形成真正的“前半段 prepared tier 和后半段 resident commit 阶段联动控流”，不再只是 prepared cache 自己感知自己的压力。
87. apply queue staged commit 现在也开始具备独立批次控制：
    - 系统会分别记录：
      - `apply_queue_commit_batches / apply_queue_commit_experts`
      - `background_apply_commit_batches / background_apply_commit_experts`
    - 并通过 `adaptive_apply_commit_limit()` 根据：
      - apply queue 压力
      - apply queue pressure EMA
      - cold promotion penalty
      - profile aggressiveness
      动态调节每批 commit 尺寸
    这让后半段 resident commit 不再只是“有个队列再逐 expert 提交”，而开始具备独立的批次控制面，为后续真正的 per-layer batch commit 做准备。
88. apply commit batch queue 现在也已具备独立预算和诊断：
    - `size / limit / utilization`
    - `enqueued / pruned / evictions`
    - `background_apply_commit_batch_queue_enqueued`
    并且 runtime 会单独累计 `offload_background_apply_commit_batch_queue_enqueued_total`
    所以后续 benchmark 已可以区分后台 worker 到底是在推进 staged commit queue，还是已经把 ready batch 推进到了最终 resident commit buffer。
89. apply commit batch queue 现在也进入了 controller 闭环：
    - `apply_commit_batch_queue_pressure`
    - `apply_commit_batch_queue_pressure_step`
    - `apply_commit_batch_queue_pressure_ema`
    - `apply_commit_batch_queue_budget_backoff`
    这组信号会继续反向约束 `adaptive_activation_limit / adaptive_prebuild_limit / adaptive_prefetch_*`
    因而系统现在不只感知 candidate queue 和 staged commit queue 的压力，也开始感知最终 resident commit batch buffer 是否拥塞。
90. apply commit batch queue 的压力信号现在也已进入 profile sweep 决策面：
    - `apply_commit_batch_queue_pressure_avg`
    - `apply_commit_batch_queue_pressure_ema_avg`
    - `apply_commit_batch_queue_budget_backoff_avg`
    这意味着 profile 对比不再只看前半段 prepared-tier 和中段 staged commit queue，也开始显式比较“最终 resident commit buffer 是否拥塞”。
91. resident commit 的后半段现在已经明确拆成两级 budget：
    - `_adaptive_apply_commit_limit()` 负责 `apply_commit_queue` 的 staged resolve / staging
    - `_adaptive_apply_commit_batch_limit()` 负责 `apply_commit_batch_queue -> resident set` 的 final batch commit
    这让后半段 resident commit 从“单一 staged commit 预算”推进成了“staged resolve 与 final commit 分离控制”的结构，更接近真正的后台 batch commit worker。
92. `apply_commit_batch_queue` 现在已经按 batch 而不是按 expert 存储 staged commit：
    - stage 阶段直接把 hot ready entries 组装成 batch
    - resident commit 消费 batch entries 做模块提交
    - 后续 finalize 再逐 expert 写回 residency / lifecycle
    这意味着 resident commit 的最后一段已经不再是“逐 expert 的 staged buffer”，而是“batch commit buffer + per-expert finalize”的结构。
93. resident commit 现在还显式区分了两种预算：
    - staged budget：控制 `apply_commit_queue -> apply_commit_batch_queue`
    - final batch budget：控制 `apply_commit_batch_queue -> resident set`
    这让后半段不仅按 batch 组织，还能对“准备多少 batch”和“每轮真正 commit 多少 batch”分别施加控制，为后续后台 batch commit worker 留出接口。
94. resident commit 的最后一段现在又明确拆出了一层 `resident_commit_batch_queue`：
    - `apply_commit_batch_queue` 负责把 staged commit 按 batch 准备好
    - `resident_commit_batch_queue` 负责保存“已经进入 final resident-commit buffer 的 batch”
    - resident set commit 只从 `resident_commit_batch_queue` 消费
    这样后半段已经从“batch buffer + finalize”推进成了“staged batch buffer + final resident-commit buffer + finalize”的三段结构，更接近真正后台 batch commit worker。
95. background tick 现在对 `resident_commit_batch_queue` 也采用“上一轮可消费、本轮新入队只 prefinalize”的规则：
    - tick 开始前已经存在的 resident-commit batch，允许进入本轮 background apply
    - 同一 tick 新推进到 `resident_commit_batch_queue` 的 batch，只会被标成 prefinalized，不会立即 commit
    这样最终 resident commit 也具备了稳定的流水线边界，避免同一轮里既组最终 batch 又立刻消费。
96. resident commit 现已进一步拆成 `resident_commit_batch_queue -> resident_commit_finalize_queue -> resident set`：
    - background tick 会先把 final resident batches 预推进到 `resident_commit_finalize_queue`
    - 同一 tick 新推进到 finalize queue 的 batch 只记作 prefinalized，不会立即消费
    - resident set apply 只消费 tick 开始前已经存在的 finalize batches
    这样最后一段已经从“resident batch buffer + finalize”推进成“resident batch staging + finalize queue + metadata finalize”的更稳定三段结构。
97. resident commit 现已再拆成 `resident_commit_finalize_queue -> resident_commit_ready_cache -> resident set`：
    - background tick 会先把 preexisting finalize batches 解析成 ready cache entries
    - 同一 tick 新进入 finalize queue 的 batch 不会立刻进入 ready cache
    - resident set apply 只消费 tick 开始前已经存在的 ready cache batches
    这样最后一段已经从“finalize queue + metadata finalize”推进成“finalize staging + ready cache + lightweight finalize”的更稳定后台提交结构。

这仍不是最终想要的“PIM resident -> GPU resident 的异步迁移”，但已经把系统推进到了“prefill 做热度探测和预热，decode 做真正 materialize”的合理分工。

## 目标演进方向

1. 非专家层继续常驻 GPU，包括 embedding、attention、norm、router 和 lm head。
2. 专家参数分成两级驻留：
   - GPU resident experts: 当前时间窗内的热点专家，直接在 GPU 上执行
   - PIM resident experts: 默认驻留在 PIM 的冷专家
3. 调度器根据路由统计和时间窗热度，决定专家 promotion / demotion：
   - 从 PIM 提升到 GPU
   - 从 GPU 回写到 PIM
4. 迁移要与推理重叠：
   - 当前 token / layer 继续在 GPU 和 PIM 上执行
   - 下一批候选热点专家在后台传输
5. 因此系统最终需要同时维护“计算路径”和“迁移路径”两条流水线，而不只是当前的静态 mask。

## Prefill 策略

- prefill 阶段通常 token 数多、专家路由分布宽，直接落到 PIM 会使传输与小批量专家调用数暴涨。
- 因此当前系统策略应优先保证：
  - prefill 默认仍以 GPU resident experts + CPU fallback 为主
  - PIM 更适合 decode 阶段的小 batch、窄路由、可持续复用的 cold experts
- 后续动态调度的合理做法是：
  - 用 prefill 收集热度
  - 在 prefill 期间预热下一阶段的 GPU resident experts
  - 到 decode 再让 PIM 承担更大比例的冷专家执行

## 关键约束

- 当前 `Qwen3-30B-A3B` 在本机 `47.41 GiB` GPU 上无法走“全专家纯 CUDA”路径，会 OOM。
- 当前环境里没有可用的 `kt-kernel` / `kt_kernel_ext`，所以 CPU offload 仍然是纯 PyTorch fallback，而不是最终性能形态。
- 当前 `pim` backend 同时支持 `linear` 和实验性 `fused` 两种 kernel variant；`fused` 已能在 DPU 上执行完整 expert 子图，但性能尚未优于 `linear3`，所以默认仍应使用 `linear`。
- 当前 PIM benchmark 测的是整数 affine workload，不是 FLOPS，也不是 MoE expert GEMM。
- 当前专家放置仍是静态的 `gpu_experts_mask`；真正的 GPU/PIM 动态迁移、权重常驻管理和 overlap 调度还没有接入系统。
- 当前 resident commit 已演进为多级 staged pipeline：
  - `apply_candidate_queue`
  - `apply_commit_queue`
  - `apply_commit_batch_queue`
  - `resident_commit_batch_queue`
  - `resident_commit_finalize_queue`
  - `resident_commit_ready_cache`
  - `resident_commit_apply_queue`
  - `resident_commit_finalize_ready_queue`
  - `resident set`
- background worker 现在可以把 preexisting resident batches 逐步推进到 `resident_commit_apply_queue` 和 `resident_commit_finalize_ready_queue`，而前台 decode 主路径主要消费 tick 开始前已存在的 finalize-ready batches，再做最终 metadata finalize。
