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
- `nano_ktrans/kernels/pim_native/`
  DPU 原生代码目录，当前同时包含 linear kernel、fused expert kernel、对应 host bridge 和 build 脚本。
- `benchmarks/benchmark_inference.py`
  统一 benchmark 入口，用于比较 `cpu`、`cuda`、`cuda_cpu_offload`、`cuda_pim_shadow`。
- `benchmarks/pim_microbench/`
  独立的 UPMEM host/DPU microbenchmark，用来测真实硬件上的传输与整数 kernel 指标。

## 数据流

1. `LLM` 从 checkpoint 读取 Hugging Face 配置，并通过 `GenericMoeConfig` 适配到统一结构。
2. decoder 层中的 `HybridMoE` 在 GPU 上完成路由。
3. 命中的 hot experts 直接在设备侧执行。
4. 冷 expert 请求通过 `ExpertOffloadBackend` 派发。
5. `CPUMoEBackend`、`PIMMoEBackend(pim_shadow)` 或 `PIMMoEBackend(pim)` 处理 offloaded expert 请求。
6. offload 输出与 GPU 输出合并，再交还推理引擎继续 decode。

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
    这样系统里的 `ready` 更接近“上一阶段已经完成”的稳定信号，而不是“本阶段临时凑出来”的同步捷径。
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
