---
created: 2026-04-07
updated: 2026-04-15
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
