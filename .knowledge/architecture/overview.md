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
  Hybrid MoE 调度层。当前负责区分 hot experts 和 offloaded experts，并在 GPU 输出和 offload 输出之间合并结果；后续要升级为动态专家驻留与迁移的核心调度层。
- `nano_ktrans/kernels/offload_backend.py`
  offload 扩展点，定义 `ExpertOffloadBackend` 抽象和 backend 命名规范化逻辑；后续应承载专家驻留状态、迁移计划和异步数据面接口。
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
