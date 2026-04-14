---
updated: 2026-04-07
tags: [pitfalls, debugging]
---

# ⚠️ 已知陷阱 & 踩坑记录

<!-- updated: 2026-04-07 20:56 -->

## 可选加速依赖不能再被当成硬依赖

- 现象：没有安装 `flash-attn` 或 `kt-kernel` 时，连导入 `nano_ktrans.layers` / `nano_ktrans.models` 都会失败。
- 根因：`attention.py`、`cpu_infer.py`、`cpu_moe.py` 之前在模块导入时直接强依赖这些包。
- 修复：改为运行时探测，可用则走加速路径，不可用则退化到纯 PyTorch attention 和 CPU fallback。

<!-- updated: 2026-04-07 20:56 -->

## 本机 pip 会被失效代理拦住

- 现象：`pip install` 持续报 `ProxyError`，目标地址是 `127.0.0.1:7897`。
- 根因：环境里预设了 `HTTP_PROXY` / `HTTPS_PROXY`，但当前会话没有可用代理服务。
- 规避：安装依赖时显式 unset 这些变量。

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
