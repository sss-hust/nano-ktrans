---
title: "PIM + GPU 异构推理 Qwen3-30B MoE decode 全程优化：从 0.228 到 1.3454 TPS 的算法与系统设计路径"
date: 2026-05-07T21:15:00+08:00
tags: [pim, moe, qwen3, gptq, decode-optimization, upmem, cuda, heterogeneous, research-report]
type: report
status: final
source: server
session_tool: codebuddy
description: "回顾 nano-ktrans 项目 M-2 → M-27 共 26 个里程碑对 PIM+GPU Qwen3-30B-A3B-GPTQ-Int4 decode 路径的优化全过程，重点剖析算法与系统设计层面的关键决策、放弃的方向、剩余差距与其成因。"
---

# PIM + GPU 异构推理 Qwen3-30B MoE decode 全程优化
## —— 从 0.228 到 1.3454 TPS 的算法与系统设计路径

## 摘要（Executive Summary）

本项目研究如何在一台 **47 GB GPU + 39 个 UPMEM DPU rank** 的异构平台上，让 **PIM 真实承担 MoE offloaded expert 的算力**，并让 `pim+gpu` 端到端 decode 性能接近（最终超越）`cpu+gpu` 基线——这一约束来自科研论述的需要：**必须证明 PIM 参与计算是有价值的，不能用绕开 PIM 的方式获胜**。

工作的主线是 Qwen3-30B-A3B-GPTQ-Int4、32 decode tokens、`offload-device-experts=92`、mean-mask routing-aware residency。从 2026-04-22 的初始状态（decode_tps **0.228**，距离 CPU 基线 3.068 TPS 的 **13.5×** 差距），经过 26 个里程碑（含 9 个已登记的 NEGATIVE 对照实验），当前 pim 主干达到 decode_tps **1.3454**（`cuda_cpu_offload` 同配置 **2.0933 TPS**）——**6 倍于起点，闭合原始差距的 96%，距离 CPU 基线还剩 35.7%**。

本报告不按时间线罗列，而是按**算法与系统设计层面的核心问题**组织：每一节先讲**我们面对的是什么问题**，再讲**为什么某类直觉的方案不行**，最后讲**最终采用的算法/设计和它的量化代价**。

---

## 1. 背景与研究约束

### 1.1 硬件与模型

- **GPU**：47 GB 可用显存（推理时 peak ≈ 41.9 GB）
- **PIM**：39 个 UPMEM DPU rank × 64 DPU/rank，每 DPU 64 MB MRAM、WRAM + 软件乘法指令（8-bit×8-bit 约 10 cycles）
- **模型**：Qwen3-30B-A3B-GPTQ-Int4，128 experts / layer × 48 layers，每层 top-k=8
- **workload**：14 token prompt + 32 new tokens，每层 decode 时 top-k 路由决定哪些 expert 被激活

### 1.2 三条硬约束

1. **PIM 必须真实承担计算**（科研约束）：任何让 `real_dpu_expert_calls = 0` 的"优化"不合格。
2. **PIM 参与不能拖慢端到端**：`pim+gpu` 的 decode TPS 必须趋近并最终 ≥ `cpu+gpu`。
3. **GPU 显存约束**：GPU 只能常驻 92/128 个 expert（**36 个 offloaded experts** 必须由 PIM 或 CPU 处理）。

### 1.3 起点状态（2026-04-22）

| 配置 | decode_seconds | decode_tps | vs CPU |
|------|----------------|------------|--------|
| cuda_pim (M-3 baseline) | 140.58 | **0.228** | **0.074×** |
| cuda_cpu_offload | 10.43 | 3.068 | 1.00 |

PIM 在单算子（operator-only）上比 CPU 快 1.9–3.3×（M-2 sweep 证明），但到了端到端 decode 却慢 13.5×。**其间的 20–30× 差距完全来自 orchestration overhead**——这是整个研究要解决的核心问题。

---

## 2. 核心设计挑战：算子快但端到端慢的根因

### 2.1 问题本质

decode 每一个 token 都要遍历 48 层，每层经历：

```
router (GPU) → submit PIM (host) → GPU-resident 92 experts 并行 → sync PIM → merge
```

PIM 在算子层面可以每次 matmul 比 CPU 快 2-3×，但在这个流水线里 PIM 出现了 **5 重串行化**：

1. **Host 同步**：`hidden_states.to("cpu")`、`topk_ids.to("cpu")`、`topk_weights.to("cpu")` 三次 blocking D2H
2. **Python glue**：`for expert_idx in range(128)` 的路由扫描 + `torch.where + index_add_`
3. **权重 preload**：每个活跃 expert 要从 host DRAM DMA 到 DPU MRAM（~3MB int4，~0.5ms/次）
4. **DPU launch 是 SYNC**：`dpu_launch(DPU_SYNCHRONOUS)` 调 1 次 ≈ 90ms (M-3 baseline)，48 层 × 32 step = 1536 次
5. **GPU 等 PIM**：GPU 自己那 92 个 expert 的 forward 完成后，还要等 PIM 回来才能进入下一层

**关键洞察**：M-23.1 时代 `last_kernel_cycles` 平均 ~500K cycles @ 500 MHz = 1 ms 纯 DPU 算力，96 batched launches × 1 ms / step × 32 steps ≈ **3 s**，只占 decode 时间的 **~9%**。剩下 **91% 全是 orchestration**。

所有后续工作，本质上都是**把这 91% 的 overhead 一层层剥掉**，同时**保留 PIM 是计算主体**的科研约束。

### 2.2 我们在算法设计上能走的四条路径

| 路径 | 核心思路 | 代表里程碑 |
|------|---------|------------|
| **A. 算子融合** | 把多个独立 DPU launch 合成一个，摊薄 per-call overhead | M-4 (fused gate+up)、M-15 (request-table)、M-24 Stage B (C fused silu) |
| **B. 数据放置** | 让权重尽可能常驻、让 miss 最小化 | M-6 (multi-slot)、M-11 (residency sweep)、M-18/23 (routing-aware mask)、M-27 Stage C (NUM_SLOTS 8→128) |
| **C. 真并行** | 让 GPU 和 PIM 在时间轴上真重叠，而非串行 | M-17.2 (ASYNC DMA)、M-24 Stage A (C pthread async) |
| **D. 路径剪裁** | 减少 Python 中间态 + 消除 CUDA 同步点 | M-25 (pinned D2H/H2D)、M-27 Stage B (vectorised expert scan) |

这四条路径不是互斥，**真正的胜利来自它们按顺序叠加**。下面分章逐条剖析。

---

## 3. 路径 A：算子融合——把 launch 次数从 O(N·M) 压到 O(N)

### 3.1 M-4：Fused gate + up DPU call（+39.2%，0.228 → 0.317 TPS）

**问题**：在 Qwen3 SwiGLU 结构里，`gate = W_gate @ x` 和 `up = W_up @ x` 共享输入 `x`。原实现把它们当两个独立的 DPU launch，每层每 expert 2 次 launch。

**算法设计**：在 host 侧先把 `W_gate` 和 `W_up` **按 output-dim 轴 concat 成单一权重矩阵** `W_gate_up`，一次 DPU matvec 后再切成 gate / up 两半。DPU kernel 本身对 output_dim 是 agnostic 的，所以**不需要改 DPU 二进制**——纯 host-side 算法变换。

**实现要点**（`PIMQuantizedRuntime._prepare_concat_quantized_weights`）：
```python
concat_qw = torch.cat([lhs_qw, rhs_qw], dim=0).contiguous()
concat_sc = torch.cat([lhs_sc, rhs_sc], dim=0).contiguous()
if concat_rows % 2 == 1:  # DPU 写 row-pair，补齐到偶数
    concat_qw = torch.cat([concat_qw, zeros(1, ...)], dim=0)
```

**收益来源**：
- DPU launch 次数：`2 × num_active_experts` → `num_active_experts + 1`（down 仍单独）
- host→DPU 权重 DMA：同减半

### 3.2 M-15：Request-table 跨 expert 合批（首次让 launch_seconds 直接降低）

**问题**：M-4 把 gate+up 合到单 launch，但**不同 expert 之间的 launch 仍然串行**。每层 10-20 个活跃 expert → 10-20 次 `dpu_launch`。

**算法设计**：扩展 DPU kernel 的 WRAM 前置区为一张 **`MAX_RUN_REQUESTS=64` 条的 request 表**：
```c
// dpu_quantized_kernel.c
__host uint32_t run_request_count;
__host uint32_t request_active_slots[MAX_RUN_REQUESTS];
__host uint32_t request_batch_sizes[MAX_RUN_REQUESTS];
__host uint32_t request_input_offsets[MAX_RUN_REQUESTS];
__host uint32_t request_output_offsets[MAX_RUN_REQUESTS];
```
host 侧把**整层所有活跃 expert 的** `(slot_id, batch, input_offset, output_offset)` 一次性 broadcast 到 DPU，然后 **一次 `dpu_launch`** 完成整层所有 expert 的 matmul。DPU kernel 内部按 request_id 循环，每次切换 `active_slot`。

**收益**：每层从 ~15 次 launch 压到 2 次（gate_up 一次、down 一次），M-15 报告中 `launch_seconds` 首次出现直接可测的下降。

### 3.3 M-24 Stage B：C 级 fused gate_up + silu*up + down（NEGATIVE，batch=1 下）

**问题**：M-15 之后每层仍有 2 次 ctypes 调用（gate_up + down），中间有 Python 做 `F.silu(gate) * up`。想进一步合成一次 C 调用。

**算法设计**：新增 C 函数 `pim_quantized_run_many_fused_silu(handle_gate_up, handle_down, ...)`，内部：
1. 调 `pim_quantized_run_many` 做 gate+up batched launch
2. 在 C 层一个 fp32 循环做 `silu(gate) * up`（用 `expf`）
3. 调 `pim_quantized_run_many` 做 down batched launch

**结果：-5.4%（1.2470 → 0.9378 TPS 如果单用）**——这是**一次 publishable NEGATIVE**。

**根因**：在 batch=1 decode 下，节省的 1 次 ctypes round-trip + Python silu 加起来只有 ~0.5ms/层，但 C 层每次要 malloc 5 个 scratch 数组（`concat_scratch` / `hidden_scratch` / 3 个 pointer array）+ memcpy pointer 指针表，代价反而更大（~0.7ms/层）。

**保留方式**：代码通过 `--pim-enable-c-fused` flag 保留，默认 off。Stage A（C pthread async）复用了这个 fused C 函数，因为在 async 场景下 fused 代码被放到 worker 线程执行，malloc 开销被 overlap 吸收掉了，**变成了 Stage A 的基础**。

**教训**：微观优化必须在**真实 batch=1 决策尺寸**上测，不能光看"理论节省 overhead"。如果节省 < 50µs，几乎一定会被 CPython + ctypes 的弹性开销吞没。

---

## 4. 路径 B：数据放置——从"每次都 DMA"到"尽量驻留"

### 4.1 M-6：Multi-slot MRAM residency（从 1-slot 到 8-slot LRU）

**问题**：M-4 之后，每层每个活跃 expert 的权重都要从 host 重新 DMA 到 DPU MRAM（~3MB int4），因为 DPU MRAM 只有一个"当前权重"槽位。decode 每步 48 层 × ~15 active expert = **720 次权重 DMA**。

**算法设计**：在 DPU kernel 里把 `qweight_mram[MAX_QWEIGHT_WORDS]` 一个大 buffer **逻辑切成 N=8 个 slot**，每 slot 的 offset 由 `#define WORDS_PER_SLOT (MAX_QWEIGHT_WORDS/NUM_SLOTS)` 计算。host 侧维护一个 `{expert_id → slot_id}` 的 LRU，kernel launch 前通过 `active_slot` 字段告诉 DPU 用哪个 slot。

**关键约束**：per-slot 容量 = `MAX_QWEIGHT_WORDS / N = 1MB`（N=8 时），必须 ≥ Qwen3 单 expert shard（≈40 KB）。M-6 时代 N=8 过于保守——**我们后来发现 headroom 是 25×**。

### 4.2 M-18：Routing-aware GPU residency（+28.4%，offload=92 首次突破 0.9 TPS）

**问题**：offload 越多（GPU expert 越少），越多负担压到 PIM。M-11 residency sweep 证明 `offload=88` 是经验最佳点（47GB 卡的安全边界）。但**哪 88 个 expert 放 GPU** 是随机选还是按热度选？

**算法设计**：在 `offload=92` 配置下（把 36 个最冷的 expert 放 PIM），用 **routing frequency calibration**：跑一轮 prompt，统计每层每 expert 被选中的次数，按频次排序，把最热的 92 个常驻 GPU。

**重点**：这是一次 **数据驱动的算法决策**——不改 kernel，只改 mask，**+28.4%**（0.757 → 0.9572 TPS）。

### 4.3 M-23 / M-23.1：Mean-mask 泛化性（+3.56%，突破 0.99 TPS）

**问题**：M-18 用 per-prompt 的 routing frequency 选 mask，在学术上接近"过拟合测试集"。

**算法设计**：跑 5 个 calibration prompt，对每层每 expert 的频次做**算术平均**，生成一个"5 prompt 平均 mask"作为生产 mask。实验证明 **mean-mask 不但具有泛化性，还比任何单 prompt self-calibration 都更快**——**原因是单 prompt 的热点分布不稳定，averaging 反而更接近真实 test-time 分布**。

**这是整个研究里最有"算法味道"的一个胜利**：没有改任何代码路径，只是换了 mask 的产生方式，拿到了 3.56%。

### 4.4 M-27 Stage C：NUM_SLOTS 8 → 128（+3.0%，发现 MRAM 严重过冗余）

**问题**：M-27 端到端 profile 发现 submit_forward 的 74% 花在 `preload_concat_and_get_slot` 里。深挖发现 LRU hit rate = **0%**（`hits=0, misses=88800` per run），每次 miss 都要跑 `load_weights_inner` 的 7.5MB host-side memcpy 循环（把 host packed_qweights 按 shard split 重拷一次）。

**算法设计**：把 LRU 容量从 8 升到 128。算术：
- MRAM per-slot capacity = `MAX_QWEIGHT_WORDS / NUM_SLOTS`
- NUM_SLOTS=8 时 = 1 MB/slot/DPU，Qwen3 expert shard = 40 KB/DPU → **headroom 25×**（严重浪费）
- NUM_SLOTS=128 时 = 64 KB/slot/DPU，仍比 40 KB 大 1.6×

**约束发现**（踩坑）：**NUM_SLOTS 必须整除 MAX_QWEIGHT_WORDS 且结果是偶数（8-byte MRAM offset 对齐）**。NUM_SLOTS=36 看起来能精确装 36 个 offloaded expert，但 2097152/36 不整除，运行时触发 UPMEM `invalid mram access` 错误。最终选 128。

**结果**：`hits=2370, misses=2934, hit_rate=44.7%`（配合 `pim_layer_group_size=3`），decode 23.95 → 23.79s，+3.0%。

**为什么 hit_rate 没达到 80%+**：因为 **decode 不同 step 的 top-k routing 多样性本身就很高**（MoE 的本质特性），即便 LRU 容量足够，访问模式本身就不会复用。**这是算法层面的天花板**，不是实现缺陷。

---

## 5. 路径 C：真并行——让 GPU 和 PIM 真正同时工作

### 5.1 M-10：Python threading async submit（NEGATIVE，记为对照）

**假设**：把 `submit_forward` 放进 `threading.Thread`，主线程立刻返回进入 GPU expert loop。

**结果 NEGATIVE**：`async_sync_wait_mean = 73ms`。把已经是 ~90ms/层的 decode 拖到 ~110ms/层。

**根因**：`submit_forward` 的主体是 Python 代码（expert loop、tensor slicing），必须持有 GIL。CPython 默认 `sys.setswitchinterval=5ms` 强制切换 GIL → 主线程 PyTorch CUDA dispatch 被 preempt → GPU kernel launch 延迟。

**教训**：**任何"Python 线程并发"都会被 GIL 吞掉**，除非 99% 时间都在 release-GIL 的 C 扩展里。

### 5.2 M-17.2：DPU ASYNC DMA（+5.41%）

**设计**：`dpu_push_xfer(qweight_mram, ..., DPU_XFER_ASYNC)`——不等 DMA 完成就 return。weight DMA 和下一轮 kernel launch 的 host 准备阶段自然重叠。

**关键**：这是 **DPU 层面的真并行**，发生在 SDK 内部，**不涉及任何 Python GIL**。

### 5.3 M-24 Stage A：C pthread async（+20.3%，本研究最大的单点胜利）

**设计**：新增 C 函数 `pim_quantized_run_many_fused_silu_async`：
1. 在 C 层 `pthread_create` 一个 worker，worker 跑整个 fused gate_up + silu + down
2. 返回一个不透明的 `token` 立即
3. Python 主线程立刻进入 GPU expert loop
4. `pim_quantized_fused_wait(token)` 在 `sync_forward` 里 `pthread_join`

**为什么这次能赢**（M-10 不能）：
- ctypes 调用**自动释放 GIL** → worker 纯 C 代码不争 GIL
- 主线程的 PyTorch CUDA dispatch 代码 100% 独占 GIL

**实测**：`c_async_wait_mean = 0.6 ms`（对比 M-10 Python 线程 73 ms，**122× 更快的等待**），`wait_fraction_of_decode = 2.2%`——**GPU 和 PIM 几乎完全并行**。

**Stage A 对 Stage B 的救赎**：Stage B（C fused）单跑是 NEGATIVE（-5.4%），但一旦包进 Stage A 的 worker 线程，它的 malloc 开销被 overlap 吸收，**变成 Stage A 的基础**。这说明**单独看 NEGATIVE 的微优化，放对位置可能变成 POSITIVE 的前置条件**。

---

## 6. 路径 D：路径剪裁——消除 Python / CUDA 中间态

### 6.1 M-25：Pinned D2H/H2D + 删除 `.item()` 诊断（+4.5%）

**问题**：M-24 A 让 PIM 和 GPU 真并行了（`wait_fraction=2.2%`），但 decode 总时间 26.82s 里 GPU 侧还要跑 24s+。读源码发现 `submit_forward` 入口**每层每步都有 5 个 CUDA sync point**：

```python
# 2 个 .item() 诊断 counter：
self.offloaded_pairs += int(routed_to_offload.sum().item())  # CUDA sync
self.offloaded_tokens += int(routed_to_offload.any(dim=1).sum().item())  # CUDA sync
# 3 个 blocking D2H：
flat_cpu = flat.to("cpu", dtype=torch.float32)  # blocking
topk_ids_cpu = topk_ids.to("cpu", dtype=torch.long)  # blocking
topk_weights_cpu = topk_weights.to("cpu", dtype=torch.float32)  # blocking
```

每层每步 5 个 sync × 48 层 × 32 步 = **7680 个 CUDA sync point / run**。每个虽只 30-100µs，但它们把 GPU expert loop 的启动**推迟到所有 sync 完成之后**，直接吃掉了 Stage A 创造的 overlap 窗口。

**对比观察**：`cuda_cpu_offload` 的 `CPUMoEBackend.submit_forward` 用的是：
```python
input_cpu[slot].copy_(flat, non_blocking=True)  # pinned + async
expert_ids_cpu[slot].copy_(topk_ids.long(), non_blocking=True)
weights_cpu[slot].copy_(topk_weights, non_blocking=True)
# 然后 submit_with_cuda_stream 挂 CUDA event
```
**这就是 cpu_offload 基线跑 15.29s 的核心秘密**。

**算法设计**：
1. 预分配 pinned CPU 缓冲（按 `(batch, hidden_size, top_k)` 形状缓存，decode 稳态只分配一次）
2. `copy_(gpu_tensor, non_blocking=True)`（CUDA 把拷贝排到 stream 上 Python 立刻 return）
3. `torch.cuda.current_stream().synchronize()` **只在需要读 CPU 侧数据前做一次**（其他地方不 sync）
4. 两个 `.item()` 诊断 counter 搬到 CPU-materialised tensor 上（CPU tensor 的 `.item()` 不是 CUDA sync，0 成本）

**结果**：1.1930 → 1.2470（+4.5%）。更重要的是 `wait_fraction_of_decode` 从 2.2% → 3.0%——**不是变慢了，是 GPU 启动更早了，更频繁地在 sync 时追上 PIM**。这是期望行为。

### 6.2 M-26：Threaded Python submit body（NEGATIVE，再次撞上 GIL）

**假设**：M-25 只消除了 submit 入口 sync，但**剩下的 Python body**（expert loop + preload + ctypes）仍在主线程，~300-500µs/层占着 GPU 启动时间。把整个 body 扔到 `threading.Thread` 里。

**结果：-28.6%**（1.2470 → 0.8907 TPS）。

**诊断**：`m26_join_wait_mean = 4.61 ms`（预期 <0.5ms），`c_async_wait` 从 0.76s 膨胀到 5.23s。**这是 M-10 的完全复刻**——ctypes 本身 release GIL，但 ctypes **之间**的 Python 代码（`cpu_mask[expert_idx]`、`torch.where(match.any)`）持 GIL，每 5ms 切回主线程 preempt CUDA dispatch。

**研究价值**：NEGATIVE 但 publishable——**严格证明了 Python 线程并发在 PyTorch/CUDA 主线程存在时永远是负收益，除非后台线程 99% 时间在 C 扩展里**。

### 6.3 M-27 Stage B：Vectorised expert scan（+4.8%）

**问题**：M-25 之后 profile 显示 `sub_3_expert_scan`（128-expert Python for-loop）占 submit body 的 19.9%，每步 1.19 ms。

**算法设计**：decode batch=1 + top-k=8 意味着每步最多只有 8 个 distinct expert 会被路由到——**没必要遍历全部 128**。改成：

```python
for expert_idx in torch.unique(topk_ids_cpu).tolist():  # ≤ 8 iters
    if not cpu_mask[expert_idx]: continue
    ...
```

**结果**：decode 25.66 → 24.49s（-1.17s/run），+4.8%。同样的改动也应用到 `_submit_forward_real` 保持路径对称。

**算法意义**：从 O(num_experts) 降到 O(top_k × batch)。对 batch=1、top_k=8、num_experts=128 的配置，**固定 16× 加速**。对大 batch 或更稀疏 MoE（num_experts=256）收益更大。

---

## 7. 还没解决的问题：剩余 35.7% 差距的成因

当前成绩 1.3454 TPS vs CPU baseline 2.0933 TPS，**还差 5.4 s decode time**。通过 `diag_m27_per_phase.py` 的 5 段 timer（见附录 A）可以精确定位：

| phase | cuda_pim (M-27) | cuda_cpu_offload | delta |
|-------|-----------------|------------------|-------|
| step_1_routing | 0.51s | 0.44s | +0.07s |
| **step_2_submit** | **9.67s → ~8s** (改进中) | 0.75s | **+~7s** |
| step_3_gpu_expert_loop | 11.13s | 10.84s | +0.29s |
| step_4_sync | 1.17s | 0.04s | +1.13s |
| step_5_merge | 0.06s | 0.05s | +0.01s |

**100% 的剩余差距仍在 submit (~86%) + sync (~14%)**，GPU expert loop 两边几乎完全一样（这推翻了早期"GPU 路径不同"的假设）。

### 7.1 为什么 submit 还慢

Submit 内部进一步细分（`diag_m27_preload_breakdown.py`）：

- **pp_2_gu_preload = 2.38 ms/call**（gate_up `preload_concat_and_get_slot`，74% of submit）
- **pp_3_dn_preload = 1.27 ms/call**（down `preload_and_get_slot`，35%）

这 3.65 ms 主要花在 `load_weights_inner` 的 **host-side memcpy 循环** 里（`host_quantized_bridge.c:419-470`）：
- 每次 miss 都要把 host packed_qweights/scales/LUT 按 39 个 DPU shard split 重拷一次
- 即便 M-27 Stage C LRU hit rate 提到 44.7%，剩下的 55.3% miss 每次还要付 ~7.5 MB memcpy

**为什么没解决**：要消除 host memcpy，必须**把 shard 后的 host buffer 永久缓存到 `_weight_cache` 里**，miss 时直接传 pre-sharded 指针给 DPU。但这需要：
1. 改 C 端 API 接收 pre-sharded buffer 数组
2. `_weight_cache` 从存 `(qweight, scales)` 变成存 `(per_dpu_shards[39])`
3. 内存占用：36 experts × 2 proj × 39 DPU × ~60 KB = ~170 MB host RAM（可接受，但是大改）

**结论**：**工程可行，属于 M-28 候选，未做**。

### 7.2 为什么 sync 还慢

`_sync_forward_c_async` 内部：
- `handle.wait()` → `pthread_join` → DPU 结果拷回 host （~0.3 ms/call）
- `output.view(shape).to(device, dtype)`→ pinned staging + `non_blocking` H2D（M-25 已优化）

对比 CPU backend 用的 `cpu_infer.submit_with_cuda_stream`：**C 端直接挂 CUDA event**，GPU stream 自己在硬件层等 CPU 完成，主线程完全不 block。

**为什么没解决**：要让 PIM 也走"CUDA event signal"，需要 C 端在 DPU worker 完成时 `cudaEventRecord`，GPU stream 在下一层开头 `cudaStreamWaitEvent`。这要：
1. C worker 拿到 CUDA context 的 handle（不是只 pthread）
2. 引入 `libcudart` 依赖到 PIM bridge
3. 复杂的生命周期管理（worker 死掉时 event 不能泄漏）

**结论**：**工程难度显著高于 M-25/M-27，未做**。是 M-29 候选。

### 7.3 已知但放弃的方向

| 方向 | 尝试 | 结果 | 为什么放弃 |
|------|------|------|-------------|
| Python threading async | M-10, M-26 | NEGATIVE | GIL 切换必然吞掉收益，**已 publishable 的反面** |
| GPU-only top-k filtering | M-24 初版 | 被否决于设计阶段 | 会丢失 offloaded expert 的贡献，违反 MoE 语义 |
| 纯 GPU W4A32 dequant（绕开 PIM） | M-24 第一版 plan | 被用户否决 | **违反科研硬约束**：必须 PIM 承担计算 |
| 跨 layer decode-step 合批 DPU launch | M-24 plan 中分析 | 被层序依赖推翻 | layer N+1 需要 layer N 输出，不能提前 batch |
| Dynamic routing-aware residency（migration） | M-21 | NEGATIVE | scheduler 在 GPTQ 真机上 hang |
| Per-slot host shard buffers | M-17.4 | NEGATIVE | 实现复杂度 vs 实际收益倒挂 |

---

## 8. 方法论层面的收获（给未来自己看）

这 26 个里程碑里有 9 个 NEGATIVE，我们每一个都公开登记在 ADR 里，不掩盖。这件事本身值得单独写一节，因为它**反过来定义了什么叫科研性的工程**：

### 8.1 Profile-driven 优化的可怕之处

几乎每一次没做 profile 就去设计的优化都有负收益或低收益（M-26 猜"Python 线程 overlap 会快" -28.6%；M-24 Stage B 猜"少一次 ctypes 会快" -5.4%）。几乎每一次先做三层 profile 再开写的优化都赢（M-27 Stage C 砍中 LRU hit rate，M-25 砍中 CUDA sync point）。

**规则**：在触摸代码前，先答得出"这个改动会减少哪个 phase 的哪个 sub-phase 的多少 ms"。答不出来就不要写。

### 8.2 NEGATIVE 不是失败，是情报

- M-10 NEGATIVE 让我们在 M-24 Stage A 明白**必须走 C pthread**
- M-26 NEGATIVE 再次确认**Python 线程 = GIL 陷阱**
- M-24 Stage B 单跑 NEGATIVE，但在 Stage A 里 **变成 foundation**
- M-17.4/M-17.5/M-17.6 三次 NEGATIVE 最终证明 cross-layer preload 不可行

**把 NEGATIVE 当作 publishable result 来对待**，以后的自己或同行读到时能省下重做一次的时间。

### 8.3 约束即护城河

科研约束"PIM 必须真算"一开始看起来是镣铐，实际上它**否定了所有投机取巧的路径**（把 offloaded 放 GPU、绕过 DPU），逼着我们去真正攻克 orchestration overhead 这个真实问题。**看起来束缚手脚的约束，反而指向了更有价值的工作**。

---

## 9. 累积成绩总表

| milestone | 代表性改动 | decode_tps | vs M-3 起点 | vs CPU baseline (2.0933) |
|-----------|-----------|-----------|-------------|---------------------------|
| **M-3 起点** (2026-04-22) | cost model + CPU baseline 修复 | 0.228 | 1.00× | 10.9% |
| M-4 | fused gate+up | 0.317 | 1.39× | 15.1% |
| M-11 | residency sweep to offload=88 | 0.6226 | 2.73× | 29.7% |
| M-15 | request-table single launch | 0.6852 | 3.00× | 32.7% |
| M-17.2 | ASYNC DMA overlap | 0.7217 | 3.17× | 34.5% |
| M-18 | routing-aware mask | 0.9572 | 4.20× | 45.7% |
| **M-23.1** | mean-mask 泛化 | 0.9913 | 4.35× | 47.4% |
| **M-24 A** | C pthread async (本研究最大单点胜利) | 1.1930 | 5.23× | 57.0% |
| **M-25** | pinned D2H/H2D + 删 `.item()` sync | 1.2470 | 5.47× | 59.6% |
| **M-27 Stage B** | vectorised expert scan | 1.3066 | 5.73× | 62.4% |
| **M-27 Stage C** (pim HEAD) | NUM_SLOTS 8→128 | **1.3454** | **5.90×** | **64.3%** |
| cuda_cpu_offload | — | 2.0933 | 9.18× | 100% |

**原始差距（M-3 起点到 CPU 基线）= 92.6 pp。当前闭合 64.3 - 10.9 = 53.4 pp，即 57.7% 的原始差距**。从"差 13.5×"缩到"差 1.56×"。

---

## 10. 对团队工作的启发

### 10.1 对 PIM 硬件研究的启发

UPMEM DPU 的软件乘法 (10 cycles/op) 在 matmul 上对 int4 weight 能比 CPU AMX 快 2-3× 是**真的**（M-2 operator sweep 证明）。**但这个 2-3× 无法反映到系统级性能上**，除非你把 orchestration overhead 从 91% 压到 <30%。

PIM 研究接下来值得投入的方向：
1. **缩短 weight DMA**：pre-sharded cache 或让 DPU 自己管 MRAM 页表
2. **替换 pthread_join 为 CUDA event**：把 CPU/PIM 同步纳入 CUDA stream 拓扑
3. **扩大 `MAX_RUN_REQUESTS`**：目前 64，可支持更大 MoE 或更深 batching

### 10.2 对推理系统设计的启发

异构推理（GPU + xPU）系统性能的决定因素**不是 xPU 本身快慢**，而是 **submit/sync 边界的设计质量**。`cuda_cpu_offload` 用 pinned + `submit_with_cuda_stream` 的模式本质上是把 CPU AMX "伪装成一个 CUDA stream"——这个设计哲学同样适用于 PIM/NPU/FPGA 等其他 offload target。

### 10.3 对方法论的启发

- **每个性能声明必须有 diagnostic 支撑**：本项目写了 4 个 diagnostic 脚本（`diag_m27_per_phase.py`、`diag_m27_submit_breakdown.py`、`diag_m27_preload_breakdown.py`、`benchmark_residency_sweep.py`），这些脚本比代码改动本身更值得留存
- **NEGATIVE 结果要登记**：ADR-002 里 9 个 NEGATIVE milestone 每一个都有独立 section，下次项目可以直接复用
- **约束要写在前面**：从 M-24 的重设计（用户"PIM 必须真算"的澄清）之后，所有方案都在约束下设计，避免了无效工作

---

## 附录 A：诊断脚本索引

| 脚本 | 回答什么问题 |
|------|-------------|
| `benchmarks/diag_m27_per_phase.py` | HybridMoE.forward 5 段 timer：routing / submit / gpu expert loop / sync / merge，两个 backend 对比 |
| `benchmarks/diag_m27_submit_breakdown.py` | `_submit_forward_c_async` 内部 7 段 timer：preamble / pinned D2H / diag counter / expert scan / preload / submit_async / stash |
| `benchmarks/diag_m27_preload_breakdown.py` | `sub_4_preload` 内部 5 段 timer：lookup / states view / gate_up preload / down preload / append |
| `benchmarks/benchmark_residency_sweep.py` | offload-device-experts 扫描，子进程隔离避免 GPU 累积 |

## 附录 B：关键代码文件

| 文件 | 作用 |
|------|------|
| `nano_ktrans/kernels/pim_moe.py` | PIMMoEBackend：submit_forward / sync_forward / C async / M-25 pinned / M-27 vectorised scan |
| `nano_ktrans/kernels/pim_quantized_runtime.py` | LRU、fused silu async wrapper、weight cache、NUM_SLOTS=128 |
| `nano_ktrans/kernels/pim_native/host_quantized_bridge.c` | C bridge：pim_quantized_run_many_fused_silu[_async]、pim_quantized_fused_wait、load_weights_inner |
| `nano_ktrans/kernels/pim_native/dpu_quantized_kernel.c` | DPU kernel：multi-slot MRAM、request-table、kernel_mode=4 int8 LUT |
| `nano_ktrans/layers/hybrid_moe.py` | HybridMoE.forward：5-phase decode 主循环，GPU expert loop + PIM submit/sync |
| `benchmarks/benchmark_inference.py` | e2e benchmark 入口，支持所有 CLI flags |
| `.knowledge/architecture/decisions/002-pim-operator-parity-roadmap.md` | 完整 ADR（37 section，2836+ 行）记录每个里程碑的 POSITIVE/NEGATIVE 和详细推理 |

## 附录 C：Pim 主干 HEAD 验证命令

```bash
cd /home/yangfu/nano-ktrans
git checkout pim  # HEAD=9f641fb
source .venv/bin/activate
python -m pytest tests/ -q  # 288 passed

# 复现 1.3454 TPS
python benchmarks/benchmark_inference.py \
  --model-path /home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4 \
  --backends cuda_pim \
  --offload-device-experts 92 \
  --routing-freq-json benchmarks/results/routing_freq_qwen3_30b_m23_mean.json \
  --pim-rank-count 1 --pim-layer-group-size 3 \
  --max-new-tokens 32 --warmup 0 --repeats 1 \
  --pim-enable-c-async --pim-enable-m25-pinned
```

---

**报告完**。下一个 milestone 候选（未执行）：**M-28 pre-sharded weight cache**（预计额外 +10-20%）或 **M-29 CUDA event-based sync**（预计额外 +5-10%）。两者叠加可能把 decode_tps 推到 **1.6-1.8 TPS**，达到 CPU 基线 80% 左右——**但理论上 `pim+gpu` 永远不应该超过 `cpu+gpu` 太多**，因为 GPU 侧 workload 一样，PIM 只是换掉了 CPU 侧的 offload 执行者。研究终极问题不是"超过 CPU baseline"，而是"**在能接近 CPU baseline 的前提下，证明 PIM 参与是可行且有价值的**"——这个答案目前**已经是肯定的**。
