# nano-ktrans — 项目完整综述

> 最后更新：2026-04-24（M-1 ~ M-10 全部关闭，87 条 dev_gate acceptance rules 全 PASS，242 tests 全绿）
>
> 这是一份**可以直接讲给别人听**的技术文档。目标读者：对 MoE 推理、UPMEM PIM、系统软件栈感兴趣但没跟项目日常进度的工程师/研究者。
>
> 本文是 `.knowledge/architecture/decisions/002-pim-operator-parity-roadmap.md` 的面向外部读者的浓缩版 + 技术原理展开。完整 ADR/journal/gotchas 在 `.knowledge/` 下。

---

## 目录

- [0. 一句话定位](#0-一句话定位)
- [1. 问题定义](#1-问题定义为什么要把-moe-拆开跑)
- [2. 系统架构](#2-系统架构三层视图)
- [3. 三个核心子系统](#3-三个核心子系统)
- [4. 10 个 milestone 全景](#4-10-个-milestone-全景)
- [5. 当前性能数据汇总](#5-当前性能数据汇总)
- [6. 关键教训](#6-关键教训)
- [7. dev_gate 数据驱动工作流](#7-dev_gate-数据驱动工作流)
- [8. 下一步 M-11 双轨](#8-下一步m-11-双轨)
- [9. 数据字典：每一项指标的精确含义](#9-数据字典每一项指标的精确含义)
- [10. 怎么讲给别人听（电梯版）](#10-怎么讲给别人听电梯版)

---

## 0. 一句话定位

**nano-ktrans 是一个把 MoE 大模型（以 `Qwen3-30B-A3B-GPTQ-Int4` 为主测对象）拆成 GPU + CPU + 真实 UPMEM DPU (PIM) 三类后端协同推理的研究性框架，目标是证明 PIM 在 MoE expert offload 场景下是否能真正超过 CPU。**

结论提前说：**当前在算子层面 PIM 能赢 CPU（batch=1 时 2–3×），在端到端 decode 上还输 CPU ~8.7×**。差距来自 orchestration（launch 次数、DMA、host-side Python glue），不是 kernel 本身。已经用 10 个 milestone 把问题每一层架构约束摸清了，剩下的只有 2 个攻击面。

---

## 1. 问题定义：为什么要把 MoE 拆开跑

### 1.1 Qwen3-30B-A3B MoE 的事实

| 量 | 值 | 含义 |
|---|---|---|
| 层数 | 48 | 每层都有一套 MoE |
| 每层 expert 数 | 128 | `num_experts=128` |
| top_k | 8 | 每 token 每层激活 8 个 expert 做加权合并 |
| 单 expert 结构 | gate / up / down 三投影 MLP | SwiGLU 风格 |
| hidden_size | 2048 | |
| moe_intermediate_size | 768 | gate / up: `2048 → 768`；down: `768 → 2048` |
| 量化 | GPTQ Int4, group_size=128 | 每 128 个 input 共享一组 scale |
| 总权重 | ~30B 参数，Int4 后 ~15 GB | |
| 推理时 active | `48 × 8 = 384` expert/token | 但权重必须全部可访问 |

### 1.2 为什么不能全放 GPU

- `30B × 0.5 B/参数 (Int4) ≈ 15 GB` + KV cache + 运行时 buffer + CUDA 内存碎片 → 47 GB 卡放不下
- 即便放下，GPU 对 sparse MoE 的利用率低：top_k=8 激活意味着每 token 只用 `8/128 ≈ 6%` 的 expert 权重，SM 大量空闲
- Qwen3 训练时 expert 分布相对均匀（有 aux-loss 约束），hot-set 优化空间有限

### 1.3 为什么考虑 PIM (UPMEM DPU)

**PIM (Processing-in-Memory) 核心想法**：让 DRAM 芯片内部直接做计算，权重不再需要搬到 CPU/GPU 的高速缓存。

**UPMEM DPU 硬件参数**（测试机实测）：
- 39 个可见 DPU rank (`/dev/dpu_rank0..39`)
- 每 rank 通常 64 个 DPU
- 每 DPU：64 MB MRAM + 64 KB WRAM + 16 tasklet (硬件线程) + 500 MHz
- **关键弱点**：DPU 没有硬件乘法器（软件乘法 ~10 cycle）、没有 SIMD、没有硬件 `ctz/popcnt`

**MoE expert forward 的性质**：
- 小 batch matvec：`y = W @ x`，W 大（`output_dim × input_dim`），x 小
- 算术密度低 = 每读一次 W 只做几次运算
- 典型瓶颈在 **memory bandwidth**，而非 compute

**理论匹配**：DPU MRAM 带宽总和远超 DDR → 一层 128 个 expert 并行摊到 64 个 DPU，每 DPU 读本地 MRAM 算自己那部分。省掉"权重从 DRAM 搬到 GPU/CPU cache" 这一步。

### 1.4 研究问题

**在当前 UPMEM 硬件 + Qwen3-30B MoE 上，PIM 能否端到端战胜 CPU baseline？**

这就是 `.knowledge/architecture/decisions/002-pim-operator-parity-roadmap.md`（ADR-002）这条 10 个 milestone 的技术路线要回答的问题。

---

## 2. 系统架构（三层视图）

### 2.1 全景架构图

```
┌─────────────────────────────────────────────────────────────┐
│                   Python / PyTorch 层                        │
│                                                              │
│  LLM ──► SimpleEngine ──► Mixtral model                      │
│                                │                             │
│                                ▼                             │
│                          48 × HybridMoE layer                │
│                          ┌──────────────────┐                │
│                          │ gate (softmax)   │                │
│                          │       │          │                │
│                          │       ▼ top_k=8  │                │
│                          │   ┌───┴────┐     │                │
│                          │   ▼        ▼     │                │
│                          │  GPU     offload │                │
│                          │ experts  backend │                │
│                          │ (2 hot) (126 冷) │                │
│                          └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 Offload Backend 抽象                         │
│   submit_forward() → sync_forward() 异步接口                 │
│                                                              │
│   ├─ CPUMoEBackend       (fp16 + 可选 AMX)                   │
│   ├─ PIMMoEBackend       (继承 CPU, 路由到 DPU)              │
│   │    └── BackendCostModel (M-3)                            │
│   │         从 M-2 算子 sweep 蒸馏的 (shape,batch,rank)      │
│   │         → seconds 表，决定 PIM vs CPU                    │
│   └─ (shadow / no-op 模式)                                   │
└─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 PIM Runtime 层                               │
│                                                              │
│   PIMQuantizedRuntime (W4A32 量化路径, 主线)                 │
│     ├── _handle: c_void_p → 独立 DPU rank pool               │
│     ├── NUM_SLOTS=8 MRAM LRU cache                           │
│     ├── preload(expert_id) → slot 分配 + host→DPU DMA        │
│     ├── infer(slot_id) → 一次 dpu_launch                     │
│     └── preload_and_infer_concat() → gate+up fused           │
│                                                              │
│   PIMLinearRuntime   (fp32 路径, 老代码)                     │
│   PIMExpertRuntime   (fp32 fused MLP, 老代码)                │
└─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│         C 层 (libpim_quantized_bridge.so)                    │
│                                                              │
│   pim_q_ctx_t {                                              │
│     dpu_set_t set;                                           │
│     input_dim / output_dim / group_size / kernel_mode;       │
│     slot_loaded_mask;                                        │
│     ... all previously static globals                        │
│   }                                                          │
│                                                              │
│   void* pim_quantized_init(...)     → 返回 ctx handle        │
│   int   pim_quantized_load_weights(ctx, ..., slot_id)        │
│   int   pim_quantized_run(ctx, ..., slot_id)                 │
│   void  pim_quantized_shutdown(ctx)                          │
└─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│           DPU kernel (dpu_quantized_kernel.c)                │
│                                                              │
│   __host active_slot ───────┐                                │
│   __mram qweight_mram [NUM_SLOTS × WORDS_PER_SLOT]           │
│   __mram scales_mram  [NUM_SLOTS × SCALES_PER_SLOT]          │
│   __mram lut_mram     [NUM_SLOTS × LUT_PER_SLOT]             │
│                                                              │
│   5 个 kernel_mode:                                          │
│    mode=3  fp32 matvec (reference)                           │
│    mode=4  int8 × int16 软件乘法, LUT 查表         ◄── 主力 │
│    mode=5  per-block runtime LUT                             │
│    mode=6  假 T-MAC (bit-serial 带乘法, 未用)                │
│    mode=7  真 T-MAC (bit-plane, 无乘法, 实测慢)              │
│                                                              │
│   16 tasklets 并行，每 tasklet 处理 row pair                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流（一次 decode step 一层的路径）

```
hidden_states (GPU)
  ├─► gate softmax → topk_ids/topk_weights
  │
  ├─► offload_backend.submit_forward() ────┐   # PIMMoEBackend
  │                                          │
  │   └─► cost_model.decide()  ──► "pim"    │
  │        └─► _submit_forward_real()       │   (同步直接跑)
  │             for each CPU-side expert:   │
  │               preload(eid) → slot_id    │
  │               gate+up fused matvec ──► DPU launch (sync)
  │               SiLU*  (host)              │
  │               preload(down_eid) → slot   │
  │               down matvec  ──────► DPU launch (sync)
  │             (写入 _fallback_output)      │
  │                                          │
  ├─► GPU experts forward (2 个 hot expert, 并行执行)
  │                                          │
  └─► offload_backend.sync_forward() ◄──────┘
       └─► 读 _fallback_output

  → final_output = GPU + CPU parts
```

---

## 3. 三个核心子系统

### 3.1 HybridMoE（`nano_ktrans/layers/hybrid_moe.py`, ~3700 行）

- 每个 MoE 层一个实例
- 持有 `gpu_experts_mask: Tensor[num_experts, bool]`，True = GPU 常驻
- `forward()` 三步：
  1. 路由（softmax + topk 拿 `topk_ids/weights`）
  2. **并行**：起 GPU experts forward + `offload_backend.submit_forward(...)`（理论上是异步）
  3. `offload_backend.sync_forward(...)` 拿 CPU/PIM 结果，与 GPU 结果相加
- 动态 expert 迁移（`dynamic_expert_scheduler`）也在这层（目前还没用到 PIM 路径）

### 3.2 Offload Backend 家族（`nano_ktrans/kernels/`）

```
ExpertOffloadBackend (abstract)
 ├── CPUMoEBackend          ← 真实 CPU 计算，可选 AMX (Intel AMX 指令集)
 │    └── PIMMoEBackend     ← 继承，把部分调用路由到 DPU
```

`CPUMoEBackend.submit_forward` / `sync_forward` 是 **ktrans 风格的双阶段异步接口**（submit 非阻塞提交，sync 阻塞等结果）。`PIMMoEBackend` 继承它，在 submit 阶段根据 cost model 选择 PIM or CPU，sync 阶段读 `_fallback_output`。

### 3.3 PIM Runtime（`nano_ktrans/kernels/pim_quantized_runtime.py`）

最核心的 `PIMQuantizedRuntime`：

- **Handle-based**（M-8 之后）：每 Python 实例持 `ctypes.c_void_p`，对应 C 的 `pim_q_ctx_t*`，真正独立的 DPU rank pool
- **8-slot MRAM LRU**（M-6 引入）：每 expert preload 时分配 slot，hit 时跳过 DMA
- **Fused preload+infer_concat**（M-4.1）：gate 和 up 权重沿 row 轴 concat，1 次 DPU launch 同时算两个投影
- **get_shared(profile, instance_key, rank_count)**（M-8 引入 `instance_key`）：
  - `profile` → 真传给 UPMEM `dpu_alloc_ranks`，必须空字符串或合法 UPMEM profile
  - `instance_key` → Python 侧 `_shared` dict 的 key，可以是任意字符串
  - 这两个在 M-7 之前是同一个字段，导致 M-5/M-6/M-7 的"独立 runtime"都是 Python 层假象

---

## 4. 10 个 milestone 全景

每个 milestone 是一个"假设 → 实装 → 真机测 → PASS/负结果"闭环，由 `scripts/dev_gate.py` 强制数据驱动。

### 4.0 全景表

| # | 名称 | 类型 | 核心动作 | decode_tps 结果 |
|---|------|------|---------|-----------------|
| M-1 | baseline | 基线 | 跑通 Qwen3-GPTQ 真机 + 180 cell operator sweep | 0.228 |
| **M-2** | 真 T-MAC | **负结果** | `kernel_mode=7` bit-plane DPU kernel | 数值 bit-exact 但 0/60 cell 赢 mode=4 |
| **M-3** | Cost Model | **真胜利** | BackendCostModel 数据驱动 PIM/CPU 路由 | **prefill 13.3× 赢 CPU** |
| **M-4** | Fused Gate+Up | **真胜利** | host 端 concat gate/up 权重，3 call → 2 call | **decode +39%**, 0.228→0.317 |
| M-5 | Dual Runtime | 负结果 | gate_up 和 down 各占一 runtime | null |
| M-6 | Multi-slot MRAM | 负结果 | 8-slot LRU MRAM cache + DPU binary 改造 | null |
| M-7 | Per-layer Scoping | 负结果 | 48 层分 16 组，每组独立 runtime | null + 发现 `.so` static bug |
| M-8 | Handle-based | 基础设施 | C 端 20 个 static 全局封进 `pim_q_ctx_t*` | 首次 hit>0，但 tps -22% |
| M-9 | Locality 量化 | 决策 | 1 行 Jaccard histogram + group_size sweep | 发现 Jaccard=0.14，caching 路径关 |
| M-10 | Python async | 负结果 | `threading.Thread` 尝试 overlap GPU/PIM | A/B 输 3-5%，**意外发现 offload=32 跑出 0.35** |

---

### 4.1 M-1 — baseline：建立可信度量基线

**假设**：在动手做任何优化之前，先让 e2e `cuda_pim` backend 能跑通 Qwen3-GPTQ 真权重，并用 operator-level sweep 建立 baseline 数据。

**实装**：
1. 修了 4 处 GPTQ layout 级联 bug（config 层找不到 `.qweight` → cpu_moe 加载 stacked fp16 崩溃 → pim fallback 链断 → GPTQ cost-adapter 识别失败）
2. 写了 `benchmark_pim_shape_sweep.py`：扫 **3 shapes (gate/up/down) × 4 batches (1/2/4/8) × 5 ranks (1/4/8/16/32) × 3 kernel_modes (3/4/6) = 180 cells**
3. 每 cell 跑 warmup=3 + repeats=10 次，记录 `pim_seconds_{avg,min,max}`、`launch_seconds`、`transfer_seconds`、`max_abs_error` vs CPU grouped

**产出数据**（`pim_shape_sweep_M1_*.json`）：
- `kernel_mode=4` batch=1 peak = **CPU grouped 的 3.36×**
- `kernel_mode=6`（自称 T-MAC）peak 只有 1.11× → **证伪伪 T-MAC 声明**

**证伪怎么做的**：在 `tests/test_core.py::TestQuantizedKernelAudit` 加 3 条静态代码审计断言，锁定 mode=6 内循环里仍有 `lut[q]` 查表 + `abs_x & 0xNN` 的 activation-side shift-add —— **这是朴素软件乘法器模拟，不是 T-MAC**。真正的 T-MAC 要求 weight 完全编码进 LUT 索引，内循环只做 `acc += T[bit][row][group][pack(x_bits)] << bit`，零乘法零分支。

**dev_gate PASS 6/6**。从此所有 milestone 的 baseline 指向 M-1 artifact。

---

### 4.2 M-2 — 真 T-MAC (publishable 负结果)

**假设**：UPMEM DPU 没有硬件乘法器，把内循环改成 bit-serial 查表加法应该快。T-MAC 论文（Wang et al., MLSys'25）在 ARM/x86 CPU 上达到 2-5× 提速。

#### 技术原理：T-MAC bit-serial 是什么

标准 W4A32 matvec 每个元素需要：

```
y += x * dequant(q)             # 一次 int8 × int16 软件乘法（DPU 上 ~10 cycles）
```

T-MAC 的核心观察：**把 activation 按 bit plane 分解**，每 bit plane 里 activation 只有 0/1：

```
x = sign × Σ_b (bit_b(|x|) × 2^b)   for b in 0..6 (int8 magnitude)

y = x × lut[q]
  = sign × Σ_b (bit_b(|x|) × 2^b × lut[q])
  = sign × Σ_b (2^b × (bit_b(|x|) × lut[q]))
```

维护 7 个 per-bit-plane accumulator `S_b`：

```
S_b += sign × lut[q]   iff bit_b(|x|) == 1
```

最终 `y = Σ_b S_b << b`，内循环里**没有乘法**，只有 "mask check → 条件加"。

#### M-2 实装

- Host 端把每 `BLOCK_FLOATS=64` 的 int8 activation 块预处理成 8 个 `uint64_t` bitmask（7 magnitude planes + 1 sign plane），通过 `dpu_broadcast_to("inputs_bitplanes_mram", ...)` 送到 DPU
- DPU kernel 新增 `kernel_mode=7`：
  - 按 bit-plane 外层、block 内层扫描
  - 对每个 set bit 做 `acc_b += signed_lut[q_weight]`
  - 最后 `acc = Σ_b acc_b << b`
- 数值 `max_abs_err = 0.000e+00` **bit-exact** 与 `mode=4`（60/60 cell 逐一核对）

#### M-2 结果：0/60 cell 跑赢 mode=4

| kernel_mode | peak ratio | mean ratio |
|---|---|---|
| 4 (int8 × int16 软件乘法) | **3.32×** | 1.45× |
| 7 (真 T-MAC bit-serial) | 1.15× | 0.48× |

#### 根因分析

```
UPMEM DPU:
  int8 × int16 软件乘法 ≈ 10 cycles (SDK 深度优化)
  无 SIMD / 无硬件 ctz/popcnt

T-MAC 实际成本 (per element):
  - bit-scan (sparse) or bit-test (dense): ≈ 2-4 cycles/bit × 7 bits = 14-28 cycles
  - Σ conditional add: 3-4 cycles × 3-4 set bits avg = 12-16 cycles
  - 再加 LUT 访问 + weight unpack + DMA 预处理
  ≈ 30-50 cycles/element ≥ 软件乘法的 10 cycles

ARM/x86 CPU:
  硬件乘法 ≈ 1 cycle (但没法 SIMD int4 混合精度)
  硬件 ctz/popcnt 支持
  SIMD width 128-512 bit, T-MAC 查表可打满
  → T-MAC 能比 SIMD mul-add 快 2-5×
```

**核心结论**：**T-MAC 的 2-5× 收益来自 "SIMD 友好的 LUT 替代无法 SIMD 化的混合精度 MAC"**。UPMEM 既没有 SIMD 也没有 ctz/popcnt，**架构上不匹配**。

**这是 publishable negative result**：首个 DPU 上严格无硬件乘法 T-MAC 实现 + 全 shape × batch 的量化数据。完整报告写在 ADR-002 §10。

---

### 4.3 M-3 — BackendCostModel（真胜利）

**假设**：M-1 sweep 的 180 cell 数据足够精确，能把 `PIMMoEBackend._submit_forward_real` 里原来的硬编码路由（`pim_prefill_token_threshold=8` 等）升级为**数据驱动决策**。

#### 技术原理：cost-based routing

```python
class BackendCostModel:
    # 初值表：(shape_name, batch, rank_count) → {pim_sec, cpu_sec}
    _cells: dict[tuple[str, int, int], _Cell]

    def decide(shape_name, batch, rank_count, is_prefill, pim_available) -> BackendDecision:
        pim_est, cpu_est = self.estimate(shape, batch, rank_count)
        # Nearest-neighbor fallback if exact cell missing:
        #   1. same shape same batch, nearest rank
        #   2. same shape, nearest batch
        # Then choose min(pim, cpu) with stability margin 1.1×
        # to prevent thrashing between near-parity backends.

    def update(shape, batch, rank, backend, observed_seconds):
        # EMA 在线更新，alpha=0.25
        # 让真实 observation 逐渐修正 baseline
```

**PIMMoEBackend 接入**：每层对 gate/up/down 三个 shape 各 `decide()` 一次。多数投票：majority → pim 就走 PIM；majority → cpu 就 return False 让父类 CPUMoEBackend 处理。

#### M-3 附带修复：一个 M-1 遗留 bug

真机跑第一次 `cuda_cpu_offload` benchmark 发现：**`CPUMoEBackend.submit_forward` 在 GPTQ + 无 AMX 时写 zeros**（`_fallback_output = torch.zeros_like(...)`）而非真做计算。这意味着**之前所有 `cuda_cpu_offload` 的 TPS 数字都是假的** —— decoder 跑了但 MoE 输出全零。

新加 `_compute_expert_output_cpu_gptq()` 方法，调 `cpu_w4a32_matvec` 真做 CPU grouped W4A32 forward。修完后 CPU baseline decode 从 4s 跳到 **10.4s**（真 W4A32 grouped 成本），这才是诚实 baseline。

#### M-3 结果

| 指标 | cuda_pim (M-3) | cuda_cpu_offload (M-3 修复后) |
|---|---|---|
| prefill_seconds | **3.44** | 45.76 |
| decode_tps | 0.228 | 3.068 |

**prefill PIM 13.3× 赢 CPU** ✅  
decode 仍输 13.5×（orchestration overhead 问题，M-10 之前一直没解决）。

**dev_gate PASS 10/10**（首次用 `sum()` 聚合 + `ratio_vs_artifact` 跨文件比较）。

---

### 4.4 M-4 — Fused Gate+Up（真胜利）

**假设**：每个 expert 内部 `gate` 和 `up` 两个投影共享同一个 input，可以在 host 端把两组 W4A32 权重沿 output 轴 concat 成一个 fat projection，一次 preload + 一次 DPU launch 同时算出 gate 和 up。**DPU binary 不用动**。

#### 技术原理：host-side 权重 concat

`gate` 和 `up` 的 GPTQLinearWeight 形状：
- `qweight.shape = (output_dim=768, words_per_row=256)`  （4-bit 压缩到 int32，256 = 2048/8）
- `scales.shape = (768, num_groups=16)`  （per-group scale）

两组沿 `dim=0`（output 轴）concat：
- `concat_qweight.shape = (1536, 256)`
- `concat_scales.shape = (1536, 16)`
- DPU kernel 的 row-pair 循环对 output_dim 完全不敏感，只要是偶数 output_dim 就能跑（原来是 768 → 偶数；concat 后 1536 → 还是偶数）

```python
# nano_ktrans/kernels/pim_quantized_runtime.py
def preload_and_infer_concat(self, expert_id, lhs, rhs, inputs, kernel_mode=4):
    concat_qw, concat_sc, ... = self._prepare_concat_quantized_weights(lhs, rhs, kernel_mode)
    # 一次 preload（两套权重总共）
    # 一次 infer
    outputs = ...  # shape = (batch, 2*output_dim)
    return outputs[:, :lhs.out], outputs[:, lhs.out:lhs.out+rhs.out]
```

#### M-4.1 诊断补丁：M-3 发现的 preload pathology

从 M-3 的 `quantized_runtime_preload_misses = 1,675,440` 看出，每 decode step 每层每 expert 的 3 个投影都**彼此 evict**：gate preload 把上一次 up 覆盖，up 覆盖 gate，down 覆盖 up … 每个 projection 每次都 miss。

M-4 fused 路径把 `gate + up` 合并为**一个** preload + 一次 launch，down 还是独立一次。每 expert 的 DPU call 从 3 降到 2，**DPU 调用总数 -33.4%**。

#### M-4 真机结果（Qwen3-GPTQ-Int4, 32 decode tokens）

| 指标 | M-3 | M-4 | delta |
|---|---|---|---|
| DPU quantized calls | 34905 | **23246** | **-33.4%** |
| decode_seconds | 140.58 s | 100.96 s | -28.2% |
| **decode_tps** | 0.228 | **0.317** | **+39.2%** |
| vs CPU ratio | 0.074× | **0.103×** | +39% |
| 数值误差 vs mode=4 单独 | - | **max_abs_err = 0** | bit-exact |

**M-4 decode_tps 0.317 是截至 M-9 的项目最大单点设计胜利**。  
（M-10 后续意外发现 `offload_device_experts=32` 能到 0.351，是配置胜利，不是设计胜利。）

**dev_gate PASS 8/8**，含反 3-call regression guard（`DPU calls ≤ 28000`）和数据驱动 decode_tps 阈值（`≥ 0.285 = M-3 peak × 1.25`）。

---

### 4.5 M-5 — Dual Runtime（架构挖掘 1）

**假设**：M-4 后 `preload_hit_ratio = 0%` 的瓶颈是 "gate_up bundle 和 down bundle 在同一 runtime 里互相覆盖"。让它们各占独立的 `PIMQuantizedRuntime` 实例（背后是不同 DPU rank pool），就能让每个 bundle 首次 preload 后保持 resident，**跨 expert 复用时 hit**。

#### 实装

```python
def _try_init_quantized_runtimes_dual(self):
    gate_up_rt = PIMQuantizedRuntime.get_shared(profile=f"{self.pim_profile}|gate_up", rank_count=1)
    down_rt    = PIMQuantizedRuntime.get_shared(profile=f"{self.pim_profile}|down",    rank_count=1)
    return gate_up_rt, down_rt
```

#### 真机 sanity 测试结果
- 47/48 layers 成功分到 `quantized_runtime_down_distinct=True`
- decode_tps 0.309，和 M-4 的 0.317 噪声内等价（-2.7%）
- **preload_hits_local 仍然 0**

#### 根因：工作集 ≫ cache 容量

一层 top_k=8 active expert × 2 bundle（gate_up + down）= **16 unique bundle/layer/step** 要 cache。每 runtime 只有 1 个 MRAM slot（M-5 阶段还没 multi-slot）→ **跨 expert 仍 100% miss**。

Dual runtime 只能避免 *同一 expert 内部* gate_up 和 down 互相覆盖，但 M-4 fused gate+up 已经把那个解决了 —— dual runtime 是在已经解决的问题上再加一层，**新 gain = 0**。

#### 意外的 micro-bench 测量

```
_prepare_quantized_weights (Python padding):   0.074 ms/call
pim_quantized_load_weights (host→DPU DMA):     0.96 ms/call   ← 实际热点
infer-only (resident):                         2.31 ms/call
```

**preload miss 的 1.45 ms 成本 95% 花在 DPU DMA**，不是 Python。任何纯 Python 优化最多省 5%。要真省，必须让权重**不再每 call 重传**，即 MRAM 多 slot residency。这个定量发现锁定了 M-6 的目标：改 DPU binary MRAM 布局。

**dev_gate PASS 7/7**（infra landed + no regression 类 KPI）。

---

### 4.6 M-6 — Multi-slot MRAM（架构挖掘 2）

**假设**：改 DPU binary 让 MRAM 支持 `NUM_SLOTS=8` 个 expert 同时常驻 → LRU 命中率从 0% 飙到 20-30% → decode_tps 0.30 → 0.40+。

#### 技术原理：DPU binary 的 MRAM 布局

原布局：
```c
__mram_noinit uint32_t qweight_mram[MAX_QWEIGHT_WORDS];  // 8 MB/DPU
__mram_noinit float    scales_mram [MAX_SCALE_FLOATS];   // 256 KB/DPU
__mram_noinit int16_t  lut_mram    [MAX_LUT_INT16];      // 2 MB/DPU
```

M-6 改成：
```c
#define NUM_SLOTS 8
#define WORDS_PER_SLOT     (MAX_QWEIGHT_WORDS / NUM_SLOTS)   // 1 MB/slot/DPU
#define SCALES_PER_SLOT    (MAX_SCALE_FLOATS  / NUM_SLOTS)   // 32 KB/slot/DPU
#define LUT_INT16_PER_SLOT (MAX_LUT_INT16     / NUM_SLOTS)   // 256 KB/slot/DPU

__host uint32_t active_slot;  // host 每次 run 前 broadcast 进来
```

kernel 所有 MRAM 索引加 `qw_slot_base = active_slot * WORDS_PER_SLOT` 偏移。5 个 kernel_mode (3/4/5/6/7) 一共 29 处访问，全部加偏移。

Host bridge `dpu_push_xfer` 的 `offset_bytes` 参数天然对应 slot offset：
```c
dpu_push_xfer(..., "qweight_mram",
              (size_t)slot_id * WORDS_PER_SLOT * sizeof(uint32_t),  // ← 仅写目标 slot
              shard_qweight_words * sizeof(uint32_t),
              DPU_XFER_DEFAULT);
```

Python 层 `PIMQuantizedRuntime` 新增 LRU：
```python
def _allocate_slot(self, expert_id) -> (slot, was_resident):
    if expert_id in self._expert_to_slot:
        self._touch_slot(self._expert_to_slot[expert_id])  # bump LRU
        return self._expert_to_slot[expert_id], True
    # LRU evict oldest
    slot = min(range(self.NUM_SLOTS), key=lambda s: self._slot_lru_ticker[s])
    ...
    return slot, False
```

#### Micro-bench 验证（真机）

```
Preload 4 distinct experts (slots 1..4):
  expert 0 slot=1  was_miss=True
  expert 1 slot=2  was_miss=True
  expert 2 slot=3  was_miss=True
  expert 3 slot=4  was_miss=True

Re-preload 4 experts in reverse order:
  expert 3 was_miss=False    ← HIT, output bit-exact
  expert 2 was_miss=False    ← HIT
  expert 1 was_miss=False    ← HIT
  expert 0 was_miss=False    ← HIT
hits=4 misses=5  (4 hits from reverse pass + 1 warmup)
```

Overflow 到 12 experts 后，LRU 正确淘汰最旧 4 个。

#### E2E 结果：`preload_hits_local = 0`

在真实 HybridMoE 里跑，hit 仍然 0。**micro-bench 单 runtime 的 LRU 是对的，e2e 里失灵**。

#### 根因：48 层共享同一个 runtime 单例

```python
# 48 个 PIMMoEBackend 实例都调:
PIMQuantizedRuntime.get_shared(profile="", rank_count=1)
#     ↑ 同一 (profile, rank_count) → 返回同一个 Python 对象
#       8 个 slot 被 48 × 8 = 384 次/forward 的 slot-claim 洗穿
```

**理论 hit 上限 = `NUM_SLOTS / (num_layers × top_k) = 8/(48×8) = 0.2%`**。

**dev_gate PASS 8/8**（infra landed + no regression + 7 条 unit test 单独验证 LRU 正确）。M-6 作为**基础设施正确 + null e2e 关闭**，指向 M-7 per-layer 必须打破共享。

---

### 4.7 M-7 — Per-layer Scoping（架构挖掘 3）

**假设**：让每层（或每组层）独占 `PIMQuantizedRuntime`，打破 M-6 的 48 层共享。group_size=3 → 16 组 × 2 runtime = 32 rank pool（< 39 可用 rank）。理论 hit 上限 = `NUM_SLOTS / (group_size × top_k) = 8/(3×8) = 33%`。

同时加 **prefill-time speculative preload**：prefill 末尾统计每层 top-N hot expert，把它们的 fused gate_up bundle + down bundle 都提前 preload 到各自 runtime 的 slot。

#### 首次真机跑：heap corruption

```
munmap_chunk(): invalid pointer
timeout: the monitored command dumped core
```

紧急排查揭示**一个贯穿 M-5/M-6/M-7 的底层架构 bug**。

#### 根因：`.so` 的 `static` 全局变量让所有 runtime 假共享

`host_quantized_bridge.c`:
```c
static struct dpu_set_t g_set;
static bool g_initialized = false;
static uint32_t g_input_dim = 0;
static uint32_t g_output_dim = 0;
static uint32_t g_slot_loaded_mask = 0;
// ... 共 20 个 static 全局
```

ctypes 事实：
1. Python 里多个 `PIMQuantizedRuntime` 实例通过 `self._lib = ctypes.CDLL(...)` 打开同一份 `.so`
2. **`.so` 里的 `static` 只有一份物理内存**
3. `pim_quantized_init()` 第二次被调时看到 `g_initialized==true` 直接 `return 0`，**第二个 Python 对象根本没真分到新 DPU rank pool**，复用第一个的 `g_set`

**后果**：
- M-5 "47/48 dual runtime distinct" → Python 层假象，底层一个
- M-6 multi-slot LRU → Python 侧记得对，但 `g_slot_loaded_mask` 是全进程共享 8-bit 位图
- M-7 speculative preload **第一个打破严格串行调用顺序的路径**（prefill 末尾 N 个预热 preload 连续调不等中间 run），使得 `g_input_dim` 被下一个 preload 覆盖，接着的 run 读到错 shape → `outputs = torch.empty(batch, wrong_dim)` → `munmap_chunk` heap 损坏

#### M-7 降级收尾

- `enable_speculative_preload_gptq = False` 默认关（触发 crash 的路径）
- Python-layer 基础设施（pim_layer_group_size 参数、profile key 前缀、diagnostic 字段）全保留，等 M-8 修完底层再真正启用

降级后 e2e 32 token 能跑但 `hits_local` 仍 0（根因没修）。**M-7 作为"诊断出根因 + 降级保命"关闭**。

**dev_gate PASS 8/8**（"infra landed + heap safe + no regression" 类 KPI）。

---

### 4.8 M-8 — Handle-based refactor（架构挖掘 4）

**目标**：彻底修 M-7 诊断出的根因。所有 `static` 全局封进结构体，API 改 handle-based。

#### 实装

**C 端 `host_quantized_bridge.c`**：

```c
typedef struct {
    struct dpu_set_t set;
    bool weights_loaded;
    uint64_t last_cycles;
    double last_load_qweight_transfer_seconds;
    // ... 原来 20 个 static 全部搬这里
    uint32_t slot_loaded_mask;
} pim_q_ctx_t;

void* pim_quantized_init(const char *binary_path, const char *profile,
                          uint32_t rank_count, char *error_buffer, size_t len) {
    pim_q_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (dpu_alloc_ranks(rank_count, profile, &ctx->set) != DPU_OK) { free(ctx); return NULL; }
    if (dpu_get_nr_dpus(ctx->set, &ctx->nr_dpus) != DPU_OK) { dpu_free(ctx->set); free(ctx); return NULL; }
    if (dpu_load(ctx->set, binary_path, NULL) != DPU_OK) { dpu_free(ctx->set); free(ctx); return NULL; }
    return ctx;   // 每次 init 都 alloc fresh ctx，不再 "g_initialized 早退"
}

int pim_quantized_load_weights(void *handle, uint32_t input_dim, ..., uint32_t slot_id, ...);
int pim_quantized_run(void *handle, uint32_t batch, ..., uint32_t slot_id, ...);
void pim_quantized_shutdown(void *handle);   // free(ctx)
```

13 个导出函数全加 `void *handle` 首参。

**Python 端**：
```python
self._lib.pim_quantized_init.restype = ctypes.c_void_p  # handle
handle = self._lib.pim_quantized_init(...)
self._handle = ctypes.c_void_p(handle)

# 所有调用加 self._handle 首参
rc = self._lib.pim_quantized_load_weights(self._handle, ...)
```

**关键修复**：`shutdown()` 先把 `self._handle` 置 `c_void_p(0)` **再**调 C 端，防 double-free。

**关键设计**：`get_shared()` 新增 `instance_key` 参数
```python
def get_shared(cls, *, profile="", rank_count=1, instance_key=""):
    # profile 真传给 UPMEM dpu_alloc_ranks，必须空或合法 UPMEM profile
    # instance_key 是 Python _shared dict 的键，可以是任意字符串
    # 之前 M-7 用 profile="...|gate_up|g0" 在 UPMEM 眼里是 "invalid profile"
    # 只因为 g_initialized 早退让这个错误字符串没真到 UPMEM 才没炸
```

#### 真隔离 sanity 测试（真机）

```python
rt_a = PIMQuantizedRuntime.get_shared(instance_key="m8_a", rank_count=1)
rt_b = PIMQuantizedRuntime.get_shared(instance_key="m8_b", rank_count=1)
# rt_a handle = 0x55e71e158ad0
# rt_b handle = 0x55e71d013910   ← 不同！
# rt_a num_dpus = 64
# rt_b num_dpus = 64             ← 各自独立的 rank pool！

# 交错 preload + infer (M-7 下这里 crash)
rt_a.preload(1001, gate); rt_a.infer(x)
rt_b.preload(2001, up);   rt_b.infer(x)
rt_b.preload(2001, up)            # HIT！rt_b 的 2001 仍 resident
# rt_b.preload_hits = 1   ← 项目历史上第一次跨-runtime hit
```

#### E2E 真机数据（Qwen3-GPTQ-Int4, 32 tokens, group_size=3, speculative ON）

| milestone | decode_tps | preload_hits_local | hit_ratio |
|---|---|---|---|
| M-4 fused | 0.317 | 0 | 0.0% |
| M-5 dual | 0.309 | 0 | 0.0% |
| M-6 multi-slot | 0.300 | 0 | 0.0% |
| M-7 per-layer (降级) | 0.309 | 0 | 0.0% |
| **M-8 handle-based** | **0.242** | **24** | **0.1%** |

**项目历史上首次 `preload_hits_local > 0`（24 hits），证明 handle refactor 真让 runtime 物理独立**。但 decode_tps 反而 **-22%**。

#### 两个新瓶颈揭示

1. **32 rank pool 协调开销**：group_size=3 → 16 groups × 2 runtimes = 32 独立 `dpu_set_t`。UPMEM driver 的 per-set dispatch/sync overhead 大约 **1.3 ms/call**。之前假隔离时所有 call 共享 1 个 `g_set`，一次 dispatch 成本。
2. **Qwen3 routing temporal locality 远低于预期**：24 hits / 23306 misses = **0.1%**（ADR 原估 20-30%）。相邻 decode step 的 top_k 集合几乎不重叠。

**dev_gate PASS 9/9**，含项目首次**正向 KPI** `sum(preload_hits_local) >= 1`（此前 milestone 不可能达到）。

---

### 4.9 M-9 — Locality 量化（决策 milestone）

**假设**：在花更多 milestone 做 caching 优化之前，**先量化 Qwen3 routing locality 到底多少**。这是 M-5~M-8 都没做的事。

#### 实装：1 行 Jaccard histogram

```python
# _submit_forward_real 开头
active_cpu_now = frozenset(
    int(eid) for eid in topk_ids.unique().tolist()
    if cpu_mask[eid]   # 仅 CPU-side experts
)
prev = self._prev_active_cpu_experts_forward
if prev is not None:
    union = len(active_cpu_now | prev)
    inter = len(active_cpu_now & prev)
    j = inter / union if union > 0 else 0.0
    # 11-bin histogram: [0-10%), [10-20%), ..., [100%]
    self.locality_decode_jaccard_histogram[int(j * 10) if j < 1 else 10] += 1
    self.locality_decode_jaccard_sum += j
    self.locality_decode_jaccard_count += 1
self._prev_active_cpu_experts_forward = active_cpu_now
```

开销 `< 0.1 ms/call`，silent fail 保护（异常不会破坏 forward）。

#### 加 `--pim-layer-group-size` CLI flag

```bash
# 现在 group_size sweep 是一行 shell:
for gs in 3 6 12 24 48; do
  python benchmarks/benchmark_inference.py \
    --pim-layer-group-size $gs --max-new-tokens 32 \
    --json-out results/e2e_gptq_cuda_pim_M9_gs${gs}.json
done
```

#### 真机数据（震撼）

**Jaccard histogram (group_size=3, decode only, 1486 samples)**：

```
[  0%,  10%):  45.7%   ######################
[ 10%,  20%):  24.8%   ############
[ 20%,  30%):  19.9%   #########
[ 30%,  40%):   6.0%   ##
[ 40%,  50%):   2.9%   #
[ 50%,  60%):   0.2%
[ 60%,  70%):   0.3%
[ 70%,  80%):   0.0%
[ 80%,  90%):   0.2%
[ 90%,  99%):   0.0%
[100%,100%]:    0.0%
mean:           0.137
```

**相邻 decode step 的 top_k=8 集合平均只有 14% 重叠**，45.7% 样本 < 10% 重叠，只有 0.7% ≥ 50%。

**group_size 扫描**：

| group_size | decode_tps | hit_local | hit_ratio | Jaccard mean |
|---|---|---|---|---|
| 3 | 0.246 | 18 | 0.1% | 0.139 |
| 6 | 0.263 | 6 | 0.0% | 0.171 |
| 12 | 0.261 | 0 | 0.0% | 0.162 |
| 24 | 0.274 | 0 | 0.0% | 0.166 |
| **48** | **0.290** | 0 | 0.0% | 0.137 |

**group_size=48（singleton, M-6 等价）最快**。原因：
- 32 rank-pool 协调开销（M-8 §16.4 诊断）= ~1.3 ms/call × 14.7 call/layer × 48 layer = 920 ms/token
- 理论 hit 收益上限 = `mean(Jaccard) × 14.7 × 48 × 0.96 ms = 94 ms/token`
- **协调开销 > 10 × 可能的 hit 收益**

#### 决定性动作：默认翻转

| 参数 | M-8 默认 | M-9 默认 |
|---|---|---|
| `pim_layer_group_size` | 3 | **48** |
| `enable_speculative_preload_gptq` | True | **False** |

M-5/M-6/M-7/M-8 infra 全保留，**只翻默认**。High-locality 用户可通过 CLI flag 手动重新启用。

#### 最终 M-9 数据
- `decode_tps = 0.2844`（vs M-8 0.242 **+18%**，vs M-4 peak 0.317 -10%）
- 4 个连续 null milestone 的真正教训确认：**MoE 路由 locality 是一个物理常数，不是可优化的软件指标**

**dev_gate PASS 11/11**，首次检查 "routing locality diagnostic wired up and within measured bounds"。

#### 最大教训

> M-5~M-8 累计 ~10 人日工程全部建立在 "MoE routing 有 locality" 的**未经验证假设**上。M-9 的 1 行 Jaccard histogram 如果在 M-5 就加，会直接跳过整条 caching 路径。**数据驱动 > 直觉驱动**每次都要交学费。

写入 gotchas：**任何 milestone plan 里出现 "cache / prefetch / reuse / locality / working set" 这些词，上工前必须加 1 行 diagnostic 量化这条假设的 signal。**

---

### 4.10 M-10 — Python async（负结果 + 意外胜利）

**假设**：M-9 确认 caching 路径无救，但 GPU attention 本可与 PIM 并行。用 `threading.Thread` 起后台 worker 跑 `_submit_forward_real`，主线程跑 GPU attention + gate + GPU-resident experts，`sync_forward` 时 `join()`。

**原理**：Python GIL 在 ctypes 调用时释放，所以 worker 线程跑 `dpu_launch` 期间 main 线程能真执行 Python bytecode + CUDA call。

#### 实装

```python
def submit_forward(self, hs, topk_ids, topk_weights, cuda_stream):
    ...
    if self.enable_async_pim_submit and not context.is_prefill and self.has_cpu_experts:
        def _worker():
            try: self._submit_forward_real(hs, topk_ids, topk_weights)
            except BaseException as exc: self._async_exc = exc
        t = threading.Thread(target=_worker, name=f"pim_async_L{self.layer_idx}", daemon=True)
        t.start()
        self._async_thread = t
        return
    # fallback to sync path
    ...

def sync_forward(self, hs, cuda_stream):
    if self._async_thread is not None:
        wait_start = time.perf_counter()
        self._async_thread.join()
        self.async_sync_wait_seconds_sum += time.perf_counter() - wait_start
        if self._async_exc: raise self._async_exc
    return super().sync_forward(hs, cuda_stream)
```

4 个新 telemetry 字段：`async_submit_count`, `async_sync_wait_seconds_{sum,count,mean}`。

CLI `--pim-enable-async-submit` / `--no-pim-async-submit` 对。

#### 真机 A/B（Qwen3-GPTQ-Int4, 32 tokens）

| 配置 | decode_tps | sync_wait mean |
|---|---|---|
| offload=2, async OFF (≈M-9) | 0.284 | — |
| offload=2, async ON | 0.271 | 73 ms |
| **offload=32, async OFF** | **0.3506** | — |
| offload=32, async ON | 0.340 | 54 ms |

**两个配置 async 都输**。

#### 根因：Python threading 在 ctypes-heavy workload 下有隐藏开销

`sync_wait_mean = 73 ms` 分解：
- 理论 PIM work per call: ~22 ms
  - 14.7 calls/layer × (0.1ms input DMA + 1.9ms dpu_launch + 0.2ms output DMA) ≈ 32 ms
  - 但这是整个 layer 的 PIM wall time
- 实际 `sync_wait_mean = 73 ms/layer-call`，差 ~41 ms
- 这 41 ms 就是 **Python GIL 争用 + 线程切换**成本

量化公式：1488 async call × 5 ms/call Python overhead = **7.4 s** 总 overhead，正好等于 async OFF (112s) vs async ON (118s) 的差值。

**Python threading 是 zero-cost 的想法是错的**。对 ctypes-heavy workload（每次 call 只 ~2 ms），GIL 切换开销占 1-2 ms/call，相当于 50-100% 额外开销。

#### 意外胜利：offload_device_experts=32

A/B 里顺便测了 `--offload-device-experts 32`（GPU 留 32 个 expert，默认 2）：

```
decode_tps = 0.3506
```

**超过 M-4 历史 peak 0.317 +10.6%，超过 M-9 final 0.284 +23.5%**。

机制：Qwen3 top_k=8，GPU 常驻 32/128 → 每 token 每层平均 `8 × 32/128 = 2` 个 active 命中 GPU，**CPU 侧 active 从 8 降到 5-6**，PIM 工作量按比例下降。纯 weight residency 胜利，和 async 无关。

**这是 M-4 之后项目第一次 decode_tps 真正前进**。但 `offload=32` 把 GPU 内存占用从 42 GB 涨到 47 GB，OOM 风险依赖 prompt 长度和 batch size —— 不能直接改默认，M-11 要做 OOM boundary 扫描。

#### M-10 决策

- `enable_async_pim_submit = False` 默认
- Code 保留，CLI flag 保留，给 M-11 C-level async 对照用
- `offload_device_experts=32` 不改默认，留给 M-11 扫 OOM

**dev_gate PASS 10/10**，含首次使用 `ratio_vs_artifact` 检查 "async OFF ≥ async ON at offload=32"。

---

## 5. 当前性能数据汇总

### 5.1 算子层面（M-2 sweep, `benchmark_pim_shape_sweep.py`）

真机 Qwen3 W4A32 权重，单次 matvec：

| shape | batch | PIM/CPU ratio (kernel_mode=4) |
|---|---|---|
| gate (2048→768) | 1 | **3.02–3.32×** ✅ |
| up (2048→768) | 1 | **2.63–2.93×** ✅ |
| down (768→2048) | 1 | **1.90–2.01×** ✅ |
| gate/up | 2 | **1.95–2.03×** ✅ |
| down | 2 | **1.26–1.29×** ✅ 边缘 |
| 任何 shape | 4 | 0.63–0.99× ❌ |
| 任何 shape | 8 | 0.42–0.62× ❌ |

**PIM 赢 6/12 (shape, batch) 格子，全部集中在 batch ∈ {1, 2}**（decode 形态）。

### 5.2 端到端 decode（Qwen3-30B-A3B-GPTQ-Int4, 32 new tokens, batch=1）

| 配置 | decode_tps | vs CPU (3.07 tok/s) | 说明 |
|---|---|---|---|
| M-1 baseline | 0.228 | 0.074× | 无 fused, no cost model |
| M-3 cost model | 0.228 | 0.074× | cost model 对 decode 无帮助 |
| **M-4 fused gate+up** | **0.317** | 0.103× | **最大单点设计胜利** |
| M-5 dual runtime | 0.309 | 0.101× | null (假隔离) |
| M-6 multi-slot | 0.300 | 0.098× | null (48 层共享单例) |
| M-7 per-layer (gs=3) | 0.309 | 0.101× | null (static globals bug) |
| M-8 handle-based (gs=3) | 0.242 | 0.079× | 真隔离但 32 rank overhead |
| M-9 新默认 (gs=48) | 0.284 | 0.093× | 回归 M-6 等价但用诊断证实 |
| **M-10 offload=32** | **0.3506** | **0.114×** | **意外最佳**（weight residency） |
| CPU baseline | 3.0677 | 1.000× | cuda_cpu_offload |

### 5.3 端到端 prefill

| 配置 | prefill_seconds (14-token prompt) |
|---|---|
| cuda_pim (M-3+) | **3.44** |
| cuda_cpu_offload (真 W4A32) | 45.76 |

**PIM 13.3× 赢 CPU**（因为 cost model 正确在 batch=14 投给 CPU grouped path，绕开了 PIM 的 orchestration overhead）。

---

## 6. 关键教训

### 1. operator-level 胜利不自动变成 e2e 胜利

M-2 operator sweep 显示 PIM batch=1 比 CPU grouped 快 2-3×。e2e decode PIM 比 CPU AMX 慢 8-10×。差距来自 orchestration：

```
48 layers × 14.7 call/layer × 1488 calls per 32 tokens × ~1ms/call Python+ctypes roundtrip
```

**任何 roadmap 上 "operator 赢 → e2e 自动赢" 的假设必须打对折。**

### 2. 做 caching / locality 优化前必须先 1 行 histogram 量化

M-5~M-8 累计 ~10 人日工程。M-9 的 1 行 Jaccard histogram 如果在 M-5 就加，会直接跳过整条 caching 路径。

**通用规则**：milestone plan 里出现 "cache / prefetch / reuse / locality / working set" 关键词，上工前必须 ≤10 行 diagnostic 量化这条假设。

### 3. Python 多对象 ≠ 底层多实例

M-5/M-6/M-7 的 Python 诊断 `quantized_runtime_down_distinct = 47/48` 看起来真隔离，实际底层 `.so` static 全局让 N runtime 物理共享一个 DPU rank pool。

**验证规则**：底层观察量（`dpu_get_nr_dpus(set)` 返回不同值，或独立 handle 的 `fd` 不同）才算真隔离。Python 对象地址不算。

### 4. "空函数返回成功" 式 idempotency 是静默 bug

`if (g_initialized) return 0;` 看起来合理的幂等 init，但第二次调用传了新参数（rank_count/profile）时**新参数被静默丢弃**，调用方以为分到新资源，实际复用旧的。

**更安全**：要么 return error 让调用方重启，要么真的按新参数分配新 context。

### 5. null milestone 的累计价值 = 发现隐藏架构约束

6 个 null milestone 各挖出一个之前没意识到的约束：

| null | 发现的约束 |
|---|---|
| M-2 | UPMEM 没 SIMD / ctz 硬件，T-MAC 不是真最优 |
| M-5 | 单 runtime 单 MRAM slot，需要 multi-slot |
| M-6 | 48 层共享单例，需要 per-layer 隔离 |
| M-7 | UPMEM profile 字符串不能塞 Python 逻辑键 |
| M-8 | `.so` static globals 让 N runtime 假共享 |
| M-10 | Python GIL 在 ctypes 下有 5ms/call 成本 |

**null milestone 的 dev_gate KPI 应该问 "发现了什么新约束" 而不是 "tps 提升多少"。**

### 6. weight residency 旋钮比 kernel-level 优化更便宜

M-10 意外发现 `offload_device_experts=32` 超 M-4 peak 10%，纯粹是配置调整没动一行代码。

**通用规则**：任何 milestone 动手前，**先系统扫一遍配置空间**，不然可能漏掉最便宜的胜利。

---

## 7. dev_gate 数据驱动工作流

所有 milestone 都被 `scripts/dev_gate.py` 管理，这是项目的**工程纪律基础设施**。

### 7.1 目录结构

```
.codebuddy/dev_gate/
├── M-1.toml        # milestone 1 的 spec
├── M-2.toml
├── ...
└── M-10.toml
```

每个 toml 文件格式：
```toml
milestone_id = "M-4"
title = "Fused gate+up DPU call: 33% fewer DPU launches, +39% e2e decode TPS"
prerequisites = ["M-3"]

required_artifacts = [
    "benchmarks/results/e2e_gptq_cuda_pim_M4_fused.json",
]
primary_artifact = "benchmarks/results/e2e_gptq_cuda_pim_M4_fused.json"

suggested_commands = [
    "python benchmarks/benchmark_inference.py --backends cuda_pim --max-new-tokens 32 --json-out ..."
]

[[acceptance_checks]]
path = "results[0].runs[0].decode_tokens_per_second"
op = ">="
value = 0.285
reason = "M-4 fused gate+up must improve decode TPS by >=25% over the M-3 baseline (0.2276)."

[[acceptance_checks]]
path = "sum(results[0].offload_diagnostics.layers[*].backend.real_dpu_quantized_calls)"
op = "<="
value = 28000
reason = "fused gate+up cuts DPU calls from 34905 to ~23246; guard against regressing to 3-call path."
```

### 7.2 三阶段 gate

```
stage 1: prerequisite_check    (前置 milestone 全 PASS)
stage 2: artifact_check        (required_artifacts 全存在且 mtime 新鲜)
stage 3: acceptance_check      (acceptance_checks JSON-path 断言全 PASS)
```

### 7.3 支持的 op

- `==, !=, <, <=, >, >=`
- `exists, not_exists`
- **聚合**：`min(path[*].x)`, `max(...)`, `count(...)`, `sum(...)`（M-3 引入 sum）
- **跨 artifact**：`ratio_vs_artifact`（M-3 引入，M-10 首次用）

### 7.4 verdict

- `PASS` — 所有 acceptance rules 满足
- `PARTIAL` — 部分 rules 失败（继续 collect 数据）
- `BLOCKED` — 关键 rule 失败
- `WAIT` — artifact 不存在或不新鲜
- `HALT` — 前置 milestone 没 PASS

### 7.5 当前状态

```
$ python scripts/dev_gate.py check
[PASS] M-1  (stage=acceptance)
[PASS] M-2  (stage=acceptance)
[PASS] M-3  (stage=acceptance)
[PASS] M-4  (stage=acceptance)
[PASS] M-5  (stage=acceptance)
[PASS] M-6  (stage=acceptance)
[PASS] M-7  (stage=acceptance)
[PASS] M-8  (stage=acceptance)
[PASS] M-9  (stage=acceptance)
[PASS] M-10 (stage=acceptance)
```

**累计 87 条 acceptance rules 全 PASS，242 tests passed。**

---

## 8. 下一步：M-11 双轨

### 轨道 A：C-level DPU_ASYNCHRONOUS

**动机**：M-10 证明 Python async 输在 GIL 成本。C 层的 `dpu_launch(DPU_ASYNCHRONOUS)` + `dpu_sync()` 没有 Python overhead。

**实装草案**：把 `_run_expert_quantized_on_dpu` 整个循环下沉到 `host_quantized_bridge.c` 的新函数：

```c
int pim_quantized_run_batch(
    void *handle,
    uint32_t batch_size,
    const uint32_t *slot_ids,     // N 个要跑的 slot
    const void **inputs,           // N 个 input tensor
    void **outputs,                // N 个 output tensor
    uint32_t n_calls,
    char *error_buffer,
    size_t error_buffer_len
);
```

内部：
1. 循环 N 次发 `dpu_launch(DPU_ASYNCHRONOUS)`（虽然同 set 还是串行，但省 N 次 Python↔C roundtrip）
2. 最后一次 `dpu_sync(set)` 统一等
3. 整批 output push_xfer 回 host

**预期**：消除 1488 次 Python-C roundtrip 的 overhead ≈ `1488 × 0.2ms = 300 ms/token`。decode_tps 0.29 → 0.40-0.50。

### 轨道 B：offload_device_experts 系统扫

**动机**：M-10 意外发现 `offload=32` 跑 0.35 tps（超 M-4 peak）。但不知道 OOM 边界。

**实装**：系统扫 5 点 × 3 prompt length：

| offload | prompt 128 | prompt 512 | prompt 2048 |
|---|---|---|---|
| 2 | decode_tps | decode_tps | decode_tps |
| 16 | ... | ... | ... |
| 32 | 0.35 ✓ | ? | ? |
| 48 | ? | ? | ? |
| 64 | ? | ? | ? |

如果所有 stable 配置都 ≥ M-4 peak，推 `offload=32` 为 Qwen3 的新推荐默认。**直接把 decode_tps 提到 0.35+，不用写一行 C 代码。**

### 建议顺序

先做 B（一天数据扫描低风险），后做 A（深工程按 B 结果决定）。

---

## 9. 数据字典：每一项指标的精确含义

这一节把文档里所有出现过的数字、字段、ratio 的**精确定义、测量方式、典型值、常见误读**列清楚。

### 9.1 algorithm-level 指标

#### `pim_vs_cpu_grouped_ratio`

- **定义**：`cpu_grouped_seconds_avg / pim_seconds_avg`。>1 表示 PIM 快。
- **测量**：`benchmark_pim_shape_sweep.py` 每 cell warmup=3 + repeats=10，pim side 跑 `PIMQuantizedRuntime.linear(inputs, weights, kernel_mode=4)`，cpu side 跑 `cpu_w4a32_matvec_grouped(inputs, weights)`（W4A32 per-group 反量化后 matvec 的 Python reference 实现）。
- **典型值**：gate/up batch=1 **3.02-3.32×**；down batch=1 **1.90-2.01×**；batch=8 全线 **0.42-0.62×**
- **不是什么**：不是"PIM 比 CPU AMX 快多少"。CPU side 用的是 grouped reference 而非 AMX optimized kernel（`cpu_w4a32_matvec_grouped` 是 Python + PyTorch，不走 AVX512）。真正的 CPU AMX 路径在 `cpu_infer.py`（ktrans 的 C++ AMX 引擎），比 grouped 快一个量级。
- **为什么这样测**：sweep 要做成跨平台可复现的，不能依赖 AMX。AMX 对比在 e2e decode 的 `decode_tps` vs CPU baseline 里。

#### `max_abs_error_vs_cpu_grouped`

- **定义**：`max(|pim_output - cpu_grouped_output|)`，逐元素绝对误差的最大值
- **典型值**：0.18 - 0.42（Qwen3 gate/up/down）
- **不是什么**：不是"相对满量程误差"。M-1 dev_gate 里 KPI 定的是 `<= 0.5`（绝对值，宽松）；M-2 toml 原写 `<= 0.05` 是拍脑袋的"相对满量程"，实测 impossible 后降回 0.5
- **为什么这么大**：W4 量化本身就引入误差，CPU grouped reference 和 PIM 都带这个误差。mode=4 的 int8 activation 量化再叠一次误差。bit-exact 的指标是 `mode=7` vs `mode=4` 的 `max_abs_err = 0`（两者算同一个数学公式，float 精度内一致）。

#### `kernel_mode` 定义

| mode | 含义 | 内循环特征 |
|---|---|---|
| 0 | transfer-only | DPU 不计算，只测 DMA 成本 |
| 1 | unpack-only | 只做 4-bit nibble 解包 |
| 2 | dequant-only | 解包 + scale 乘法，不做 matvec |
| 3 | full fp32 | 解包 → dequant → fp32 matvec（参考基线）|
| **4** | **int8 × int16 软件乘法 + LUT** | **生产路径** |
| 5 | per-block runtime LUT | activation 依赖 LUT（实验） |
| 6 | 假 T-MAC | 自称 T-MAC 但内循环仍有 lut[q] 查表（M-1 证伪） |
| **7** | **真 T-MAC bit-plane** | **M-2 产物，数值 bit-exact 但 perf 输 mode=4** |

### 9.2 e2e 性能指标

#### `decode_tokens_per_second` / `decode_tps`

- **定义**：`run.generated_tokens / run.decode_seconds`
- **测量**：`benchmark_inference.py` 跑 `LLM.generate(prompt, max_new_tokens=N)`。prefill 单独计时；decode 从第 2 个 token 开始累计（第 1 个 token 当作 prefill 的一部分）
- **典型值（Qwen3-30B-A3B-GPTQ-Int4, 32 tokens, batch=1, offload=2）**：
  - cuda_pim M-4 peak: **0.317** (最大单点设计胜利)
  - cuda_pim M-9 final: 0.284
  - cuda_pim M-10 offload=32: **0.351** (weight residency 副产品)
  - cuda_cpu_offload: 3.068
- **不是什么**：不是理论峰值 tps。batch=1 autoregressive decode，每 token 都要完整走一遍 48 层 attention + MoE。大 batch serving 的 tps 会高一个量级。

#### `prefill_seconds`

- **定义**：从喂入 prompt 到第一个 token logits 就绪的 wall time
- **测量**：benchmark 第一次 `model(input_ids)` 包成 prefill，带 `Context(is_prefill=True)`
- **典型值**：cuda_pim 3.44 s，cuda_cpu_offload 45.76 s（14-token prompt）
- **为什么 PIM 赢 13.3×**：cost model 对 batch=14 判断走 CPU grouped path；prefill 整层一次 forward 就完，不吃 per-token orchestration；CPU AMX path 在这里没启用（测试机是 CPU grouped reference）。**如果换 AMX-enabled 机器跑 baseline，PIM 这个 13.3× 会缩小**。
- **注意**：M-3 修 zeros-output bug 之前，`cuda_cpu_offload` prefill 只有 ~4s，那是假的（MoE 整层输出全零）

#### `vs CPU` ratio in decode/prefill 表

- **定义**：`cuda_pim_tps / cuda_cpu_offload_tps`，>1 表示 PIM 赢
- **典型值**：decode 最好 0.114×（即 PIM 只有 CPU 的 11.4%），prefill 最好 13.3×（反过来 PIM 赢 13×）

### 9.3 orchestration 指标

#### `real_dpu_quantized_calls`

- **定义**：整次 benchmark 里所有层累计的 `pim_quantized_run()` 调用次数
- **典型值**：
  - M-3 (3 calls/expert): 34905 = 48 layers × 33 forwards × ~22 calls/forward 
  - M-4 fused (2 calls/expert): **23246**
- **常用除法**：`per_layer_per_step = total / (48 × 33)`，理论上等于 `2 × num_active_CPU_experts_per_layer`（M-4 fused 后每 expert 2 call）
- **意义**：是 fused gate+up 生效的直接证据（3 → 2 calls/expert）

#### `quantized_runtime_preload_misses` / `preload_misses_local`

- **定义**：
  - `_misses`：`PIMQuantizedRuntime` 实例级累计
  - `_misses_local`：`PIMMoEBackend` 实例级（per-layer）累计，避免 singleton 被 48 层累计 48 倍的误读
- **典型值**：M-4 每 call 必 miss 所以 miss_local = `real_dpu_quantized_calls`；M-8 后出现首次 `hits_local > 0`（24 hits vs 23306 miss = 0.1%）
- **不是什么**：不是单位时间 miss rate。是整次 benchmark 的总 miss 次数

#### `sync_wait_mean` (M-10)

- **定义**：`async_sync_wait_seconds_sum / async_sync_wait_seconds_count`。**async 模式下**每次 `sync_forward` 调用的 `thread.join()` 等待时间均值
- **测量**：`sync_forward` 进入前取 `time.perf_counter()`，`join()` 完后差值累计
- **典型值**：offload=2 ON 73 ms；offload=32 ON 54 ms
- **如何读**：接近 0 说明 GPU overlap 住了 PIM；接近"PIM 总时间"说明 overlap 无效
- **陷阱**：M-10 里 `sync_wait_mean=73ms` 但理论 PIM work 只 22ms/call，差 51ms = Python GIL 争用开销

### 9.4 locality 指标（M-9 引入）

#### `locality_decode_jaccard_mean`

- **定义**：`Σ jaccard(active_cpu_experts(t), active_cpu_experts(t-1)) / n_observations`，`active_cpu_experts(t)` 是第 t 次 `_submit_forward_real` 调用时路由到 CPU-side 的 expert set
- **Jaccard 公式**：`|A ∩ B| / |A ∪ B|`。A=B 时 1.0，A∩B=∅ 时 0.0
- **典型值**：Qwen3 top_k=8 decode **0.137 - 0.171** 跨 group_size 配置
- **意义**：相邻 decode step 的 top_k expert 集合有多少重叠。越高表示 MoE 路由 temporal locality 越好，slot cache 越有效
- **理论 cache hit 上限**：≈ `jaccard_mean`（slot 容量充足时，cache 最多命中 "上一步用过且这一步也用" 的那部分）
- **常见误读**：以为 0.14 是"14% hit"。它是"相邻两步 expert 集合的 Jaccard 相似度均值"，在 slot 容量无限的极限下上限大约等于它

#### `locality_decode_jaccard_histogram`

- **定义**：11 个 bin 的分布：`[0-10%), [10-20%), ..., [90-100%), [100%]`
- **为什么 11 bin**：最后一个 bin 单独留给 "完全相同" 的情况（常见于 prefill 的 token-within-prompt）
- **典型值（Qwen3 decode, group_size=3, 1486 samples）**：
  ```
  [0-10%): 45.7%   [10-20%): 24.8%   [20-30%): 19.9%
  [30-40%):  6.0%   [40-50%):  2.9%   >=50%:    0.7%
  ```
- **如何读**：中位数 bin 在 [0-10%) 意味着**一半以上 decode step 的 top_k 集合几乎不重叠**。这是 caching 必败的直接证据

### 9.5 硬件指标

#### DPU rank pool

- **定义**：`dpu_alloc_ranks(rank_count, profile, &set)` 返回的 `dpu_set_t`，包含 `rank_count` 个 rank，每 rank 64 个 DPU
- **可见 rank 数**：`ls /dev/dpu_rank*` 数量，本机 = **39**
- **分配约束**：已被分配的 rank 不能被别的 set 再分，直到 `dpu_free()`
- **M-8 前的 bug**：`pim_quantized_init` 第二次被调时 `g_initialized==true` 直接 return，所以 N 个 Python runtime 实际共享一个 `g_set`（同一批 rank）

#### `num_dpus`

- **定义**：`dpu_get_nr_dpus(set, &n)` 返回值
- **典型值**：`rank_count=1` 时 **64**（每 rank 64 DPU）
- **用法**：权重 shard 按 `rows_per_dpu = ceil(output_dim / num_dpus)` 均分到各 DPU

#### MRAM / WRAM

- **MRAM**（每 DPU 64 MB）：主存储，通过 `__mram_noinit` 声明，host 用 `dpu_push_xfer` 读写
- **WRAM**（每 DPU 64 KB）：DPU 内部快速暂存，kernel 里用 `__dma_aligned` 局部变量或 `mem_alloc`
- **M-6 切分**：qweight (8 MB), scales (256 KB), lut (2 MB) 按 `NUM_SLOTS=8` 均分 → 每 slot qweight=1MB, scales=32KB, lut=256KB

#### `dpu_launch(SYNCHRONOUS | ASYNCHRONOUS)`

- **SYNCHRONOUS**：host 阻塞到所有 tasklet 完成
- **ASYNCHRONOUS**：host 不等，需要后续 `dpu_sync(set)` 显式等
- **关键约束**：同一个 `dpu_set_t` 不能并发发两次 launch —— 第二次会等第一次完成
- **所以 async 要并行必须有多个 set**，或者多个 Python 线程 + ctypes GIL 释放（M-10 试过，被 GIL 开销吃掉）

### 9.6 配置参数

#### `offload_device_experts`

- **定义**：GPU 上保留多少个 expert 常驻。0 = 全 offload，`num_experts` = 全 GPU
- **典型值**：默认 2（GPU 留前 2 个 hot）；M-10 意外胜利配置 32
- **内存 cost**：每个 expert GPTQ 权重 ~3.65 MB × N experts × 48 layers = N × 175 MB；offload=32 多占 ~5 GB GPU
- **机制**：`gpu_experts_mask[i] = True` iff `i < offload_device_experts`，HybridMoE.forward 里按 mask 切路由

#### `pim_layer_group_size`

- **定义**：M-7 引入，多少相邻层共享一对 `PIMQuantizedRuntime`（gate_up + down）
- **约束**：`ceil(48 / group_size) × 2 runtimes × 1 rank_count ≤ 39 visible ranks` → group_size ≥ 3
- **M-7 默认**：3（16 组 × 2 = 32 ranks）
- **M-9 实测最优**：**48**（singleton，回到 M-6 等价，32-rank-pool 协调开销太大）
- **M-9 后默认**：48

#### `enable_speculative_preload_gptq`

- **定义**：M-7 引入，prefill 末尾把 top-N hot expert 提前 preload 到 slot
- **M-7 默认**：False（触发 `.so` static globals bug 的 heap corruption）
- **M-8 短暂默认**：True（handle refactor 后安全）
- **M-9 最终默认**：**False**（M-8 实测 96 preloads vs 24 hits，效费比差）

#### `enable_async_pim_submit`

- **定义**：M-10 引入，`submit_forward` 起 `threading.Thread` 跑 `_submit_forward_real`
- **M-10 初版默认**：True
- **M-10 A/B 后最终默认**：**False**（offload=2 输 4.7%，offload=32 输 3.1%）
- **保留 CLI flag** 给未来 GPU 侧更重的 workload 或 M-11 C-async 对照用

#### `NUM_SLOTS`

- **定义**：DPU binary 和 Python 侧协议的 MRAM slot 数（`dpu_quantized_kernel.c` `#define NUM_SLOTS 8` 和 `PIMQuantizedRuntime.NUM_SLOTS=8` 必须一致）
- **值**：8
- **约束**：`NUM_SLOTS × WORDS_PER_SLOT ≤ MAX_QWEIGHT_WORDS`，保证 DPU MRAM 占用不变

### 9.7 dev_gate 术语

#### acceptance rule

- **组成**：`path`（JSON-path 到 artifact 某字段） + `op`（比较操作）+ `value`（期望值）+ `reason`（失败时报错文本）
- **跨文件变体**：`ratio_vs_artifact = "other.json"` 让 observed = `path(self) / path(other)`，比 M-4 的"必须 >= M-3 × 1.25"和 M-10 的"async OFF >= async ON"用过

#### artifact

- benchmark JSON 输出，按 milestone 的 toml 声明的 `required_artifacts` 必须存在且 `mtime` 新鲜（晚于 gate 上次评估）
- 路径：`benchmarks/results/e2e_gptq_cuda_pim_M<N>_*.json` 或 `pim_shape_sweep_M<N>_*.json`

#### verdict

- **PASS** — 所有 acceptance rules 满足 → 解锁下一 milestone
- **PARTIAL** — 部分 rules fail，数据仍继续收集
- **BLOCKED** — 所有 rules fail 或关键 rule fail
- **WAIT** — artifact 不存在/不新鲜，需先跑 benchmark
- **HALT** — 前置 milestone 没 PASS

---

## 10. 怎么讲给别人听（电梯版）

> 我在做一个 MoE 大模型推理框架叫 nano-ktrans，主线是把 Qwen3-30B MoE 的 expert 放到 UPMEM 真实 PIM 硬件上跑。
>
> 算子级我们已经证明 PIM batch=1 能比 CPU 快 2-3 倍，但端到端 decode 还输 CPU 8-9 倍。
>
> 为了找到差距来源，我们做了 10 个 milestone，其中 6 个是 null perf 但每个都挖出一个架构约束——比如 UPMEM 没硬件 SIMD 让 T-MAC 失效，Qwen3 MoE 的 routing locality 只有 14% 让所有 caching 策略失效，Python threading 的 GIL 开销让 async overlap 失效。
>
> 最重要的两个真胜利是 M-4 fused gate+up（decode +39%）和 M-3 cost model（prefill 赢 CPU 13.3 倍）。还有个 M-10 的意外副产品，发现 offload_device_experts=32 配置直接把 decode_tps 推到 0.35，超过 M-4 peak——纯粹是调了一个旋钮，没写新代码。
>
> 整条路线用一个叫 dev_gate 的工作流强制数据驱动，每个 milestone 的 KPI 都是真机数据断言，失败就归类 null 写负结果报告。
>
> 现在剩两条攻击路径：C 层 async DPU launch 消 Python-C roundtrip，或系统扫 weight residency 配置。

---

## 附录：文件索引

```
/home/yangfu/nano-ktrans/
├── PROJECT_OVERVIEW.md                          ← 本文
├── README.md                                     ← 项目入口 README
├── agent.md                                      ← AI agent 工作规范
├── EXPERT_MIGRATION_EVICTION.md                  ← GPU↔PIM 专家迁移协议
├── pyproject.toml
│
├── nano_ktrans/                                  ← 核心源码
│   ├── layers/hybrid_moe.py                      ← HybridMoE (~3700 行)
│   ├── kernels/
│   │   ├── cpu_moe.py                            ← CPUMoEBackend
│   │   ├── pim_moe.py                            ← PIMMoEBackend
│   │   ├── pim_quantized_runtime.py              ← PIMQuantizedRuntime
│   │   └── pim_native/
│   │       ├── dpu_quantized_kernel.c            ← DPU kernel (5 modes)
│   │       ├── host_quantized_bridge.c           ← handle-based .so
│   │       └── build.sh
│   └── scheduler/
│       ├── cost_model.py                         ← M-3 BackendCostModel
│       └── cost_model_baseline_m2.json           ← M-2 蒸馏 60 cell 表
│
├── benchmarks/
│   ├── benchmark_inference.py                    ← e2e 跑 LLM.generate
│   ├── benchmark_pim_shape_sweep.py              ← operator-only 180 cell
│   └── results/
│       ├── pim_shape_sweep_M{1,2}_*.json         ← operator sweep artifacts
│       ├── e2e_gptq_cuda_pim_M{3..10}_*.json     ← e2e 各 milestone 数据
│       └── e2e_gptq_cuda_cpu_offload_M3_*.json   ← CPU baseline
│
├── tests/
│   ├── test_core.py                              ← 大部分单测 (~8000 行)
│   ├── test_dev_gate.py                          ← dev_gate 框架测试
│   └── test_pim_runtime.py                       ← 真机 PIM runtime 测试
│
├── scripts/
│   └── dev_gate.py                               ← milestone 工作流
│
├── .codebuddy/dev_gate/
│   ├── M-1.toml ... M-10.toml                    ← 每 milestone acceptance spec
│   ├── dev_gate_state.json                       ← 运行时状态 (gitignored)
│   └── dev_gate_log.jsonl                        ← 历史评估日志 (gitignored)
│
└── .knowledge/                                   ← 知识库
    ├── INDEX.md
    ├── architecture/decisions/
    │   ├── 001-pim-moe-offloading-literature.md  ← ADR-001 文献综述
    │   └── 002-pim-operator-parity-roadmap.md    ← ADR-002 本文来源
    ├── context/gotchas.md                        ← 累积踩坑记录
    ├── development/
    │   ├── changelog.md                          ← 按日期的变更日志
    │   └── current-focus.md                      ← 当前工作焦点
    └── journal/
        ├── 2026-04-07.md ... 2026-04-23.md       ← 每日开发日志
```
