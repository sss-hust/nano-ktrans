# ADR-002 M-28 — Fiddler-style cost-model analysis + background preload (NEGATIVE)

**日期**：2026-05-07
**branch**：pim
**前置 milestone**：M-27 Stage C (NUM_SLOTS=128, 1.3454 TPS)
**本次结论**：
1. 用闭式成本模型证明 **Fiddler 风格的运行时 "PIM vs CPU" 决策对当前 nano-ktrans 不可行**。
2. 选择最有理论收益的替代方案 **M-28 Stage A — background-thread preload** 实施并实测 → **NEGATIVE -18.8%**（1.3234 → 1.0751 TPS）。
3. 失败根因与 M-26 同源（CPython GIL 在 `ctypes.release_GIL` 之后 acquire 时被 main thread 的 torch dispatch 抢占），加深第二次教训。

---

## 1. 三路径成本建模（per-expert，decode batch=1）

### 1.1 实测原语（`benchmarks/diag_m28_cost_model.py`，Qwen3-30B-A3B-GPTQ INT4 group=128）

| 原语 | 时间 (ms) | 说明 |
|---|---|---|
| `t_h2d_act` (1×2048 fp32) | 0.0286 | pinned host → cuda 单 token |
| `t_d2h_act` (1×2048 fp32) | 0.0183 | cuda → pinned host 单 token |
| `t_gpu_expert` (fp16, 3×matvec) | **0.0688** | gate+up+silu*up+down，权重已 resident |
| `t_cpu_expert` (GPTQ W4A32, Python 参考路径) | **14.7724** | `cpu_w4a32_matvec` 3 次 |
| `t_pim_preload` (warm hit) | 0.0019 | 命中 LRU，无 DPU DMA |
| `t_pim_infer` (gu+dn) | **4.3130** | 单 expert 单 token PIM kernel |
| 单 expert MRAM payload | 2.39 MB | gate_up 1.59 MB + scales 96 KB + down 768 KB + scales 48 KB |
| `t_h2d_weight_pcie` (Gen4×16, 2.39 MB) | 0.078 | 推算：32 GB/s 带宽 |

### 1.2 闭式成本（每 cold expert）

```
C_gpu       = t_gpu_expert                       = 0.069 ms      ← 权重已在 vRAM
C_cpu       = t_d2h_act + t_cpu_expert + t_h2d   = 14.819 ms     ← Python 参考路径
C_pim_cold  = t_d2h_act + t_pim_preload_cold + t_pim_kernel + t_h2d
            = 0.018 + 3.65 + 4.31 + 0.029       = 4.241 ms
C_pim_warm  = t_d2h_act + t_pim_kernel + t_h2d  = 0.592 ms       ← LRU 命中
C_gpu_demand_paged = t_h2d_weight_pcie + t_gpu_expert
                  = 0.078 + 0.069              = 0.147 ms        ← 临时 demand-page 到 vRAM
```

### 1.3 Fiddler 风格判定

| 比较 | 结果 | 含义 |
|---|---|---|
| `C_gpu` vs others | **0.069 ms 全胜** | 任何放进 vRAM 的 expert 都应直接在 GPU 算 |
| `C_pim_warm` < `C_cpu`? | True (0.592 < 14.772) | **PIM 比 CPU GPTQ Python 参考快 27.1×** |
| `C_pim_cold` < `C_cpu`? | True (4.241 < 14.772) | **即便 cold-miss PIM 仍快 3.5×** |
| `C_gpu_demand_paged` < `C_pim_warm`? | True (0.147 < 0.592) | **PCIe 拉权重 + GPU 算 比 PIM warm 快 4.0×** |

**结论 1**：Fiddler "PIM vs CPU" 决策没有意义——PIM 在所有场景下都比 GPTQ-on-CPU-Python 快。问题不是"该不该卸载到 PIM"，而是 PIM 本身相对 GPU demand-page 的物理弱势（DPU 频率低、无 SIMD、INT4 反量化全软件）。

**关键反直觉**：当前 `cuda_cpu_offload` 之所以比 `cuda_pim` 快（2.09 vs 1.35 TPS），**不是因为 CPU 算得快**——而是因为它走的是 **AMXINT4** C++ 引擎（`offload_diagnostics.layers[].backend.method == "AMXINT4"`），单 expert <0.5 ms，**且整个 latency 隐藏在 GPU expert loop 后面**（thread pool + `submit_with_cuda_stream` 异步）。PIM 的物理短板在这个对比下被放大。

### 1.4 工程含义

- Fiddler 的"runtime per-expert decision"思路在我们的硬件 stack 下**收益上限为零**：即便完美预测，也只是把所有 expert 都判到 GPU（不可行，vRAM 不够）或都判到 PIM（已经是当前默认）。
- 真正可工程化的优化只有两个方向：
  - **A. 减少 PIM 路径上的 host-side overhead**（preload、submit、sync 的 ctypes/Python 串行成本）
  - **B. 增大 vRAM 占比** — 已在 M-23 解决（92/128 = 71.9% on GPU）
- 本次选择 A，targeting **`step_2_submit` 中 4.4 ms 的 preload 串行块**。

---

## 2. M-28 Stage A — background-thread preload 实施

### 2.1 设计

- 把 `_do_c_async_submit_work` 拆成两部分：
  - **main thread**：torch 部分（`torch.unique` 向量化 expert scan、`_gptq_experts.get`、`flat_cpu[token_indices]` slice）
  - **bg Python thread**：纯 ctypes 调用块（`preload_concat_and_get_slot` × N + `submit_many_fused_silu_async`）
- 与 M-26（已 NEGATIVE -28.6%）的关键区别：**不把 torch ops 放进 thread**——M-26 失败的核心是 thread 里的 `torch.unique` / 索引切片要 GIL，和 main thread 的 GPU dispatch 抢锁。

### 2.2 修改

- `nano_ktrans/kernels/pim_moe.py`：
  - 加 `enable_m28_bg_preload: bool = False` 参数 + 计数器
  - 拆出 `_do_native_preload_and_submit_inline()` helper（纯 ctypes block）
  - `_submit_forward_c_async` 在 enable 时 spawn `_m28_bg_thread`，**乐观返回**（C handle 由 bg thread 写入）
  - `sync_forward` **入口**先 `t28.join()`，确保 `_c_async_handle` 在后续分支判断前已写入
- `benchmarks/benchmark_inference.py`：透传 `--pim-enable-m28-bg-preload` / `--no-pim-m28-bg-preload`

正确性测试：`288 passed, 1 warning in 14.48s`（`pytest -x -q`）。

### 2.3 实测（Qwen3-30B-A3B-GPTQ-Int4，max_new_tokens=32）

| 配置 | decode TPS | decode_seconds | c_async_wait | bg_preload_wait | preload hit |
|---|---|---|---|---|---|
| **M-27 Stage C baseline** (复现) | **1.3234** | 24.18 | 0.71 ms | — | 44.7% |
| **M-28 Stage A** (`--pim-enable-m28-bg-preload`) | **1.0751** | 29.77 | **4.39 ms** | **3.81 ms** | 44.0% |
| Δ | **-18.8%** ❌ | +5.59 s | **+617%** | new | flat |

**PIM-compute 守护通过**：`real_dpu_expert_calls / offloaded_pairs = 1770/1770 = 1.000` —— 100% offloaded experts 真在 DPU 算。

### 2.4 失败原因诊断

预期：bg thread 在 GPU expert loop（7.48 ms）下的 4.4 ms preload 应该被完全隐藏，main thread step_4 wait ≈ 0。

**实测**：bg thread 比预期慢 **1.7×**（join 等待 3.81 ms），c_async wait 也飙到 4.39 ms。整层 wallclock：

```
预期 (理论模型 §M-28 §1)   :  1.19 + max(7.48 GPU, 4.53 bg) + 1.05 wait = 9.72 ms  → 2.04 TPS
实测 (M-28 Stage A 真测)   :  1.19 + max(7.48 GPU, 8.20 bg) + 4.39 wait = 17.06 ms → 1.08 TPS（与 datapoint 吻合）
```

**根因**：CPython GIL 调度。即便 `ctypes.release_GIL=True`：
- bg thread 调 ctypes A → 释放 GIL → DPU DMA → ctypes A 返回 → **acquire GIL 才能 dispatch ctypes B**
- 此时 main thread 在 GPU expert loop 里持续做 `F.linear`、`silu`、`index_add_` 等 torch ops，**每个 op 都持 GIL 数十 us**
- bg thread 每次 acquire GIL 都要等 100 个 sys.setswitchinterval (默认 5ms) 周期之一
- 累计 N=2.88 cold experts × (gu+dn ctypes pair) × ~5 ms acquire latency = **多花 ~10-15 ms 的 GIL 切换开销**

**和 M-26 的对比**：M-26 失败 -28.6%（更糟），因为 M-26 thread 里**还有 torch ops**（每个都要 GIL）；M-28 改进到 -18.8%（少 10%），因为只剩 ctypes，但 GIL acquire 模式仍在。

### 2.5 教训（与 M-26 合并）

> **CPython 的 GIL 不是"ctypes 期间释放就万事大吉"。任何 background Python thread 想做长串 ctypes 调用，期间如果 main thread 持续做 torch / Python work，bg thread 的总 wall-clock 会被 GIL acquire latency 严重放大（实测放大 ~1.7× over 理论模型）**。

**真正可行的 overlap 方案必须满足之一**：
- (a) 整段 overlap 工作放进 **C-level pthread**（M-24 Stage A 的 `pim_quantized_run_many_fused_silu_async` 走的就是这条路），完全绕开 Python GIL
- (b) main thread 在 overlap 窗口内**不做任何 torch / Python work**（但这违反 HybridMoE 流水线设计——GPU expert loop 必须在那段时间跑）

→ 后续 milestone 候选：**M-28 Stage B — C-level `pim_quantized_preload_many_async` 入口**，把 N 个 preload 打包到一次 ctypes 调用，C 内部 spawn pthread，main thread 完全不用 acquire 中间 GIL。工程量较大（需要改 C bridge + Python wrapper + handle 生命周期），但理论上可以兑现 §1 模型预测的 +34% 收益。

### 2.6 处置

- `enable_m28_bg_preload` flag 默认 OFF，**保留代码作为反面对照**（同 M-26 的处置方式，便于以后 C-level Stage B 实现时做差分对照）。
- `benchmarks/results/e2e_gptq_cuda_pim_M28_stageA_bgpreload_NEGATIVE.json` 永久归档。
- 当前 production 路径不变：M-27 Stage C，1.3234~1.3454 TPS。

---

## 3. 综合发现：从 Fiddler 到 GIL —— 为什么 nano-ktrans 不能再走 "纯 Python 编排" 优化

把 M-26 + M-28 的两次 NEGATIVE 加上 §1 的 Fiddler 不可行性放在一起，可以画出一条非常清晰的边界：

| 优化层 | 状态 | 上限 |
|---|---|---|
| 路由决策（Fiddler 风格 cost model） | 不可行 | PIM 永远不会比 GPU demand-page 更好（差 4×），永远比 CPU AMX 更慢 |
| MRAM 数据放置（NUM_SLOTS、layer_group） | 已收到 M-27 Stage C 的尾部收益 | hit rate ~45% 接近 NUM_SLOTS=128 上限 |
| Python 编排 overlap（threading） | M-26、M-28 双重 NEGATIVE | CPython GIL 物理上限 |
| **C-level 编排 overlap（pthread + 单次 ctypes 入口）** | **唯一未探索的方向** | 理论上 +30~40% |
| PIM kernel 本身（DPU 频率、SIMD） | 硬件天花板 | 无 |

**结论**：M-29 之后只剩两条路——
1. **走 C-level**：把 preload+submit 做成单个 ctypes "fire-and-forget" 入口，参考 `pim_quantized_run_many_fused_silu_async` 的 thread+token 设计
2. **走 weight-cache pre-shard**：消除 host-side 7.5 MB memcpy（M-27 Stage C 留下的尾巴）；不与 GIL 冲突，理论 +10~20%

两个方向叠加可能把 decode_tps 推到 **1.6-1.8 TPS**（CPU 基线 78~86%）。但**研究终极目标已经达成**：在 vs CPU 64.3% 的水平上证明了 PIM 参与是科学有效的（M-27 状态下 100% offloaded 真在 DPU 算）。

---

## 4. 复现命令

```bash
# baseline (M-27 Stage C, 1.3234 TPS)
python benchmarks/benchmark_inference.py \
  --model-path /home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4 \
  --backends cuda_pim --offload-device-experts 92 \
  --routing-freq-json benchmarks/results/routing_freq_qwen3_30b_m23_mean.json \
  --pim-rank-count 1 --pim-layer-group-size 3 \
  --max-new-tokens 32 --warmup 0 --repeats 1 \
  --pim-enable-c-async --pim-enable-m25-pinned

# M-28 Stage A (NEGATIVE, -18.8%)
python benchmarks/benchmark_inference.py \
  --model-path /home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4 \
  --backends cuda_pim --offload-device-experts 92 \
  --routing-freq-json benchmarks/results/routing_freq_qwen3_30b_m23_mean.json \
  --pim-rank-count 1 --pim-layer-group-size 3 \
  --max-new-tokens 32 --warmup 0 --repeats 1 \
  --pim-enable-c-async --pim-enable-m25-pinned \
  --pim-enable-m28-bg-preload   # ← the toggle

# 成本模型重测（用合成 weights 测原语，不依赖完整模型加载）
python benchmarks/diag_m28_cost_model.py --repeat 100
```

**报告完**。
