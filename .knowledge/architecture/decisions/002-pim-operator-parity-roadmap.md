---
id: ADR-002
title: PIM 算子级超越 CPU 的优化路线与科研计划
status: proposed
created: 2026-04-22
updated: 2026-04-22
tags: [architecture, pim, quantized, t-mac, roadmap, research-plan]
related: [ADR-001]
---

# ADR-002: PIM 算子级超越 CPU 的优化路线与科研计划

## 1. 目标

**核心 KPI**：在 `Qwen/Qwen3-30B-A3B-GPTQ-Int4` 的 MoE expert linear（`gate/up/down`）上，
在 **full shape × {batch=1, 4, 8} × {1/4/8/16/32 ranks}** 的组合里，PIM 稳定超过 CPU
grouped baseline **≥ 1.5×**，且误差 `max_abs_error ≤ 0.05` 相对满量程。

**次级 KPI（验证路线有效性）**：
- `kernel_mode=6`（T-MAC）在 `batch=1` 下比 `kernel_mode=4` 再快 **≥ 30%**
- `batch=8` 时 PIM/CPU ≥ 0.9×（不再出现 `0.56x` 悬崖）
- 端到端 `cuda_pim` e2e decode TPS ≥ `cuda_cpu_offload` 的 1.0×

上述数字均以真实 UPMEM 硬件（`/dev/dpu_rank*`）为准，simulator 结果不作为验收。

---

## 2. 当前状态快照（截至 2026-04-22）

### 2.1 已经做对的部分

| 能力 | 状态 | 证据 |
|------|------|------|
| 权重常驻（fp32 path） | ✅ | `host_expert_bridge.c::preload/infer/evict` |
| 权重常驻（quantized path） | ✅ | `pim_quantized_runtime.py::preload/infer/evict` + `weight_cache` |
| 分项 profiling（transfer/unpack/dequant/full） | ✅ | `dpu_quantized_kernel.c::kernel_mode=0..4` |
| `kernel_mode=4` int8 定点 batch=1 | ✅ | `down` 2.54×、`gate` 1.30× CPU grouped |
| Quantized PIM 接入主 MoE 链路 | ✅ | `pim_moe.py::_run_expert_quantized_on_dpu` |
| GPU↔PIM demotion eviction 钩子 | ✅ | `notify_expert_evicted()` |
| 8 级 staged commit pipeline | ✅ | `HybridMoE` |

### 2.2 关键差距（按严重性排序）

#### Gap A — `kernel_mode=6` 不是真正的 T-MAC，没有消除 DPU 软件乘法

**证据**：[`dpu_quantized_kernel.c` 166-240 行]

```c
if (kernel_mode == 6) {
    ...
    const uint32_t abs_x = ...;
    /* Unrolled bit scan for 7-bit magnitude */
    if (abs_x & 0x01u) { partial0 += w0; ... }
    if (abs_x & 0x02u) { partial0 += w0 << 1; ... }
    ...
    if (abs_x & 0x40u) { partial0 += w0 << 6; ... }
    acc0 += sign * partial0;
}
```

这是 **activation 侧的朴素 shift-add 乘法器模拟**，内循环 per element 要做
7 次条件分支 + 7 次 shift + 7 次 add，比 `kernel_mode=4` 的一次 `int8 × int16`
软件乘法很可能**更慢**（DPU 软件乘法 ~10 cycles，7 次分支 + shift ≈ 10-15 cycles，
且增加 WRAM spill 风险）。

**真正的 T-MAC 核心思想**（Wang et al., MLSys'25）：
```
预计算：对每组 g 个权重位，枚举所有 2^g 种 activation 组合 → T[j][a_bits]
推理：acc += T[j][pack(x[j*g..(j+1)*g])]          // 纯查表 + 加法，无乘法
```

**本项目当前实现偏离了这一核心**。修复后预计对 DPU 友好度陡升（DPU 天然适合查表）。

#### Gap B — LUT 布局与 WRAM 占用未优化，`batch>1` 读取放大

当前 `kernel_mode=4/6` 的 LUT 是 `[output_row, group, 16 entries × int16]`，每次都按
`(row, group)` 索引从 MRAM 读 32 Bytes。16 tasklets 并行时每组 LUT 读取命中率低。

**问题**：
- batch=B 时 activation 读取放大 B 倍，但 LUT 没有按 batch 复用
- LUT 是 int16，而 activations 被 host 量化为 int8；T-MAC 可以允许 LUT 变 int8 进一步压缩

#### Gap C — 前台/后台 overlap 未真正形成

**证据**：`benchmarks/results/e2e_base_pim_2026-04-20.json` 里
`runtime_evictions=0, runtime_deferred_for_prefetch=0`，说明 scheduler profile
未触发实质迁移；`profile_sweep_summary.comparison_table` 为空。

端到端路径下，**PIM 的计算并没有与 GPU/CPU 真正 overlap**，即使算子本身变快，
总延迟也被串行依赖吃掉。

#### Gap D — e2e GPTQ benchmark 从未跑通

`e2e_gptq_cpu_vs_pim_2026-04-20.json` 两个 run 都 abort 在：

```
Weight key 'model.layers.0.mlp.experts.0.gate_up_proj.weight' not found
```

Qwen3 checkpoint 真实存的是 unpacked `gate_proj/up_proj/down_proj`，但 e2e benchmark
入口仍在用 packed `gate_up_proj` key。**所有关于"PIM 打赢 CPU"的量化证据
目前只来自 operator-only micro benchmark，端到端尚未验证过一次**。

#### Gap E — 未建立 cost model，backend 选择仍是硬阈值

`pim_prefill_token_threshold=8` 硬编码，既不跟 rank_count 联动，也不跟 shape 联动。
大量 `gate batch=4/8` 场景实际应该走 CPU，但现在仍会误路由到 PIM。

---

## 3. 瓶颈定量分析（基于 `2026-04-19` 真机 breakdown）

以真实 `Qwen/Qwen3-30B-A3B-GPTQ-Int4` 的 `gate` projection `(input=2048, output=768, group=128)` 为例：

| kernel_mode | 阶段 | 耗时占比 | 说明 |
|-------------|------|---------|------|
| 0 (transfer_only) | host→DPU input / DPU→host output | ~3% / ~2% | 完全不瓶颈 |
| 1 (unpack_only) | 4-bit nibble 解包 | +15% | 偏内存访问 |
| 2 (dequant_only) | 反量化 = unpack + scale 乘法 | +25% | 软件乘法开始显著 |
| 3 (full soft-float) | full matvec | 100% baseline | bottleneck = 软件乘法 |
| 4 (int8 fixed-point) | int8×int16 软件乘法 | ~50% of mode=3 | batch=1 已超 CPU |
| 6 (current "T-MAC") | 7 轮条件 shift-add | ~实测 ≈ mode=4 | 没有明显改善 |

**结论**：DPU 软件乘法依然是核心瓶颈（UPMEM DPU 每条 8-bit×8-bit multiply 软件实现
是 ~10 cycles，32×32 是 ~40 cycles）。**消除乘法**是唯一真正能拉开差距的路径。

---

## 4. 技术路线（4 个里程碑）

### M-1：修复阻塞性 bug + 建立正确的测量基线（1–2 天）

**目标**：让 e2e benchmark 能跑起来，并在同一 shape 下拿到可信的 CPU/PIM 对照数据。

- [ ] **M1-T1**：修复 e2e `gate_up_proj` key 适配问题
  - 让 `benchmark_inference.py` 在 `cuda_pim` backend 下使用 checkpoint 自适应 layout（已在 config.py 做过一次，e2e 路径未接入）
  - 验收：`e2e_gptq_cpu_vs_pim_2026-04-22.json` 有真实数字，而非 error string

- [ ] **M1-T2**：将 operator-only benchmark 扩到**全 shape × batch × rank** sweep
  - 覆盖 Qwen3 的 3 种 shape：`gate (2048→768)`、`up (2048→768)`、`down (768→2048)`
  - 每个 shape × `batch ∈ {1, 2, 4, 8}` × `rank ∈ {1, 4, 8, 16, 32}` × `kernel_mode ∈ {3, 4, 6}`
  - 记录 `seconds_{avg,min,max}`、`launch_seconds`、`transfer_seconds`、`max_abs_error`
  - 产物：`benchmarks/results/pim_shape_batch_rank_sweep_2026-04-22.json`
  - 这个 sweep 是所有后续决策的**唯一数据来源**

- [ ] **M1-T3**：把 `kernel_mode=6` 的"真 T-MAC vs 假 T-MAC"用数据验证
  - 不做任何代码改动，只跑 sweep，确认当前 mode=6 确实 ≤ mode=4
  - 作为后续重写的 baseline

**退出标准**：所有 sweep 结果归档、真实 e2e 数字归档。

---

### M-2：真正的 T-MAC bit-serial kernel（kernel_mode=7，1 周）

**目标**：把 DPU 软件乘法彻底从内循环拿掉。

#### 4.2.1 算法设计

按 T-MAC 论文，针对 Qwen3 GPTQ W4A32 做如下选择：

- **activation 量化**：承袭 `kernel_mode=4` 的 per-batch int8 activation + per-tensor scale
- **weight 视角的 bit-slicing**：4-bit weight 拆成 4 个 1-bit 平面
- **group size `g`**：选 `g=4`（4 个 weight bit 一组 → 2^4=16 条表项，WRAM 友好）
- **LUT 数值类型**：int16（保留 activation × 1-bit-weight 累加的精度），每组 16 × 2B = 32B
- **预计算**（host 端一次）：对每对 (output_row, weight_group)，枚举 16 种权重组合，
  计算 `Σ_{j=0}^{g-1} a_j · (q_j >> bit) · 2^j` 并存 `T_bit[row][group][16]`
- **DPU 内循环**：
  ```c
  for (bit = 0; bit < 4; bit++) {
      for (group = 0; group < num_groups; group++) {
          idx = pack_activation_bits(x[group*g..(group+1)*g]);  // 索引
          acc += T_bit[bit][row][group][idx] << bit;            // 纯查表
      }
  }
  ```

**相对当前实现的差异**：`T_bit` 的索引是 **activation 的量化 bit pack**，不是 weight；
这样 weight 被彻底编码进了表，DPU 内循环只剩表查找 + 整数加 + 移位。

#### 4.2.2 代码改动

- `pim_quantized_runtime.py`：host 端 `_build_tmac_luts(qweight, scales, zero_points)`
  生成 `(num_bit_planes, output_dim, num_groups, 2^g)` int16 张量，preload 到 MRAM
- `dpu_quantized_kernel.c`：新增 `kernel_mode=7`
  - 取消 per-element 分支
  - 每个 output_row tasklet 只做 `mram_read(LUT_slice) + acc += LUT[idx] << bit`
- `host_quantized_bridge.c`：增加 `g_kernel_mode == 7` 的 LUT 布局感知分发

#### 4.2.3 风险与对策

| 风险 | 缓解 |
|------|------|
| LUT 大小爆炸（`output_dim × num_groups × 16 × 2B`） | 典型 shape 下：768 × 16 × 32B = 393 KB / DPU-rank；Qwen3 MRAM 64MB/DPU，绰绰有余 |
| activation bit-pack 本身也是个软件操作 | 做 per-group 一次 pack 后缓存，摊到 `output_dim` 行上可忽略 |
| int16 LUT 在累加时可能溢出（大 `num_groups`） | 用 int32 accumulator + int16 LUT，数学上最大 `2^15 × 512 = 16M < 2^31` 安全 |
| 与现有 kernel_mode=4 并行维护导致代码分叉 | kernel_mode=7 完全并行独立，直到 sweep 验证其稳定可超 mode=4 再切默认 |

#### 4.2.4 验收标准

- `batch=1` 下 `kernel_mode=7` ≥ `kernel_mode=4` × 1.3（DPU 不再瓶颈在软件乘法）
- `batch=4/8` 下 `kernel_mode=7` 不出现 < CPU_grouped × 0.9 的悬崖（LUT 复用生效）
- `max_abs_error < 0.05`（相对满量程）

**退出标准**：`kernel_mode=7` 在 M1-T2 的全 sweep 下至少有 70% 配置打过 CPU grouped ≥ 1.2×。

---

### M-3：Cost-Model 驱动的自适应分发 + 真正的后台 overlap（1 周）

**目标**：即使 `kernel_mode=7` 在某些 shape 下仍输给 CPU，系统也能自动路由到正确的 backend；
并让 PIM 的计算与 GPU/CPU 真正 overlap，不被串行依赖吃掉收益。

#### 4.3.1 BackendCostModel（落地 ADR-001 的 P3 + P6）

- 新增 `nano_ktrans/scheduler/cost_model.py`
- 每个 backend（CPU / GPU-resident / PIM）维护一个 `latency_per_token(shape, batch)`
  的滑动平均表
- 首次调用时按 M1-T2 sweep 结果做初值；runtime 每次 `submit_forward` 结束后更新 EMA
- `HybridMoE._select_backend_for_experts()` 用 cost model 查询，选最小 predicted latency
- 退化路径：cost model 未收敛时回落到现在的 `pim_prefill_token_threshold` 规则

#### 4.3.2 Throttle-Aware（ADR-001 P6）

- `PIMLinearRuntime.last_profile()` 已输出 `runtime_dpu_count` / `launch_seconds`
- 在 cost model 里增加 `dpu_pressure = active_dpu_count / total_dpu_count`
- 当 `dpu_pressure > 0.8` 时，把 PIM latency 预测上调 1.5× 作为 throttle 惩罚

#### 4.3.3 真正的后台 PIM 提交

现在 `HybridMoE.submit_forward` 是同步调用。改造：
- `PIMMoEBackend.submit_forward_async(expert_idx, inputs)` 只把请求入队，返回 future
- 背景 worker（已有）负责 dispatch 到 `PIMQuantizedRuntime.infer`
- GPU 端的 attention/dense forward 与 PIM 专家 forward **完全并行**
- 新增 `pim_submit_queue_depth` 诊断观察排队

#### 4.3.4 验收标准

- 端到端 `cuda_pim` decode TPS ≥ `cuda_cpu_offload` × 1.0（目前落后）
- cost model 在 sweep 数据上预测误差 < 20%
- decode overlap 可见：`pipeline_promotion_source_activated_ratio > 60%`

---

### M-4：进一步的算法级优化（研究探索，2–3 周）

只有 M-1 ~ M-3 完成、数据稳定后才启动。这是论文级的增量。

#### 4.4.1 Mixed-Precision Expert（ADR-001 P4 / HOBBIT）

- 高 gate-score 专家用 `kernel_mode=3`（soft-float，精度高）
- 低 gate-score 专家用 `kernel_mode=7`（T-MAC，速度高）
- 每专家保留 `(int4, int8_scale)` 双版本

#### 4.4.2 Sub-Batch Interleaving（ADR-001 P5 / NeuPIMs）

- 依赖 batch>1 支持（目前单序列）
- 把单 decode step 的 batch 拆成 A/B
- A: GPU attention + gate → B: PIM experts → roll forward

#### 4.4.3 LUT 压缩与共享

- 观察：同层内大量专家的 `scales` 分布相似
- 可以做 **k-means 聚类的 shared LUT**，每专家只存一个 cluster_id
- 预期对 MRAM 占用下降 ~50%

---

## 5. 验证方法论

### 5.1 三级基准

```
  Level 1: operator-only micro benchmark
           (benchmark_quant_matvec.py, 每次 runtime 变更必跑)
           ├── 产出: seconds_avg, max_abs_error, launch/transfer breakdown
           └── 验收: 用于判定 kernel 算法变更是否真的消除了瓶颈

  Level 2: profile-aware single-shape sweep
           (新增 benchmark_pim_shape_sweep.py)
           ├── 产出: {shape × batch × rank × kernel_mode} 网格
           └── 验收: 用于选择每 shape 的最优 kernel_mode + rank

  Level 3: end-to-end decode
           (benchmark_inference.py --backend cuda_pim)
           ├── 产出: prefill/decode TPS, cold_promotion_ratio, prefetch_hit_rate
           └── 验收: 唯一判定路线成败的指标
```

### 5.2 每个里程碑的验收清单

每个 milestone 必须产出一份 `benchmarks/results/<milestone>_<date>.json`，
并在 journal 里写一份"验收报告"，包含：

1. baseline 数字（重跑 M1-T2 sweep 的对应子集）
2. 新代码下的数字
3. 误差分析（相对 CPU grouped 的 `max_abs_error`）
4. 若未达验收标准，列出已观测到的次要瓶颈

### 5.3 不允许的"胜利"

- ❌ simulator 结果
- ❌ 只跑一次的数字（至少 warmup=3 + repeats=10）
- ❌ 只报告 best-case shape 不报告 worst-case shape
- ❌ 只报告 batch=1 不报告 batch=4/8

---

## 6. 预期产出（科研向）

- **工程产出**：
  - `kernel_mode=7` 真正的 T-MAC DPU kernel（首个 DPU 上的 T-MAC 实现）
  - `BackendCostModel` + 自适应 PIM/CPU/GPU 分发
  - 全 shape × batch × rank × kernel 的公开 benchmark 数据

- **论文素材**：
  - "T-MAC on UPMEM DPU: Making MoE Expert Offloading Practical"
    - Novelty: 首个 DPU 上严格无硬件乘法的 quantized GEMV
    - Evaluation: Qwen3-30B-A3B-GPTQ-Int4, vs CPU grouped baseline
    - Key result: `kernel_mode=7` vs `kernel_mode=4` 的跨 shape × batch 对照
  - 或增量投 **MLSys workshop / Systems for ML**：
    - "PIM + Hybrid MoE: When Does UPMEM Help?"
    - 用 cost model 的收敛曲线做"PIM 何时可用"的定量判据

---

## 7. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 真 T-MAC 在 UPMEM 上 LUT 访问延迟过高（MRAM 未缓存） | 中 | 高 | 准备 per-tasklet WRAM LUT 预取版本（增加 WRAM 占用换带宽） |
| Qwen3 group_size=128 导致 LUT 过宽 | 中 | 中 | 二次分组：每 4 个 1-bit slice 再打包 |
| `dpu.driver` 不稳定导致 M-3 的 async 分发失败 | 低 | 中 | 保留 C bridge 的 sync 回退 |
| 宿主机 GPU 同时只有 1 个，M-3 验证需要排期 | 中 | 低 | M-1/M-2 可在 CPU 环境上完成算子层 |

---

## 8. 与 ADR-001 的关系

| ADR-001 编号 | ADR-002 位置 |
|-------------|-------------|
| P1 (MRS) | 已落地（2026-04-21） |
| P2 (Expert Map Store) | 已落地（2026-04-21） |
| P3 (Fiddler cost model) | M-3 的 BackendCostModel |
| P4 (HOBBIT mixed precision) | M-4.1 |
| P5 (Sub-batch interleaving) | M-4.2 |
| P6 (PIMoE throttle) | M-3.2 |

ADR-001 关注**系统级策略**；ADR-002 关注**算子层能效**。两者正交，合起来构成
"算子赢 + 调度正确"的完整叙事。

---

## 9. 决策

采用上述 M-1 ~ M-4 四阶段路线。**M-1 与 M-2 是必须完成项**（保证 KPI），
M-3 是端到端能否看到收益的关键，M-4 是科研论文素材。

下一步立即开始 **M-1**。
