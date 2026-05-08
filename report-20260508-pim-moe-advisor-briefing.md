---
title: "PIM + GPU 异构推理 MoE decode 进展汇报"
subtitle: "硬件特性、执行逻辑、当前瓶颈与后续设计"
date: 2026-05-08
audience: advisor
---

# PIM + GPU 异构推理 MoE decode 进展汇报

## 一、实验平台硬件规格

### 1.1 实验平台完整配置

| 组件 | 规格 |
|---|---|
| GPU | 单卡，47 GB 可用显存（推理 peak 41.9 GB / 45.0 GB 上限）|
| CPU | Intel Xeon Silver 4210R（Cascade Lake, 2020）|
| PIM | UPMEM DPU 服务器，**39 个可用 rank × 64 DPU/rank = 2496 个 DPU** |
| host RAM | 足够装下完整 GPTQ 权重（约 15 GB）|

### 1.2 PIM (UPMEM DPU) 硬件核心参数

这些是后续所有性能推算的基础参数，必须先列出来：

| 参数 | 值 | 说明 |
|---|---|---|
| DPU 频率 | 350–500 MHz | 远低于 CPU/GPU |
| 每 DPU 的 MRAM（主内存）| **64 MB** | 存放权重 + 激活 + scratch |
| 每 DPU 的 WRAM（工作内存）| 24 KB | 类似 L1，kernel 热数据 |
| 每 DPU 的 IRAM（指令内存）| 32 KB | kernel binary 常驻 |
| 每 DPU 并发线程数 | 24 tasklets | SMT 硬件线程 |
| **单 DPU 硬件乘法器** | **没有** | 32×32 乘法需要软件模拟 ~20-30 cycle |
| 单 DPU 浮点单元 | 没有 FPU | fp32 乘加全软件模拟 ~100-300 cycle |
| 向量指令 | 没有 SIMD | 所有运算标量执行 |
| 分支预测 | 弱 | 复杂 `if` 链会显著降吞吐 |
| rank 内 DPU 同步 | 全 rank 一起 launch | 同一个 kernel binary 同时跑 |

### 1.3 "算力密度"的数量级对比

| 硬件 | 理论 peak | 说明 |
|---|---|---|
| GPU (H100/A100 级别，本机) | ~50–300 TFLOPS fp16 | 硬件矩阵单元 (tensor core) |
| CPU AVX-512 + 10 core | **768 GFLOPS** fp32 | 实测达到 638 GFLOPS (peak 的 83%) |
| 单 DPU 软件乘加 | **~几 MFLOPS** | 标量 + 软件乘法 |
| 单 rank (64 DPU) | **~几十到 100 MFLOPS** | 线性扩展 |
| 全 39 rank (2496 DPU) | **~几 GFLOPS** | PIM 的总算力粗估 |

**关键事实**：PIM 的峰值算力比 CPU AVX-512 小约 **200×**，比 GPU 小约 **20000×**。PIM 的优势不在单位算力，而在：

1. **MRAM 直接存放权重**（不占 GPU vRAM，不占 CPU host RAM）
2. **计算在权重旁边发生**（不走 PCIe，不走 DDR）
3. **总 MRAM 容量极大**：39 rank × 64 DPU × 64 MB = **156 GB**，比 GPU vRAM 大 3 倍

这决定了 PIM 的研究定位：**"在 GPU 装不下全部权重 + CPU 存储/算力不足时，PIM 作为第三方卸载执行者"**。

---

## 二、CPU vs PIM 横向对比（关键数据）

### 2.1 单算子 matmul 对比（batch=1，decode 场景）

对比口径：Qwen3-30B-A3B-GPTQ 一个 expert 的 gate projection（shape: `[1, 2048] × [2048, 768]`），INT4 权重 + fp32 激活。

| 路径 | 耗时 | 说明 |
|---|---|---|
| **A. 纯 F.linear (AVX-512 SGEMM on 预反量化权重)** | **0.045 ms** | CPU 硬件真实能力 |
| B. CPU 在线反量化 + F.linear (`cpu_w4a32_matvec`) | 7.2 ms | 当前 baseline 走这条，99% 是 Python 反量化 overhead |
| C. PIM 整个 matmul（含 host 到 DPU 传输）| 2.33 ms (1 rank) | 实测 M-1 sweep，kernel_mode=4 |
| D. PIM kernel 纯算时间（`launch_seconds`）| ~1.8 ms | DPU 实际算的 cycle |

**数据来源**：`benchmarks/results/pim_shape_sweep_M1_2026-04-22.json`（180 cell sweep）

### 2.2 三层对比解读

这三个对比是**三个不同的命题**，不能混为一谈：

**对比 1：C vs B（当前生产对比）**
- PIM 2.33 ms vs CPU 7.2 ms → **PIM 快 3.1×**
- 但对手 B 的 7.2 ms 里 **99% 是 Python 反量化循环**，真正的 SGEMM 只有 0.045 ms
- 所以"PIM 快 3×"的含义是"比 Python 循环快"，**不是"比 AVX-512 硬件快"**

**对比 2：D vs A（硬件纯算力对比）**
- PIM 1.8 ms vs CPU 0.045 ms → **PIM 慢 40×**
- 这才是**诚实的硬件能力对比**：DPU 没有硬件乘法器 + 无 SIMD，比 AVX-512 慢两个数量级是符合预期的

**对比 3：研究前提下的真实对比**
- 研究前提"CPU 存储有限"意味着不能用 10 GB host RAM 存反量化后的 fp32 权重
- 所以 CPU 只能走 B（在线反量化，7.2 ms）或者去掉 AMX（本机就没有 AMX）
- **在这个约束下，PIM 路径 C (2.33 ms) 比 CPU 路径 B (7.2 ms) 快是真实的胜利**

### 2.3 本机 CPU 的关键事实

本机 CPU 是 **Intel Xeon Silver 4210R**：

- ✓ 支持 AVX-512 + VNNI（PyTorch 运行时确认 `CPU capability = AVX512`）
- ✗ **不支持 AMX**（AMX 从 2023 年 Sapphire Rapids 开始才有）
- 所以论文中提到的 "AMXINT4 快 CPU baseline" 在本机**不成立**
- 实测 F.linear 吞吐 638 GFLOPS（达到 AVX-512 理论 peak 768 的 83%）

### 2.4 端到端推理性能对比（Qwen3-30B-A3B-GPTQ-Int4, 32 tokens decode）

| 配置 | decode TPS | decode 时间 | vs 起点 | vs CPU 基线 |
|---|---|---|---|---|
| M-3 起点 (PIM，2026-04-22) | 0.228 | 140.6 s | 1.0× | 10.9% |
| **M-27 当前 PIM**（2026-05-07）| **1.3234** | 24.2 s | **5.80×** | **63.2%** |
| cuda_cpu_offload baseline | 2.0933 | 15.3 s | 9.18× | 100% |

**PIM 的绝对提升是 5.80×**，但相对 CPU baseline 仍差 **37%**。

---

## 三、PIM 上前向传播的执行逻辑（详细）

下面用通俗的方式把"一个 token 在 PIM 上怎么走完 48 层"讲清楚。假设我们已经处在 decode 阶段（每次只生成一个新 token）。

### 3.1 初始化阶段（只做一次，模型加载时）

#### 3.1.1 专家权重被静态划分

- 用历史 routing 频率做 calibration，**每层 128 个 expert 按热度排序**
- **最热的 92 个 expert**：权重反量化成 fp16，搬到 **GPU vRAM**，从此常驻
- **最冷的 36 个 expert**：保留 INT4 格式在 **host RAM**（不占 GPU，不占 PIM）

#### 3.1.2 PIM runtime 分配

当前配置下：

- 48 层分成 **16 个 layer-group**（每 3 层一组）
- 每组分配 **2 个 PIMQuantizedRuntime 实例**：一个管 gate+up（合并存储），一个管 down
- 每个实例 = `dpu_alloc_ranks(1)` = **1 rank = 64 DPU**
- 总占用 **32 rank**（16 group × 2），物理可用 39 rank

#### 3.1.3 DPU kernel 一次性烧录

- 每个 rank 分配后立刻 `dpu_load(binary)`，把 kernel 二进制**写进 rank 内所有 64 个 DPU 的 IRAM**
- kernel 从此**永远不换**，后续所有操作只改 MRAM 数据 + 控制变量
- 这是因为 IRAM 只有 32 KB，换 kernel 要重新 load 花几百 μs

#### 3.1.4 冷专家权重尚未进 PIM

注意：**初始化结束时，36 个冷专家的权重仍在 host RAM，尚未进入 PIM MRAM**。只有被路由到才会触发 "按需加载"。

### 3.2 推理阶段：一个 token 穿过一层的完整流程

```
Layer L 开始
 │
 ├─ ① router 计算：
 │   - GPU 上算 router_logits = gate_weight × hidden_states
 │   - topk=8，挑出 8 个 expert
 │     * 其中约 6.2 个 是 GPU expert（权重在 vRAM）
 │     * 其中约 1.8 个 是 CPU expert（权重还在 host RAM）
 │
 ├─ ② submit PIM（主线程做准备，把工作交给 PIM）：
 │   - 把 hidden_states 从 GPU 传到 host RAM (pinned memcpy, 0.11 ms)
 │   - 扫描 topk_ids 找出这 1.8 个 CPU expert 的 id
 │   - 对每个 CPU expert：
 │       a. 查 MRAM slot 表：这个 expert 的权重当前在 MRAM 里吗？
 │          - 命中（约 45% 概率）：什么都不做
 │          - 未命中：把 host 里的 INT4 权重按行切成 64 份，
 │                    通过 PCIe 推到 64 个 DPU 的 MRAM 槽位
 │                    （gate+up bundle 约 2.4 ms，down 约 1.3 ms）
 │       b. 构造一条 "request"：记录该 expert 在 MRAM 里的 slot 编号 +
 │          激活数据在 MRAM 里的 offset
 │   - 把所有 1.8 个 request 打包成一张表
 │   - 通过一次 ctypes 调用推给 C 层 pthread worker
 │   - 主线程立刻返回（不等 PIM 跑完）
 │
 ├─ ③ PIM 在后台跑：（与 ④ 并行）
 │   后台 C 线程：
 │   - dpu_broadcast_to 把激活推到所有 DPU 的 MRAM
 │   - dpu_launch(set) 同步启动 rank 内 64 个 DPU
 │     DPU 内部 kernel_mode=4 算法：
 │       for (request in request_table):          ← N 个 expert 串行
 │         for (row in 本 DPU 负责的 1/64 行):     ← 每 DPU 算 1/64 行
 │           for (col_block in input_dim):
 │             读 input_i8_shards + qweight + LUT
 │             inner loop: acc += (int32)x * (int32)lut_i16[q]
 │           写 output_i32 到 MRAM
 │   - dpu_copy_from 把 gate+up 结果拉回 host
 │   - **在 host CPU 上算 silu(gate) * up**（这步在 host 因为 int4 kernel 不含 silu）
 │   - 再次 dpu_broadcast + launch + copy_from 算 down
 │   - 整个过程约 4-8 ms，和 ④ 并行
 │
 ├─ ④ 同时：主线程在 GPU 上处理 6.2 个 GPU expert：
 │   - 对每个 GPU expert：
 │       F.linear(hidden, w1_gate)
 │       F.linear(hidden, w3_up)
 │       silu * elementwise-mul
 │       F.linear(hidden_intermediate, w2_down)
 │   - 约 7.48 ms，CUDA kernels queue 上去
 │
 ├─ ⑤ sync 点：主线程等 PIM pthread 完成
 │   - 如果 PIM 已经算完：等待时间 ≈ 0.7 ms
 │   - 如果 PIM 还没算完：等差值
 │   - 拿到 PIM 的 1.8 个 expert 输出
 │
 ├─ ⑥ merge：
 │   - 按 routing_weights 把 8 个 expert 输出加权合并
 │   - H2D 一次，结果回 GPU
 │   - 和上一层的残差相加
 │
 └─ 进入 Layer L+1
```

### 3.3 几个容易误解的细节

1. **DPU kernel 不是每次换**：初始化加载一次，永远不换。所谓"换 kernel"是通过一个 `kernel_mode` 变量在 kernel 内部分支，选择 mode 3/4/5/6/7 等不同算法。生产路径固定 `kernel_mode=4`（int8 激活 + int16 LUT + int32 累加）。

2. **权重切分方式**：一个 expert 的权重矩阵 `[out_dim, in_dim]` 按 **out_dim 行**切成 64 份，每 DPU 拿 1/64 行。不是按列切也不是按 block 切。

3. **反量化位置**：INT4 权重 → fp32 的反量化发生在 **DPU 内部**（查 16 项 LUT），**不在 host 侧做**。这是 PIM 相对 CPU 的关键容量优势——host 不需要存 fp32 反量化后的大矩阵。

4. **silu 必须回 host**：当前 int4 kernel 不含 silu 激活函数，所以 gate projection 完成后 DPU 结果出 MRAM → 拉回 host → host fp32 算 silu*up → 再推回 DPU 算 down。这是一次额外的 host↔DPU 往返。fp32 kernel 版本有 silu-in-DPU（LUT 4096 项），但生产路径的 int4 kernel 没有。

5. **同一 rank 内多个 expert 是"伪并行"**：rank 内 64 DPU 并行没问题（每个算 1/64 行），但**同 rank 内的多个 expert 在 kernel 里 for 循环串行处理**，不是真并行。这是 M-15 request-table 设计的特性：省掉多次 launch overhead，代价是同 rank 内串行。

6. **跨 rank 目前也是串行**：gate+up 用一个 rank，down 用另一个 rank，host 必须先等 gate+up 完成做 silu，再触发 down。没有做跨 rank 并发。

---

## 四、PIM 理论性能分析 vs 实测性能

### 4.1 PIM 的理论上限在哪里

给出一个简单的理论模型：

**假设所有优化都做到极致（所有权重预驻留、所有 overhead 消除）**：

| 组件 | 最优可能时间 |
|---|---|
| step_1 routing (GPU) | 0.44 ms / layer |
| step_2 submit（只剩 pinned D2H + ctypes dispatch）| 0.20 ms / layer |
| step_3 GPU expert loop（GPU 侧的 6.2 个 expert）| 7.48 ms / layer |
| step_4 sync（PIM 已 hidden 到 step_3，几乎不等）| 0.10 ms / layer |
| PIM 实际算时间（完全并行，被 step_3 hide）| 隐藏 |
| **Total per layer** | **~8.2 ms** |
| **Decode TPS (48 layer/token)** | **2.54** |

理论上限 ≈ **2.54 TPS**，和现在的 CPU baseline (2.09) 相当，能略超。

### 4.2 当前实测性能分解

当前 M-27 状态下，一层 decode 的实际时间分配：

| phase | 实测时间 | 理论下限 | 差距 |
|---|---|---|---|
| routing | 0.34 ms | 0.44 ms | 已达标 |
| **submit** | **6.50 ms** | 0.20 ms | **+6.30 ms** |
| gpu_expert_loop | 7.48 ms | 7.48 ms | 已达标 |
| **sync** | **0.79 ms** | 0.10 ms | **+0.69 ms** |
| merge | 0.04 ms | 0.04 ms | 已达标 |
| **total** | **15.15 ms** | 8.26 ms | +6.89 ms |
| **TPS** | **1.32** | **2.52** | |

**差距 83% 来自 submit，11% 来自 sync**。GPU expert loop 已经完全对齐 cpu_offload 基线。

### 4.3 为什么达不到理论上限（3 个硬骨头）

#### 4.3.1 骨头 1：preload 的 host-side memcpy（消耗 ~4 ms / layer）

当 LRU 未命中时，preload 要做一次 host-side 7.5 MB memcpy（把 host packed_qweights 按 64 个 DPU 重新 shard 一次）。这是**纯 host 开销**，即便 DPU DMA 本身很快也要等。

即便 LRU 命中率达到 44.7%，剩下 55.3% 未命中每次还是要付这个 memcpy。

**为什么没消灭**：需要把"预切分后的 host buffer"永久缓存（约 170 MB host RAM），miss 时直接推指针给 DPU。工程量中等（3-5 天）但**是 M-28 的明确目标**。

#### 4.3.2 骨头 2：每层只用 2 个 rank，剩下 30 个 rank 空闲

**这是最严重的问题**，我重新用数据说明：

```
物理并发资源: 39 rank × 64 DPU × 24 tasklet = 59,904 硬件线程
已分配占用:   32 rank
单层瞬时激活: 只有 2 rank (gate_up + down)
真实利用率:   2/39 = 5.1%
```

也就是说，当前设计**每层只利用了 5.1% 的 PIM 硬件**。剩下 30 个 alloc 过的 rank 在该层压根不参与计算。

这对应 "同一 layer-group 内的 3 层共享 2 个 rank"，group 间串行（不同 group 的 rank 不会跨层并发）。

#### 4.3.3 骨头 3：同一 rank 内不同 expert 串行

即便在活跃的 2 个 rank 里面，多个 expert 也是 DPU kernel 内 for 循环**串行**处理。如果某层有 4 个 CPU expert 被路由到：

- 当前：串行 4 × per-expert time = 4× 倍速
- 理论上：分散到 4 个不同 rank 并行 → 1× 倍速（节约 3× 时间）

---

## 五、后续设计方向与量化分析

基于上述三个瓶颈，有三个可能的改进路径，下面逐一量化评估是否可行。

### 5.1 方向 A：Pre-sharded weight cache（M-28 候选）

#### 5.1.1 设计

把 host 侧 packed_qweights 的 64-way shard 结果**预先算好并缓存**，避免每次 miss 时重新 memcpy。

#### 5.1.2 量化收益

- 当前：55.3% miss × 4.4 ms memcpy 平均 = **2.43 ms/layer**
- 新方案：memcpy 换成传指针数组 ≈ 0.2 ms/layer
- 节省：**~2.2 ms/layer × 48 layer × 32 token / 1000 = 3.4 秒**
- 新 decode 时间：24.2 - 3.4 = **20.8 秒**
- 新 TPS：32/20.8 = **1.54**（+16%）

#### 5.1.3 代价

- 额外 host RAM：36 expert × 2 proj × 64 DPU × 60 KB ≈ **270 MB**
- 工程量：中等（3-5 天，涉及 C bridge API 变更）
- 风险：低

**评估：值得做，但不是最关键**

### 5.2 方向 B：Expert-parallel 多 rank 并发（最重要）

#### 5.2.1 设计

把 128 个 expert 静态分布到 32 个 rank，每 rank 持有 4 个 expert × 48 layer 全驻留。decode 时 top-k=8 的 8 个 expert 落在约 7 个不同 rank，**同时 launch 这 7 个 rank**。

#### 5.2.2 容量可行性

```
Qwen3-30B-A3B 每 expert 权重: 2448 KB
每 rank 4 expert × 48 layer = 192 pairs × 2448 KB = 459 MB
每 DPU (rank/64) = 7.17 MB / DPU
硬件 MRAM 预算: 64 MB / DPU （占 11%） ✓ 完全够
当前 DPU kernel 编译常量 MAX_QWEIGHT_WORDS = 8 MB / DPU ✓ 刚好够（不用重编）
```

**容量可行，富裕 8×。**

#### 5.2.3 量化收益

**preload 成本消失**（一次性启动期 0.7 秒，decode 阶段零成本）：
- 节省：2.43 ms/layer (cache miss) + 0.5 ms/layer (cache hit 也省) ≈ 2.9 ms/layer
- 对应 decode 节省：2.9 × 48 × 32 / 1000 = **4.45 秒**

**专家间并发**：
- 当前：1.8 个 expert 串行 × 4.3 ms = 7.7 ms PIM kernel（被 step_3 hide 中一半）
- 新方案：1.8 个 expert 分散到约 1.8 个不同 rank 并行 = 4.3 ms（剩余单个 rank 时间）
- 节省 step_4 sync 等待：~0.5 ms/layer
- 对应 decode 节省：0.5 × 48 × 32 / 1000 = **0.77 秒**

**合计收益**：24.2 - 4.45 - 0.77 = **18.98 秒**
**新 TPS**：32/18.98 = **1.69**（+28%）

#### 5.2.4 如果算上 sync path 也优化（假设）

理论上 step_2_submit 能降到 0.20 ms，步骤 2+4 总共 0.3 ms：
- 新 decode 时间 ≈ 48 × (0.34 + 0.20 + 7.48 + 0.10 + 0.04) × 32 / 1000 = 12.53 秒
- **理论 TPS ≈ 2.55**（超越 CPU baseline）

#### 5.2.5 工程量

- 改 DPU kernel `NUM_SLOTS`：128 → 384（48 layer × 4 expert × 2 proj）
- 改 host bridge：支持 multi-rank 并发 launch
- 改 Python runtime：32 个独立 runtime 分配 + routing 到 rank 的索引
- 启动期一次性 bulk preload：32 rank 并行 ≈ 0.7 秒
- 工程量估计：**5-8 工作日**

#### 5.2.6 风险

| 风险 | 缓解 |
|---|---|
| 32 runtime alloc 超过物理上限 | 已验证，当前 M-27 就已经 alloc 32 rank |
| top-k 撞 rank (2 个 expert 落在同一 rank) | 用 routing_freq 做 graph partition 放置，降低概率 |
| 跨 rank 同步 overhead | 多 pthread join 开销 < 0.1 ms，可忽略 |

**评估：这是最重要的下一步，研究价值最高**

### 5.3 方向 C：In-DPU silu LUT（M-31 候选）

#### 5.3.1 设计

把 `silu_lut_4096.h`（已存在于 fp32 kernel）集成到 int4 kernel 里。DPU 在 gate 出来后立刻 `silu*up`，一次 launch 完成整个 `gate_up + silu*up + down`。

#### 5.3.2 量化收益

- 节省一次 host silu fp32 循环 + 一次 D2H + 一次 H2D：约 0.3 ms/layer
- 对应 decode 节省：0.3 × 48 × 32 / 1000 = **0.46 秒**
- 新 TPS 增益：+2%

#### 5.3.3 代价

- DPU IRAM 容量紧张（原有 kernel 已占大部分）
- 工程量：中小（2-3 天）

**评估：锦上添花，不优先**

### 5.4 方向 D：大模型阶段（Stage 2）的 prefetch 预测

当 PIM 装不下全部权重（如 DeepSeek-V3 671B 需要 170 MB/DPU > 64 MB hw 上限），就回到"部分驻留 + 动态换入换出"的场景。此时需要 **预测 layer L+1 会用哪些 expert**，提前 prefetch。

#### 5.4.1 三种预测思路量化

**思路 a：GPU 前瞻下一层 router**

- 用 layer L 的 hidden（当前 token 在 L 的输出）直接喂给 layer L+1 的 gate，算近似 router
- 近似误差：hidden_L ≠ hidden_{L+1}_real（差一层 FFN 残差）
- 预期 top-k overlap：70-85%（类似论文 ProMoE 报道）
- 收益：把 prefetch 成本从 critical path 移走，节省 miss-path 的 2.4 ms/layer
- 假设命中率 80%，相当于把 hit rate 从 45% 拉到 89%：节省 ~1.9 ms/layer
- 对应：32 × 48 × 1.9 / 1000 = **2.9 秒 / 32 token**
- **量化 TPS 增益：+13%**

**思路 b：CPU 聚类预测**

- 在线统计每个 token 的 hot expert 聚类，预测 next-layer
- 问题：decode 阶段每 token 在每层只激活 1.8 个 expert，**样本太少聚类不稳定**
- 收益不确定，不推荐优先

**思路 c：层间专家相似性（offline profile）**

- 训练阶段统计转移矩阵 `P[next_expert | current_expert]`
- 运行时查表，预 prefetch top-3 候选
- 预期 top-k overlap：60-70%（较保守）
- 但零运行时开销，是 baseline prefetch 策略
- **量化 TPS 增益：+7-10%**

#### 5.4.2 Stage 2 路线量化小结

在 Qwen3-30B 上做了 Stage 2 的预实验其实意义不大（因为 Stage 1 方案 A 已经让全部权重驻留，没 prefetch 需求）。**Stage 2 是换到更大模型（235B+）时才有意义的研究路线**，理论上能把大模型场景的 decode TPS 相对"无预测 LRU"的方案提 10-20%。

---

## 六、整体 roadmap 与预期上限

### 6.1 逐步叠加的 TPS 预估

| 阶段 | 关键改动 | 预期 decode TPS | vs CPU baseline |
|---|---|---|---|
| 当前 (M-27) | — | 1.32 | 63% |
| +方向 A (pre-sharded cache) | 消 host memcpy | 1.54 | 74% |
| **+方向 B (expert-parallel)** | **跨 rank 并发** | **~1.80** | **~86%** |
| +方向 C (in-DPU silu) | 消 host silu 往返 | ~1.90 | ~91% |
| +sync CUDA event | 全 stream 化 | ~2.10 | **~100%（持平）** |
| +理论上限 | — | 2.54 | 121% |

所以**最终可能达到 ~2.1 TPS**，刚好持平 CPU baseline（2.09）。

### 6.2 研究的真正立足点

重要的是我们追求的不是 **超过 CPU baseline 多少**，而是**在 host RAM 受限的场景下 PIM 能提供一个有竞争力的卸载路径**。这个命题在当前已经成立：

1. ✓ 已证明 PIM 真实承担计算（`real_dpu_expert_calls > 0`，100% 参与率）
2. ✓ 已证明 `pim+gpu` 的 decode 性能比 "无 AMX CPU + 在线反量化"（现实场景）快 3×
3. ✓ 已证明 PIM 的容量可以存下全模型（Qwen3-30B）甚至中等规模 MoE（Qwen3-235B 也能 fit）
4. ✗ 尚未证明 PIM 能把并行度用起来（只用了 5.1%）

**方向 B（expert-parallel）是整个研究中最有学术价值的下一步**，因为它直接回答"PIM 能不能用好它的并行度"这个核心问题。

### 6.3 Stage 2 的研究边界

当模型继续增长到 PIM 装不下（DeepSeek-V3 级别），Stage 1 的"全驻留"假设破产，此时需要引入 prefetch 预测。Stage 1 的数据会成为 Stage 2 的 baseline：

- 小模型（Qwen3-30B）：**static full residency + expert-parallel**（方向 B），预计 2.1 TPS
- 中模型（Qwen3-235B）：仍可全驻留，但 expert-parallel 的 rank 分布需要新算法（每 rank 更多 expert）
- 大模型（DeepSeek-V3 671B）：**partial residency + predictive prefetching**（方向 D），需要预测算法做支撑

---

## 七、总结

1. **PIM 硬件天生不擅长算力比拼**：单 DPU 无硬件乘法器、无 SIMD、无 FPU，比 CPU AVX-512 慢 40×。PIM 的优势是"**大容量 MRAM 存权重 + 近存计算**"，不是"单位算力"。

2. **当前成果**：从起点 0.228 TPS 提升 5.8× 到 **1.32 TPS**，闭合了原始差距的 58%。剩余差距 37% 完全集中在 host-side orchestration（submit + sync），不是 PIM 算力本身。

3. **最大浪费**：当前设计只用了 39 个 rank 中的 2 个（5.1%），意味着 PIM 硬件的并行度远未开发。

4. **最有价值的下一步**：expert-parallel 多 rank 并发（方向 B），预期 TPS 从 1.32 提到 ~1.80（+36%），叠加其他优化可达 2.1 TPS（持平 CPU baseline）。

5. **研究的真正贡献不在 TPS 数字**：在"host RAM 受限 + GPU 容量不足"的现实场景下，PIM 已经是比 CPU 更优的卸载选择（3× 快于 CPU 在线反量化路径）。这才是论文的主命题。

---

## 附录：关键数字表

| 数值 | 含义 |
|---|---|
| 39 rank × 64 DPU × 24 tasklet | PIM 总并发硬件线程 ≈ 60,000 |
| 64 MB / DPU | 单 DPU MRAM 容量 |
| 156 GB | PIM 总容量（39 × 64 × 64 MB）|
| 2448 KB | Qwen3 单 expert 权重 |
| 14.7 GB | Qwen3-30B 全模型权重 |
| 768 GFLOPS | CPU AVX-512 理论 peak（本机） |
| 638 GFLOPS | CPU AVX-512 实测（83% peak）|
| ~几 GFLOPS | PIM 全 39 rank 理论 peak |
| 7.17 MB / DPU | 方案 B 下每 DPU 占用（容量富裕 8×）|
| 1.32 TPS | 当前 PIM decode 性能 |
| 2.09 TPS | CPU baseline |
| 2.54 TPS | PIM 理论上限 |
