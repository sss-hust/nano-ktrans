# ADR-002 M-30 — Expert-parallel static residency (POSITIVE +37.2%)

**日期**: 2026-05-08
**分支**: pim
**前置**: M-27 Stage C (1.3234 TPS)
**结果**: **+37.2% over M-27 baseline**, vs CPU baseline 从 63% 提升到 **86.7%**

## 1. 核心思想

用户提出：权重只存 GPU 和 PIM（消掉 host-side preload），**同时让不同 expert 落在不同 rank 上并行**。

实施形态：
- **每个 cold expert 独享 1 个 PIM rank**（36 expert → 36 rank）
- **启动时一次性把该 expert 的所有 48 层权重 bulk-preload** 到它自己的 rank
- decode 时 top-k=8 的 routed expert **自然落在不同 rank 上**，各 rank 并发 launch

替代了 M-7 的"16 layer-group × 2 rank = 32 rank，每层只用 2 rank"布局。

## 2. 关键容量校核

NUM_SLOTS=128 不变的前提下：
```
每 rank 持有 1 expert × 48 layer × 2 projection = 96 slot
NUM_SLOTS=128 → 96 < 128 ✓
每 DPU 4 MB / 硬件 64 MB = 6% ✓
物理 rank 占用 36 / 39 = 92% ✓
```

**无需重新编译 DPU kernel 二进制。** 纯 host-side 重组。

## 3. 实测结果

### 3.1 32-token decode（Qwen3-30B-A3B-GPTQ, offload=92）

| 指标 | M-27 baseline | **M-30** | Δ |
|---|---|---|---|
| **decode_tps** | 1.3234 | **1.8152** | **+37.2%** |
| decode_seconds | 24.18 | 17.63 | -6.55 s |
| sync_wait_mean | 0.71 ms | **0.081 ms** | **-88.6%** |
| active_ranks_mean (per layer) | 1 | **2.45** | 2.45× |
| active_ranks_max | 1 | **7** | 7× |
| PIM compute participation | 1.000 | **1.000** | ✓ |
| **vs CPU baseline (2.0933)** | **63.2%** | **86.7%** | 差距 1.68× → 1.15× |

### 3.2 诊断数据（benchmarks/results/e2e_gptq_cuda_pim_M30_expert_parallel.json）

- `m30_runtime_pool_size = 36` — 每个 cold expert 独占 1 rank
- `m30_bulk_preload_seconds = 0.159`（layer 0 自身的 preload；全 48 层累加 7.72 s 一次性启动开销）
- `m30_parallel_submit_count = 1300`（32 tokens × ~40.6 per-token cold-expert-groups）
- `c_async_sync_wait_mean = 0.081 ms` — PIM kernel 几乎完全被 GPU expert loop 隐藏

## 4. 为什么效果这么好

**主要收益来源两块，互补叠加**：

### 4.1 消掉 preload miss（-55% × 4.4 ms/layer）

M-27 的 preload 路径：每 layer-step 平均 ~2.88 cold expert 需要 DMA，hit rate 44.7% 仍有 4.4 ms/layer 花在 host shard memcpy + DPU DMA 上。

M-30：启动期一次性把全部权重推进 MRAM，decode 时 `preload_concat_and_get_slot` 全部 cache hit，零 DMA。

### 4.2 跨 rank 真并行（sync_wait 0.71 ms → 0.08 ms）

M-27：1 rank 内串行处理 1.8 个 cold expert，总 PIM kernel 时间 ~7.7 ms。GPU expert loop 7.48 ms 只 hide 其中一半，尾部溢出 ~0.71 ms 让主线程等。

M-30：1.8 个 cold expert 分散到 1.8 个不同 rank **并行**跑，单 rank 只处理 1 个 expert → 4.3 ms。**完全被 GPU loop 7.48 ms 包住**，主线程 join 时 PIM 早结束。sync_wait 从 0.71 ms 降到 0.08 ms，**-88.6%**。

## 5. 启动期开销

一次性 bulk preload（完整 48 层 × 36 expert × 2 projection = 3456 次 ctypes preload）：
- 实测：**~7.7 秒**（相对模型加载 217 秒只增加 3.5%）
- 所有 preload 在 init 阶段完成，decode 阶段零 preload

## 6. Host RAM 副作用

GPTQ weights 启动后仍留在 host `self._gptq_experts` 里（以备 fallback 路径使用），暂未释放。如果要彻底消除 host 权重副本，可以：
1. M-30 init 完成后调用 `self._gptq_experts.clear()` + `gc.collect()`
2. 把 fallback 路径设为不可达（m30 不允许 fallback，出错直接 raise）

这是可选的进一步工作，当前实现保守地保留 host 副本。

## 7. 正确性验证

- **288/288 tests pass**（`pytest tests/ -x -q`，14.86 s）
- **PIM compute participation ratio = 1.000**（100% offloaded expert 真在 DPU 算，科研守护通过）
- Smoke benchmark (4 token) + e2e benchmark (32 token) 都 `status: ok`

## 8. 代码改动

仅 2 个文件：
- `nano_ktrans/kernels/pim_moe.py`：加 `enable_m30_expert_parallel` 参数、`_try_init_quantized_runtimes_expert_parallel`、`_do_m30_expert_parallel_submit`、`_sync_forward_m30_multi`
- `benchmarks/benchmark_inference.py`：透传 `--pim-enable-m30-expert-parallel` / `--no-pim-m30-expert-parallel`

**没有改 C bridge，没有改 DPU kernel，没有改 NUM_SLOTS。** 纯 host 侧 Python 重组。

## 9. 后续可能的叠加优化

M-30 已接近 CPU baseline (86.7%)，剩余 13% 差距来源：

| 项目 | 大致时间 | 可否消除 |
|---|---|---|
| step_2_submit 内 torch.unique scan + Python dispatch | ~1.2 ms/layer | 小幅优化空间 |
| step_2_submit 内 D2H pinned copy + cuda_stream sync | ~0.1 ms/layer | 已优化 |
| step_3_gpu_expert_loop | 7.48 ms/layer | GPU 侧相同负载，与 CPU baseline 对齐 |
| step_4_sync + merge | ~0.15 ms/layer | 已 hide |

继续叠加空间有限，要进一步提升需要：
- **in-DPU silu LUT**（M-31 候选，消除 host-side silu*up 往返，+2-3%）
- **CUDA event sync**（M-32 候选，把 PIM 完成信号挂到 CUDA stream 不用 Python join，+5-8%）

或者到大模型（235B+）场景时，M-30 的 "每 expert 一 rank" 会不够，需要引入 Stage 2 的 prefetch 预测策略。

## 10. 复现命令

```bash
python benchmarks/benchmark_inference.py \
  --model-path /home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4 \
  --backends cuda_pim --offload-device-experts 92 \
  --routing-freq-json benchmarks/results/routing_freq_qwen3_30b_m23_mean.json \
  --pim-rank-count 1 --pim-layer-group-size 3 \
  --max-new-tokens 32 --warmup 0 --repeats 1 \
  --pim-enable-c-async --pim-enable-m25-pinned --pim-enable-m30-expert-parallel \
  --json-out benchmarks/results/e2e_gptq_cuda_pim_M30_expert_parallel.json
```

**报告完**。M-30 是整个 nano-ktrans PIM 研究路径上最大的一次单点胜利（+37.2%），且首次真正发挥了 PIM 的多 rank 并行能力（从 5.1% 利用率提升到 19% 峰值 × 7 rank 瞬时）。
