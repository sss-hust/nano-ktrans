# Benchmarks

This directory contains two benchmark entry points:

- `benchmark_inference.py`: benchmarks nano-ktrans inference on `cpu`, `cuda`, `cuda_cpu_offload`, `cuda_pim`, and `cuda_pim_shadow`.
- `benchmark_quant_matvec.py`: benchmarks operator-only `W4A32` CPU vs PIM matvec on GPTQ-Int4 or synthetic weights.
- `pim_microbench/`: contains a standalone UPMEM microbenchmark for transfer and kernel timing.

## Inference Benchmark

Run on the local Qwen3 checkpoint:

```bash
cd /home/yangfu/nano-ktrans
./.venv/bin/python benchmarks/benchmark_inference.py \
  --model-path /home/yangfu/models/Qwen--Qwen3-30B-A3B-Base \
  --backends cpu cuda cuda_cpu_offload cuda_pim \
  --max-new-tokens 2 \
  --repeats 1
```

Notes:

- `cpu` keeps all experts on CPU.
- `cuda` keeps all experts on GPU.
- `cuda_cpu_offload` keeps only `--offload-device-experts` experts on GPU and offloads the rest to the CPU backend.
- `cuda_pim` routes offloaded experts through the real experimental PIM backend. In the current implementation, expert linear projections run on DPU while SiLU/gating stays on the host, and larger flattened batches still fall back to CPU.
- `cuda_pim_shadow` uses the same numerically-correct CPU fallback for offloaded experts but exposes PIM visibility and routing counters inside the main inference path.
- If CUDA is not available in the current session, CUDA backends are reported as `unavailable` instead of crashing.
- `benchmark_inference.py` now supports `--enable-background-offload-worker` and `--background-offload-poll-interval-seconds`, so real `cuda_pim` / dynamic-scheduler runs can exercise the staged background migration pipeline during benchmarked generation instead of only inside `example.py`.
- When `--scheduler-profile-sweep` is used, the output JSON now includes `profile_sweep_summary`, which auto-compares:
  - `decode_tokens_per_second`
  - non-cold promotion totals/ratios (`activated + warm`)
  - overlap hits and promotion source breakdown
  - layer-level apply batch metrics
  - step-level runtime apply batch totals
  - prepared-tier controller metrics, including:
    - `prepared_cache_budget` / `prepared_cache_budget_heuristic`
      - `baseline`: `max(2 * decode_promote_k, prefetch_candidate_budget, 2)`
      - `overlap_safe`: baseline 再上调一档，优先降低 strict ready-only 下的冷启动
      - `eager`: 再上调一档，配合更激进的候选预取与 prepared-tier 推进
    - `prepared_cache_budget_backoff_avg`
    - `prepared_cache_rebalance_pressure_avg / _ema_avg`
    - `cold_promotion_penalty_avg`
    - `adaptive_activation_limit_avg / adaptive_prebuild_limit_avg`
    - `adaptive_prefetch_pending_limit_avg / adaptive_prefetch_candidate_budget_avg`
  - background offload worker metrics:
    - `background_worker_enabled`
    - `background_worker_ticks / background_worker_work_ticks`
    - `background_worker_work_ratio`
  - cache eviction regression pressure (`activated -> warmed`, `warmed -> ready`)
  - deferred-for-prefetch counts
  - a ranked `comparison_table` and `best_by_metric` summary for quick profile selection

## PIM Microbenchmark

Build and run:

```bash
cd /home/yangfu/nano-ktrans
benchmarks/pim_microbench/run.sh --ranks 1 --bytes-per-dpu 1048576 --repetitions 2
```

Optional profile override:

```bash
benchmarks/pim_microbench/build/pim_microbench_host \
  --ranks 1 \
  --bytes-per-dpu 1048576 \
  --repetitions 2 \
  --profile backend=simulator,chipId=0x42
```

Metrics:

- `host_to_pim_seconds`
- `host_to_pim_gbps`
- `kernel_seconds`
- `kernel_workload`
- `kernel_effective_gbps`
- `kernel_element_gops`
- `kernel_int32_gops_estimate`
- `pim_to_host_seconds`
- `pim_to_host_gbps`
- `end_to_end_seconds`
- `kernel_cycles_min/avg/max`

Important:

- Real hardware benchmarking requires allocatable UPMEM rank devices. If `dpu_alloc_ranks` fails, the benchmark binary will exit with a DPU allocation error.
- Simulator runs are useful to validate the benchmark implementation, but not to claim hardware PIM performance.
- The current PIM microbenchmark is memory-oriented and uses integer arithmetic (`y = x * scalar + 1`). It does **not** measure floating-point throughput or MoE GEMM performance.

## Quantized Operator Benchmark

This benchmark compares only the `W4A32` matvec operator:

- CPU grouped: GPTQ-style symmetric INT4 dequantization per group + `matvec`
- CPU dense: dequantize full weight first, then run dense `matvec`
- PIM: resident INT4 weight shards + on-DPU dequantization + `matvec`

Synthetic validation:

```bash
cd /home/yangfu/nano-ktrans
source /usr/upmem_env.sh hw
./.venv/bin/python benchmarks/benchmark_quant_matvec.py \
  --synthetic \
  --synthetic-input-dim 2048 \
  --synthetic-output-dim 768 \
  --synthetic-group-size 128 \
  --batch-size 1 \
  --repeats 5 \
  --rank-count 4
```

Real GPTQ checkpoint:

```bash
cd /home/yangfu/nano-ktrans
source /usr/upmem_env.sh hw
./.venv/bin/python benchmarks/benchmark_quant_matvec.py \
  --model-path /home/yangfu/models/Qwen--Qwen3-30B-A3B-GPTQ-Int4 \
  --layer-idx 0 \
  --expert-idx 0 \
  --proj-name gate \
  --batch-size 1 \
  --repeats 5 \
  --rank-count 4 \
  --json-out benchmarks/results/qwen3_gptq_w4a32_gate.json
```

Notes:

- The current implementation assumes GPTQ `sym=true` and sequential `g_idx`.
- JSON output now includes both `cpu_grouped` and `cpu_dense` baselines:
  - `cpu_grouped` is closer to the operator path actually executed on PIM
  - `cpu_dense` is a lower-bound CPU baseline after full dequantization
- PIM weights are persisted inside the quantized runtime after `load_weights`; benchmark timing focuses on repeated operator execution, not full model inference.
- This benchmark is intentionally operator-only and does not include routing, overlap scheduling, or full MoE execution.
