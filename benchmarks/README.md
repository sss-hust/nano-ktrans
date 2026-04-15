# Benchmarks

This directory contains two benchmark entry points:

- `benchmark_inference.py`: benchmarks nano-ktrans inference on `cpu`, `cuda`, `cuda_cpu_offload`, `cuda_pim`, and `cuda_pim_shadow`.
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
- When `--scheduler-profile-sweep` is used, the output JSON now includes `profile_sweep_summary`, which auto-compares:
  - `decode_tokens_per_second`
  - overlap hits and promotion source breakdown
  - layer-level apply batch metrics
  - step-level runtime apply batch totals
  - deferred-for-prefetch counts

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
