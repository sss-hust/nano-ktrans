# PIM Guide For nano-ktrans

## What PIM Does Here

In this framework, PIM is the target backend for `cold` experts that are not kept on GPU.
The intended execution split is:

- `router + hot experts` stay on GPU
- `cold experts` move out of GPU memory
- the offloaded path first uses CPU as the compatibility backend
- later, the same offload interface can dispatch selected experts onto PIM

The key point is that PIM is not replacing the whole model. It is replacing the
backend used by offloaded experts.

## Why It Matters

For MoE inference, the bottleneck is often not the active parameter count, but
expert residency:

- keeping all experts on GPU wastes memory
- pushing all cold experts back to CPU creates PCIe and host-memory pressure
- PIM gives a third tier with much larger near-memory capacity and high parallelism

So in `nano-ktrans`, PIM should be modeled as an expert execution tier:

- Tier 1: GPU resident experts
- Tier 2: PIM resident experts
- Tier 3: CPU fallback experts

## Where It Fits In This Codebase

Current path:

- [llm.py](/home/yangfu/nano-ktrans/nano_ktrans/llm.py)
- [mixtral.py](/home/yangfu/nano-ktrans/nano_ktrans/models/mixtral.py)
- [hybrid_moe.py](/home/yangfu/nano-ktrans/nano_ktrans/layers/hybrid_moe.py)
- [cpu_moe.py](/home/yangfu/nano-ktrans/nano_ktrans/kernels/cpu_moe.py)

The current offload path is:

1. GPU computes router logits.
2. `HybridMoE` splits hot vs cold experts.
3. Hot experts run as PyTorch GPU modules.
4. Cold experts are submitted through `ExpertOffloadBackend`.
5. GPU and CPU outputs are merged.

The implemented extension today is:

1. `HybridMoE` remains the scheduler.
2. `ExpertOffloadBackend` abstracts the offloaded expert path.
3. `CPUMoEBackend` provides the compatible numerical fallback.
4. `PIMMoEBackend` currently powers the `pim_shadow` mode for PIM visibility and routing diagnostics.
5. CPU fallback is still responsible for numerical correctness when a real DPU kernel is unavailable.

## Recommended Runtime Design

The most practical design for your machine is:

- route every token on GPU
- keep a small hot set of experts on GPU
- place the medium-frequency experts on PIM
- leave rare experts on CPU fallback

That gives you three benefits:

- GPU memory pressure drops
- host CPU is no longer the only cold-expert executor
- expert placement becomes a research variable you can tune

## Suggested PIM-Specific Metrics

When you benchmark this framework, record at least:

- token latency split by prefill and decode
- per-layer expert hit distribution
- GPU-expert hit ratio
- PIM-expert hit ratio
- CPU-fallback hit ratio
- host-to-device traffic
- host-to-PIM transfer volume
- PIM kernel occupancy
- end-to-end throughput under different hot-set sizes

## Your Machine's PIM Notes

### What is directly confirmed locally

Your local DPU runtime log shows a hardware-backed rank device:

- [dpu-2026_04_01-10_39_19-463585.log](/home/yangfu/dpu-2026_04_01-10_39_19-463585.log#L1)
- [dpu-2026_04_01-10_39_19-463585.log](/home/yangfu/dpu-2026_04_01-10_39_19-463585.log#L2)
- [dpu-2026_04_01-10_39_19-463585.log](/home/yangfu/dpu-2026_04_01-10_39_19-463585.log#L6)

That log references `/dev/dpu_rank39`, which implies the machine had at least
40 enumerated ranks at the time of the run.

### What this strongly implies

UPMEM systems commonly expose:

- `1 rank = 64 DPUs`
- `2 ranks per DIMM`
- `20 DIMMs = 40 ranks = 2560 DPUs`

This matches your statement that the machine has `2560` PIM cores.

### Per-DPU capacity and local memory

For UPMEM DPUs, public documentation and published evaluations consistently
report:

- `64 MB MRAM` per DPU
- `64 KB WRAM` per DPU
- `24 KB IRAM` per DPU
- `24 hardware tasklets` per DPU

From that:

- total MRAM across `2560` DPUs is about `160 GB`
- total WRAM across `2560` DPUs is about `160 MB`
- total IRAM across `2560` DPUs is about `60 MB`

### Compute characteristics

The DPU is a small in-order 32-bit core designed for massive throughput via
parallelism, not single-core latency. In practice, the useful metric is system
parallelism across thousands of DPUs, not per-core peak FLOPS in the GPU sense.

For this framework, the relevant properties are:

- very large aggregate near-memory capacity
- high concurrency across many DPUs
- no native all-to-all between DPUs
- host CPU usually orchestrates transfers and synchronization

### Bandwidth notes

Bandwidth has to be tracked at three levels:

- host DRAM <-> PIM DIMM transfer bandwidth
- intra-DPU MRAM <-> WRAM movement
- aggregate bandwidth across all active DPUs

For the public UPMEM reference platform:

- `1 GB/s` memory bandwidth per DPU
- `128 DPUs` per DIMM
- `20 DIMMs` gives `2560 DPUs`
- total platform memory bandwidth is advertised as about `2.56 TB/s`

These are platform specifications, not yet a local measurement on your host.
The current local log is enough to confirm the rank device path, but it does not
report measured GB/s or measured clock frequency.

## What Still Needs Local Measurement

To make this guide fully machine-specific, you still need:

- measured rank count from the loaded runtime
- measured DPU clock on this host
- measured host-to-PIM transfer bandwidth
- measured single-DPU and full-system expert GEMV/GEMM throughput
- measured overlap efficiency against GPU execution

## Immediate Next Step

The next meaningful step is no longer backend refactoring. That part is already
done. The remaining gap is a real DPU numerical path:

- implement a minimal expert kernel on DPU, ideally starting from single-token single-expert GEMV
- add a reliable host bridge from Python into the UPMEM runtime
- upgrade `PIMMoEBackend` from `pim_shadow` to a true execution backend
- then compare end-to-end latency against `cuda_cpu_offload`

## Source Notes

Use the following separation when you cite this machine in notes or papers:

- `Locally confirmed`: presence of hardware-backed DPU rank device in the log
- `Official platform specification`: DPU count, DIMM count, per-DPU memory, and advertised aggregate bandwidth
- `Not yet locally measured`: sustained GB/s, expert throughput, and overlap efficiency on this exact host

Reference sources:

- Local runtime log:
  [dpu-2026_04_01-10_39_19-463585.log](/home/yangfu/dpu-2026_04_01-10_39_19-463585.log#L1)
  [dpu-2026_04_01-10_39_19-463585.log](/home/yangfu/dpu-2026_04_01-10_39_19-463585.log#L2)
  [dpu-2026_04_01-10_39_19-463585.log](/home/yangfu/dpu-2026_04_01-10_39_19-463585.log#L6)
- UPMEM technology page:
  https://www.upmem.com/technology/
- UPMEM developer page:
  https://www.upmem.com/developer/
- UPMEM ABUMPIMP 2024 hardware slides:
  https://www.upmem.com/wp-content/uploads/2024/09/240826-ABUMPIMP-2024-Keynote_-UPMEM-PIM-platform-for-Data-Intensive-Applications.pdf
