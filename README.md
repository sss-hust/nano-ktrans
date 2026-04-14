# nano-ktrans 🚀

A minimal, educational CPU/GPU Hybrid MoE inference framework. 
Simplified from the excellent [KTransformers](https://github.com/kvcache-ai/ktransformers) project, `nano-ktrans` aims to be the clearest educational implementation of Hybrid MoE (expert offloading) in Python. Like `nano-vLLM`, code readability, educational value, and minimalism are preferred over absolute edge-case performance.

Currently supports **Mixtral-style MoE**, **Qwen2-MoE**, and a compatibility
path for **Qwen3-MoE** checkpoints with automatic expert-layout detection.
DeepSeek-V2/V3 style MLA attention is not yet implemented, so those models
still require a dedicated adaptation pass.

## Project Highlights

- 🧠 **Hybrid Mixture-of-Experts:** Dynamically routes experts. "Hot" experts remain on GPU memory, while "Cold" experts are computed in CPU RAM using AVX512/AMX acceleration.
- ⚡ **High-Performance CPU Backend:** Ships with Python wrappers (`kernels/`) calling the `kt-kernel` C++ extension, allowing efficient multi-threaded CPU GEMMs without blocking the GPU.
- 📦 **Zero-Configuration Load:** Native SafeTensor weight loader designed to stream models without crazy conversions.
- 🏎️ **FlashAttention & Triton KV Cache:** Minimalist, transparent implementation of prefill and decode attention layers.

## Project Structure

```text
nano-ktrans/
├── nano_ktrans/
│   ├── models/                  # Core Models / architecture adapters
│   ├── layers/                  # Layers (Attention, Linear, RMSNorm, HybridMoE)
│   ├── kernels/                 # CPU Matrix Acceleration logic wrapper (AMX)
│   ├── engine/                  # Core Inference Loop
│   └── utils/                   # Globals, context, and loading utils
├── example.py                   # Minimal running example
├── tests/                       # Pytest directory
└── pyproject.toml               # uv / Python packaging definition
```

## Setup & Installation

You can bring up the core project with a standard Python virtualenv. CUDA/PIM-specific
accelerators are optional.

```bash
# 1. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the CPU-safe core dependencies
pip install -e ".[dev]"
```

Optional extras:

```bash
# CUDA attention kernels
pip install -e ".[cuda]"

# kt-kernel CPU expert backend
pip install -e ".[cpu-kernel]"

# All accelerators
pip install -e ".[accel]"
```

Without these extras, nano-ktrans now falls back to pure PyTorch attention and
keeps experts on the active device by default, which is the easiest way to
bring the project up before enabling PIM or custom kernels.

## Current Runtime Status

As of `v0.2.0`, the repository has been validated on the local
`Qwen3-30B-A3B-Base` checkpoint with these paths:

- `cpu`: end-to-end inference works.
- `cuda_cpu_offload`: works on the host machine with a small hot-expert set on GPU.
- `cuda`: still OOM on the local `47.41 GiB` GPU for this model when all experts stay on device.
- `cuda_pim_shadow`: integrated into the main inference path and records PIM visibility and routing counters, but the numerical expert compute still falls back to CPU.
- `benchmarks/pim_microbench`: runs on real UPMEM hardware and reports transfer plus integer-kernel metrics; it is not a floating-point MoE expert benchmark.

## Testing

A comprehensive test suite is provided via `pytest` to test core numerical stability, routing algorithms, linear slicing, and rope math independent of weights or GPUs.

```bash
pytest tests -v
```

## Running an Example

To execute inference, provide the local path or Hugging Face repo id for a
supported MoE checkpoint. When you do not pass `--num-device-experts`, all
experts stay on the active device, which is the simplest non-PIM path.

```bash
python example.py /path/to/local/model/weights --device cpu
```

If you have a CUDA runtime and want hybrid expert placement later:

```bash
python example.py /path/to/local/model/weights --device cuda --num-device-experts 2
```

Current status:

- Mixtral-style MoE: supported
- Qwen2-MoE style sparse MLP + shared expert: supported in the Python model path
- Qwen3-MoE style sparse MLP + packed expert projections + q/k norm: supported
- DeepSeek-V2/V3: blocked on MLA attention support

## License
Apache-2.0 License. See the KTransformers project for the original implementation.
