# nano-ktrans 🚀

A minimal, educational CPU/GPU Hybrid MoE inference framework. 
Simplified from the excellent [KTransformers](https://github.com/kvcache-ai/ktransformers) project, `nano-ktrans` aims to be the clearest educational implementation of Hybrid MoE (expert offloading) in Python. Like `nano-vLLM`, code readability, educational value, and minimalism are preferred over absolute edge-case performance.

Currently focused solely on **Mixtral-8x7B**.

## Project Highlights

- 🧠 **Hybrid Mixture-of-Experts:** Dynamically routes experts. "Hot" experts remain on GPU memory, while "Cold" experts are computed in CPU RAM using AVX512/AMX acceleration.
- ⚡ **High-Performance CPU Backend:** Ships with Python wrappers (`kernels/`) calling the `kt-kernel` C++ extension, allowing efficient multi-threaded CPU GEMMs without blocking the GPU.
- 📦 **Zero-Configuration Load:** Native SafeTensor weight loader designed to stream models without crazy conversions.
- 🏎️ **FlashAttention & Triton KV Cache:** Minimalist, transparent implementation of prefill and decode attention layers.

## Project Structure

```text
nano-ktrans/
├── nano_ktrans/
│   ├── models/                  # Core Models (Mixtral-8x7B)
│   ├── layers/                  # Layers (Attention, Linear, RMSNorm, HybridMoE)
│   ├── kernels/                 # CPU Matrix Acceleration logic wrapper (AMX)
│   ├── engine/                  # Core Inference Loop
│   └── utils/                   # Globals, context, and loading utils
├── example.py                   # Minimal running example
├── tests/                       # Pytest directory
└── pyproject.toml               # uv / Python packaging definition
```

## Setup & Installation

We recommend using [`uv`](https://github.com/astral-sh/uv) to manage the environment and dependencies quickly.

```bash
# 1. Clone the repo
git clone https://github.com/your-repo/nano-ktrans.git
cd nano-ktrans

# 2. Create and activate a fast virtual environment via uv
uv venv .venv
source .venv/bin/activate

# 3. Install the package and dependencies
uv pip install -e ".[dev]"
```

*Note: Ensure you have AVX512 and `kt-kernel` build prerequisites if you are compiling the backend locally.*

## Testing

A comprehensive test suite is provided via `pytest` to test core numerical stability, routing algorithms, linear slicing, and rope math independent of weights or GPUs.

```bash
pytest tests -v
```

## Running an Example

To execute inference, provide the local path to your unloaded Mixtral-8x7B safetensors folder:

```bash
python example.py /path/to/local/mixtral/weights
```

## License
Apache-2.0 License. See the KTransformers project for the original implementation.
