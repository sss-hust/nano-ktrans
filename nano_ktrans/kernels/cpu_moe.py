"""
CPUMoEBackend: CPU 端 MoE 专家计算的核心后端。

这是 nano-ktrans 性能的关键所在。它将 kt-kernel 的 C++ AMX/AVX 矩阵加速
封装成一个简洁的 Python 接口，负责：

1. **Pinned Memory Buffer 管理**: 预分配 CPU 固定内存 buffer，用于 GPU ↔ CPU 的
   高效异步数据传输（避免每次推理都重新分配内存）。

2. **权重加载和量化**: 从 SafeTensor 文件中读取 FP16/BF16 权重，通过 C++ 在线
   量化为 INT4/INT8 格式并存储在 CPU 内存中。支持 GPTQ 量化权重直接加载。

3. **异步 Forward**: 利用 CPUInfer 线程池的 submit_with_cuda_stream 接口，
   将专家计算任务提交到 CPU 上异步执行，同时 GPU 继续处理其他专家。

4. **结果同步**: 等待 CPU 计算完成后，将结果从 pinned memory 拷贝回 GPU。

架构概览：
                    ┌──────────────┐
                    │  HybridMoE   │   (层级调度)
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌──────────────────┐     ┌──────────────────┐
    │  GPU Experts     │     │  CPUMoEBackend   │
    │  (nn.Module)     │     │  (本文件)         │
    │  PyTorch Forward │     │                  │
    └──────────────────┘     │  ┌─────────────┐ │
                             │  │ CPUInfer    │ │  ← CPU 线程池
                             │  │ (cpu_infer) │ │
                             │  └─────┬───────┘ │
                             │        │         │
                             │  ┌─────▼───────┐ │
                             │  │ kt_kernel   │ │  ← C++ AMX GEMM
                             │  │ AMXInt4_MOE │ │
                             │  └─────────────┘ │
                             └──────────────────┘

依赖：
- kt_kernel_ext.moe.MOEConfig, AMXInt4_MOE 等
- kernels.cpu_infer.CPUInferEngine
- kernels.weight_loader.ExpertWeightLoader
"""

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from nano_ktrans.utils.context import get_context

from .cpu_infer import CPUInferEngine
from .offload_backend import ExpertOffloadBackend
from .weight_loader import ExpertWeightLoader, GPTQLinearWeight

logger = logging.getLogger("nano_ktrans.cpu_moe")

try:
    from kt_kernel import kt_kernel_ext  # noqa: F401
    import kt_kernel_ext.moe as _moe_mod
except ImportError:
    _moe_mod = None

MOEConfig = getattr(_moe_mod, "MOEConfig", None) if _moe_mod is not None else None
AMXInt4_MOE = getattr(_moe_mod, "AMXInt4_MOE", None) if _moe_mod is not None else None
AMXInt8_MOE = getattr(_moe_mod, "AMXInt8_MOE", None) if _moe_mod is not None else None
AMXBF16_MOE = getattr(_moe_mod, "AMXBF16_MOE", None) if _moe_mod is not None else None


class PinnedBufferPool:
    """
    管理 CPU pinned memory buffer 的池。

    在 Hybrid MoE 中，GPU 和 CPU 之间需要频繁传输数据。
    使用 pinned memory（锁页内存）可以让 cudaMemcpyAsync 达到最高带宽。

    Buffer 布局（每个 buffer slot）：
    - input_cpu:   [batch, hidden_size]  BF16, 存放从 GPU 异步拷贝的输入
    - expert_ids:  [batch, top_k]        INT64, 专家路由 ID
    - weights:     [batch, top_k]        FP32,  路由置信度权重
    - output_cpu:  [batch, hidden_size]  BF16, CPU 专家计算结果
    - output_gpu:  [batch, hidden_size]  GPU,  从 CPU 拷贝回的结果
    - bsz:         [1]                   INT32, 当前 batch size
    """

    # 双缓冲深度：允许流水线化（layer N 还在 CPU 上算，layer N+1 已经提交）
    BUFFER_DEPTH = 2

    def __init__(self):
        self._cache: Dict[int, tuple] = {}  # batch_size → buffers

    def get_buffers(self, batch_size: int, hidden_size: int, top_k: int, device: torch.device):
        """获取或创建指定 batch size 的 buffer 组"""
        if batch_size in self._cache:
            return self._cache[batch_size]

        buffers = self._allocate(batch_size, hidden_size, top_k, device)
        self._cache[batch_size] = buffers
        return buffers

    def _allocate(self, batch_size: int, hidden_size: int, top_k: int, device: torch.device):
        """分配一组双缓冲"""
        d = self.BUFFER_DEPTH
        use_pinned = torch.cuda.is_available()

        input_cpu = [
            torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, pin_memory=use_pinned, device="cpu")
            for _ in range(d)
        ]
        expert_ids_cpu = [
            torch.zeros(batch_size, top_k, dtype=torch.long, pin_memory=use_pinned, device="cpu")
            for _ in range(d)
        ]
        weights_cpu = [
            torch.zeros(batch_size, top_k, dtype=torch.float32, pin_memory=use_pinned, device="cpu")
            for _ in range(d)
        ]
        output_cpu = [
            torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, pin_memory=use_pinned, device="cpu")
            for _ in range(d)
        ]
        bsz_cpu = [
            torch.full((1,), batch_size, dtype=torch.int32, pin_memory=use_pinned, device="cpu")
            for _ in range(d)
        ]
        output_gpu = [
            torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
            for _ in range(d)
        ]

        return (input_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_cpu, output_gpu)


# 全局 buffer 池
_buffer_pool = PinnedBufferPool()


class CPUMoEBackend(ExpertOffloadBackend):
    """
    单层 CPU MoE 专家的后端。

    每一层 MoE 创建一个 CPUMoEBackend 实例。
    它负责加载该层 CPU 专家的权重，并在推理时异步执行 forward。

    使用示例：
        backend = CPUMoEBackend(
            layer_idx=0, num_experts=8, top_k=2,
            hidden_size=4096, intermediate_size=14336,
            gpu_experts_mask=mask,
            weight_path="/path/to/model",
            method="AMXINT4",
        )
        # 推理时：
        backend.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
        output = backend.sync_forward(hidden_states, cuda_stream)
    """

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        gpu_experts_mask: torch.Tensor,
        weight_path: str,
        num_threads: int = 16,
        numa_pools: int = 1,
        chunked_prefill_size: int = 512,
        method: str = "AMXINT4",
        expert_key_template: Optional[str] = None,
        expert_proj_names: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.method = method
        self.use_fallback = True
        self.moe = None
        self.cpu_infer = None
        self._fallback_output = None
        self.fallback_dtype = torch.bfloat16
        self.has_cpu_experts = bool((~gpu_experts_mask.bool()).any().item())
        self.cpu_expert_lookup: Dict[int, int] = {}
        self.backend_name = "cpu"
        self.is_gptq = False
        # Per-expert GPTQ weights: cpu_slot → {proj: GPTQLinearWeight}
        self._gptq_experts: Dict[int, Dict[str, GPTQLinearWeight]] = {}

        if not self.has_cpu_experts:
            self.gpu_experts_mask = gpu_experts_mask.to(dtype=torch.uint8, device="cpu")
            return

        # 硬件检测：检查是否有 amx 或 avx512 标志
        has_amx = False
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read().lower()
                has_amx = "amx" in content
        except OSError as exc:
            logger.debug("Failed to read /proc/cpuinfo for AMX detection: %s", exc)

        # GPU 专家掩码 (pinned, 供 C++ 读取)
        self.gpu_experts_mask = torch.empty(
            num_experts,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        self.gpu_experts_mask.copy_(gpu_experts_mask)

        # ===== 加载原始专家权重 =====
        loader = ExpertWeightLoader(weight_path)

        # ===== 检测 GPTQ 量化权重 =====
        self._detect_and_load_gptq(
            loader, layer_idx, num_experts, gpu_experts_mask,
            expert_key_template, expert_proj_names,
        )

        # GPTQ path: per-expert quantized weights already loaded into
        # ``self._gptq_experts`` and consumed downstream by the PIM
        # quantized runtime.  The fp16 stacked tensors below do NOT exist
        # on disk for GPTQ checkpoints (only ``.qweight / .scales`` do),
        # so calling ``load_layer_experts_stacked`` in that mode aborts
        # with ``Weight key '...gate_proj.weight' not found``.  Skip the
        # stacked fp16 load for GPTQ layers; the C++ AMX path and the
        # torch-fallback path below are also GPTQ-unaware so we take the
        # same early return and let downstream PIM routing handle decode.
        if self.is_gptq:
            self._gate_proj = None
            self._up_proj = None
            self._down_proj = None
            self.use_fallback = False  # dispatched by PIM quantized path
            # Downstream PIM routing uses cpu_expert_lookup to map
            # expert_idx → per-expert GPTQ weight slot in
            # self._gptq_experts, so it must be populated even though no
            # stacked fp16 tensors exist.
            _cpu_expert_indices = torch.where(~gpu_experts_mask.bool())[0].to(torch.long).cpu()
            self.cpu_expert_lookup = {
                int(expert_idx): slot
                for slot, expert_idx in enumerate(_cpu_expert_indices.tolist())
            }
            logger.info(
                "Layer %s: CPU stacked fp16 load skipped (GPTQ mode, %s CPU experts, handled by quantized DPU path)",
                layer_idx, len(self.cpu_expert_lookup),
            )
            return

        stacked = loader.load_layer_experts_stacked(
            layer_idx,
            num_experts,
            key_template=expert_key_template or "model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight",
            proj_name_map=expert_proj_names,
        )
        self._gate_proj = stacked["gate"].contiguous()
        self._up_proj = stacked["up"].contiguous()
        self._down_proj = stacked["down"].contiguous()

        if has_amx and MOEConfig is not None:
            try:
                self.cpu_infer = CPUInferEngine.get_instance(num_threads, numa_pools)

                # ===== 创建 C++ MOE 实例 =====
                moe_config = MOEConfig(
                    num_experts,
                    top_k,
                    hidden_size,
                    intermediate_size,
                    self.gpu_experts_mask.data_ptr(),
                )
                moe_config.layer_idx = layer_idx
                moe_config.pool = self.cpu_infer.backend
                moe_config.max_len = chunked_prefill_size
                moe_config.save = True
                moe_config.load = False
                
                # 重定向量化缓存
                cache_path = os.path.join(os.getcwd(), "quant_cache")
                os.makedirs(cache_path, exist_ok=True)
                moe_config.path = cache_path

                moe_config.gate_proj = self._gate_proj.data_ptr()
                moe_config.up_proj = self._up_proj.data_ptr()
                moe_config.down_proj = self._down_proj.data_ptr()

                if method == "AMXINT4":
                    self.moe = AMXInt4_MOE(moe_config)
                elif method == "AMXINT8":
                    self.moe = AMXInt8_MOE(moe_config)
                
                # 提交量化
                self._identity_map = torch.arange(num_experts, dtype=torch.long)
                self.cpu_infer.submit(self.moe.load_weights_task(self._identity_map.data_ptr()))
                self.cpu_infer.sync()
                logger.info("Layer %s: Accelerated by %s", layer_idx, method)
                self.use_fallback = False
                return # 初始化成功
            except Exception as e:
                logger.warning(
                    "Layer %s: Failed to init %s (%s). Falling back to PyTorch.",
                    layer_idx, method, e,
                )
        else:
            logger.info(
                "Layer %s: kt-kernel/AMX unavailable. Using PyTorch fallback.",
                layer_idx,
            )

        # ===== Fallback 模式：仅保留 CPU 专家权重，直接用 F.linear 计算，避免复制一整套 nn.Linear =====
        cpu_expert_indices = torch.where(~gpu_experts_mask.bool())[0].to(torch.long).cpu()
        self.cpu_expert_lookup = {
            int(expert_idx): slot for slot, expert_idx in enumerate(cpu_expert_indices.tolist())
        }
        self._gate_proj = self._gate_proj.index_select(0, cpu_expert_indices).to(dtype=self.fallback_dtype).contiguous()
        self._up_proj = self._up_proj.index_select(0, cpu_expert_indices).to(dtype=self.fallback_dtype).contiguous()
        self._down_proj = self._down_proj.index_select(0, cpu_expert_indices).to(dtype=self.fallback_dtype).contiguous()

    def _detect_and_load_gptq(
        self,
        loader: ExpertWeightLoader,
        layer_idx: int,
        num_experts: int,
        gpu_experts_mask: torch.Tensor,
        expert_key_template: Optional[str],
        expert_proj_names: Optional[Dict[str, str]],
    ) -> None:
        """Detect if model uses GPTQ quantization and load per-expert quantized weights."""
        if not loader._quantize_config:
            return
        bits = loader._quantize_config.get("bits")
        if bits is None:
            return

        # Build key template for GPTQ loading
        gptq_key_template = expert_key_template or \
            "model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight"

        # Check if qweight keys exist for the first CPU expert
        cpu_expert_indices = torch.where(~gpu_experts_mask.bool())[0].tolist()
        if not cpu_expert_indices:
            return

        first_expert = cpu_expert_indices[0]
        proj_map = expert_proj_names or {"gate": "w1", "up": "w3", "down": "w2"}
        # Try to find a qweight key for the first projection of the first CPU expert
        test_proj = list(proj_map.values())[0]
        test_key = gptq_key_template.format(layer=layer_idx, expert=first_expert, proj=test_proj)
        if test_key.endswith(".weight"):
            test_prefix = test_key[:-len(".weight")]
        else:
            test_prefix = test_key
        qweight_key = test_prefix + ".qweight"
        if qweight_key not in loader._key_to_file:
            return

        # GPTQ detected — load per-expert quantized weights
        self.is_gptq = True
        for slot, expert_idx in enumerate(cpu_expert_indices):
            expert_gptq: Dict[str, GPTQLinearWeight] = {}
            for proj_name in ["gate", "up", "down"]:
                try:
                    gptq_weight = loader.load_gptq_expert_linear(
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        proj_name=proj_name,
                        key_template=gptq_key_template,
                        proj_name_map=expert_proj_names,
                    )
                    expert_gptq[proj_name] = gptq_weight
                except (KeyError, ValueError):
                    # If any projection fails, abandon GPTQ for this layer
                    self.is_gptq = False
                    self._gptq_experts.clear()
                    return
            self._gptq_experts[slot] = expert_gptq

        if self.is_gptq:
            logger.info(
                "Layer %s: GPTQ-Int%s weights detected (%s CPU experts)",
                layer_idx, bits, len(self._gptq_experts),
            )

    def update_gpu_expert_mask(self, gpu_experts_mask: torch.Tensor) -> None:
        self.gpu_experts_mask = gpu_experts_mask.to(dtype=torch.uint8, device="cpu")

    def export_expert_weights(self, expert_idx: int) -> Dict[str, torch.Tensor] | None:
        cpu_slot = self.cpu_expert_lookup.get(int(expert_idx))
        if cpu_slot is None:
            return None
        # GPTQ path never materializes fp16 stacked tensors; promotion
        # back to GPU goes through per-expert safetensor reload via
        # ExpertMaterializationManager, not through this fast path.
        if self.is_gptq or self._gate_proj is None:
            return None
        return {
            "gate": self._gate_proj[cpu_slot].to(dtype=self.fallback_dtype).contiguous().cpu(),
            "up": self._up_proj[cpu_slot].to(dtype=self.fallback_dtype).contiguous().cpu(),
            "down": self._down_proj[cpu_slot].to(dtype=self.fallback_dtype).contiguous().cpu(),
        }

    def _compute_expert_output_cpu(self, states: torch.Tensor, cpu_slot: int) -> torch.Tensor:
        if self.is_gptq or self._gate_proj is None:
            raise RuntimeError(
                "CPUMoEBackend fp16 expert compute is unavailable in GPTQ mode; "
                "this layer must be routed through the quantized DPU path."
            )
        states = states.to(dtype=self.fallback_dtype)
        gate = F.linear(states, self._gate_proj[cpu_slot])
        up = F.linear(states, self._up_proj[cpu_slot])
        hidden = F.silu(gate) * up
        return F.linear(hidden, self._down_proj[cpu_slot])

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream: int | None,
    ):
        """
        异步提交 CPU 专家计算（非阻塞）。

        数据流：
        1. hidden_states (GPU) → input_cpu (pinned) via non_blocking copy
        2. topk_ids, topk_weights → pinned buffers via non_blocking copy
        3. submit_with_cuda_stream: CPU 线程池在 CUDA 拷贝完成后开始计算
        """
        self.submit_calls += 1
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        device = hidden_states.device
        context = get_context()

        if not self.has_cpu_experts:
            self._fallback_output = torch.zeros_like(hidden_states)
            return

        # In GPTQ mode the CPU backend's fp16 stacked weights were never
        # loaded and the C++ AMX engine was never initialised, so neither
        # the ``use_fallback`` F.linear path nor the ``cpu_infer`` path
        # below can run.  A subclass (e.g. PIMMoEBackend) is expected to
        # have already filled ``self._fallback_output`` via its own
        # ``_submit_forward_real``; if we still get here it means the
        # subclass bailed out (prefill_force_cpu, batch_too_large, ...)
        # and there is no honest fp16 fallback available — emit zeros so
        # downstream sync_forward returns a well-shaped tensor instead of
        # dereferencing None pointers.
        if self.is_gptq and self.cpu_infer is None:
            if self._fallback_output is None or self._fallback_output.shape != hidden_states.shape:
                self._fallback_output = torch.zeros_like(hidden_states)
            return

        if self.use_fallback:
            cpu_mask = ~self.gpu_experts_mask.bool()
            flat_cpu = flat.to("cpu", dtype=self.fallback_dtype)
            topk_ids_cpu = topk_ids.to("cpu", dtype=torch.long)
            topk_weights_cpu = topk_weights.to("cpu", dtype=torch.float32)

            output = torch.zeros(batch_size, self.hidden_size, dtype=self.fallback_dtype, device="cpu")

            for expert_idx in range(self.num_experts):
                if not cpu_mask[expert_idx]:
                    continue

                cpu_slot = self.cpu_expert_lookup.get(int(expert_idx))
                if cpu_slot is None:
                    continue

                match = topk_ids_cpu == expert_idx
                token_indices = torch.where(match.any(dim=1))[0]
                if len(token_indices) == 0:
                    continue
                expert_output = self._compute_expert_output_cpu(flat_cpu[token_indices], cpu_slot)

                row_idx, col_idx = torch.where(match[token_indices])
                weights = topk_weights_cpu[token_indices[row_idx], col_idx].to(dtype=expert_output.dtype).unsqueeze(1)
                output.index_add_(0, token_indices[row_idx], expert_output[row_idx] * weights)

            self._fallback_output = output.to(device=device, dtype=hidden_states.dtype)
            return

        (input_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_cpu, _output_gpu) = \
            _buffer_pool.get_buffers(batch_size, self.hidden_size, self.top_k, device)

        slot = self.layer_idx % PinnedBufferPool.BUFFER_DEPTH

        # 异步拷贝: GPU → CPU pinned memory
        input_cpu[slot].copy_(flat, non_blocking=True)
        expert_ids_cpu[slot].copy_(topk_ids.long(), non_blocking=True)
        weights_cpu[slot].copy_(topk_weights, non_blocking=True)

        # 提交 CPU forward 任务
        self.cpu_infer.submit_with_cuda_stream(
            cuda_stream,
            self.moe.forward_task(
                bsz_cpu[slot].data_ptr(),
                expert_ids_cpu[slot].size(-1),
                expert_ids_cpu[slot].data_ptr(),
                weights_cpu[slot].data_ptr(),
                input_cpu[slot].data_ptr(),
                output_cpu[slot].data_ptr(),
                not context.is_prefill,  # incremental
            ),
        )

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream: int | None) -> torch.Tensor:
        """
        同步等待 CPU 计算完成，并将结果拷回 GPU。

        Returns:
            output_gpu: [batch_size, hidden_size] on GPU
        """
        self.sync_calls += 1
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        device = hidden_states.device

        if self.use_fallback:
            if self._fallback_output is None:
                return torch.zeros_like(hidden_states)
            return self._fallback_output

        # GPTQ-with-no-cpu_infer path: PIMMoEBackend (or subclass) has
        # already populated self._fallback_output from its quantized
        # submit; the C++ AMX path below is not available here.
        if self.is_gptq and self.cpu_infer is None:
            if self._fallback_output is None:
                return torch.zeros_like(hidden_states)
            return self._fallback_output.to(device=device)

        (_input_cpu, _expert_ids_cpu, _weights_cpu, output_cpu, _bsz_cpu, output_gpu) = \
            _buffer_pool.get_buffers(batch_size, self.hidden_size, self.top_k, device)

        slot = self.layer_idx % PinnedBufferPool.BUFFER_DEPTH

        # 等待 CPU 完成
        self.cpu_infer.sync_with_cuda_stream(cuda_stream)

        # 异步拷贝: CPU pinned → GPU
        output_gpu[slot].copy_(output_cpu[slot], non_blocking=True)

        return output_gpu[slot]

    def diagnostics(self) -> dict:
        diagnostics = super().diagnostics()
        diagnostics.update(
            {
                "backend_name": self.backend_name,
                "layer_idx": self.layer_idx,
                "has_cpu_experts": self.has_cpu_experts,
                "use_fallback": self.use_fallback,
                "method": self.method,
                "cpu_expert_count": len(self.cpu_expert_lookup),
            }
        )
        return diagnostics
