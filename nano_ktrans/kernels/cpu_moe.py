"""
CPUMoEBackend: CPU 端 MoE 专家计算的核心后端。

这是 nano-ktrans 性能的关键所在。它将 kt-kernel 的 C++ AMX/AVX 矩阵加速
封装成一个简洁的 Python 接口，负责：

1. **Pinned Memory Buffer 管理**: 预分配 CPU 固定内存 buffer，用于 GPU ↔ CPU 的
   高效异步数据传输（避免每次推理都重新分配内存）。

2. **权重加载和量化**: 从 SafeTensor 文件中读取 FP16/BF16 权重，通过 C++ 在线
   量化为 INT4/INT8 格式并存储在 CPU 内存中。

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

import os
import torch
from typing import Optional, List, Dict
from kt_kernel import kt_kernel_ext

# 导入 C++ MoE 模块
import kt_kernel_ext.moe as _moe_mod
MOEConfig = _moe_mod.MOEConfig

# 检测可用的 AMX 后端
AMXInt4_MOE = getattr(_moe_mod, "AMXInt4_MOE", None)
AMXInt8_MOE = getattr(_moe_mod, "AMXInt8_MOE", None)
AMXBF16_MOE = getattr(_moe_mod, "AMXBF16_MOE", None)
from torch import nn

from .cpu_infer import CPUInferEngine
from .weight_loader import ExpertWeightLoader


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

        input_cpu = [
            torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, pin_memory=True, device="cpu")
            for _ in range(d)
        ]
        expert_ids_cpu = [
            torch.zeros(batch_size, top_k, dtype=torch.long, pin_memory=True, device="cpu")
            for _ in range(d)
        ]
        weights_cpu = [
            torch.zeros(batch_size, top_k, dtype=torch.float32, pin_memory=True, device="cpu")
            for _ in range(d)
        ]
        output_cpu = [
            torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, pin_memory=True, device="cpu")
            for _ in range(d)
        ]
        bsz_cpu = [
            torch.full((1,), batch_size, dtype=torch.int32, pin_memory=True, device="cpu")
            for _ in range(d)
        ]
        output_gpu = [
            torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
            for _ in range(d)
        ]

        return (input_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_cpu, output_gpu)


# 全局 buffer 池
_buffer_pool = PinnedBufferPool()


class CPUMoEBackend:
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
    ):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.method = method
        self.use_fallback = False

        # 硬件检测：检查是否有 amx 或 avx512 标志
        has_amx = False
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read().lower()
                has_amx = "amx" in content
        except: pass

        # GPU 专家掩码 (pinned, 供 C++ 读取)
        self.gpu_experts_mask = torch.empty(num_experts, dtype=torch.uint8, device="cpu", pin_memory=True)
        self.gpu_experts_mask.copy_(gpu_experts_mask)

        # 获取 CPU 推理引擎单例
        self.cpu_infer = CPUInferEngine.get_instance(num_threads, numa_pools)

        # ===== 加载原始专家权重 =====
        loader = ExpertWeightLoader(weight_path)
        stacked = loader.load_layer_experts_stacked(layer_idx, num_experts)
        self._gate_proj = stacked["gate"].contiguous()
        self._up_proj = stacked["up"].contiguous()
        self._down_proj = stacked["down"].contiguous()

        if has_amx:
            try:
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
                print(f"  [CPUMoEBackend] Layer {layer_idx}: Accelerated by {method}")
                return # 初始化成功
            except Exception as e:
                print(f"  [CPUMoEBackend] Layer {layer_idx}: Failed to init {method} ({e}). Falling back to PyTorch.")
        else:
            print(f"  [CPUMoEBackend] Layer {layer_idx}: AMX not supported. Using PyTorch Fallback.")

        # ===== Fallback 模式：初始化简单的 PyTorch CPU 专家 =====
        self.use_fallback = True
        self.fallback_experts = nn.ModuleList()
        for i in range(num_experts):
            # 强制指定 device="cpu"，防止被 torch.device("cuda") 上下文拦截
            exp = nn.Module().to("cpu")
            exp.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, device="cpu").to(torch.bfloat16)
            exp.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, device="cpu").to(torch.bfloat16)
            exp.w3 = nn.Linear(hidden_size, intermediate_size, bias=False, device="cpu").to(torch.bfloat16)
            
            exp.w1.weight.data.copy_(self._gate_proj[i])
            exp.w2.weight.data.copy_(self._down_proj[i])
            exp.w3.weight.data.copy_(self._up_proj[i])
            self.fallback_experts.append(exp)
        
        # 定义同步同步 buffer (fallback 模式下 sync 会用到)
        self._fallback_output = None

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream: int,
    ):
        """
        异步提交 CPU 专家计算（非阻塞）。

        数据流：
        1. hidden_states (GPU) → input_cpu (pinned) via non_blocking copy
        2. topk_ids, topk_weights → pinned buffers via non_blocking copy
        3. submit_with_cuda_stream: CPU 线程池在 CUDA 拷贝完成后开始计算
        """
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        device = hidden_states.device

        (input_cpu, expert_ids_cpu, weights_cpu, output_cpu, bsz_cpu, _output_gpu) = \
            _buffer_pool.get_buffers(batch_size, self.hidden_size, self.top_k, device)

        slot = self.layer_idx % PinnedBufferPool.BUFFER_DEPTH

        # 异步拷贝: GPU → CPU pinned memory
        input_cpu[slot].copy_(flat, non_blocking=True)
        expert_ids_cpu[slot].copy_(topk_ids.long(), non_blocking=True)
        weights_cpu[slot].copy_(topk_weights, non_blocking=True)

        if self.use_fallback:
            # Fallback 模式：直接在 CPU 上计算
            # 既然是 Python 计算，我们需要确保 GPU 到 CPU 的异步拷贝已完成
            torch.cuda.synchronize()
            
            # 找到哪些 token 指向了 CPU 专家 (gpu_experts_mask 中为 False 的专家)
            cpu_mask = ~self.gpu_experts_mask.bool()
            
            output = torch.zeros(batch_size, self.hidden_size, dtype=torch.bfloat16, device="cpu")
            flat_cpu = input_cpu[slot] # 刚才已经 copy 到了 pinned mem，直接复用
            
            for expert_idx in range(self.num_experts):
                if not cpu_mask[expert_idx]:
                    continue
                
                # 寻找选择该专家的 token
                match = (expert_ids_cpu[slot] == expert_idx)
                token_indices = torch.where(match.any(dim=1))[0]
                
                if len(token_indices) == 0:
                    continue
                
                # 执行投影
                exp = self.fallback_experts[expert_idx]
                states = flat_cpu[token_indices]
                
                # MoE 核心计算：w2(silu(w1(x)) * w3(x))
                expert_output = exp.w2(torch.nn.functional.silu(exp.w1(states)) * exp.w3(states))
                
                # 乘路由权重
                # 我们需要找到每个 token-expert 对对应的 weight
                # topk_weights 是 [batch, top_k]
                row_idx, col_idx = torch.where(match[token_indices])
                w = weights_cpu[slot][token_indices[row_idx], col_idx].unsqueeze(1)
                
                output.index_add_(0, token_indices[row_idx], expert_output[row_idx] * w)
            
            self._fallback_output = output
            return

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
                False,  # incremental
            ),
        )

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream: int) -> torch.Tensor:
        """
        同步等待 CPU 计算完成，并将结果拷回 GPU。

        Returns:
            output_gpu: [batch_size, hidden_size] on GPU
        """
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat.shape[0]
        device = hidden_states.device

        (_input_cpu, _expert_ids_cpu, _weights_cpu, output_cpu, _bsz_cpu, output_gpu) = \
            _buffer_pool.get_buffers(batch_size, self.hidden_size, self.top_k, device)

        slot = self.layer_idx % PinnedBufferPool.BUFFER_DEPTH

        if self.use_fallback:
            # 将 CPU 计算结果拷回 GPU
            output_gpu[slot].copy_(self._fallback_output, non_blocking=True)
            return output_gpu[slot]

        # 等待 CPU 完成
        self.cpu_infer.sync_with_cuda_stream(cuda_stream)

        # 异步拷贝: CPU pinned → GPU
        output_gpu[slot].copy_(output_cpu[slot], non_blocking=True)

        return output_gpu[slot]
