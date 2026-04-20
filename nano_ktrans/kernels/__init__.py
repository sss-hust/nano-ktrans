"""
nano-ktrans kernels 包。

这个包封装了 kt-kernel 的 C++ CPU 加速算子，提供简洁的 Python 接口。

核心组件：
- CPUInferEngine:     CPU 异步推理线程池
- CPUMoEBackend:      MoE 专家 CPU 后端（buffer 管理 + AMX GEMM）
- ExpertWeightLoader: SafeTensor 权重加载器

架构：
    CPUMoEBackend 内部使用 CPUInferEngine 提交异步任务，
    使用 ExpertWeightLoader 加载权重，
    使用 kt_kernel_ext 的 AMXInt4_MOE 执行实际的矩阵乘法。
"""

from .cpu_infer import CPUInferEngine
from .expert_migration import ExpertMigrationManager
from .cpu_moe import CPUMoEBackend
from .offload_backend import ExpertOffloadBackend, count_visible_pim_ranks, normalize_offload_backend_name
from .pim_expert_runtime import PIMExpertRuntime
from .pim_linear_runtime import PIMLinearRuntime
from .pim_moe import PIMMoEBackend
from .pim_quantized_runtime import PIMQuantizedRuntime
from .weight_loader import ExpertWeightLoader

__all__ = [
    "CPUInferEngine",
    "ExpertMigrationManager",
    "ExpertOffloadBackend",
    "CPUMoEBackend",
    "PIMExpertRuntime",
    "PIMLinearRuntime",
    "PIMMoEBackend",
    "PIMQuantizedRuntime",
    "ExpertWeightLoader",
    "count_visible_pim_ranks",
    "normalize_offload_backend_name",
]
