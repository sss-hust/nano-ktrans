"""
HybridMoE: CPU/GPU 混合专家层。

这是 nano-ktrans 的核心创新模块。它将 MoE 的专家分为两类：
- GPU 专家：常驻 GPU 显存，使用 PyTorch 直接计算
- CPU 专家：存放在 CPU 内存，通过 kt-kernel 的 AMX/AVX 加速计算

两类专家的计算是 **并发** 的：
1. 先通过 CPUMoEBackend.submit_forward() 异步启动 CPU 计算
2. 然后在 GPU 上同步计算 GPU 专家
3. 最后调用 CPUMoEBackend.sync_forward() 等待 CPU 结果
4. 合并两者的输出

这种设计充分利用了 CPU 和 GPU 的计算资源，使得即使只有少量 GPU 显存，
也能运行大型 MoE 模型。
"""

import torch
from torch import nn
from typing import Optional, Dict

from nano_ktrans.kernels.cpu_moe import CPUMoEBackend
from nano_ktrans.kernels.offload_backend import normalize_offload_backend_name
from nano_ktrans.kernels.pim_moe import PIMMoEBackend
from nano_ktrans.scheduler import DynamicExpertScheduler
from nano_ktrans.utils.expert_runtime_state import ExpertResidencyPlan
from nano_ktrans.utils.context import get_context


class HybridMoE(nn.Module):
    """
    CPU/GPU 混合 MoE 层。

    参数：
        num_experts:           专家总数 (e.g., 8)
        top_k:                 每个 token 选择的专家数 (e.g., 2)
        hidden_size:           隐藏层维度
        moe_intermediate_size: 专家 FFN 中间维度
        gpu_experts:           GPU 上的专家模块 (nn.ModuleDict)
        gpu_experts_mask:      布尔掩码, True 表示该专家在 GPU 上
        layer_idx:             层索引
        weight_path:           权重文件路径
        num_threads:           CPU 推理线程数
        numa_pools:            NUMA 子池数量
        method:                CPU 后端方法 ("AMXINT4", "AMXINT8")
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts: nn.ModuleDict,
        gpu_experts_mask: torch.Tensor,
        layer_idx: int,
        weight_path: str,
        num_threads: int = 16,
        numa_pools: int = 1,
        chunked_prefill_size: int = 512,
        method: str = "AMXINT4",
        offload_backend: str = "cpu",
        offload_backend_kwargs: Optional[Dict[str, object]] = None,
        residency_plan: Optional[ExpertResidencyPlan] = None,
        dynamic_expert_scheduler: Optional[DynamicExpertScheduler] = None,
        router_use_softmax: bool = False,
        normalize_topk_prob: bool = True,
        expert_key_template: Optional[str] = None,
        expert_proj_names: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.gpu_experts_mask = gpu_experts_mask
        self.gpu_experts = gpu_experts
        self.layer_idx = layer_idx
        self.dynamic_expert_scheduler = dynamic_expert_scheduler
        self.residency_plan = residency_plan
        self.router_use_softmax = router_use_softmax
        self.normalize_topk_prob = normalize_topk_prob
        self.has_cpu_experts = bool((~gpu_experts_mask.bool()).any().item())
        self.offload_backend_name = normalize_offload_backend_name(offload_backend)
        self.offload_backend_kwargs = offload_backend_kwargs or {}

        # 只有存在离线 CPU 专家时才初始化 CPU backend。
        self.offload_backend = None
        if self.has_cpu_experts:
            backend_kwargs = dict(
                layer_idx=layer_idx,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size,
                gpu_experts_mask=gpu_experts_mask,
                weight_path=weight_path,
                num_threads=num_threads,
                numa_pools=numa_pools,
                chunked_prefill_size=chunked_prefill_size,
                method=method,
                expert_key_template=expert_key_template,
                expert_proj_names=expert_proj_names,
            )
            backend_kwargs.update(self.offload_backend_kwargs)
            if self.offload_backend_name in {"pim", "pim_shadow"}:
                backend_kwargs.setdefault(
                    "pim_execution_mode",
                    "real" if self.offload_backend_name == "pim" else "shadow",
                )
                self.offload_backend = PIMMoEBackend(**backend_kwargs)
            else:
                self.offload_backend = CPUMoEBackend(**backend_kwargs)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        """
        前向计算。

        Args:
            hidden_states: [batch * seq_len, hidden_size]
            router_logits: [batch * seq_len, num_experts]  来自 gate 的原始 logits

        Returns:
            output: [batch * seq_len, hidden_size]  加权混合后的专家输出
        """
        # ===== Step 1: 路由 =====
        if self.router_use_softmax:
            router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
            topk_weights, topk_ids = torch.topk(router_probs, self.top_k, dim=-1)
            if self.normalize_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights, topk_ids = torch.topk(router_logits, self.top_k, dim=-1)
            topk_weights = torch.softmax(topk_weights, dim=-1)

        context = get_context()
        phase = "prefill" if context.is_prefill else "decode"
        if self.dynamic_expert_scheduler is not None and self.dynamic_expert_scheduler.enabled:
            self.dynamic_expert_scheduler.observe(self.layer_idx, topk_ids, phase=phase)
            if phase == "prefill":
                planned_ops = self.dynamic_expert_scheduler.plan_layer(self.layer_idx, phase=phase)
                if planned_ops:
                    self.dynamic_expert_scheduler.apply_plan(planned_ops)
                    if self.residency_plan is not None:
                        self.gpu_experts_mask = self.residency_plan.layer_state(self.layer_idx).gpu_mask()
                        if self.offload_backend is not None:
                            self.offload_backend.update_gpu_expert_mask(self.gpu_experts_mask)

        batch_seq_len, hidden_dim = hidden_states.shape

        # ===== Step 2: 提交 CPU 专家（异步） =====
        cuda_stream = None
        if self.offload_backend is not None:
            if hidden_states.is_cuda and torch.cuda.is_available():
                cuda_stream = torch.cuda.current_stream().cuda_stream
            self.offload_backend.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

        # ===== Step 3: 并行执行 GPU 专家 =====
        final_gpu_states = torch.zeros_like(hidden_states)

        # 构建专家掩码: [batch_seq_len, num_experts]
        expert_mask = torch.nn.functional.one_hot(
            topk_ids, num_classes=self.num_experts
        ).sum(dim=1).bool()

        for expert_idx in range(self.num_experts):
            if not self.gpu_experts_mask[expert_idx]:
                continue  # CPU 专家，跳过

            expert_key = str(expert_idx)
            if expert_key not in self.gpu_experts:
                continue

            # 找到路由到这个专家的 token
            token_indices = torch.where(expert_mask[:, expert_idx])[0]
            if len(token_indices) == 0:
                continue

            # 提取对应 token 的 hidden states
            current_state = hidden_states[token_indices]

            # GPU 专家前向
            expert_output = self.gpu_experts[expert_key](current_state)

            # 提取对应的路由权重
            expert_match = (topk_ids[token_indices] == expert_idx)
            weights = topk_weights[token_indices][expert_match]
            expert_output = expert_output * weights.unsqueeze(1)

            # 累加到结果
            final_gpu_states.index_add_(0, token_indices, expert_output)

        # ===== Step 4: 同步 CPU 专家结果 =====
        if self.offload_backend is not None:
            cpu_output = self.offload_backend.sync_forward(hidden_states, cuda_stream)
        else:
            cpu_output = torch.zeros_like(hidden_states)

        # ===== Step 5: 合并 (CPU 和 GPU 专家处理不同的 token-expert 对) =====
        return final_gpu_states + cpu_output

    def diagnostics(self) -> dict:
        layer_residency = None
        pending_migrations = []
        if self.residency_plan is not None:
            state = self.residency_plan.layer_state(self.layer_idx)
            layer_residency = {
                "gpu_experts": int(state.gpu_mask().sum().item()),
                "pim_experts": int(state.pim_mask().sum().item()),
                "cpu_experts": int(state.cpu_mask().sum().item()),
                "epoch": state.epoch,
            }
            pending_migrations = [
                {
                    "expert_idx": op.expert_idx,
                    "src": op.src.value,
                    "dst": op.dst.value,
                    "reason": op.reason,
                }
                for op in state.pending_ops
            ]
        return {
            "layer_idx": self.layer_idx,
            "offload_backend_name": self.offload_backend_name,
            "has_cpu_experts": self.has_cpu_experts,
            "layer_residency": layer_residency,
            "pending_migrations": pending_migrations,
            "backend": None if self.offload_backend is None else self.offload_backend.diagnostics(),
        }
