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
from nano_ktrans.kernels.expert_materialization import ExpertMaterializationManager
from nano_ktrans.kernels.offload_backend import normalize_offload_backend_name
from nano_ktrans.kernels.pim_moe import PIMMoEBackend
from nano_ktrans.layers.expert_mlp import build_expert_module, load_expert_weights
from nano_ktrans.scheduler import DynamicExpertScheduler
from nano_ktrans.utils.expert_runtime_state import ExpertResidency, ExpertResidencyPlan
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
        experts_are_packed: bool = False,
        hidden_act: str = "silu",
        expert_prefetch_cache_size: int = 8,
        expert_prefetch_workers: int = 1,
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
        self.experts_are_packed = experts_are_packed
        self.hidden_act = hidden_act
        self.weight_path = weight_path
        self.expert_key_template = (
            expert_key_template
            or "model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight"
        )
        self.expert_proj_names = expert_proj_names
        self.has_cpu_experts = bool((~gpu_experts_mask.bool()).any().item())
        self.offload_backend_name = normalize_offload_backend_name(offload_backend)
        self.offload_backend_kwargs = offload_backend_kwargs or {}
        self.applied_migration_ops = 0
        self.last_applied_migration_phase = ""
        self.applied_migration_history: list[dict[str, object]] = []
        self.prefetch_requested = 0
        self.prefetch_materialized = 0
        self.runtime_evictions = 0
        self.decode_prefetch_hits = 0
        self.decode_prefetch_misses = 0
        self.materialization_manager = ExpertMaterializationManager(
            weight_path=weight_path,
            expert_key_template=self.expert_key_template,
            expert_proj_names=self.expert_proj_names,
            max_cached_experts=expert_prefetch_cache_size,
            prefetch_workers=expert_prefetch_workers,
        )

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

    def _build_runtime_expert(self, expert_idx: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
        expert = build_expert_module(
            hidden_size=self.hidden_size,
            intermediate_size=self.offload_backend.intermediate_size if self.offload_backend is not None else 0,
            hidden_act=self.hidden_act,
            experts_are_packed=self.experts_are_packed,
        )
        expert = expert.to(device=device, dtype=dtype)
        weights = self.materialization_manager.get_expert(
            self.layer_idx,
            expert_idx,
        )
        load_expert_weights(expert, weights)
        expert.eval()
        return expert

    def _request_prefetch(self, expert_idx: int) -> None:
        self.materialization_manager.prefetch(self.layer_idx, expert_idx)
        self.prefetch_requested += 1

    def _set_residency(self, expert_idx: int, residency: ExpertResidency) -> None:
        if self.residency_plan is None:
            return
        state = self.residency_plan.layer_state(self.layer_idx)
        if residency == ExpertResidency.GPU:
            state.residency[expert_idx] = 1
        elif residency == ExpertResidency.PIM:
            state.residency[expert_idx] = 2
        else:
            state.residency[expert_idx] = 3

    def _runtime_gpu_budget(self) -> int:
        if self.dynamic_expert_scheduler is None or not self.dynamic_expert_scheduler.enabled:
            return int(self.gpu_experts_mask.bool().sum().item())
        return max(0, int(self.dynamic_expert_scheduler.config.gpu_budget_per_layer))

    def _runtime_gpu_residents(self) -> list[int]:
        return [int(expert_idx) for expert_idx in torch.where(self.gpu_experts_mask.bool())[0].tolist()]

    def _synchronize_gpu_mask(self) -> None:
        if self.residency_plan is not None:
            layer_state = self.residency_plan.layer_state(self.layer_idx)
            self.gpu_experts_mask = layer_state.gpu_mask().to(device=self.gpu_experts_mask.device)
        if self.offload_backend is not None:
            self.offload_backend.update_gpu_expert_mask(self.gpu_experts_mask)

    def _promote_expert_to_gpu(self, expert_idx: int, device: torch.device, dtype: torch.dtype) -> bool:
        expert_key = str(expert_idx)
        if expert_key not in self.gpu_experts:
            self.gpu_experts[expert_key] = self._build_runtime_expert(expert_idx, device, dtype)
        self.gpu_experts_mask[expert_idx] = True
        self._set_residency(expert_idx, ExpertResidency.GPU)
        return True

    def _demote_expert_from_gpu(self, expert_idx: int, dst: ExpertResidency) -> bool:
        expert_key = str(expert_idx)
        if expert_key in self.gpu_experts:
            del self.gpu_experts[expert_key]
        self.gpu_experts_mask[expert_idx] = False
        self._set_residency(expert_idx, dst)
        return True

    def _coalesce_migration_ops(self, queued_ops: list) -> list:
        latest_by_expert: dict[int, object] = {}
        ordered_expert_ids: list[int] = []
        for op in queued_ops:
            expert_idx = int(op.expert_idx)
            if expert_idx in latest_by_expert:
                ordered_expert_ids.remove(expert_idx)
            latest_by_expert[expert_idx] = op
            ordered_expert_ids.append(expert_idx)
        return [latest_by_expert[expert_idx] for expert_idx in ordered_expert_ids]

    def _pick_eviction_candidate(
        self,
        protected_experts: set[int],
        fallback_dst: ExpertResidency,
    ) -> Optional[tuple[int, ExpertResidency]]:
        gpu_residents = self._runtime_gpu_residents()
        candidates = [expert_idx for expert_idx in gpu_residents if expert_idx not in protected_experts]
        if not candidates:
            return None

        hotness = None
        if self.residency_plan is not None:
            hotness = self.residency_plan.layer_state(self.layer_idx).hotness

        if hotness is not None and hotness.numel() > 0:
            coldest = min(candidates, key=lambda expert_idx: float(hotness[expert_idx].item()))
        else:
            coldest = min(candidates)
        return coldest, fallback_dst

    def _promotion_sort_key(self, op, active_experts: set[int], phase: str) -> tuple[int, int, float, int]:
        hotness_score = 0.0
        if self.residency_plan is not None:
            state = self.residency_plan.layer_state(self.layer_idx)
            if int(op.expert_idx) < state.hotness.numel():
                hotness_score = float(state.hotness[int(op.expert_idx)].item())
        is_active = 1 if int(op.expert_idx) in active_experts else 0
        is_ready = 1 if phase == "decode" and self.materialization_manager.is_ready(self.layer_idx, int(op.expert_idx)) else 0
        return (-is_ready, -is_active, -hotness_score, int(op.expert_idx))

    def _apply_queued_migrations(
        self,
        hidden_states: torch.Tensor,
        active_experts: set[int],
        *,
        phase: str,
    ) -> None:
        if (
            phase != "decode"
            or self.offload_backend is None
            or self.dynamic_expert_scheduler is None
            or not self.dynamic_expert_scheduler.enabled
        ):
            return

        queued_ops = self.offload_backend.migration_manager.drain_layer(self.layer_idx)
        if not queued_ops:
            return

        queued_ops = self._coalesce_migration_ops(queued_ops)
        max_promotions = max(1, int(self.dynamic_expert_scheduler.config.decode_promote_k))
        gpu_budget = self._runtime_gpu_budget()
        applied = 0
        applied_promotions = 0
        deferred = []
        device = hidden_states.device
        dtype = hidden_states.dtype
        promotion_ops = [op for op in queued_ops if op.dst == ExpertResidency.GPU]
        demotion_ops = [
            op for op in queued_ops if op.src == ExpertResidency.GPU and op.dst != ExpertResidency.GPU
        ]
        promotion_ops.sort(key=lambda op: self._promotion_sort_key(op, active_experts, phase))

        for op in demotion_ops:
            if int(op.expert_idx) in active_experts:
                deferred.append(op)
                continue
            applied_ok = self._demote_expert_from_gpu(op.expert_idx, op.dst)
            if applied_ok:
                applied += 1
                self.applied_migration_history.append(
                    {
                        "phase": phase,
                        "expert_idx": op.expert_idx,
                        "src": op.src.value,
                        "dst": op.dst.value,
                        "reason": op.reason,
                    }
                )
            else:
                deferred.append(op)

        protected_experts = set(active_experts)
        protected_experts.update(int(op.expert_idx) for op in promotion_ops)

        for op in promotion_ops:
            if applied_promotions >= max_promotions:
                deferred.append(op)
                continue

            if bool(self.gpu_experts_mask[int(op.expert_idx)].item()):
                continue

            while gpu_budget > 0 and int(self.gpu_experts_mask.bool().sum().item()) >= gpu_budget:
                fallback_dst = (
                    self.dynamic_expert_scheduler.config.offload_tier
                    if self.dynamic_expert_scheduler is not None
                    else ExpertResidency.PIM
                )
                victim = self._pick_eviction_candidate(protected_experts, fallback_dst)
                if victim is None:
                    deferred.append(op)
                    break
                victim_idx, victim_dst = victim
                self._demote_expert_from_gpu(victim_idx, victim_dst)
                self.runtime_evictions += 1
                applied += 1
                self.applied_migration_history.append(
                    {
                        "phase": phase,
                        "expert_idx": victim_idx,
                        "src": ExpertResidency.GPU.value,
                        "dst": victim_dst.value,
                        "reason": "evict_for_promotion",
                    }
                )
            else:
                if self.materialization_manager.is_ready(self.layer_idx, int(op.expert_idx)):
                    self.decode_prefetch_hits += 1
                else:
                    self.decode_prefetch_misses += 1
                applied_ok = self._promote_expert_to_gpu(op.expert_idx, device, dtype)
                if applied_ok:
                    applied += 1
                    applied_promotions += 1
                    self.prefetch_materialized += 1
                    self.applied_migration_history.append(
                        {
                            "phase": phase,
                            "expert_idx": op.expert_idx,
                            "src": op.src.value,
                            "dst": op.dst.value,
                            "reason": op.reason,
                        }
                    )
                else:
                    deferred.append(op)
                continue

            continue

        self.applied_migration_history = self.applied_migration_history[-64:]

        if deferred:
            self.offload_backend.queue_migration_plan(deferred, phase=f"{phase}_deferred")

        if applied:
            self.applied_migration_ops += applied
            self.last_applied_migration_phase = phase
            self._synchronize_gpu_mask()

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
        active_experts = {int(expert_idx) for expert_idx in torch.unique(topk_ids).tolist()}
        self._apply_queued_migrations(hidden_states, active_experts, phase=phase)

        if self.dynamic_expert_scheduler is not None and self.dynamic_expert_scheduler.enabled:
            self.dynamic_expert_scheduler.observe(self.layer_idx, topk_ids, phase=phase)
            planned_ops = self.dynamic_expert_scheduler.plan_layer(self.layer_idx, phase=phase)
            if planned_ops and self.offload_backend is not None:
                for op in planned_ops:
                    if (
                        op.dst == ExpertResidency.GPU
                        and not self.materialization_manager.has_cached(self.layer_idx, int(op.expert_idx))
                    ):
                        self._request_prefetch(op.expert_idx)
                # 当前系统只有迁移控制面，没有真实 GPU/PIM 权重迁移数据面。
                # 这里先排队；decode 阶段会消费队列并执行最小 GPU materialize/demote。
                self.offload_backend.queue_migration_plan(planned_ops, phase=phase)

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
            "applied_migration_ops": self.applied_migration_ops,
            "last_applied_migration_phase": self.last_applied_migration_phase,
            "runtime_gpu_experts": sorted(int(expert_idx) for expert_idx in self.gpu_experts.keys()),
            "gpu_experts_mask_sum": int(self.gpu_experts_mask.bool().sum().item()),
            "prefetch_requested": self.prefetch_requested,
            "prefetch_materialized": self.prefetch_materialized,
            "runtime_evictions": self.runtime_evictions,
            "decode_prefetch_hits": self.decode_prefetch_hits,
            "decode_prefetch_misses": self.decode_prefetch_misses,
            "materialization_manager": self.materialization_manager.diagnostics(),
            "layer_residency": layer_residency,
            "pending_migrations": pending_migrations,
            "backend": None if self.offload_backend is None else self.offload_backend.diagnostics(),
        }
