from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class SparseExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.hidden_dim = hidden_size
        self.ffn_dim = intermediate_size
        self.hidden_act = hidden_act

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        act_fn = getattr(nn.functional, self.hidden_act)
        current_hidden_states = act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        return self.w2(current_hidden_states)


class PackedSparseExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.gate_up_proj = nn.Parameter(
            torch.empty(2 * self.intermediate_size, self.hidden_dim)
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        act_fn = getattr(nn.functional, self.hidden_act)
        gate, up = nn.functional.linear(hidden_states, self.gate_up_proj).chunk(2, dim=-1)
        current_hidden_states = act_fn(gate) * up
        return self.down_proj(current_hidden_states)


def build_expert_module(
    *,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
    experts_are_packed: bool,
) -> nn.Module:
    if experts_are_packed:
        return PackedSparseExpertMLP(hidden_size, intermediate_size, hidden_act)
    return SparseExpertMLP(hidden_size, intermediate_size, hidden_act)


def load_expert_weights(module: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        if isinstance(module, PackedSparseExpertMLP):
            gate_up = torch.cat([weights["gate"], weights["up"]], dim=0)
            module.gate_up_proj.copy_(
                gate_up.to(
                    device=module.gate_up_proj.device,
                    dtype=module.gate_up_proj.dtype,
                )
            )
            module.down_proj.weight.copy_(
                weights["down"].to(
                    device=module.down_proj.weight.device,
                    dtype=module.down_proj.weight.dtype,
                )
            )
            return

        if not isinstance(module, SparseExpertMLP):
            raise TypeError(f"Unsupported expert module type: {type(module)!r}")

        module.w1.weight.copy_(
            weights["gate"].to(device=module.w1.weight.device, dtype=module.w1.weight.dtype)
        )
        module.w2.weight.copy_(
            weights["down"].to(device=module.w2.weight.device, dtype=module.w2.weight.dtype)
        )
        module.w3.weight.copy_(
            weights["up"].to(device=module.w3.weight.device, dtype=module.w3.weight.dtype)
        )
