"""
weight_loader: CPU 专家权重的 SafeTensor 加载器。

这个模块负责从本地 SafeTensor 文件中加载 MoE 专家的权重，
并将它们组织成 kt-kernel C++ 层可以直接读取的格式。

支持的权重格式：
- GPTQ INT4（带 scales + zeros）
- BF16 / FP16 原始权重

加载流程：
1. 扫描目录下的 .safetensors 文件
2. 按 layer_idx 和 expert_idx 提取 gate_proj / up_proj / down_proj
3. 返回权重张量列表，供 CPUMoEBackend 传递给 C++ MOE 实例
"""

import os
from glob import glob
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from safetensors import safe_open


@dataclass
class GPTQLinearWeight:
    qweight: torch.Tensor
    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]
    group_size: int
    bits: int
    sym: bool
    linear_prefix: str

    @property
    def input_dim(self) -> int:
        return int(self.qweight.shape[1]) * self.values_per_word

    @property
    def output_dim(self) -> int:
        return int(self.qweight.shape[0])

    @property
    def num_groups(self) -> int:
        return int(self.scales.shape[1])

    @property
    def values_per_word(self) -> int:
        return 32 // self.bits

    def unpack_qvalues(self) -> torch.Tensor:
        shifts = torch.arange(self.values_per_word, dtype=torch.int32) * self.bits
        packed = self.qweight.to(dtype=torch.int32, copy=True).contiguous()
        unpacked = ((packed.unsqueeze(-1) >> shifts) & ((1 << self.bits) - 1)).to(dtype=torch.float32)
        return unpacked.reshape(self.output_dim, self.qweight.shape[1] * self.values_per_word)

    def dequantize(self) -> torch.Tensor:
        weight = torch.empty(self.output_dim, self.input_dim, dtype=torch.float32)
        qvalues = self.unpack_qvalues()
        for group_idx in range(self.num_groups):
            start = group_idx * self.group_size
            end = min(start + self.group_size, self.input_dim)
            scale = self.scales[:, group_idx].unsqueeze(1)
            if self.zero_points is None:
                zero = torch.full_like(scale, float(1 << (self.bits - 1)))
            else:
                zero = self.zero_points[:, group_idx].unsqueeze(1)
            weight[:, start:end] = (qvalues[:, start:end] - zero) * scale
        return weight


class ExpertWeightLoader:
    """
    从 SafeTensor 文件中加载指定层的 MoE 专家权重。

    用法：
        loader = ExpertWeightLoader("/path/to/model")
        weights = loader.load_layer_experts(layer_idx=0, num_experts=8)
        # weights["gate"] = [tensor_expert_0, tensor_expert_1, ...]
    """

    def __init__(self, weight_path: str):
        self.weight_path = weight_path
        self._files = sorted(glob(os.path.join(weight_path, "*.safetensors")))
        if not self._files:
            raise FileNotFoundError(f"No .safetensors files found in {weight_path}")

        # 建立 key → file 的索引, 避免每次加载都扫描所有文件
        self._key_to_file: Dict[str, str] = {}
        for f in self._files:
            with safe_open(f, "pt", "cpu") as sf:
                for key in sf.keys():
                    self._key_to_file[key] = f
        self._quantize_config = self._load_quantize_config()

    def _load_quantize_config(self) -> Dict[str, object]:
        for filename in ("quantize_config.json", "config.json"):
            path = os.path.join(self.weight_path, filename)
            if not os.path.exists(path):
                continue
            try:
                import json

                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            if filename == "quantize_config.json":
                return payload
            quant = payload.get("quantization_config")
            if isinstance(quant, dict):
                return quant
        return {}

    def _load_tensor(self, key: str) -> torch.Tensor:
        file_path = self._key_to_file.get(key)
        if file_path is None:
            raise KeyError(
                f"Weight key '{key}' not found in safetensors files. "
                f"Available keys can be listed with the safetensors CLI."
            )
        with safe_open(file_path, "pt", "cpu") as sf:
            return sf.get_tensor(key).contiguous()

    @staticmethod
    def _normalize_scale_layout(
        scales: torch.Tensor,
        *,
        output_dim: int,
        num_groups: int,
    ) -> torch.Tensor:
        if tuple(scales.shape) == (output_dim, num_groups):
            return scales.to(dtype=torch.float32).contiguous()
        if tuple(scales.shape) == (num_groups, output_dim):
            return scales.transpose(0, 1).to(dtype=torch.float32).contiguous()
        raise ValueError(
            "Unsupported GPTQ scales shape: "
            f"expected {(output_dim, num_groups)} or {(num_groups, output_dim)}, got {tuple(scales.shape)}"
        )

    @staticmethod
    def _linear_prefix_from_template(
        *,
        key_template: str,
        layer_idx: int,
        expert_idx: int,
        proj_name: str,
    ) -> str:
        key = key_template.format(layer=layer_idx, expert=expert_idx, proj=proj_name)
        if key.endswith(".weight"):
            return key[: -len(".weight")]
        return key

    def load_layer_experts(
        self,
        layer_idx: int,
        num_experts: int,
        key_template: str = "model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight",
        proj_name_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        加载指定层全部专家的三组权重。

        Args:
            layer_idx:      层索引
            num_experts:    专家总数
            key_template:   SafeTensor 中的 key 模板
                           Mixtral 默认: model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight

        Returns:
            字典，包含三个列表:
            {
                "gate": [expert_0_w1, expert_1_w1, ...],  # gate_proj (w1)
                "up":   [expert_0_w3, expert_1_w3, ...],  # up_proj   (w3)
                "down": [expert_0_w2, expert_1_w2, ...],  # down_proj (w2)
            }
        """
        result = {"gate": [], "up": [], "down": []}

        proj_map = proj_name_map or {"gate": "w1", "up": "w3", "down": "w2"}

        uses_gate_up = "gate_up" in proj_map

        for expert_idx in range(num_experts):
            for proj_name, w_name in proj_map.items():
                key = key_template.format(
                    layer=layer_idx, expert=expert_idx, proj=w_name
                )
                if key not in self._key_to_file:
                    raise KeyError(
                        f"Weight key '{key}' not found in safetensors files. "
                        f"Available keys can be listed with the safetensors CLI."
                    )
                file_path = self._key_to_file[key]
                with safe_open(file_path, "pt", "cpu") as sf:
                    tensor = sf.get_tensor(key)
                    if proj_name == "gate_up":
                        gate, up = tensor.chunk(2, dim=0)
                        result["gate"].append(gate.contiguous())
                        result["up"].append(up.contiguous())
                    else:
                        result[proj_name].append(tensor.contiguous())

        if uses_gate_up:
            assert len(result["gate"]) == num_experts
            assert len(result["up"]) == num_experts

        return result

    def load_gptq_linear(
        self,
        linear_prefix: str,
        *,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
        sym: Optional[bool] = None,
    ) -> GPTQLinearWeight:
        bits = int(bits or self._quantize_config.get("bits", 4))
        group_size = int(group_size or self._quantize_config.get("group_size", 128))
        sym = bool(self._quantize_config.get("sym", True) if sym is None else sym)

        qweight_raw = self._load_tensor(f"{linear_prefix}.qweight")
        scales_raw = self._load_tensor(f"{linear_prefix}.scales")
        qweight = qweight_raw.transpose(0, 1).to(dtype=torch.int32).contiguous()
        output_dim = qweight.shape[0]
        input_dim = qweight.shape[1] * (32 // bits)
        num_groups = (input_dim + group_size - 1) // group_size

        g_idx_key = f"{linear_prefix}.g_idx"
        if g_idx_key in self._key_to_file:
            g_idx = self._load_tensor(g_idx_key).to(dtype=torch.long)
            expected = torch.arange(input_dim, dtype=torch.long) // group_size
            if g_idx.numel() < input_dim or not torch.equal(g_idx[:input_dim], expected):
                raise NotImplementedError(
                    "Only sequential GPTQ grouping is currently supported for W4A32 benchmarking."
                )

        scales = self._normalize_scale_layout(
            scales_raw,
            output_dim=output_dim,
            num_groups=num_groups,
        )
        qzeros_key = f"{linear_prefix}.qzeros"
        zero_points: Optional[torch.Tensor] = None
        if qzeros_key in self._key_to_file:
            if not sym:
                raise NotImplementedError("Asymmetric GPTQ INT4 is not supported yet.")
            _ = self._load_tensor(qzeros_key)
        return GPTQLinearWeight(
            qweight=qweight,
            scales=scales,
            zero_points=zero_points,
            group_size=group_size,
            bits=bits,
            sym=sym,
            linear_prefix=linear_prefix,
        )

    def load_gptq_expert_linear(
        self,
        *,
        layer_idx: int,
        expert_idx: int,
        proj_name: str,
        key_template: str = "model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
        proj_name_map: Optional[Dict[str, str]] = None,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
        sym: Optional[bool] = None,
    ) -> GPTQLinearWeight:
        proj_map = proj_name_map or {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
        if proj_name not in proj_map:
            raise KeyError(f"Unknown expert projection name: {proj_name}")
        linear_prefix = self._linear_prefix_from_template(
            key_template=key_template,
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            proj_name=proj_map[proj_name],
        )
        return self.load_gptq_linear(
            linear_prefix,
            bits=bits,
            group_size=group_size,
            sym=sym,
        )

    def load_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        key_template: str = "model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight",
        proj_name_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        加载单个专家的权重。

        Returns:
            {
                "gate": Tensor[intermediate, hidden],
                "up":   Tensor[intermediate, hidden],
                "down": Tensor[hidden, intermediate],
            }
        """
        proj_map = proj_name_map or {"gate": "w1", "up": "w3", "down": "w2"}
        result: Dict[str, torch.Tensor] = {}

        for proj_name, w_name in proj_map.items():
            key = key_template.format(layer=layer_idx, expert=expert_idx, proj=w_name)
            if key not in self._key_to_file:
                raise KeyError(
                    f"Weight key '{key}' not found in safetensors files. "
                    f"Available keys can be listed with the safetensors CLI."
                )
            file_path = self._key_to_file[key]
            with safe_open(file_path, "pt", "cpu") as sf:
                tensor = sf.get_tensor(key)
                if proj_name == "gate_up":
                    gate, up = tensor.chunk(2, dim=0)
                    result["gate"] = gate.contiguous()
                    result["up"] = up.contiguous()
                else:
                    result[proj_name] = tensor.contiguous()

        return result

    def load_layer_experts_stacked(
        self,
        layer_idx: int,
        num_experts: int,
        key_template: str = "model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight",
        proj_name_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        加载并堆叠为 [num_experts, ...] 形状（用于 AMX 在线量化模式）。

        Returns:
            {
                "gate": Tensor[num_experts, intermediate_size, hidden_size],
                "up":   Tensor[num_experts, intermediate_size, hidden_size],
                "down": Tensor[num_experts, hidden_size, intermediate_size],
            }
        """
        experts = self.load_layer_experts(
            layer_idx,
            num_experts,
            key_template=key_template,
            proj_name_map=proj_name_map,
        )
        return {
            "gate": torch.stack(experts["gate"]),
            "up": torch.stack(experts["up"]),
            "down": torch.stack(experts["down"]),
        }
