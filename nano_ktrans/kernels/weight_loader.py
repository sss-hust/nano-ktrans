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
from typing import Dict, List, Optional
import torch
from safetensors import safe_open


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
