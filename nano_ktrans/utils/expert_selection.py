"""
GPU Expert Selection: 基于激活频率的"热门"专家选择。

这是 ktransformers Hybrid MoE 的核心策略：
- 分析校准数据集上各专家的激活频率
- 将最频繁激活的专家放在 GPU 上（"热"专家）
- 将其余专家放在 CPU 上（"冷"专家）

这种策略比简单地"选前 N 个专家"效果好得多：
不同层的热门专家可能不同，基于数据驱动的选择能最大化 GPU 利用率。

使用流程：
    1. 准备校准文本 → profile_expert_activation() 收集激活频率
    2. generate_gpu_experts_masks() 根据频率选择 GPU 专家
    3. 将生成的 masks 传入模型构造
"""

import torch
from typing import List


def generate_gpu_experts_masks(
    activation_freq: torch.Tensor,
    num_gpu_experts: int,
) -> List[torch.Tensor]:
    """
    基于激活频率，为每一层选择放在 GPU 上的热门专家。

    与 ktransformers 的 generate_gpu_experts_masks() 功能一致。

    Args:
        activation_freq: [num_layers, num_experts] 每个专家的激活频率统计
        num_gpu_experts: 每层放在 GPU 上的专家数量

    Returns:
        masks: 长度为 num_layers 的列表，每项是 [num_experts] 的 bool Tensor
               True = GPU 专家, False = CPU 专家

    示例:
        >>> freq = torch.tensor([[10, 2, 8, 1, 5, 3, 7, 4],   # layer 0
        ...                      [1, 9, 3, 8, 2, 6, 4, 7]])   # layer 1
        >>> masks = generate_gpu_experts_masks(freq, num_gpu_experts=2)
        >>> masks[0]  # layer 0: expert 0 (freq=10) 和 expert 2 (freq=8)
        tensor([ True, False,  True, False, False, False, False, False])
        >>> masks[1]  # layer 1: expert 1 (freq=9) 和 expert 3 (freq=8)
        tensor([False,  True, False,  True, False, False, False, False])
    """
    num_layers, num_experts = activation_freq.shape
    k = min(num_gpu_experts, num_experts)
    masks = []
    for layer_idx in range(num_layers):
        freq = activation_freq[layer_idx]
        _, top_indices = torch.topk(freq, k)
        mask = torch.zeros(num_experts, dtype=torch.bool)
        mask[top_indices] = True
        masks.append(mask)
    return masks


def uniform_gpu_experts_masks(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
) -> List[torch.Tensor]:
    """
    简单的均匀专家放置策略（不需要激活频率数据）。

    每层固定选择前 num_gpu_experts 个专家放在 GPU 上。
    适用于没有校准数据时的 fallback。
    """
    masks = []
    k = min(num_gpu_experts, num_experts)
    for _ in range(num_layers):
        mask = torch.zeros(num_experts, dtype=torch.bool)
        mask[:k] = True
        masks.append(mask)
    return masks


def profile_expert_activation(
    model,
    tokenizer,
    calibration_texts: List[str],
    device: str = "cuda",
) -> torch.Tensor:
    """
    在校准数据上运行模型，统计每个专家的激活频率。

    通过在 gate 层注册 forward hook，记录 top-k 路由结果。

    Args:
        model:    已初始化的 MixtralForCausalLM 模型
        tokenizer: 对应的 tokenizer
        calibration_texts: 校准文本列表
        device:   推理设备

    Returns:
        activation_freq: [num_layers, num_experts] 归一化的激活频率
    """
    config = model.model.config
    num_layers = config.num_hidden_layers
    num_experts = config.num_local_experts
    top_k = config.num_experts_per_tok

    activation_counts = torch.zeros(num_layers, num_experts)
    total_tokens = 0

    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        total_tokens += input_ids.shape[1]

        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            def make_hook(lidx):
                def hook_fn(module, input, output):
                    # gate output: [batch*seq_len, num_experts] logits
                    topk_ids = torch.topk(output, top_k, dim=-1).indices
                    for eid in topk_ids.flatten():
                        activation_counts[lidx, eid.item()] += 1
                return hook_fn
            h = layer.gate.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        with torch.no_grad():
            positions = torch.arange(input_ids.shape[1], device=device)
            model(input_ids, positions)

        for h in hooks:
            h.remove()

    if total_tokens > 0:
        activation_freq = activation_counts / total_tokens
    else:
        activation_freq = activation_counts

    return activation_freq
