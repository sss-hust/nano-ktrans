import torch

from nano_ktrans.engine.simple_engine import SimpleEngine
from nano_ktrans.models.mixtral import MixtralConfig, MixtralForCausalLM
from nano_ktrans.utils.expert_selection import uniform_gpu_experts_masks


def _init_tiny_model(model: MixtralForCausalLM) -> None:
    with torch.no_grad():
        for param in model.parameters():
            if not param.is_floating_point():
                continue
            if param.ndim == 1:
                param.fill_(0.02)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)


def test_cpu_only_smoke_generation_path():
    config = MixtralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        decoder_sparse_step=1,
    )
    layer_gpu_expert_masks = uniform_gpu_experts_masks(
        num_layers=config.num_hidden_layers,
        num_experts=config.num_local_experts,
        num_gpu_experts=config.num_local_experts,
    )

    full_model = MixtralForCausalLM(config, layer_gpu_expert_masks).to("cpu")
    _init_tiny_model(full_model)
    full_engine = SimpleEngine(full_model, max_seq_len=32, chunk_size=8)

    full_input_ids = torch.randint(0, config.vocab_size, (1, 4))
    full_prefill_logits = full_engine.prefill(full_input_ids)
    assert full_prefill_logits.shape == (1, 4, config.vocab_size)

    chunked_model = MixtralForCausalLM(config, layer_gpu_expert_masks).to("cpu")
    _init_tiny_model(chunked_model)
    chunked_engine = SimpleEngine(chunked_model, max_seq_len=32, chunk_size=4)

    chunked_input_ids = torch.randint(0, config.vocab_size, (1, 6))
    chunked_prefill_logits = chunked_engine.prefill(chunked_input_ids)
    assert chunked_prefill_logits.shape == (1, 2, config.vocab_size)

    next_token = torch.argmax(chunked_prefill_logits[:, -1, :], dim=-1, keepdim=True)
    decode_logits = chunked_engine.decode_step(next_token, current_seq_len=6)
    assert decode_logits.shape == (1, 1, config.vocab_size)
