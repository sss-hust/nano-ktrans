import torch
from transformers import AutoTokenizer, PretrainedConfig
from nano_ktrans.models.mixtral import MixtralForCausalLM, MixtralConfig
from nano_ktrans.utils.loader import load_model
from nano_ktrans.engine.simple_engine import SimpleEngine

class LLM:
    """
    User-facing class for interacting with the nano-ktrans Mixtral model.
    """
    def __init__(
        self, 
        model_path: str, 
        max_seq_len: int = 2048, 
        device: str = "cuda", 
        num_gpu_experts: int = 2
    ):
        self.model_path = model_path
        self.device = device
        self.max_seq_len = max_seq_len
        
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Loading config from {model_path}...")
        hf_config = PretrainedConfig.from_pretrained(model_path)
        
        # We manually map HF config to our simplified MixtralConfig
        config = MixtralConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            num_local_experts=hf_config.num_local_experts,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            rms_norm_eps=hf_config.rms_norm_eps,
            max_position_embeddings=hf_config.max_position_embeddings,
            rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
        )
        
        # Hybrid MoE Allocation: We place `num_gpu_experts` experts on GPU and the rest on CPU
        num_layers = config.num_hidden_layers
        num_local_experts = config.num_local_experts
        
        layer_gpu_expert_masks = []
        for _ in range(num_layers):
            mask = torch.zeros(num_local_experts, dtype=torch.bool)
            mask[:num_gpu_experts] = True # Top `num_gpu_experts` experts conceptually on GPU
            layer_gpu_expert_masks.append(mask)

        print(f"Instantiating Hybrid MoE model on {device}...")
        # To avoid OOM, initialize empty dummy weights first, then load Safetensors
        with torch.device(device):
            self.model = MixtralForCausalLM(config, layer_gpu_expert_masks, weight_path=model_path)
            self.model = self.model.to(torch.bfloat16) # use bf16 for inference
            
        print(f"Loading weights from {model_path} into Python model...")
        load_model(self.model, model_path)
        
        print("Initializing SimpleEngine...")
        self.engine = SimpleEngine(self.model, max_seq_len=max_seq_len)
        print("Model loaded successfully.")

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        # 1. Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        seq_len = input_ids.shape[1]
        print(f"Prompt tokens: {seq_len}. Starts generation...")
        
        # 2. Prefill
        logits = self.engine.prefill(input_ids)
        next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
        
        generated_ids = [next_token.item()]
        
        # 3. Decode Loop
        for i in range(max_new_tokens - 1):
            logits = self.engine.decode_step(next_token, seq_len + i)
            next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
            
            token_id = next_token.item()
            generated_ids.append(token_id)
            
            # Simple stopping criteria
            if token_id == self.tokenizer.eos_token_id:
                break
                
        # 4. Decode output
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text
