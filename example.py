import sys
from nano_ktrans.llm import LLM

def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1" # Default to Mixtral

    print(f"Starting nano-ktrans with model: {model_path}")
    
    # Initialize the minimal LLM
    model = LLM(model_path, num_gpu_experts=2) # Keep 2 experts on GPU, rest on CPU
    
    # Generation test
    prompt = "[INST] How do I write a good Python script? [/INST]"
    print(f"\nUser: {prompt}\n")
    print("Agent: Generating response...")
    
    generated_text = model.generate(prompt, max_new_tokens=256)
    print("\n" + "="*40 + "\n")
    print(generated_text)
    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
