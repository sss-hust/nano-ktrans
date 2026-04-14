import argparse

from nano_ktrans.llm import LLM

def main():
    parser = argparse.ArgumentParser(description="Run nano-ktrans on a local or Hugging Face MoE checkpoint.")
    parser.add_argument("model_path", help="Local model directory or Hugging Face repo id.")
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, or cuda[:index].")
    parser.add_argument(
        "--num-device-experts",
        type=int,
        default=None,
        help="Experts kept on the active device. Default keeps all experts on the active device.",
    )
    parser.add_argument(
        "--offload-backend",
        default="cpu",
        choices=["cpu", "pim_shadow"],
        help="Backend used for experts that are not kept on the active device.",
    )
    parser.add_argument(
        "--pim-rank-count",
        type=int,
        default=1,
        help="Visible PIM ranks to report when using the experimental pim_shadow backend.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    model_path = args.model_path

    print(f"Starting nano-ktrans with model: {model_path}")
    
    model = LLM(
        model_path,
        device=args.device,
        num_gpu_experts=args.num_device_experts,
        offload_backend=args.offload_backend,
        offload_backend_kwargs={"pim_rank_count": args.pim_rank_count},
    )
    
    # Generation test
    prompt = "请解释一下如何写出结构清晰的 Python 脚本。"
    print(f"\nUser: {prompt}\n")
    print("Agent: Generating response...")
    
    generated_text = model.generate(prompt, max_new_tokens=args.max_new_tokens)
    print("\n" + "="*40 + "\n")
    print(generated_text)
    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
