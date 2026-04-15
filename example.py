import argparse

from nano_ktrans.llm import LLM
from nano_ktrans.scheduler import SCHEDULER_PROFILE_NAMES

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
        choices=["cpu", "pim", "pim_shadow"],
        help="Backend used for experts that are not kept on the active device.",
    )
    parser.add_argument(
        "--pim-rank-count",
        type=int,
        default=1,
        help="Visible PIM ranks to report when using experimental PIM backends.",
    )
    parser.add_argument(
        "--pim-profile",
        default="",
        help="Optional libdpu allocation profile passed to the real PIM backend.",
    )
    parser.add_argument(
        "--pim-max-batch-tokens",
        type=int,
        default=1,
        help="Maximum token rows routed through the real PIM backend before falling back to CPU.",
    )
    parser.add_argument(
        "--pim-kernel-variant",
        default="linear",
        choices=["linear", "fused"],
        help="Real PIM kernel variant: 'linear' runs three DPU linears with host activation, 'fused' runs the full expert MLP on DPU.",
    )
    parser.add_argument(
        "--pim-prefill-policy",
        default="cpu",
        choices=["cpu", "pim"],
        help="Policy for prefill-routed experts. Recommended: keep prefill on CPU/GPU and reserve PIM mainly for decode.",
    )
    parser.add_argument(
        "--pim-prefill-token-threshold",
        type=int,
        default=8,
        help="Maximum flattened tokens allowed to use real PIM during prefill before forcing fallback.",
    )
    parser.add_argument(
        "--enable-dynamic-expert-scheduler",
        action="store_true",
        help="Enable experimental dynamic GPU/PIM expert residency scheduler.",
    )
    parser.add_argument(
        "--scheduler-prefill-force-gpu-budget-per-layer",
        type=int,
        default=None,
        help="During prefill, temporarily target at least this many GPU-resident experts per layer.",
    )
    parser.add_argument(
        "--scheduler-prefill-collect-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="During prefill, only collect hotness and prefetch candidates without emitting migrations.",
    )
    parser.add_argument(
        "--scheduler-step-stride-prefill",
        type=int,
        default=None,
        help="Logical step stride used when updating hotness during prefill.",
    )
    parser.add_argument(
        "--scheduler-step-stride-decode",
        type=int,
        default=None,
        help="Logical step stride used when updating hotness during decode.",
    )
    parser.add_argument(
        "--scheduler-demotion-idle-steps",
        type=int,
        default=None,
        help="Minimum logical steps since last access before a GPU expert may be demoted.",
    )
    parser.add_argument(
        "--scheduler-migration-cooldown-steps",
        type=int,
        default=None,
        help="Minimum logical steps between consecutive residency changes for the same expert.",
    )
    parser.add_argument(
        "--scheduler-decode-require-prefetch-ready",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="During decode, only promote experts whose staging prefetch is already ready; otherwise defer migration.",
    )
    parser.add_argument(
        "--scheduler-prefetch-candidate-budget-per-layer",
        type=int,
        default=None,
        help="Number of offloaded experts per layer to proactively prefetch based on hotness, even without an immediate migration op.",
    )
    parser.add_argument(
        "--scheduler-profile",
        default="baseline",
        choices=list(SCHEDULER_PROFILE_NAMES),
        help="Scheduler preset. 'baseline' keeps current behavior, 'overlap_safe' prefers ready-only decode promotions, 'eager' is more aggressive.",
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
        offload_backend_kwargs={
            "pim_rank_count": args.pim_rank_count,
            "pim_profile": args.pim_profile,
            "pim_max_batch_tokens": args.pim_max_batch_tokens,
            "pim_kernel_variant": args.pim_kernel_variant,
            "pim_prefill_policy": args.pim_prefill_policy,
            "pim_prefill_token_threshold": args.pim_prefill_token_threshold,
        },
        enable_dynamic_expert_scheduler=args.enable_dynamic_expert_scheduler,
        scheduler_prefill_force_gpu_budget_per_layer=args.scheduler_prefill_force_gpu_budget_per_layer,
        scheduler_prefill_collect_only=args.scheduler_prefill_collect_only,
        scheduler_step_stride_prefill=args.scheduler_step_stride_prefill,
        scheduler_step_stride_decode=args.scheduler_step_stride_decode,
        scheduler_demotion_idle_steps=args.scheduler_demotion_idle_steps,
        scheduler_migration_cooldown_steps=args.scheduler_migration_cooldown_steps,
        scheduler_decode_require_prefetch_ready=args.scheduler_decode_require_prefetch_ready,
        scheduler_prefetch_candidate_budget_per_layer=args.scheduler_prefetch_candidate_budget_per_layer,
        scheduler_profile=args.scheduler_profile,
    )
    
    # Generation test
    prompt = "请解释一下如何写出结构清晰的 Python 脚本。"
    print(f"\nUser: {prompt}\n")
    print("Agent: Generating response...")
    
    generated_text = model.generate(prompt, max_new_tokens=args.max_new_tokens)
    print("\n" + "="*40 + "\n")
    print(generated_text)
    print("\n" + "="*40 + "\n")
    diagnostics = model.get_offload_diagnostics()
    print("Scheduler Summary:")
    from nano_ktrans.scheduler import summarize_offload_diagnostics
    print(summarize_offload_diagnostics(diagnostics))

if __name__ == "__main__":
    main()
