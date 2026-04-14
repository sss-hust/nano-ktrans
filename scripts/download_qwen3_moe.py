#!/usr/bin/env python3
import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a Qwen3 MoE checkpoint for nano-ktrans.")
    parser.add_argument(
        "--repo",
        default="Qwen/Qwen3-30B-A3B-Base",
        help="Hugging Face repo id to download.",
    )
    parser.add_argument(
        "--dest",
        default=os.path.expanduser("~/models"),
        help="Destination root directory.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only fetch config/tokenizer files.",
    )
    args = parser.parse_args()

    allow_patterns = ["*.json", "tokenizer*", "vocab.json", "merges.txt"]
    if not args.config_only:
        allow_patterns.append("*.safetensors")

    local_dir = os.path.join(args.dest, args.repo.replace("/", "--"))
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {args.repo} -> {local_dir}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        resume_download=True,
        local_dir_use_symlinks=False,
    )
    print("Done.")


if __name__ == "__main__":
    main()
