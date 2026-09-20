#!/usr/bin/env python3
"""Merge a compact LingBot-VLA LoRA adapter into a full HF checkpoint."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.utils.lora_utils import merge_lora_adapter_into_hf_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline merge of a LingBot-VLA LoRA export into deployable Hugging Face safetensors.",
    )
    parser.add_argument("--base-model", required=True, help="Original full HF checkpoint used for LoRA training.")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Checkpoint lora_adapter directory (or its adapter_model.safetensors file).",
    )
    parser.add_argument("--output", required=True, help="New full HF checkpoint directory to create.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    args = parser.parse_args()

    merged_path = merge_lora_adapter_into_hf_checkpoint(
        base_model_dir=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(f"Merged deployable HF checkpoint: {merged_path}")


if __name__ == "__main__":
    main()
