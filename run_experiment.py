import argparse
import os
from typing import Dict, List

import torch

from common import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    ModelBundle,
    MODEL_PAIR_PRESETS,
    SamplingConfig,
    apply_model_pair_preset,
    build_runtime_config,
    build_sampling_config,
    parse_int_list,
    write_csv,
)
from spec_decode import speculative_generate


def parse_args():
    parser = argparse.ArgumentParser(description="Single-level speculative decoding experiment runner.")
    parser.add_argument("--prompt", default="Explain speculative decoding in simple terms:")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--ks", default="1,2,4,6,8")
    parser.add_argument("--strategy", choices=["greedy", "top_k", "top_p"], default="greedy")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model_pair", choices=sorted(MODEL_PAIR_PRESETS), default="")
    parser.add_argument("--draft", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--out_dir", default="results")
    return parser.parse_args()
def format_summary_row(k_value: int, result, args, sampling: SamplingConfig) -> Dict[str, object]:
    stats = result.stats
    return {
        "draft_model": args.draft,
        "target_model": args.target,
        "strategy": sampling.strategy,
        "top_k": sampling.top_k,
        "top_p": sampling.top_p,
        "temperature": sampling.temperature,
        "candidate_length_k": k_value,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": stats["generated_tokens"],
        "acceptance_rate": stats["acceptance_rate"],
        "avg_accepted_prefix_length": stats["avg_accepted_prefix_length"],
        "total_generation_time": stats["total_generation_time"],
        "tokens_per_s": stats["tokens_per_s"],
        "proposed_tokens": stats["proposed_tokens"],
        "accepted_tokens": stats["accepted_tokens"],
        "rejection_events": stats["rejection_events"],
        "verify_rounds": stats["verify_rounds"],
    }


def main():
    args = parse_args()
    apply_model_pair_preset(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = build_runtime_config(args)
    sampling = build_sampling_config(args)
    k_values = parse_int_list(args.ks)

    target_bundle = ModelBundle.load(args.target, args.target, device, runtime)
    draft_bundle = ModelBundle.load(args.draft, args.target, device, runtime)

    rows: List[Dict[str, object]] = []
    for index, candidate_length in enumerate(k_values):
        result = speculative_generate(
            prompt=args.prompt,
            draft_bundle=draft_bundle,
            target_bundle=target_bundle,
            max_new_tokens=args.max_new_tokens,
            candidate_length=candidate_length,
            sampling=sampling,
            seed=args.seed + index,
        )
        row = format_summary_row(candidate_length, result, args, sampling)
        rows.append(row)

        print(
            f"k={candidate_length} strategy={sampling.strategy} "
            f"acceptance_rate={row['acceptance_rate']:.4f} "
            f"avg_prefix={row['avg_accepted_prefix_length']:.4f} "
            f"time={row['total_generation_time']:.4f}s "
            f"tokens/s={row['tokens_per_s']:.2f}"
        )

    output_path = os.path.join(args.out_dir, "single_level_speculative_results.csv")
    write_csv(output_path, rows)
    print(f"\nresults_csv={output_path}")


if __name__ == "__main__":
    main()
