import argparse
import os
import statistics as stats
from typing import Dict, List, Sequence, Tuple

import torch

from baseline_generate import baseline_generate
from common import (
    DEFAULT_DRAFT_MODEL,
    DEFAULT_TARGET_MODEL,
    ModelBundle,
    MODEL_PAIR_PRESETS,
    apply_model_pair_preset,
    build_runtime_config,
    build_sampling_config,
    models_share_tokenizer_family,
    parse_int_list,
    write_csv,
)
from spec_decode import speculative_generate


PROMPT_PRESETS: Dict[str, List[Tuple[str, str]]] = {
    "default": [
        ("explain_sd_simple", "Explain speculative decoding in simple terms."),
    ],
    "analysis": [
        ("explain_sd_simple", "Explain speculative decoding in simple terms."),
        ("sampling_diff", "Compare greedy decoding, top-k sampling, and top-p sampling in one paragraph."),
        ("draft_target_interaction", "Explain how draft models and target models interact during speculative decoding."),
        ("low_acceptance_reason", "Why can a low acceptance rate reduce the speedup of speculative decoding?"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark single-level speculative decoding across strategies and k.")
    parser.add_argument("--prompt", default="Explain speculative decoding in simple terms:")
    parser.add_argument("--prompt_preset", default="")
    parser.add_argument("--prompt_file", default="")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--baseline_impls", default="hf,manual")
    parser.add_argument("--ks", default="1,2,4,6,8")
    parser.add_argument("--strategies", default="greedy,top_k,top_p")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model_pair", choices=sorted(MODEL_PAIR_PRESETS), default="")
    parser.add_argument("--draft", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--draft_quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--out_dir", default="results")
    return parser.parse_args()


def load_prompts(args) -> List[Tuple[str, str]]:
    if args.prompt_file:
        prompts = []
        with open(args.prompt_file, "r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if "\t" in line:
                    prompt_id, prompt = line.split("\t", 1)
                else:
                    prompt_id = f"prompt_{index + 1}"
                    prompt = line
                prompts.append((prompt_id, prompt))
        return prompts

    if args.prompt_preset:
        if args.prompt_preset not in PROMPT_PRESETS:
            choices = ", ".join(sorted(PROMPT_PRESETS))
            raise ValueError(f"Unknown prompt_preset: {args.prompt_preset}. Available presets: {choices}")
        return PROMPT_PRESETS[args.prompt_preset]

    return [("prompt_1", args.prompt)]
def summarize_rows(rows: List[Dict[str, object]], numeric_keys: Sequence[str]) -> Dict[str, object]:
    summary = {"n": len(rows)}
    for key in numeric_keys:
        values = []
        for row in rows:
            value = row.get(key, "")
            if value == "":
                continue
            values.append(float(value))

        if not values:
            summary[f"{key}_mean"] = ""
            summary[f"{key}_std"] = ""
        elif len(values) == 1:
            summary[f"{key}_mean"] = values[0]
            summary[f"{key}_std"] = 0.0
        else:
            summary[f"{key}_mean"] = stats.mean(values)
            summary[f"{key}_std"] = stats.pstdev(values)
    return summary


def make_sampling_args(base_args, strategy: str):
    namespace = argparse.Namespace(**vars(base_args))
    namespace.strategy = strategy
    return namespace


def baseline_row(prompt_id: str, prompt: str, strategy: str, report: Dict[str, object], args) -> Dict[str, object]:
    total_time = report["total_generation_time"]
    return {
        "prompt_id": prompt_id,
        "prompt_chars": len(prompt),
        "mode": "baseline",
        "baseline_impl": report["baseline_impl"],
        "draft_model": "",
        "target_model": args.target,
        "strategy": strategy,
        "candidate_length_k": "",
        "top_k": args.top_k if strategy == "top_k" else "",
        "top_p": args.top_p if strategy == "top_p" else "",
        "temperature": 1.0 if strategy == "greedy" else args.temperature,
        "generated_tokens": report["generated_tokens"],
        "acceptance_rate": "",
        "avg_accepted_prefix_length": "",
        "total_generation_time": total_time,
        "tokens_per_s": report["tokens_per_s"],
        "proposed_tokens": "",
        "accepted_tokens": "",
        "rejection_events": "",
        "verify_rounds": "",
        "draft_time": 0.0,
        "verify_time": total_time,
        "rebuild_time": 0.0,
        "draft_time_ratio": 0.0,
        "verify_time_ratio": 1.0 if total_time > 0 else 0.0,
        "rebuild_time_ratio": 0.0,
    }


def speculative_row(
    prompt_id: str,
    prompt: str,
    strategy: str,
    candidate_length: int,
    report: Dict[str, object],
    args,
) -> Dict[str, object]:
    total_time = report["total_generation_time"]
    draft_time = report.get("draft_time", 0.0)
    verify_time = report.get("verify_time", 0.0)
    rebuild_time = report.get("rebuild_time", 0.0)
    return {
        "prompt_id": prompt_id,
        "prompt_chars": len(prompt),
        "mode": "speculative",
        "baseline_impl": "manual",
        "draft_model": args.draft,
        "target_model": args.target,
        "strategy": strategy,
        "candidate_length_k": candidate_length,
        "top_k": args.top_k if strategy == "top_k" else "",
        "top_p": args.top_p if strategy == "top_p" else "",
        "temperature": 1.0 if strategy == "greedy" else args.temperature,
        "generated_tokens": report["generated_tokens"],
        "acceptance_rate": report["acceptance_rate"],
        "avg_accepted_prefix_length": report["avg_accepted_prefix_length"],
        "total_generation_time": total_time,
        "tokens_per_s": report["tokens_per_s"],
        "proposed_tokens": report["proposed_tokens"],
        "accepted_tokens": report["accepted_tokens"],
        "rejection_events": report["rejection_events"],
        "verify_rounds": report["verify_rounds"],
        "draft_time": draft_time,
        "verify_time": verify_time,
        "rebuild_time": rebuild_time,
        "draft_time_ratio": draft_time / total_time if total_time > 0 else 0.0,
        "verify_time_ratio": verify_time / total_time if total_time > 0 else 0.0,
        "rebuild_time_ratio": rebuild_time / total_time if total_time > 0 else 0.0,
    }


def main():
    args = parse_args()
    apply_model_pair_preset(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = build_runtime_config(args)
    prompts = load_prompts(args)
    baseline_impls = [item.strip() for item in args.baseline_impls.split(",") if item.strip()]
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    k_values = parse_int_list(args.ks)

    if not models_share_tokenizer_family(args.draft, args.target):
        raise ValueError(
            "This implementation assumes draft and target share the target tokenizer family, "
            f"but got draft={args.draft!r} and target={args.target!r}."
        )

    target_bundle = ModelBundle.load(args.target, args.target, device, runtime, quantization="none")
    draft_bundle = ModelBundle.load(
        args.draft,
        args.target,
        device,
        runtime,
        quantization=args.draft_quantization,
    )
    raw_rows: List[Dict[str, object]] = []
    for prompt_index, (prompt_id, prompt) in enumerate(prompts):
        for strategy in strategies:
            sampling = build_sampling_config(make_sampling_args(args, strategy))

            for baseline_impl in baseline_impls:
                for run_index in range(args.warmup + args.repeats):
                    seed = args.seed + prompt_index * 1000 + run_index
                    baseline_report = baseline_generate(
                        prompt=prompt,
                        target_bundle=target_bundle,
                        max_new_tokens=args.max_new_tokens,
                        sampling=sampling,
                        seed=seed,
                        impl=baseline_impl,
                    )
                    baseline_report["baseline_impl"] = baseline_impl

                    if run_index < args.warmup:
                        continue

                    raw_rows.append(baseline_row(prompt_id, prompt, strategy, baseline_report, args))
                    print(
                        f"[baseline] prompt={prompt_id} strategy={strategy} impl={baseline_impl} "
                        f"rep={run_index - args.warmup + 1}/{args.repeats} "
                        f"tps={baseline_report['tokens_per_s']:.2f}"
                    )

            for candidate_length in k_values:
                for run_index in range(args.warmup + args.repeats):
                    seed = args.seed + prompt_index * 1000 + run_index
                    result = speculative_generate(
                        prompt=prompt,
                        draft_bundle=draft_bundle,
                        target_bundle=target_bundle,
                        max_new_tokens=args.max_new_tokens,
                        candidate_length=candidate_length,
                        sampling=sampling,
                        seed=seed,
                    )

                    if run_index < args.warmup:
                        continue

                    raw_rows.append(
                        speculative_row(
                            prompt_id=prompt_id,
                            prompt=prompt,
                            strategy=strategy,
                            candidate_length=candidate_length,
                            report=result.stats,
                            args=args,
                        )
                    )
                    print(
                        f"[spec] prompt={prompt_id} strategy={strategy} k={candidate_length} "
                        f"rep={run_index - args.warmup + 1}/{args.repeats} "
                        f"acc={result.stats['acceptance_rate']:.4f} "
                        f"tps={result.stats['tokens_per_s']:.2f} "
                        f"draft={result.stats['draft_time']:.4f}s "
                        f"verify={result.stats['verify_time']:.4f}s "
                        f"rebuild={result.stats['rebuild_time']:.4f}s"
                    )

    raw_path = os.path.join(args.out_dir, "bench_raw.csv")
    write_csv(raw_path, raw_rows)

    numeric_keys = [
        "prompt_chars",
        "generated_tokens",
        "acceptance_rate",
        "avg_accepted_prefix_length",
        "total_generation_time",
        "tokens_per_s",
        "proposed_tokens",
        "accepted_tokens",
        "rejection_events",
        "verify_rounds",
        "draft_time",
        "verify_time",
        "rebuild_time",
        "draft_time_ratio",
        "verify_time_ratio",
        "rebuild_time_ratio",
    ]

    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    overall_groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in raw_rows:
        per_prompt_key = (
            row["prompt_id"],
            row["mode"],
            row["baseline_impl"],
            row["draft_model"],
            row["target_model"],
            row["strategy"],
            row["candidate_length_k"],
            row["top_k"],
            row["top_p"],
            row["temperature"],
        )
        groups.setdefault(per_prompt_key, []).append(row)

        overall_key = per_prompt_key[1:]
        overall_groups.setdefault(overall_key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, rows in groups.items():
        prompt_id, mode, baseline_impl, draft_model, target_model, strategy, candidate_length, top_k, top_p, temperature = key
        summary_rows.append(
            {
                "prompt_id": prompt_id,
                "mode": mode,
                "baseline_impl": baseline_impl,
                "draft_model": draft_model,
                "target_model": target_model,
                "strategy": strategy,
                "candidate_length_k": candidate_length,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                **summarize_rows(rows, numeric_keys),
            }
        )

    for key, rows in overall_groups.items():
        mode, baseline_impl, draft_model, target_model, strategy, candidate_length, top_k, top_p, temperature = key
        summary_rows.append(
            {
                "prompt_id": "ALL",
                "mode": mode,
                "baseline_impl": baseline_impl,
                "draft_model": draft_model,
                "target_model": target_model,
                "strategy": strategy,
                "candidate_length_k": candidate_length,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                **summarize_rows(rows, numeric_keys),
            }
        )

    summary_path = os.path.join(args.out_dir, "bench_summary.csv")
    write_csv(summary_path, summary_rows)

    print("\n== Done ==")
    print(f"raw_csv={raw_path}")
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
