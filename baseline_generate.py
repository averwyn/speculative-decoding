import argparse
import time

import torch

from common import (
    MODEL_PAIR_PRESETS,
    DEFAULT_TARGET_MODEL,
    ModelBundle,
    SamplingConfig,
    apply_model_pair_preset,
    build_runtime_config,
    build_sampling_config,
    to_cache,
)
from spec_decode import normalize_logits, sample_token


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline generation with the target model only.")
    parser.add_argument("--prompt", default="Explain speculative decoding in simple terms:")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--model_pair", choices=sorted(MODEL_PAIR_PRESETS), default="")
    parser.add_argument("--target", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--impl", choices=["hf", "manual"], default="hf")
    parser.add_argument("--strategy", choices=["greedy", "top_k", "top_p"], default="greedy")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def hf_generate(
    prompt: str,
    target_bundle: ModelBundle,
    max_new_tokens: int,
    sampling: SamplingConfig,
    seed: int,
):
    tokenizer = target_bundle.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(target_bundle.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sampling.is_greedy:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = sampling.temperature
        if sampling.strategy == "top_k":
            generate_kwargs["top_k"] = sampling.top_k
        if sampling.strategy == "top_p":
            generate_kwargs["top_p"] = sampling.top_p

    if target_bundle.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        outputs = target_bundle.model.generate(**inputs, **generate_kwargs)
    if target_bundle.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :].tolist()
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "text": text,
        "generated_ids": generated_ids,
        "generated_tokens": int(new_tokens),
        "total_generation_time": elapsed,
        "tokens_per_s": float(new_tokens / elapsed) if elapsed > 0 else 0.0,
    }


@torch.inference_mode()
def manual_autoregressive_generate(
    prompt: str,
    target_bundle: ModelBundle,
    max_new_tokens: int,
    sampling: SamplingConfig,
    seed: int,
):
    tokenizer = target_bundle.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = prompt_ids.to(target_bundle.device)

    rng = torch.Generator(device=target_bundle.device)
    rng.manual_seed(seed)

    if target_bundle.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    outputs = target_bundle.model(input_ids=input_ids, use_cache=True)
    past = to_cache(outputs.past_key_values)
    next_logits = outputs.logits[:, -1, :]

    generated_ids = []
    for _ in range(max_new_tokens):
        probs = normalize_logits(next_logits, sampling)
        token = sample_token(probs, sampling, rng)
        generated_ids.append(int(token.item()))

        outputs = target_bundle.model(input_ids=token, past_key_values=past, use_cache=True)
        past = to_cache(outputs.past_key_values)
        next_logits = outputs.logits[:, -1, :]

    if target_bundle.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    full_ids = torch.cat(
        [prompt_ids, torch.tensor([generated_ids], dtype=prompt_ids.dtype)],
        dim=1,
    )
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return {
        "text": text,
        "generated_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "total_generation_time": elapsed,
        "tokens_per_s": float(len(generated_ids) / elapsed) if elapsed > 0 else 0.0,
    }


def baseline_generate(
    prompt: str,
    target_bundle: ModelBundle,
    max_new_tokens: int,
    sampling: SamplingConfig,
    seed: int,
    impl: str = "hf",
):
    if impl == "hf":
        return hf_generate(
            prompt=prompt,
            target_bundle=target_bundle,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            seed=seed,
        )
    return manual_autoregressive_generate(
        prompt=prompt,
        target_bundle=target_bundle,
        max_new_tokens=max_new_tokens,
        sampling=sampling,
        seed=seed,
    )


def main():
    args = parse_args()
    apply_model_pair_preset(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = build_runtime_config(args)
    sampling = build_sampling_config(args)
    target_bundle = ModelBundle.load(args.target, args.target, device, runtime)

    result = baseline_generate(
        prompt=args.prompt,
        target_bundle=target_bundle,
        max_new_tokens=args.max_new_tokens,
        sampling=sampling,
        seed=args.seed,
        impl=args.impl,
    )

    print(f"target={args.target} strategy={sampling.strategy} device={device} impl={args.impl}")
    print(
        f"generated_tokens={result['generated_tokens']} "
        f"time={result['total_generation_time']:.4f}s "
        f"tokens/s={result['tokens_per_s']:.2f}"
    )
    print("\n[text]")
    print(result["text"])


if __name__ == "__main__":
    main()
