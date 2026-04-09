import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from common import ModelBundle, SamplingConfig, crop_cache, get_cache_seq_length, to_cache


def timed_model_call(model, *, device: str, **kwargs):
    # Synchronize around segmented timings so CUDA work is attributed to the
    # correct phase instead of spilling into later measurements.
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    outputs = model(**kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    return outputs, time.time() - t0


@dataclass
class DecodeResult:
    text: str
    generated_ids: List[int]
    stats: Dict[str, float]


@dataclass
class TopKDistribution:
    indices: torch.Tensor
    probs: torch.Tensor

    def token_prob(self, token_id: int) -> float:
        matches = self.indices == token_id
        if not torch.any(matches):
            return 0.0
        return float(self.probs[matches][0].item())

    def sample_token(self, rng: Optional[torch.Generator]) -> torch.Tensor:
        sampled = torch.multinomial(self.probs, num_samples=1, generator=rng)
        return self.indices[sampled].view(1, 1)


def build_topk_distribution(logits: torch.Tensor, k: int, temperature: float) -> TopKDistribution:
    logits = logits.float().squeeze(0) / max(temperature, 1e-5)
    k = max(1, min(k, logits.shape[-1]))
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    topk_probs = torch.softmax(topk_logits, dim=-1)
    return TopKDistribution(indices=topk_indices, probs=topk_probs)


def normalize_logits(logits: torch.Tensor, sampling: SamplingConfig) -> torch.Tensor:
    logits = logits.float()

    if sampling.is_greedy:
        one_hot = torch.full_like(logits, float("-inf"))
        one_hot.scatter_(dim=-1, index=logits.argmax(dim=-1, keepdim=True), value=0.0)
        return torch.softmax(one_hot, dim=-1)

    logits = logits / max(sampling.temperature, 1e-5)

    if sampling.strategy == "top_k":
        k = max(1, min(sampling.top_k, logits.shape[-1]))
        topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        probs = torch.zeros_like(logits)
        probs.scatter_(dim=-1, index=topk_indices, src=topk_probs)
        return probs
    elif sampling.strategy == "top_p":
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > sampling.top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        kept_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        kept_probs = kept_probs / kept_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.zeros_like(logits)
        probs.scatter_(dim=-1, index=sorted_indices, src=kept_probs)
        return probs

    probs = torch.softmax(logits, dim=-1)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def sample_token(probs: torch.Tensor, sampling: SamplingConfig, rng: Optional[torch.Generator]) -> torch.Tensor:
    if sampling.is_greedy:
        return probs.argmax(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=rng)


def greedy_token_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1, keepdim=True)


def corrective_distribution(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    correction = torch.clamp(target_probs - draft_probs, min=0.0)
    total = correction.sum(dim=-1, keepdim=True)
    zero_mask = total <= 1e-12
    if torch.any(zero_mask):
        correction = torch.where(zero_mask, target_probs, correction)
        total = correction.sum(dim=-1, keepdim=True)
    return correction / total.clamp_min(1e-12)


def corrective_distribution_topk(
    target_dist: TopKDistribution,
    draft_dist: TopKDistribution,
) -> TopKDistribution:
    correction = target_dist.probs.clone()
    for draft_index, draft_prob in zip(draft_dist.indices, draft_dist.probs):
        matches = target_dist.indices == draft_index
        if torch.any(matches):
            correction[matches] -= draft_prob

    correction = torch.clamp(correction, min=0.0)
    total = correction.sum()
    if float(total.item()) <= 1e-12:
        correction = target_dist.probs.clone()
        total = correction.sum()
    correction = correction / total.clamp_min(1e-12)
    return TopKDistribution(indices=target_dist.indices, probs=correction)


def init_stats() -> Dict[str, float]:
    return {
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "rejection_events": 0,
        "verify_rounds": 0,
        "accepted_prefix_sum": 0.0,
        "draft_time": 0.0,
        "verify_time": 0.0,
        "rebuild_time": 0.0,
    }


@torch.inference_mode()
def propose_block(
    input_ids: torch.Tensor,
    draft_bundle: ModelBundle,
    sampling: SamplingConfig,
    candidate_length: int,
    draft_past=None,
    next_logits: Optional[torch.Tensor] = None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[List[int], List[torch.Tensor], object, torch.Tensor, Dict[str, float]]:
    stats = {"draft_time": 0.0}

    if draft_past is None:
        outputs, elapsed = timed_model_call(
            draft_bundle.model,
            device=draft_bundle.device,
            input_ids=input_ids,
            use_cache=True,
        )
        stats["draft_time"] += elapsed
        draft_past = to_cache(outputs.past_key_values)
        next_logits = outputs.logits[:, -1, :]

    proposal_ids: List[int] = []
    proposal_probs: List[torch.Tensor] = []
    for _ in range(candidate_length):
        if sampling.is_greedy:
            token = greedy_token_from_logits(next_logits)
        elif sampling.strategy == "top_k":
            topk_dist = build_topk_distribution(next_logits, sampling.top_k, sampling.temperature)
            token = topk_dist.sample_token(rng)
        else:
            probs = normalize_logits(next_logits, sampling)
            token = sample_token(probs, sampling, rng)
        proposal_ids.append(int(token.item()))
        if not sampling.is_greedy:
            if sampling.strategy == "top_k":
                proposal_probs.append(topk_dist)
            else:
                proposal_probs.append(probs.squeeze(0).detach())

        outputs, elapsed = timed_model_call(
            draft_bundle.model,
            device=draft_bundle.device,
            input_ids=token,
            past_key_values=draft_past,
            use_cache=True,
        )
        stats["draft_time"] += elapsed
        draft_past = to_cache(outputs.past_key_values)
        next_logits = outputs.logits[:, -1, :]

    return proposal_ids, proposal_probs, draft_past, next_logits, stats


@torch.inference_mode()
def verify_block(
    input_ids: torch.Tensor,
    proposal_ids: Sequence[int],
    proposal_probs: Sequence[torch.Tensor],
    target_bundle: ModelBundle,
    sampling: SamplingConfig,
    target_past=None,
    prefix_next_logits: Optional[torch.Tensor] = None,
    sample_rng: Optional[torch.Generator] = None,
    accept_rng: Optional[torch.Generator] = None,
) -> Tuple[List[int], torch.Tensor, Optional[object], bool, Dict[str, float]]:
    stats = {"verify_time": 0.0}
    device = target_bundle.device
    proposal_tensor = torch.tensor([proposal_ids], device=device, dtype=input_ids.dtype)

    if target_past is None:
        outputs, elapsed = timed_model_call(
            target_bundle.model,
            device=target_bundle.device,
            input_ids=input_ids,
            use_cache=True,
        )
        stats["verify_time"] += elapsed
        target_past = to_cache(outputs.past_key_values)
        prefix_next_logits = outputs.logits[:, -1, :]

    prefix_cache_len = get_cache_seq_length(target_past)

    verify_outputs, elapsed = timed_model_call(
        target_bundle.model,
        device=target_bundle.device,
        input_ids=proposal_tensor,
        past_key_values=target_past,
        use_cache=True,
    )
    stats["verify_time"] += elapsed
    verify_past = to_cache(verify_outputs.past_key_values)
    verify_logits = verify_outputs.logits.squeeze(0)
    if prefix_next_logits is None:
        raise ValueError("prefix_next_logits must be provided when target_past is supplied.")

    accepted: List[int] = []
    final_token: Optional[torch.Tensor] = None
    all_accepted = True

    def _sampling_rng_for(_probs: torch.Tensor) -> Optional[torch.Generator]:
        return sample_rng

    for idx, draft_token in enumerate(proposal_ids):
        if idx == 0:
            current_logits = prefix_next_logits
        else:
            current_logits = verify_logits[idx - 1 : idx, :]

        if sampling.is_greedy:
            accept = int(current_logits.argmax(dim=-1).item()) == draft_token
        elif sampling.strategy == "top_k":
            target_dist = build_topk_distribution(current_logits, sampling.top_k, sampling.temperature)
            draft_step_dist = proposal_probs[idx]
            p_x = target_dist.token_prob(draft_token)
            q_x = draft_step_dist.token_prob(draft_token)
            accept_prob = min(1.0, p_x / max(q_x, 1e-12))
            random_value = (
                torch.rand(1, device=current_logits.device, generator=accept_rng).item()
                if accept_rng is not None
                else torch.rand(1, device=current_logits.device).item()
            )
            accept = random_value <= accept_prob
        else:
            target_probs = normalize_logits(current_logits, sampling).squeeze(0).detach()
            draft_step_probs = proposal_probs[idx]
            p_x = float(target_probs[draft_token].item())
            q_x = float(draft_step_probs[draft_token].item())
            accept_prob = min(1.0, p_x / max(q_x, 1e-12))
            random_value = (
                torch.rand(1, device=current_logits.device, generator=accept_rng).item()
                if accept_rng is not None
                else torch.rand(1, device=current_logits.device).item()
            )
            accept = random_value <= accept_prob

        if accept:
            accepted.append(draft_token)
            continue

        all_accepted = False
        if sampling.is_greedy:
            final_token = greedy_token_from_logits(current_logits).squeeze(0)
            break
        if sampling.strategy == "top_k":
            target_dist = build_topk_distribution(current_logits, sampling.top_k, sampling.temperature)
            draft_step_dist = proposal_probs[idx]
            correction = corrective_distribution_topk(target_dist, draft_step_dist)
            final_token = correction.sample_token(_sampling_rng_for(correction.probs)).squeeze(0)
            break

        target_probs = normalize_logits(current_logits, sampling).squeeze(0).detach()
        draft_step_probs = proposal_probs[idx]
        correction = corrective_distribution(target_probs.unsqueeze(0), draft_step_probs.unsqueeze(0))
        final_token = sample_token(correction, sampling, _sampling_rng_for(correction)).squeeze(0)
        break

    if all_accepted:
        if sampling.is_greedy:
            final_token = greedy_token_from_logits(verify_logits[-1:, :]).squeeze(0)
        elif sampling.strategy == "top_k":
            last_dist = build_topk_distribution(verify_logits[-1:, :], sampling.top_k, sampling.temperature)
            final_token = last_dist.sample_token(_sampling_rng_for(last_dist.probs)).squeeze(0)
        else:
            last_probs = normalize_logits(verify_logits[-1:, :], sampling)
            final_token = sample_token(last_probs, sampling, _sampling_rng_for(last_probs)).squeeze(0)
        return accepted, final_token.view(1, 1), verify_past, True, stats

    accepted_target_past = crop_cache(verify_past, prefix_cache_len + len(accepted))
    return accepted, final_token.view(1, 1), accepted_target_past, False, stats


def finalize_stats(stats: Dict[str, float], elapsed: float, generated_tokens: int) -> Dict[str, float]:
    proposed = stats["proposed_tokens"]
    accepted = stats["accepted_tokens"]
    verify_rounds = stats["verify_rounds"]
    stats["acceptance_rate"] = accepted / proposed if proposed > 0 else 0.0
    stats["avg_accepted_prefix_length"] = (
        stats["accepted_prefix_sum"] / verify_rounds if verify_rounds > 0 else 0.0
    )
    stats["total_generation_time"] = elapsed
    stats["generated_tokens"] = generated_tokens
    stats["tokens_per_s"] = generated_tokens / elapsed if elapsed > 0 else 0.0
    return stats


@torch.inference_mode()
def speculative_generate(
    prompt: str,
    draft_bundle: ModelBundle,
    target_bundle: ModelBundle,
    max_new_tokens: int,
    candidate_length: int,
    sampling: SamplingConfig,
    seed: Optional[int] = None,
) -> DecodeResult:
    tokenizer = target_bundle.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    target_prompt_ids = prompt_ids.to(target_bundle.device)
    draft_prompt_ids = prompt_ids.to(draft_bundle.device)
    sample_rng = None
    accept_rng = None
    if seed is not None:
        sample_rng = torch.Generator(device=target_bundle.device)
        sample_rng.manual_seed(seed)
        accept_rng = torch.Generator(device=target_bundle.device)
        accept_rng.manual_seed(seed + 1)

    stats = init_stats()

    if target_bundle.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    target_outputs, elapsed = timed_model_call(
        target_bundle.model,
        device=target_bundle.device,
        input_ids=target_prompt_ids,
        use_cache=True,
    )
    target_past = to_cache(target_outputs.past_key_values)
    target_next_logits = target_outputs.logits[:, -1, :]
    stats["verify_time"] += elapsed

    draft_outputs, elapsed = timed_model_call(
        draft_bundle.model,
        device=draft_bundle.device,
        input_ids=draft_prompt_ids,
        use_cache=True,
    )
    draft_past = to_cache(draft_outputs.past_key_values)
    draft_next_logits = draft_outputs.logits[:, -1, :]
    stats["draft_time"] += elapsed

    generated_ids: List[int] = []

    while len(generated_ids) < max_new_tokens:
        proposal_len = min(candidate_length, max_new_tokens - len(generated_ids))
        draft_prefix_cache_len = get_cache_seq_length(draft_past)

        (
            proposal_ids,
            proposal_probs,
            draft_past,
            draft_next_logits,
            draft_stats,
        ) = propose_block(
            input_ids=draft_prompt_ids,
            draft_bundle=draft_bundle,
            sampling=sampling,
            candidate_length=proposal_len,
            draft_past=draft_past,
            next_logits=draft_next_logits,
            rng=sample_rng,
        )

        stats["draft_time"] += draft_stats["draft_time"]
        stats["proposed_tokens"] += len(proposal_ids)
        stats["verify_rounds"] += 1

        accepted_ids, final_token, accepted_target_past, all_accepted, verify_stats = verify_block(
            input_ids=target_prompt_ids,
            proposal_ids=proposal_ids,
            proposal_probs=proposal_probs,
            target_bundle=target_bundle,
            sampling=sampling,
            target_past=target_past,
            prefix_next_logits=target_next_logits,
            sample_rng=sample_rng,
            accept_rng=accept_rng,
        )

        accept_len = len(accepted_ids)
        stats["verify_time"] += verify_stats["verify_time"]
        stats["accepted_tokens"] += accept_len
        stats["rejection_events"] += 0 if all_accepted else 1
        stats["accepted_prefix_sum"] += accept_len

        if accepted_ids:
            generated_ids.extend(accepted_ids)

        if len(generated_ids) >= max_new_tokens:
            break

        generated_ids.append(int(final_token.item()))
        if len(generated_ids) >= max_new_tokens:
            break

        if all_accepted:
            target_past = accepted_target_past
            target_step, elapsed = timed_model_call(
                target_bundle.model,
                device=target_bundle.device,
                input_ids=final_token.to(device=target_bundle.device, dtype=target_prompt_ids.dtype),
                past_key_values=target_past,
                use_cache=True,
            )
            stats["rebuild_time"] += elapsed
            target_past = to_cache(target_step.past_key_values)
            target_next_logits = target_step.logits[:, -1, :]

            draft_step, elapsed = timed_model_call(
                draft_bundle.model,
                device=draft_bundle.device,
                input_ids=final_token.to(device=draft_bundle.device, dtype=draft_prompt_ids.dtype),
                past_key_values=draft_past,
                use_cache=True,
            )
            stats["rebuild_time"] += elapsed
            draft_past = to_cache(draft_step.past_key_values)
            draft_next_logits = draft_step.logits[:, -1, :]
        else:
            target_step, elapsed = timed_model_call(
                target_bundle.model,
                device=target_bundle.device,
                input_ids=final_token.to(device=target_bundle.device, dtype=target_prompt_ids.dtype),
                past_key_values=accepted_target_past,
                use_cache=True,
            )
            stats["rebuild_time"] += elapsed
            target_past = to_cache(target_step.past_key_values)
            target_next_logits = target_step.logits[:, -1, :]

            draft_past = crop_cache(draft_past, draft_prefix_cache_len + accept_len)
            draft_step, elapsed = timed_model_call(
                draft_bundle.model,
                device=draft_bundle.device,
                input_ids=final_token.to(device=draft_bundle.device, dtype=draft_prompt_ids.dtype),
                past_key_values=draft_past,
                use_cache=True,
            )
            stats["rebuild_time"] += elapsed
            draft_past = to_cache(draft_step.past_key_values)
            draft_next_logits = draft_step.logits[:, -1, :]

    if target_bundle.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    generated_ids = generated_ids[:max_new_tokens]
    full_ids = torch.cat(
        [prompt_ids, torch.tensor([generated_ids], dtype=prompt_ids.dtype)],
        dim=1,
    )
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return DecodeResult(text=text, generated_ids=generated_ids, stats=finalize_stats(stats, elapsed, len(generated_ids)))
