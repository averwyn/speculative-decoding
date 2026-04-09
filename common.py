import os
import csv
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig

    _HAS_BITSANDBYTES_CONFIG = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BITSANDBYTES_CONFIG = False

try:
    from transformers.cache_utils import DynamicCache

    _HAS_DYNAMIC_CACHE = True
except Exception:
    DynamicCache = None
    _HAS_DYNAMIC_CACHE = False


DEFAULT_LOCAL_CACHE_DIR = r"D:\hf_cache"
DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-1.5B"
MODEL_PAIR_PRESETS = {
    "qwen_main": (DEFAULT_DRAFT_MODEL, DEFAULT_TARGET_MODEL),
    "distilgpt2_gpt2": ("distilgpt2", "gpt2"),
    "gpt2_gpt2_medium": ("gpt2", "gpt2-medium"),
    "pythia_410m_1.4b": ("EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b"),
    "pythia_160m_1.4b": ("EleutherAI/pythia-160m", "EleutherAI/pythia-1.4b"),
    "pythia_70m_1.4b": ("EleutherAI/pythia-70m", "EleutherAI/pythia-1.4b"),
    "pythia_31m_1.4b": ("EleutherAI/pythia-31m", "EleutherAI/pythia-1.4b"),
    "pythia_14m_1.4b": ("EleutherAI/pythia-14m", "EleutherAI/pythia-1.4b"),
}


def models_share_tokenizer_family(draft_model: str, target_model: str) -> bool:
    if draft_model == target_model:
        return True

    def _normalize(name: str) -> str:
        return str(name).strip().lower()

    draft = _normalize(draft_model)
    target = _normalize(target_model)

    qwen_prefix = "qwen/qwen2.5-"
    if draft.startswith(qwen_prefix) and target.startswith(qwen_prefix):
        return True

    gpt2_family = {"distilgpt2", "gpt2", "gpt2-medium"}
    if draft in gpt2_family and target in gpt2_family:
        return True

    pythia_prefix = "eleutherai/pythia-"
    if draft.startswith(pythia_prefix) and target.startswith(pythia_prefix):
        return True

    return False


@dataclass
class RuntimeConfig:
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    draft_quantization: str = "none"


@dataclass
class SamplingConfig:
    strategy: str = "greedy"
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 1.0

    @property
    def is_greedy(self) -> bool:
        return self.strategy == "greedy"


@dataclass
class ModelBundle:
    name: str
    tokenizer: object
    model: object
    device: str

    @classmethod
    def load(
        cls,
        model_name: str,
        tokenizer_name: Optional[str],
        device: str,
        runtime: RuntimeConfig,
        quantization: str = "none",
    ) -> "ModelBundle":
        # The experiment framework assumes draft and target can share the target
        # tokenizer family, which holds for the current Qwen and GPT-2 presets.
        source_name = model_name
        tok_name = tokenizer_name or model_name
        if runtime.local_files_only:
            source_name = resolve_local_snapshot_path(model_name, runtime.cache_dir) or model_name
            tok_name = resolve_local_snapshot_path(tok_name, runtime.cache_dir) or tok_name

        kwargs = {}
        if runtime.cache_dir:
            kwargs["cache_dir"] = runtime.cache_dir
        if runtime.local_files_only:
            kwargs["local_files_only"] = True

        tokenizer = AutoTokenizer.from_pretrained(tok_name, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = dict(kwargs)
        use_quantization = quantization != "none"
        if use_quantization:
            if device != "cuda":
                raise ValueError("bitsandbytes quantization requires CUDA in this setup.")
            if not _HAS_BITSANDBYTES_CONFIG:
                raise ValueError("BitsAndBytesConfig is unavailable. Please check the transformers install.")

            if quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            else:
                raise ValueError(f"Unsupported quantization mode: {quantization}")
            model_kwargs["device_map"] = {"": 0}

        model = AutoModelForCausalLM.from_pretrained(source_name, **model_kwargs)
        if not use_quantization:
            model = model.to(device)
        model = model.eval()
        return cls(name=model_name, tokenizer=tokenizer, model=model, device=device)


def resolve_cache_dir(requested_cache_dir: Optional[str] = None) -> Optional[str]:
    if requested_cache_dir:
        return requested_cache_dir

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hub_dir = os.path.join(hf_home, "hub")
        return hub_dir if os.path.isdir(hub_dir) else hf_home

    if os.path.isdir(DEFAULT_LOCAL_CACHE_DIR):
        return DEFAULT_LOCAL_CACHE_DIR

    return None


def repo_id_to_cache_dir_name(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def resolve_local_snapshot_path(name: str, cache_dir: Optional[str]) -> Optional[str]:
    if os.path.isdir(name):
        return name
    if not cache_dir:
        return None

    candidates = [cache_dir, os.path.join(cache_dir, "hub")]
    cache_leaf = repo_id_to_cache_dir_name(name)
    for base_dir in candidates:
        model_cache_dir = os.path.join(base_dir, cache_leaf)
        snapshots_dir = os.path.join(model_cache_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        snapshot_names = sorted(entry.name for entry in os.scandir(snapshots_dir) if entry.is_dir())
        for snapshot_name in reversed(snapshot_names):
            snapshot_dir = os.path.join(snapshots_dir, snapshot_name)
            if os.path.isfile(os.path.join(snapshot_dir, "config.json")):
                return snapshot_dir
    return None


def to_cache(past_key_values):
    if _HAS_DYNAMIC_CACHE and past_key_values is not None and not isinstance(past_key_values, DynamicCache):
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values


def crop_cache(cache, max_length: int):
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return cache
    if isinstance(cache, tuple):
        cropped_items = []
        for item in cache:
            if isinstance(item, tuple):
                cropped_items.append(
                    tuple(
                        tensor[..., :max_length, :].clone() if torch.is_tensor(tensor) else tensor
                        for tensor in item
                    )
                )
            else:
                cropped_items.append(crop_cache(item, max_length))
        return tuple(cropped_items)
    if isinstance(cache, list):
        cropped_items = []
        for item in cache:
            if isinstance(item, tuple):
                cropped_items.append(
                    tuple(
                        tensor[..., :max_length, :].clone() if torch.is_tensor(tensor) else tensor
                        for tensor in item
                    )
                )
            else:
                cropped_items.append(crop_cache(item, max_length))
        return cropped_items
    return cache


def get_cache_seq_length(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        first_layer = cache[0]
        if isinstance(first_layer, tuple) and first_layer and torch.is_tensor(first_layer[0]):
            return int(first_layer[0].shape[-2])
    raise TypeError(f"Unsupported cache type for seq length: {type(cache)!r}")


def build_sampling_config(args) -> SamplingConfig:
    if args.strategy == "greedy":
        return SamplingConfig(strategy="greedy", top_k=0, top_p=1.0, temperature=1.0)
    if args.strategy == "top_k":
        return SamplingConfig(strategy="top_k", top_k=args.top_k, top_p=1.0, temperature=args.temperature)
    return SamplingConfig(strategy="top_p", top_k=0, top_p=args.top_p, temperature=args.temperature)


def build_runtime_config(args) -> RuntimeConfig:
    cache_dir = resolve_cache_dir(args.cache_dir or None)
    return RuntimeConfig(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        draft_quantization=getattr(args, "draft_quantization", "none"),
    )


def parse_int_list(raw_value: str) -> List[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def apply_model_pair_preset(args) -> None:
    preset_name = getattr(args, "model_pair", "") or ""
    if not preset_name:
        return

    try:
        draft_model, target_model = MODEL_PAIR_PRESETS[preset_name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_PAIR_PRESETS))
        raise ValueError(f"Unknown model_pair preset: {preset_name}. Available presets: {choices}") from exc

    args.draft = draft_model
    args.target = target_model


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
