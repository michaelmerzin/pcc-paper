"""Model loading, weight snapshot/restore, architecture introspection.

Works with Llama/Mistral/Phi-3/Gemma — anything exposing
`model.model.layers[i].self_attn.o_proj`.
"""
from __future__ import annotations
import gc
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import ExperimentConfig, MODELS


_DTYPE_MAP = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


def free() -> None:
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_dtype(s: str):
    if s not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{s}'")
    return _DTYPE_MAP[s]


def get_layers(model) -> List:
    """Return the list of transformer blocks.

    Works for Llama, Mistral, Phi-3, Gemma — all expose `.model.layers`.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(
        f"Cannot locate transformer blocks on {type(model).__name__}. "
        f"This package supports decoder-only HF models with `.model.layers[i].self_attn.o_proj`."
    )


def load_model_and_tokenizer(cfg: ExperimentConfig):
    """Load model + tokenizer; mutates cfg.num_layers and cfg.hidden_size in place."""
    spec = MODELS[cfg.model_key]
    hf_name = spec["hf_name"]
    dtype_str = cfg.dtype or spec["dtype"]
    cfg.dtype = dtype_str

    print(f"[load] {hf_name}  dtype={dtype_str}  device_map={cfg.device_map}")

    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=_resolve_dtype(dtype_str),
        device_map=cfg.device_map,
    )
    model.eval()

    layers = get_layers(model)
    cfg.num_layers = len(layers)
    cfg.hidden_size = layers[0].self_attn.o_proj.weight.shape[0]
    print(f"[load] num_layers={cfg.num_layers}  hidden={cfg.hidden_size}")

    # Sanity check against expected dimensions
    if cfg.num_layers != spec["expected_layers"]:
        print(f"  WARNING: expected {spec['expected_layers']} layers, got {cfg.num_layers}")
    if cfg.hidden_size != spec["expected_hidden"]:
        print(f"  WARNING: expected hidden={spec['expected_hidden']}, got {cfg.hidden_size}")

    free()
    return model, tok


def snapshot_o_proj(model) -> Dict[int, torch.Tensor]:
    """Snapshot all o_proj weights to CPU (for restoration between edits)."""
    return {
        l: blk.self_attn.o_proj.weight.detach().clone().cpu()
        for l, blk in enumerate(get_layers(model))
    }


def restore_o_proj(model, snap: Dict[int, torch.Tensor]) -> None:
    """Restore o_proj weights from a snapshot."""
    for l, blk in enumerate(get_layers(model)):
        W = blk.self_attn.o_proj.weight
        W.data.copy_(snap[l].to(W.device, dtype=W.dtype))
