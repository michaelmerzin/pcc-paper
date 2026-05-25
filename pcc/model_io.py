"""Model load + weight snapshot/restore. Works for any HF decoder with .model.layers[i].self_attn.o_proj."""
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import MODELS


_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_layers(model):
    # llama / mistral / phi-3 / gemma all expose .model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(f"can't find .model.layers on {type(model).__name__}")


def load_model_and_tokenizer(cfg):
    spec = MODELS[cfg.model_key]
    hf_name = spec["hf_name"]
    dt = cfg.dtype or spec["dtype"]
    cfg.dtype = dt

    print(f"loading {hf_name} ({dt})")

    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=_DTYPES[dt],
        device_map=cfg.device_map,
    )
    model.eval()

    layers = get_layers(model)
    cfg.num_layers = len(layers)
    cfg.hidden_size = layers[0].self_attn.o_proj.weight.shape[0]

    if cfg.num_layers != spec["expected_layers"]:
        print(f"  warning: expected {spec['expected_layers']} layers, got {cfg.num_layers}")
    if cfg.hidden_size != spec["expected_hidden"]:
        print(f"  warning: expected hidden={spec['expected_hidden']}, got {cfg.hidden_size}")

    print(f"  L={cfg.num_layers} d={cfg.hidden_size}")
    free()
    return model, tok


def snapshot_o_proj(model):
    return {l: blk.self_attn.o_proj.weight.detach().clone().cpu()
            for l, blk in enumerate(get_layers(model))}


def restore_o_proj(model, snap):
    for l, blk in enumerate(get_layers(model)):
        W = blk.self_attn.o_proj.weight
        W.data.copy_(snap[l].to(W.device, dtype=W.dtype))
