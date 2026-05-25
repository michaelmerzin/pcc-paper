"""LoReFT (Wu et al. 2024) — low-rank intervention on the residual stream.

  Phi(h) = h + R^T (W h + b - R h)

where R has orthonormal rows. We do NOT use pyreft because its pyvene dep
breaks against newer transformers; this is a from-scratch impl.

Stability notes:
  - Late-quarter layer subset by default (the paper says this captures most of
    the gain at a fraction of the memory).
  - Training in bf16 autocast where available, fp16 otherwise. fp16 sometimes
    underflows on long sequences — we skip NaN steps rather than crashing.
  - Gradient checkpointing on the frozen base because Mistral + MMLU's 2048
    context OOMs on a T4 without it.
"""
import time
import torch
import torch.nn as nn

from ..model_io import get_layers
from .peft import _pairs, _encode


class LoReFTIntervention(nn.Module):
    def __init__(self, hidden_size, rank, dtype=torch.float32):
        super().__init__()
        self.d = hidden_size
        self.r = rank
        self.R = nn.Parameter(torch.empty(rank, hidden_size, dtype=dtype))
        self.W = nn.Parameter(torch.empty(rank, hidden_size, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(rank, dtype=dtype))
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(hidden_size, rank, dtype=dtype))
            self.R.copy_(q.T)
            nn.init.normal_(self.W, std=0.02)

    def forward(self, h):
        # operate in h's dtype — extra fp32 here blows the grad-ckpt budget
        Wh = torch.matmul(h, self.W.T.to(h.dtype))
        Rh = torch.matmul(h, self.R.T.to(h.dtype))
        delta = torch.matmul(Wh + self.b.to(h.dtype) - Rh, self.R.to(h.dtype))
        return h + delta


def _attach_hooks(model, interventions):
    layers = get_layers(model)
    handles = []

    def mk(li, interv):
        def hook(_m, _i, output):
            if isinstance(output, tuple):
                y, rest = output[0], output[1:]
            else:
                y, rest = output, None
            if y.device.type == "meta":
                return output
            first_p = next(interv.parameters())
            if first_p.device != y.device:
                if first_p.device.type == "meta":
                    interv.to_empty(device=y.device)
                else:
                    interv.to(y.device)
            y_new = interv(y)
            return ((y_new,) + rest) if rest is not None else y_new
        return hook

    for li, interv in interventions.items():
        h = layers[li].register_forward_hook(mk(li, interv))
        handles.append(h)
    return handles


def _layer_subset(num_layers, mode):
    if mode == "all":
        return list(range(num_layers))
    if mode == "late_half":
        return list(range(num_layers // 2, num_layers))
    if mode == "late_quarter":
        return list(range(3 * num_layers // 4, num_layers))
    raise ValueError(f"bad layer_subset: {mode!r}")


def train_loreft(base, tok, task, examples, demos, max_seq_len,
                 num_layers, hidden_size,
                 rank=4, lr=5e-4, epochs=3, batch_size=1,
                 layer_subset="late_quarter",
                 train_max_seq_len=256,
                 use_grad_checkpoint=True,
                 verbose=True):
    # NOTE: paper suggests lr~4e-3, but fp16 inference of Mistral underflows
    # there. 5e-4 (2.5x our LoRA lr) is stable with bf16/fp16 autocast on T4.
    layers_to_edit = _layer_subset(num_layers, layer_subset)
    if verbose:
        print(f"  [LoReFT] {layer_subset} -> {len(layers_to_edit)} layers: {layers_to_edit}")

    interventions = {l: LoReFTIntervention(hidden_size, rank, dtype=torch.float32)
                     for l in layers_to_edit}
    layers = get_layers(base)
    for l, interv in interventions.items():
        host_dev = next(layers[l].parameters()).device
        interventions[l] = interv.to(host_dev)

    for p in base.parameters():
        p.requires_grad = False

    if use_grad_checkpoint and hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            base.gradient_checkpointing_enable()
        base.train()
        # we want grad-ckpt recomputation but NOT dropout — base is frozen
        for m in base.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        if verbose:
            print("  [LoReFT] grad ckpt on")
    else:
        base.eval()

    trainable = []
    for interv in interventions.values():
        for p in interv.parameters():
            p.requires_grad = True
            trainable.append(p)
    n_params = sum(p.numel() for p in trainable)
    if verbose:
        print(f"  [LoReFT] interventions={len(interventions)} rank={rank} trainable={n_params:,}")

    handles = _attach_hooks(base, interventions)

    pairs = _pairs(task, examples, demos)
    eff_train_len = min(max_seq_len, train_max_seq_len)
    if verbose:
        print(f"  [LoReFT] training {n_params:,} params on {len(pairs)} pairs "
              f"(lr={lr}, ep={epochs}, train_msl={eff_train_len}, eval_msl={max_seq_len})")

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    device = next(base.parameters()).device
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    t0 = time.time()
    for ep in range(epochs):
        tot, n_done, nan_skipped = 0.0, 0, 0
        for prompt, target in pairs:
            enc, labels = _encode(tok, prompt, target, eff_train_len, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = base(**enc, labels=labels)
                loss = out.loss
            if not torch.isfinite(loss):
                nan_skipped += 1
                continue
            loss.backward()
            grad_nan = any((p.grad is not None and not torch.isfinite(p.grad).all())
                           for p in trainable)
            if grad_nan:
                nan_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tot += loss.item()
            n_done += 1
        if verbose:
            print(f"  [LoReFT] epoch {ep+1}/{epochs} avg_loss={tot/max(1, n_done):.4f} "
                  f"steps={n_done} nan_skip={nan_skipped}")

    wall = time.time() - t0

    for interv in interventions.values():
        interv.eval()
        for p in interv.parameters():
            p.requires_grad = False
    if use_grad_checkpoint and hasattr(base, "gradient_checkpointing_disable"):
        try:
            base.gradient_checkpointing_disable()
        except Exception:
            pass
    base.eval()

    return {"wall_clock_s": wall, "n_train": len(pairs), "n_params": n_params,
            "interventions": interventions, "handles": handles,
            "rank": rank, "layer_subset": layer_subset,
            "layers_edited": layers_to_edit}


def unload_loreft(base, info):
    for h in info.get("handles", []):
        h.remove()
    info["handles"] = []
    info["interventions"] = {}
    return base
