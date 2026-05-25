"""DoRA (Liu et al. 2024) — LoRA with separately-learned magnitude/direction.
Needs peft>=0.10 (use_dora=True).
"""
import time
import torch

from .peft import _pairs, _encode


def train_dora(base, tok, task, examples, demos, max_seq_len,
               rank=8, lr=2e-4, epochs=3,
               target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
               verbose=True):
    from peft import LoraConfig, get_peft_model, TaskType
    try:
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                         target_modules=list(target_modules),
                         task_type=TaskType.CAUSAL_LM, bias="none",
                         use_dora=True)
    except TypeError:
        raise RuntimeError("peft too old for use_dora. install peft>=0.10")

    pm = get_peft_model(base, cfg)
    if verbose:
        pm.print_trainable_parameters()

    pairs = _pairs(task, examples, demos)
    n_params = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    if verbose:
        print(f"  [DoRA] {n_params:,} params on {len(pairs)} pairs (lr={lr}, ep={epochs})")

    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    device = next(pm.parameters()).device
    pm.train()
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for prompt, target in pairs:
            enc, labels = _encode(tok, prompt, target, max_seq_len, device)
            out = pm(**enc, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += out.loss.item()
        if verbose:
            print(f"  [DoRA] epoch {ep+1}/{epochs} avg_loss={tot/len(pairs):.4f}")

    pm.eval()
    return {"wall_clock_s": time.time() - t0,
            "n_train": len(pairs),
            "n_params": n_params,
            "peft_model": pm,
            "rank": rank}
