"""PEFT baselines for Table 2 (LoRA, IA³, Prompt Tuning, Prefix Tuning).

All four methods are trained on the SAME correction-set protocol that PCC
uses, for an apples-to-apples comparison:
  - Same set of examples (G_full or CALIB ⊆ G_full)
  - Same supervision (no-shot prompt → gold label)
  - Same number of epochs (3, paper §5.1 line 442-443)

The instability of these methods at this data scale is what motivates PCC
(paper §5.1 line 452-458: 'On MRPC, LoRA drops to 34.31%, ... Prefix Tuning
collapses to 0.00%, while PCC reaches 80.88%').
"""
from __future__ import annotations
import time
from typing import List, Dict, Any
import torch


def _build_training_pairs(task, examples, demos):
    """Build (no_shot_prompt, gold_string) pairs.

    We train on no-shot prompts because the edited model is deployed in
    no-shot mode — direct match between training and inference.
    """
    pairs = []
    for ex, _ in zip(examples, demos):
        prompt = task.build_prompt(ex, [], with_fewshot=False)
        target = str(task.gold(ex))
        if target:
            pairs.append((prompt, target))
    if not pairs:
        raise ValueError("No training pairs (check task.gold returns non-empty strings).")
    return pairs


def _encode_pair(tokenizer, prompt, target, max_seq_len, device):
    full = prompt + " " + target
    enc = tokenizer(full, return_tensors="pt", truncation=True,
                    max_length=max_seq_len).to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_seq_len)["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :prompt_len] = -100
    return enc, labels


def _train_peft(model, tokenizer, task, examples, demos, max_seq_len: int,
                 lr: float, epochs: int, peft_label: str, verbose: bool = True):
    pairs = _build_training_pairs(task, examples, demos)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  [{peft_label}] training {n_params:,} params on {len(pairs)} pairs (lr={lr}, ep={epochs})")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device
    model.train()
    t0 = time.time()

    for ep in range(epochs):
        total = 0.0
        for i, (prompt, target) in enumerate(pairs):
            enc, labels = _encode_pair(tokenizer, prompt, target, max_seq_len, device)
            out = model(**enc, labels=labels)
            optim.zero_grad()
            out.loss.backward()
            optim.step()
            total += out.loss.item()
        if verbose:
            print(f"  [{peft_label}] epoch {ep+1}/{epochs}  avg_loss={total/len(pairs):.4f}")

    wall = time.time() - t0
    model.eval()
    return {"wall_clock_s": wall, "n_train": len(pairs), "n_params": n_params}


# ============================================================================ #
#   LoRA                                                                       #
# ============================================================================ #
def train_lora(base_model, tokenizer, task, examples, demos, max_seq_len: int,
               rank: int = 8, lr: float = 1e-4, epochs: int = 3,
               target_modules=("o_proj",), verbose: bool = True):
    """Standard LoRA on o_proj (paper Table 2: r=8, 3 epochs)."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise RuntimeError("Install peft: pip install peft")

    cfg = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    peft_model = get_peft_model(base_model, cfg)
    if verbose:
        peft_model.print_trainable_parameters()
    info = _train_peft(peft_model, tokenizer, task, examples, demos, max_seq_len,
                       lr=lr, epochs=epochs, peft_label="LoRA", verbose=verbose)
    info["peft_model"] = peft_model
    return info


# ============================================================================ #
#   IA³                                                                        #
# ============================================================================ #
def train_ia3(base_model, tokenizer, task, examples, demos, max_seq_len: int,
              lr: float = 1e-3, epochs: int = 3,
              target_modules=("o_proj",), feedforward_modules=(), verbose: bool = True):
    """IA³ multiplicative scaling on o_proj (paper Table 2: 3 epochs)."""
    try:
        from peft import IA3Config, get_peft_model, TaskType
    except ImportError:
        raise RuntimeError("Install peft: pip install peft")

    cfg = IA3Config(
        target_modules=list(target_modules),
        feedforward_modules=list(feedforward_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(base_model, cfg)
    if verbose:
        peft_model.print_trainable_parameters()
    info = _train_peft(peft_model, tokenizer, task, examples, demos, max_seq_len,
                       lr=lr, epochs=epochs, peft_label="IA3", verbose=verbose)
    info["peft_model"] = peft_model
    return info


# ============================================================================ #
#   Prompt Tuning                                                              #
# ============================================================================ #
def train_prompt_tuning(base_model, tokenizer, task, examples, demos, max_seq_len: int,
                         n_virtual_tokens: int = 20, lr: float = 3e-3, epochs: int = 3,
                         verbose: bool = True):
    """Soft prompt tuning (Lester et al., paper Table 2: 3 epochs)."""
    try:
        from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
    except ImportError:
        raise RuntimeError("Install peft: pip install peft")

    cfg = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=n_virtual_tokens,
        tokenizer_name_or_path=None,
    )
    peft_model = get_peft_model(base_model, cfg)
    if verbose:
        peft_model.print_trainable_parameters()
    info = _train_peft(peft_model, tokenizer, task, examples, demos, max_seq_len,
                       lr=lr, epochs=epochs, peft_label="PromptTuning", verbose=verbose)
    info["peft_model"] = peft_model
    return info


# ============================================================================ #
#   Prefix Tuning                                                              #
# ============================================================================ #
def train_prefix_tuning(base_model, tokenizer, task, examples, demos, max_seq_len: int,
                         n_virtual_tokens: int = 20, lr: float = 3e-3, epochs: int = 3,
                         verbose: bool = True):
    """Prefix tuning (Li & Liang, paper Table 2: 3 epochs).

    Note: Prefix tuning is notoriously unstable on instruction-tuned models;
    Table 2 shows it collapsing to 0.00% on MRPC and 4.82% on SST-2.
    """
    try:
        from peft import PrefixTuningConfig, get_peft_model, TaskType
    except ImportError:
        raise RuntimeError("Install peft: pip install peft")

    cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=n_virtual_tokens,
    )
    peft_model = get_peft_model(base_model, cfg)
    if verbose:
        peft_model.print_trainable_parameters()
    info = _train_peft(peft_model, tokenizer, task, examples, demos, max_seq_len,
                       lr=lr, epochs=epochs, peft_label="PrefixTuning", verbose=verbose)
    info["peft_model"] = peft_model
    return info


def unload_peft(peft_model):
    """Merge adapters back into base model (LoRA/IA3) and return base."""
    if hasattr(peft_model, "unload"):
        return peft_model.unload()
    return peft_model
