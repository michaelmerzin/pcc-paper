"""Evaluation: batch generation, scoring, correction-set construction.

The correction set G = {i : ŷ_fs_i = y_i ∧ ŷ_ns_i ≠ y_i}  (paper Eq. 1)
filters for examples where few-shot demonstrably helps; this is what
ensures the calibration signal H^ns→Z^fs encodes the SUCCESSFUL part
of the ICL behavior.
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import random
import torch

from .model_io import free


def batch_generate(model, tokenizer, task, examples, shots_list, with_fs: bool,
                   max_seq_len: int, bs: int = 1):
    """Greedy generation in batches. Returns list of parsed predictions."""
    model.eval()
    preds = []
    device = next(model.parameters()).device

    for s in range(0, len(examples), bs):
        batch_ex = examples[s:s + bs]
        batch_sh = shots_list[s:s + bs]
        prompts = [task.build_prompt(ex, sh, with_fs) for ex, sh in zip(batch_ex, batch_sh)]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_seq_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=task.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        preds.extend(task.parse_pred(t) for t in texts)
        free()

    return preds


def evaluate(model, tokenizer, task, examples, shots_per_example,
             max_seq_len: int, bs: int = 1, verbose: bool = False) -> Tuple[float, List, Dict]:
    """Evaluate on `examples`. Returns (accuracy, predictions, stats)."""
    with_fs = any(len(s) > 0 for s in shots_per_example)
    preds = batch_generate(model, tokenizer, task, examples, shots_per_example,
                            with_fs, max_seq_len, bs)
    correct = sum(int(task.score(p, ex)) for p, ex in zip(preds, examples))
    acc = correct / max(1, len(examples))
    parsed = sum(int(p is not None) for p in preds)

    stats = {
        "n": len(examples),
        "correct": correct,
        "acc": acc,
        "parsed": parsed,
        "parse_rate": parsed / max(1, len(examples)),
    }
    if verbose:
        kind = "few-shot" if with_fs else "no-shot"
        print(f"  [eval {kind}] acc={acc*100:.2f}%  parse_rate={stats['parse_rate']*100:.0f}%")
    return acc, preds, stats


def build_shots_per_example(task, train_pool, eval_set, k_shots: int, seed: int,
                             use_canonical: bool = False) -> List[List]:
    """Build per-example shot lists.

    For classification tasks with balanced samplers, every eval example gets
    its own balanced shot set rotated by label (paper §4 setup).
    For canonical-exemplar tasks (GSM8K), the same demos are used for every example.
    """
    if use_canonical:
        demos = list(train_pool[:k_shots])
        return [demos for _ in eval_set]

    out = []
    for i, ex in enumerate(eval_set):
        # deterministic per-example seed
        per_seed = (seed * 1_000_003 + i) & 0x7FFFFFFF
        rng = random.Random(per_seed)
        if hasattr(task, "sample_demos") and task.sample_demos is not None:
            shots = task.sample_demos(train_pool, k_shots, rng, example=ex)
        else:
            shots = rng.sample(train_pool, k_shots)
        out.append(shots)
    return out


def build_correction_set(examples, ns_preds, fs_preds, task) -> List[int]:
    """G_full = {i : few-shot fixes no-shot}  (paper Eq. 1).

    Returns indices into `examples`.
    """
    g_idx = []
    for i, ex in enumerate(examples):
        ns_correct = task.score(ns_preds[i], ex)
        fs_correct = task.score(fs_preds[i], ex)
        if fs_correct and not ns_correct:
            g_idx.append(i)
    return g_idx
