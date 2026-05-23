"""Unsupervised KL-divergence ranking for label-free calibration (paper §3.5, Table 4-5).

When labels are unavailable, we cannot construct G via Eq. 1. Instead:

    KL_i = D_KL(p^ns_i ‖ p^fs_i)                              (paper Eq. 5)

We rank candidate examples by KL_i and pick the top |G|, balancing by the
predicted few-shot label. The intuition: large KL = the few-shot prompt
moves the label distribution a lot, which is a proxy for "ICL helped here."

Hybrid variant (Table 5):
    1. First filter by gold correctness (need labels, but only for the filter).
    2. Within the correct subset, rank by KL and take the top n.
    Result: strictly cleaner activation shifts → stronger ridge updates.
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from collections import Counter
import random

from .model_io import free


def _classification_label_token_ids(tokenizer, label_strings: List[str], device) -> torch.Tensor:
    """Tokenize each label as a leading-space single token (first token only)."""
    ids = []
    for lab in label_strings:
        toks = tokenizer(" " + str(lab), add_special_tokens=False)["input_ids"]
        if toks:
            ids.append(toks[0])
    return torch.tensor(ids, device=device)


def rank_by_kl(
    model, tokenizer, task, calib_pool, demo_pool,
    k_shots: int, max_seq_len: int,
    seed: int = 0,
    use_canonical: bool = False,
    n_scan: int = 200,
    verbose: bool = True,
) -> List[Tuple[int, float, str, str]]:
    """Score each example in calib_pool by KL between p^ns and p^fs over label tokens.

    Only meaningful for classification tasks (need a fixed label vocabulary).

    Returns:
        List of (index, kl, ns_pred, fs_pred) sorted descending by kl.
    """
    if task.num_classes is None:
        raise ValueError(f"KL ranking is for classification tasks; {task.name} is generative.")

    device = next(model.parameters()).device
    label_strings = list(set(task.gold(ex) for ex in calib_pool[:min(200, len(calib_pool))]))
    label_ids = _classification_label_token_ids(tokenizer, label_strings, device)
    if len(label_ids) == 0:
        raise ValueError("Could not tokenize any label strings as single tokens.")

    pool = calib_pool[:n_scan]
    out_rows = []

    for i, ex in enumerate(pool):
        # Per-example deterministic shots
        per_seed = (seed * 1_000_003 + i) & 0x7FFFFFFF
        rng = random.Random(per_seed)
        if use_canonical:
            shots = list(demo_pool[:k_shots])
        elif hasattr(task, "sample_demos") and task.sample_demos is not None:
            shots = task.sample_demos(demo_pool, k_shots, rng, example=ex)
        else:
            shots = rng.sample(demo_pool, k_shots)

        # No-shot logits at last position
        enc_ns = tokenizer(task.build_prompt(ex, [], False),
                            return_tensors="pt", truncation=True, max_length=max_seq_len).to(device)
        with torch.no_grad():
            out_ns = model(**enc_ns)
        # Few-shot logits at last position
        enc_fs = tokenizer(task.build_prompt(ex, shots, True),
                            return_tensors="pt", truncation=True, max_length=max_seq_len).to(device)
        with torch.no_grad():
            out_fs = model(**enc_fs)

        logits_ns = out_ns.logits[0, -1][label_ids].float()
        logits_fs = out_fs.logits[0, -1][label_ids].float()
        p_ns = F.softmax(logits_ns, dim=-1).clamp(min=1e-8)
        p_fs = F.softmax(logits_fs, dim=-1).clamp(min=1e-8)
        kl = F.kl_div(p_ns.log(), p_fs, reduction="sum").item()

        ns_pred = label_strings[logits_ns.argmax().item()]
        fs_pred = label_strings[logits_fs.argmax().item()]
        out_rows.append((i, kl, ns_pred, fs_pred))
        free()

        if verbose and (i + 1) % max(1, n_scan // 4) == 0:
            print(f"  [kl rank] {i+1}/{n_scan}", end="\r")

    out_rows.sort(key=lambda r: r[1], reverse=True)
    return out_rows


def select_top_kl_balanced(
    ranked: List[Tuple[int, float, str, str]],
    calib_pool: List,
    n_target: int,
    num_classes: int,
) -> Tuple[List, List]:
    """Take top-KL examples, balanced by predicted few-shot label.

    Returns:
        (selected_examples, selected_indices_into_calib_pool)
    """
    by_label: Dict[str, List] = {}
    for (i, kl, ns_pred, fs_pred) in ranked:
        by_label.setdefault(fs_pred, []).append(i)

    per_class = max(1, n_target // max(1, num_classes))
    selected_idx = []
    for lab, idxs in by_label.items():
        selected_idx.extend(idxs[:per_class])
        if len(selected_idx) >= n_target:
            break
    selected_idx = selected_idx[:n_target]
    selected_examples = [calib_pool[i] for i in selected_idx]
    return selected_examples, selected_idx


def estimate_unsupervised_precision(
    selected_examples: List, selected_idx: List[int],
    calib_pool: List, ranked: List[Tuple[int, float, str, str]], task,
) -> float:
    """Measure post-hoc how often the (few-shot pred = gold) on the selected set.

    Not used in selection — purely diagnostic, matches the 'Prec.' column of Table 4.
    """
    idx_to_fs_pred = {i: fs_pred for (i, kl, ns_pred, fs_pred) in ranked}
    correct = 0
    for i, ex in zip(selected_idx, selected_examples):
        fs_pred = idx_to_fs_pred.get(i)
        if fs_pred is not None and fs_pred == task.gold(ex):
            correct += 1
    return correct / max(1, len(selected_examples))
