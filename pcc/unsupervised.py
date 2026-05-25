"""Label-free calibration: rank candidates by KL(p_ns || p_fs) over label tokens.
Used when we don't have access to gold labels for filtering.

Hybrid (table 5): filter G_full by correctness first, THEN rank by KL.
"""
import random
import torch
import torch.nn.functional as F

from .model_io import free


def _label_token_ids(tokenizer, labels, device):
    ids = []
    for lab in labels:
        toks = tokenizer(" " + str(lab), add_special_tokens=False)["input_ids"]
        if toks:
            ids.append(toks[0])
    return torch.tensor(ids, device=device)


def rank_by_kl(model, tokenizer, task, calib_pool, demo_pool, k_shots, max_seq_len,
               seed=0, use_canonical=False, n_scan=200, verbose=True):
    """Returns sorted list of (index, kl, ns_pred, fs_pred), descending by KL.
    Only meaningful for classification."""
    if task.num_classes is None:
        raise ValueError(f"KL ranking needs a class vocab; {task.name} is generative.")

    device = next(model.parameters()).device
    # collect label set from the head of calib_pool
    label_strings = list(set(task.gold(ex) for ex in calib_pool[:min(200, len(calib_pool))]))
    label_ids = _label_token_ids(tokenizer, label_strings, device)
    if len(label_ids) == 0:
        raise ValueError("couldn't tokenize any label string as a single token")

    pool = calib_pool[:n_scan]
    rows = []
    for i, ex in enumerate(pool):
        per_seed = (seed * 1_000_003 + i) & 0x7FFFFFFF
        rng = random.Random(per_seed)
        if use_canonical:
            shots = list(demo_pool[:k_shots])
        elif hasattr(task, "sample_demos") and task.sample_demos is not None:
            shots = task.sample_demos(demo_pool, k_shots, rng, example=ex)
        else:
            shots = rng.sample(demo_pool, k_shots)

        enc_ns = tokenizer(task.build_prompt(ex, [], False),
                           return_tensors="pt", truncation=True, max_length=max_seq_len).to(device)
        with torch.no_grad():
            out_ns = model(**enc_ns)
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
        rows.append((i, kl, ns_pred, fs_pred))
        free()

        if verbose and (i + 1) % max(1, n_scan // 4) == 0:
            print(f"  kl rank {i+1}/{n_scan}", end="\r")

    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def select_top_kl_balanced(ranked, calib_pool, n_target, num_classes):
    """Take top-KL but spread across predicted-FS labels."""
    by_label = {}
    for (i, kl, ns_pred, fs_pred) in ranked:
        by_label.setdefault(fs_pred, []).append(i)

    per_class = max(1, n_target // max(1, num_classes))
    picked = []
    for lab, idxs in by_label.items():
        picked.extend(idxs[:per_class])
        if len(picked) >= n_target:
            break
    picked = picked[:n_target]
    return [calib_pool[i] for i in picked], picked


def estimate_unsupervised_precision(selected_examples, selected_idx, calib_pool, ranked, task):
    # diagnostic only: how often fs_pred == gold on the selected examples
    fs_lookup = {i: fs_pred for (i, kl, ns_pred, fs_pred) in ranked}
    correct = 0
    for i, ex in zip(selected_idx, selected_examples):
        fs_pred = fs_lookup.get(i)
        if fs_pred is not None and fs_pred == task.gold(ex):
            correct += 1
    return correct / max(1, len(selected_examples))
