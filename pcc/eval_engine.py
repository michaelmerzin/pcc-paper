import random
import torch

from .model_io import free


def batch_generate(model, tokenizer, task, examples, shots_list, with_fs, max_seq_len, bs=1):
    model.eval()
    device = next(model.parameters()).device
    preds = []

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


def evaluate(model, tokenizer, task, examples, shots_per_example, max_seq_len, bs=1, verbose=False):
    with_fs = any(len(s) > 0 for s in shots_per_example)
    preds = batch_generate(model, tokenizer, task, examples, shots_per_example,
                           with_fs, max_seq_len, bs)
    correct = sum(int(task.score(p, ex)) for p, ex in zip(preds, examples))
    n = len(examples)
    acc = correct / n if n else 0.0
    parsed = sum(int(p is not None) for p in preds)
    stats = {"n": n, "correct": correct, "acc": acc,
             "parsed": parsed, "parse_rate": parsed / n if n else 0.0}
    if verbose:
        print(f"  eval {'fs' if with_fs else 'ns'}: acc={acc*100:.2f} parse={stats['parse_rate']*100:.0f}%")
    return acc, preds, stats


def build_shots_per_example(task, train_pool, eval_set, k_shots, seed, use_canonical=False):
    # For canonical-exemplar tasks (gsm8k Wei et al. shots) every example gets the same demos.
    if use_canonical:
        demos = list(train_pool[:k_shots])
        return [demos for _ in eval_set]

    out = []
    for i, ex in enumerate(eval_set):
        per_seed = (seed * 1_000_003 + i) & 0x7FFFFFFF
        rng = random.Random(per_seed)
        if hasattr(task, "sample_demos") and task.sample_demos is not None:
            shots = task.sample_demos(train_pool, k_shots, rng, example=ex)
        else:
            shots = rng.sample(train_pool, k_shots)
        out.append(shots)
    return out


def build_correction_set(examples, ns_preds, fs_preds, task):
    # examples where FS got it right but NS didn't
    g = []
    for i, ex in enumerate(examples):
        if task.score(fs_preds[i], ex) and not task.score(ns_preds[i], ex):
            g.append(i)
    return g
