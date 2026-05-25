"""LoRA, IA3, Prompt Tuning, Prefix Tuning. All trained on the same G_full as PCC
with no-shot prompt -> gold label, 3 epochs.
"""
import time
import torch


def _pairs(task, examples, demos):
    out = []
    for ex, _ in zip(examples, demos):
        prompt = task.build_prompt(ex, [], with_fewshot=False)
        target = str(task.gold(ex))
        if target:
            out.append((prompt, target))
    if not out:
        raise ValueError("no training pairs — task.gold returning empty?")
    return out


def _encode(tokenizer, prompt, target, max_seq_len, device):
    full = prompt + " " + target
    enc = tokenizer(full, return_tensors="pt", truncation=True,
                    max_length=max_seq_len).to(device)
    p_len = tokenizer(prompt, return_tensors="pt", truncation=True,
                      max_length=max_seq_len)["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :p_len] = -100
    return enc, labels


def _train(peft_model, tokenizer, task, examples, demos, max_seq_len,
           lr, epochs, tag, verbose=True):
    pairs = _pairs(task, examples, demos)
    n_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if verbose:
        print(f"  [{tag}] {n_params:,} params, {len(pairs)} pairs (lr={lr}, ep={epochs})")

    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    device = next(peft_model.parameters()).device
    peft_model.train()
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for prompt, target in pairs:
            enc, labels = _encode(tokenizer, prompt, target, max_seq_len, device)
            out = peft_model(**enc, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            tot += out.loss.item()
        if verbose:
            print(f"  [{tag}] epoch {ep+1}/{epochs} avg_loss={tot/len(pairs):.4f}")
    wall = time.time() - t0
    peft_model.eval()
    return {"wall_clock_s": wall, "n_train": len(pairs), "n_params": n_params}


def train_lora(base, tok, task, examples, demos, max_seq_len,
               rank=8, lr=1e-4, epochs=3,
               target_modules=("o_proj",), verbose=True):
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                     target_modules=list(target_modules),
                     task_type=TaskType.CAUSAL_LM, bias="none")
    pm = get_peft_model(base, cfg)
    if verbose:
        pm.print_trainable_parameters()
    info = _train(pm, tok, task, examples, demos, max_seq_len, lr, epochs, "LoRA", verbose)
    info["peft_model"] = pm
    return info


def train_ia3(base, tok, task, examples, demos, max_seq_len,
              lr=1e-3, epochs=3,
              target_modules=("o_proj",), feedforward_modules=(), verbose=True):
    from peft import IA3Config, get_peft_model, TaskType
    cfg = IA3Config(target_modules=list(target_modules),
                    feedforward_modules=list(feedforward_modules),
                    task_type=TaskType.CAUSAL_LM)
    pm = get_peft_model(base, cfg)
    if verbose:
        pm.print_trainable_parameters()
    info = _train(pm, tok, task, examples, demos, max_seq_len, lr, epochs, "IA3", verbose)
    info["peft_model"] = pm
    return info


def train_prompt_tuning(base, tok, task, examples, demos, max_seq_len,
                        n_virtual_tokens=20, lr=3e-3, epochs=3, verbose=True):
    from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
    cfg = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                             prompt_tuning_init=PromptTuningInit.RANDOM,
                             num_virtual_tokens=n_virtual_tokens,
                             tokenizer_name_or_path=None)
    pm = get_peft_model(base, cfg)
    if verbose:
        pm.print_trainable_parameters()
    info = _train(pm, tok, task, examples, demos, max_seq_len, lr, epochs, "PromptTuning", verbose)
    info["peft_model"] = pm
    return info


def train_prefix_tuning(base, tok, task, examples, demos, max_seq_len,
                        n_virtual_tokens=20, lr=3e-3, epochs=3, verbose=True):
    # prefix tuning is unstable on instruction-tuned models — expect collapses on some tasks
    from peft import PrefixTuningConfig, get_peft_model, TaskType
    cfg = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                             num_virtual_tokens=n_virtual_tokens)
    pm = get_peft_model(base, cfg)
    if verbose:
        pm.print_trainable_parameters()
    info = _train(pm, tok, task, examples, demos, max_seq_len, lr, epochs, "PrefixTuning", verbose)
    info["peft_model"] = pm
    return info


def unload_peft(pm):
    if hasattr(pm, "unload"):
        return pm.unload()
    return pm


# expose as the same names the old code used
_build_training_pairs = _pairs
_encode_pair = _encode
