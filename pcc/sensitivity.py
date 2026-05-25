"""Hooks that capture the last-token o_proj input/output at every layer,
plus the sensitivity score (mean abs delta over CALIB)."""
import torch

from .model_io import get_layers, free


class OProjLastTokenTap:
    """Captures o_proj input/output at position -1 for every layer.

    We only take the last token because that's the position immediately before
    generation starts — for all of our tasks the answer token follows. Storing
    one vector per layer per example keeps things cheap: 32 * 4096 * 4 bytes
    on Mistral is ~500KB per example, so 64 calib examples ~ 32MB cpu-side.
    """

    def __init__(self, model, capture_input=True, capture_output=True):
        self.model = model
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.in_buf = {}
        self.out_buf = {}
        self.handles = []

    def __enter__(self):
        self.handles = []
        for li, blk in enumerate(get_layers(self.model)):
            self.handles.append(
                blk.self_attn.o_proj.register_forward_hook(self._mk_hook(li))
            )
        return self

    def _mk_hook(self, li):
        def hook(module, inp, out):
            if self.capture_input:
                self.in_buf[li] = inp[0][:, -1, :].detach().cpu().float()
            if self.capture_output:
                self.out_buf[li] = out[:, -1, :].detach().cpu().float()
        return hook

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        self.in_buf.clear()
        self.out_buf.clear()


def _forward_once(model, tokenizer, prompt, max_seq_len):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    with torch.no_grad():
        model(**enc)


def compute_sensitivity(model, tokenizer, task, calib_examples, calib_demos,
                        num_layers, hidden_size, max_seq_len, verbose=True):
    """Return (L, d) tensor of mean |z_fs - z_ns| over the calibration examples."""
    scores = torch.zeros((num_layers, hidden_size), dtype=torch.float32)
    n = len(calib_examples)

    with OProjLastTokenTap(model, capture_input=False, capture_output=True) as tap:
        for i, ex in enumerate(calib_examples):
            tap.clear()
            _forward_once(model, tokenizer,
                          task.build_prompt(ex, [], with_fewshot=False), max_seq_len)
            y_ns = {l: tap.out_buf[l][0].clone() for l in tap.out_buf}
            free()

            tap.clear()
            _forward_once(model, tokenizer,
                          task.build_prompt(ex, calib_demos[i], with_fewshot=True), max_seq_len)
            y_fs = {l: tap.out_buf[l][0].clone() for l in tap.out_buf}
            free()

            for l in range(num_layers):
                if l in y_ns and l in y_fs:
                    scores[l] += (y_fs[l] - y_ns[l]).abs()

            if verbose and (i + 1) % max(1, n // 4) == 0:
                print(f"  sensitivity {i+1}/{n}", end="\r")

    scores /= max(1, n)

    if verbose:
        flat = scores.view(-1)
        top = flat.argmax().item()
        l, c = top // hidden_size, top % hidden_size
        print(f"  top neuron: L{l} N{c}  s={scores[l, c]:.4f}")
    return scores


def collect_calibration_tensors(model, tokenizer, task, calib_examples, calib_demos,
                                num_layers, hidden_size, max_seq_len, verbose=True):
    """H_ns (o_proj input) and Z_fs (o_proj output) at last token, stacked over CALIB."""
    h_list = {}
    z_list = {}
    n = len(calib_examples)

    with OProjLastTokenTap(model, capture_input=True, capture_output=True) as tap:
        for i, ex in enumerate(calib_examples):
            # no-shot: capture the input
            tap.clear()
            _forward_once(model, tokenizer,
                          task.build_prompt(ex, [], with_fewshot=False), max_seq_len)
            for l in range(num_layers):
                if l in tap.in_buf:
                    h_list.setdefault(l, []).append(tap.in_buf[l][0].clone())
            free()

            # few-shot: capture the output
            tap.clear()
            _forward_once(model, tokenizer,
                          task.build_prompt(ex, calib_demos[i], with_fewshot=True), max_seq_len)
            for l in range(num_layers):
                if l in tap.out_buf:
                    z_list.setdefault(l, []).append(tap.out_buf[l][0].clone())
            free()

            if verbose and (i + 1) % max(1, n // 4) == 0:
                print(f"  calib {i+1}/{n}", end="\r")

    H = {l: torch.stack(v, 0) for l, v in h_list.items()}
    Z = {l: torch.stack(v, 0) for l, v in z_list.items()}
    return H, Z
