"""Sensitivity scoring and calibration tensor collection.

CRITICAL — Token-position alignment (paper §3.6):
  We extract H^ns and Z^fs at the FINAL INPUT TOKEN of each prompt.
  This is the activation immediately preceding the position where the model
  emits its first output token. The convention works uniformly across:
    - classification    (last token = verbalizer cue token like ':')
    - multiple-choice   (last token = answer prefix)
    - chain-of-thought  (last token = generation cue like '[/INST]')

Memory:
  - We store only the last-token vector per layer per example.
  - For Mistral-7B (32 layers × 4096): ~512 KB / example.
  - For 64 calibration examples: ~32 MB total on CPU. Fits any GPU host.
"""
from __future__ import annotations
from typing import Dict, List
import torch

from .model_io import get_layers, free


# ============================================================================ #
#   Forward hooks that capture last-token activations at every o_proj         #
# ============================================================================ #
class OProjLastTokenTap:
    """Context manager that captures the LAST TOKEN's o_proj input/output at every layer.

    This is the dimensionally clean tensor needed for the ridge solve
    (paper §3.6, lines 355-360): one vector per layer per example.
    """

    def __init__(self, model, capture_input: bool = True, capture_output: bool = True):
        self.model = model
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.in_buf: Dict[int, torch.Tensor] = {}
        self.out_buf: Dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        self.handles = []
        for li, blk in enumerate(get_layers(self.model)):
            self.handles.append(
                blk.self_attn.o_proj.register_forward_hook(self._make_hook(li))
            )
        return self

    def _make_hook(self, li: int):
        def hook(module, inp, out):
            # inp[0]: (B, T, hidden)   ← input to o_proj
            # out:    (B, T, hidden)   ← output of o_proj
            # We take position -1 (final input token) on the first batch row.
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


def _forward_once(model, tokenizer, prompt: str, max_seq_len: int) -> None:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    with torch.no_grad():
        _ = model(**enc)


# ============================================================================ #
#   Paper Eq. 2 — sensitivity SCORES                                          #
# ============================================================================ #
def compute_sensitivity(
    model, tokenizer, task,
    calib_examples: List,
    calib_demos: List[List],
    num_layers: int,
    hidden_size: int,
    max_seq_len: int,
    verbose: bool = True,
) -> torch.Tensor:
    """Compute SCORES[l, n] = mean over CALIB of |z_fs[l,n] - z_ns[l,n]|.

    Returns:
        (L, hidden) tensor of sensitivity scores. Used by select_rows() in edit.py.
    """
    SCORES = torch.zeros((num_layers, hidden_size), dtype=torch.float32)
    n = len(calib_examples)

    with OProjLastTokenTap(model, capture_input=False, capture_output=True) as tap:
        for i, ex in enumerate(calib_examples):
            # no-shot forward
            tap.clear()
            _forward_once(model, tokenizer,
                          task.build_prompt(ex, [], with_fewshot=False), max_seq_len)
            y_ns = {l: tap.out_buf[l][0].clone() for l in tap.out_buf}
            free()

            # few-shot forward
            tap.clear()
            _forward_once(model, tokenizer,
                          task.build_prompt(ex, calib_demos[i], with_fewshot=True), max_seq_len)
            y_fs = {l: tap.out_buf[l][0].clone() for l in tap.out_buf}
            free()

            for l in range(num_layers):
                if l in y_ns and l in y_fs:
                    SCORES[l] += (y_fs[l] - y_ns[l]).abs()

            if verbose and (i + 1) % max(1, n // 4) == 0:
                print(f"  [sensitivity] {i+1}/{n}", end="\r")

    SCORES /= max(1, n)

    if verbose:
        flat = SCORES.view(-1)
        top = torch.argmax(flat).item()
        l, c = top // hidden_size, top % hidden_size
        print(f"  [sensitivity] top neuron: L{l}-N{c}  s={SCORES[l, c]:.4f}")

    return SCORES


# ============================================================================ #
#   Calibration tensors H_ns and Z_fs (used by the ridge solve in edit.py)    #
# ============================================================================ #
def collect_calibration_tensors(
    model, tokenizer, task,
    calib_examples: List,
    calib_demos: List[List],
    num_layers: int,
    hidden_size: int,
    max_seq_len: int,
    verbose: bool = True,
):
    """Returns (H_NS, Z_FS) where each is dict[layer] -> Tensor (N, hidden).

    H_NS[l]  : no-shot   inputs  to o_proj at layer l, stacked over CALIB
    Z_FS[l]  : few-shot  outputs of o_proj at layer l, stacked over CALIB

    These are exactly the H^ns and Z^fs of paper Eq. 3 / Eq. 4.
    """
    H_NS_list: Dict[int, List[torch.Tensor]] = {}
    Z_FS_list: Dict[int, List[torch.Tensor]] = {}
    n = len(calib_examples)

    with OProjLastTokenTap(model, capture_input=True, capture_output=True) as tap:
        for i, ex in enumerate(calib_examples):
            # no-shot — capture H_ns (input to o_proj at last position)
            tap.clear()
            _forward_once(model, tokenizer,
                          task.build_prompt(ex, [], with_fewshot=False), max_seq_len)
            for l in range(num_layers):
                if l in tap.in_buf:
                    H_NS_list.setdefault(l, []).append(tap.in_buf[l][0].clone())
            free()

            # few-shot — capture Z_fs (output of o_proj at last position)
            tap.clear()
            _forward_once(model, tokenizer,
                          task.build_prompt(ex, calib_demos[i], with_fewshot=True), max_seq_len)
            for l in range(num_layers):
                if l in tap.out_buf:
                    Z_FS_list.setdefault(l, []).append(tap.out_buf[l][0].clone())
            free()

            if verbose and (i + 1) % max(1, n // 4) == 0:
                print(f"  [calibration] {i+1}/{n}", end="\r")

    H_NS = {l: torch.stack(v, 0) for l, v in H_NS_list.items()}
    Z_FS = {l: torch.stack(v, 0) for l, v in Z_FS_list.items()}

    if verbose:
        any_l = next(iter(H_NS))
        print(f"  [calibration] tensors shape per layer: ({n}, {H_NS[any_l].shape[1]})")

    return H_NS, Z_FS
