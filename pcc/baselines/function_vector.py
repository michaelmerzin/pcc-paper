"""Function Vector baseline (Todd et al. 2024).

The function-vector method injects the MEAN no-shot/few-shot activation delta
as a fixed additive offset at inference time, on the selected o_proj output
rows.

PCC's mechanistic argument (paper §5.1, lines 469-493):
    FV applies the SAME fixed shift ∆̄_l to every input. PCC, by contrast,
    encodes the shift into the weight matrix as an OPERATOR that scales
    contextually with each input. This is why PCC outperforms FV by 6.9pp
    on Mistral-7B SST-2 (88.65 vs. 81.77) and 1.96pp on MRPC.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch

from .. model_io import get_layers, free
from ..sensitivity import OProjLastTokenTap, _forward_once
from ..eval_engine import evaluate


def compute_function_vector(
    H_NS: Dict[int, torch.Tensor],
    Z_FS: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """ ∆̄_l = mean over CALIB of (z_fs_l - z_ns_l)  — per-layer mean delta.

    Note: H_NS here is the o_proj INPUT (not output). The function vector is
    computed on the OUTPUT delta. Since we already have Z_FS (the few-shot
    o_proj output) and we need the no-shot o_proj output too — but we don't
    store it separately. We approximate by H_NS @ W_0^T, which equals the
    no-shot output for unedited rows.

    To avoid extra forwards, the cleaner version computes Z_NS by re-running
    on no-shot prompts during sensitivity capture. For this clean reference
    implementation we take the cleaner path: just collect Z_NS too.
    """
    raise NotImplementedError(
        "Use compute_function_vector_from_zns_zfs(Z_NS, Z_FS) — call "
        "collect_full_outputs(...) below first to obtain Z_NS.")


def collect_full_outputs(
    model, tokenizer, task, calib_examples, calib_demos, calib_shots_ns,
    num_layers: int, max_seq_len: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Collect Z_NS (no-shot o_proj outputs) and Z_FS (few-shot o_proj outputs).

    Used by the FV baseline. calib_shots_ns must be empty lists for each example.
    """
    Z_NS: Dict[int, List[torch.Tensor]] = {}
    Z_FS: Dict[int, List[torch.Tensor]] = {}
    with OProjLastTokenTap(model, capture_input=False, capture_output=True) as tap:
        for i, ex in enumerate(calib_examples):
            tap.clear()
            _forward_once(model, tokenizer, task.build_prompt(ex, [], False), max_seq_len)
            for l in range(num_layers):
                if l in tap.out_buf:
                    Z_NS.setdefault(l, []).append(tap.out_buf[l][0].clone())
            free()

            tap.clear()
            _forward_once(model, tokenizer, task.build_prompt(ex, calib_demos[i], True), max_seq_len)
            for l in range(num_layers):
                if l in tap.out_buf:
                    Z_FS.setdefault(l, []).append(tap.out_buf[l][0].clone())
            free()

    Z_NS = {l: torch.stack(v, 0) for l, v in Z_NS.items()}
    Z_FS = {l: torch.stack(v, 0) for l, v in Z_FS.items()}
    return Z_NS, Z_FS


def compute_function_vector_delta(
    Z_NS: Dict[int, torch.Tensor],
    Z_FS: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """ ∆̄_l = mean_{i ∈ G} (z_fs_{l,i} - z_ns_{l,i})  for each layer l.

    Returns dict[layer] → (hidden,) vector.
    """
    return {l: (Z_FS[l] - Z_NS[l]).mean(dim=0).float() for l in Z_NS}


class FunctionVectorInjector:
    """Context manager that adds ∆̄_l to o_proj output at every forward pass.

    Optionally restricts the injection to a selected row subset (parallel to
    the PCC row selection — same rows, different mechanism).
    """

    def __init__(self, model, deltas: Dict[int, torch.Tensor],
                 selected_rows: List[Tuple[int, int]] = None):
        self.model = model
        self.deltas = deltas
        # Group selected rows by layer
        self.by_layer: Dict[int, torch.Tensor] = {}
        if selected_rows is not None:
            for l, n in selected_rows:
                self.by_layer.setdefault(l, []).append(n)
            self.by_layer = {l: torch.tensor(sorted(set(v))) for l, v in self.by_layer.items()}
        self.handles = []

    def __enter__(self):
        for l, blk in enumerate(get_layers(self.model)):
            if l not in self.deltas:
                continue
            delta_l = self.deltas[l]
            row_idx = self.by_layer.get(l)  # None means inject on all rows

            def _make_hook(delta, rows):
                def hook(module, inp, out):
                    # out: (B, T, hidden) — we add delta to the LAST token only
                    # (matches the calibration convention)
                    d = delta.to(out.device, dtype=out.dtype)
                    if rows is not None:
                        mask = torch.zeros_like(d)
                        mask[rows.to(d.device)] = d[rows.to(d.device)]
                        out[:, -1, :] = out[:, -1, :] + mask
                    else:
                        out[:, -1, :] = out[:, -1, :] + d
                    return out
                return hook

            self.handles.append(
                blk.self_attn.o_proj.register_forward_hook(_make_hook(delta_l, row_idx))
            )
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
