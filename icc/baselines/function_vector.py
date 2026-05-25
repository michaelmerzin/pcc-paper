"""Function Vector (Todd et al. 2024) — mean o_proj-output delta as fixed offset."""
import torch

from ..model_io import get_layers, free
from ..sensitivity import OProjLastTokenTap, _forward_once


def collect_full_outputs(model, tokenizer, task, examples, demos, calib_shots_ns,
                        num_layers, max_seq_len):
    """Get Z_NS and Z_FS (o_proj outputs at last token) for FV's mean delta."""
    Z_NS, Z_FS = {}, {}
    with OProjLastTokenTap(model, capture_input=False, capture_output=True) as tap:
        for i, ex in enumerate(examples):
            tap.clear()
            _forward_once(model, tokenizer, task.build_prompt(ex, [], False), max_seq_len)
            for l in range(num_layers):
                if l in tap.out_buf:
                    Z_NS.setdefault(l, []).append(tap.out_buf[l][0].clone())
            free()
            tap.clear()
            _forward_once(model, tokenizer, task.build_prompt(ex, demos[i], True), max_seq_len)
            for l in range(num_layers):
                if l in tap.out_buf:
                    Z_FS.setdefault(l, []).append(tap.out_buf[l][0].clone())
            free()
    return ({l: torch.stack(v, 0) for l, v in Z_NS.items()},
            {l: torch.stack(v, 0) for l, v in Z_FS.items()})


def compute_function_vector_delta(Z_NS, Z_FS):
    return {l: (Z_FS[l] - Z_NS[l]).mean(dim=0).float() for l in Z_NS}


class FunctionVectorInjector:
    """Add delta to o_proj output at the final token, on every forward pass."""

    def __init__(self, model, deltas, selected_rows=None):
        self.model = model
        self.deltas = deltas
        self.by_layer = {}
        if selected_rows is not None:
            for l, n in selected_rows:
                self.by_layer.setdefault(l, []).append(n)
            self.by_layer = {l: torch.tensor(sorted(set(v))) for l, v in self.by_layer.items()}
        self.handles = []

    def __enter__(self):
        for l, blk in enumerate(get_layers(self.model)):
            if l not in self.deltas:
                continue
            d = self.deltas[l]
            rows = self.by_layer.get(l)

            def mk(delta, row_idx):
                def hook(module, inp, out):
                    dd = delta.to(out.device, dtype=out.dtype)
                    if row_idx is not None:
                        mask = torch.zeros_like(dd)
                        ri = row_idx.to(dd.device)
                        mask[ri] = dd[ri]
                        out[:, -1, :] = out[:, -1, :] + mask
                    else:
                        out[:, -1, :] = out[:, -1, :] + dd
                    return out
                return hook

            self.handles.append(blk.self_attn.o_proj.register_forward_hook(mk(d, rows)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
