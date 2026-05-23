"""Experiment configuration — model + task + edit hyperparameters.

The defaults here exactly match the paper:
  - mu (ridge anchor)     = 1e-2
  - alpha (blend weight)  = 0.8
  - sparsity sweep        = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
  - fs_seeds              = (0, 1, 2)
  - n_calib               = 64 (supervised), 32 (unsupervised)

Override via constructor kwargs or dataclasses.replace().
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict
import json
import os


# ============================================================================ #
#   Model registry — the 3 models evaluated in the paper                      #
# ============================================================================ #
MODELS: Dict[str, Dict[str, object]] = {
    "mistral7b": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "dtype": "float16",
        "expected_layers": 32,
        "expected_hidden": 4096,
    },
    "phi3_mini": {
        "hf_name": "microsoft/Phi-3-mini-4k-instruct",
        "dtype": "float16",
        "expected_layers": 32,
        "expected_hidden": 3072,
    },
    "gemma2_2b": {
        "hf_name": "google/gemma-2-2b-it",
        "dtype": "bfloat16",
        "expected_layers": 26,
        "expected_hidden": 2048,
    },
}


# ============================================================================ #
#   Default sparsity sweep                                                    #
# ============================================================================ #
DEFAULT_TOPK_PERCENTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass
class ExperimentConfig:
    """All knobs for one PCC run. Defaults match the paper."""

    # ── Model ──────────────────────────────────────────────────────────────
    model_key: str = "mistral7b"            # key into MODELS dict
    dtype: str = "float16"                  # set by load_model_and_tokenizer
    device_map: str = "auto"
    num_layers: int = 0                     # populated at load time
    hidden_size: int = 0                    # populated at load time

    # ── Task ───────────────────────────────────────────────────────────────
    task_key: str = "sst2"
    n_eval: int = 150                       # eval-set size (paper uses 150-500)
    n_train_pool: int = 128                 # demo pool size
    k_shots: Optional[int] = None           # None → use task.default_k_shots
    use_canonical_demos: bool = True        # for GSM8K (Wei et al. exemplars)
    max_seq_len: Optional[int] = None       # None → use task.recommended_max_seq_len
    bs_eval: int = 1

    # ── Correction set & calibration ────────────────────────────────────────
    n_calib_pool: int = 500                 # examples to build G_full from
    n_calib: int = 64                       # |CALIB| ⊆ G_full
    calib_from_g_full: bool = True

    # ── Edit hyperparameters (paper Eq. 3-4) ───────────────────────────────
    mu: float = 1e-2
    alpha: float = 0.8
    topk_percents: Tuple[float, ...] = DEFAULT_TOPK_PERCENTS
    selection_mode: str = "top"             # 'top' | 'random' | 'early' | 'late'

    # ── Multi-seed rigor ────────────────────────────────────────────────────
    fs_seeds: Tuple[int, ...] = (0, 1, 2)
    rand_baseline_seeds: Tuple[int, ...] = (0, 1, 2)

    # ── Output ──────────────────────────────────────────────────────────────
    out_dir: str = "runs/default"
    run_name: str = ""
    save_predictions: bool = False

    def finalize(self) -> "ExperimentConfig":
        """Fill in run_name and create out_dir."""
        if not self.run_name:
            mk = self.model_key
            tk = self.task_key
            self.run_name = f"{mk}_{tk}"
        self.out_dir = os.path.join(self.out_dir, self.run_name)
        os.makedirs(self.out_dir, exist_ok=True)
        return self

    @property
    def model_name(self) -> str:
        return MODELS[self.model_key]["hf_name"]

    def dump(self) -> str:
        path = os.path.join(self.out_dir, "config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path
