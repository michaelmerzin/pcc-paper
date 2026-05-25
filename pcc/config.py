"""Experiment config + model registry."""
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


MODELS = {
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
        "dtype": "bfloat16",  # bf16 mandatory, fp16 overflows on the gating
        "expected_layers": 26,
        "expected_hidden": 2048,
    },
}


# Sweep grid. We extended to 0.40 once Gemma BoolQ wanted 0.35.
DEFAULT_P_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


@dataclass
class ExperimentConfig:
    model_key: str = "mistral7b"
    dtype: str = "float16"
    device_map: str = "auto"
    num_layers: int = 0       # filled in load_model_and_tokenizer
    hidden_size: int = 0

    task_key: str = "sst2"
    n_eval: int = 150
    n_train_pool: int = 128
    k_shots: Optional[int] = None
    use_canonical_demos: bool = True
    max_seq_len: Optional[int] = None
    bs_eval: int = 1

    n_calib_pool: int = 500
    n_calib: int = 64
    calib_from_g_full: bool = True

    mu: float = 1e-2
    alpha: float = 0.8
    topk_percents: Tuple[float, ...] = DEFAULT_P_GRID
    selection_mode: str = "top"   # top | random | early | late

    # Defaults are the standard-task seeds. MMLU and BoolQ/ARC override these
    # in the runner scripts.
    fs_seeds: Tuple[int, ...] = (1, 2)
    rand_baseline_seeds: Tuple[int, ...] = (42, 137, 271)

    out_dir: str = "runs/default"
    run_name: str = ""
    save_predictions: bool = False

    def finalize(self):
        if not self.run_name:
            self.run_name = f"{self.model_key}_{self.task_key}"
        self.out_dir = os.path.join(self.out_dir, self.run_name)
        os.makedirs(self.out_dir, exist_ok=True)
        return self

    @property
    def model_name(self):
        return MODELS[self.model_key]["hf_name"]

    def dump(self):
        path = os.path.join(self.out_dir, "config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path
