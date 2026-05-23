"""PCC — Parametric Context Compression.

Clean reproduction package for the EMNLP 2026 paper:
'Parametric Context Compression: Compiling Few-Shot Behavior into Sparse Weight Edits'

Top-level usage:
    from pcc.pipeline import run_pcc
    result = run_pcc(model_key='mistral7b', task_key='sst2', sparsity=0.10)

Each script under `experiments/` reproduces exactly one paper table.
"""
__version__ = "1.0.0"
