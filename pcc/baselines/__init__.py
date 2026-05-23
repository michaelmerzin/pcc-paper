"""Baselines for the PCC comparison (paper §5.1, Table 2):

    - function_vector  : Todd et al. (2024) — mean activation-delta injection
    - peft             : LoRA, IA³, Prompt Tuning, Prefix Tuning (via peft library)
    - random_fill      : U[-1,1] noise into selected rows (paper §6.2 sanity check)
"""
