"""LoRA+ optimizer parameter grouping utility.

Reference:
    Hayou et al. (2024), "LoRA+: Efficient Low Rank Adaptation of Large Models", ICML 2024.
    https://arxiv.org/abs/2402.12354
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn


def get_lora_plus_param_groups(
    model: nn.Module,
    lr: float = 1e-4,
    loraplus_lr_ratio: float = 16.0,
    weight_decay: float = 0.0,
) -> List[Dict[str, Any]]:
    """Partition model parameters into separate optimizer parameter groups for LoRA+.

    In standard LoRA, both A and B matrices share the same learning rate.
    Hayou et al. (2024) prove that for optimal feature learning, matrix B requires
    a significantly higher learning rate than matrix A (typically lr_B = 16 * lr_A).

    Args:
        model: Adapted PyTorch model with LoRA layers.
        lr: Base learning rate applied to matrix A (and other trainable parameters).
        loraplus_lr_ratio: Ratio lambda by which learning rate for matrix B is scaled (lr_B = lambda * lr).
        weight_decay: Weight decay applied to all parameters.

    Returns:
        List of parameter group dictionaries compatible with torch.optim.Optimizer.
    """
    param_group_A = []
    param_group_B = []
    param_group_other = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "lora_B" in name:
            param_group_B.append(param)
        elif "lora_A" in name:
            param_group_A.append(param)
        else:
            param_group_other.append(param)

    lr_B = lr * loraplus_lr_ratio

    groups = [
        {"params": param_group_A, "lr": lr, "weight_decay": weight_decay, "group_name": "lora_A"},
        {"params": param_group_B, "lr": lr_B, "weight_decay": weight_decay, "group_name": "lora_B"},
    ]

    if param_group_other:
        groups.append({
            "params": param_group_other,
            "lr": lr,
            "weight_decay": weight_decay,
            "group_name": "other_trainable",
        })

    return groups
