"""Dynamic module injector and parameter freezing utilities for LoRA.

Recursively replaces target linear layers with LoRALinear / LoRAConv1D and
manages gradient freezing across transformer architectures.
"""

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .lora_layer import LoRAConfig, LoRAConv1D, LoRALayer, LoRALinear


def _is_target_module(module_name: str, target_modules: List[str]) -> bool:
    """Check if the given module name matches any target in target_modules.

    Supports exact matching or suffix matching (e.g., 'attn.c_attn' matches 'c_attn').
    """
    for target in target_modules:
        if module_name == target or module_name.endswith("." + target) or target in module_name.split("."):
            return True
    return False


def _is_hf_conv1d(module: nn.Module) -> bool:
    """Check if the module is Hugging Face's Conv1D (used in GPT-2)."""
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight") and hasattr(module, "bias")


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Recursively search and replace target linear modules with LoRA layers.

    Args:
        model: Pretrained PyTorch transformer model (e.g., GPT-2 or Qwen2.5).
        config: LoRA configuration detailing rank, alpha, dropout, and target modules.

    Returns:
        The adapted model with frozen base parameters and trainable low-rank adapters.
    """
    # Collect modules to replace to avoid modifying dictionary during iteration
    named_modules = dict(model.named_modules())
    modules_to_replace: List[Tuple[str, nn.Module]] = []

    for name, module in named_modules.items():
        if _is_target_module(name, config.target_modules):
            if isinstance(module, nn.Linear):
                modules_to_replace.append((name, module))
            elif _is_hf_conv1d(module):
                modules_to_replace.append((name, module))

    if not modules_to_replace:
        raise ValueError(
            f"No target modules matching {config.target_modules} found in model. "
            f"Available module suffixes: {list({n.split('.')[-1] for n in named_modules.keys()})[:20]}"
        )

    # Replace target modules
    for name, base_layer in modules_to_replace:
        parent_name, _, child_name = name.rpartition(".")
        parent = named_modules[parent_name] if parent_name else model

        if isinstance(base_layer, nn.Linear):
            lora_layer = LoRALinear(
                base_layer=base_layer,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                merge_weights=config.merge_weights,
                fan_in_fan_out=config.fan_in_fan_out,
            )
        elif _is_hf_conv1d(base_layer):
            lora_layer = LoRAConv1D(
                base_layer=base_layer,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                merge_weights=config.merge_weights,
            )
        else:
            continue

        setattr(parent, child_name, lora_layer)

    # Freeze base model weights and configure trainable parameters
    mark_only_lora_as_trainable(model, bias=config.bias)

    return model


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all base parameters, leaving exclusively LoRA parameters trainable.

    Args:
        model: Adapted model.
        bias: Bias configuration: 'none', 'all', or 'lora_only'.
    """
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Bias strategy '{bias}' not supported. Choose from 'none', 'all', 'lora_only'.")


def merge_adapters(model: nn.Module) -> None:
    """Merge all LoRA adapters in the model into their base weights in-place."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.merge()


def unmerge_adapters(model: nn.Module) -> None:
    """Unmerge all LoRA adapters in the model to restore base weights."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.unmerge()


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Calculate and return total, trainable, and percentage of trainable parameters.

    Returns:
        Dict containing 'total', 'trainable', 'frozen', and 'percent_trainable'.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    percent_trainable = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "percent_trainable": round(percent_trainable, 4),
    }
