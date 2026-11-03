"""Core mathematical modules and layer injection for LoRA."""

from .dora_layer import DoRALinear
from .injector import (
    count_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
    merge_adapters,
    unmerge_adapters,
)
from .lora_layer import LoRAConfig, LoRAConv1D, LoRALayer, LoRALinear
from .lora_plus import get_lora_plus_param_groups
from .qlora_layer import QLoRALinear, build_nf4_config, load_nf4_config

__all__ = [
    "LoRAConfig",
    "LoRALayer",
    "LoRALinear",
    "LoRAConv1D",
    "DoRALinear",
    "get_lora_plus_param_groups",
    "inject_lora",
    "mark_only_lora_as_trainable",
    "merge_adapters",
    "unmerge_adapters",
    "count_parameters",
    "QLoRALinear",
    "build_nf4_config",
    "load_nf4_config",
]
