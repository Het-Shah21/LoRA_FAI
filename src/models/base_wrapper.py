"""Base model wrapper defining the adaptation interface for transformer architectures."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.core.injector import count_parameters, inject_lora, merge_adapters, unmerge_adapters
from src.core.lora_layer import LoRAConfig


class BaseAdapterWrapper(ABC):
    """Abstract base class wrapping a transformer model for LoRA / QLoRA adaptation."""

    def __init__(self, model_name_or_path: str, config: Optional[LoRAConfig] = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.lora_config = config
        self.model: Optional[nn.Module] = None
        self.tokenizer: Any = None
        self.is_adapted = False

    @abstractmethod
    def load_base_model(self, **kwargs) -> nn.Module:
        """Instantiate and return the underlying pretrained transformer backbone."""
        pass

    @abstractmethod
    def get_default_target_modules(self) -> List[str]:
        """Return the architecture-specific default linear projection module names."""
        pass

    def apply_lora(self, config: Optional[LoRAConfig] = None) -> nn.Module:
        """Inject LoRA adapters into designated target modules and freeze base weights."""
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Call load_base_model() first.")

        cfg = config or self.lora_config
        if cfg is None:
            cfg = LoRAConfig(target_modules=self.get_default_target_modules())
        elif not cfg.target_modules:
            cfg.target_modules = self.get_default_target_modules()

        self.lora_config = cfg
        self.model = inject_lora(self.model, cfg)
        self.is_adapted = True
        return self.model

    def merge(self) -> None:
        """Fold adapter weights into base weights in-place for zero-latency inference."""
        if self.model is not None and self.is_adapted:
            merge_adapters(self.model)

    def unmerge(self) -> None:
        """Unfold adapter weights from base weights."""
        if self.model is not None and self.is_adapted:
            unmerge_adapters(self.model)

    def get_parameter_summary(self) -> Dict[str, float]:
        """Return parameter count statistics (trainable vs frozen)."""
        if self.model is None:
            return {"total": 0, "trainable": 0, "frozen": 0, "percent_trainable": 0.0}
        return count_parameters(self.model)
