"""Qwen2.5 model wrapper for Low-Rank Adaptation."""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.core.lora_layer import LoRAConfig
from .base_wrapper import BaseAdapterWrapper


class Qwen2AdapterWrapper(BaseAdapterWrapper):
    """Adapter wrapper tailored for Qwen2 / Qwen2.5 architectures."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
        config: Optional[LoRAConfig] = None,
        task: str = "causal_lm",
        num_labels: int = 2,
    ) -> None:
        super().__init__(model_name_or_path, config)
        self.task = task
        self.num_labels = num_labels

    def get_default_target_modules(self) -> List[str]:
        """Default target modules for Qwen attention adaptation."""
        return ["q_proj", "v_proj"]

    def get_all_linear_modules(self) -> List[str]:
        """All linear modules including attention and feed-forward (MLP) layers."""
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def load_base_model(self, torch_dtype: torch.dtype = torch.float32, **kwargs) -> nn.Module:
        """Load pretrained Qwen2.5 model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.task == "sequence_classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                torch_dtype=torch_dtype,
                **kwargs,
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch_dtype,
                **kwargs,
            )

        return self.model
