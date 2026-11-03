"""GPT-2 model wrapper for Low-Rank Adaptation."""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2Config, GPT2ForSequenceClassification, GPT2LMHeadModel

from src.core.lora_layer import LoRAConfig
from .base_wrapper import BaseAdapterWrapper


class GPT2AdapterWrapper(BaseAdapterWrapper):
    """Adapter wrapper tailored for GPT-2 architectures (using Conv1D projection layers)."""

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        config: Optional[LoRAConfig] = None,
        task: str = "causal_lm",
        num_labels: int = 2,
    ) -> None:
        super().__init__(model_name_or_path, config)
        self.task = task
        self.num_labels = num_labels

    def get_default_target_modules(self) -> List[str]:
        """In GPT-2, c_attn packs Query, Key, and Value projections."""
        return ["c_attn"]

    def load_base_model(self, torch_dtype: torch.dtype = torch.float32, **kwargs) -> nn.Module:
        """Load pretrained GPT-2 model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.task == "sequence_classification":
            self.model = GPT2ForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                torch_dtype=torch_dtype,
                **kwargs,
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.model = GPT2LMHeadModel.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch_dtype,
                **kwargs,
            )

        return self.model
