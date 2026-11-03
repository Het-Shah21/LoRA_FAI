"""QLoRA linear adapter compatible with Member A's LoRA layer interface."""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .lora_layer import LoRALayer

try:
    import bitsandbytes as bnb
except (ImportError, RuntimeError):  # pragma: no cover - platform dependent
    bnb = None


def build_nf4_config(config=None):
    """Build Transformers' ``BitsAndBytesConfig`` from YAML-like settings."""
    from transformers import BitsAndBytesConfig

    values = dict(config or {})
    dtype = values.get("bnb_4bit_compute_dtype", torch.float16)
    if isinstance(dtype, str):
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype.lower()]
    return BitsAndBytesConfig(
        load_in_4bit=bool(values.get("load_in_4bit", True)),
        bnb_4bit_quant_type=values.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=bool(values.get("bnb_4bit_use_double_quant", True)),
    )


def load_nf4_config(path):
    """Load an NF4 YAML config without loading a model or dataset."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as stream:
        return build_nf4_config(yaml.safe_load(stream) or {})


class QLoRALinear(nn.Module, LoRALayer):
    """LoRA adapter around a frozen linear or bitsandbytes NF4 projection.

    The public adapter interface matches ``LoRALinear``: ``r``,
    ``lora_alpha``, ``scaling``, ``lora_dropout``, ``merged``, ``lora_A`` and
    ``lora_B``. NF4 weights cannot safely be mutated by adding a full-precision
    delta, so merge/unmerge explicitly reject quantized bases.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        fan_in_fan_out: bool = False,
        compute_dtype: Optional[torch.dtype] = None,
        use_4bit: bool = True,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)
        if not isinstance(base_layer, nn.Linear) and type(base_layer).__name__ != "Linear4bit":
            raise TypeError(
                "QLoRALinear supports nn.Linear and bitsandbytes Linear4bit bases; "
                "use Member A's LoRAConv1D for GPT-2 Conv1D."
            )
        self.base_layer = base_layer
        self.in_features = getattr(base_layer, "in_features", base_layer.weight.shape[1])
        self.out_features = getattr(base_layer, "out_features", base_layer.weight.shape[0])
        self.fan_in_fan_out = fan_in_fan_out
        self.compute_dtype = compute_dtype or getattr(base_layer.weight, "dtype", torch.float32)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False, dtype=torch.float32)
        self.reset_parameters()
        if self.merge_weights:
            self.merge()

    @classmethod
    def from_linear(cls, layer: nn.Linear, **kwargs) -> "QLoRALinear":
        """Build from ``nn.Linear``; use NF4 only with CUDA and bitsandbytes."""
        use_4bit = kwargs.pop("use_4bit", True)
        if use_4bit and bnb is not None and torch.cuda.is_available():
            compute_dtype = kwargs.get("compute_dtype") or torch.float16
            quantized = bnb.nn.Linear4bit(
                layer.in_features, layer.out_features, bias=layer.bias is not None,
                compute_dtype=compute_dtype, compress_statistics=True, quant_type="nf4",
            )
            quantized.load_state_dict(layer.state_dict())
            layer = quantized
        return cls(layer, **kwargs)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        return delta.T if self.fan_in_fan_out else delta

    @property
    def is_quantized_base(self) -> bool:
        return type(self.base_layer).__name__ == "Linear4bit"

    def merge(self) -> None:
        if self.is_quantized_base:
            raise RuntimeError(
                "Cannot merge into an NF4 base layer without losing quantization metadata."
            )
        if not self.merged:
            delta = self.get_delta_weight().to(
                self.base_layer.weight.device, dtype=self.base_layer.weight.dtype
            )
            self.base_layer.weight.data.add_(delta)
            self.merged = True

    def unmerge(self) -> None:
        if self.is_quantized_base:
            raise RuntimeError(
                "Cannot unmerge an NF4 base layer because no quantized weight mutation was performed."
            )
        if self.merged:
            delta = self.get_delta_weight().to(
                self.base_layer.weight.device, dtype=self.base_layer.weight.dtype
            )
            self.base_layer.weight.data.sub_(delta)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
        result = self.base_layer(x)
        adapter_input = self.lora_dropout(x).to(self.lora_A.weight.dtype)
        delta = self.lora_B(self.lora_A(adapter_input)) * self.scaling
        return result + delta.to(result.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, merged={self.merged}, "
            f"base={type(self.base_layer).__name__}"
        )
