"""From-scratch PyTorch implementation of Low-Rank Adaptation (LoRA).

Reference:
    Hu et al. (2022), "LoRA: Low-Rank Adaptation of Large Language Models", ICLR.
    https://arxiv.org/abs/2106.09685
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration class for Low-Rank Adaptation (LoRA).

    Attributes:
        r: Intrinsic low-rank dimension (r << min(d_in, d_out)).
        lora_alpha: Constant scaling factor alpha.
        lora_dropout: Dropout probability applied to the input of matrix A.
        target_modules: List of module names or suffixes to replace with LoRA.
        merge_weights: Whether to merge weights upon initialization or call.
        fan_in_fan_out: Set to True if the layer stores weights as (fan_in, fan_out),
            such as Hugging Face's Conv1D layer used in GPT-2.
        bias: Bias training strategy: 'none', 'all', or 'lora_only'.
    """

    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    merge_weights: bool = False
    fan_in_fan_out: bool = False
    bias: str = "none"


class LoRALayer:
    """Base interface for LoRA layers managing hyperparameters and merge state."""

    def __init__(
        self,
        r: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ) -> None:
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinear(nn.Module, LoRALayer):
    """LoRA adapted Linear layer wrapping a frozen base linear layer.

    Computes:
        y = x W_0^T + (alpha / r) * (Dropout(x) @ A^T @ B^T) + b

    Where:
        W_0 in R^(d_out x d_in) is the frozen pretrained weight.
        A in R^(r x d_in) is initialized with Kaiming uniform.
        B in R^(d_out x r) is strictly initialized to 0, ensuring delta_W = 0 at step 0.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        fan_in_fan_out: bool = False,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.fan_in_fan_out = fan_in_fan_out

        # Freeze the base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank decomposition matrices:
        # lora_A computes x @ A^T, so its weight shape is (r, in_features)
        # lora_B computes (x @ A^T) @ B^T, so its weight shape is (out_features, r)
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False)

        self.reset_parameters()

        if self.merge_weights:
            self.merge()

    def reset_parameters(self) -> None:
        """Initialize low-rank matrices per Hu et al. (2022).

        A is initialized with Kaiming uniform initialization (or Gaussian).
        B is initialized strictly to zero, ensuring W + (alpha/r)*B*A = W at initialization.
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        """Compute the full low-rank weight update matrix delta_W = (alpha / r) * (B @ A).

        Returns:
            Tensor of shape (out_features, in_features), or (in_features, out_features)
            if fan_in_fan_out is True.
        """
        # lora_B.weight: (out_features, r)
        # lora_A.weight: (r, in_features)
        # delta_w: (out_features, in_features)
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        if self.fan_in_fan_out:
            delta_w = delta_w.T
        return delta_w

    def merge(self) -> None:
        """Fold low-rank adapter delta_W directly into the frozen base layer weight in-place.

        Guarantees zero additional inference latency and zero memory overhead at deployment.
        """
        if not self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
            self.base_layer.weight.data.add_(delta_w)
            self.merged = True

    def unmerge(self) -> None:
        """Subtract low-rank adapter delta_W from the base layer weight to restore original state."""
        if self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
            self.base_layer.weight.data.sub_(delta_w)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapted layer.

        If merged: directly computes base_layer(x) with zero latency.
        If unmerged: computes base_layer(x) + (alpha/r) * lora_B(lora_A(dropout(x))).
        """
        if self.merged:
            return self.base_layer(x)

        # Base forward pass
        result = self.base_layer(x)

        # Adapter forward pass
        lora_in = self.lora_dropout(x).to(self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(lora_in)) * self.scaling

        return result + lora_out.to(result.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, scaling={self.scaling}, "
            f"merged={self.merged}"
        )


class LoRAConv1D(nn.Module, LoRALayer):
    """LoRA adapted wrapper for Hugging Face's Conv1D layer (used in GPT-2).

    HF Conv1D stores weight with shape (nx, nf) = (in_features, out_features)
    and computes: x @ weight + bias.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.base_layer = base_layer
        # In HF Conv1D, weight is shape (nx, nf)
        self.in_features = base_layer.weight.shape[0]
        self.out_features = base_layer.weight.shape[1]

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False)

        self.reset_parameters()

        if self.merge_weights:
            self.merge()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        """Returns delta_W transposed to match HF Conv1D shape (in_features, out_features)."""
        # (lora_B.weight @ lora_A.weight) is (out_features, in_features)
        # Transpose to (in_features, out_features) to match base_layer.weight.shape
        return (self.lora_B.weight @ self.lora_A.weight).T * self.scaling

    def merge(self) -> None:
        if not self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
            self.base_layer.weight.data.add_(delta_w)
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
            self.base_layer.weight.data.sub_(delta_w)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        result = self.base_layer(x)
        lora_in = self.lora_dropout(x).to(self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(lora_in)) * self.scaling
        return result + lora_out.to(result.dtype)
