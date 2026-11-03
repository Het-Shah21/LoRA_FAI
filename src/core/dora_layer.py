"""Weight-Decomposed Low-Rank Adaptation (DoRA) implementation.

Reference:
    Liu et al. (2024), "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024.
    https://arxiv.org/abs/2402.09353
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_layer import LoRALayer


class DoRALinear(nn.Module, LoRALayer):
    """DoRA adapted Linear layer decomposing weight into magnitude and direction.

    Formulation:
        W = m * (V / ||V||_c)
        where V = W_0 + (alpha / r) * (B @ A)
        and m is a trainable magnitude parameter initialized to ||W_0||_c.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
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
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Low-rank directional update
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False)

        # Trainable magnitude vector: one scalar per output feature (row of weight)
        # Compute initial Frobenius norm of each row of W_0
        with torch.no_grad():
            initial_norm = torch.linalg.norm(self.base_layer.weight, dim=1, keepdim=True)  # (out_features, 1)
        self.magnitude = nn.Parameter(initial_norm.clone())

        self.reset_parameters()

        if self.merge_weights:
            self.merge()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if hasattr(self, "base_layer"):
            with torch.no_grad():
                self.magnitude.copy_(torch.linalg.norm(self.base_layer.weight, dim=1, keepdim=True))

    def get_effective_weight(self) -> torch.Tensor:
        """Compute W = m * (V / ||V||_c), where V = W_0 + delta_W."""
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        v = self.base_layer.weight + delta_w
        v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
        # Avoid division by zero
        v_norm = torch.clamp(v_norm, min=1e-12)
        return self.magnitude * (v / v_norm)

    def merge(self) -> None:
        if not self.merged:
            eff_w = self.get_effective_weight().detach().to(self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
            self.base_layer.weight.data.copy_(eff_w)
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            raise RuntimeError(
                "DoRA modifies weight direction and magnitude non-linearly; "
                "exact unmerge requires caching the pre-merge base weights."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        # Compute adapted weight and forward
        eff_weight = self.get_effective_weight()
        bias = self.base_layer.bias
        return F.linear(x, eff_weight, bias)
