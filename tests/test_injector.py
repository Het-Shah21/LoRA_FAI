"""Unit tests for dynamic LoRA injector, parameter freezing, and module traversal."""

import pytest
import torch
import torch.nn as nn

from src.core.injector import (
    count_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
    merge_adapters,
    unmerge_adapters,
)
from src.core.lora_layer import LoRAConfig, LoRALinear


class ToyAttention(nn.Module):
    """Mock multi-head attention module."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + k + v)


class ToyTransformerBlock(nn.Module):
    """Mock transformer block containing attention and feedforward layers."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.attn = ToyAttention(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class ToyModel(nn.Module):
    """Mock multi-layer transformer model."""

    def __init__(self, num_layers: int = 4, d_model: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([ToyTransformerBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def test_selective_injection():
    """Verify that only specified target modules are replaced with LoRALinear."""
    model = ToyModel(num_layers=2, d_model=32)
    config = LoRAConfig(
        r=4,
        lora_alpha=8.0,
        target_modules=["q_proj", "v_proj"],
    )

    model = inject_lora(model, config)

    # q_proj and v_proj should be LoRALinear
    for layer in model.layers:
        assert isinstance(layer.attn.q_proj, LoRALinear)
        assert isinstance(layer.attn.v_proj, LoRALinear)
        # k_proj and out_proj should remain standard Linear
        assert isinstance(layer.attn.k_proj, nn.Linear)
        assert not isinstance(layer.attn.k_proj, LoRALinear)
        assert isinstance(layer.attn.out_proj, nn.Linear)
        assert not isinstance(layer.attn.out_proj, LoRALinear)


def test_parameter_freezing_and_counting():
    """Verify that all non-LoRA parameters are frozen and only adapter matrices are trainable."""
    model = ToyModel(num_layers=3, d_model=64)
    config = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])
    model = inject_lora(model, config)

    stats = count_parameters(model)
    assert stats["trainable"] > 0
    assert stats["frozen"] > 0
    assert stats["percent_trainable"] < 10.0  # Much smaller than full model

    # Check requires_grad flag for every parameter
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad is True, f"{name} should be trainable!"
        else:
            assert param.requires_grad is False, f"{name} should be frozen!"


def test_model_end_to_end_backward():
    """Verify forward and backward gradients flow cleanly through an injected model."""
    torch.manual_seed(42)
    model = ToyModel(num_layers=2, d_model=32)
    config = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])
    model = inject_lora(model, config)

    # Note: When B = 0 at step 0, dL/dA is 0 by chain rule (Hu et al., 2022).
    # Initialize B with small weights to verify backward gradient flow through both A and B.
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B.weight, std=0.01)

    x = torch.randn(4, 10, 32)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # LoRA adapters must have non-zero gradients for both A and B
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.grad is not None
            assert not torch.all(param.grad == 0.0), f"Gradient was zero for {name}"
        else:
            assert param.grad is None


def test_model_global_merge_unmerge():
    """Verify global merge_adapters and unmerge_adapters on entire model."""
    torch.manual_seed(42)
    model = ToyModel(num_layers=2, d_model=32)
    config = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])
    model = inject_lora(model, config)

    # Initialize non-zero adapter weights
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_A.weight, std=0.05)
            torch.nn.init.normal_(m.lora_B.weight, std=0.05)

    x = torch.randn(2, 5, 32)
    out_before = model(x)

    # Global merge
    merge_adapters(model)
    out_merged = model(x)
    assert torch.allclose(out_before, out_merged, atol=1e-6)

    # Global unmerge
    unmerge_adapters(model)
    out_unmerged = model(x)
    assert torch.allclose(out_before, out_unmerged, atol=1e-6)
