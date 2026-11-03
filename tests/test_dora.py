"""Unit tests for DoRA (Weight-Decomposed LoRA) and LoRA+."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.dora_layer import DoRALinear
from src.core.lora_plus import get_lora_plus_param_groups


def test_dora_initialization():
    """Verify DoRA produces identical outputs to base layer at step 0."""
    torch.manual_seed(42)
    in_features, out_features, r = 32, 64, 4
    base = nn.Linear(in_features, out_features, bias=True)
    dora = DoRALinear(base_layer=base, r=r, lora_alpha=8.0)

    # 1. Base weights must be frozen
    assert not dora.base_layer.weight.requires_grad

    # 2. Matrix B must be zero
    assert torch.all(dora.lora_B.weight == 0.0)

    # 3. Magnitude must match initial row norms
    expected_norm = torch.linalg.norm(base.weight, dim=1, keepdim=True)
    assert torch.allclose(dora.magnitude, expected_norm, atol=1e-6)

    # 4. Output at step 0 must match base layer output
    x = torch.randn(4, in_features)
    out_dora = dora(x)
    out_base = base(x)

    assert torch.allclose(out_dora, out_base, atol=1e-6)


def test_dora_backward_and_magnitude_gradient():
    """Verify gradients flow cleanly to magnitude vector, matrix A, and matrix B in DoRA."""
    torch.manual_seed(99)
    in_features, out_features, r = 16, 32, 2
    base = nn.Linear(in_features, out_features, bias=False)
    dora = DoRALinear(base_layer=base, r=r, lora_alpha=4.0)

    # Set non-zero B to allow gradient flow to A
    torch.nn.init.normal_(dora.lora_B.weight, std=0.05)

    x = torch.randn(3, in_features)
    out = dora(x)
    loss = out.sum()
    loss.backward()

    assert dora.magnitude.grad is not None
    assert not torch.all(dora.magnitude.grad == 0.0)
    assert dora.lora_A.weight.grad is not None
    assert dora.lora_B.weight.grad is not None


def test_dora_merge():
    """Verify DoRA in-place merge output invariance."""
    torch.manual_seed(1234)
    in_features, out_features, r = 24, 48, 4
    base = nn.Linear(in_features, out_features)
    dora = DoRALinear(base_layer=base, r=r, lora_alpha=8.0)
    torch.nn.init.normal_(dora.lora_A.weight, std=0.05)
    torch.nn.init.normal_(dora.lora_B.weight, std=0.05)

    x = torch.randn(4, in_features)
    out_unmerged = dora(x)

    dora.merge()
    assert dora.merged is True
    out_merged = dora(x)

    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)


def test_lora_plus_parameter_groups():
    """Verify that get_lora_plus_param_groups creates distinct learning rates for A and B."""
    from src.core.lora_layer import LoRALinear
    base = nn.Linear(32, 64)
    lora = LoRALinear(base_layer=base, r=4, lora_alpha=8.0)

    groups = get_lora_plus_param_groups(lora, lr=1e-4, loraplus_lr_ratio=16.0)

    assert len(groups) == 2
    group_A = next(g for g in groups if g["group_name"] == "lora_A")
    group_B = next(g for g in groups if g["group_name"] == "lora_B")

    assert group_A["lr"] == 1e-4
    assert group_B["lr"] == 16e-4
    assert group_A["params"][0] is lora.lora_A.weight
    assert group_B["params"][0] is lora.lora_B.weight
