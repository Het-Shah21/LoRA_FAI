"""Unit tests and mathematical parity validation for LoRALinear vs Hugging Face PEFT.

Validates that custom LoRALinear produces numerically equivalent forward logits
and backward gradients within strict tolerance (atol <= 1e-5).
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig as PEFTLoraConfig
from peft.tuners.lora import Linear as PEFTLoraLinear

from src.core.lora_layer import LoRAConfig, LoRALinear


def test_lora_initialization():
    """Verify that B is zero-initialized and output equals base layer at step 0."""
    torch.manual_seed(42)
    in_features, out_features, r = 32, 64, 4
    base = nn.Linear(in_features, out_features, bias=True)
    lora = LoRALinear(base_layer=base, r=r, lora_alpha=8.0, lora_dropout=0.0)

    # 1. Base weights must be frozen
    assert not lora.base_layer.weight.requires_grad
    assert not lora.base_layer.bias.requires_grad

    # 2. Matrix B must be strictly zero
    assert torch.all(lora.lora_B.weight == 0.0)

    # 3. Matrix A must be non-zero (Kaiming uniform)
    assert not torch.all(lora.lora_A.weight == 0.0)

    # 4. At step 0, forward pass output must be identical to base layer output
    x = torch.randn(8, in_features)
    out_lora = lora(x)
    out_base = base(x)

    assert torch.allclose(out_lora, out_base, atol=1e-7), (
        f"Maximum divergence at step 0: {(out_lora - out_base).abs().max().item()}"
    )


def test_lora_forward_parity_vs_peft():
    """Verify forward pass output parity against Hugging Face PEFT within 1e-5."""
    torch.manual_seed(1337)
    in_features, out_features, r, alpha = 64, 128, 8, 16.0
    batch_size, seq_len = 4, 16

    # 1. Create base layer with identical weights
    base_linear = nn.Linear(in_features, out_features, bias=True)
    base_linear_peft = nn.Linear(in_features, out_features, bias=True)
    base_linear_peft.weight.data.copy_(base_linear.weight.data)
    base_linear_peft.bias.data.copy_(base_linear.bias.data)

    # 2. Instantiate custom LoRALinear
    custom_lora = LoRALinear(
        base_layer=base_linear,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
    )

    # 3. Instantiate HF PEFT LoraLinear
    peft_cfg = PEFTLoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0)
    peft_lora = PEFTLoraLinear(
        base_layer=base_linear_peft,
        adapter_name="default",
        config=peft_cfg,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
    )

    # 4. Synchronize adapter weights so both have identical random parameters
    # Note: Initialize B with non-zero values to test delta computation
    torch.nn.init.normal_(custom_lora.lora_A.weight, std=0.02)
    torch.nn.init.normal_(custom_lora.lora_B.weight, std=0.02)

    peft_lora.lora_A["default"].weight.data.copy_(custom_lora.lora_A.weight.data)
    peft_lora.lora_B["default"].weight.data.copy_(custom_lora.lora_B.weight.data)

    # 5. Evaluate forward pass on random 3D input tensor (batch, seq_len, in_features)
    x = torch.randn(batch_size, seq_len, in_features)
    out_custom = custom_lora(x)
    out_peft = peft_lora(x)

    diff = (out_custom - out_peft).abs().max().item()
    assert torch.allclose(out_custom, out_peft, atol=1e-5), (
        f"Forward parity failed! Max absolute discrepancy vs PEFT: {diff:.6e}"
    )


def test_lora_backward_gradient_parity_vs_peft():
    """Verify backward gradient update parity against Hugging Face PEFT within 1e-5."""
    torch.manual_seed(2024)
    in_features, out_features, r, alpha = 48, 96, 4, 8.0

    # 1. Base layers
    base_custom = nn.Linear(in_features, out_features, bias=False)
    base_peft = nn.Linear(in_features, out_features, bias=False)
    base_peft.weight.data.copy_(base_custom.weight.data)

    # 2. Adapters
    custom_lora = LoRALinear(base_layer=base_custom, r=r, lora_alpha=alpha, lora_dropout=0.0)
    peft_cfg = PEFTLoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0)
    peft_lora = PEFTLoraLinear(
        base_layer=base_peft,
        adapter_name="default",
        config=peft_cfg,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
    )

    # Synchronize weights with non-zero values
    torch.nn.init.normal_(custom_lora.lora_A.weight, std=0.1)
    torch.nn.init.normal_(custom_lora.lora_B.weight, std=0.1)
    peft_lora.lora_A["default"].weight.data.copy_(custom_lora.lora_A.weight.data)
    peft_lora.lora_B["default"].weight.data.copy_(custom_lora.lora_B.weight.data)

    # 3. Identical input tensor
    x_custom = torch.randn(6, in_features, requires_grad=False)
    x_peft = x_custom.clone().detach()

    # 4. Forward passes
    out_custom = custom_lora(x_custom)
    out_peft = peft_lora(x_peft)

    # 5. Target loss computation
    target = torch.randn_like(out_custom)
    loss_custom = F.mse_loss(out_custom, target)
    loss_peft = F.mse_loss(out_peft, target)

    assert torch.allclose(loss_custom, loss_peft, atol=1e-6)

    # 6. Backward passes
    loss_custom.backward()
    loss_peft.backward()

    # 7. Compare gradients of adapter matrix A
    grad_A_custom = custom_lora.lora_A.weight.grad
    grad_A_peft = peft_lora.lora_A["default"].weight.grad
    diff_grad_A = (grad_A_custom - grad_A_peft).abs().max().item()

    assert torch.allclose(grad_A_custom, grad_A_peft, atol=1e-5), (
        f"Gradient parity failed for matrix A! Max diff: {diff_grad_A:.6e}"
    )

    # 8. Compare gradients of adapter matrix B
    grad_B_custom = custom_lora.lora_B.weight.grad
    grad_B_peft = peft_lora.lora_B["default"].weight.grad
    diff_grad_B = (grad_B_custom - grad_B_peft).abs().max().item()

    assert torch.allclose(grad_B_custom, grad_B_peft, atol=1e-5), (
        f"Gradient parity failed for matrix B! Max diff: {diff_grad_B:.6e}"
    )
