"""Unit tests verifying mathematical invariance and zero drift in merge / unmerge."""

import pytest
import torch
import torch.nn as nn

from src.core.lora_layer import LoRAConv1D, LoRALinear


def test_linear_merge_output_invariance():
    """Verify that calling merge() produces numerically identical output to the unmerged adapter."""
    torch.manual_seed(42)
    in_features, out_features, r, alpha = 32, 64, 4, 8.0
    base = nn.Linear(in_features, out_features, bias=True)
    lora = LoRALinear(base_layer=base, r=r, lora_alpha=alpha, lora_dropout=0.0)

    # Initialize non-zero weights in adapter
    torch.nn.init.normal_(lora.lora_A.weight, std=0.05)
    torch.nn.init.normal_(lora.lora_B.weight, std=0.05)

    x = torch.randn(8, in_features)

    # Forward before merge
    out_before_merge = lora(x)

    # Execute merge
    lora.merge()
    assert lora.merged is True

    # Forward after merge
    out_after_merge = lora(x)

    diff = (out_before_merge - out_after_merge).abs().max().item()
    assert torch.allclose(out_before_merge, out_after_merge, atol=1e-6), (
        f"Output shifted after merge! Max difference: {diff:.6e}"
    )


def test_linear_unmerge_weight_restoration():
    """Verify that unmerge() cleanly restores the base layer weights with zero drift (< 1e-7)."""
    torch.manual_seed(99)
    in_features, out_features, r = 24, 48, 4
    base = nn.Linear(in_features, out_features, bias=False)
    orig_base_weight = base.weight.clone().detach()

    lora = LoRALinear(base_layer=base, r=r, lora_alpha=8.0, lora_dropout=0.0)
    torch.nn.init.normal_(lora.lora_A.weight, std=0.1)
    torch.nn.init.normal_(lora.lora_B.weight, std=0.1)

    # 1. Merge
    lora.merge()
    assert not torch.allclose(lora.base_layer.weight, orig_base_weight, atol=1e-3)

    # 2. Unmerge
    lora.unmerge()
    assert lora.merged is False

    # 3. Check base weights are restored
    drift = (lora.base_layer.weight - orig_base_weight).abs().max().item()
    assert torch.allclose(lora.base_layer.weight, orig_base_weight, atol=1e-7), (
        f"Weight drift detected after unmerge: {drift:.6e}"
    )


def test_merge_unmerge_idempotency():
    """Verify that repeated calls to merge() or unmerge() are strictly idempotent."""
    torch.manual_seed(123)
    base = nn.Linear(16, 32)
    lora = LoRALinear(base_layer=base, r=2, lora_alpha=4.0)
    torch.nn.init.normal_(lora.lora_A.weight, std=0.05)
    torch.nn.init.normal_(lora.lora_B.weight, std=0.05)

    # First merge
    lora.merge()
    w_merged_once = lora.base_layer.weight.clone()

    # Second merge (should do nothing)
    lora.merge()
    w_merged_twice = lora.base_layer.weight.clone()
    assert torch.equal(w_merged_once, w_merged_twice)

    # First unmerge
    lora.unmerge()
    w_unmerged_once = lora.base_layer.weight.clone()

    # Second unmerge (should do nothing)
    lora.unmerge()
    w_unmerged_twice = lora.base_layer.weight.clone()
    assert torch.equal(w_unmerged_once, w_unmerged_twice)


class DummyHFConv1D(nn.Module):
    """Minimal mock of Hugging Face's Conv1D layer."""

    def __init__(self, nf: int, nx: int):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.randn(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


def test_conv1d_merge_unmerge():
    """Verify merge and unmerge for HF Conv1D layer architecture."""
    torch.manual_seed(777)
    nx, nf, r = 16, 32, 4
    base_conv = DummyHFConv1D(nf=nf, nx=nx)
    orig_conv_weight = base_conv.weight.clone().detach()

    lora_conv = LoRAConv1D(base_layer=base_conv, r=r, lora_alpha=8.0)
    torch.nn.init.normal_(lora_conv.lora_A.weight, std=0.05)
    torch.nn.init.normal_(lora_conv.lora_B.weight, std=0.05)

    x = torch.randn(2, 5, nx)

    # Output before merge
    out_before = lora_conv(x)

    # Merge
    lora_conv.merge()
    out_after = lora_conv(x)

    assert torch.allclose(out_before, out_after, atol=1e-6)

    # Unmerge
    lora_conv.unmerge()
    assert torch.allclose(base_conv.weight, orig_conv_weight, atol=1e-7)
