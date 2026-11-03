"""Unit tests for dynamic LoRA injector, parameter freezing, and module traversal."""

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM

from src.core.injector import (
    count_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
    merge_adapters,
    unmerge_adapters,
)
from src.core.lora_layer import LoRAConfig, LoRAConv1D, LoRALinear


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

    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B.weight, std=0.01)

    x = torch.randn(4, 10, 32)
    out = model(x)
    loss = out.sum()
    loss.backward()

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

    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_A.weight, std=0.05)
            torch.nn.init.normal_(m.lora_B.weight, std=0.05)

    x = torch.randn(2, 5, 32)
    out_before = model(x)

    merge_adapters(model)
    out_merged = model(x)
    assert torch.allclose(out_before, out_merged, atol=1e-6)

    unmerge_adapters(model)
    out_unmerged = model(x)
    assert torch.allclose(out_before, out_unmerged, atol=1e-6)


def test_real_gpt2_injection():
    """Verify LoRA injection into real Hugging Face GPT-2 architecture."""
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=500, bos_token_id=0, eos_token_id=1)
    model = GPT2LMHeadModel(cfg)

    lora_cfg = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["c_attn"])
    model = inject_lora(model, lora_cfg)

    # Check c_attn replaced by LoRAConv1D
    for block in model.transformer.h:
        assert isinstance(block.attn.c_attn, LoRAConv1D)

    stats = count_parameters(model)
    assert stats["percent_trainable"] < 3.0  # < 3% for tiny 2-layer toy model (< 1% for full 124M model)
    assert stats["trainable"] < 3000

    x = torch.randint(0, 500, (2, 8))
    out = model(x)
    assert out.logits.shape == (2, 8, 500)

    out.logits.sum().backward()
    for name, param in model.named_parameters():
        if "lora_B" in name:
            assert param.grad is not None


def test_real_qwen2_injection():
    """Verify LoRA injection into real Hugging Face Qwen2.5 architecture."""
    cfg = Qwen2Config(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = Qwen2ForCausalLM(cfg)

    lora_cfg = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])
    model = inject_lora(model, lora_cfg)

    for layer in model.model.layers:
        assert isinstance(layer.self_attn.q_proj, LoRALinear)
        assert isinstance(layer.self_attn.v_proj, LoRALinear)
        assert isinstance(layer.self_attn.k_proj, nn.Linear)

    stats = count_parameters(model)
    assert stats["percent_trainable"] < 3.0  # < 3% for tiny 2-layer toy model (< 1% for full 124M model)
    assert stats["trainable"] < 3000

    x = torch.randint(0, 500, (2, 8))
    out = model(x)
    assert out.logits.shape == (2, 8, 500)

    out.logits.sum().backward()
    for name, param in model.named_parameters():
        if "lora_B" in name:
            assert param.grad is not None
