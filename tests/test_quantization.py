import torch
import pytest
from src.core.qlora_layer import QLoRALinear, bnb
from src.core.injector import merge_adapters, unmerge_adapters
from src.core.lora_layer import LoRALayer


def test_qlora_forward_and_adapter_gradients():
    layer = QLoRALinear(torch.nn.Linear(8, 4), r=2, use_4bit=False)
    layer(torch.randn(3, 8)).sum().backward()
    assert layer.base_layer.weight.grad is None
    assert layer.lora_A.weight.grad is not None
    assert layer.lora_B.weight.grad is not None


def test_initial_output_matches_base():
    base = torch.nn.Linear(8, 4)
    layer = QLoRALinear(base, r=2, use_4bit=False)
    x = torch.randn(2, 8)
    assert torch.allclose(base(x), layer(x))


def test_qlora_matches_member_a_interface_and_freezes_base():
    layer = QLoRALinear(torch.nn.Linear(8, 4), r=2, lora_alpha=6.0, lora_dropout=0.1)
    assert isinstance(layer, LoRALayer)
    assert (layer.r, layer.lora_alpha, layer.scaling) == (2, 6.0, 3.0)
    assert hasattr(layer, "lora_dropout") and hasattr(layer, "merged")
    assert {name for name, _ in layer.named_parameters()} == {
        "base_layer.weight", "base_layer.bias", "lora_A.weight", "lora_B.weight"
    }
    assert all(not p.requires_grad for p in layer.base_layer.parameters())
    assert all(p.requires_grad for p in layer.lora_A.parameters())
    assert all(p.requires_grad for p in layer.lora_B.parameters())


def test_qlora_adapter_gradients_only():
    layer = QLoRALinear(torch.nn.Linear(8, 4), r=2)
    torch.nn.init.normal_(layer.lora_B.weight, std=0.05)
    layer(torch.randn(3, 8)).sum().backward()
    assert layer.base_layer.weight.grad is None
    assert layer.base_layer.bias.grad is None
    assert layer.lora_A.weight.grad is not None
    assert layer.lora_B.weight.grad is not None


def test_qlora_cpu_merge_unmerge_is_real_and_invariant():
    layer = QLoRALinear(torch.nn.Linear(8, 4), r=2)
    torch.nn.init.normal_(layer.lora_B.weight, std=0.05)
    original = layer.base_layer.weight.detach().clone()
    x = torch.randn(2, 8)
    before = layer(x)
    layer.merge()
    assert layer.merged
    assert not torch.equal(layer.base_layer.weight, original)
    assert torch.allclose(before, layer(x), atol=1e-6)
    layer.unmerge()
    assert not layer.merged
    assert torch.allclose(layer.base_layer.weight, original, atol=1e-7)


def test_qlora_participates_in_global_injector_lifecycle_on_cpu():
    layer = QLoRALinear(torch.nn.Linear(8, 4), r=2)
    model = torch.nn.Sequential(layer)
    torch.nn.init.normal_(layer.lora_B.weight, std=0.05)
    merge_adapters(model)
    assert layer.merged
    unmerge_adapters(model)
    assert not layer.merged


def test_from_linear_cpu_does_not_claim_nf4_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CPU fallback test is not applicable with CUDA")
    layer = QLoRALinear.from_linear(torch.nn.Linear(8, 4), use_4bit=True)
    assert not layer.is_quantized_base


def test_gpt2_conv1d_support_is_intentionally_deferred():
    class Conv1DLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 4))

    with pytest.raises(TypeError, match="LoRAConv1D"):
        QLoRALinear(Conv1DLike(), r=2)


@pytest.mark.skipif(bnb is None or not torch.cuda.is_available(), reason="requires CUDA and bitsandbytes")
def test_nf4_merge_is_rejected_instead_of_faking_a_merge():
    layer = QLoRALinear.from_linear(torch.nn.Linear(8, 4), use_4bit=True)
    assert layer.is_quantized_base
    assert all(not p.requires_grad for p in layer.base_layer.parameters())
    with pytest.raises(RuntimeError, match="NF4"):
        layer.merge()
