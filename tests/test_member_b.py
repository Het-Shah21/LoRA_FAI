"""CPU-only integration tests for Member B utilities."""

import torch
from torch import nn

from src.core.lora_layer import LoRALinear
from src.training.baselines import full_finetuning, head_only, zero_shot
from src.training.checkpoint import load_adapter, save_adapter
from src.training.profiler import VRAMProfiler, parameter_counts


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, input_ids, labels=None):
        logits = self.classifier(torch.tanh(self.transformer(input_ids.float())))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def _batches():
    return [
        {"input_ids": torch.randn(4, 4), "labels": torch.tensor([0, 1, 0, 1])},
        {"input_ids": torch.randn(4, 4), "labels": torch.tensor([1, 0, 1, 0])},
    ]


def test_profiler_cpu_schema_and_sampling():
    with VRAMProfiler(tokens=8, device="cpu") as profiler:
        profiler.sample("synthetic", tokens=8)
    record = profiler.as_dict()
    assert record["tokens"] == 8
    assert record["tokens_per_second"] >= 0
    assert record["samples"][0]["allocated_bytes"] == 0


def test_parameter_schema_matches_member_a():
    stats = parameter_counts(nn.Linear(4, 2))
    assert {"trainable", "frozen", "total", "percent_trainable"} <= set(stats)


def test_baselines_use_synthetic_batches_only():
    zero = zero_shot(TinyClassifier(), _batches(), device="cpu")
    assert zero["mode"] == "zero_shot"
    head = head_only(TinyClassifier(), _batches(), epochs=1, device="cpu")
    full = full_finetuning(TinyClassifier(), _batches(), epochs=1, device="cpu")
    assert head["mode"] == "head"
    assert full["mode"] == "full"
    assert head["parameters"]["trainable"] < full["parameters"]["trainable"]


def test_adapter_checkpoint_round_trip(tmp_path):
    first = LoRALinear(nn.Linear(4, 3), r=2)
    nn.init.normal_(first.lora_B.weight, std=0.1)
    path = save_adapter(nn.Sequential(first), tmp_path / "adapter.pt")
    second = LoRALinear(nn.Linear(4, 3), r=2)
    result = load_adapter(nn.Sequential(second), path)
    assert result["missing"] == []
    assert result["unexpected"] == []
    assert torch.equal(first.lora_A.weight, second.lora_A.weight)
    assert torch.equal(first.lora_B.weight, second.lora_B.weight)
