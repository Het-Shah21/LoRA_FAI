"""Adapter-only checkpoint utilities for Member C's future demo."""

from pathlib import Path
from typing import Any, Dict

import torch


def adapter_state_dict(model) -> Dict[str, torch.Tensor]:
    """Return only LoRA tensors, detached and moved to CPU."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_" in key
    }


def save_adapter(model, save_path) -> Path:
    """Save strictly adapter weights; the frozen base model is never serialized."""
    state = adapter_state_dict(model)
    if not state:
        raise ValueError("No parameters containing 'lora_' were found")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def _load_state(path) -> Dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older supported torch releases
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise TypeError("Adapter checkpoint must be a state-dict mapping string keys to tensors")
    return state


def load_adapter(model, load_path, strict: bool = True) -> Dict[str, Any]:
    """Load only adapter tensors and report missing/unexpected adapter keys.

    ``strict=True`` raises on either a missing adapter tensor or an unexpected
    checkpoint key. Non-adapter keys are never accepted or applied.
    """
    state = _load_state(load_path)
    checkpoint_keys = set(state)
    model_adapter_keys = {
        key for key in model.state_dict() if "lora_" in key
    }
    unexpected = sorted(checkpoint_keys - model_adapter_keys)
    missing = sorted(model_adapter_keys - checkpoint_keys)
    non_tensor = sorted(key for key, value in state.items() if not torch.is_tensor(value))
    if non_tensor:
        unexpected.extend(non_tensor)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Adapter checkpoint mismatch: missing={missing}, unexpected={sorted(set(unexpected))}"
        )
    compatible = {
        key: value for key, value in state.items()
        if key in model_adapter_keys and torch.is_tensor(value)
    }
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "missing": missing,
        "unexpected": sorted(set(unexpected)),
    }
