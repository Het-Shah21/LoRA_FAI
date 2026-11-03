"""Zero-shot, head-only, and full fine-tuning baseline workflows."""

from typing import Any, Dict, Iterable, Optional

import torch

from .profiler import VRAMProfiler, parameter_counts


def _move(batch, device):
    if isinstance(batch, dict):
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    return batch.to(device) if torch.is_tensor(batch) else batch


def _model_output(model, batch):
    return model(**batch) if isinstance(batch, dict) else model(batch)


def _loss_logits(output):
    if isinstance(output, dict):
        return output.get("loss"), output.get("logits")
    return getattr(output, "loss", None), getattr(output, "logits", None)


def _batch_tokens(batch) -> int:
    if isinstance(batch, dict):
        labels = batch.get("labels")
        if torch.is_tensor(labels):
            return int((labels != -100).sum().item())
        input_ids = batch.get("input_ids")
        if torch.is_tensor(input_ids):
            return int(input_ids.numel())
    return 0


def _classification_metrics(logits, labels):
    if logits is None or labels is None or logits.ndim != labels.ndim + 1:
        return 0, 0
    mask = labels != -100
    predictions = logits.argmax(dim=-1)
    return int((predictions[mask] == labels[mask]).sum().item()), int(mask.sum().item())


@torch.no_grad()
def zero_shot(model, dataloader: Iterable, device: Optional[Any] = None, metric_fn=None):
    """Evaluate without backward or optimizer steps.

    Classification batches return accuracy; causal-LM batches return loss and
    perplexity from the model's mean cross-entropy loss.
    """
    device = device or next(model.parameters()).device
    model.eval()
    model.to(device)
    total_correct = total_labels = 0
    losses = []
    tokens = 0
    with VRAMProfiler(device=device) as profiler:
        for batch in dataloader:
            batch = _move(batch, device)
            output = _model_output(model, batch)
            loss, logits = _loss_logits(output)
            if loss is not None:
                losses.append(float(loss.detach().cpu()))
            if isinstance(batch, dict):
                correct, count = _classification_metrics(logits, batch.get("labels"))
                total_correct += correct
                total_labels += count
            tokens += _batch_tokens(batch)
            profiler.tokens = tokens
            profiler.sample("eval_step", tokens=_batch_tokens(batch))
    result = _result("zero_shot", losses, total_correct, total_labels, profiler, model)
    return metric_fn(result) if metric_fn else result


def set_trainable_mode(model, mode: str = "head") -> None:
    """Set exactly the requested baseline's trainable parameters."""
    if mode not in {"head", "full"}:
        raise ValueError("mode must be 'head' or 'full'")
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode == "head":
        head_names = ("classifier", "lm_head", "score", "head", "qa_outputs")
        for name, module in model.named_modules():
            if any(name.lower().endswith(token) for token in head_names):
                for parameter in module.parameters():
                    parameter.requires_grad = True
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("No trainable head was found for head-only baseline")


def _enable_gradient_checkpointing(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()


def _result(mode, losses, correct, count, profiler, model):
    mean_loss = sum(losses) / len(losses) if losses else None
    return {
        "mode": mode,
        "loss": mean_loss,
        "perplexity": float(torch.exp(torch.tensor(mean_loss))) if mean_loss is not None else None,
        "accuracy": correct / count if count else None,
        "parameters": parameter_counts(model),
        "profile": profiler.as_dict(),
        "peak_vram_gb": profiler.allocated_gb,
        "reserved_vram_gb": profiler.reserved_gb,
        "elapsed_seconds": profiler.elapsed_time,
        "tokens_per_second": profiler.tokens_per_second,
    }


def train_baseline(
    model,
    dataloader: Iterable,
    mode: str = "head",
    epochs: int = 1,
    lr: float = 5e-5,
    device: Optional[Any] = None,
    gradient_checkpointing: bool = False,
):
    """Train a baseline with AdamW and return metrics plus profiling data."""
    set_trainable_mode(model, mode)
    if gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    device = device or next(model.parameters()).device
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr
    )
    model.train()
    losses = []
    total_correct = total_labels = 0
    token_count = 0
    with VRAMProfiler(device=device) as profiler:
        for _ in range(epochs):
            for batch in dataloader:
                batch = _move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = _model_output(model, batch)
                loss, logits = _loss_logits(output)
                if loss is None:
                    if torch.is_tensor(output):
                        loss = output
                    else:
                        raise ValueError("Model output must provide .loss or be a scalar tensor")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if isinstance(batch, dict):
                    correct, count = _classification_metrics(logits, batch.get("labels"))
                    total_correct += correct
                    total_labels += count
                step_tokens = _batch_tokens(batch)
                token_count += step_tokens
                profiler.tokens = token_count
                profiler.sample("train_step", tokens=step_tokens)
    return _result(mode, losses, total_correct, total_labels, profiler, model)


def head_only(model, dataloader, **kwargs):
    return train_baseline(model, dataloader, mode="head", **kwargs)


def full_finetuning(model, dataloader, **kwargs):
    return train_baseline(model, dataloader, mode="full", **kwargs)
