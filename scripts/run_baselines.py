"""CLI entry point for Member B's SST-2 baseline workflows.

The model and dataset are loaded only when this command is executed. Unit
tests can import the runners without triggering downloads.
"""

import argparse
import json
from pathlib import Path


def _load_yaml(path):
    import yaml

    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _build_parser():
    parser = argparse.ArgumentParser(description="Run a zero-shot or fine-tuning baseline.")
    parser.add_argument("--baseline", required=True, choices=("zero_shot", "head_only", "full", "full_finetuning"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.yaml"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import DataCollatorWithPadding

    from src.models.gpt2_wrapper import GPT2AdapterWrapper
    from src.models.qwen_wrapper import Qwen2AdapterWrapper
    from src.core.qlora_layer import load_nf4_config
    from src.training.baselines import full_finetuning, head_only, zero_shot

    args = _build_parser().parse_args(argv)
    config = _load_yaml(args.config) if args.config.exists() else {}
    model_name = args.model or config.get("model", "gpt2")
    batch_size = args.batch_size or config.get("batch_size", 8)
    epochs = args.epochs or config.get("epochs", 1)
    lr = args.lr or config.get("learning_rate", 5e-5)
    device_name = args.device or config.get("device", "auto")
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name)

    wrapper_cls = Qwen2AdapterWrapper if "qwen" in model_name.lower() else GPT2AdapterWrapper
    wrapper = wrapper_cls(model_name, task="sequence_classification")
    load_kwargs = {}
    if config.get("load_in_4bit", False):
        load_kwargs["quantization_config"] = load_nf4_config(Path("configs/qlora_nf4.yaml"))
        load_kwargs["device_map"] = "auto"
    model = wrapper.load_base_model(**load_kwargs)
    tokenizer = wrapper.tokenizer
    dataset = load_dataset("glue", "sst2", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = dataset.map(lambda row: tokenizer(row["sentence"], truncation=True), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=args.baseline != "zero_shot",
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
    )
    if not load_kwargs:
        model.to(device)
    if args.baseline == "zero_shot":
        result = zero_shot(model, loader, device=device)
    elif args.baseline == "head_only":
        result = head_only(model, loader, epochs=epochs, lr=lr, device=device,
                           gradient_checkpointing=args.gradient_checkpointing)
    else:
        result = full_finetuning(model, loader, epochs=epochs, lr=lr, device=device,
                                 gradient_checkpointing=args.gradient_checkpointing)
    print(json.dumps(result, indent=2, default=str))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
