"""Standalone verification script validating custom LoRALinear against Hugging Face PEFT.

Runs forward and backward parity checks and prints an audit report.
"""

import math
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from peft import LoraConfig as PEFTLoraConfig
    from peft.tuners.lora import Linear as PEFTLoraLinear
except ImportError:
    print("Error: 'peft' library is required to run parity verification.")
    sys.exit(1)

from src.core.injector import count_parameters, inject_lora
from src.core.lora_layer import LoRAConfig, LoRALinear


def run_forward_parity(dim_in=128, dim_out=256, r=8, alpha=16.0, tol=1e-5):
    """Assert forward output equivalence between custom LoRA and HF PEFT."""
    torch.manual_seed(42)
    base_custom = nn.Linear(dim_in, dim_out, bias=True)
    base_peft = nn.Linear(dim_in, dim_out, bias=True)
    base_peft.weight.data.copy_(base_custom.weight.data)
    base_peft.bias.data.copy_(base_custom.bias.data)

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

    # Use random non-zero weights
    torch.nn.init.normal_(custom_lora.lora_A.weight, std=0.02)
    torch.nn.init.normal_(custom_lora.lora_B.weight, std=0.02)
    peft_lora.lora_A["default"].weight.data.copy_(custom_lora.lora_A.weight.data)
    peft_lora.lora_B["default"].weight.data.copy_(custom_lora.lora_B.weight.data)

    x = torch.randn(4, 16, dim_in)
    out_custom = custom_lora(x)
    out_peft = peft_lora(x)

    max_diff = (out_custom - out_peft).abs().max().item()
    passed = max_diff <= tol
    return passed, max_diff


def run_backward_parity(dim_in=64, dim_out=128, r=4, alpha=8.0, tol=1e-5):
    """Assert backward gradient equivalence between custom LoRA and HF PEFT."""
    torch.manual_seed(999)
    base_custom = nn.Linear(dim_in, dim_out, bias=False)
    base_peft = nn.Linear(dim_in, dim_out, bias=False)
    base_peft.weight.data.copy_(base_custom.weight.data)

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

    torch.nn.init.normal_(custom_lora.lora_A.weight, std=0.05)
    torch.nn.init.normal_(custom_lora.lora_B.weight, std=0.05)
    peft_lora.lora_A["default"].weight.data.copy_(custom_lora.lora_A.weight.data)
    peft_lora.lora_B["default"].weight.data.copy_(custom_lora.lora_B.weight.data)

    x = torch.randn(8, dim_in)
    out_custom = custom_lora(x)
    out_peft = peft_lora(x)

    target = torch.randn_like(out_custom)
    loss_c = F.mse_loss(out_custom, target)
    loss_p = F.mse_loss(out_peft, target)

    loss_c.backward()
    loss_p.backward()

    diff_grad_A = (custom_lora.lora_A.weight.grad - peft_lora.lora_A["default"].weight.grad).abs().max().item()
    diff_grad_B = (custom_lora.lora_B.weight.grad - peft_lora.lora_B["default"].weight.grad).abs().max().item()

    passed = (diff_grad_A <= tol) and (diff_grad_B <= tol)
    return passed, max(diff_grad_A, diff_grad_B)


def run_merge_invariance(dim_in=64, dim_out=64, r=4, tol=1e-6):
    """Assert in-place weight merge invariance and unmerge restoration."""
    torch.manual_seed(123)
    base = nn.Linear(dim_in, dim_out)
    orig_w = base.weight.clone().detach()

    lora = LoRALinear(base_layer=base, r=r, lora_alpha=8.0)
    torch.nn.init.normal_(lora.lora_A.weight, std=0.05)
    torch.nn.init.normal_(lora.lora_B.weight, std=0.05)

    x = torch.randn(4, dim_in)
    out_unmerged = lora(x)

    lora.merge()
    out_merged = lora(x)
    diff_merge = (out_unmerged - out_merged).abs().max().item()

    lora.unmerge()
    drift_unmerge = (lora.base_layer.weight - orig_w).abs().max().item()

    passed = (diff_merge <= tol) and (drift_unmerge <= 1e-7)
    return passed, diff_merge, drift_unmerge


def main():
    print("=" * 70)
    print("  LoRA from Scratch: Mathematical Parity & Invariance Audit")
    print("=" * 70)

    fwd_pass, fwd_diff = run_forward_parity()
    print(f"[{'PASS' if fwd_pass else 'FAIL'}] Forward Logits Parity vs HF PEFT  : Max |Diff| = {fwd_diff:.4e} (tol=1e-5)")

    bwd_pass, bwd_diff = run_backward_parity()
    print(f"[{'PASS' if bwd_pass else 'FAIL'}] Backward Gradient Parity vs PEFT : Max |Diff| = {bwd_diff:.4e} (tol=1e-5)")

    mrg_pass, mrg_diff, unmrg_drift = run_merge_invariance()
    print(f"[{'PASS' if mrg_pass else 'FAIL'}] In-place Merge Invariance        : Max |Diff| = {mrg_diff:.4e} (tol=1e-6)")
    print(f"[{'PASS' if mrg_pass else 'FAIL'}] Unmerge Weight Restoration Drift   : Max |Diff| = {unmrg_drift:.4e} (tol=1e-7)")

    print("-" * 70)
    all_passed = fwd_pass and bwd_pass and mrg_pass
    if all_passed:
        print(">>> ALL VERIFICATION CHECKS PASSED WITH EXTREME NUMERICAL PRECISION <<<")
    else:
        print(">>> SOME VERIFICATION CHECKS FAILED <<<")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
