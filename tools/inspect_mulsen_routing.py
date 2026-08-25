"""Read-only routing diagnostics for a trained MulSen-AD checkpoint.

The hooks in this script observe the inputs to each MoE adapter and recompute
its gate logits.  They do not modify model inputs, outputs, parameters, or the
checkpoint.  CLS and patch tokens are reported separately because v1 supplies
region/thermal context only for patch routing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset.mulsen_protocol import build_evaluation_dataset
from evaluate_mulsen import _read_checkpoint_metadata, _sha256, build_model_from_config
from model.moe_adapter import BaseIndependentMoE
from mulsen_checkpoint import load_mulsen_checkpoint


class RoutingStats:
    """Streaming token-level gate statistics for one token subset."""

    def __init__(self, num_experts: int, top_k: int) -> None:
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.token_count = 0
        self.probability_sum = torch.zeros(self.num_experts, dtype=torch.float64)
        self.top1_count = torch.zeros(self.num_experts, dtype=torch.int64)
        self.topk_count = torch.zeros(self.num_experts, dtype=torch.int64)
        self.entropy_sum = 0.0
        self.max_probability_sum = 0.0
        self.topk_mass_sum = 0.0
        self.base_abs_sum = 0.0
        self.context_abs_sum = 0.0
        self.total_abs_sum = 0.0
        self.context_changed_top1 = 0
        self.context_observed = False

    def update(
        self,
        base_logits: torch.Tensor,
        context_logits: Optional[torch.Tensor] = None,
    ) -> None:
        base = base_logits.detach().reshape(-1, self.num_experts).float()
        if base.numel() == 0:
            return
        if context_logits is None:
            context = torch.zeros_like(base)
        else:
            context = context_logits.detach().reshape_as(base).float()
            self.context_observed = True
        total = base + context
        probabilities = F.softmax(total, dim=-1, dtype=torch.float32)
        topk_probabilities, topk_indices = torch.topk(
            probabilities, self.top_k, dim=-1
        )
        top1_indices = probabilities.argmax(dim=-1)

        self.token_count += int(base.shape[0])
        self.probability_sum += probabilities.sum(dim=0).double().cpu()
        self.top1_count += torch.bincount(
            top1_indices, minlength=self.num_experts
        ).cpu()
        self.topk_count += torch.bincount(
            topk_indices.reshape(-1), minlength=self.num_experts
        ).cpu()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        self.entropy_sum += float(entropy.sum().cpu())
        self.max_probability_sum += float(probabilities.max(dim=-1).values.sum().cpu())
        self.topk_mass_sum += float(topk_probabilities.sum(dim=-1).sum().cpu())
        self.base_abs_sum += float(base.abs().sum().cpu())
        self.context_abs_sum += float(context.abs().sum().cpu())
        self.total_abs_sum += float(total.abs().sum().cpu())
        self.context_changed_top1 += int(
            (top1_indices != base.argmax(dim=-1)).sum().cpu()
        )

    @staticmethod
    def _cv_squared(values: torch.Tensor) -> float:
        values = values.double()
        mean = values.mean()
        if float(mean) == 0.0:
            return 0.0
        return float(((values - mean).pow(2).mean() / mean.pow(2)).item())

    def summary(self) -> Dict[str, object]:
        if self.token_count == 0:
            raise RuntimeError("routing statistics contain no tokens")
        mean_probabilities = self.probability_sum / self.token_count
        top1_share = self.top1_count.double() / self.token_count
        topk_share = self.topk_count.double() / (self.token_count * self.top_k)
        distribution_entropy = -(
            mean_probabilities
            * mean_probabilities.clamp_min(1e-12).log()
        ).sum()
        log_experts = math.log(self.num_experts)
        logit_count = self.token_count * self.num_experts
        return {
            "token_count": self.token_count,
            "mean_probabilities": mean_probabilities.tolist(),
            "top1_share": top1_share.tolist(),
            "topk_share": topk_share.tolist(),
            "mean_normalized_token_entropy": self.entropy_sum
            / (self.token_count * log_experts),
            "mean_max_probability": self.max_probability_sum / self.token_count,
            "mean_topk_probability_mass": self.topk_mass_sum / self.token_count,
            "soft_load_cv_squared": self._cv_squared(self.probability_sum),
            "top1_load_cv_squared": self._cv_squared(self.top1_count),
            "effective_experts_from_mean_probabilities": float(
                distribution_entropy.exp().item()
            ),
            "mean_abs_base_logit": self.base_abs_sum / logit_count,
            "mean_abs_context_logit": self.context_abs_sum / logit_count,
            "mean_abs_total_logit": self.total_abs_sum / logit_count,
            "context_to_base_abs_ratio": self.context_abs_sum
            / max(self.base_abs_sum, 1e-12),
            "context_changed_top1_fraction": self.context_changed_top1
            / self.token_count,
            "context_observed": self.context_observed,
        }


class ThermalAttentionStats:
    """Streaming entropy diagnostics over valid RGB-region attention rows."""

    def __init__(self) -> None:
        self.region_count = 0
        self.zero_attention_region_count = 0
        self.thermal_token_count: Optional[int] = None
        self.entropy_sum = 0.0
        self.normalized_entropy_sum = 0.0
        self.effective_token_sum = 0.0
        self.normalized_entropy_min = math.inf
        self.normalized_entropy_max = -math.inf

    def update(self, output) -> None:
        attention = output.thermal_attention.detach().float()
        valid_regions = output.pool.valid_regions.detach().bool()
        if attention.ndim != 3:
            raise ValueError(
                "expected thermal attention [batch,regions,tokens], got "
                f"{tuple(attention.shape)}"
            )
        if valid_regions.shape != attention.shape[:2]:
            raise ValueError(
                "thermal-attention regions do not match the valid-region mask"
            )
        token_count = int(attention.shape[-1])
        if token_count == 0:
            self.zero_attention_region_count += int(valid_regions.sum().cpu())
            return
        if self.thermal_token_count not in (None, token_count):
            raise ValueError("thermal token count changed during routing audit")
        self.thermal_token_count = token_count

        row_sums = attention.sum(dim=-1)
        active = valid_regions & (row_sums > 0)
        self.zero_attention_region_count += int((valid_regions & ~active).sum().cpu())
        probabilities = attention[active]
        if probabilities.numel() == 0:
            return
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        normalized = (
            entropy / math.log(token_count)
            if token_count > 1
            else torch.zeros_like(entropy)
        )
        self.region_count += int(probabilities.shape[0])
        self.entropy_sum += float(entropy.sum().cpu())
        self.normalized_entropy_sum += float(normalized.sum().cpu())
        self.effective_token_sum += float(entropy.exp().sum().cpu())
        self.normalized_entropy_min = min(
            self.normalized_entropy_min, float(normalized.min().cpu())
        )
        self.normalized_entropy_max = max(
            self.normalized_entropy_max, float(normalized.max().cpu())
        )

    def summary(self) -> Optional[Dict[str, object]]:
        if self.thermal_token_count is None:
            return None
        if self.region_count == 0:
            return {
                "region_count": 0,
                "zero_attention_region_count": self.zero_attention_region_count,
                "thermal_token_count": self.thermal_token_count,
            }
        return {
            "region_count": self.region_count,
            "zero_attention_region_count": self.zero_attention_region_count,
            "thermal_token_count": self.thermal_token_count,
            "mean_entropy_nats": self.entropy_sum / self.region_count,
            "mean_normalized_entropy": self.normalized_entropy_sum
            / self.region_count,
            "min_normalized_entropy": self.normalized_entropy_min,
            "max_normalized_entropy": self.normalized_entropy_max,
            "mean_effective_thermal_tokens": self.effective_token_sum
            / self.region_count,
        }


class RoutingCollector:
    """Forward-pre-hook collector for one sequence-first MoE adapter."""

    def __init__(self, module: BaseIndependentMoE) -> None:
        self.module = module
        num_experts = int(module.config.num_experts_)
        top_k = int(module.config.top_k_)
        self.all_tokens = RoutingStats(num_experts, top_k)
        self.class_tokens = RoutingStats(num_experts, top_k)
        self.patch_tokens = RoutingStats(num_experts, top_k)

    def hook(self, module, args, kwargs) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        context = kwargs.get("router_context")
        if hidden.ndim != 3:
            raise ValueError(
                f"expected sequence-first [tokens,batch,width], got {tuple(hidden.shape)}"
            )
        with torch.no_grad():
            flat_hidden = hidden.reshape(-1, module.d_model)
            base = module.gate(flat_hidden).reshape(
                hidden.shape[0], hidden.shape[1], -1
            )
            context_logits = None
            if context is not None:
                flat_context = context.reshape(-1, module.router_context_dim).to(
                    dtype=module.context_gate.weight.dtype
                )
                context_logits = module.context_gate(flat_context)
                context_logits = context_logits * module.context_expert_mask.to(
                    context_logits.dtype
                )
                if module.context_alpha is not None:
                    context_logits = context_logits * module.context_alpha.to(
                        dtype=context_logits.dtype
                    )
                context_logits = context_logits.reshape_as(base).to(base.dtype)
            self.all_tokens.update(base, context_logits)
            self.class_tokens.update(
                base[:1], None if context_logits is None else context_logits[:1]
            )
            self.patch_tokens.update(
                base[1:], None if context_logits is None else context_logits[1:]
            )

    def summary(self, layer_index: int, block_index: int) -> Dict[str, object]:
        base_norm = float(self.module.gate.weight.detach().float().norm().cpu())
        context_norm = (
            float(self.module.context_gate.weight.detach().float().norm().cpu())
            if hasattr(self.module, "context_gate")
            else 0.0
        )
        context_alpha = self.module.context_alpha
        return {
            "adapter_index": layer_index,
            "transformer_block_index": block_index,
            "base_gate_weight_l2": base_norm,
            "context_gate_weight_l2": context_norm,
            "context_to_base_weight_l2_ratio": context_norm / max(base_norm, 1e-12),
            "context_alpha": (
                float(context_alpha.detach().float().cpu())
                if context_alpha is not None
                else None
            ),
            "all_tokens": self.all_tokens.summary(),
            "class_tokens": self.class_tokens.summary(),
            "patch_tokens": self.patch_tokens.summary(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MulSen MoE routing")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite routing report: {output_path}")

    metadata = _read_checkpoint_metadata(checkpoint_path)
    config: Mapping[str, object] = metadata["config"]
    thermal_config = config.get("thermal_normalization")
    if bool(config.get("use_thermal")) and not isinstance(thermal_config, Mapping):
        raise ValueError("thermal checkpoint lacks normalization provenance")
    dataset = build_evaluation_dataset(
        args.data_root,
        str(config["protocol_stage"]),
        img_size=int(config["img_size"]),
        use_region_routing=bool(config["use_region_routing"]),
        slic_segments=int(config["slic_segments"]),
        slic_compactness=float(config["slic_compactness"]),
        thermal_mean=float(thermal_config["mean"]) if thermal_config else None,
        thermal_std=float(thermal_config["std"]) if thermal_config else None,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model_from_config(config, device)
    load_mulsen_checkpoint(
        checkpoint_path,
        model=model,
        expected_config=config,
        restore_rng=False,
        map_location="cpu",
    )
    model.eval()

    adapters = list(model.image_adapter["moe_adapters"])
    collectors = [RoutingCollector(adapter) for adapter in adapters]
    region_contexts = (
        list(model.image_adapter["region_contexts"])
        if "region_contexts" in model.image_adapter
        else []
    )
    attention_collectors = [ThermalAttentionStats() for _ in region_contexts]
    handles = [
        adapter.register_forward_pre_hook(collector.hook, with_kwargs=True)
        for adapter, collector in zip(adapters, collectors)
    ]
    handles.extend(
        module.register_forward_hook(
            lambda _module, _inputs, output, collector=collector: collector.update(
                output
            )
        )
        for module, collector in zip(region_contexts, attention_collectors)
    )
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="routing audit", leave=False):
                image = batch["image"].to(device, non_blocking=True)
                thermal = (
                    batch["thermal"].to(device, non_blocking=True)
                    if model.use_thermal
                    else None
                )
                region_map = (
                    batch["region_map"].to(device, non_blocking=True)
                    if model.use_region_routing
                    else None
                )
                model(image, thermal=thermal, region_map=region_map)
    finally:
        for handle in handles:
            handle.remove()

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "epoch": int(metadata["epoch"]),
        "protocol_version": config["protocol_version"],
        "protocol_stage": config["protocol_stage"],
        "variant": config["variant"],
        "evaluation_sample_count": len(dataset),
        "semantics": {
            "topk_share": "fraction of selected expert slots; sums to one",
            "soft_load_cv_squared": "population CV^2 of accumulated pre-top-k probabilities",
            "context_logits": "context-gate residual after the configured expert mask",
            "class_context": "zero by v1 design; thermal/region context directly conditions patches only",
            "thermal_attention_entropy": "mean over valid RGB regions; each region is weighted equally and entropy is normalized by log(thermal token count)",
        },
        "layers": [
            {
                **collector.summary(index, int(config["moe_layers"][index])),
                "thermal_attention": (
                    attention_collectors[index].summary()
                    if index < len(attention_collectors)
                    else None
                ),
            }
            for index, collector in enumerate(collectors)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variant": config["variant"], "epoch": metadata["epoch"]}, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
