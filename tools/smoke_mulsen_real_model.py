"""Run a bounded real-data/real-model MulSen-AD gradient and memory smoke.

This is not an experiment runner. It decodes one locked-protocol sample, makes
one optimizer update to the deliberately zero-initialized context-router head,
then runs a second backward pass to verify that multimodal conditioning becomes
trainable after that stabilization step. No checkpoint or result is written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset.mulsen_protocol import build_training_dataset, get_protocol
from dataset.mulsen_stats import ThermalNormalization
from model.clip import create_model
from model.moe_adapter import MoECLIP
from train_mulsen import batch_loss, configure_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded one-sample MulSen-AD ViT-L/14 CUDA smoke"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--thermal-stats", type=Path, required=True)
    parser.add_argument("--protocol-stage", choices=("development",), default="development")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--use-segment-paa", action="store_true")
    return parser.parse_args()


def _gradient_l1(module: torch.nn.Module) -> float:
    terms = [
        parameter.grad.detach().abs().sum().float()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not terms:
        return 0.0
    return float(torch.stack(terms).sum().item())


def _gradient_report(model: MoECLIP) -> Dict[str, object]:
    context_modules = model.image_adapter["region_contexts"]
    moe_modules = model.image_adapter["moe_adapters"]
    expert_gradients = [
        _gradient_l1(expert)
        for adapter in moe_modules
        for expert in adapter.experts
    ]
    return {
        "thermal_encoder_l1": _gradient_l1(model.thermal_branch),
        "thermal_attention_l1": float(
            sum(_gradient_l1(context.thermal_attention) for context in context_modules)
        ),
        "context_mlp_l1": float(
            sum(_gradient_l1(context.context_mlp) for context in context_modules)
        ),
        "context_router_head_l1": float(
            sum(_gradient_l1(adapter.context_gate) for adapter in moe_modules)
        ),
        "active_rgb_lora_experts": sum(value > 0.0 for value in expert_gradients),
        "rgb_lora_expert_l1": float(sum(expert_gradients)),
        "segmentation_projection_l1": _gradient_l1(
            model.image_adapter["seg_proj"]
        ),
        "text_adapter_l1": _gradient_l1(model.text_adapter),
    }


def _require_finite(report: Dict[str, object], pass_name: str) -> None:
    for key, value in report.items():
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            raise FloatingPointError(
                f"{pass_name} gradient summary is non-finite for {key}: {value}"
            )


def _step(
    model: MoECLIP,
    batch: Dict[str, object],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    img_size: int,
    *,
    update: bool,
    amp_enabled: bool,
) -> Dict[str, object]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp_enabled
    ):
        loss, components = batch_loss(
            model,
            batch,
            device,
            img_size,
            balance_weight=0.01,
            etf_weight=0.01,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite smoke loss: {components}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = _gradient_report(model)
    if update:
        scaler.step(optimizer)
        scaler.update()
    return {
        "loss": float(loss.detach()),
        "components": {
            name: float(value.detach()) for name, value in components.items()
        },
        "gradients": gradients,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this memory smoke requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    protocol = get_protocol(args.protocol_stage)
    normalization = ThermalNormalization.load(
        args.thermal_stats,
        expected_categories=protocol.train_categories,
        expected_stage=protocol.stage,
    )
    dataset = build_training_dataset(
        args.data_root,
        args.protocol_stage,
        img_size=args.img_size,
        use_region_routing=True,
        slic_segments=64,
        slic_compactness=10.0,
        thermal_mean=normalization.mean,
        thermal_std=normalization.std,
        augment=False,
        geometry_seed=args.seed,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"sample-index {args.sample_index} is outside 0..{len(dataset) - 1}"
        )
    sample = dataset[args.sample_index]
    batch = default_collate([sample])

    torch.cuda.empty_cache()
    # The released Windows/PyTorch environment accepts the current-device form
    # but rejects an explicit device argument for the memory-stat APIs.
    torch.cuda.reset_peak_memory_stats()
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = MoECLIP(
        clip_model=clip_model,
        use_paa=True,
        use_segment_paa=args.use_segment_paa,
        seg_proj_sharing_strategy="shared",
        image_adapt_weight=0.1,
        moe_r=8,
        moe_lora_alpha=16,
        moe_num_experts=4,
        moe_top_k=2,
        router_init="normal",
        use_fofs=True,
        moe_layers=[5, 11, 17, 23],
        use_thermal=True,
        use_region_routing=True,
        thermal_depth=4,
        thermal_width=256,
        region_context_dim=256,
        region_attention_heads=4,
        region_coordinate_bias=1.0,
        region_coordinate_sigma=0.75,
        modality_dropout=0.0,
        stable_adapter_norm=True,
        adapter_norm_floor=1.0,
    ).to(device)
    trainable = configure_trainable_parameters(model, use_fofs=True)
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.0)
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0.0:
        raise ValueError("amp-init-scale must be finite and positive")
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp,
        init_scale=args.amp_init_scale,
    )

    output_shapes: Dict[str, object] = {}

    def capture_outputs(_module, _inputs, outputs):
        segmentation, detection, _balance, _etf = outputs
        output_shapes.update(
            {
                "segmentation_maps": len(segmentation),
                "segmentation_shape": list(segmentation[0].shape),
                "detection_shape": list(detection.shape),
            }
        )

    hook = model.register_forward_hook(capture_outputs)
    first = _step(
        model,
        batch,
        optimizer,
        scaler,
        device,
        args.img_size,
        update=True,
        amp_enabled=args.amp,
    )
    second = _step(
        model,
        batch,
        optimizer,
        scaler,
        device,
        args.img_size,
        update=False,
        amp_enabled=args.amp,
    )
    hook.remove()
    torch.cuda.synchronize()

    _require_finite(first["gradients"], "first-pass")
    _require_finite(second["gradients"], "second-pass")
    if first["gradients"]["context_router_head_l1"] <= 0.0:
        raise AssertionError("zero-initialized context router head received no gradient")
    for key in (
        "thermal_encoder_l1",
        "thermal_attention_l1",
        "context_mlp_l1",
        "context_router_head_l1",
        "rgb_lora_expert_l1",
        "segmentation_projection_l1",
        "text_adapter_l1",
    ):
        if second["gradients"][key] <= 0.0:
            raise AssertionError(f"second-pass gradient is zero for {key}")
    if output_shapes.get("segmentation_maps") != 12:
        raise AssertionError(f"expected 12 PAA maps, got {output_shapes}")

    result = {
        "status": "passed",
        "scope": "two backward passes on one sample; one optimizer update; no save",
        "gpu": torch.cuda.get_device_name(0),
        "sample": {
            "dataset_records": len(dataset),
            "index": args.sample_index,
            "category": sample["class_name"],
            "anomaly_type": sample["anomaly_type"],
            "file_name": sample["file_name"],
            "rgb_shape": list(sample["image"].shape),
            "thermal_shape": list(sample["thermal"].shape),
            "region_count": int(sample["region_map"].unique().numel()),
        },
        "normalization": {
            "mean": normalization.mean,
            "std": normalization.std,
            "sample_count": normalization.sample_count,
        },
        "amp_initial_scale": args.amp_init_scale,
        "amp_enabled": args.amp,
        "segment_paa": args.use_segment_paa,
        "outputs": output_shapes,
        "first_pass": first,
        "second_pass": second,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "cuda_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / (1024**2), 2
        ),
        "cuda_peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / (1024**2), 2
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
