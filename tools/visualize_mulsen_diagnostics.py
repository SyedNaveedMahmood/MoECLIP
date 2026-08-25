"""Read-only qualitative diagnostics for explicitly selected MulSen samples.

This tool is intentionally development-only.  It loads a checkpoint in eval
mode, runs only the exact sample keys supplied by the caller, and writes
figures/JSON; it never consumes anomaly masks while computing model inputs.
Masks are rendered only as clearly labelled evaluation overlays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset.mulsen_protocol import build_evaluation_dataset
from evaluate_mulsen import (
    _batched_text_features,
    _pixel_score_maps,
    _read_checkpoint_metadata,
    _sha256,
    build_model_from_config,
)
from mulsen_checkpoint import load_mulsen_checkpoint


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("._") or "sample"


def validate_visualization_scope(stage: str) -> None:
    """Allow qualitative visualization only on the sealed development set."""

    if str(stage).lower() != "development":
        raise ValueError(
            "qualitative visualization is sealed to the development protocol; "
            "final checkpoints are never scanned"
        )


class _RegionCapture:
    def __init__(self) -> None:
        self.output = None

    def hook(self, module, inputs, output) -> None:
        del module, inputs
        self.output = output


class _RouterCapture:
    """Capture base/context Top-1 assignment for one MoE insertion layer."""

    def __init__(self) -> None:
        self.base_top1: Optional[np.ndarray] = None
        self.context_top1: Optional[np.ndarray] = None
        self.context_changed: Optional[np.ndarray] = None
        self.alpha: Optional[float] = None

    def hook(self, module, args, kwargs) -> None:
        hidden = args[0] if args else kwargs["hidden_states"]
        context = kwargs.get("router_context")
        with torch.no_grad():
            base = module.gate(hidden.reshape(-1, module.d_model))
            base = base.reshape(hidden.shape[0], hidden.shape[1], -1)
            total = base
            if context is not None and hasattr(module, "context_gate"):
                context_flat = context.reshape(-1, module.router_context_dim)
                context_logits = module.context_gate(
                    context_flat.to(dtype=module.context_gate.weight.dtype)
                )
                context_logits = context_logits * module.context_expert_mask.to(
                    dtype=context_logits.dtype
                )
                context_logits = context_logits.reshape_as(base).to(base.dtype)
                if hasattr(module, "context_scale_logit"):
                    context_logits = context_logits * torch.sigmoid(
                        module.context_scale_logit
                    ).to(dtype=context_logits.dtype)
                total = base + context_logits
                self.context_top1 = total.argmax(dim=-1)[1:, 0].cpu().numpy()
                self.context_changed = (
                    self.context_top1 != base.argmax(dim=-1)[1:, 0].cpu().numpy()
                )
            self.base_top1 = base.argmax(dim=-1)[1:, 0].cpu().numpy()
            if hasattr(module, "context_alpha") and module.context_alpha is not None:
                self.alpha = float(module.context_alpha.detach().cpu())


def _to_display_rgb(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=image.dtype)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=image.dtype)
    output = image.detach().cpu() * std[:, None, None] + mean[:, None, None]
    return output.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _to_display_gray(thermal: torch.Tensor) -> np.ndarray:
    output = thermal.detach().cpu().float().squeeze(0).numpy()
    low, high = float(output.min()), float(output.max())
    if high > low:
        output = (output - low) / (high - low)
    else:
        output = np.zeros_like(output)
    return output


def _patch_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    side = int(round(values.size**0.5))
    if side * side != values.size:
        raise ValueError(f"router token count is not square: {values.size}")
    return values.reshape(side, side)


def _attention_display(attention: torch.Tensor, thermal_tokens: int) -> np.ndarray:
    """Collapse region-to-thermal attention into a thermal-grid heat map."""

    values = attention.detach().float().cpu().numpy()
    if values.ndim != 2:
        raise ValueError(f"expected [regions, thermal_tokens], got {values.shape}")
    side = int(round(thermal_tokens**0.5))
    if side * side != thermal_tokens:
        raise ValueError(f"thermal token count is not square: {thermal_tokens}")
    # Each region contributes equally to the displayed aggregate.  This is a
    # visualization of the learned attention, not an anomaly score.
    return values.mean(axis=0).reshape(side, side)


def _make_figure(
    *,
    rgb: np.ndarray,
    thermal: np.ndarray,
    regions: np.ndarray,
    anomaly_map: np.ndarray,
    attention_map: Optional[np.ndarray],
    rgb_mask: np.ndarray,
    thermal_mask: np.ndarray,
    routers: List[_RouterCapture],
    display_layers: List[int],
    output_path: Path,
) -> None:
    expected_layers = [6, 12, 18, 24]
    if len(routers) != 4 or len(display_layers) != 4:
        raise ValueError(
            "visualization requires exactly four MoE layers, labeled 6/12/18/24"
        )
    if list(display_layers) != expected_layers:
        raise ValueError(
            "visualization requires MoE implementation layers 5/11/17/23 "
            "(display labels 6/12/18/24)"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    axes = axes.reshape(-1)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB input")
    axes[1].imshow(thermal, cmap="inferno")
    axes[1].set_title("Infrared input (display-scaled)")
    axes[2].imshow(regions, cmap="tab20")
    axes[2].set_title("SLIC regions")
    image = axes[3].imshow(anomaly_map, cmap="magma")
    axes[3].set_title("RGB anomaly map (model output)")
    figure.colorbar(image, ax=axes[3], fraction=0.046)

    if attention_map is None:
        axes[4].axis("off")
        axes[4].set_title("Region-to-thermal attention unavailable")
    else:
        image = axes[4].imshow(attention_map, cmap="viridis")
        axes[4].set_title("Region-to-thermal attention\n(visualization only)")
        figure.colorbar(image, ax=axes[4], fraction=0.046)

    axes[5].imshow(rgb)
    axes[5].imshow(rgb_mask, cmap="Reds", alpha=(rgb_mask > 0) * 0.6)
    axes[5].set_title("RGB GT mask (evaluation-only overlay)")
    axes[6].imshow(thermal, cmap="inferno")
    axes[6].imshow(thermal_mask, cmap="Blues", alpha=(thermal_mask > 0) * 0.6)
    axes[6].set_title("IR GT mask (evaluation-only overlay)")
    axes[7].axis("off")

    for axis_index, (layer, capture) in enumerate(
        zip(display_layers, routers), start=8
    ):
        assignment_values = (
            capture.context_top1
            if capture.context_top1 is not None
            else capture.base_top1
        )
        if assignment_values is None:
            axes[axis_index].axis("off")
            axes[axis_index].set_title(f"Layer {layer}: routing unavailable")
            continue
        assignment = _patch_map(assignment_values)
        axes[axis_index].imshow(assignment, cmap="tab10", interpolation="nearest")
        if capture.context_top1 is None:
            axes[axis_index].set_title(f"Layer {layer}: base Top-1")
        else:
            changed = (
                int(capture.context_changed.sum())
                if capture.context_changed is not None
                else 0
            )
            total = (
                int(capture.context_changed.size)
                if capture.context_changed is not None
                else 0
            )
            suffix = f"; changed {changed}/{total}" if total else ""
            axes[axis_index].set_title(f"Layer {layer}: context Top-1{suffix}")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--sample_key",
        action="append",
        required=True,
        help="Exact development sample key; repeat for multiple selected samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata = _read_checkpoint_metadata(checkpoint_path)
    config: Mapping[str, object] = metadata["config"]
    validate_visualization_scope(str(config.get("protocol_stage")))
    requested_keys = list(dict.fromkeys(str(key) for key in args.sample_key))
    if not requested_keys:
        raise ValueError("at least one explicit --sample_key is required")

    thermal_config = config.get("thermal_normalization")
    if bool(config.get("use_thermal")) and not isinstance(thermal_config, Mapping):
        raise ValueError("thermal checkpoint lacks normalization provenance")
    dataset = build_evaluation_dataset(
        args.data_root,
        "development",
        img_size=int(config["img_size"]),
        use_region_routing=bool(config["use_region_routing"]),
        slic_segments=int(config["slic_segments"]),
        slic_compactness=float(config["slic_compactness"]),
        thermal_mean=float(thermal_config["mean"]) if thermal_config else None,
        thermal_std=float(thermal_config["std"]) if thermal_config else None,
    )
    selected = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        key = str(sample["sample_key"])
        if key in requested_keys:
            selected[key] = sample
    missing = [key for key in requested_keys if key not in selected]
    if missing:
        raise KeyError(
            "requested sample key(s) are not in the development validation set: "
            + ", ".join(missing)
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config, device)
    load_mulsen_checkpoint(
        checkpoint_path,
        model=model,
        expected_config=config,
        restore_rng=False,
        map_location="cpu",
    )
    model.eval()
    display_layers = [int(layer) + 1 for layer in config["moe_layers"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol_stage": "development",
        "sample_count": len(selected),
        "samples": [],
        "semantics": {
            "router_layers": "implementation block index + 1 (5,11,17,23 -> 6,12,18,24)",
            "gt_masks": "shown only as evaluation-only overlays; never model inputs",
            "thermal_display": "per-image display scaling; not model normalization",
        },
    }

    for key in requested_keys:
        sample = selected[key]
        image = sample["image"].unsqueeze(0).to(device)
        thermal = sample["thermal"].unsqueeze(0).to(device) if model.use_thermal else None
        region_map = (
            sample["region_map"].unsqueeze(0).to(device)
            if model.use_region_routing
            else None
        )
        region_captures = [_RegionCapture() for _ in model.image_adapter["region_contexts"]] if model.use_context_routing else []
        router_captures = [_RouterCapture() for _ in model.image_adapter["moe_adapters"]]
        handles = []
        if model.use_context_routing:
            handles.extend(
                module.register_forward_hook(capture.hook)
                for module, capture in zip(
                    model.image_adapter["region_contexts"], region_captures
                )
            )
        handles.extend(
            module.register_forward_pre_hook(capture.hook, with_kwargs=True)
            for module, capture in zip(
                model.image_adapter["moe_adapters"], router_captures
            )
        )
        try:
            with torch.no_grad():
                text = _batched_text_features(
                    model, [str(sample["class_name"])], device
                )
                patch_features, detection, _balance, _etf = model(
                    image, thermal=thermal, region_map=region_map
                )
                anomaly_map = _pixel_score_maps(
                    patch_features, text, int(config["img_size"])
                )[0].cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()

        attention_map = None
        if region_captures:
            output = region_captures[-1].output
            if output is not None and output.thermal_attention.numel():
                attention = output.thermal_attention[0]
                attention_map = _attention_display(attention, attention.shape[-1])

        sample_name = _safe_name(key)
        figure_path = output_dir / f"{sample_name}.png"
        _make_figure(
            rgb=_to_display_rgb(sample["image"]),
            thermal=_to_display_gray(sample["thermal"]),
            regions=sample.get("region_map", torch.zeros_like(sample["mask_rgb"][0])).numpy(),
            anomaly_map=anomaly_map,
            attention_map=attention_map,
            rgb_mask=sample["mask_rgb"][0].numpy(),
            thermal_mask=sample["mask_thermal"][0].numpy(),
            routers=router_captures,
            display_layers=display_layers,
            output_path=figure_path,
        )
        json_sample = {
            "sample_key": key,
            "category": str(sample["class_name"]),
            "anomaly_type": str(sample["anomaly_type"]),
            "label_rgb": int(sample["label_rgb"]),
            "label_thermal": int(sample["label_thermal"]),
            "label_rgbt": int(sample["label_rgbt"]),
            "figure": str(figure_path),
            "router_layers": {
                str(layer): {
                    "base_top1": capture.base_top1.tolist() if capture.base_top1 is not None else None,
                    "context_top1": capture.context_top1.tolist() if capture.context_top1 is not None else None,
                    "context_changed_top1": capture.context_changed.tolist() if capture.context_changed is not None else None,
                    "context_alpha": capture.alpha,
                }
                for layer, capture in zip(display_layers, router_captures)
            },
            "region_to_thermal_attention": (
                attention_map.tolist() if attention_map is not None else None
            ),
        }
        report["samples"].append(json_sample)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": len(selected), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
