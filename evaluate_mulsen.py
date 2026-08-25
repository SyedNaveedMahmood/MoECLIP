"""Leakage-safe category-held-out evaluation for MulSen-AD checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.mulsen_protocol import PROTOCOL_VERSION, build_evaluation_dataset
from forward_utils import get_adapted_single_class_text_embedding
from model.clip import create_model
from model.moe_adapter import MoECLIP
from mulsen_checkpoint import CHECKPOINT_VERSION, load_mulsen_checkpoint


_EPOCH_PATTERN = re.compile(r"mulsen_epoch_(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate locked category-held-out MulSen-AD checkpoints"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_epoch(path: Path) -> int:
    match = _EPOCH_PATTERN.search(path.name)
    return int(match.group(1)) if match else -1


def discover_checkpoints(
    checkpoint: Path | None, checkpoint_dir: Path | None
) -> Tuple[Path, ...]:
    if checkpoint is not None:
        path = checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        return (path,)
    directory = checkpoint_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {directory}")
    paths = tuple(
        sorted(directory.glob("mulsen_epoch_*.pth"), key=_checkpoint_epoch)
    )
    if not paths:
        raise FileNotFoundError(f"no epoch checkpoints found in {directory}")
    if any(_checkpoint_epoch(path) < 0 for path in paths):
        raise ValueError("checkpoint directory contains an invalid epoch filename")
    return paths


def validate_evaluation_scope(stage: str, checkpoint_count: int) -> None:
    if stage not in {"development", "final"}:
        raise ValueError("checkpoint has an invalid protocol_stage")
    if checkpoint_count <= 0:
        raise ValueError("evaluation requires at least one checkpoint")
    if stage == "final" and checkpoint_count != 1:
        raise ValueError(
            "final unseen categories cannot select among multiple checkpoints"
        )


def select_development_checkpoint(
    evaluations: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not evaluations:
        raise ValueError("no development evaluations are available")
    return max(
        evaluations,
        key=lambda item: (
            item["metrics"]["macro"]["selection_score"],
            -item["epoch"],
        ),
    )


def _read_checkpoint_metadata(path: Path) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version in {path}")
    config = checkpoint.get("experiment_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"checkpoint has no experiment config: {path}")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"checkpoint uses a different MulSen protocol: {path}")
    return {
        "epoch": int(checkpoint["epoch"]),
        "config": dict(config),
    }


def _require_config(config: Mapping[str, object], names: Sequence[str]) -> None:
    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(f"checkpoint config is missing fields: {missing}")


def build_model_from_config(
    config: Mapping[str, object], device: torch.device
) -> MoECLIP:
    _require_config(
        config,
        (
            "model_name",
            "img_size",
            "variant",
            "use_thermal",
            "use_region_routing",
            "use_paa",
            "use_segment_paa",
            "seg_proj_sharing_strategy",
            "image_adapt_weight",
            "moe_r",
            "moe_lora_alpha",
            "moe_num_experts",
            "moe_top_k",
            "moe_layers",
            "router_init",
            "use_fofs",
            "thermal_depth",
            "thermal_width",
            "region_context_dim",
            "region_attention_heads",
            "region_coordinate_bias",
            "region_coordinate_sigma",
            "num_context_experts",
            "modality_dropout",
            "stable_adapter_norm",
            "adapter_norm_floor",
            "relu",
        ),
    )
    clip_model = create_model(
        model_name=str(config["model_name"]),
        img_size=int(config["img_size"]),
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    return MoECLIP(
        clip_model=clip_model,
        use_paa=bool(config["use_paa"]),
        use_segment_paa=bool(config["use_segment_paa"]),
        seg_proj_sharing_strategy=str(config["seg_proj_sharing_strategy"]),
        image_adapt_weight=float(config["image_adapt_weight"]),
        levels=list(config.get("levels", [6, 12, 18, 24])),
        moe_r=int(config["moe_r"]),
        moe_lora_alpha=int(config["moe_lora_alpha"]),
        moe_num_experts=int(config["moe_num_experts"]),
        moe_top_k=int(config["moe_top_k"]),
        router_init=str(config["router_init"]),
        use_fofs=bool(config["use_fofs"]),
        moe_layers=list(config["moe_layers"]),
        relu=bool(config["relu"]),
        use_thermal=bool(config["use_thermal"]),
        use_region_routing=bool(config["use_region_routing"]),
        thermal_depth=int(config["thermal_depth"]),
        thermal_width=int(config["thermal_width"]),
        region_context_dim=int(config["region_context_dim"]),
        region_attention_heads=int(config["region_attention_heads"]),
        region_coordinate_bias=float(config["region_coordinate_bias"]),
        region_coordinate_sigma=float(config["region_coordinate_sigma"]),
        num_context_experts=config["num_context_experts"],
        # ``use_global_context`` was introduced after the released v1
        # checkpoints.  Missing means the original v1 path, preserving
        # reconstruction of legacy configs without weakening required keys.
        use_global_context=bool(config.get("use_global_context", False)),
        modality_dropout=float(config["modality_dropout"]),
        stable_adapter_norm=bool(config["stable_adapter_norm"]),
        adapter_norm_floor=float(config["adapter_norm_floor"]),
    ).to(device)


def _safe_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(values.min())
    span = float(values.max()) - low
    if span <= np.finfo(np.float64).eps:
        return np.zeros_like(values)
    return (values - low) / span


def _normalize_if_outside_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        return _safe_minmax(values)
    return values


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.unique(labels).size != 2:
        raise ValueError("AUROC/AP require both normal and anomalous labels")
    if not np.isfinite(scores).all():
        raise FloatingPointError("evaluation scores contain non-finite values")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def category_metrics(
    *,
    image_labels: np.ndarray,
    detection_scores: np.ndarray,
    pixel_maps: np.ndarray,
    rgb_masks: np.ndarray,
    pixel_valid: np.ndarray,
) -> Dict[str, object]:
    image_labels = np.asarray(image_labels, dtype=np.uint8)
    detection_scores = np.asarray(detection_scores, dtype=np.float64)
    pixel_maps = np.asarray(pixel_maps, dtype=np.float32)
    rgb_masks = np.asarray(rgb_masks, dtype=np.uint8)
    pixel_valid = np.asarray(pixel_valid, dtype=bool)
    if pixel_maps.shape != rgb_masks.shape:
        raise ValueError("pixel maps and RGB masks must have identical shapes")
    if pixel_maps.shape[0] != image_labels.shape[0]:
        raise ValueError("image and pixel sample counts differ")
    if not pixel_valid.any():
        raise ValueError("category has no RGB-valid pixel evaluation samples")

    # Match the released industrial scoring rule: the summed patch map is
    # min-max normalized only when it leaves [0,1], while cosine-derived
    # detection scores normally remain in their native [0,1] range.
    normalized_maps = _normalize_if_outside_unit(pixel_maps)
    patch_max_scores = normalized_maps.reshape(len(pixel_maps), -1).max(axis=1)
    raw_patch_max_scores = pixel_maps.reshape(len(pixel_maps), -1).max(axis=1)
    normalized_detection = _normalize_if_outside_unit(detection_scores)
    combined_scores = 0.5 * patch_max_scores + 0.5 * normalized_detection
    valid_masks = rgb_masks[pixel_valid]
    valid_maps = pixel_maps[pixel_valid]
    return {
        "sample_count": int(len(image_labels)),
        "anomalous_images": int(image_labels.sum()),
        "rgb_pixel_sample_count": int(pixel_valid.sum()),
        "image_combined": _binary_metrics(image_labels, combined_scores),
        "image_detection_only": _binary_metrics(image_labels, detection_scores),
        "rgb_pixel": _binary_metrics(valid_masks, valid_maps),
        "combined_scores": combined_scores.tolist(),
        "detection_scores": detection_scores.tolist(),
        # These per-sample values use exactly the same category-local
        # normalization as the historical combined score.  They are retained
        # for diagnostic reporting only and never affect checkpoint selection.
        "patch_max_scores_raw": raw_patch_max_scores.tolist(),
        "patch_max_scores_normalized": patch_max_scores.tolist(),
    }


def _score_statistics(values: Sequence[float]) -> Dict[str, object]:
    """Return finite, JSON-friendly distribution statistics for scores."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"sample_count": 0, "values": []}
    if not np.isfinite(array).all():
        raise FloatingPointError("diagnostic scores contain non-finite values")
    return {
        "sample_count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def _sample_modality_subgroup(prediction: Mapping[str, object]) -> str:
    """Classify a sample using modality labels, never masks or predictions."""

    rgb = int(prediction["label_rgb"])
    thermal = int(prediction["label_thermal"])
    if rgb == 0 and thermal == 0:
        return "good"
    if rgb == 1 and thermal == 0:
        return "rgb_only"
    if rgb == 0 and thermal == 1:
        return "ir_only"
    if rgb == 1 and thermal == 1:
        return "rgb_ir"
    raise ValueError(
        f"invalid modality labels: label_rgb={rgb}, label_thermal={thermal}"
    )


def subgroup_diagnostics(
    predictions: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Summarize image scores by RGB/IR visibility subgroup.

    This is deliberately diagnostic-only.  The existing combined score and
    selection metric remain implemented by ``category_metrics`` and
    ``summarize_categories`` unchanged.
    """

    required = {
        "category",
        "label_rgb",
        "label_thermal",
        "label_rgbt",
        "detection_score",
        "patch_max_score_raw",
        "patch_max_score_normalized",
        "combined_score",
    }
    groups = {name: [] for name in ("good", "rgb_only", "ir_only", "rgb_ir")}
    by_category: Dict[str, Dict[str, list]] = {}
    for prediction in predictions:
        missing = required.difference(prediction)
        if missing:
            raise KeyError(f"prediction is missing diagnostic fields: {sorted(missing)}")
        subgroup = _sample_modality_subgroup(prediction)
        groups[subgroup].append(prediction)
        category = str(prediction["category"])
        by_category.setdefault(category, {name: [] for name in groups})[subgroup].append(
            prediction
        )

    def summarize(items: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        labels = [int(item["label_rgbt"]) for item in items]
        return {
            "sample_count": len(items),
            "anomalous_images": int(sum(labels)),
            "detection_score": _score_statistics(
                [float(item["detection_score"]) for item in items]
            ),
            "patch_max_score_raw": _score_statistics(
                [float(item["patch_max_score_raw"]) for item in items]
            ),
            "patch_max_score_normalized": _score_statistics(
                [float(item["patch_max_score_normalized"]) for item in items]
            ),
            "combined_score": _score_statistics(
                [float(item["combined_score"]) for item in items]
            ),
        }

    return {
        "subgroups": {name: summarize(items) for name, items in groups.items()},
        "category_subgroups": {
            category: {
                name: summarize(items) for name, items in subgroup_items.items()
            }
            for category, subgroup_items in sorted(by_category.items())
        },
        "detection_only": _binary_metrics(
            np.asarray([int(item["label_rgbt"]) for item in predictions]),
            np.asarray([float(item["detection_score"]) for item in predictions]),
        )
        if predictions
        else None,
    }


def summarize_categories(
    category_results: Mapping[str, Mapping[str, object]]
) -> Dict[str, float]:
    if not category_results:
        raise ValueError("no category results to summarize")
    macro_image = float(
        np.mean(
            [result["image_combined"]["auroc"] for result in category_results.values()]
        )
    )
    macro_pixel = float(
        np.mean([result["rgb_pixel"]["auroc"] for result in category_results.values()])
    )
    return {
        "macro_image_combined_auroc": macro_image,
        "macro_image_combined_ap": float(
            np.mean(
                [
                    result["image_combined"]["average_precision"]
                    for result in category_results.values()
                ]
            )
        ),
        "macro_image_detection_auroc": float(
            np.mean(
                [
                    result["image_detection_only"]["auroc"]
                    for result in category_results.values()
                ]
            )
        ),
        "macro_rgb_pixel_auroc": macro_pixel,
        "macro_rgb_pixel_ap": float(
            np.mean(
                [
                    result["rgb_pixel"]["average_precision"]
                    for result in category_results.values()
                ]
            )
        ),
        "selection_score": 0.5 * (macro_image + macro_pixel),
    }


def _batched_text_features(
    model: MoECLIP, class_names: Sequence[str], device: torch.device
) -> torch.Tensor:
    by_class = {}
    for class_name in dict.fromkeys(class_names):
        by_class[class_name] = get_adapted_single_class_text_embedding(
            model, "MulSenAD", class_name, device
        )
    return torch.stack([by_class[name] for name in class_names], dim=0)


def _pixel_score_maps(
    patch_features: Sequence[torch.Tensor],
    text_features: torch.Tensor,
    img_size: int,
) -> torch.Tensor:
    maps = []
    for features in patch_features:
        logits = 100.0 * torch.bmm(features, text_features)
        side = int(round(features.shape[1] ** 0.5))
        if side * side != features.shape[1]:
            raise RuntimeError("patch features do not form a square grid")
        score = (logits[:, :, 1] + 1.0 - logits[:, :, 0]) / 2.0
        score = score.reshape(features.shape[0], 1, side, side)
        score = gaussian_blur2d(score, (7, 7), (1.0, 1.0))
        maps.append(
            F.interpolate(
                score,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=True,
            )
        )
    return torch.cat(maps, dim=1).sum(dim=1)


def evaluate_checkpoint(
    model: MoECLIP,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    collected: Dict[str, Dict[str, list]] = {}
    predictions: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            class_names = list(batch["class_name"])
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
            text = _batched_text_features(model, class_names, device)
            patch_features, detection, _balance, _etf = model(
                image, thermal=thermal, region_map=region_map
            )
            detection_logits = torch.bmm(detection.unsqueeze(1), text)[:, 0]
            detection_scores = ((detection_logits[:, 1] + 1.0) / 2.0).cpu().numpy()
            pixel_maps = _pixel_score_maps(
                patch_features, text, img_size
            ).cpu().numpy()
            labels = batch["label_rgbt"].cpu().numpy()
            labels_rgb = batch["label_rgb"].cpu().numpy()
            labels_thermal = batch["label_thermal"].cpu().numpy()
            masks = batch["mask_rgb"][:, 0].cpu().numpy().astype(np.uint8)
            anomaly_types = list(batch["anomaly_type"])
            sample_keys = list(batch["sample_key"])

            for index, category in enumerate(class_names):
                pixel_is_valid = anomaly_types[index] == "good" or labels_rgb[index] == 1
                bucket = collected.setdefault(
                    category,
                    {
                        "image_labels": [],
                        "detection_scores": [],
                        "pixel_maps": [],
                        "rgb_masks": [],
                        "pixel_valid": [],
                    },
                )
                bucket["image_labels"].append(int(labels[index]))
                bucket["detection_scores"].append(float(detection_scores[index]))
                bucket["pixel_maps"].append(pixel_maps[index])
                bucket["rgb_masks"].append(masks[index])
                bucket["pixel_valid"].append(bool(pixel_is_valid))
                predictions.append(
                    {
                        "sample_key": sample_keys[index],
                        "category": category,
                        "anomaly_type": anomaly_types[index],
                        "label_rgbt": int(labels[index]),
                        "label_rgb": int(labels_rgb[index]),
                        "label_thermal": int(labels_thermal[index]),
                        "rgb_pixel_metric_included": bool(pixel_is_valid),
                        "detection_score": float(detection_scores[index]),
                        "patch_max_score_raw": float(pixel_maps[index].max()),
                    }
                )

    category_results = {
        category: category_metrics(
            image_labels=np.asarray(bucket["image_labels"]),
            detection_scores=np.asarray(bucket["detection_scores"]),
            pixel_maps=np.stack(bucket["pixel_maps"]),
            rgb_masks=np.stack(bucket["rgb_masks"]),
            pixel_valid=np.asarray(bucket["pixel_valid"]),
        )
        for category, bucket in collected.items()
    }
    # Annotate predictions only after category-local map normalization is
    # known.  This preserves the historical category-level score exactly
    # while making the normalized component inspectable per sample.
    prediction_offsets: Dict[str, int] = {category: 0 for category in category_results}
    for prediction in predictions:
        category = str(prediction["category"])
        offset = prediction_offsets[category]
        result = category_results[category]
        prediction["patch_max_score_normalized"] = float(
            result["patch_max_scores_normalized"][offset]
        )
        prediction["combined_score"] = float(result["combined_scores"][offset])
        prediction_offsets[category] = offset + 1
    diagnostics = subgroup_diagnostics(predictions)
    return {
        "categories": category_results,
        "macro": summarize_categories(category_results),
        "diagnostics": diagnostics,
    }, predictions


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_path}")
    checkpoints = discover_checkpoints(args.checkpoint, args.checkpoint_dir)
    metadata = [_read_checkpoint_metadata(path) for path in checkpoints]
    config = metadata[0]["config"]
    if any(item["config"] != config for item in metadata[1:]):
        raise ValueError("checkpoint directory mixes different experiment configs")
    stage = str(config.get("protocol_stage"))
    validate_evaluation_scope(stage, len(checkpoints))

    thermal_config = config.get("thermal_normalization")
    if bool(config.get("use_thermal")) and not isinstance(thermal_config, Mapping):
        raise ValueError("thermal checkpoint lacks normalization provenance")
    thermal_mean = float(thermal_config["mean"]) if thermal_config else None
    thermal_std = float(thermal_config["std"]) if thermal_config else None
    dataset = build_evaluation_dataset(
        args.data_root,
        stage,
        img_size=int(config["img_size"]),
        use_region_routing=bool(config["use_region_routing"]),
        slic_segments=int(config["slic_segments"]),
        slic_compactness=float(config["slic_compactness"]),
        thermal_mean=thermal_mean,
        thermal_std=thermal_std,
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
    evaluations = []
    for path, item in zip(checkpoints, metadata):
        load_mulsen_checkpoint(
            path,
            model=model,
            expected_config=config,
            restore_rng=False,
            map_location="cpu",
        )
        metrics, predictions = evaluate_checkpoint(
            model, loader, device, int(config["img_size"])
        )
        evaluations.append(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
                "epoch": item["epoch"],
                "metrics": metrics,
                "predictions": predictions,
            }
        )

    selected = None
    if stage == "development":
        selected = select_development_checkpoint(evaluations)
    output = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_stage": stage,
        "metric_policy": {
            "image_labels": "label_rgbt over good and RGB-or-IR-visible anomalies",
            "image_score": "released industrial rule: normalize a component only if outside [0,1], then 0.5 detection + 0.5 max RGB patch map",
            "pixel_labels": "RGB masks for good and RGB-visible anomalies only; IR-only anomalies excluded",
            "selection": "mean of macro image-combined AUROC and macro RGB-pixel AUROC",
            "tie_break": "earliest epoch",
            "diagnostics": "detection-only AUROC/AP and modality-subgroup score distributions are diagnostic only and never affect checkpoint selection",
        },
        "experiment_config": config,
        "evaluation_sample_count": len(dataset),
        "evaluations": evaluations,
        "selected_checkpoint": (
            {
                "checkpoint": selected["checkpoint"],
                "epoch": selected["epoch"],
                "selection_score": selected["metrics"]["macro"]["selection_score"],
            }
            if selected is not None
            else None
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["selected_checkpoint"] or evaluations[0]["metrics"]["macro"], indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
