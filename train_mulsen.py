"""Train category-held-out segment-guided RGB-thermal MoECLIP.

This script is intentionally separate from ``train.py`` so the released RGB
reproduction command and its historical behavior remain available unchanged.
It never evaluates unseen categories and requires stage-matched thermal
statistics for variants that use IR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.constants import DATA_PATH
from dataset.mulsen_protocol import (
    PROTOCOL_VERSION,
    build_training_dataset,
    get_protocol,
)
from dataset.mulsen_stats import ThermalNormalization
from forward_utils import (
    calculate_seg_loss,
    get_adapted_single_class_text_embedding,
)
from model.clip import create_model
from model.moe_adapter import MoECLIP
from mulsen_checkpoint import load_mulsen_checkpoint, save_mulsen_checkpoint


VARIANT_FLAGS = {
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
    "D": (True, True),
}


def parse_int_list(value: str) -> Tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train region-guided RGB-thermal MoECLIP on MulSen-AD"
    )
    parser.add_argument("--dataset", choices=("MulSenAD",), default="MulSenAD")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(DATA_PATH["MulSenAD"]),
    )
    parser.add_argument(
        "--protocol_stage",
        choices=("development", "final"),
        default="development",
    )
    parser.add_argument("--variant", choices=tuple(VARIANT_FLAGS), default="D")
    parser.add_argument("--thermal_stats", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)

    parser.add_argument("--model_name", default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--moe_r", type=int, default=8)
    parser.add_argument("--moe_lora_alpha", type=int, default=16)
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument(
        "--moe_layers", type=parse_int_list, default=(5, 11, 17, 23)
    )
    parser.add_argument(
        "--router_init", choices=("zero", "normal"), default="normal"
    )
    parser.add_argument("--no_use_fofs", dest="use_fofs", action="store_false")
    parser.set_defaults(use_fofs=True)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--no_use_paa", dest="use_paa", action="store_false")
    parser.set_defaults(use_paa=True)
    parser.add_argument(
        "--seg_proj_sharing_strategy",
        choices=("shared", "separate"),
        default="shared",
    )
    parser.add_argument("--relu", action="store_true")

    parser.add_argument("--region_method", choices=("slic",), default="slic")
    parser.add_argument("--slic_segments", type=int, default=64)
    parser.add_argument("--slic_compactness", type=float, default=10.0)
    parser.add_argument("--thermal_depth", type=int, default=4)
    parser.add_argument("--thermal_width", type=int, default=256)
    parser.add_argument("--region_context_dim", type=int, default=256)
    parser.add_argument("--region_attention_heads", type=int, default=4)
    parser.add_argument("--region_coordinate_bias", type=float, default=1.0)
    parser.add_argument("--region_coordinate_sigma", type=float, default=0.75)
    parser.add_argument("--num_context_experts", type=int)
    parser.add_argument("--modality_dropout", type=float, default=0.2)
    parser.add_argument("--align_loss_lambda", type=float, default=0.0)
    parser.add_argument("--adapter_norm_floor", type=float, default=1.0)
    parser.add_argument(
        "--legacy_adapter_norm",
        dest="stable_adapter_norm",
        action="store_false",
        help="use the released near-zero-singular adapter normalization",
    )
    parser.set_defaults(stable_adapter_norm=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--lr_milestones", type=parse_int_list, default=(12, 16)
    )
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--balance_loss_lambda", type=float, default=0.01)
    parser.add_argument("--etf_loss_lambda", type=float, default=0.01)
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--amp_init_scale",
        type=float,
        default=1024.0,
        help="initial GradScaler scale; 1024 is validated for batch-one ViT-L/14",
    )
    parser.add_argument("--seed", type=int, default=111)
    return parser.parse_args()


def validate_args(args) -> Tuple[bool, bool]:
    use_thermal, use_region_routing = VARIANT_FLAGS[args.variant]
    if use_thermal and args.thermal_stats is None:
        raise ValueError("variants B/D require --thermal_stats")
    if args.align_loss_lambda != 0.0:
        raise ValueError(
            "cross-modal alignment is intentionally disabled in v1; "
            "--align_loss_lambda must be 0"
        )
    if args.img_size <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("img_size, epochs, and batch_size must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if not np.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0.0:
        raise ValueError("amp_init_scale must be finite and positive")
    if not np.isfinite(args.adapter_norm_floor) or args.adapter_norm_floor <= 0.0:
        raise ValueError("adapter_norm_floor must be finite and positive")
    if args.moe_num_experts < args.moe_top_k or args.moe_top_k <= 0:
        raise ValueError("moe_top_k must be within 1..moe_num_experts")
    if args.num_context_experts is not None and not (
        0 <= args.num_context_experts <= args.moe_num_experts
    ):
        raise ValueError("num_context_experts must be within 0..moe_num_experts")
    return use_thermal, use_region_routing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_experiment_config(
    args,
    *,
    use_thermal: bool,
    use_region_routing: bool,
    normalization: Optional[ThermalNormalization],
) -> Dict[str, object]:
    protocol = get_protocol(args.protocol_stage)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_stage": protocol.stage,
        "train_categories": list(protocol.train_categories),
        "evaluation_categories": list(protocol.evaluation_categories),
        "variant": args.variant,
        "use_thermal": use_thermal,
        "use_region_routing": use_region_routing,
        "data_root": str(args.data_root.expanduser().resolve()),
        "model_name": args.model_name,
        "img_size": args.img_size,
        "moe_r": args.moe_r,
        "moe_lora_alpha": args.moe_lora_alpha,
        "moe_num_experts": args.moe_num_experts,
        "moe_top_k": args.moe_top_k,
        "moe_layers": list(args.moe_layers),
        "router_init": args.router_init,
        "use_fofs": args.use_fofs,
        "use_paa": args.use_paa,
        "seg_proj_sharing_strategy": args.seg_proj_sharing_strategy,
        "image_adapt_weight": args.image_adapt_weight,
        "thermal_depth": args.thermal_depth,
        "thermal_width": args.thermal_width,
        "region_context_dim": args.region_context_dim,
        "region_attention_heads": args.region_attention_heads,
        "region_coordinate_bias": args.region_coordinate_bias,
        "region_coordinate_sigma": args.region_coordinate_sigma,
        "num_context_experts": args.num_context_experts,
        "modality_dropout": args.modality_dropout,
        "stable_adapter_norm": args.stable_adapter_norm,
        "adapter_norm_floor": args.adapter_norm_floor,
        "slic_segments": args.slic_segments,
        "slic_compactness": args.slic_compactness,
        "augment": args.augment,
        "thermal_normalization": (
            {
                "mean": normalization.mean,
                "std": normalization.std,
                "sample_count": normalization.sample_count,
                "pixel_count": normalization.pixel_count,
                "file": str(args.thermal_stats.expanduser().resolve()),
                "sha256": _sha256(args.thermal_stats.expanduser().resolve()),
            }
            if normalization is not None
            else None
        ),
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_milestones": list(args.lr_milestones),
        "lr_gamma": args.lr_gamma,
        "balance_loss_lambda": args.balance_loss_lambda,
        "etf_loss_lambda": args.etf_loss_lambda,
        "align_loss_lambda": 0.0,
        "batch_size": args.batch_size,
        "amp": args.amp,
        "amp_init_scale": args.amp_init_scale,
        "seed": args.seed,
    }


def configure_trainable_parameters(model: MoECLIP, use_fofs: bool):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.image_adapter.named_parameters():
        parameter.requires_grad = not (use_fofs and "lora_A" in name)
    for parameter in model.text_adapter.parameters():
        parameter.requires_grad = True
    if model.use_thermal:
        for parameter in model.thermal_branch.parameters():
            parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters were selected")
    return trainable


def batch_loss(model, batch, device, img_size, balance_weight, etf_weight):
    image = batch["image"].to(device, non_blocking=True)
    mask = batch["mask_rgb"].to(device, non_blocking=True)
    label = batch["label_rgbt"].to(device, non_blocking=True)
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
    class_names = list(batch["class_name"])
    text_by_class = {}
    for class_name in dict.fromkeys(class_names):
        text_by_class[class_name] = get_adapted_single_class_text_embedding(
            model, "MulSenAD", class_name, device
        )
    text_features = torch.stack(
        [text_by_class[class_name] for class_name in class_names], dim=0
    )

    patch_features, detection, balance, etf = model(
        image, thermal=thermal, region_map=region_map
    )
    classification = F.cross_entropy(
        torch.matmul(detection.unsqueeze(1), text_features)[:, 0], label
    )
    segmentation = image.new_zeros(())
    for feature in patch_features:
        scores = 100.0 * torch.matmul(feature, text_features)
        side = int(round(feature.shape[1] ** 0.5))
        if side * side != feature.shape[1]:
            raise RuntimeError("segmentation features do not form a square grid")
        scores = scores.permute(0, 2, 1).reshape(
            feature.shape[0], 2, side, side
        )
        scores = F.interpolate(
            scores, size=img_size, mode="bilinear", align_corners=True
        )
        scores = torch.softmax(scores, dim=1)
        # RGB-negative modality records have a valid all-zero RGB target. IR-only
        # anomalies remain image-positive via label_rgbt but are not assigned a
        # fabricated RGB anomaly mask.
        segmentation = segmentation + calculate_seg_loss(scores, mask)
    total = (
        classification
        + segmentation
        + balance_weight * balance
        + etf_weight * etf
    )
    return total, {
        "classification": classification.detach(),
        "segmentation": segmentation.detach(),
        "balance": balance.detach(),
        "etf": etf.detach(),
    }


def main() -> None:
    args = parse_args()
    use_thermal, use_region_routing = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protocol = get_protocol(args.protocol_stage)
    normalization = None
    if use_thermal:
        normalization = ThermalNormalization.load(
            args.thermal_stats,
            expected_categories=protocol.train_categories,
            expected_stage=protocol.stage,
        )
    experiment_config = build_experiment_config(
        args,
        use_thermal=use_thermal,
        use_region_routing=use_region_routing,
        normalization=normalization,
    )

    output_dir = args.output_dir.expanduser().resolve()
    config_path = output_dir / "experiment_config.json"
    if config_path.exists() and args.resume is None:
        raise FileExistsError(
            f"refusing to overwrite an existing experiment: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(experiment_config, indent=2) + "\n", encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("train_mulsen")
    logger.info("experiment config: %s", experiment_config)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = MoECLIP(
        clip_model=clip_model,
        use_paa=args.use_paa,
        seg_proj_sharing_strategy=args.seg_proj_sharing_strategy,
        image_adapt_weight=args.image_adapt_weight,
        moe_r=args.moe_r,
        moe_lora_alpha=args.moe_lora_alpha,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        router_init=args.router_init,
        use_fofs=args.use_fofs,
        moe_layers=list(args.moe_layers),
        relu=args.relu,
        use_thermal=use_thermal,
        use_region_routing=use_region_routing,
        thermal_depth=args.thermal_depth,
        thermal_width=args.thermal_width,
        region_context_dim=args.region_context_dim,
        region_attention_heads=args.region_attention_heads,
        region_coordinate_bias=args.region_coordinate_bias,
        region_coordinate_sigma=args.region_coordinate_sigma,
        num_context_experts=args.num_context_experts,
        modality_dropout=args.modality_dropout,
        stable_adapter_norm=args.stable_adapter_norm,
        adapter_norm_floor=args.adapter_norm_floor,
    ).to(device)
    trainable = configure_trainable_parameters(model, args.use_fofs)
    model.train()
    logger.info(
        "trainable parameters: %d; frozen CLIP eval=%s",
        sum(parameter.numel() for parameter in trainable),
        not model.clipmodel.training,
    )

    dataset = build_training_dataset(
        args.data_root,
        args.protocol_stage,
        img_size=args.img_size,
        use_region_routing=use_region_routing,
        slic_segments=args.slic_segments,
        slic_compactness=args.slic_compactness,
        thermal_mean=normalization.mean if normalization else None,
        thermal_std=normalization.std if normalization else None,
        augment=args.augment,
        geometry_seed=None,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.lr_milestones), gamma=args.lr_gamma
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
        init_scale=args.amp_init_scale,
    )
    start_epoch = 0
    if args.resume is not None:
        checkpoint = load_mulsen_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_config={
                "protocol_version": PROTOCOL_VERSION,
                "protocol_stage": args.protocol_stage,
                "variant": args.variant,
                "model_name": args.model_name,
                "img_size": args.img_size,
            },
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"])
        logger.info("resumed after epoch %d", start_epoch)

    for epoch_index in range(start_epoch, args.epochs):
        model.train()
        running = []
        progress = tqdm(loader, desc=f"epoch {epoch_index + 1}/{args.epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss, components = batch_loss(
                    model,
                    batch,
                    device,
                    args.img_size,
                    args.balance_loss_lambda,
                    args.etf_loss_lambda,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss: {components}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach()))
            progress.set_postfix(
                loss=f"{running[-1]:.4f}",
                balance=f"{float(components['balance']):.4g}",
                etf=f"{float(components['etf']):.4g}",
            )
        scheduler.step()
        completed_epoch = epoch_index + 1
        logger.info(
            "epoch %d mean loss %.6f", completed_epoch, float(np.mean(running))
        )
        epoch_path = output_dir / f"mulsen_epoch_{completed_epoch:03d}.pth"
        checkpoint_args = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": completed_epoch,
            "experiment_config": experiment_config,
        }
        save_mulsen_checkpoint(epoch_path, **checkpoint_args)
        save_mulsen_checkpoint(output_dir / "mulsen_last.pth", **checkpoint_args)


if __name__ == "__main__":
    main()
