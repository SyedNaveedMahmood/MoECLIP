"""Complete checkpoints for the MulSen-AD region-guided extension."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
from torch import nn


CHECKPOINT_VERSION = 1


def _model_components(model: nn.Module) -> Mapping[str, Any]:
    components = {
        "image_adapter": model.image_adapter.state_dict(),
        "text_adapter": model.text_adapter.state_dict(),
    }
    if getattr(model, "use_thermal", False):
        components["thermal_branch"] = model.thermal_branch.state_dict()
    return components


def save_mulsen_checkpoint(
    path: Union[str, Path],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[Any],
    epoch: int,
    experiment_config: Mapping[str, Any],
) -> None:
    """Atomically save every trainable component and training state."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "model_components": _model_components(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "experiment_config": dict(experiment_config),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_mulsen_checkpoint(
    path: Union[str, Path],
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[Any] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
    restore_rng: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> Mapping[str, Any]:
    """Strictly restore components and reject incompatible experiment config."""

    checkpoint = torch.load(
        Path(path).expanduser().resolve(), map_location=map_location
    )
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported MulSen checkpoint version")
    stored_config = checkpoint.get("experiment_config")
    if not isinstance(stored_config, Mapping):
        raise ValueError("checkpoint has no experiment configuration")
    if expected_config is not None:
        for key, expected_value in expected_config.items():
            if stored_config.get(key) != expected_value:
                raise ValueError(
                    f"checkpoint config mismatch for {key!r}: "
                    f"stored={stored_config.get(key)!r}, expected={expected_value!r}"
                )

    components = checkpoint.get("model_components", {})
    model.image_adapter.load_state_dict(components["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(components["text_adapter"], strict=True)
    if getattr(model, "use_thermal", False):
        if "thermal_branch" not in components:
            raise ValueError("thermal model checkpoint has no thermal branch")
        model.thermal_branch.load_state_dict(
            components["thermal_branch"], strict=True
        )
    elif "thermal_branch" in components:
        raise ValueError("RGB-only model cannot load a thermal checkpoint")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if checkpoint["scheduler"] is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    if restore_rng:
        rng = checkpoint["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"].cpu())
        if torch.cuda.is_available() and rng["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "load_mulsen_checkpoint",
    "save_mulsen_checkpoint",
]
