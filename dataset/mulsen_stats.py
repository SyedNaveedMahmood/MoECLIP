"""Validated thermal-normalization metadata for MulSen-AD experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence, Tuple, Union

from .mulsen_protocol import PROTOCOL_VERSION


@dataclass(frozen=True)
class ThermalNormalization:
    mean: float
    std: float
    categories: Tuple[str, ...]
    sample_count: int
    pixel_count: int
    protocol_stage: str

    @staticmethod
    def load(
        path: Union[str, Path],
        *,
        expected_categories: Sequence[str],
        expected_stage: str,
    ) -> "ThermalNormalization":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"thermal statistics file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "protocol_version",
            "protocol_stage",
            "source_split",
            "categories",
            "sample_count",
            "pixel_count",
            "mean",
            "std",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"thermal statistics are missing fields: {sorted(missing)}")
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("thermal statistics use a different protocol version")
        if payload["protocol_stage"] != expected_stage:
            raise ValueError("thermal statistics use a different protocol stage")
        if payload["source_split"] != "train-normal-only":
            raise ValueError("thermal statistics were not computed from normal train data")
        categories = tuple(payload["categories"])
        if categories != tuple(expected_categories):
            raise ValueError(
                "thermal-stat categories do not exactly match training categories"
            )
        mean = float(payload["mean"])
        std = float(payload["std"])
        sample_count = int(payload["sample_count"])
        pixel_count = int(payload["pixel_count"])
        if not 0.0 <= mean <= 1.0:
            raise ValueError("thermal mean must be in [0,1]")
        if not 0.0 < std <= 1.0:
            raise ValueError("thermal std must be in (0,1]")
        if sample_count <= 0 or pixel_count <= 0:
            raise ValueError("thermal statistics must contain positive counts")
        return ThermalNormalization(
            mean=mean,
            std=std,
            categories=categories,
            sample_count=sample_count,
            pixel_count=pixel_count,
            protocol_stage=expected_stage,
        )


__all__ = ["ThermalNormalization"]
