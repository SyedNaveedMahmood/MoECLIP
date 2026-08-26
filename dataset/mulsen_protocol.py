"""Locked category-held-out MulSen-AD protocol and dataset composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

from torch.utils.data import ConcatDataset, Dataset

from .mulsen_ad import MulSenAD


ALL_CATEGORIES: Tuple[str, ...] = (
    "button_cell",
    "capsule",
    "cotton",
    "cube",
    "flat_pad",
    "light",
    "nut",
    "piggy",
    "plastic_cylinder",
    "screen",
    "screw",
    "solar_panel",
    "spring_pad",
    "toothbrush",
    "zipper",
)

DEVELOPMENT_TRAIN_CATEGORIES: Tuple[str, ...] = (
    "button_cell",
    "capsule",
    "cube",
    "flat_pad",
    "light",
    "screen",
    "spring_pad",
    "zipper",
)

DEVELOPMENT_VALIDATION_CATEGORIES: Tuple[str, ...] = (
    "plastic_cylinder",
    "screw",
)

FINAL_SEEN_CATEGORIES: Tuple[str, ...] = (
    *DEVELOPMENT_TRAIN_CATEGORIES,
    *DEVELOPMENT_VALIDATION_CATEGORIES,
)

FINAL_UNSEEN_CATEGORIES: Tuple[str, ...] = (
    "cotton",
    "nut",
    "piggy",
    "solar_panel",
    "toothbrush",
)

PROTOCOL_VERSION = "mulsen-rgbt-zsad-v1"


@dataclass(frozen=True)
class MulSenProtocol:
    """Category sets for development selection or locked final evaluation."""

    stage: str
    train_categories: Tuple[str, ...]
    evaluation_categories: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.stage not in {"development", "final"}:
            raise ValueError("stage must be 'development' or 'final'")
        train = set(self.train_categories)
        evaluation = set(self.evaluation_categories)
        if train & evaluation:
            raise ValueError("training and evaluation categories must be disjoint")
        if train | evaluation != set(ALL_CATEGORIES):
            if self.stage == "final":
                raise ValueError("the final protocol must partition all categories")
            expected = set(FINAL_SEEN_CATEGORIES)
            if train | evaluation != expected:
                raise ValueError(
                    "the development protocol must partition final seen categories"
                )


def get_protocol(stage: str) -> MulSenProtocol:
    stage = str(stage).lower()
    if stage == "development":
        return MulSenProtocol(
            stage=stage,
            train_categories=DEVELOPMENT_TRAIN_CATEGORIES,
            evaluation_categories=DEVELOPMENT_VALIDATION_CATEGORIES,
        )
    if stage == "final":
        return MulSenProtocol(
            stage=stage,
            train_categories=FINAL_SEEN_CATEGORIES,
            evaluation_categories=FINAL_UNSEEN_CATEGORIES,
        )
    raise ValueError("stage must be 'development' or 'final'")


class FilteredDataset(Dataset):
    """Index-only view that preserves the strict base loader's output schema."""

    def __init__(self, dataset: MulSenAD, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def _loader_kwargs(
    *,
    img_size: int,
    train: bool,
    use_region_routing: bool,
    slic_segments: int,
    slic_compactness: float,
    thermal_mean: Optional[float] = None,
    thermal_std: Optional[float] = None,
    augment: bool = False,
    geometry_seed: Optional[int] = None,
) -> Mapping[str, object]:
    return {
        "img_size": img_size,
        "train": train,
        "joint_geometry": bool(augment),
        "geometry_seed": geometry_seed,
        "rotation_degrees": 15.0 if augment else 0.0,
        "translation_fraction": 0.1 if augment else 0.0,
        "horizontal_flip_prob": 0.5 if augment else 0.0,
        "vertical_flip_prob": 0.0,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "region_method": "slic" if use_region_routing else None,
        "slic_segments": slic_segments,
        "slic_compactness": slic_compactness,
    }


def build_training_dataset(
    data_root: Union[str, Path],
    protocol_stage: str,
    *,
    img_size: int = 518,
    use_region_routing: bool = True,
    slic_segments: int = 64,
    slic_compactness: float = 10.0,
    thermal_mean: Optional[float] = None,
    thermal_std: Optional[float] = None,
    augment: bool = True,
    geometry_seed: Optional[int] = None,
) -> ConcatDataset:
    """Compose normal train images and visible anomalies from seen categories.

    Official test ``good`` images are not moved into training. Point-cloud-only
    and all-zero anomaly-folder records are excluded, never relabeled.
    """

    protocol = get_protocol(protocol_stage)
    kwargs = _loader_kwargs(
        img_size=img_size,
        train=True,
        use_region_routing=use_region_routing,
        slic_segments=slic_segments,
        slic_compactness=slic_compactness,
        thermal_mean=thermal_mean,
        thermal_std=thermal_std,
        augment=augment,
        geometry_seed=geometry_seed,
    )
    normal = MulSenAD(
        data_root, split="train", categories=protocol.train_categories, **kwargs
    )
    anomaly_base = MulSenAD(
        data_root, split="test", categories=protocol.train_categories, **kwargs
    )
    anomaly_indices = [
        index
        for index, record in enumerate(anomaly_base.records)
        if record.label_rgbt == 1
    ]
    return ConcatDataset([normal, FilteredDataset(anomaly_base, anomaly_indices)])


def build_evaluation_dataset(
    data_root: Union[str, Path],
    protocol_stage: str,
    *,
    img_size: int = 518,
    use_region_routing: bool = True,
    slic_segments: int = 64,
    slic_compactness: float = 10.0,
    thermal_mean: Optional[float] = None,
    thermal_std: Optional[float] = None,
) -> FilteredDataset:
    """Return locked-category good samples and RGB-or-IR-visible anomalies."""

    protocol = get_protocol(protocol_stage)
    kwargs = _loader_kwargs(
        img_size=img_size,
        train=False,
        use_region_routing=use_region_routing,
        slic_segments=slic_segments,
        slic_compactness=slic_compactness,
        thermal_mean=thermal_mean,
        thermal_std=thermal_std,
    )
    base = MulSenAD(
        data_root,
        split="test",
        categories=protocol.evaluation_categories,
        **kwargs,
    )
    indices = [
        index
        for index, record in enumerate(base.records)
        if record.anomaly_type == "good" or record.label_rgbt == 1
    ]
    return FilteredDataset(base, indices)


__all__ = [
    "ALL_CATEGORIES",
    "DEVELOPMENT_TRAIN_CATEGORIES",
    "DEVELOPMENT_VALIDATION_CATEGORIES",
    "FINAL_SEEN_CATEGORIES",
    "FINAL_UNSEEN_CATEGORIES",
    "FilteredDataset",
    "MulSenProtocol",
    "PROTOCOL_VERSION",
    "build_evaluation_dataset",
    "build_training_dataset",
    "get_protocol",
]
