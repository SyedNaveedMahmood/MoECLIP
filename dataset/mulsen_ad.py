"""MulSen-AD RGB/infrared dataset adapter.

The official MulSen-AD release is arranged as::

    <root>/<category>/<RGB|Infrared>/{train,test,GT}/...

Training contains normal RGB and infrared files directly below ``train``.
Test files are below ``test/<anomaly_type>`` and their modality labels are in
``RGB/GT/<anomaly_type>/data.csv``.  The CSV has RGB, infrared and point-cloud
columns.  This adapter deliberately returns ``label == label_rgbt`` (RGB OR
infrared), because this project does not use point clouds; ``label_any`` is
preserved for audits and is never used to label an RGB+IR sample.

SLIC is computed from the transformed, *unnormalized RGB image only*.  Ground
truth masks are never read by the SLIC path.  Resizing to a common square grid
does not establish calibrated RGB/IR registration; it is only the input grid
used by the prototype.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:  # Optional at import time so the RGB baseline does not need skimage.
    from skimage.segmentation import slic
except ImportError:  # pragma: no cover - exercised only in minimal installs
    slic = None


_IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
OFFICIAL_IR_MAX_CHANNEL_DELTA = 20
# Full-archive audit maximum: capsule/Infrared/train/5.png at
# 0.09185872395833333, with maximum channel delta 4.  The independent delta
# cap still rejects materially color-encoded payloads.
OFFICIAL_IR_MAX_NON_GRAY_FRACTION = 0.10


class MulSenIntegrityError(RuntimeError):
    """Raised when the on-disk MulSen-AD structure is ambiguous or incomplete."""


@dataclass(frozen=True)
class MulSenSample:
    category: str
    split: str
    anomaly_type: str
    file_name: str
    image_path: Path
    thermal_path: Path
    mask_rgb_path: Optional[Path]
    mask_thermal_path: Optional[Path]
    label_rgb: int
    label_thermal: int
    label_pointcloud: int

    @property
    def label_rgbt(self) -> int:
        return int(bool(self.label_rgb or self.label_thermal))

    @property
    def label_any(self) -> int:
        return int(bool(self.label_rgb or self.label_thermal or self.label_pointcloud))

    @property
    def key(self) -> str:
        return f"{self.category}/{self.split}/{self.anomaly_type}/{self.file_name}"


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS


def _image_files(path: Path, *, recursive: bool = False) -> List[Path]:
    if not path.is_dir():
        return []
    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted((p for p in iterator if _is_image(p)), key=lambda p: p.name.lower())


def _unique_id_map(paths: Iterable[Path], description: str, *, by_stem: bool = True) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in paths:
        key = path.stem if by_stem else path.name
        if key in result:
            raise MulSenIntegrityError(
                f"Duplicate {description} ID {key!r}: {result[key]} and {path}"
            )
        result[key] = path
    return result


def _parse_label(value: object, field: str, csv_path: Path, row_number: int) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n", ""}:
        return 0
    raise MulSenIntegrityError(
        f"Invalid {field} label {value!r} in {csv_path} row {row_number}"
    )


def _read_labels(csv_path: Path) -> Dict[str, Tuple[int, int, int]]:
    if not csv_path.is_file():
        raise MulSenIntegrityError(f"Missing label CSV: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise MulSenIntegrityError(f"Empty or malformed label CSV: {csv_path}")
    fields = {str(k).strip().lower(): k for k in rows[0].keys() if k is not None}
    required = {"object", "rgb", "infrared", "pointcloud"}
    missing = required.difference(fields)
    if missing:
        raise MulSenIntegrityError(f"CSV {csv_path} lacks columns: {sorted(missing)}")
    result: Dict[str, Tuple[int, int, int]] = {}
    for row_number, row in enumerate(rows, start=2):
        object_id = str(row[fields["object"]]).strip()
        if not object_id:
            raise MulSenIntegrityError(f"Blank object ID in {csv_path} row {row_number}")
        if object_id in result:
            raise MulSenIntegrityError(f"Duplicate object ID {object_id!r} in {csv_path}")
        result[object_id] = (
            _parse_label(row[fields["rgb"]], "RGB", csv_path, row_number),
            _parse_label(row[fields["infrared"]], "infrared", csv_path, row_number),
            _parse_label(row[fields["pointcloud"]], "pointcloud", csv_path, row_number),
        )
    return result


def _open_rgb(path: Path, require_opaque_alpha: bool) -> np.ndarray:
    with Image.open(path) as image:
        native = np.asarray(image)
        mode = image.mode
        if native.dtype != np.uint8:
            raise MulSenIntegrityError(f"RGB image is not uint8: {path} ({native.dtype})")
        if mode == "RGBA":
            if require_opaque_alpha and not np.all(native[..., 3] == 255):
                raise MulSenIntegrityError(f"RGB RGBA image has non-opaque alpha: {path}")
            native = native[..., :3]
        elif mode != "RGB":
            raise MulSenIntegrityError(
                f"RGB image must be native RGB or RGBA, got {mode!r}: {path}"
            )
        return np.ascontiguousarray(native).copy()


def _open_thermal(
    path: Path,
    require_gray: bool,
    max_channel_delta: int,
    max_non_gray_fraction: float,
) -> np.ndarray:
    with Image.open(path) as image:
        native = np.asarray(image)
        mode = image.mode
        if native.dtype != np.uint8:
            raise MulSenIntegrityError(f"Infrared image is not uint8: {path} ({native.dtype})")
        if mode == "RGBA":
            if not np.all(native[..., 3] == 255):
                raise MulSenIntegrityError(f"Infrared RGBA image has non-opaque alpha: {path}")
            native = native[..., :3]
        elif mode == "L":
            return np.ascontiguousarray(native).copy()
        elif mode != "RGB":
            raise MulSenIntegrityError(
                f"Infrared image must be native RGB, RGBA, or L, got {mode!r}: {path}"
            )
        channels = native[..., :3].astype(np.int16)
        channel_span = channels.max(axis=2) - channels.min(axis=2)
        observed_delta = int(channel_span.max())
        non_gray_fraction = float(np.count_nonzero(channel_span) / channel_span.size)
        if require_gray and (
            observed_delta > max_channel_delta
            or non_gray_fraction > max_non_gray_fraction
        ):
            raise MulSenIntegrityError(
                "Infrared RGB payload exceeds grayscale-encoding tolerances: "
                f"{path} (max channel delta={observed_delta}, non-gray fraction="
                f"{non_gray_fraction:.6f}; allowed {max_channel_delta} and "
                f"{max_non_gray_fraction:.6f})"
            )
        # The release is intended to be grayscale but 30/2035 files contain
        # sparse per-channel encoding differences.  Channel mean is symmetric
        # and avoids arbitrarily privileging R, G, or B.  No per-image range
        # normalization is performed.
        grayscale = np.rint(channels.mean(axis=2)).astype(np.uint8)
        return np.ascontiguousarray(grayscale).copy()


def _open_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return np.ascontiguousarray(array).copy()


class MulSenAD(Dataset):
    """Strict paired RGB/IR MulSen-AD loader.

    Args:
        data_root: Directory containing category directories (the extracted
            ``MulSen_AD`` directory, not its parent).
        split: ``"train"`` or ``"test"``.  Training is normal-only.
        categories: Optional explicit category subset.
        train: Enables joint geometry only when true; normally set this to
            ``split == "train"``.  It is separate from ``split`` so callers can
            make a deterministic augmented validation fixture.
        joint_geometry: Apply one sampled flip/rotation/translation to all
            modalities and masks.  It is disabled by default.
        require_ir_grayscale: Validate that decoded IR RGB channels are equal
            or differ only within the two configured encoding tolerances.
            Accepted RGB channels are averaged, divided by 255, and never
            normalized with per-image extrema.
        ir_max_channel_delta: Largest allowed per-pixel RGB channel spread.
            The official archive's audited maximum is 20.
        ir_max_non_gray_fraction: Largest allowed fraction of pixels whose IR
            RGB channels differ.  The archive-wide fraction is about 0.000237.
        region_method: ``None`` or ``"slic"``.  SLIC returns ``LongTensor``
            ``[H,W]``.  When disabled, the ``region_map`` key is omitted so
            PyTorch's default collator never receives a ``None`` value.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        categories: Optional[Sequence[str]] = None,
        img_size: int = 518,
        train: Optional[bool] = None,
        joint_geometry: bool = False,
        geometry_seed: Optional[int] = None,
        geometry_generator: Optional[torch.Generator] = None,
        rotation_degrees: float = 0.0,
        translation_fraction: float = 0.0,
        horizontal_flip_prob: float = 0.0,
        vertical_flip_prob: float = 0.0,
        rgb_mean: Sequence[float] = _CLIP_MEAN,
        rgb_std: Sequence[float] = _CLIP_STD,
        thermal_mean: Optional[float] = None,
        thermal_std: Optional[float] = None,
        require_opaque_alpha: bool = True,
        require_ir_grayscale: bool = True,
        ir_max_channel_delta: int = OFFICIAL_IR_MAX_CHANNEL_DELTA,
        ir_max_non_gray_fraction: float = OFFICIAL_IR_MAX_NON_GRAY_FRACTION,
        region_method: Optional[str] = None,
        slic_segments: int = 64,
        slic_compactness: float = 10.0,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = str(split).lower()
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"MulSen-AD root does not exist: {self.data_root}")
        self.categories = list(categories) if categories is not None else sorted(
            p.name for p in self.data_root.iterdir() if p.is_dir()
        )
        if not self.categories:
            raise MulSenIntegrityError(f"No category directories under {self.data_root}")
        if any(not isinstance(category, str) or not category.strip() for category in self.categories):
            raise ValueError("categories must contain non-empty strings")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must not contain duplicates")
        self.img_size = int(img_size)
        if self.img_size <= 0:
            raise ValueError("img_size must be positive")
        self.train = self.split == "train" if train is None else bool(train)
        self.joint_geometry = bool(joint_geometry)
        self.rotation_degrees = float(rotation_degrees)
        self.translation_fraction = float(translation_fraction)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.vertical_flip_prob = float(vertical_flip_prob)
        self.rgb_mean = torch.tensor(tuple(rgb_mean), dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor(tuple(rgb_std), dtype=torch.float32).view(3, 1, 1)
        if self.rgb_mean.shape != (3, 1, 1) or self.rgb_std.shape != (3, 1, 1):
            raise ValueError("rgb_mean and rgb_std must each have three values")
        if not torch.isfinite(self.rgb_mean).all() or not torch.isfinite(self.rgb_std).all():
            raise ValueError("rgb_mean and rgb_std must be finite")
        if torch.any(self.rgb_std <= 0):
            raise ValueError("rgb_std values must be positive")
        if (thermal_mean is None) != (thermal_std is None):
            raise ValueError("thermal_mean and thermal_std must be supplied together")
        self.thermal_mean = thermal_mean
        self.thermal_std = thermal_std
        if thermal_std is not None and thermal_std <= 0:
            raise ValueError("thermal_std must be positive")
        self.require_opaque_alpha = bool(require_opaque_alpha)
        self.require_ir_grayscale = bool(require_ir_grayscale)
        self.ir_max_channel_delta = int(ir_max_channel_delta)
        self.ir_max_non_gray_fraction = float(ir_max_non_gray_fraction)
        if not 0 <= self.ir_max_channel_delta <= 255:
            raise ValueError("ir_max_channel_delta must be between 0 and 255")
        if not 0.0 <= self.ir_max_non_gray_fraction <= 1.0:
            raise ValueError("ir_max_non_gray_fraction must be between 0 and 1")
        self.region_method = None if region_method in {None, "", "none"} else str(region_method).lower()
        if self.region_method not in {None, "slic"}:
            raise ValueError("region_method must be None or 'slic'")
        if self.region_method == "slic" and slic is None:
            raise ImportError("region_method='slic' requires scikit-image")
        self.slic_segments = int(slic_segments)
        self.slic_compactness = float(slic_compactness)
        if self.slic_segments <= 0 or self.slic_compactness <= 0:
            raise ValueError("slic_segments and slic_compactness must be positive")
        if geometry_seed is not None and geometry_generator is not None:
            raise ValueError("supply geometry_seed or geometry_generator, not both")
        self._geometry_generator = geometry_generator
        if geometry_seed is not None:
            self._geometry_generator = torch.Generator(device="cpu").manual_seed(int(geometry_seed))
        self.warnings: List[str] = []
        self.records: List[MulSenSample] = []
        for category in self.categories:
            self.records.extend(self._index_category(str(category)))
        if not self.records:
            raise MulSenIntegrityError(f"No paired samples found for {self.split} under {self.data_root}")

    @property
    def audit_warnings(self) -> Tuple[str, ...]:
        return tuple(self.warnings)

    def __len__(self) -> int:
        return len(self.records)

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _index_category(self, category: str) -> List[MulSenSample]:
        category_root = self.data_root / category
        if not category_root.is_dir():
            raise MulSenIntegrityError(f"Missing category directory: {category_root}")
        rgb_root, ir_root = category_root / "RGB", category_root / "Infrared"
        if not rgb_root.is_dir() or not ir_root.is_dir():
            raise MulSenIntegrityError(f"Category lacks RGB and Infrared directories: {category_root}")
        if self.split == "train":
            rgb_files = _image_files(rgb_root / "train")
            ir_files = _image_files(ir_root / "train")
            # Pairing is deliberately exact at the relative filename level;
            # ``0.png`` and ``0.bmp`` are not silently treated as the same
            # registered capture.  Mask/CSV IDs below are extensionless.
            rgb_map = _unique_id_map(rgb_files, f"RGB train {category}", by_stem=False)
            ir_map = _unique_id_map(ir_files, f"infrared train {category}", by_stem=False)
            if set(rgb_map) != set(ir_map):
                raise MulSenIntegrityError(
                    f"RGB/infrared train pairing mismatch in {category}: "
                    f"RGB-only={sorted(set(rgb_map)-set(ir_map))}, IR-only={sorted(set(ir_map)-set(rgb_map))}"
                )
            return [
                MulSenSample(category, "train", "good", name, rgb_map[name], ir_map[name], None, None, 0, 0, 0)
                for name in sorted(rgb_map)
            ]

        rgb_test, ir_test = rgb_root / "test", ir_root / "test"
        rgb_types = {p.name for p in rgb_test.iterdir() if p.is_dir()} if rgb_test.is_dir() else set()
        ir_types = {p.name for p in ir_test.iterdir() if p.is_dir()} if ir_test.is_dir() else set()
        if rgb_types != ir_types:
            raise MulSenIntegrityError(
                f"RGB/infrared anomaly-type mismatch in {category}: "
                f"RGB-only={sorted(rgb_types-ir_types)}, IR-only={sorted(ir_types-rgb_types)}"
            )
        records: List[MulSenSample] = []
        for anomaly_type in sorted(rgb_types):
            rgb_files = _image_files(rgb_test / anomaly_type)
            ir_files = _image_files(ir_test / anomaly_type)
            rgb_pairs = _unique_id_map(rgb_files, f"RGB test {category}/{anomaly_type}", by_stem=False)
            ir_pairs = _unique_id_map(ir_files, f"infrared test {category}/{anomaly_type}", by_stem=False)
            if set(rgb_pairs) != set(ir_pairs):
                raise MulSenIntegrityError(
                    f"RGB/infrared pairing mismatch in {category}/{anomaly_type}: "
                    f"RGB-only={sorted(set(rgb_pairs)-set(ir_pairs))}, IR-only={sorted(set(ir_pairs)-set(rgb_pairs))}"
                )
            rgb_map = _unique_id_map(rgb_files, f"RGB test IDs {category}/{anomaly_type}")
            ir_map = _unique_id_map(ir_files, f"infrared test IDs {category}/{anomaly_type}")
            labels = _read_labels(rgb_root / "GT" / anomaly_type / "data.csv")
            if set(labels) != set(rgb_map):
                raise MulSenIntegrityError(
                    f"CSV/images mismatch in {category}/{anomaly_type}: "
                    f"CSV-only={sorted(set(labels)-set(rgb_map))}, image-only={sorted(set(rgb_map)-set(labels))}"
                )
            rgb_masks = _unique_id_map(
                _image_files(rgb_root / "GT" / anomaly_type), f"RGB mask {category}/{anomaly_type}"
            )
            ir_masks = _unique_id_map(
                _image_files(ir_root / "GT" / anomaly_type), f"infrared mask {category}/{anomaly_type}"
            )
            for mask_id in sorted(set(rgb_masks) - set(rgb_map)):
                self._warn(f"Ignoring orphan RGB mask {rgb_masks[mask_id]}")
            for mask_id in sorted(set(ir_masks) - set(ir_map)):
                self._warn(f"Ignoring orphan infrared mask {ir_masks[mask_id]}")
            for name in sorted(rgb_map):
                label_rgb, label_ir, label_pc = labels[name]
                rgb_mask = rgb_masks.get(name)
                ir_mask = ir_masks.get(name)
                if label_rgb and rgb_mask is None:
                    raise MulSenIntegrityError(f"Positive RGB label has no mask: {category}/{anomaly_type}/{name}")
                if label_ir and ir_mask is None:
                    raise MulSenIntegrityError(f"Positive infrared label has no mask: {category}/{anomaly_type}/{name}")
                if not label_rgb and rgb_mask is not None:
                    self._warn(
                        f"Ignoring unexpected RGB mask for a non-positive label: {rgb_mask}"
                    )
                if not label_ir and ir_mask is not None:
                    self._warn(
                        f"Ignoring unexpected infrared mask for a non-positive label: {ir_mask}"
                    )
                records.append(MulSenSample(
                    category, "test", anomaly_type, name, rgb_map[name], ir_map[name],
                    rgb_mask if label_rgb else None, ir_mask if label_ir else None,
                    label_rgb, label_ir, label_pc,
                ))
        return records

    def _geometry(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if not (self.joint_geometry and self.train):
            return tensors
        generator = self._geometry_generator
        rand = lambda: float(torch.rand((), generator=generator).item()) if generator is not None else float(torch.rand(()).item())
        angle = (2.0 * rand() - 1.0) * self.rotation_degrees
        translate = (
            int(round((2.0 * rand() - 1.0) * self.translation_fraction * self.img_size)),
            int(round((2.0 * rand() - 1.0) * self.translation_fraction * self.img_size)),
        )
        hflip = rand() < self.horizontal_flip_prob
        vflip = rand() < self.vertical_flip_prob
        transformed = []
        for index, tensor in enumerate(tensors):
            is_mask = index >= 2
            value = TF.affine(
                tensor, angle=angle, translate=list(translate), scale=1.0, shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR,
                fill=0.0,
            )
            if hflip:
                value = TF.hflip(value)
            if vflip:
                value = TF.vflip(value)
            transformed.append(value)
        return transformed

    def _load(self, record: MulSenSample) -> Mapping[str, object]:
        rgb_array = _open_rgb(record.image_path, self.require_opaque_alpha)
        ir_array = _open_thermal(
            record.thermal_path,
            self.require_ir_grayscale,
            self.ir_max_channel_delta,
            self.ir_max_non_gray_fraction,
        )
        rgb = TF.resize(torch.from_numpy(rgb_array).permute(2, 0, 1).float() / 255.0,
                        [self.img_size, self.img_size], InterpolationMode.BICUBIC, antialias=True)
        thermal = TF.resize(torch.from_numpy(ir_array).unsqueeze(0).float() / 255.0,
                            [self.img_size, self.img_size], InterpolationMode.BILINEAR, antialias=True)

        def load_mask(path: Optional[Path]) -> torch.Tensor:
            if path is None:
                return torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            mask_array = _open_mask(path)
            expected_shape = (
                rgb_array.shape[:2]
                if path == record.mask_rgb_path
                else ir_array.shape[:2]
            )
            if mask_array.shape != expected_shape:
                raise MulSenIntegrityError(
                    f"Mask/image size mismatch for {record.key}: {path} has "
                    f"{mask_array.shape[::-1]}, expected {expected_shape[::-1]}"
                )
            mask = torch.from_numpy(mask_array).unsqueeze(0).float() / 255.0
            mask = TF.resize(mask, [self.img_size, self.img_size], InterpolationMode.NEAREST)
            return (mask > 0).float()

        mask_rgb = load_mask(record.mask_rgb_path)
        mask_ir = load_mask(record.mask_thermal_path)
        rgb, thermal, mask_rgb, mask_ir = self._geometry([rgb, thermal, mask_rgb, mask_ir])
        rgb_unnormalized = rgb.clamp(0.0, 1.0)
        region_map = None
        if self.region_method == "slic":
            region_np = slic(
                rgb_unnormalized.permute(1, 2, 0).numpy(),
                n_segments=self.slic_segments,
                compactness=self.slic_compactness,
                start_label=0,
                channel_axis=-1,
                enforce_connectivity=True,
            )
            _, inverse = np.unique(region_np, return_inverse=True)
            region_map = torch.from_numpy(inverse.reshape(self.img_size, self.img_size).astype(np.int64))
        rgb = (rgb_unnormalized - self.rgb_mean) / self.rgb_std
        if self.thermal_mean is not None:
            thermal = (thermal - float(self.thermal_mean)) / float(self.thermal_std)
        output: Dict[str, object] = {
            "image": rgb,
            "thermal": thermal,
            "mask": mask_rgb,  # RGB-space alias; modality masks are never unioned.
            "mask_rgb": mask_rgb,
            "mask_thermal": mask_ir,
            # `mask` aliases RGB, so its validity must also be RGB-specific.
            "mask_valid": torch.tensor(bool(record.label_rgb), dtype=torch.bool),
            "mask_valid_rgb": torch.tensor(bool(record.label_rgb), dtype=torch.bool),
            "mask_valid_thermal": torch.tensor(bool(record.label_thermal), dtype=torch.bool),
            "label": torch.tensor(record.label_rgbt, dtype=torch.int64),
            "label_rgb": torch.tensor(record.label_rgb, dtype=torch.int64),
            "label_thermal": torch.tensor(record.label_thermal, dtype=torch.int64),
            "label_pointcloud": torch.tensor(record.label_pointcloud, dtype=torch.int64),
            "label_rgbt": torch.tensor(record.label_rgbt, dtype=torch.int64),
            "label_any": torch.tensor(record.label_any, dtype=torch.int64),
            "class_name": record.category,
            "anomaly_type": record.anomaly_type,
            "file_name": record.file_name,
            "sample_key": record.key,
            "source_paths": {
                "rgb": str(record.image_path),
                "thermal": str(record.thermal_path),
                # Empty strings keep the nested mapping compatible with the
                # default DataLoader collator; absence is already explicit in
                # the mask-valid fields.
                "mask_rgb": str(record.mask_rgb_path) if record.mask_rgb_path else "",
                "mask_thermal": str(record.mask_thermal_path) if record.mask_thermal_path else "",
            },
        }
        if region_map is not None:
            output["region_map"] = region_map
        return output

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self._load(self.records[index])


MulSenADDataset = MulSenAD

__all__ = [
    "MulSenAD",
    "MulSenADDataset",
    "MulSenIntegrityError",
    "MulSenSample",
    "OFFICIAL_IR_MAX_CHANNEL_DELTA",
    "OFFICIAL_IR_MAX_NON_GRAY_FRACTION",
]
