"""Read-only RGB/Infrared pairing and alignment audit for MulSen-AD.

The script deliberately treats the RGB and Infrared PNGs as independent input
modalities.  It never opens files below ``*/GT`` while making image statistics,
edges, contact sheets, or shift estimates.  CSV files below ``*/GT`` are read
separately as label metadata and are never used to form image diagnostics.

The diagnostic overlay has one explicit, limited convention: each RGB image is
resized with PIL bilinear interpolation to the native Infrared width/height;
the Infrared image is not warped.  Edges are then computed on those two images
on the common grid.  The reported shift is a small-window edge-overlap
diagnostic in Infrared pixels, *not* a camera calibration or registration claim.

Example (PowerShell):

    python tools/inspect_mulsen_alignment.py `
      --data-root data/MulSenAD_official/MulSen_AD `
      --output-dir artifacts/mulsen_alignment `
      --sample-count 16 --seed 111

The output directory must be outside ``--data-root``.  No source file is
written or modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover - requirements.txt includes OpenCV
    cv2 = None


PNG_SUFFIXES = {".png"}
MODALITIES = ("RGB", "Infrared")
SPLITS = ("train", "test")
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
SCRIPT_VERSION = "1.2"


@dataclass(frozen=True, order=True)
class PairKey:
    category: str
    split: str
    anomaly_type: Optional[str]
    filename: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "split": self.split,
            "anomaly_type": self.anomaly_type,
            "filename": self.filename,
        }

    def compact(self) -> str:
        anomaly = self.anomaly_type if self.anomaly_type is not None else "<none>"
        return f"{self.category}/{self.split}/{anomaly}/{self.filename}"


@dataclass(frozen=True)
class PairRecord:
    key: PairKey
    rgb_path: Path
    infrared_path: Path


class IntegrityError(RuntimeError):
    """Raised after a report has been written when source pairing is invalid."""


def _jsonable_counter(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _assert_non_gt_input(path: Path) -> None:
    """Guard the input-only diagnostic path against accidental GT reads."""

    if any(part.casefold() == "gt" for part in path.parts):
        raise AssertionError(f"GT path is forbidden for image diagnostics: {path}")


def _modality_map(
    category_root: Path,
    category: str,
    modality: str,
) -> Tuple[Dict[PairKey, List[Path]], List[str]]:
    """Index non-GT PNGs under one category/modality by semantic path key."""

    modality_root = category_root / modality
    if not modality_root.is_dir():
        return {}, [f"missing modality directory: {modality_root}"]

    indexed: Dict[PairKey, List[Path]] = defaultdict(list)
    ignored: List[str] = []
    for split in SPLITS:
        split_root = modality_root / split
        if not split_root.is_dir():
            ignored.append(f"missing split directory: {split_root}")
            continue
        for path in sorted(split_root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in PNG_SUFFIXES:
                continue
            _assert_non_gt_input(path)
            rel = path.relative_to(split_root)
            if split == "test":
                if len(rel.parts) < 2:
                    ignored.append(f"test PNG has no anomaly_type directory: {path}")
                    continue
                anomaly_type: Optional[str] = rel.parts[0]
            else:
                # MulSen-AD stores train PNGs directly under train.  Preserve a
                # nested first component if a future release adds one.
                anomaly_type = rel.parts[0] if len(rel.parts) > 1 else None
            key = PairKey(category, split, anomaly_type, path.name)
            indexed[key].append(path)
    return dict(indexed), ignored


def discover_pairs(data_root: Path) -> Tuple[List[PairRecord], Dict[str, Any]]:
    """Discover exact RGB/Infrared pairs and return a loud integrity report."""

    if not data_root.is_dir():
        raise FileNotFoundError(f"MulSen data root does not exist: {data_root}")
    categories = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    if not categories:
        raise FileNotFoundError(f"No category directories under {data_root}")

    pairs: List[PairRecord] = []
    duplicate_entries: List[Dict[str, Any]] = []
    missing_rgb: List[PairKey] = []
    missing_infrared: List[PairKey] = []
    source_issues: List[str] = []
    inventory: Dict[str, Any] = {}

    for category in categories:
        category_root = data_root / category
        maps: Dict[str, Dict[PairKey, List[Path]]] = {}
        ignored_by_modality: Dict[str, List[str]] = {}
        for modality in MODALITIES:
            maps[modality], ignored_by_modality[modality] = _modality_map(
                category_root, category, modality
            )
            for key, paths in maps[modality].items():
                if len(paths) > 1:
                    duplicate_entries.append(
                        {
                            "modality": modality,
                            "key": key.as_dict(),
                            "paths": [str(p) for p in paths],
                        }
                    )

        rgb_keys = set(maps["RGB"])
        infrared_keys = set(maps["Infrared"])
        for key in sorted(rgb_keys - infrared_keys):
            missing_infrared.append(key)
        for key in sorted(infrared_keys - rgb_keys):
            missing_rgb.append(key)
        for key in sorted(rgb_keys & infrared_keys):
            # Duplicates are retained in the report and also make the run fail;
            # use the first path only to keep a useful partial report.
            pairs.append(PairRecord(key, maps["RGB"][key][0], maps["Infrared"][key][0]))

        inventory[category] = {
            "RGB_png_count": sum(len(v) for v in maps["RGB"].values()),
            "Infrared_png_count": sum(len(v) for v in maps["Infrared"].values()),
            "paired_key_count": len(rgb_keys & infrared_keys),
            "ignored_or_missing_directories": ignored_by_modality,
        }

    if duplicate_entries:
        source_issues.append(f"duplicate modality keys: {len(duplicate_entries)}")
    if missing_rgb:
        source_issues.append(f"missing RGB counterparts: {len(missing_rgb)}")
    if missing_infrared:
        source_issues.append(f"missing Infrared counterparts: {len(missing_infrared)}")

    integrity = {
        "status": "ok" if not source_issues else "failed",
        "category_count": len(categories),
        "categories": categories,
        "paired_count": len(pairs),
        "issues": source_issues,
        "missing_rgb": [key.as_dict() for key in missing_rgb],
        "missing_infrared": [key.as_dict() for key in missing_infrared],
        "duplicates": duplicate_entries,
        "inventory": inventory,
    }
    return sorted(pairs, key=lambda pair: pair.key), integrity


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _casefold_column(fieldnames: Sequence[str], wanted: str) -> Optional[str]:
    wanted = wanted.casefold()
    for field in fieldnames:
        if field.casefold() == wanted:
            return field
    return None


def _label_semantic(value_counts: Counter) -> str:
    values = set(value_counts)
    if not values:
        return "empty"
    if values <= {"0", "1"}:
        return "binary indicator (0/1); meaning is retained as the CSV column name"
    return "non-binary/categorical values; no coercion performed"


def _read_label_csv(path: Path, data_root: Path) -> Dict[str, Any]:
    """Read one CSV as metadata; never use it to create image diagnostics."""

    rel = path.relative_to(data_root)
    parts = rel.parts
    category = parts[0] if len(parts) > 0 else "<unknown>"
    source = parts[1] if len(parts) > 1 else "<unknown>"
    anomaly_type = parts[3] if len(parts) > 3 else "<unknown>"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    object_column = _casefold_column(fieldnames, "object")
    values_by_column: Dict[str, Counter] = {}
    for field in fieldnames:
        values_by_column[field] = Counter(_csv_value(row.get(field)) for row in rows)
    object_values = [_csv_value(row.get(object_column)) for row in rows] if object_column else []
    duplicate_objects = sorted(value for value, count in Counter(object_values).items() if count > 1)

    semantics = {
        field: {
            "unique_values": _jsonable_counter(values_by_column[field]),
            "semantic_summary": _label_semantic(values_by_column[field]),
        }
        for field in fieldnames
        if field != object_column
    }
    return {
        "path": _relative(path, data_root),
        "category": category,
        "source_directory": source,
        "anomaly_type": anomaly_type,
        "columns": fieldnames,
        "row_count": len(rows),
        "object_column": object_column,
        "object_ids": object_values,
        "duplicate_object_ids": duplicate_objects,
        "columns_semantics": semantics,
        "rows": rows,
    }


def _positive_label(value: Any) -> Optional[bool]:
    """Interpret the release's binary label convention without coercing unknowns."""

    text = _csv_value(value).casefold()
    if text in {"1", "true", "yes", "positive"}:
        return True
    if text in {"0", "false", "no", "negative", ""}:
        return False
    return None


def _test_image_index(data_root: Path, category: str, modality: str) -> Dict[Tuple[str, str], Dict[str, Path]]:
    """Index test images by (anomaly_type, object filename), without reading pixels."""

    root = data_root / category / modality / "test"
    result: Dict[Tuple[str, str], Dict[str, Path]] = defaultdict(dict)
    if not root.is_dir():
        return result
    for anomaly_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(anomaly_dir.iterdir()):
            if not path.is_file() or path.suffix.casefold() != ".png":
                continue
            _assert_non_gt_input(path)
            result[(anomaly_dir.name, path.name)][modality] = path
    return result


def _mask_index(data_root: Path, category: str, modality: str) -> Dict[Tuple[str, str], Path]:
    """Index mask filenames only; mask pixels are deliberately never opened."""

    root = data_root / category / modality / "GT"
    result: Dict[Tuple[str, str], Path] = {}
    if not root.is_dir():
        return result
    for anomaly_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(anomaly_dir.iterdir()):
            if not path.is_file() or path.suffix.casefold() != ".png":
                continue
            result[(anomaly_dir.name, path.name)] = path
    return result


def _reconcile_mask_labels(data_root: Path, summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconcile CSV metadata and mask *paths* for every test row.

    This is intentionally metadata-only: no mask image is opened and no mask
    path or mask content participates in edge computation or sample selection.
    Positive labels without masks and CSV/image pairing mismatches are errors.
    Orphan or unexpected masks are warnings because the released archive is
    known to contain at least one such artifact (nut/RGB/GT/color/8.png).
    """

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    csv_rows_by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    csv_ids_by_file: Dict[str, List[str]] = {}
    modalities = {"RGB": "RGB", "infrared": "Infrared"}

    def issue(target: List[Dict[str, Any]], kind: str, **fields: Any) -> None:
        target.append({"kind": kind, **fields})

    for summary in summaries:
        category = str(summary["category"])
        anomaly = str(summary["anomaly_type"])
        csv_path = str(summary["path"])
        fields = summary["columns"]
        object_column = summary["object_column"]
        if not object_column:
            issue(errors, "csv_missing_object_column", csv_path=csv_path)
            continue
        csv_ids = [_csv_value(row.get(object_column)) for row in summary["rows"]]
        csv_ids_by_file[csv_path] = csv_ids
        duplicate_ids = sorted(value for value, count in Counter(csv_ids).items() if count > 1)
        for object_id in duplicate_ids:
            issue(errors, "duplicate_csv_object_id", csv_path=csv_path, object_id=object_id)
        for row in summary["rows"]:
            object_id = _csv_value(row.get(object_column))
            csv_rows_by_key[(category, anomaly, object_id)].append(
                {"summary": summary, "row": row, "csv_path": csv_path}
            )

        image_indexes = {
            modality: _test_image_index(data_root, category, modality)
            for modality in MODALITIES
        }
        mask_indexes = {
            modality: _mask_index(data_root, category, modality)
            for modality in MODALITIES
        }
        for row in summary["rows"]:
            object_id = _csv_value(row.get(object_column))
            key = (category, anomaly, object_id)
            for column_name, modality in modalities.items():
                column = _casefold_column(fields, column_name)
                if not column:
                    issue(errors, "csv_missing_modality_column", csv_path=csv_path, column=column_name)
                    continue
                value = _csv_value(row.get(column))
                positive = _positive_label(value)
                if positive is None:
                    issue(errors, "unknown_modality_label", csv_path=csv_path, object_id=object_id,
                          modality=modality, value=value)
                    continue
                image_entry = image_indexes[modality].get((anomaly, f"{object_id}.png"))
                if not image_entry:
                    issue(errors, "csv_row_without_test_image", csv_path=csv_path, category=category,
                          anomaly_type=anomaly, object_id=object_id, modality=modality)
                mask_path = mask_indexes[modality].get((anomaly, f"{object_id}.png"))
                if positive and mask_path is None:
                    issue(errors, "positive_label_missing_mask", csv_path=csv_path, category=category,
                          anomaly_type=anomaly, object_id=object_id, modality=modality)
                if not positive and mask_path is not None:
                    issue(warnings, "negative_label_has_unexpected_mask", csv_path=csv_path,
                          category=category, anomaly_type=anomaly, object_id=object_id,
                          modality=modality, mask_path=_relative(mask_path, data_root))

    categories = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    for category in categories:
        for modality in MODALITIES:
            image_index = _test_image_index(data_root, category, modality)
            mask_index = _mask_index(data_root, category, modality)
            for (anomaly, filename), path_map in sorted(image_index.items()):
                if not csv_rows_by_key.get((category, anomaly, filename.removesuffix(".png"))):
                    issue(errors, "test_image_without_csv_row", category=category,
                          anomaly_type=anomaly, object_id=filename.removesuffix(".png"),
                          modality=modality, image_path=_relative(path_map[modality], data_root))
            for (anomaly, filename), mask_path in sorted(mask_index.items()):
                rows = csv_rows_by_key.get((category, anomaly, filename.removesuffix(".png")), [])
                matching_column = "RGB" if modality == "RGB" else "infrared"
                positive = False
                for entry in rows:
                    column = _casefold_column(entry["summary"]["columns"], matching_column)
                    if column and _positive_label(entry["row"].get(column)) is True:
                        positive = True
                if not rows:
                    issue(warnings, "orphan_mask_without_csv_row", category=category,
                          anomaly_type=anomaly, object_id=filename.removesuffix(".png"),
                          modality=modality, mask_path=_relative(mask_path, data_root),
                          severity="warning")
                elif not positive:
                    issue(warnings, "unexpected_mask_for_nonpositive_label", category=category,
                          anomaly_type=anomaly, object_id=filename.removesuffix(".png"),
                          modality=modality, mask_path=_relative(mask_path, data_root),
                          severity="warning")

    # Report numeric holes as metadata warnings, but do not treat them as
    # missing images: object IDs in a release need not be contiguous.
    csv_id_gaps: List[Dict[str, Any]] = []
    for csv_path, ids in csv_ids_by_file.items():
        numeric = sorted({int(value) for value in ids if value.isdigit()})
        if numeric:
            expected = set(range(numeric[0], numeric[-1] + 1))
            missing = sorted(expected - set(numeric))
            if missing:
                csv_id_gaps.append({"csv_path": csv_path, "missing_numeric_ids": missing})
    return {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "csv_id_gaps": csv_id_gaps,
        "mask_pixels_opened": False,
        "selection_or_edge_inputs_include_masks": False,
        "known_release_orphan_policy": "orphan/unexpected mask paths are warnings; positive-label missing masks and CSV/image mismatches are errors",
    }


def read_label_reports(data_root: Path, pairs: Sequence[PairRecord]) -> Dict[str, Any]:
    """Summarize label CSVs independently, including explicit disagreements."""

    csv_paths = sorted(data_root.rglob("data.csv"))
    summaries = [_read_label_csv(path, data_root) for path in csv_paths]
    summary_without_rows = []
    disagreement_records: List[Dict[str, Any]] = []
    disagreement_keys: List[PairKey] = []

    pair_lookup = {pair.key: pair for pair in pairs}
    for summary in summaries:
        rows = summary["rows"]
        summary_without_rows.append({key: value for key, value in summary.items() if key != "rows"})
        fieldnames = summary["columns"]
        object_column = summary["object_column"]
        rgb_column = _casefold_column(fieldnames, "RGB")
        ir_column = _casefold_column(fieldnames, "infrared")
        if not object_column or not rgb_column or not ir_column:
            continue
        for row in rows:
            object_id = _csv_value(row.get(object_column))
            rgb_value = _csv_value(row.get(rgb_column))
            ir_value = _csv_value(row.get(ir_column))
            if rgb_value == ir_value:
                continue
            category = summary["category"]
            anomaly_type = summary["anomaly_type"]
            key = PairKey(category, "test", anomaly_type, f"{object_id}.png")
            disagreement_records.append(
                {
                    "category": category,
                    "anomaly_type": anomaly_type,
                    "object_id": object_id,
                    "RGB_column_value": rgb_value,
                    "infrared_column_value": ir_value,
                    "csv_path": summary["path"],
                    "paired_image_exists": key in pair_lookup,
                }
            )
            if key in pair_lookup:
                disagreement_keys.append(key)

    sources = Counter(str(summary["source_directory"]) for summary in summary_without_rows)
    reconciliation = _reconcile_mask_labels(data_root, summaries)
    return {
        "csv_count": len(summary_without_rows),
        "csv_count_by_source_directory": _jsonable_counter(sources),
        "csv_summaries": summary_without_rows,
        "explicit_RGB_vs_infrared_column_disagreements": disagreement_records,
        "disagreement_pair_count": len(set(disagreement_keys)),
        "mask_label_reconciliation": reconciliation,
        "note": (
            "RGB and infrared columns are summarized independently.  The script "
            "does not assume that modality labels should agree; disagreements "
            "are reported, not corrected or used as anomaly ground truth."
        ),
    }


def _image_array(path: Path) -> Tuple[Image.Image, np.ndarray]:
    _assert_non_gt_input(path)
    with Image.open(path) as source:
        image = source.copy()
    return image, np.asarray(image)


def image_stats(path: Path, data_root: Path, role: str) -> Dict[str, Any]:
    image, array = _image_array(path)
    arrays = [array] if array.ndim == 2 else [array[..., index] for index in range(array.shape[-1])]
    per_channel = []
    for index, channel in enumerate(arrays):
        quantiles = np.percentile(channel.astype(np.float64), PERCENTILES).tolist()
        per_channel.append(
            {
                "channel": index,
                "min": float(np.min(channel)),
                "max": float(np.max(channel)),
                "percentiles": {
                    str(percentile): float(value)
                    for percentile, value in zip(PERCENTILES, quantiles)
                },
            }
        )
    exact_rgb_channels: Optional[bool] = None
    alpha_stats: Optional[Dict[str, Any]] = None
    if array.ndim == 3 and array.shape[-1] >= 3:
        # Compare only the RGB channels.  In particular, an RGBA image's
        # alpha channel is not part of this storage/channel-identity check.
        exact_rgb_channels = bool(
            np.array_equal(array[..., 0], array[..., 1])
            and np.array_equal(array[..., 1], array[..., 2])
        )
        if array.shape[-1] >= 4:
            alpha = array[..., 3]
            alpha_quantiles = np.percentile(alpha.astype(np.float64), PERCENTILES).tolist()
            alpha_stats = {
                "min": float(np.min(alpha)),
                "max": float(np.max(alpha)),
                "percentiles": {
                    str(percentile): float(value)
                    for percentile, value in zip(PERCENTILES, alpha_quantiles)
                },
                "all_opaque": bool(np.all(alpha == np.iinfo(alpha.dtype).max))
                if np.issubdtype(alpha.dtype, np.integer)
                else bool(np.all(alpha >= 1.0)),
            }
    return {
        "role": role,
        "path": _relative(path, data_root),
        "pil_mode": image.mode,
        "dimensions_wh": [int(image.width), int(image.height)],
        "numpy_dtype": str(array.dtype),
        "numpy_shape": list(array.shape),
        "channel_count": int(array.shape[-1]) if array.ndim == 3 else 1,
        "channels_0_1_2_exactly_identical": exact_rgb_channels,
        "alpha_channel": alpha_stats,
        "global_min": float(np.min(array)),
        "global_max": float(np.max(array)),
        "per_channel": per_channel,
    }


def _png_header(path: Path) -> Dict[str, Any]:
    """Read PNG IHDR without decoding or converting the image."""

    with path.open("rb") as handle:
        data = handle.read(33)
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a valid PNG signature/header")
    length = struct.unpack(">I", data[8:12])[0]
    if data[12:16] != b"IHDR" or length != 13 or len(data) < 33:
        raise ValueError("PNG does not begin with a 13-byte IHDR")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    return {
        "width": int(width),
        "height": int(height),
        "bit_depth": int(bit_depth),
        "color_type": int(color_type),
        "color_type_name": {
            0: "grayscale", 2: "truecolor", 3: "indexed-color",
            4: "grayscale+alpha", 6: "truecolor+alpha",
        }.get(color_type, "unknown"),
        "compression": int(compression),
        "filter_method": int(filter_method),
        "interlace": int(interlace),
    }


def _bmp_header(path: Path) -> Dict[str, Any]:
    """Read the BMP file/DIB headers without decoding or trusting the suffix.

    MulSen-AD contains files whose suffix is ``.png`` but whose payload is a
    BMP.  The DIB header is the portable source of BMP dimensions and storage
    bit depth; Pillow is still used below to validate that the image decodes.
    """

    with path.open("rb") as handle:
        data = handle.read(54)
    if len(data) < 18 or data[:2] != b"BM":
        raise ValueError("not a valid BMP signature/header")
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 12:
        raise ValueError(f"unsupported BMP DIB header size: {dib_size}")
    required = 14 + dib_size
    if len(data) < required:
        with path.open("rb") as handle:
            data = handle.read(required)
    if len(data) < required:
        raise ValueError("truncated BMP DIB header")
    if dib_size == 12:
        width, height, planes, bits_per_pixel = struct.unpack_from("<HHHH", data, 18)
        signed_height = int(height)
    else:
        if dib_size < 40:
            raise ValueError(f"unsupported BMP DIB header size: {dib_size}")
        signed_width, signed_height, planes, bits_per_pixel = struct.unpack_from(
            "<iiHH", data, 18
        )
        width, height = abs(int(signed_width)), abs(int(signed_height))
    if width <= 0 or height <= 0 or planes != 1 or bits_per_pixel <= 0:
        raise ValueError("invalid BMP dimensions, planes, or bits-per-pixel")
    return {
        "width": int(width),
        "height": int(height),
        "dib_header_size": int(dib_size),
        "bits_per_pixel": int(bits_per_pixel),
        "height_signed": int(signed_height),
    }


def _container_header(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Identify the on-disk container from its signature, not its extension."""

    with path.open("rb") as handle:
        signature = handle.read(8)
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG", _png_header(path)
    if signature[:2] == b"BM":
        return "BMP", _bmp_header(path)
    raise ValueError("unknown image container signature")


def image_header_inventory(
    paths: Iterable[Path],
    *,
    audit_grayscale_channels: bool = False,
) -> Dict[str, Any]:
    modes: Counter = Counter()
    dimensions: Counter = Counter()
    containers: Counter = Counter()
    headers: Counter = Counter()
    bmp_headers: Counter = Counter()
    decode_errors: List[str] = []
    container_errors: List[str] = []
    pil_records: List[Dict[str, Any]] = []
    decoded_dtypes: Counter = Counter()
    decoded_global_min: Optional[float] = None
    decoded_global_max: Optional[float] = None
    exact_channel_images = 0
    nonexact_channel_images = 0
    nonexact_channel_pixels = 0
    audited_channel_pixels = 0
    max_channel_delta = 0
    nonexact_channel_records: List[Dict[str, Any]] = []
    grayscale_audit_errors: List[str] = []
    count = 0
    for path in paths:
        record: Dict[str, Any] = {"path": str(path), "suffix": path.suffix.casefold()}
        try:
            _assert_non_gt_input(path)
            container, header = _container_header(path)
            containers[container] += 1
            record["container"] = container
            record["container_header"] = header
            if container == "PNG":
                header_key = json.dumps(
                    {key: header[key] for key in ("width", "height", "bit_depth", "color_type")},
                    sort_keys=True,
                )
                headers[header_key] += 1
            else:
                header_key = json.dumps(
                    {key: header[key] for key in ("width", "height", "dib_header_size", "bits_per_pixel")},
                    sort_keys=True,
                )
                bmp_headers[header_key] += 1
        except Exception as exc:
            text = f"{path}: {exc}"
            container_errors.append(text)
            record["container_error"] = str(exc)
            # Keep attempting Pillow decode so a report identifies every
            # input, even when the independent signature parser rejects it.
        try:
            with Image.open(path) as image:
                pil_format = image.format
                pil_mode = image.mode
                pil_width, pil_height = image.width, image.height
                image.load()
                modes[image.mode] += 1
                dimensions[f"{image.width}x{image.height}"] += 1
                record.update({
                    "pil_format": pil_format,
                    "pil_mode": pil_mode,
                    "pil_dimensions_wh": [int(pil_width), int(pil_height)],
                })
                if audit_grayscale_channels:
                    array = np.asarray(image)
                    decoded_dtypes[str(array.dtype)] += 1
                    item_min = float(array.min())
                    item_max = float(array.max())
                    decoded_global_min = (
                        item_min if decoded_global_min is None else min(decoded_global_min, item_min)
                    )
                    decoded_global_max = (
                        item_max if decoded_global_max is None else max(decoded_global_max, item_max)
                    )
                    if array.ndim == 2:
                        exact_channel_images += 1
                        audited_channel_pixels += int(array.size)
                    elif array.ndim == 3 and array.shape[-1] >= 3:
                        rgb = array[..., :3].astype(np.int16)
                        span = rgb.max(axis=2) - rgb.min(axis=2)
                        item_nonexact = int(np.count_nonzero(span))
                        item_delta = int(span.max())
                        audited_channel_pixels += int(span.size)
                        nonexact_channel_pixels += item_nonexact
                        max_channel_delta = max(max_channel_delta, item_delta)
                        if item_nonexact:
                            nonexact_channel_images += 1
                            nonexact_channel_records.append({
                                "path": str(path),
                                "max_channel_delta": item_delta,
                                "non_equal_pixel_count": item_nonexact,
                                "non_equal_pixel_fraction": float(item_nonexact / span.size),
                            })
                        else:
                            exact_channel_images += 1
                    else:
                        grayscale_audit_errors.append(
                            f"{path}: unsupported decoded shape for grayscale audit: {array.shape}"
                        )
                count += 1
        except Exception as exc:
            text = f"{path}: {exc}"
            decode_errors.append(text)
            record["pil_error"] = str(exc)
        pil_records.append(record)
    header_records = []
    for encoded, item_count in sorted(headers.items()):
        record = json.loads(encoded)
        record["count"] = int(item_count)
        header_records.append(record)
    bmp_records = []
    for encoded, item_count in sorted(bmp_headers.items()):
        record = json.loads(encoded)
        record["count"] = int(item_count)
        bmp_records.append(record)
    signature_count = len(headers)
    all_dimensions = set(dimensions)
    return {
        "count": count,
        "input_count": len(pil_records),
        "pil_modes": _jsonable_counter(modes),
        "dimensions": _jsonable_counter(dimensions),
        "container_counts": _jsonable_counter(containers),
        "png_ihdr_records": header_records,
        "bmp_dib_records": bmp_records,
        "pil_records": pil_records,
        "header_inconsistency": signature_count > 1,
        "header_inconsistency_reason": (
            "more than one width/height/bit-depth/color-type IHDR signature"
            if signature_count > 1 else None
        ),
        "dimension_inconsistency": len(all_dimensions) > 1,
        "dimension_inconsistency_reason": (
            "more than one decodable PIL width/height in this modality"
            if len(all_dimensions) > 1 else None
        ),
        "container_errors": container_errors,
        # Backward-compatible alias for consumers of the pre-1.2 report.
        "header_errors": container_errors,
        "decode_errors": decode_errors,
        "pixel_encoding_audit": {
            "enabled": bool(audit_grayscale_channels),
            "numpy_dtypes": _jsonable_counter(decoded_dtypes),
            "global_min": decoded_global_min,
            "global_max": decoded_global_max,
            "exact_grayscale_channel_image_count": int(exact_channel_images),
            "nonexact_grayscale_channel_image_count": int(nonexact_channel_images),
            "audited_pixel_count": int(audited_channel_pixels),
            "nonexact_channel_pixel_count": int(nonexact_channel_pixels),
            "nonexact_channel_pixel_fraction": (
                float(nonexact_channel_pixels / audited_channel_pixels)
                if audited_channel_pixels else None
            ),
            "max_channel_delta": int(max_channel_delta),
            "nonexact_images": nonexact_channel_records,
            "errors": grayscale_audit_errors,
        },
    }


def _all_input_paths(data_root: Path, modality: str) -> List[Path]:
    """Enumerate every non-GT PNG, including unmatched files, for integrity checks."""

    paths: List[Path] = []
    for category_root in sorted(path for path in data_root.iterdir() if path.is_dir()):
        modality_root = category_root / modality
        for split in SPLITS:
            split_root = modality_root / split
            if split_root.is_dir():
                paths.extend(
                    path for path in sorted(split_root.rglob("*"))
                    if path.is_file() and path.suffix.casefold() == ".png"
                )
    for path in paths:
        _assert_non_gt_input(path)
    return paths


def _grayscale_uint8(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    if gray.dtype == np.uint8:
        return gray
    values = gray.astype(np.float64)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def edge_map(gray: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.Canny(gray, 50, 150) > 0
    # Lightweight fallback when OpenCV is unavailable.
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = (gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)) * 0.5
    gy[1:-1, :] = (gray[2:, :].astype(np.float32) - gray[:-2, :].astype(np.float32)) * 0.5
    magnitude = np.hypot(gx, gy)
    threshold = float(np.percentile(magnitude, 90)) if magnitude.size else 0.0
    return magnitude > max(threshold, 1.0)


def _edge_overlap(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> Dict[str, float]:
    """Compare a shifted right/down by (dx,dy) against b without wraparound."""

    height, width = a.shape
    ax0, ax1 = max(0, -dx), min(width, width - dx)
    ay0, ay1 = max(0, -dy), min(height, height - dy)
    bx0, bx1 = max(0, dx), min(width, width + dx)
    by0, by1 = max(0, dy), min(height, height + dy)
    if ax1 <= ax0 or ay1 <= ay0:
        return {"f1": 0.0, "iou": 0.0, "overlap": 0.0, "a_count": 0.0, "b_count": 0.0}
    left = a[ay0:ay1, ax0:ax1]
    right = b[by0:by1, bx0:bx1]
    overlap = float(np.logical_and(left, right).sum())
    a_count = float(left.sum())
    b_count = float(right.sum())
    union = a_count + b_count - overlap
    f1 = 2.0 * overlap / (a_count + b_count) if a_count + b_count else 0.0
    iou = overlap / union if union else 0.0
    return {"f1": f1, "iou": iou, "overlap": overlap, "a_count": a_count, "b_count": b_count}


def _rect_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def _best_edge_shift(a: np.ndarray, b: np.ndarray, max_shift: int) -> Tuple[int, int, Dict[str, float]]:
    """Find the best bounded shift without a Python loop over image pixels.

    ``matchTemplate`` computes all binary edge overlaps in one OpenCV call;
    integral images provide the valid-window edge counts.  This keeps a
    65-by-65 search bounded by the image library rather than repeating a
    640x480 reduction thousands of times.
    """

    height, width = a.shape
    if cv2 is not None:
        radius = int(max_shift)
        padded = cv2.copyMakeBorder(b.astype(np.float32), radius, radius, radius, radius,
                                    cv2.BORDER_CONSTANT, value=0)
        overlap_map = cv2.matchTemplate(padded, a.astype(np.float32), cv2.TM_CCORR)
        integral_a = cv2.integral(a.astype(np.float32))
        integral_b = cv2.integral(b.astype(np.float32))
        best: Optional[Tuple[float, int, int, Dict[str, float]]] = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ax0, ax1 = max(0, -dx), min(width, width - dx)
                ay0, ay1 = max(0, -dy), min(height, height - dy)
                bx0, bx1 = max(0, dx), min(width, width + dx)
                by0, by1 = max(0, dy), min(height, height + dy)
                a_count = _rect_sum(integral_a, ax0, ay0, ax1, ay1)
                b_count = _rect_sum(integral_b, bx0, by0, bx1, by1)
                overlap = float(overlap_map[radius + dy, radius + dx])
                union = a_count + b_count - overlap
                f1 = 2.0 * overlap / (a_count + b_count) if a_count + b_count else 0.0
                iou = overlap / union if union else 0.0
                metrics = {"f1": f1, "iou": iou, "overlap": overlap,
                           "a_count": a_count, "b_count": b_count}
                candidate = (f1, dx, dy, metrics)
                if best is None or (candidate[0], -abs(dx) - abs(dy)) > (best[0], -abs(best[1]) - abs(best[2])):
                    best = candidate
        assert best is not None
        return int(best[1]), int(best[2]), best[3]

    # Dependency-light fallback: downsample before the bounded search.  It is
    # explicitly a coarse diagnostic, never a registration claim.
    scale = max(1, int(np.ceil(max(height, width) / 160)))
    coarse_a = a[::scale, ::scale]
    coarse_b = b[::scale, ::scale]
    radius = max(1, int(np.ceil(max_shift / scale)))
    candidates: List[Tuple[float, int, int, Dict[str, float]]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            metrics = _edge_overlap(coarse_a, coarse_b, dx, dy)
            candidates.append((metrics["f1"], dx, dy, metrics))
    _, dx, dy, coarse_metrics = max(candidates, key=lambda item: (item[0], -abs(item[1]) - abs(item[2])))
    return dx * scale, dy * scale, {**coarse_metrics, "coarse_scale": float(scale)}


def edge_diagnostic(rgb_path: Path, infrared_path: Path, max_shift: int = 32) -> Dict[str, Any]:
    """Resize RGB to native IR dimensions and estimate only small edge shifts."""

    rgb, _ = _image_array(rgb_path)
    infrared, _ = _image_array(infrared_path)
    target_size = infrared.size  # explicit convention: use native IR grid
    rgb_aligned = rgb.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    ir_aligned = infrared.convert("RGB")
    rgb_edges = edge_map(_grayscale_uint8(rgb_aligned))
    ir_edges = edge_map(_grayscale_uint8(ir_aligned))

    baseline = _edge_overlap(rgb_edges, ir_edges, 0, 0)
    best_dx, best_dy, best = _best_edge_shift(rgb_edges, ir_edges, max_shift)
    best_f1 = best["f1"]
    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        rgb_tolerant = cv2.dilate(rgb_edges.astype(np.uint8), kernel) > 0
        ir_tolerant = cv2.dilate(ir_edges.astype(np.uint8), kernel) > 0
        tolerant_zero = _edge_overlap(rgb_tolerant, ir_tolerant, 0, 0)
        tolerant_dx, tolerant_dy, tolerant_best = _best_edge_shift(rgb_tolerant, ir_tolerant, max_shift)
    else:
        tolerant_zero = None
        tolerant_dx, tolerant_dy, tolerant_best = best_dx, best_dy, None
    return {
        "resize_convention": "RGB resized with PIL bilinear interpolation to native Infrared (width,height); Infrared is not warped",
        "common_grid_wh": [int(target_size[0]), int(target_size[1])],
        "edge_detector": "OpenCV Canny thresholds 50/150" if cv2 is not None else "numpy central-gradient 90th-percentile fallback",
        "rgb_edge_pixels": int(rgb_edges.sum()),
        "infrared_edge_pixels": int(ir_edges.sum()),
        "zero_shift": baseline,
        "best_shift": {
            "dx_right_pixels": int(best_dx),
            "dy_down_pixels": int(best_dy),
            **best,
        },
        "toleranced_edge_overlap": {
            "method": "3x3 binary dilation before overlap/search" if cv2 is not None else "unavailable without OpenCV",
            "zero_shift": tolerant_zero,
            "best_shift": ({"dx_right_pixels": tolerant_dx, "dy_down_pixels": tolerant_dy, **tolerant_best}
                           if tolerant_best is not None else None),
        },
        "shift_search_radius_pixels": int(max_shift),
        "best_minus_zero_f1": float(best_f1 - baseline["f1"]),
        "interpretation": "Cautious exact/toleranced edge-overlap diagnostic only; not camera calibration, registration proof, or GT-derived alignment. Search uses matchTemplate and is bounded to the requested radius.",
    }


def _panel(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)


def _overlay_edges(rgb_edges: np.ndarray, ir_edges: np.ndarray) -> Image.Image:
    overlay = np.zeros((*rgb_edges.shape, 3), dtype=np.uint8)
    overlay[:] = [24, 24, 24]
    overlay[rgb_edges] = [245, 70, 70]
    overlay[ir_edges] = [60, 220, 245]
    overlay[rgb_edges & ir_edges] = [255, 255, 255]
    return Image.fromarray(overlay)


def _diagnostic_panels(pair: PairRecord) -> List[Image.Image]:
    rgb, _ = _image_array(pair.rgb_path)
    infrared, _ = _image_array(pair.infrared_path)
    target_size = infrared.size
    rgb_aligned = rgb.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    ir_aligned = infrared.convert("RGB")
    rgb_gray = _grayscale_uint8(rgb_aligned)
    ir_gray = _grayscale_uint8(ir_aligned)
    rgb_edges = edge_map(rgb_gray)
    ir_edges = edge_map(ir_gray)
    return [
        rgb_aligned,
        ir_aligned,
        Image.fromarray(np.where(rgb_edges, 255, 0).astype(np.uint8)).convert("RGB"),
        Image.fromarray(np.where(ir_edges, 255, 0).astype(np.uint8)).convert("RGB"),
        _overlay_edges(rgb_edges, ir_edges),
    ]


def write_contact_sheet(
    pairs: Sequence[PairRecord],
    reasons: Mapping[PairKey, Sequence[str]],
    output_path: Path,
) -> None:
    labels = ["RGB (resized)", "Infrared", "RGB edges", "Infrared edges", "edge overlay"]
    panel_size = (280, 210)
    title_height = 46
    row_height = title_height + panel_size[1]
    sheet = Image.new("RGB", (len(labels) * panel_size[0], max(1, len(pairs)) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(pairs):
        y = row * row_height
        reason = ", ".join(reasons.get(pair.key, ("deterministic sample",)))
        title = f"{pair.key.compact()} [{reason}]"
        draw.text((4, y + 3), title[:180], fill="black")
        for column, (label, image) in enumerate(zip(labels, _diagnostic_panels(pair))):
            x = column * panel_size[0]
            draw.text((x + 4, y + 22), label, fill="black")
            fitted = _panel(image, panel_size)
            px = x + (panel_size[0] - fitted.width) // 2
            py = y + title_height + (panel_size[1] - fitted.height) // 2
            sheet.paste(fitted, (px, py))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def _pick_one(rng: random.Random, choices: Sequence[PairRecord]) -> Optional[PairRecord]:
    if not choices:
        return None
    return choices[rng.randrange(len(choices))]


def sample_pairs(
    pairs: Sequence[PairRecord],
    disagreement_keys: Sequence[PairKey],
    sample_count: int,
    seed: int,
) -> Tuple[List[PairRecord], Dict[PairKey, List[str]]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = random.Random(seed)
    by_key = {pair.key: pair for pair in pairs}
    selected: List[PairRecord] = []
    reasons: Dict[PairKey, List[str]] = defaultdict(list)

    def selected_keys() -> set[PairKey]:
        return {item.key for item in selected}

    def add(pair: Optional[PairRecord], reason: str) -> None:
        if pair is None or pair.key in selected_keys() or len(selected) >= sample_count:
            return
        selected.append(pair)
        reasons[pair.key].append(reason)

    categories = sorted({pair.key.category for pair in pairs})
    disagreements = [by_key[key] for key in sorted(set(disagreement_keys)) if key in by_key]
    disagreement_set = {pair.key for pair in disagreements}

    # First cover distinct categories.  The first two slots deliberately
    # reserve one normal and one anomalous pair when the caller permits it;
    # subsequent slots maximize category coverage in stable seed order.
    category_order = categories[:]
    rng.shuffle(category_order)
    target_strata = (["good", "anomalous"] if sample_count >= 2 else ["good"])
    for index, category in enumerate(category_order):
        if len(selected) >= sample_count:
            break
        preferred = target_strata[index] if index < len(target_strata) else ("good" if index % 2 == 0 else "anomalous")
        good = [pair for pair in pairs if pair.key.category == category and pair.key.split == "test" and pair.key.anomaly_type == "good"]
        anomalous = [pair for pair in pairs if pair.key.category == category and pair.key.split == "test" and pair.key.anomaly_type != "good"]
        candidates = good if preferred == "good" else anomalous
        fallback = anomalous if preferred == "good" else good
        add(_pick_one(rng, candidates or fallback), f"{preferred} coverage" if candidates else f"{preferred} unavailable; fallback coverage")

    # Inject disagreements only after category/class coverage.  Replace an
    # existing anomalous pick from the same category where possible, so
    # disagreements cannot crowd out normal coverage or distinct categories.
    requested_disagreements = min(len(disagreements), max(1, sample_count // 8)) if disagreements else 0
    disagreement_replacements = 0
    for disagreement in disagreements:
        if disagreement.key in selected_keys():
            reasons[disagreement.key].append("RGB/infrared label-column disagreement")
            disagreement_replacements += 1
            if disagreement_replacements >= requested_disagreements:
                break
            continue
        if disagreement_replacements >= requested_disagreements:
            break
        for index, current in enumerate(selected):
            if (current.key.category == disagreement.key.category
                    and current.key.anomaly_type != "good"
                    and current.key not in disagreement_set):
                selected[index] = disagreement
                reasons.pop(current.key, None)
                reasons[disagreement.key].extend(["anomalous coverage", "RGB/infrared label-column disagreement"])
                disagreement_replacements += 1
                break

    # If category coverage exhausted the budget, fill only after the above
    # invariants.  The seeded fill is explicitly reported as such.

    remaining = [pair for pair in pairs if pair.key not in {item.key for item in selected}]
    rng.shuffle(remaining)
    for pair in remaining:
        add(pair, "seeded fill")
    return selected, {key: value for key, value in reasons.items()}


def _stats_for_sampled(pairs: Sequence[PairRecord], data_root: Path) -> List[Dict[str, Any]]:
    output = []
    for pair in pairs:
        output.append(
            {
                "key": pair.key.as_dict(),
                "RGB": image_stats(pair.rgb_path, data_root, "RGB"),
                "Infrared": image_stats(pair.infrared_path, data_root, "Infrared"),
            }
        )
    return output


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/MulSenAD_official/MulSen_AD"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mulsen_alignment"))
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max-shift", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == data_root or data_root in output_dir.parents:
        raise ValueError(f"output-dir must be outside data-root: {output_dir}")
    if args.max_shift < 0:
        raise ValueError("max-shift must be non-negative")

    pairs, integrity = discover_pairs(data_root)
    labels = read_label_reports(data_root, pairs)
    disagreement_keys = []
    for item in labels["explicit_RGB_vs_infrared_column_disagreements"]:
        if item["paired_image_exists"]:
            disagreement_keys.append(
                PairKey(item["category"], "test", item["anomaly_type"], f"{item['object_id']}.png")
            )
    sampled, reasons = sample_pairs(pairs, disagreement_keys, args.sample_count, args.seed)

    all_rgb = _all_input_paths(data_root, "RGB")
    all_ir = _all_input_paths(data_root, "Infrared")
    image_inventory = {
        "RGB": image_header_inventory(all_rgb),
        "Infrared": image_header_inventory(all_ir, audit_grayscale_channels=True),
    }
    integrity_errors = []
    integrity_warnings = []
    for modality, inventory in image_inventory.items():
        if inventory["header_inconsistency"]:
            integrity_warnings.append(
                f"{modality} has multiple PNG IHDR signatures; this is an inventory statistic, not a failure"
            )
        if inventory["dimension_inconsistency"]:
            integrity_warnings.append(
                f"{modality} has multiple decodable dimensions; per-pair resize remains the diagnostic convention"
            )
        if inventory["container_counts"].get("BMP", 0):
            integrity_warnings.append(
                f"{modality} contains {inventory['container_counts']['BMP']} BMP payload(s), including possible .png-suffixed files"
            )
        if inventory["container_errors"]:
            integrity_errors.append(f"{modality} image container errors: {len(inventory['container_errors'])}")
        if inventory["decode_errors"]:
            integrity_errors.append(f"{modality} image decode errors: {len(inventory['decode_errors'])}")
        if inventory["pixel_encoding_audit"]["errors"]:
            integrity_errors.append(
                f"{modality} pixel-encoding audit errors: "
                f"{len(inventory['pixel_encoding_audit']['errors'])}"
            )
    label_reconciliation = labels["mask_label_reconciliation"]
    if label_reconciliation["errors"]:
        integrity_errors.append(
            f"metadata reconciliation errors: {len(label_reconciliation['errors'])}"
        )
    if label_reconciliation["warnings"]:
        integrity_warnings.append(
            f"metadata reconciliation warnings: {len(label_reconciliation['warnings'])}"
        )
    integrity["errors"] = integrity_errors
    integrity["warnings"] = integrity_warnings
    integrity["issues"].extend(integrity_errors)
    integrity["status"] = "failed" if integrity["issues"] else "ok"
    strata_counts = Counter()
    for pair in sampled:
        if pair.key.anomaly_type == "good":
            strata_counts["good"] += 1
        else:
            strata_counts["anomalous"] += 1
        if pair.key in set(disagreement_keys):
            strata_counts["RGB/infrared disagreement"] += 1
    sampled_stats: List[Dict[str, Any]] = []
    for pair in sampled:
        try:
            sampled_stats.append({
                "key": pair.key.as_dict(),
                "RGB": image_stats(pair.rgb_path, data_root, "RGB"),
                "Infrared": image_stats(pair.infrared_path, data_root, "Infrared"),
            })
        except Exception as exc:
            integrity["issues"].append(f"sampled image statistics error: {pair.key.compact()}: {exc}")
            integrity["status"] = "failed"
    report: Dict[str, Any] = {
        "script": {"name": Path(__file__).name, "version": SCRIPT_VERSION},
        "arguments": {
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "sample_count": args.sample_count,
            "seed": args.seed,
            "max_shift": args.max_shift,
        },
        "diagnostic_input_policy": {
            "gt_masks_used_for_diagnostics": False,
            "label_csvs_used_only_as_metadata": True,
            "diagnostic_image_roots": ["*/RGB/{train,test}", "*/Infrared/{train,test}"],
            "gt_paths_opened_for_image_stats_edges_or_overlays": [],
            "alignment_convention": "RGB is bilinearly resized to native Infrared dimensions; no affine/projective warp is estimated or applied.",
            "same_index_claim": "Pair identity is filename plus category/split/anomaly_type only. Same-index pixel correspondence is not asserted; edge shift is a cautious diagnostic.",
        },
        "integrity": integrity,
        "image_inventory": image_inventory,
        "labels": labels,
        "sampling": {
            "selected_count": len(sampled),
            "selected_keys": [pair.key.as_dict() for pair in sampled],
            "selection_reasons": {pair.key.compact(): reasons.get(pair.key, []) for pair in sampled},
            "disagreement_candidates": len(set(disagreement_keys)),
            "selected_strata_counts": _jsonable_counter(strata_counts),
            "selection_policy": "distinct-category coverage first; reserve good+anomalous when sample_count >= 2; inject a bounded number of disagreement pairs without displacing category/normal coverage",
            "unmet_constraints": (["both good and anomalous require sample_count >= 2"]
                                  if args.sample_count < 2 else []),
        },
        "sampled_image_stats": sampled_stats,
        "edge_diagnostics": [],
    }
    for pair in sampled:
        try:
            diagnostic = edge_diagnostic(pair.rgb_path, pair.infrared_path, args.max_shift)
            diagnostic["key"] = pair.key.as_dict()
            report["edge_diagnostics"].append(diagnostic)
        except Exception as exc:
            report["edge_diagnostics"].append({"key": pair.key.as_dict(), "error": str(exc)})
            integrity["issues"].append(f"sampled image decode/edge error: {pair.key.compact()}: {exc}")
            integrity["status"] = "failed"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mulsen_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contact_path = output_dir / "mulsen_alignment_contact_sheet.png"
    try:
        write_contact_sheet(sampled, reasons, contact_path)
    except Exception as exc:
        report["contact_sheet_error"] = str(exc)
        integrity["issues"].append(f"contact-sheet image error: {exc}")
        integrity["status"] = "failed"
    # Edge/contact failures are added after the first JSON construction, so
    # rewrite once with the final integrity status and diagnostics.
    report["integrity"] = integrity
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if integrity["status"] != "ok":
        print(json.dumps({"status": "failed", "issues": integrity["issues"], "report": str(report_path)}, indent=2), file=sys.stderr)
        raise IntegrityError("MulSen RGB/Infrared pairing integrity failed; see report JSON")

    print(json.dumps({
        "status": "ok",
        "paired_count": integrity["paired_count"],
        "sampled_count": len(sampled),
        "label_csv_count": labels["csv_count"],
        "label_disagreement_pairs": labels["disagreement_pair_count"],
        "report": str(report_path),
        "contact_sheet": str(contact_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IntegrityError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
