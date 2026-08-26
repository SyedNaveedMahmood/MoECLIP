"""Compute MulSen-AD IR mean/std from seen-category normal training data.

This is intentionally a user-run preprocessing command. It decodes only the
official IR files selected by the locked protocol and never reads anomaly masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset.mulsen_ad import (
    MulSenAD,
    OFFICIAL_IR_MAX_CHANNEL_DELTA,
    OFFICIAL_IR_MAX_NON_GRAY_FRACTION,
    _open_thermal,
)
from dataset.mulsen_protocol import PROTOCOL_VERSION, get_protocol


class StreamingMoments:
    """Float64 population moments without retaining decoded images."""

    def __init__(self) -> None:
        self.pixel_count = 0
        self.value_sum = 0.0
        self.square_sum = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.pixel_count += int(values.size)
        self.value_sum += float(values.sum(dtype=np.float64))
        self.square_sum += float(np.square(values).sum(dtype=np.float64))

    def finalize(self):
        if self.pixel_count <= 0:
            raise ValueError("cannot finalize empty thermal statistics")
        mean = self.value_sum / self.pixel_count
        variance = max(0.0, self.square_sum / self.pixel_count - mean * mean)
        std = variance**0.5
        if std <= 0.0:
            raise ValueError("thermal standard deviation is zero")
        return mean, std


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute leakage-safe MulSen-AD thermal normalization"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--protocol-stage",
        required=True,
        choices=("development", "final"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing file: {output}")
    protocol = get_protocol(args.protocol_stage)
    dataset = MulSenAD(
        args.data_root,
        split="train",
        categories=protocol.train_categories,
        img_size=14,
        train=False,
        joint_geometry=False,
        region_method=None,
    )
    moments = StreamingMoments()
    for index, record in enumerate(dataset.records, start=1):
        thermal = _open_thermal(
            record.thermal_path,
            require_gray=True,
            max_channel_delta=OFFICIAL_IR_MAX_CHANNEL_DELTA,
            max_non_gray_fraction=OFFICIAL_IR_MAX_NON_GRAY_FRACTION,
        )
        moments.update(thermal.astype(np.float64) / 255.0)
        if index % 100 == 0 or index == len(dataset.records):
            print(f"decoded {index}/{len(dataset.records)} normal IR images")
    mean, std = moments.finalize()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_stage": protocol.stage,
        "source_split": "train-normal-only",
        "categories": list(protocol.train_categories),
        "sample_count": len(dataset.records),
        "pixel_count": moments.pixel_count,
        "mean": mean,
        "std": std,
        "decoded_storage": "uint8 RGB payload averaged to one channel, divided by 255",
        "resize_policy": "native 640x480 pixels; no resize and no per-image min/max",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
