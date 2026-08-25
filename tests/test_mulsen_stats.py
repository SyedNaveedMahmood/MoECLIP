"""Validation tests for leakage-safe thermal statistics metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.mulsen_protocol import DEVELOPMENT_TRAIN_CATEGORIES, PROTOCOL_VERSION
from dataset.mulsen_stats import ThermalNormalization
from tools.compute_mulsen_thermal_stats import StreamingMoments


class MulSenStatsTest(unittest.TestCase):
    def test_streaming_population_moments(self) -> None:
        moments = StreamingMoments()
        moments.update(np.array([0.0, 0.5], dtype=np.float32))
        moments.update(np.array([1.0, 0.5], dtype=np.float32))
        mean, std = moments.finalize()
        self.assertAlmostEqual(mean, 0.5)
        self.assertAlmostEqual(std, np.sqrt(0.125))

    def test_metadata_requires_exact_stage_and_categories(self) -> None:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_stage": "development",
            "source_split": "train-normal-only",
            "categories": list(DEVELOPMENT_TRAIN_CATEGORIES),
            "sample_count": 705,
            "pixel_count": 1000,
            "mean": 0.4,
            "std": 0.2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stats.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            stats = ThermalNormalization.load(
                path,
                expected_categories=DEVELOPMENT_TRAIN_CATEGORIES,
                expected_stage="development",
            )
            self.assertEqual(stats.sample_count, 705)
            self.assertEqual(stats.mean, 0.4)

            with self.assertRaisesRegex(ValueError, "different protocol stage"):
                ThermalNormalization.load(
                    path,
                    expected_categories=DEVELOPMENT_TRAIN_CATEGORIES,
                    expected_stage="final",
                )
            with self.assertRaisesRegex(ValueError, "do not exactly match"):
                ThermalNormalization.load(
                    path,
                    expected_categories=DEVELOPMENT_TRAIN_CATEGORIES[:-1],
                    expected_stage="development",
                )


if __name__ == "__main__":
    unittest.main()
