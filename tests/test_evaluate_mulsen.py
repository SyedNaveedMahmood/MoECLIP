"""Tests for leakage-safe MulSen evaluation metrics and selection."""

from __future__ import annotations

import unittest

import numpy as np

from evaluate_mulsen import (
    _safe_minmax,
    category_metrics,
    select_development_checkpoint,
    summarize_categories,
    validate_evaluation_scope,
)


class EvaluateMulSenTest(unittest.TestCase):
    def test_constant_minmax_is_finite(self) -> None:
        normalized = _safe_minmax(np.full((3, 2), 7.0))
        np.testing.assert_array_equal(normalized, np.zeros((3, 2)))

    def test_ir_only_sample_is_image_scored_but_excluded_from_rgb_pixels(self) -> None:
        masks = np.zeros((4, 2, 2), dtype=np.uint8)
        masks[2, 0, 0] = 1
        maps = np.zeros((4, 2, 2), dtype=np.float32)
        maps[2, 0, 0] = 1.0
        maps[3, 1, 1] = 0.8
        result = category_metrics(
            image_labels=np.array([0, 0, 1, 1]),
            detection_scores=np.array([0.1, 0.2, 0.8, 0.9]),
            pixel_maps=maps,
            rgb_masks=masks,
            pixel_valid=np.array([True, True, True, False]),
        )

        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["anomalous_images"], 2)
        self.assertEqual(result["rgb_pixel_sample_count"], 3)
        self.assertEqual(result["image_combined"]["auroc"], 1.0)
        self.assertEqual(result["rgb_pixel"]["auroc"], 1.0)

        summary = summarize_categories({"toy_a": result, "toy_b": result})
        self.assertEqual(summary["selection_score"], 1.0)

    def test_final_scope_cannot_select_and_development_ties_choose_earliest(self) -> None:
        validate_evaluation_scope("development", 3)
        validate_evaluation_scope("final", 1)
        with self.assertRaisesRegex(ValueError, "cannot select"):
            validate_evaluation_scope("final", 2)

        evaluations = [
            {"epoch": 4, "metrics": {"macro": {"selection_score": 0.7}}},
            {"epoch": 2, "metrics": {"macro": {"selection_score": 0.7}}},
            {"epoch": 1, "metrics": {"macro": {"selection_score": 0.6}}},
        ]
        selected = select_development_checkpoint(evaluations)
        self.assertEqual(selected["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
