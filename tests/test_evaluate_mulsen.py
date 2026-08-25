"""Tests for leakage-safe MulSen evaluation metrics and selection."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from evaluate_mulsen import (
    _safe_minmax,
    category_metrics,
    select_development_checkpoint,
    subgroup_diagnostics,
    summarize_categories,
    validate_evaluation_scope,
    build_model_from_config,
)


class EvaluateMulSenTest(unittest.TestCase):
    def test_legacy_config_reconstructs_v1_without_global_context(self) -> None:
        config = {
            "model_name": "fixture",
            "img_size": 28,
            "variant": "D",
            "use_thermal": True,
            "use_region_routing": True,
            "use_paa": True,
            "use_segment_paa": False,
            "seg_proj_sharing_strategy": "shared",
            "image_adapt_weight": 0.1,
            "moe_r": 2,
            "moe_lora_alpha": 4,
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_layers": [0],
            "router_init": "normal",
            "use_fofs": False,
            "thermal_depth": 1,
            "thermal_width": 8,
            "region_context_dim": 8,
            "region_attention_heads": 2,
            "region_coordinate_bias": 1.0,
            "region_coordinate_sigma": 0.75,
            "num_context_experts": None,
            "modality_dropout": 0.2,
            "stable_adapter_norm": True,
            "adapter_norm_floor": 1.0,
            "relu": False,
        }
        device = MagicMock()
        clip = MagicMock()
        model = MagicMock()
        model.to.return_value = model
        with patch("evaluate_mulsen.create_model", return_value=clip), patch(
            "evaluate_mulsen.MoECLIP", return_value=model
        ) as constructor:
            self.assertIs(build_model_from_config(config, device), model)

        self.assertFalse(constructor.call_args.kwargs["use_global_context"])

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

    def test_subgroup_diagnostics_preserve_modality_semantics(self) -> None:
        def prediction(category, rgb, thermal, detection, raw, normalized, combined):
            return {
                "category": category,
                "label_rgb": rgb,
                "label_thermal": thermal,
                "label_rgbt": int(rgb or thermal),
                "detection_score": detection,
                "patch_max_score_raw": raw,
                "patch_max_score_normalized": normalized,
                "combined_score": combined,
            }

        diagnostics = subgroup_diagnostics(
            [
                prediction("screw", 0, 0, 0.1, 0.2, 0.0, 0.05),
                prediction("screw", 1, 0, 0.8, 1.2, 0.9, 0.85),
                prediction("screw", 0, 1, 0.7, 0.3, 0.2, 0.45),
                prediction("plastic_cylinder", 1, 1, 0.9, 1.1, 1.0, 0.95),
            ]
        )
        self.assertEqual(diagnostics["subgroups"]["good"]["sample_count"], 1)
        self.assertEqual(diagnostics["subgroups"]["rgb_only"]["anomalous_images"], 1)
        self.assertEqual(diagnostics["subgroups"]["ir_only"]["sample_count"], 1)
        self.assertEqual(
            diagnostics["category_subgroups"]["screw"]["ir_only"]["sample_count"],
            1,
        )
        self.assertEqual(diagnostics["detection_only"]["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
