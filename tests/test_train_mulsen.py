"""Offline unit tests for MulSen-AD training configuration and loss wiring."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from train_mulsen import (
    _segmentation_supervision_mask,
    batch_loss,
    validate_args,
)


def _args(**overrides):
    values = {
        "variant": "D",
        "thermal_stats": "stats.json",
        "align_loss_lambda": 0.0,
        "img_size": 28,
        "epochs": 1,
        "batch_size": 1,
        "workers": 0,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "num_context_experts": None,
        "use_global_context": False,
        "amp_init_scale": 1024.0,
        "adapter_norm_floor": 1.0,
        "use_paa": True,
        "use_segment_paa": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _LossFixture(nn.Module):
    use_thermal = True
    use_region_routing = True

    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Parameter(torch.randn(1, 4, 768))
        self.detection = nn.Parameter(torch.randn(1, 768))

    def forward(self, image, thermal=None, region_map=None):
        if thermal is None or region_map is None:
            raise AssertionError("multimodal region inputs were not forwarded")
        batch_size = image.shape[0]
        patch = torch.nn.functional.normalize(
            self.patch.expand(batch_size, -1, -1), dim=-1
        )
        detection = torch.nn.functional.normalize(
            self.detection.expand(batch_size, -1), dim=-1
        )
        nonzero_aux = patch.square().mean()
        return [patch], detection, nonzero_aux, nonzero_aux


class TrainMulSenTest(unittest.TestCase):
    def test_variant_contract_and_rejected_alignment(self) -> None:
        self.assertEqual(validate_args(_args(variant="A", thermal_stats=None)), (False, False))
        self.assertEqual(validate_args(_args(variant="B")), (True, False))
        self.assertEqual(validate_args(_args(variant="C", thermal_stats=None)), (False, True))
        self.assertEqual(validate_args(_args(variant="D")), (True, True))
        with self.assertRaisesRegex(ValueError, "require --thermal_stats"):
            validate_args(_args(variant="D", thermal_stats=None))
        with self.assertRaisesRegex(ValueError, "alignment is intentionally disabled"):
            validate_args(_args(align_loss_lambda=0.1))
        with self.assertRaisesRegex(ValueError, "amp_init_scale"):
            validate_args(_args(amp_init_scale=float("nan")))
        with self.assertRaisesRegex(ValueError, "amp_init_scale"):
            validate_args(_args(amp_init_scale=0.0))
        with self.assertRaisesRegex(ValueError, "adapter_norm_floor"):
            validate_args(_args(adapter_norm_floor=0.0))
        with self.assertRaisesRegex(ValueError, "variants C/D"):
            validate_args(_args(variant="B", use_segment_paa=True))
        with self.assertRaisesRegex(ValueError, "variants C/D"):
            validate_args(_args(variant="B", use_global_context=True))
        self.assertEqual(
            validate_args(_args(variant="C", use_global_context=True)),
            (False, True),
        )
        with self.assertRaisesRegex(ValueError, "standard PAA"):
            validate_args(_args(use_segment_paa=True, use_paa=False))

    def test_loss_is_finite_and_uses_multimodal_region_inputs(self) -> None:
        torch.manual_seed(31)
        model = _LossFixture()
        text = torch.nn.functional.normalize(torch.randn(768, 2), dim=0)
        batch = {
            "image": torch.randn(1, 3, 28, 28),
            "thermal": torch.randn(1, 1, 28, 28),
            "region_map": torch.zeros(1, 28, 28, dtype=torch.long),
            "mask_rgb": torch.zeros(1, 1, 28, 28),
            "mask_thermal": torch.ones(1, 1, 28, 28),
            "label_rgbt": torch.ones(1, dtype=torch.long),
            "class_name": ["capsule"],
        }
        with patch(
            "train_mulsen.get_adapted_single_class_text_embedding",
            return_value=text,
        ):
            loss, components = batch_loss(
                model,
                batch,
                torch.device("cpu"),
                28,
                balance_weight=0.01,
                etf_weight=0.01,
            )

        self.assertTrue(torch.isfinite(loss))
        for value in components.values():
            self.assertTrue(torch.isfinite(value))
        loss.backward()
        self.assertGreater(float(model.patch.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.detection.grad.abs().sum()), 0.0)

    def test_segmentation_supervision_is_rgb_specific(self) -> None:
        batch = {
            "anomaly_type": ["good", "rgb_only", "rgb_and_ir", "ir_only"],
            "label_rgb": torch.tensor([0, 1, 1, 0]),
        }
        valid = _segmentation_supervision_mask(batch, torch.device("cpu"), 4)
        self.assertEqual(valid.tolist(), [True, True, True, False])

    def test_mixed_batch_excludes_ir_only_from_segmentation_denominator(self) -> None:
        torch.manual_seed(37)
        model = _LossFixture()
        text = torch.nn.functional.normalize(torch.randn(768, 2), dim=0)
        mask_rgb = torch.stack(
            [torch.full((1, 28, 28), value) for value in (0.0, 0.25, 0.75, 1.0)]
        )
        batch = {
            "image": torch.randn(4, 3, 28, 28),
            "thermal": torch.randn(4, 1, 28, 28),
            "region_map": torch.zeros(4, 28, 28, dtype=torch.long),
            "mask_rgb": mask_rgb,
            "mask_thermal": torch.ones(4, 1, 28, 28),
            "label_rgbt": torch.tensor([0, 1, 1, 1]),
            "label_rgb": torch.tensor([0, 1, 1, 0]),
            "anomaly_type": ["good", "rgb_only", "rgb_and_ir", "ir_only"],
            "class_name": ["capsule"] * 4,
        }
        captured = []

        def capture_segmentation_loss(scores, masks):
            captured.append((scores.shape[0], masks.detach().clone()))
            return scores.mean() * 0.0

        with patch(
            "train_mulsen.get_adapted_single_class_text_embedding",
            return_value=text,
        ), patch(
            "train_mulsen.calculate_seg_loss",
            side_effect=capture_segmentation_loss,
        ):
            loss, _ = batch_loss(
                model,
                batch,
                torch.device("cpu"),
                28,
                balance_weight=0.01,
                etf_weight=0.01,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(captured), 1)
        supervised_count, selected_masks = captured[0]
        self.assertEqual(supervised_count, 3)
        torch.testing.assert_close(
            selected_masks[:, 0, 0, 0], torch.tensor([0.0, 0.25, 0.75])
        )


if __name__ == "__main__":
    unittest.main()
