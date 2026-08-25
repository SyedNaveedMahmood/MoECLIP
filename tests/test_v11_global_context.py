"""Focused tests for the explicitly gated v1.1 global-context router path."""

from __future__ import annotations

import io
import unittest

import torch

from model.moe_adapter import BaseIndependentMoE
from model.region_context import (
    RegionContextEncoder,
    count_weighted_global_context,
)
from tests.test_region_moeclip import _inputs, _model
from tests.test_region_router import _config


class V11GlobalContextTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)

    def test_count_weighted_pooling_uses_valid_patch_counts(self) -> None:
        regions = torch.tensor(
            [[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]],
            requires_grad=True,
        )
        counts = torch.tensor([[3, 1, 99]], dtype=torch.long)
        valid = torch.tensor([[True, True, False]])

        pooled = count_weighted_global_context(regions, counts, valid)

        torch.testing.assert_close(
            pooled, torch.tensor([[3.25, 6.5]])
        )
        pooled.sum().backward()
        self.assertIsNotNone(regions.grad)
        self.assertGreater(float(regions.grad[:, :2].abs().sum()), 0.0)
        self.assertEqual(float(regions.grad[:, 2:].abs().sum()), 0.0)

    def test_global_context_encoder_shapes_and_gradients(self) -> None:
        encoder = RegionContextEncoder(
            rgb_dim=8,
            thermal_dim=6,
            context_dim=8,
            num_heads=2,
            use_global_context=True,
        )
        rgb = torch.randn(2, 4, 8, requires_grad=True)
        thermal = torch.randn(2, 9, 6, requires_grad=True)
        region_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

        output = encoder(rgb, region_ids, (2, 2), thermal, (3, 3))

        self.assertEqual(output.patch_context.shape, (2, 4, 8))
        self.assertEqual(output.global_context.shape, (2, 8))
        (output.patch_context.square().mean() + output.global_context.square().mean()).backward()
        self.assertGreater(float(rgb.grad.abs().sum()), 0.0)
        self.assertGreater(float(thermal.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(encoder.patch_context_projection[0].weight.grad.abs().sum()), 0.0
        )

    def test_alpha_is_initialized_to_point_two_and_checkpointed(self) -> None:
        router = BaseIndependentMoE(
            d_model=16,
            config=_config(),
            use_fofs=False,
            router_context_dim=8,
            use_context_scale=True,
        )

        self.assertAlmostEqual(float(router.context_alpha), 0.2, places=6)
        self.assertIn("context_scale_logit", router.state_dict())
        checkpoint = io.BytesIO()
        torch.save(router.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = BaseIndependentMoE(
            d_model=16,
            config=_config(),
            use_fofs=False,
            router_context_dim=8,
            use_context_scale=True,
        )
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        torch.testing.assert_close(restored.context_alpha, router.context_alpha)

    def test_alpha_scales_context_logits_without_changing_rgb_experts(self) -> None:
        router = BaseIndependentMoE(
            d_model=16,
            config=_config(),
            use_fofs=False,
            router_context_dim=8,
            use_context_scale=True,
        )
        hidden = torch.randn(3, 2, 16)
        context = torch.randn(3, 2, 8)
        with torch.no_grad():
            router.context_gate.weight.fill_(1.0)

        base = router.compute_router_logits(hidden)
        conditioned = router.compute_router_logits(hidden, context)
        self.assertGreater(float((conditioned - base).abs().sum()), 0.0)

        with torch.no_grad():
            router.context_scale_logit.copy_(torch.tensor(-20.0))
        nearly_disabled = router.compute_router_logits(hidden, context)
        self.assertLess(float((nearly_disabled - base).abs().max()), 1e-5)

    def test_alpha_receives_finite_nonzero_gradient_when_context_is_active(self) -> None:
        router = BaseIndependentMoE(
            d_model=16,
            config=_config(),
            use_fofs=False,
            router_context_dim=8,
            use_context_scale=True,
        )
        with torch.no_grad():
            router.context_gate.weight.fill_(1.0)
        hidden = torch.zeros(2, 3, 16)
        context = torch.ones(2, 3, 8)
        router.compute_router_logits(hidden, context).sum().backward()
        self.assertIsNotNone(router.context_scale_logit.grad)
        self.assertTrue(torch.isfinite(router.context_scale_logit.grad))
        self.assertGreater(float(router.context_scale_logit.grad.abs()), 0.0)

    def test_context_scale_requires_router_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "router_context_dim"):
            BaseIndependentMoE(
                d_model=16,
                config=_config(),
                use_fofs=False,
                use_context_scale=True,
            )

    def test_rgb_only_path_remains_valid_and_cls_context_is_zero(self) -> None:
        model = _model(use_global_context=False).eval()
        image, _, _ = _inputs()
        with torch.no_grad():
            segmentation, detection, _, _ = model(image)
        self.assertEqual(len(segmentation), 12)
        self.assertEqual(detection.shape, (1, 768))

        model = _model(
            use_thermal=True,
            use_region_routing=True,
            use_global_context=True,
        ).eval()
        captured = []

        def capture(module, args, kwargs):
            captured.append(kwargs["router_context"].detach().clone())

        handle = model.image_adapter["moe_adapters"][0].register_forward_pre_hook(
            capture, with_kwargs=True
        )
        try:
            image, thermal, region_map = _inputs()
            with torch.no_grad():
                model(image, thermal=thermal, region_map=region_map)
        finally:
            handle.remove()

        self.assertTrue(captured)
        self.assertEqual(float(captured[0][0].abs().sum()), 0.0)
        self.assertIsNone(getattr(model, "anomaly_mask", None))


if __name__ == "__main__":
    unittest.main()
