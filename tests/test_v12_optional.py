"""Focused tests for disabled-by-default v1.2 infrastructure."""

from __future__ import annotations

import unittest

import torch

from model.region_context import RegionContextEncoder
from tests.test_region_moeclip import _inputs, _model
from train_mulsen import validate_args
from tests.test_train_mulsen import _args
from tools.smoke_mulsen_real_model import _output_shape_report


class V12OptionalTest(unittest.TestCase):
    def test_reliability_disabled_preserves_encoder_state_and_output(self) -> None:
        torch.manual_seed(71)
        enabled = RegionContextEncoder(
            rgb_dim=8, thermal_dim=6, context_dim=8, num_heads=2
        )
        gated = RegionContextEncoder(
            rgb_dim=8,
            thermal_dim=6,
            context_dim=8,
            num_heads=2,
            use_thermal_reliability_gate=False,
        )
        gated.load_state_dict(enabled.state_dict(), strict=True)
        rgb = torch.randn(1, 4, 8)
        thermal = torch.randn(1, 9, 6)
        region_ids = torch.tensor([[0, 0, 1, 1]])
        first = enabled(rgb, region_ids, (2, 2), thermal, (3, 3))
        second = gated(rgb, region_ids, (2, 2), thermal, (3, 3))
        torch.testing.assert_close(first.patch_context, second.patch_context)
        self.assertIsNone(second.thermal_reliability)
        self.assertFalse(any("reliab" in key for key in gated.state_dict()))

    def test_reliability_is_bounded_and_zero_gate_keeps_rgb_context(self) -> None:
        torch.manual_seed(73)
        encoder = RegionContextEncoder(
            rgb_dim=8,
            thermal_dim=6,
            context_dim=8,
            num_heads=2,
            use_thermal_reliability_gate=True,
        )
        with torch.no_grad():
            encoder.reliability_mlp[-1].weight.zero_()
            encoder.reliability_mlp[-1].bias.fill_(-20.0)
        rgb = torch.randn(1, 4, 8)
        thermal = torch.randn(1, 9, 6)
        region_ids = torch.tensor([[0, 0, 1, 1]])
        gated = encoder(rgb, region_ids, (2, 2), thermal, (3, 3))
        no_thermal = encoder(rgb, region_ids, (2, 2))
        self.assertTrue(torch.isfinite(gated.patch_context).all())
        self.assertTrue(((gated.thermal_reliability >= 0) &
                         (gated.thermal_reliability <= 1)).all())
        self.assertLess(float(gated.thermal_reliability.max()), 1e-4)
        # With rho approximately zero, the RGB-derived context remains and
        # agrees with the same encoder when the thermal component is absent.
        torch.testing.assert_close(
            gated.patch_context, no_thermal.patch_context, atol=2e-4, rtol=2e-4
        )

    def test_reliability_zero_for_missing_dropped_and_padded_regions(self) -> None:
        encoder = RegionContextEncoder(
            rgb_dim=8,
            thermal_dim=6,
            context_dim=8,
            num_heads=2,
            use_thermal_reliability_gate=True,
        )
        rgb = torch.randn(2, 4, 8)
        thermal = torch.randn(2, 4, 6)
        # Sample one has two valid regions; sample two has one, leaving a
        # padded region in the batch representation.
        region_ids = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]])
        output = encoder(
            rgb,
            region_ids,
            (2, 2),
            thermal,
            (2, 2),
            thermal_available=torch.tensor([False, True]),
        )
        self.assertTrue(torch.equal(output.thermal_reliability[0],
                                    torch.zeros_like(output.thermal_reliability[0])))
        self.assertEqual(float(output.thermal_reliability[1, 1]), 0.0)
        missing = encoder(rgb[:1], region_ids[:1], (2, 2))
        self.assertTrue(torch.equal(missing.thermal_reliability,
                                    torch.zeros_like(missing.thermal_reliability)))

    def test_aux_lambda_zero_has_no_head_and_default_forward_api(self) -> None:
        model = _model(use_thermal=True, use_region_routing=True)
        self.assertEqual(model.thermal_aux_lambda, 0.0)
        self.assertNotIn("thermal_aux_head", model.image_adapter)
        image, thermal, region_map = _inputs()
        output = model.eval()(image, thermal=thermal, region_map=region_map)
        self.assertEqual(len(output), 4)

    def test_aux_head_gradients_and_checkpoint_state(self) -> None:
        model = _model(
            use_thermal=True,
            use_region_routing=True,
            thermal_aux_lambda=0.2,
        ).train()
        image, thermal, region_map = _inputs()
        output = model(image, thermal=thermal, region_map=region_map,
                       return_thermal_aux=True)
        self.assertEqual(output[-1].shape, (1, 2))
        output[-1].sum().backward()
        self.assertIsNotNone(model.thermal_branch.patch_embed.weight.grad)
        self.assertIsNotNone(
            model.image_adapter["thermal_aux_head"][1].weight.grad
        )
        state = model.state_dict()
        restored = _model(
            use_thermal=True,
            use_region_routing=True,
            thermal_aux_lambda=0.2,
        )
        restored.load_state_dict(state, strict=True)

    def test_dropout_suppresses_router_but_not_genuine_aux_thermal(self) -> None:
        model = _model(
            use_thermal=True,
            use_region_routing=True,
            modality_dropout=1.0,
            thermal_aux_lambda=0.2,
            use_thermal_reliability_gate=True,
        ).train()
        captured = []
        handle = model.image_adapter["region_contexts"][0].register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        image, thermal, region_map = _inputs()
        try:
            real = model(
                image,
                thermal=thermal,
                region_map=region_map,
                return_thermal_aux=True,
            )
            real[-1].sum().backward()
            real_aux = real[-1].detach().clone()
            self.assertFalse(model.last_thermal_available.any())
            self.assertTrue(torch.equal(
                captured[0].thermal_region_features,
                torch.zeros_like(captured[0].thermal_region_features),
            ))
            self.assertTrue(torch.equal(
                captured[0].thermal_reliability,
                torch.zeros_like(captured[0].thermal_reliability),
            ))
            self.assertGreater(
                float(model.thermal_branch.patch_embed.weight.grad.abs().sum()),
                0.0,
            )
            model.zero_grad(set_to_none=True)
            zero_aux = model(
                image,
                thermal=torch.zeros_like(thermal),
                region_map=region_map,
                return_thermal_aux=True,
            )[-1].detach()
            self.assertGreater(float((real_aux - zero_aux).abs().max()), 1e-6)
        finally:
            handle.remove()

    def test_cli_rejects_invalid_optional_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires both thermal"):
            validate_args(_args(variant="C", use_thermal_reliability_gate=True))
        with self.assertRaisesRegex(ValueError, "requires a thermal variant"):
            validate_args(_args(variant="A", thermal_aux_lambda=0.1))
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            validate_args(_args(thermal_aux_lambda=float("nan")))

    def test_smoke_output_shape_helper_accepts_auxiliary_fifth_value(self) -> None:
        segmentation = [torch.zeros(1, 4, 8) for _ in range(12)]
        detection = torch.zeros(1, 8)
        report = _output_shape_report(
            (segmentation, detection, torch.zeros(()), torch.zeros(()),
             torch.zeros(1, 2))
        )
        self.assertEqual(report["segmentation_maps"], 12)
        self.assertEqual(report["detection_shape"], [1, 8])


if __name__ == "__main__":
    unittest.main()
