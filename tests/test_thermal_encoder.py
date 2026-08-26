"""Focused smoke tests for the MulSen-AD thermal conditioning encoder."""

from __future__ import annotations

import io
import unittest

import torch

from model.thermal_branch import ThermalEncoder


class ThermalEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_518_input_produces_clip_sized_patch_grid_without_cls(self) -> None:
        encoder = ThermalEncoder(width=32, output_dim=48, depth=4)
        thermal = torch.rand(1, 1, 518, 518)

        with torch.no_grad():
            output = encoder(thermal)

        self.assertEqual(output.grid_size, (37, 37))
        self.assertEqual(output.tokens.shape, (1, 1369, 48))
        self.assertEqual(len(output.taps), 4)
        for tap in output.taps:
            self.assertEqual(tap.shape, (1, 1369, 48))

    def test_backward_reaches_stem_blocks_and_all_tap_projections(self) -> None:
        encoder = ThermalEncoder(width=32, output_dim=48, depth=4)
        thermal = torch.rand(2, 1, 56, 56)

        output = encoder(thermal)
        loss = sum(tap.square().mean() for tap in output.taps)
        loss.backward()

        parameters = [
            encoder.patch_embed.weight,
            encoder.blocks[0].depthwise.weight,
            encoder.blocks[0].mlp[0].weight,
            encoder.blocks[-1].depthwise.weight,
            encoder.blocks[-1].mlp[0].weight,
        ]
        parameters.extend(projection.weight for projection in encoder.tap_projections)
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_state_dict_round_trip_preserves_eval_output(self) -> None:
        encoder = ThermalEncoder(width=24, output_dim=32, depth=2).eval()
        thermal = torch.rand(1, 1, 56, 56)
        with torch.no_grad():
            expected = encoder(thermal).tokens

        checkpoint = io.BytesIO()
        torch.save(encoder.state_dict(), checkpoint)
        checkpoint.seek(0)

        restored = ThermalEncoder(width=24, output_dim=32, depth=2).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        with torch.no_grad():
            actual = restored(thermal).tokens

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_input_contract_rejects_ambiguous_thermal_tensors(self) -> None:
        encoder = ThermalEncoder(width=16, output_dim=16, depth=1)
        invalid_inputs = (
            torch.rand(1, 3, 56, 56),
            torch.zeros(1, 1, 56, 56, dtype=torch.uint8),
            torch.rand(1, 1, 55, 56),
        )
        expected_errors = (ValueError, TypeError, ValueError)

        for thermal, expected_error in zip(invalid_inputs, expected_errors):
            with self.subTest(shape=tuple(thermal.shape), dtype=str(thermal.dtype)):
                with self.assertRaises(expected_error):
                    encoder(thermal)

    def test_default_encoder_remains_modest(self) -> None:
        encoder = ThermalEncoder()
        trainable_parameters = sum(
            parameter.numel() for parameter in encoder.parameters()
            if parameter.requires_grad
        )
        self.assertLess(trainable_parameters, 2_000_000)


if __name__ == "__main__":
    unittest.main()
