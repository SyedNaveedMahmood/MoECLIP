"""Unit tests for RGB region pooling and thermal context lookup."""

from __future__ import annotations

import unittest

import torch

from model.region_context import (
    RegionContextEncoder,
    RegionThermalAttention,
    identity_patch_regions,
    pixel_regions_to_patch_regions,
    pool_patch_regions,
)


class RegionContextTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_pixel_regions_use_modal_label_per_patch(self) -> None:
        region_map = torch.tensor(
            [[
                [0, 0, 1, 1],
                [0, 2, 1, 1],
                [3, 3, 4, 4],
                [3, 3, 5, 4],
            ]],
            dtype=torch.long,
        )
        patch_regions = pixel_regions_to_patch_regions(region_map, (2, 2))
        torch.testing.assert_close(
            patch_regions, torch.tensor([[0, 1, 3, 4]], dtype=torch.long)
        )

    def test_region_pooling_and_patch_membership_are_exact(self) -> None:
        tokens = torch.tensor(
            [[[1.0, 3.0], [3.0, 5.0], [10.0, 20.0], [14.0, 24.0]]]
        )
        region_ids = torch.tensor([[7, 7, 9, 9]])
        output = pool_patch_regions(tokens, region_ids, (2, 2))

        torch.testing.assert_close(
            output.region_features,
            torch.tensor([[[2.0, 4.0], [12.0, 22.0]]]),
        )
        torch.testing.assert_close(output.region_counts, torch.tensor([[2, 2]]))
        torch.testing.assert_close(
            output.patch_region_indices, torch.tensor([[0, 0, 1, 1]])
        )
        self.assertTrue(output.valid_regions.all())

    def test_soft_coordinate_bias_prefers_but_does_not_force_nearby_tokens(self) -> None:
        attention = RegionThermalAttention(
            rgb_dim=4,
            thermal_dim=4,
            attention_dim=4,
            num_heads=1,
            coordinate_bias_strength=2.0,
            coordinate_bias_sigma=0.75,
        ).eval()
        with torch.no_grad():
            attention.query.weight.zero_()
            attention.key.weight.zero_()
        rgb_regions = torch.zeros(1, 2, 4)
        region_coordinates = torch.tensor([[[-0.9, 0.0], [0.9, 0.0]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        thermal = torch.randn(1, 2, 4)

        _, weights = attention(
            rgb_regions, region_coordinates, valid, thermal, (1, 2)
        )

        self.assertGreater(float(weights[0, 0, 0]), float(weights[0, 0, 1]))
        self.assertGreater(float(weights[0, 1, 1]), float(weights[0, 1, 0]))
        self.assertTrue((weights > 0.0).all())
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2))

    def test_each_region_attends_over_full_thermal_grid_with_gradients(self) -> None:
        encoder = RegionContextEncoder(
            rgb_dim=8,
            thermal_dim=6,
            context_dim=8,
            num_heads=2,
        )
        rgb = torch.randn(1, 4, 8, requires_grad=True)
        region_ids = torch.tensor([[0, 0, 1, 1]])
        thermal = torch.randn(1, 9, 6, requires_grad=True)

        output = encoder(rgb, region_ids, (2, 2), thermal, (3, 3))
        self.assertEqual(output.patch_context.shape, (1, 4, 8))
        self.assertEqual(output.thermal_attention.shape, (1, 2, 9))
        torch.testing.assert_close(
            output.thermal_attention.sum(dim=-1), torch.ones(1, 2)
        )

        output.thermal_region_features.square().sum().backward()
        self.assertIsNotNone(thermal.grad)
        per_token_gradient = thermal.grad.abs().sum(dim=-1)
        self.assertTrue((per_token_gradient > 0.0).all())

    def test_padded_regions_and_missing_thermal_are_masked(self) -> None:
        encoder = RegionContextEncoder(
            rgb_dim=8,
            thermal_dim=6,
            context_dim=8,
            num_heads=2,
        ).eval()
        rgb = torch.randn(2, 4, 8)
        region_ids = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]])
        thermal = torch.randn(2, 4, 6)
        available = torch.tensor([False, True])

        output = encoder(
            rgb, region_ids, (2, 2), thermal, (2, 2), available
        )

        self.assertEqual(output.region_context.shape, (2, 4, 8))
        self.assertEqual(output.pool.valid_regions[0].sum().item(), 1)
        self.assertEqual(output.pool.valid_regions[1].sum().item(), 4)
        self.assertEqual(output.thermal_region_features[0].abs().sum().item(), 0.0)
        self.assertEqual(output.thermal_attention[0].abs().sum().item(), 0.0)
        self.assertEqual(output.region_context[0, 1:].abs().sum().item(), 0.0)

    def test_identity_regions_support_patch_conditioned_ablation(self) -> None:
        identities = identity_patch_regions(2, (2, 3))
        self.assertEqual(identities.shape, (2, 6))
        torch.testing.assert_close(identities[0], torch.arange(6))
        torch.testing.assert_close(identities[1], torch.arange(6))


if __name__ == "__main__":
    unittest.main()
