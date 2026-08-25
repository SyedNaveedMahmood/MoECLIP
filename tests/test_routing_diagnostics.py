"""Deterministic unit tests for the read-only routing statistics."""

import math
import unittest

import torch

from tools.inspect_mulsen_routing import RoutingStats


class RoutingStatsTest(unittest.TestCase):
    def test_balanced_uniform_router(self) -> None:
        stats = RoutingStats(num_experts=4, top_k=2)
        stats.update(torch.zeros(8, 4))
        summary = stats.summary()

        self.assertEqual(summary["token_count"], 8)
        self.assertEqual(summary["mean_probabilities"], [0.25] * 4)
        self.assertAlmostEqual(summary["mean_normalized_token_entropy"], 1.0)
        self.assertAlmostEqual(summary["soft_load_cv_squared"], 0.0)
        self.assertAlmostEqual(
            summary["effective_experts_from_mean_probabilities"], 4.0
        )
        self.assertAlmostEqual(summary["mean_topk_probability_mass"], 0.5)

    def test_context_changes_top1_without_mutating_inputs(self) -> None:
        base = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        context = torch.tensor([[-3.0, 3.0], [3.0, -3.0]])
        base_copy = base.clone()
        context_copy = context.clone()
        stats = RoutingStats(num_experts=2, top_k=1)
        stats.update(base, context)
        summary = stats.summary()

        self.assertTrue(torch.equal(base, base_copy))
        self.assertTrue(torch.equal(context, context_copy))
        self.assertEqual(summary["context_changed_top1_fraction"], 1.0)
        self.assertTrue(summary["context_observed"])
        self.assertTrue(math.isfinite(summary["context_to_base_abs_ratio"]))


if __name__ == "__main__":
    unittest.main()
