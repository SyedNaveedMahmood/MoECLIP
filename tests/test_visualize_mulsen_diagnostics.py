"""Focused tests for the read-only qualitative diagnostics helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from tools.visualize_mulsen_diagnostics import (
    _RouterCapture,
    _patch_map,
    _reliability_display,
    validate_visualization_scope,
)


class VisualizationHelperTest(unittest.TestCase):
    def test_visualization_scope_is_development_only(self) -> None:
        validate_visualization_scope("development")
        with self.assertRaisesRegex(ValueError, "development protocol"):
            validate_visualization_scope("final")

    def test_router_capture_keeps_patch_assignments_and_context_changes(self) -> None:
        module = SimpleNamespace(
            d_model=2,
            router_context_dim=2,
            gate=nn.Linear(2, 2, bias=False),
            context_gate=nn.Linear(2, 2, bias=False),
            context_expert_mask=torch.ones(2),
        )
        with torch.no_grad():
            module.gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            module.context_gate.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        capture = _RouterCapture()
        hidden = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]], dtype=torch.float32
        )
        context = torch.tensor(
            [[[0.0, 0.0]], [[0.0, 0.0]], [[2.0, 0.0]]], dtype=torch.float32
        )
        capture.hook(module, (hidden,), {"router_context": context})
        self.assertEqual(capture.base_top1.tolist(), [1, 0])
        self.assertEqual(capture.context_top1.tolist(), [1, 1])
        self.assertEqual(capture.context_changed.tolist(), [False, True])

    def test_patch_map_requires_square_token_grid(self) -> None:
        mapped = _patch_map(torch.tensor([0, 1, 2, 3]))
        self.assertEqual(mapped.shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "not square"):
            _patch_map(torch.tensor([0, 1, 2]))

    def test_reliability_is_broadcast_from_regions_to_patch_grid(self) -> None:
        class Pool:
            patch_region_indices = torch.tensor([[0, 0, 1, 1]])

        class Output:
            thermal_reliability = torch.tensor([[0.2, 0.8]])
            pool = Pool()

        mapped = _reliability_display(Output())
        self.assertEqual(mapped.shape, (2, 2))
        np.testing.assert_allclose(
            mapped, np.array([[0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
