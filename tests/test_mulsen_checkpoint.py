"""Round-trip tests for complete MulSen-AD experiment checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from mulsen_checkpoint import load_mulsen_checkpoint, save_mulsen_checkpoint


class _ToyExtension(nn.Module):
    def __init__(self, use_thermal: bool = True) -> None:
        super().__init__()
        self.use_thermal = use_thermal
        self.image_adapter = nn.ModuleDict(
            {
                "moe_adapters": nn.ModuleList([nn.Linear(4, 4)]),
                "region_contexts": nn.ModuleList([nn.Linear(4, 4)]),
                "seg_proj": nn.ModuleList([nn.Linear(4, 3)]),
                "det_proj": nn.Linear(4, 3),
            }
        )
        self.text_adapter = nn.ModuleList([nn.Linear(3, 3)])
        if use_thermal:
            self.thermal_branch = nn.Sequential(nn.Linear(2, 4), nn.GELU())

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        value = self.image_adapter["moe_adapters"][0](rgb)
        value = value + self.image_adapter["region_contexts"][0](value)
        if self.use_thermal:
            value = value + self.thermal_branch(thermal)
        return self.image_adapter["seg_proj"][0](value)


class MulSenCheckpointTest(unittest.TestCase):
    def test_components_optimizer_scheduler_and_config_round_trip(self) -> None:
        torch.manual_seed(29)
        model = _ToyExtension()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2], gamma=0.1
        )
        rgb = torch.randn(2, 4)
        thermal = torch.randn(2, 2)
        model(rgb, thermal).square().mean().backward()
        optimizer.step()
        scheduler.step()
        expected = model(rgb, thermal).detach().clone()
        config = {"protocol_stage": "development", "variant": "D"}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            save_mulsen_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=1,
                experiment_config=config,
            )
            restored = _ToyExtension()
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=9e-2)
            restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                restored_optimizer, milestones=[2], gamma=0.1
            )
            checkpoint = load_mulsen_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                expected_config=config,
                restore_rng=False,
            )

            torch.testing.assert_close(restored(rgb, thermal), expected)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(
                restored_optimizer.state_dict()["param_groups"][0]["lr"],
                optimizer.state_dict()["param_groups"][0]["lr"],
            )
            self.assertEqual(
                restored_scheduler.state_dict(), scheduler.state_dict()
            )

            with self.assertRaisesRegex(ValueError, "config mismatch"):
                load_mulsen_checkpoint(
                    path,
                    model=_ToyExtension(),
                    expected_config={"variant": "B"},
                    restore_rng=False,
                )


if __name__ == "__main__":
    unittest.main()
