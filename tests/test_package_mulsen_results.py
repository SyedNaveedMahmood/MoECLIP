from __future__ import annotations

import unittest

from tools.package_mulsen_results import (
    _portable_config,
    _portable_evaluation,
    _without_values,
)


class PackageMulSenResultsTest(unittest.TestCase):
    def test_portable_config_changes_only_machine_paths(self) -> None:
        source = {
            "data_root": r"C:\private\MulSen_AD",
            "lr": 5e-5,
            "thermal_normalization": {
                "mean": 0.5,
                "file": r"C:\private\stats.json",
            },
        }
        portable = _portable_config(source, "development")

        self.assertEqual(portable["data_root"], "${MULSEN_DATA_ROOT}")
        self.assertEqual(portable["lr"], source["lr"])
        self.assertEqual(portable["thermal_normalization"]["mean"], 0.5)
        self.assertEqual(
            portable["thermal_normalization"]["file"],
            "results/mulsen/development/thermal_stats_development.json",
        )
        self.assertEqual(source["data_root"], r"C:\private\MulSen_AD")

    def test_large_per_sample_arrays_are_removed_recursively(self) -> None:
        compact = _without_values(
            {
                "sample_count": 2,
                "values": [1, 2],
                "nested": {
                    "mean": 1.5,
                    "combined_scores": [0.1, 0.2],
                },
            }
        )
        self.assertEqual(compact, {"sample_count": 2, "nested": {"mean": 1.5}})

    def test_portable_evaluation_retains_scores_and_removes_paths(self) -> None:
        report = {
            "experiment_config": {
                "data_root": r"C:\private\MulSen_AD",
                "thermal_normalization": {
                    "file": r"C:\private\stats.json",
                },
            },
            "evaluations": [
                {
                    "checkpoint": r"C:\private\mulsen_epoch_003.pth",
                    "scores": [0.1, 0.2],
                }
            ],
            "selected_checkpoint": {
                "checkpoint": r"C:\private\mulsen_epoch_003.pth",
                "epoch": 3,
            },
        }

        portable = _portable_evaluation(report, "final")

        self.assertEqual(
            portable["experiment_config"]["data_root"],
            "${MULSEN_DATA_ROOT}",
        )
        self.assertEqual(
            portable["evaluations"][0]["checkpoint"],
            "mulsen_epoch_003.pth",
        )
        self.assertEqual(portable["evaluations"][0]["scores"], [0.1, 0.2])
        self.assertEqual(
            portable["selected_checkpoint"]["checkpoint"],
            "mulsen_epoch_003.pth",
        )
        self.assertEqual(
            report["evaluations"][0]["checkpoint"],
            r"C:\private\mulsen_epoch_003.pth",
        )


if __name__ == "__main__":
    unittest.main()
