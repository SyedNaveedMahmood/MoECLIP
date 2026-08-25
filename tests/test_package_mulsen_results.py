from __future__ import annotations

import unittest

from tools.package_mulsen_results import _portable_config, _without_values


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


if __name__ == "__main__":
    unittest.main()
