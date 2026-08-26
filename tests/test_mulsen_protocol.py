"""Leakage and record-selection tests for the locked MulSen-AD protocol."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dataset.constants import CLASS_NAMES, REAL_NAMES
from dataset.mulsen_protocol import (
    ALL_CATEGORIES,
    DEVELOPMENT_TRAIN_CATEGORIES,
    DEVELOPMENT_VALIDATION_CATEGORIES,
    FINAL_SEEN_CATEGORIES,
    FINAL_UNSEEN_CATEGORIES,
    build_evaluation_dataset,
    build_training_dataset,
    get_protocol,
)


class _FakeMulSenAD:
    calls = []

    def __init__(self, data_root, split, categories, **kwargs) -> None:
        self.split = split
        self.categories = tuple(categories)
        self.kwargs = kwargs
        self.calls.append(self)
        if split == "train":
            self.records = [
                SimpleNamespace(anomaly_type="good", label_rgbt=0)
                for _ in range(3)
            ]
        else:
            self.records = [
                SimpleNamespace(anomaly_type="good", label_rgbt=0),
                SimpleNamespace(anomaly_type="rgb_visible", label_rgbt=1),
                SimpleNamespace(anomaly_type="pointcloud_only", label_rgbt=0),
                SimpleNamespace(anomaly_type="all_zero", label_rgbt=0),
            ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return self.records[index]


class MulSenProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeMulSenAD.calls.clear()

    def test_category_partitions_are_disjoint_and_complete(self) -> None:
        self.assertFalse(
            set(DEVELOPMENT_TRAIN_CATEGORIES)
            & set(DEVELOPMENT_VALIDATION_CATEGORIES)
        )
        self.assertEqual(
            set(DEVELOPMENT_TRAIN_CATEGORIES)
            | set(DEVELOPMENT_VALIDATION_CATEGORIES),
            set(FINAL_SEEN_CATEGORIES),
        )
        self.assertFalse(set(FINAL_SEEN_CATEGORIES) & set(FINAL_UNSEEN_CATEGORIES))
        self.assertEqual(
            set(FINAL_SEEN_CATEGORIES) | set(FINAL_UNSEEN_CATEGORIES),
            set(ALL_CATEGORIES),
        )
        self.assertEqual(
            get_protocol("final").evaluation_categories,
            FINAL_UNSEEN_CATEGORIES,
        )

    def test_training_uses_normals_and_visible_anomalies_only(self) -> None:
        with patch("dataset.mulsen_protocol.MulSenAD", _FakeMulSenAD):
            dataset = build_training_dataset(
                "unused",
                "development",
                use_region_routing=True,
                augment=True,
                geometry_seed=19,
            )

        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(_FakeMulSenAD.calls), 2)
        normal, anomaly = _FakeMulSenAD.calls
        self.assertEqual(normal.categories, DEVELOPMENT_TRAIN_CATEGORIES)
        self.assertEqual(anomaly.categories, DEVELOPMENT_TRAIN_CATEGORIES)
        self.assertTrue(normal.kwargs["train"])
        self.assertTrue(anomaly.kwargs["train"])
        self.assertTrue(normal.kwargs["joint_geometry"])
        self.assertEqual(normal.kwargs["region_method"], "slic")
        selected = dataset.datasets[1]
        self.assertEqual(selected.indices, (1,))

    def test_evaluation_excludes_unavailable_records_without_relabeling(self) -> None:
        with patch("dataset.mulsen_protocol.MulSenAD", _FakeMulSenAD):
            dataset = build_evaluation_dataset("unused", "final")

        self.assertEqual(dataset.indices, (0, 1))
        self.assertEqual(
            dataset.dataset.categories,
            FINAL_UNSEEN_CATEGORIES,
        )
        self.assertFalse(dataset.dataset.kwargs["train"])
        self.assertFalse(dataset.dataset.kwargs["joint_geometry"])

    def test_prompt_metadata_covers_every_category(self) -> None:
        self.assertEqual(set(CLASS_NAMES["MulSenAD"]), set(ALL_CATEGORIES))
        self.assertEqual(set(REAL_NAMES["MulSenAD"]), set(ALL_CATEGORIES))


if __name__ == "__main__":
    unittest.main()
