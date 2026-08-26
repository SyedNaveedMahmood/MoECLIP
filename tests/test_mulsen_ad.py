"""Synthetic, offline tests for the strict MulSen-AD RGB/IR adapter."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset.mulsen_ad import MulSenAD, MulSenIntegrityError


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _make_fixture(root: Path) -> Path:
    category = root / "toy"
    types = {
        "all_zero_anomaly": (0, 0, 0),
        "both": (1, 1, 0),
        "good": (0, 0, 0),
        "rgb_only": (1, 0, 0),
        "ir_only": (0, 1, 0),
        "pc_only": (0, 0, 1),
    }
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    thermal = np.full((8, 10, 3), 40, dtype=np.uint8)
    mask_rgb = np.zeros((8, 10), dtype=np.uint8)
    mask_rgb[2:5, 3:7] = 255
    mask_ir = np.zeros((8, 10), dtype=np.uint8)
    mask_ir[1:4, 6:9] = 255

    # BMP payloads verify that the loader trusts Pillow's decoded format, not
    # an assumed acquisition/storage container.
    _save(category / "RGB" / "train" / "0.bmp", rgb)
    _save(category / "Infrared" / "train" / "0.bmp", thermal)
    for anomaly, (label_rgb, label_ir, label_pc) in types.items():
        suffix = ".bmp" if anomaly == "ir_only" else ".png"
        _save(category / "RGB" / "test" / anomaly / f"0{suffix}", rgb)
        _save(category / "Infrared" / "test" / anomaly / f"0{suffix}", thermal)
        gt_rgb = category / "RGB" / "GT" / anomaly
        gt_ir = category / "Infrared" / "GT" / anomaly
        gt_rgb.mkdir(parents=True, exist_ok=True)
        gt_ir.mkdir(parents=True, exist_ok=True)
        if label_rgb:
            _save(gt_rgb / "0.png", mask_rgb)
        if label_ir:
            _save(gt_ir / "0.png", mask_rgb if anomaly == "both" else mask_ir)
        if anomaly == "rgb_only":
            _save(gt_rgb / "orphan.png", mask_rgb)
        (gt_rgb / "data.csv").write_text(
            "object,RGB,infrared,pointcloud\n0,{},{},{}\n".format(
                label_rgb, label_ir, label_pc
            ),
            encoding="utf-8",
        )
    return category


class MulSenADTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _test_dataset(self, root: Path, **kwargs) -> MulSenAD:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return MulSenAD(root.parent, split="test", categories=["toy"], **kwargs)

    def test_pairing_shapes_labels_and_separate_masks(self) -> None:
        root = _make_fixture(self.tmp_path)
        train = MulSenAD(root.parent, split="train", categories=["toy"], img_size=16)
        self.assertEqual(len(train), 1)
        self.assertEqual(train[0]["image"].shape, (3, 16, 16))
        self.assertEqual(train[0]["thermal"].shape, (1, 16, 16))
        self.assertAlmostEqual(float(train[0]["thermal"].mean()), 40.0 / 255.0, places=6)

        test = self._test_dataset(root, img_size=16)
        by_type = {sample["anomaly_type"]: sample for sample in test}
        self.assertEqual(by_type["rgb_only"]["label"].item(), 1)
        self.assertEqual(by_type["rgb_only"]["label_rgb"].item(), 1)
        self.assertEqual(by_type["rgb_only"]["label_thermal"].item(), 0)
        self.assertTrue(by_type["rgb_only"]["mask"].equal(by_type["rgb_only"]["mask_rgb"]))
        self.assertFalse(by_type["rgb_only"]["mask_valid_thermal"].item())
        self.assertEqual(by_type["ir_only"]["label"].item(), 1)
        self.assertFalse(by_type["ir_only"]["mask_valid"].item())
        self.assertEqual(by_type["ir_only"]["mask_rgb"].sum().item(), 0)
        self.assertGreater(by_type["ir_only"]["mask_thermal"].sum().item(), 0)
        self.assertEqual(by_type["pc_only"]["label"].item(), 0)
        self.assertEqual(by_type["pc_only"]["label_any"].item(), 1)
        self.assertEqual(by_type["pc_only"]["mask"].sum().item(), 0)
        self.assertEqual(by_type["all_zero_anomaly"]["label"].item(), 0)
        self.assertEqual(by_type["all_zero_anomaly"]["label_any"].item(), 0)
        self.assertNotEqual(by_type["all_zero_anomaly"]["anomaly_type"], "good")
        self.assertTrue(any("orphan RGB mask" in item for item in test.audit_warnings))

    def test_ir_payload_must_be_equal_channel_grayscale(self) -> None:
        root = _make_fixture(self.tmp_path)
        bad = np.zeros((8, 10, 3), dtype=np.uint8)
        bad[..., 0] = 10
        bad[..., 1] = 20
        _save(root / "Infrared" / "train" / "0.bmp", bad)
        dataset = MulSenAD(root.parent, split="train", categories=["toy"], img_size=8)
        with self.assertRaisesRegex(MulSenIntegrityError, "grayscale-encoding tolerances"):
            _ = dataset[0]

    def test_audited_sparse_low_amplitude_ir_channel_differences_are_allowed(self) -> None:
        root = _make_fixture(self.tmp_path)
        sparse = np.full((8, 10, 3), 40, dtype=np.uint8)
        # 7/80 = 8.75%, below the audited 10% per-image bound. The spread of
        # four matches the release file that exposed the previous 5% mismatch.
        sparse.reshape(-1, 3)[:7, 0] = 44
        _save(root / "Infrared" / "train" / "0.bmp", sparse)
        dataset = MulSenAD(root.parent, split="train", categories=["toy"], img_size=8)

        sample = dataset[0]

        self.assertEqual(sample["thermal"].shape, (1, 8, 8))
        self.assertTrue(torch.isfinite(sample["thermal"]).all())

    def test_slic_uses_rgb_not_ground_truth(self) -> None:
        root = _make_fixture(self.tmp_path)
        first = self._test_dataset(
            root, img_size=16, region_method="slic", slic_segments=4, slic_compactness=1
        )
        first_map = next(
            sample["region_map"]
            for sample in first
            if sample["anomaly_type"] == "rgb_only"
        ).clone()

        # Changing only supervision cannot change RGB-only SLIC regions.
        _save(root / "RGB" / "GT" / "rgb_only" / "0.png", np.full((8, 10), 255, np.uint8))
        second = self._test_dataset(
            root, img_size=16, region_method="slic", slic_segments=4, slic_compactness=1
        )
        second_map = next(
            sample["region_map"]
            for sample in second
            if sample["anomaly_type"] == "rgb_only"
        )
        self.assertEqual(first_map.dtype, torch.int64)
        self.assertEqual(first_map.shape, (16, 16))
        self.assertTrue(torch.equal(first_map, second_map))
        labels = torch.unique(first_map)
        self.assertTrue(torch.equal(labels, torch.arange(labels.numel(), dtype=torch.int64)))

    def test_joint_geometry_is_deterministic_and_shared(self) -> None:
        root = _make_fixture(self.tmp_path)
        kwargs = dict(
            img_size=16,
            train=True,
            joint_geometry=True,
            geometry_seed=13,
            rotation_degrees=20,
            translation_fraction=0.1,
            horizontal_flip_prob=1,
            vertical_flip_prob=1,
        )

        def get_both(dataset: MulSenAD):
            return next(sample for sample in dataset if sample["anomaly_type"] == "both")

        first = get_both(self._test_dataset(root, **kwargs))
        second = get_both(self._test_dataset(root, **kwargs))
        self.assertTrue(torch.equal(first["image"], second["image"]))
        self.assertTrue(torch.equal(first["thermal"], second["thermal"]))
        self.assertTrue(torch.equal(first["mask_rgb"], second["mask_rgb"]))
        self.assertGreater(first["mask_rgb"].sum().item(), 0)
        self.assertTrue(torch.equal(first["mask_rgb"], first["mask_thermal"]))

    def test_exact_filename_pairing_and_missing_positive_mask_fail(self) -> None:
        root = _make_fixture(self.tmp_path / "pairing")
        (root / "Infrared" / "test" / "good" / "0.png").unlink()
        _save(
            root / "Infrared" / "test" / "good" / "0.bmp",
            np.full((8, 10, 3), 40, dtype=np.uint8),
        )
        with self.assertRaisesRegex(MulSenIntegrityError, "pairing mismatch"):
            self._test_dataset(root, img_size=8)

        missing_root = _make_fixture(self.tmp_path / "missing_mask")
        (missing_root / "RGB" / "GT" / "rgb_only" / "0.png").unlink()
        with self.assertRaisesRegex(MulSenIntegrityError, "Positive RGB label has no mask"):
            self._test_dataset(missing_root, img_size=8)

    def test_default_collation_with_and_without_regions(self) -> None:
        root = _make_fixture(self.tmp_path)
        plain = self._test_dataset(root, img_size=16)
        batch = next(iter(DataLoader(plain, batch_size=2, shuffle=False)))
        self.assertEqual(batch["image"].shape, (2, 3, 16, 16))
        self.assertEqual(batch["thermal"].shape, (2, 1, 16, 16))
        self.assertNotIn("region_map", batch)
        self.assertEqual(len(batch["source_paths"]["mask_rgb"]), 2)

        segmented = self._test_dataset(
            root, img_size=16, region_method="slic", slic_segments=4, slic_compactness=1
        )
        segmented_batch = next(iter(DataLoader(segmented, batch_size=2, shuffle=False)))
        self.assertEqual(segmented_batch["region_map"].shape, (2, 16, 16))

    def test_native_mask_size_must_match_its_modality(self) -> None:
        root = _make_fixture(self.tmp_path)
        _save(root / "RGB" / "GT" / "rgb_only" / "0.png", np.ones((7, 10), np.uint8) * 255)
        dataset = self._test_dataset(root, img_size=16)
        sample_index = next(
            index
            for index, record in enumerate(dataset.records)
            if record.anomaly_type == "rgb_only"
        )
        with self.assertRaisesRegex(MulSenIntegrityError, "Mask/image size mismatch"):
            _ = dataset[sample_index]


if __name__ == "__main__":
    unittest.main()
