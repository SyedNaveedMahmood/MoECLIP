"""Small synthetic smoke tests for the read-only MulSen inspector.

These tests never touch the real dataset and never open a GT image.  Run with:
    python -m unittest tools.test_inspect_mulsen_alignment -v
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.inspect_mulsen_alignment import (
    _image_array,
    discover_pairs,
    image_header_inventory,
    image_stats,
)


class InspectorSmokeTest(unittest.TestCase):
    def test_gt_path_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "GT" / "mask.png"
            path.parent.mkdir()
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(path)
            with self.assertRaises(AssertionError):
                _image_array(path)

    def test_missing_pair_is_reported_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "MulSen_AD"
            rgb = root / "cube" / "RGB" / "test" / "good"
            ir = root / "cube" / "Infrared" / "test" / "good"
            rgb.mkdir(parents=True)
            ir.mkdir(parents=True)
            image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
            image.save(rgb / "0.png")
            pairs, integrity = discover_pairs(root)
            self.assertEqual(pairs, [])
            self.assertEqual(integrity["status"], "failed")
            self.assertIn("missing Infrared counterparts: 1", integrity["issues"])

    def test_container_inventory_uses_signature_not_png_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # This deliberately has a .png suffix but a BMP payload, matching
            # the mixed-container failure mode found in the distributed data.
            bmp_path = root / "bmp_payload.png"
            Image.fromarray(np.full((5, 7, 3), 17, dtype=np.uint8)).save(bmp_path, format="BMP")
            rgba_path = root / "rgba.png"
            rgba = np.zeros((5, 7, 4), dtype=np.uint8)
            rgba[..., :3] = [10, 10, 10]
            rgba[..., 3] = np.arange(35, dtype=np.uint8).reshape(5, 7)
            Image.fromarray(rgba).save(rgba_path, format="PNG")

            inventory = image_header_inventory(
                [bmp_path, rgba_path], audit_grayscale_channels=True
            )

            self.assertEqual(inventory["count"], 2)
            self.assertEqual(inventory["container_counts"], {"BMP": 1, "PNG": 1})
            self.assertEqual(inventory["container_errors"], [])
            self.assertEqual(inventory["decode_errors"], [])
            self.assertEqual(inventory["bmp_dib_records"][0]["bits_per_pixel"], 24)
            self.assertEqual(inventory["png_ihdr_records"][0]["color_type"], 6)
            by_name = {Path(item["path"]).name: item for item in inventory["pil_records"]}
            self.assertEqual(by_name["bmp_payload.png"]["pil_format"], "BMP")
            self.assertEqual(by_name["rgba.png"]["pil_format"], "PNG")
            self.assertEqual(by_name["rgba.png"]["pil_mode"], "RGBA")
            pixel_audit = inventory["pixel_encoding_audit"]
            self.assertEqual(pixel_audit["numpy_dtypes"], {"uint8": 2})
            self.assertEqual(pixel_audit["exact_grayscale_channel_image_count"], 2)
            self.assertEqual(pixel_audit["nonexact_grayscale_channel_image_count"], 0)
            self.assertEqual(pixel_audit["global_min"], 0.0)
            self.assertEqual(pixel_audit["global_max"], 34.0)

            stats = image_stats(rgba_path, root, "RGB")
            self.assertEqual(stats["numpy_dtype"], "uint8")
            self.assertEqual(stats["alpha_channel"]["min"], 0.0)
            self.assertEqual(stats["alpha_channel"]["max"], 34.0)
            self.assertFalse(stats["alpha_channel"]["all_opaque"])
            self.assertTrue(stats["channels_0_1_2_exactly_identical"])


if __name__ == "__main__":
    unittest.main()
