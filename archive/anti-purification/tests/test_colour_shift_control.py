"""從 PNG 反查 tone curve 的那一段；不需要 GPU。

要擋的失效是靜默的：反查出來的查找表若有一格是錯的，套用之後仍然是一張
看起來合理的圖，指標照樣算得出來。
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.colour_shift_control import apply_lut, build_lut


def curve(values: np.ndarray, channel: int) -> np.ndarray:
    """逐通道不同的單調映射，扮演 tone curve。"""
    gamma = (0.7, 1.0, 1.4)[channel]
    return np.clip(np.rint(255.0 * (values / 255.0) ** gamma), 0, 255).astype(np.uint8)


class BuildLutTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        rng = np.random.default_rng(0)
        self.orig = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        self.defended = np.stack(
            [curve(self.orig[..., c], c) for c in range(3)], axis=-1)
        self.orig_path = self.directory / "x__orig.png"
        self.def_path = self.directory / "x__def.png"
        Image.fromarray(self.orig).save(self.orig_path)
        Image.fromarray(self.defended).save(self.def_path)

    def test_a_global_mapping_has_zero_spread(self):
        """同一個輸入色階只對到一個輸出值，這是反查成立的前提。"""
        _, spread, _ = build_lut(self.orig_path, self.def_path)
        self.assertEqual(spread, 0)

    def test_the_recovered_lut_is_the_curve(self):
        lut, _, missing = build_lut(self.orig_path, self.def_path)
        self.assertEqual(missing, 0)
        for c in range(3):
            np.testing.assert_array_equal(lut[c], curve(np.arange(256), c))

    def test_applying_it_reproduces_the_defended_image(self):
        lut, _, _ = build_lut(self.orig_path, self.def_path)
        np.testing.assert_array_equal(apply_lut(self.orig_path, lut), self.defended)


class MissingLevelTests(unittest.TestCase):
    """原圖沒有出現過的色階要被數出來，並且由相鄰色階內插，不是留 0。"""

    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        levels = np.arange(0, 256, 4, dtype=np.uint8)          # 只有 64 個色階
        orig = np.tile(levels, (64, 1))[..., None].repeat(3, axis=-1)
        defended = np.stack([curve(orig[..., c], c) for c in range(3)], axis=-1)
        self.orig_path = self.directory / "x__orig.png"
        self.def_path = self.directory / "x__def.png"
        Image.fromarray(orig).save(self.orig_path)
        Image.fromarray(defended).save(self.def_path)

    def test_missing_levels_are_counted(self):
        _, spread, missing = build_lut(self.orig_path, self.def_path)
        self.assertEqual(spread, 0)
        self.assertEqual(missing, 3 * (256 - 64))

    def test_missing_levels_are_interpolated_not_left_at_zero(self):
        lut, _, _ = build_lut(self.orig_path, self.def_path)
        for c in range(3):
            self.assertTrue(np.all(np.diff(lut[c].astype(int)) >= 0),
                            "填補之後仍然要是單調的")
            # 未觀察到的色階（例：1、2、3）不可以塌成 0。
            self.assertGreater(int(lut[c][3]), 0)
            self.assertLessEqual(abs(int(lut[c][4]) - int(curve(np.array([4]), c)[0])), 1)


class SpreadDetectionTests(unittest.TestCase):
    def test_a_spatially_varying_map_is_reported_not_averaged_away(self):
        """空間相依的映射反查不出曲線，這時 spread 必須大於 0。"""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        orig = np.zeros((8, 8, 3), np.uint8) + 100
        defended = np.zeros((8, 8, 3), np.uint8) + 120
        defended[:4] = 140                                      # 同一色階兩個輸出
        orig_path = directory / "x__orig.png"
        def_path = directory / "x__def.png"
        Image.fromarray(orig).save(orig_path)
        Image.fromarray(defended).save(def_path)
        _, spread, _ = build_lut(orig_path, def_path)
        self.assertEqual(spread, 20)


if __name__ == "__main__":
    unittest.main()
