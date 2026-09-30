"""DiffVax 接進 defence_run 的接線；不需要 GPU，也不需要官方權重。

權重本身由 `tests/test_diffvax.py` 管，這裡只釘住接線：條件清單、報表欄、
遮罩的必要性與極性。網路用隨機初始化的 `NestedUNet`——這些性質是接線的
性質，與權重的值無關。
"""

import unittest

import torch

from scripts import defence_run as run
from src.baselines import diffvax


def _immunizer() -> diffvax.NestedUNet:
    """隨機初始化的 immunizer。不載權重：本檔要測的是接線。"""
    model = diffvax.NestedUNet(num_classes=3)
    return model.eval().requires_grad_(False)


def _mask(size: int = 64) -> torch.Tensor:
    """右半白（重繪），左半黑（保留）。白＝重繪，與 `make_masks.py` 同極性。"""
    mask = torch.zeros(1, 1, size, size)
    mask[:, :, :, size // 2:] = 1.0
    return mask


class ConditionListTests(unittest.TestCase):
    def test_diffvax_is_one_of_the_conditions(self):
        self.assertIn("diffvax", run.CONDITIONS)
        self.assertIn("diffvax", run.FEEDFORWARD_CONDITIONS)

    def test_diffvax_is_not_a_pgd_condition(self):
        """`BaselineSpec` 的每一欄在前饋式免疫器上都沒有對應值。"""
        self.assertNotIn("diffvax", run.PGD_SPECS)
        self.assertNotIn("diffvax", run.DCT_CONDITIONS)

    def test_the_copied_prompts_still_match_their_modules(self):
        run._check_prompts()

    def test_diffvax_takes_no_text_condition(self):
        value, source = run.solver_prompt_of("diffvax", {"class": "man"})
        self.assertEqual(value, "")
        self.assertIn("前向", source)


class SolveTests(unittest.TestCase):
    def test_the_budget_columns_are_blank_not_zero(self):
        """填 0 會被讀成「預算為零」；這一條根本沒有硬性 L∞ 預算。"""
        x01 = torch.rand(1, 3, 64, 64)
        _, cfg = run.solve(None, "diffvax", x01, seed=0,
                           immunizer=_immunizer(), mask01=_mask())
        self.assertEqual(cfg["eps"], "")
        self.assertEqual(cfg["eps_pixel01"], "")
        self.assertEqual(cfg["norm"], "none")
        self.assertEqual(cfg["steps"], 1)
        self.assertIn("2411.17957", cfg["spec_source"])

    def test_the_defended_image_keeps_shape_and_range(self):
        x01 = torch.rand(1, 3, 64, 64)
        x_def, _ = run.solve(None, "diffvax", x01, seed=0,
                             immunizer=_immunizer(), mask01=_mask())
        self.assertEqual(x_def.shape, x01.shape)
        self.assertGreaterEqual(float(x_def.min()), 0.0)
        self.assertLessEqual(float(x_def.max()), 1.0)

    def test_the_repaint_region_comes_back_untouched(self):
        """擾動落在白區之外；白區的像素在交付圖上必須是原圖的像素。"""
        x01 = torch.rand(1, 3, 64, 64)
        mask = _mask()
        x_def, _ = run.solve(None, "diffvax", x01, seed=0,
                             immunizer=_immunizer(), mask01=mask)
        white = mask.expand_as(x01) > 0.5
        self.assertTrue(torch.allclose(x_def[white], x01[white], atol=1e-6))
        self.assertFalse(torch.allclose(x_def[~white], x01[~white], atol=1e-6))

    def test_a_missing_mask_stops_the_run(self):
        """沒有遮罩就沒有「哪裡該加擾動」的定義，不可退回全圖。"""
        x01 = torch.rand(1, 3, 64, 64)
        with self.assertRaisesRegex(SystemExit, "遮罩"):
            run.solve(None, "diffvax", x01, seed=0,
                      immunizer=_immunizer(), mask01=None)
        with self.assertRaisesRegex(SystemExit, "immunizer"):
            run.solve(None, "diffvax", x01, seed=0,
                      immunizer=None, mask01=_mask())


if __name__ == "__main__":
    unittest.main()
