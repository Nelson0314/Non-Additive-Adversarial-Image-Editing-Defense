"""防禦圖當輸入、以及本管線採用的 image guidance；都不需要 GPU。"""

from pathlib import Path
import tempfile
import unittest

from scripts import edit_preflight as preflight
from src.models import ip2p


def items(directory, names=("man_00", "woman_00")):
    return [{"name": name, "class": "man", "path": directory / f"{name}.png",
             "content": "a man", "mask": directory / "masks" / f"{name}.png"}
            for name in names]


class DefendedInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def touch(self, *filenames):
        for filename in filenames:
            (self.directory / filename).write_bytes(b"")

    def test_the_condition_name_sits_in_the_middle_of_the_filename(self):
        """`defence_run.py` 寫的是 `<名稱>__<條件>__def.png`。"""
        self.touch("man_00__mist__def.png", "woman_00__mist__def.png")
        rows = items(self.directory)
        preflight.apply_defended(rows, self.directory)
        self.assertEqual([row["path"].name for row in rows],
                         ["man_00__mist__def.png", "woman_00__mist__def.png"])

    def test_a_filename_without_the_condition_is_also_accepted(self):
        self.touch("man_00__def.png")
        self.assertEqual(
            preflight.defended_image(self.directory, "man_00").name,
            "man_00__def.png")

    def test_two_candidates_for_one_image_are_refused(self):
        """目錄裡混了兩個條件時，拿哪一張都是錯的。"""
        self.touch("man_00__mist__def.png", "man_00__dia_r__def.png")
        with self.assertRaisesRegex(SystemExit, "對到 2 個"):
            preflight.defended_image(self.directory, "man_00")

    def test_missing_defence_images_are_named_rather_than_skipped(self):
        self.touch("man_00__mist__def.png")
        with self.assertRaisesRegex(SystemExit, "woman_00"):
            preflight.apply_defended(items(self.directory), self.directory)

    def test_nothing_but_the_input_moves(self):
        self.touch("man_00__mist__def.png", "woman_00__mist__def.png")
        rows = items(self.directory)
        before = [(row["mask"], row["content"], row["class"]) for row in rows]
        preflight.apply_defended(rows, self.directory)
        self.assertEqual([(r["mask"], r["content"], r["class"]) for r in rows], before)

    def test_the_whole_set_is_refused_when_one_image_is_absent(self):
        """七個條件要共用同一個分母，少一張就不是同一條管線。"""
        self.touch("man_00__mist__def.png")
        rows = items(self.directory)
        with self.assertRaises(SystemExit):
            preflight.apply_defended(rows, self.directory)
        self.assertEqual(rows[0]["path"].name, "man_00.png")


class AdoptedGuidanceTests(unittest.TestCase):
    def test_the_edit_pipeline_pins_its_own_image_guidance(self):
        """1.8 是掃描之後這條管線自己的值，不跟著 IP2P 封裝的預設走。"""
        self.assertEqual(preflight.IP2P_EDIT_IMAGE_GUIDANCE, 1.8)
        self.assertNotEqual(preflight.IP2P_EDIT_IMAGE_GUIDANCE,
                            ip2p.IP2P_IMAGE_GUIDANCE)

    def test_text_guidance_still_comes_from_the_wrapper(self):
        self.assertEqual(preflight.IP2P_TEXT_GUIDANCE, ip2p.IP2P_TEXT_GUIDANCE)


if __name__ == "__main__":
    unittest.main()
