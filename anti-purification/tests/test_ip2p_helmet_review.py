"""判讀合併與接觸印樣的排版測試；不產生任何影像證據。"""

import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts import ip2p_helmet_contactsheet as sheets
from scripts import ip2p_helmet_review as review


SWEEP_FIELDS = ("cell", "image", "s_t", "s_i", "output_png", "negative_prompt",
                "helmet_appeared", "extra_person", "face_swapped", "usable",
                "review_status", "review_notes")


def write(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def sweep_rows(directory, cells=("a",), images=("man_00", "woman_00")):
    return [dict.fromkeys(SWEEP_FIELDS, "") | dict(
        cell=cell, image=image, s_t="7.5", s_i="1.8",
        output_png=str(directory / f"{image}__p2.png"), review_status="unreviewed")
        for cell in cells for image in images]


def review_rows(cells=("a",), images=("man_00", "woman_00"), **overrides):
    return [dict(cell=cell, image=image, helmet_appeared="1", extra_person="0",
                 face_swapped="0", usable="1", review_notes="") | overrides
            for cell in cells for image in images]


class ReviewMergeTests(unittest.TestCase):
    def merge(self, review_data, sweep_data=None):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        sweep_path, review_path = directory / "sweep.csv", directory / "review.csv"
        write(sweep_path, SWEEP_FIELDS, sweep_data or sweep_rows(directory))
        write(review_path, list(review_data[0]), review_data)
        merged = review.check(review.read_csv(review_path), review.read_csv(sweep_path))
        return merged, sweep_path

    def test_every_scanned_cell_needs_a_judgement(self):
        with self.assertRaisesRegex(SystemExit, "對不起來"):
            self.merge(review_rows(images=("man_00",)))

    def test_judgement_for_a_cell_that_was_never_run_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "對不起來"):
            self.merge(review_rows(images=("man_00", "woman_00", "man_01")))

    def test_values_other_than_zero_and_one_are_refused(self):
        with self.assertRaisesRegex(SystemExit, "只能填 0 或 1"):
            self.merge(review_rows(usable="yes"))

    def test_blank_face_swapped_needs_a_reason(self):
        """臉被帽體遮住時留空是允許的，但必須說明；留空又不說明是漏填。"""
        with self.assertRaisesRegex(SystemExit, "只能填 0 或 1"):
            self.merge(review_rows(face_swapped=""))
        merged, _ = self.merge(review_rows(face_swapped="", review_notes="臉被遮住"))
        self.assertEqual(merged[("a", "man_00")]["face_swapped"], "")

    def test_blank_is_only_allowed_for_face_swapped(self):
        with self.assertRaisesRegex(SystemExit, "只能填 0 或 1"):
            self.merge(review_rows(helmet_appeared="", review_notes="看不出來"))

    def test_duplicate_judgement_rows_are_refused(self):
        rows = review_rows() + review_rows(images=("man_00",))
        with self.assertRaisesRegex(SystemExit, "重複"):
            self.merge(rows)


class ContactSheetTests(unittest.TestCase):
    def test_missing_output_is_named_rather_than_skipped(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        rows = [dict(cell="a", image=name, output_png=str(directory / f"{name}.png"),
                     s_t="7.5", s_i="1.8", negative_prompt="")
                for name in sheets.PORTRAITS]
        with self.assertRaisesRegex(SystemExit, "不存在"):
            sheets.cell_images(rows, "a", directory)
        rows.pop()
        with self.assertRaisesRegex(SystemExit, "woman_03"):
            sheets.cell_images(rows, "a", directory)

    def test_grid_keeps_every_tile_and_stays_four_wide(self):
        tiles = [Image.new("RGB", (8, 6)) for _ in range(8)]
        sheet = sheets.grid(tiles)
        self.assertEqual(sheet.width, 4 * 8 + 5 * sheets.MARGIN)
        self.assertEqual(sheet.height, 2 * 6 + 3 * sheets.MARGIN)

    def test_label_is_scaled_so_it_stays_readable(self):
        strip = sheets.label_strip("man_00", 300)
        self.assertEqual(strip.width, 300)
        self.assertEqual(strip.height, 14 * sheets.LABEL_SCALE)


if __name__ == "__main__":
    unittest.main()
