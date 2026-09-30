"""CPU-only wiring/CSV regression tests; these are not generation evidence."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch
import yaml

from scripts import edit_preflight as preflight
from scripts import ip2p_helmet_sweep as sweep


class SweepPlanTests(unittest.TestCase):
    def test_dispatch_race_stops_only_the_owned_child(self):
        process = Mock(pid=123, returncode=None)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (patch.object(sweep, "gpu_state", side_effect=[(0, set()), (200, {999})]),
                  patch.object(sweep.subprocess, "Popen", return_value=process),
                  patch.object(sweep.subprocess, "run") as ps):
                with self.assertRaisesRegex(RuntimeError, "GPU dispatch race"):
                    sweep.run_cell(["python", "job.py"], 0, {}, root / "log", root)
                ps.assert_called_once_with(["ps", "-p", "123", "-o", "pid,ppid,args"], check=True)
                process.terminate.assert_called_once_with()
                process.wait.assert_called_once_with()

    def test_occupied_gpu_rejected_before_dispatch(self):
        with (patch.object(sweep, "gpu_state", return_value=(800, {999})),
              patch.object(sweep.subprocess, "Popen") as dispatch):
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                sweep.run_cell(["python", "job.py"], 0, {}, Path("unused.log"), Path.cwd())
            dispatch.assert_not_called()

    def test_unique_arms_and_single_variable_comparisons(self):
        cells = sweep.cells("test")
        self.assertEqual(len(cells), 11)
        self.assertEqual(len({c["arm"] for c in cells}), 11)
        self.assertEqual({(c["s_t"], c["s_i"]) for c in cells},
                         {(t, i) for t in (5.0, 6.0, 7.5) for i in (1.5, 1.8, 2.2)})
        by_id = {c["cell"]: c for c in cells}
        for cell in cells[1:]:
            parent = by_id[cell["compare_to"]]
            changed = [field for field in ("s_t", "s_i", "instruction", "negative_prompt",
                                           "seed", "steps") if parent[field] != cell[field]]
            self.assertEqual(changed, [cell["changed_variable"]])

    def test_csv_growth_detects_overwrite_duplicate_and_parameter_mismatch(self):
        cell = sweep.cells("test")[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.png"
            Image.new("RGB", (8, 8)).save(output)
            rows = [dict(cell, image=name, prompt_index="2", output_png=str(output))
                    for name in sweep.PORTRAITS]
            self.assertEqual(len(sweep.verify_growth([], rows, cell, root)), 8)
            with self.assertRaisesRegex(RuntimeError, "eight distinct"):
                sweep.verify_growth(rows, rows, cell, root)
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                sweep.verify_growth([], rows + [rows[0]], cell, root)
            bad = [dict(row) for row in rows]
            bad[0]["negative_prompt"] = "unrecorded change"
            with self.assertRaisesRegex(RuntimeError, "negative_prompt"):
                sweep.verify_growth([], bad, cell, root)
            previous = dict(rows[0], arm="ip2p_previous")
            altered = dict(previous, seed=123)
            with self.assertRaisesRegex(RuntimeError, "earlier CSV row"):
                sweep.verify_growth([previous], [altered, *rows], cell, root)


class PreflightWiringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        (self.data / "man").mkdir(parents=True)
        Image.new("RGB", (16, 16), (70, 80, 90)).save(self.data / "man" / "man_00.png")
        (self.data / "prompts.yaml").write_text(yaml.safe_dump({
            "man": {"content": "man"},
            "edits": {"ip2p": ["sunglasses", "police suit", sweep.ORIGINAL, "bowtie"]},
        }), encoding="utf-8")
        self.out = self.root / "out"
        self.model = Mock(device=torch.device("cpu"))
        self.model.edit.side_effect = lambda image, instruction, **kw: image.clone()
        suite = Mock()
        suite.pairwise.return_value = {"lpips": 0.0, "psnr": 99.0}
        suite.semantic.return_value = {"clip": 0.0}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.factory = self.stack.enter_context(patch.object(preflight, "IP2PWrapper",
                                                          return_value=self.model))
        self.stack.enter_context(patch.object(preflight, "MetricSuite", return_value=suite))
        self.stack.enter_context(patch.object(preflight, "RESOLUTION", 16))
        self.stack.enter_context(patch.object(torch.cuda, "empty_cache"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def run_preflight(self, *extra):
        argv = ["edit_preflight.py", "--data", str(self.data), "--out", str(self.out),
                "--scenarios", "ip2p", *extra]
        with patch.object(sys, "argv", argv):
            preflight.main()

    def test_negative_instruction_seed_and_original_prompt_index_reach_edit_and_csv(self):
        self.run_preflight("--prompt-indices", "2", "--ip2p-instruction", sweep.SPECIFIC,
                           "--negative-prompt", sweep.NEGATIVE, "--seed", "17",
                           "--s-t", "6.0", "--s-i", "1.8", "--suffix", "_first",
                           "--require-new-arm")
        args, kwargs = self.model.edit.call_args
        self.assertEqual(args[1], sweep.SPECIFIC)
        self.assertEqual(kwargs, dict(seed=17, steps=50, s_t=6.0, s_i=1.8,
                                     negative_prompt=sweep.NEGATIVE))
        rows = sweep.read_csv(self.out / "preflight.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt_index"], "2")
        self.assertEqual(rows[0]["instruction"], sweep.SPECIFIC)
        self.assertEqual(rows[0]["negative_prompt"], sweep.NEGATIVE)
        self.assertEqual(rows[0]["seed"], "17")
        self.assertTrue(Path(rows[0]["output_png"]).is_file())
        self.run_preflight("--prompt-indices", "2", "--suffix", "_second", "--require-new-arm")
        after = sweep.read_csv(self.out / "preflight.csv")
        self.assertEqual(len(after), 2)
        self.assertEqual(after[0], rows[0])

    def test_reused_arm_rejected_without_changing_evidence(self):
        self.run_preflight("--prompt-indices", "2", "--suffix", "_duplicate", "--require-new-arm")
        path = self.out / "preflight.csv"
        previous = path.read_bytes()
        self.factory.reset_mock()
        with self.assertRaises(SystemExit):
            self.run_preflight("--suffix", "_duplicate", "--require-new-arm")
        self.factory.assert_not_called()
        self.assertEqual(path.read_bytes(), previous)
        # CSV-only collision must also fail, even when its image folder is absent.
        orphan = sweep.read_csv(path)
        orphan[0]["arm"] = "ip2p_orphan"
        sweep.write_csv(path, orphan)
        with self.assertRaises(SystemExit):
            self.run_preflight("--suffix", "_orphan", "--require-new-arm")
        self.factory.assert_not_called()

    def test_ip2p_overrides_cannot_change_inpaint(self):
        with self.assertRaises(SystemExit):
            self.run_preflight("--scenarios", "inpaint", "--negative-prompt", sweep.NEGATIVE)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
