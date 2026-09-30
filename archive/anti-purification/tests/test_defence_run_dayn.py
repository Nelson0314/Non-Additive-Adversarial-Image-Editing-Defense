"""DAYN 接進 defence_run 的接線；不需要 GPU。"""

from pathlib import Path
import tempfile
import unittest

import yaml

from scripts import defence_run as run
from src.baselines import dayn


PROMPTS = {
    "man": {"content": "a man"},
    "woman": {"content": "a woman"},
    "edits": {"ip2p": ["Let the person wear a helmet"]},
}


class ConditionListTests(unittest.TestCase):
    def test_dayn_is_one_of_the_conditions(self):
        self.assertIn("dayn", run.CONDITIONS)
        self.assertIs(run.PGD_SPECS["dayn"], dayn.SPEC_PAPER)

    def test_the_copied_prompts_still_match_their_modules(self):
        run._check_prompts()

    def test_dayn_has_no_constant_prompt(self):
        """c_a 逐類別不同，寫成常數就等於對每一類都保護同一個東西。"""
        self.assertNotIn("dayn", run.SOLVER_PROMPT)


class SolverPromptTests(unittest.TestCase):
    def test_dayn_takes_the_class_content(self):
        value, source = run.solver_prompt_of("dayn", {"class": "man", "content": "a man"})
        self.assertEqual(value, "a man")
        self.assertIn("content", source)

    def test_other_conditions_keep_their_module_constant(self):
        self.assertEqual(run.solver_prompt_of("mist", {"class": "man"})[0],
                         run.SOLVER_PROMPT["mist"][0])

    def test_absent_content_is_refused_rather_than_silently_empty(self):
        """空的 c_a 仍然解得出一張圖，但那張圖保護的不是這裡面的東西。"""
        for item in ({"class": "man"}, {"class": "man", "content": ""}):
            with self.assertRaisesRegex(SystemExit, "content"):
                run.solver_prompt_of("dayn", item)


class ContentLoadingTests(unittest.TestCase):
    def test_only_the_content_key_is_read(self):
        """攻擊端的 `edits` 不進求解端，兩者在威脅模型裡屬於不同的人。"""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (directory / "prompts.yaml").write_text(
            yaml.safe_dump(PROMPTS, allow_unicode=True), encoding="utf-8")
        self.assertEqual(run.content_by_class(directory),
                         {"man": "a man", "woman": "a woman"})


if __name__ == "__main__":
    unittest.main()
