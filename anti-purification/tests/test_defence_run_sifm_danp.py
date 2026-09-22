"""SIFM 與 DANP 接進 defence_run 的接線；不需要 GPU。

每個測試釘的是一個「錯了不會拋錯、只會讓那一列的數字被誤讀」的接法。
"""

import unittest
from unittest import mock

import torch

from scripts import defence_run as run
from src.baselines import danp, photoguard, sifm


class _FakeResult:
    def __init__(self, x01):
        self.x_adv01 = x01


class ConditionListTests(unittest.TestCase):
    def test_both_are_conditions_bound_to_their_paper_spec(self):
        """接錯 spec 不會拋錯，只會讓整列的預算欄寫成另一篇的數。"""
        for name, spec in (("sifm", sifm.SPEC_PAPER), ("danp", danp.SPEC_PAPER)):
            self.assertIn(name, run.CONDITIONS)
            self.assertIs(run.PGD_SPECS[name], spec)

    def test_both_read_the_dataset_content(self):
        """漏掉這一行的話 `--conditions sifm` 完全不讀 prompts.yaml，
        `item["content"]` 恆為空，求解端拿到空字串——仍然產得出一張防禦圖。"""
        for name in ("sifm", "danp"):
            self.assertIn(name, run.CONTENT_CONDITIONS)

    def test_the_copied_prompts_still_match_their_modules(self):
        run._check_prompts()

    def test_neither_has_a_constant_prompt(self):
        """兩者的文字條件逐類別不同，寫成常數就等於對每一類都條件在同一個詞上。"""
        for name in ("sifm", "danp"):
            self.assertNotIn(name, run.SOLVER_PROMPT)


class SolverPromptTests(unittest.TestCase):
    def test_both_take_the_class_content(self):
        """CSV 的 solver_prompt 欄是「防禦方看到了什麼」的唯一書面證據，
        抄成別的值不會影響求解，只會讓報表說錯威脅模型。"""
        for name in ("sifm", "danp"):
            value, source = run.solver_prompt_of(name, {"class": "man", "content": "man"})
            self.assertEqual(value, "man")
            self.assertIn("content", source)
            self.assertEqual(source, run.CONTENT_PROMPT_SOURCE)

    def test_absent_content_is_refused_rather_than_silently_empty(self):
        """空字串在兩篇都解得出一張圖：SIFM 會變成無條件去噪器的特徵、
        DANP 的 cross-attention 遮罩會落在空 prompt 上，兩者都沒有症狀。"""
        for name in ("sifm", "danp"):
            for item in ({"class": "man"}, {"class": "man", "content": ""}):
                with self.assertRaisesRegex(SystemExit, "content"):
                    run.solver_prompt_of(name, item)


class SolveKwargTests(unittest.TestCase):
    """`solve` 實際傳下去的 kwargs。`run_pgd` 用 mock 換掉，不載模型。"""

    def _kwargs_for(self, cond, content):
        x01 = torch.zeros(1, 3, 8, 8)
        with mock.patch.object(run, "run_pgd",
                               return_value=_FakeResult(x01)) as fake:
            x_def, cfg = run.solve(None, cond, x01, seed=0, content=content)
        self.assertEqual(fake.call_count, 1)
        return fake.call_args, cfg

    def test_sifm_gets_the_content_as_its_prompt(self):
        """`sifm.prepare` 的參數名是 `prompt`。傳成 `content` 會被 `**_`
        吞掉，`prompt` 留在 None，那時才會拋 ValueError；但若哪天補了預設值，
        就變成靜默用另一個條件求解。"""
        call, _ = self._kwargs_for("sifm", "man")
        self.assertEqual(call.kwargs["prompt"], "man")

    def test_sifm_is_not_given_a_checkpoint_switch(self):
        """SIFM 用 forward hook 取中間特徵，checkpoint 區塊的前向在 no_grad 下
        執行，hook 取到的張量不在計算圖上——梯度靜默歸零，輸出仍是一張合理的圖
        （見 sifm.py「為什麼不提供 use_ckpt」）。"""
        call, _ = self._kwargs_for("sifm", "man")
        self.assertNotIn("use_ckpt", call.kwargs)
        self.assertNotIn("vae_ckpt", call.kwargs)

    def test_danp_gets_the_content_as_its_prompt_with_checkpointing(self):
        """`danp.prepare` 的參數名同樣是 `prompt`；`use_ckpt` 要開，
        一次 loss_fn 有兩次 UNet 前向，不開會 OOM——那是會拋錯的失效，
        但參數名傳錯則不會。"""
        call, _ = self._kwargs_for("danp", "woman")
        self.assertEqual(call.kwargs["prompt"], "woman")
        self.assertIs(call.kwargs["use_ckpt"], True)

    def test_neither_branch_rescales_the_image_before_run_pgd(self):
        """`run_pgd` 自己用 `spec.value_range.from01` 進值域、`to01` 出來，
        呼叫端一律傳 [0,1]。若在分支裡多做一次換算，預算會差一倍而不拋錯。"""
        for cond in ("sifm", "danp"):
            x01 = torch.rand(1, 3, 8, 8)
            with mock.patch.object(run, "run_pgd",
                                   return_value=_FakeResult(x01)) as fake:
                run.solve(None, cond, x01, seed=0, content="man")
            self.assertIs(fake.call_args.args[1], x01)

    def test_the_budget_columns_come_from_the_spec(self):
        """eps 欄寫死字面值的話，換 spec 時這一欄不會跟著動。"""
        for cond, spec in (("sifm", sifm.SPEC_PAPER), ("danp", danp.SPEC_PAPER)):
            _, cfg = self._kwargs_for(cond, "man")
            self.assertEqual(cfg["eps"], spec.eps)
            self.assertEqual(cfg["eps_pixel01"], spec.eps_pixel01)
            self.assertEqual(cfg["steps"], spec.steps)
            self.assertEqual(cfg["grad_reps"], spec.grad_reps)
            self.assertEqual(cfg["spec_source"], spec.source)


class ValueRangeTests(unittest.TestCase):
    def test_both_optimise_in_zero_to_one_unlike_the_older_five(self):
        """既有五篇的值域是 [-1,1]，這兩篇是 [0,1]。同一個 eps=0.03 在兩個值域
        代表的像素幅度差一倍，混淆了不會拋錯，只會讓預算欄被讀成同一件事。"""
        for spec in (sifm.SPEC_PAPER, danp.SPEC_PAPER):
            self.assertEqual((spec.value_range.lo, spec.value_range.hi), (0.0, 1.0))
        self.assertEqual((photoguard.SPEC.value_range.lo,
                          photoguard.SPEC.value_range.hi), (-1.0, 1.0))

    def test_eps_needs_no_conversion_at_the_call_site(self):
        """值域寬度是 1，故 `eps_pixel01 == eps`；`BaselineSpec.__post_init__`
        已經釘住 `eps_pixel01 == eps / scale`，呼叫端不必、也不該再換算一次。"""
        for spec in (sifm.SPEC_PAPER, danp.SPEC_PAPER):
            self.assertEqual(spec.value_range.scale, 1.0)
            self.assertEqual(spec.eps_pixel01, spec.eps)

    def test_run_pgd_does_the_conversion_itself(self):
        """若 `run_pgd` 改成要求呼叫端先進值域，這個往返就不再是恆等，
        而 [0,1] 的兩篇正好看不出差別——用 [-1,1] 的 spec 一起釘。"""
        for spec in (sifm.SPEC_PAPER, danp.SPEC_PAPER, photoguard.SPEC):
            x01 = torch.rand(1, 3, 4, 4)
            vr = spec.value_range
            self.assertTrue(torch.allclose(vr.to01(vr.from01(x01)), x01, atol=1e-6))
            self.assertGreaterEqual(float(vr.from01(x01).min()), vr.lo - 1e-6)
            self.assertLessEqual(float(vr.from01(x01).max()), vr.hi + 1e-6)


if __name__ == "__main__":
    unittest.main()
