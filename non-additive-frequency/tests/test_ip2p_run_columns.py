"""`scripts/ip2p_run.py` 寫進 CSV 的參數欄位。

存在理由：CLAUDE.md 的規則是「每個未載的參數都要成為 CSV 的**欄位**，不是
註解」。`quantile`／`hop`／`gate_edge_power`／`defense_steps` 四者此前只活在
CLI 的預設值裡，於是掃描它們的批次在合併之後與基準列長得一模一樣——這與
`r_min` 當初漏記是同一型的缺陷，而且沒有症狀。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ip2p_run  # noqa: E402


REQUIRED = ("quantile", "hop", "gate_edge_power", "defense_steps")


def _args(**over):
    ap = ip2p_run.build_parser()
    ns = ap.parse_args(["--out", "x"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_required_columns_are_written_literally():
    """欄位名必須以字面字串出現在組列的地方。

    比對字面字串而非執行整條管線：這裡不載入 IP2P（需要 GPU 與權重），而
    漏掉欄位的失效方式正是「那一行不存在」。
    """
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in REQUIRED:
        assert f'"{col}":' in src, col


@pytest.mark.parametrize("cond,expect_attr", [
    ("phase", "steps"),
    ("phase_gain", "steps"),
    ("dct_shield", "dct_steps"),
    ("dct_shield_y", "dct_steps"),
    ("advdrop", "advdrop_steps"),
    ("dct_wm", "wm_tau"),
])
def test_defense_steps_follows_the_flag_each_condition_actually_uses(
        cond, expect_attr):
    """每個條件走的是不同的步數旗標，欄位必須取到它自己那一個。

    本方法預設 100 步、DCT-Shield 是該篇 §5.4 的 1000 步——頭對頭表把兩者
    並排，而預算差十倍這件事此前不在任何欄位裡。
    """
    args = _args()
    assert ip2p_run.defense_steps(args, cond) == int(getattr(args, expect_attr))


def test_phase_and_dct_defaults_really_do_differ_tenfold():
    """釘住那個十倍差本身。它若哪天被對齊了，這條測試會提醒去改報表的敘述。"""
    args = _args()
    assert ip2p_run.defense_steps(args, "dct_shield") == 1000
    assert ip2p_run.defense_steps(args, "phase") == 100


def test_gate_edge_power_default_reproduces_current_behaviour():
    args = _args()
    assert args.gate_edge_power == 1.0
    assert args.quantile == 0.5


def test_every_driver_script_imports():
    """驅動腳本必須 import 得進來。

    存在理由：`scripts/ip2p_run.py` 曾在 HEAD 上完全 import 不進來——它引用的
    `WatermarkSpec` 與 `run_dct_watermark` 早已改名為 `DJSMASpec` 與
    `run_djsma`，而**沒有任何測試 import 這支驅動**，於是 456 條測試全過、
    主線驅動卻是死的。這一條把那個盲區補上。

    只 import 不執行：這些腳本的 `main()` 要載 Stable Diffusion 權重。
    """
    import importlib.util

    skip = {"__init__"}
    broken = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if path.stem in skip:
            continue
        spec = importlib.util.spec_from_file_location(f"drv_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass
        except ImportError as exc:
            broken.append(f"{path.name}: {exc}")
    assert not broken, broken
