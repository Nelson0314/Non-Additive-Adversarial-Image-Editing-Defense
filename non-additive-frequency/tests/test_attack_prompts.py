"""三類攻擊指令與臉部主體：驅動端的接線。

為什麼要換指令
────────────────────────────────────────────────────────────────────
資料集自帶的句子有不少指向畫面裡不重要的東西（坐墊、飛盤、披薩盒）。
新的三類是有意義的編輯：改變衣著或顏色、加上配件、變換背景與背景物品。

**兩類完全固定，只有 clothing 的衣物名詞逐張變。** 跨影像比較時不可以混進
「句子寫法不同」這個變因，故本檔釘住那件事：accessory 與 background 在所有
影像上必須逐字相同。
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ip2p_run  # noqa: E402

CAT = ROOT / "data" / "attack_prompts.yaml"


def _spec():
    return yaml.safe_load(CAT.read_text(encoding="utf-8"))


def test_目錄存在且三類齊全():
    spec = _spec()
    assert spec["images"], "沒有影像"
    for name, r in spec["images"].items():
        assert set(r["prompts"]) == {"clothing", "accessory", "background"}, name


def test_兩類在所有影像上逐字相同():
    """句子寫法是變因；只有 clothing 的衣物名詞被允許逐張變。"""
    spec = _spec()["images"]
    for cat in ("accessory", "background"):
        vals = {r["prompts"][cat] for r in spec.values()}
        assert len(vals) == 1, (cat, sorted(vals))


def test_clothing_的句子由衣物名詞與目標色組出來():
    for name, r in _spec()["images"].items():
        assert r["prompts"]["clothing"] == f"turn {r['garment']} {r['target_colour']}", name


def test_目標色是量出來的且寫成獨立欄位():
    """埋在句子裡的話，跨影像比較時「目標色不同」這個變因看不見。"""
    for name, r in _spec()["images"].items():
        assert r["target_colour"] in ("red", "blue"), name
        assert 0.0 <= r["carrier_hue"] <= 1.0, name


def test_目標色選的是離現有色相較遠的那一個():
    """這是選色的規則本身，寫錯會讓某些影像拿到與衣服同色的指令。"""
    targets = {"red": 0.0, "blue": 0.6111}

    def dist(a, b):
        d = abs(a - b) % 1.0
        return min(d, 1.0 - d)

    for name, r in _spec()["images"].items():
        h = r["carrier_hue"]
        best = max(targets, key=lambda c: dist(h, targets[c]))
        assert r["target_colour"] == best, (name, h, r["target_colour"], best)


def test_排除的影像有理由且不在清單裡():
    spec = _spec()
    for name, why in spec["dropped"].items():
        assert name not in spec["images"], name
        assert len(why) > 20, name          # 理由要寫得出來，不是一句「不好」


def test_驅動接受三類且未知類別拋錯():
    ap = ip2p_run.build_parser()
    for c in ("clothing", "accessory", "background"):
        ns = ap.parse_args(["--out", "x", "--attack-category", c])
        assert ns.attack_category == c
    with pytest.raises(SystemExit):
        ap.parse_args(["--out", "x", "--attack-category", "亂填"])


def test_指令覆寫需要目錄檔():
    """給了類別卻沒給目錄，指令會靜默沿用資料集自帶的句子。"""
    ns = ip2p_run.build_parser().parse_args(
        ["--out", "x", "--attack-category", "clothing"])
    with pytest.raises(SystemExit, match="attack-prompts"):
        ip2p_run.validate_attack_args(ns)


def test_目錄裡沒有該影像時拋錯而不是沿用舊句子():
    spec = {"images": {"a": {"prompts": {"clothing": "turn the shirt red"}}}}
    with pytest.raises(SystemExit, match="沒有"):
        ip2p_run.attack_instruction(spec, "不存在的影像", "clothing")
    assert ip2p_run.attack_instruction(spec, "a", "clothing") == "turn the shirt red"


def test_三個攻擊欄位以字面字串寫進列():
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in ("attack_category", "attack_prompts", "subject_source"):
        assert f'"{col}":' in src, col


def test_四個載體欄位以字面字串寫進列():
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in ("carrier_refine", "carrier_erode", "carrier_feather",
                "carrier_scatter"):
        assert f'"{col}":' in src, col


def test_全部新欄位登記進_SETTING_DEFAULTS():
    import distortion_axis_analysis as daa
    for col, val in (("attack_category", ""), ("attack_prompts", ""),
                     ("subject_source", "clipseg"), ("carrier_refine", "0"),
                     ("carrier_erode", "0"), ("carrier_feather", "0"),
                     ("carrier_scatter", "0")):
        assert daa.SETTING_DEFAULTS[col] == val, col


def test_face_來源不需要名詞目錄():
    """`--subject-source face` 的主體由 ATR 直接給。仍去查目錄的話，換一個
    目錄檔就會死在「objects 裡沒有這張影像」——而那個訊息與真正的原因無關。

    比對字面字串：這一段在 GPU 路徑上，測試不載權重。**錨點用 `PATCH_CONDS`
    而不是元組的字面量**——那個元組此前在四處各展開一次，已經集中成常數
    （`tests/test_deliver_jpeg.py::test_補丁條件的元組只有一份實作` 釘住只剩
    一份），錨在字面量上會讓這一條隨著那次集中而失效。
    """
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    i = src.index('if cond in PATCH_CONDS:')
    j = src.index("run_extras.update(mask_stats(m))", i)
    block = src[i:j]
    assert 'if args.subject_source == "face":' in block
    # face 分支要在「讀目錄檔」之前，否則仍會先死在目錄查找上
    assert (block.index('if args.subject_source == "face":')
            < block.index("safe_load"))
