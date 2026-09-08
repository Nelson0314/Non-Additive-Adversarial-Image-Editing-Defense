"""`--images` 指名而資料集裡沒有的影像，必須拋錯而不是靜默丟掉。

存在理由
────────────────────────────────────────────────────────────────────
選圖走的是集合交集：

    dataset = [d for d in dataset if d["name"] in keep]

指名 150 張、資料集裡只有 147 張時，**少的那三張被靜默丟掉**，唯一的守門是
「一張都不剩才拋錯」。印出來的那行寫著實際張數，沒有人會回頭跟要求的張數
對照，於是報表上的「150 張」與實際跑的 147 張差別不會有任何症狀。

實際踩到：`runs/ip2p_fair_comparison/images{13,25,75,150}.txt` 四份清單都
指名 `task_env_weather_246440`、`task_env_weather_63722`、`task_obj_add_40931`，
而 `data/omniedit150/` 底下只有 147 個目錄（`provenance.json` 列 150 筆）。
FID 的「低於 150 張不可信」那條門檻因此從來沒有真的達到過。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ip2p_run  # noqa: E402


def test_缺影像時拋錯並指名是哪幾張():
    dataset = [{"name": "a"}, {"name": "b"}]
    with pytest.raises(SystemExit) as e:
        ip2p_run.select_images(dataset, ["a", "b", "c", "d"], Path("data/x"))
    msg = str(e.value)
    assert "c" in msg and "d" in msg


def test_全部都在時逐位元回傳同一份且保持順序():
    dataset = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    got = ip2p_run.select_images(dataset, ["c", "a"], Path("data/x"))
    assert [d["name"] for d in got] == ["a", "c"]


def test_沒給清單時原樣回傳():
    dataset = [{"name": "a"}]
    assert ip2p_run.select_images(dataset, None, Path("data/x")) is dataset


def test_資料集為空時仍然拋錯():
    with pytest.raises(SystemExit):
        ip2p_run.select_images([], None, Path("data/x"))


# `provenance.json` 列 150 筆，`data/omniedit150/` 底下只有 147 個目錄。
# 這三張**目前取不回來**：provenance 裡的簽名 URL 帶 `Expires=1787157429`
# （2026-08-17）已過期，而 `scripts/fetch_omniedit.py` 走的 HF
# datasets-server `/rows` 端點回 502。
KNOWN_MISSING = frozenset({
    "task_env_weather_246440",
    "task_env_weather_63722",
    "task_obj_add_40931",
})


def test_磁碟與_provenance_的差集就是已知的那三張():
    """**釘住已知狀態，兩個方向都要偵測**。

    再少一張要有人知道；三張被補回來也要有人知道（那時把 `KNOWN_MISSING`
    清空，並把 FID 的「低於 150 張不可信」重新算過）。
    """
    import json
    import os
    data = ROOT / "data" / "omniedit150"
    if not data.is_dir():
        pytest.skip("資料集不在這棵樹裡")
    prov = json.loads((data / "provenance.json").read_text(encoding="utf-8"))
    want = {os.path.splitext(os.path.basename(r["output"]))[0] for r in prov}
    have = {p.name for p in data.iterdir() if p.is_dir()}
    assert want - have == KNOWN_MISSING
    assert not have - want


def test_影像清單只缺已知的那三張():
    """清單指名的每一張，除了已知缺的三張之外都必須存在。"""
    data = ROOT / "data" / "omniedit150"
    if not data.is_dir():
        pytest.skip("資料集不在這棵樹裡")
    have = {p.name for p in data.iterdir() if p.is_dir()}
    unexpected = {}
    for lst in sorted((ROOT / "runs" / "ip2p_fair_comparison").glob("images*.txt")):
        names = [l.strip() for l in lst.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        bad = [n for n in names if n not in have and n not in KNOWN_MISSING]
        if bad:
            unexpected[lst.name] = bad
    assert not unexpected, f"清單指名了磁碟上沒有、且不在已知缺漏裡的影像：{unexpected}"


def test_用整份清單跑會被新的守門擋下():
    """`images150.txt` 含已知缺的三張，故拿它去跑必須拋錯而不是安靜跑 147 張。

    這條把「守門有沒有真的接上主線」釘住——守門寫了但沒接進 `main` 是同一型
    的靜默失效。
    """
    lst = ROOT / "runs" / "ip2p_fair_comparison" / "images150.txt"
    if not lst.is_file():
        pytest.skip("清單不在這棵樹裡")
    names = [l.strip() for l in lst.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    data = ROOT / "data" / "omniedit150"
    dataset = [{"name": p.name} for p in data.iterdir() if p.is_dir()]
    with pytest.raises(SystemExit, match="沒有的影像"):
        ip2p_run.select_images(dataset, names, data)
