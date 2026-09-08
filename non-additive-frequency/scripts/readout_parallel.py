"""同一批資料的三個讀數並列：位移、SigLIP、身分。

存在理由
────────────────────────────────────────────────────────────────────
`docs/GOAL.md` 寫的是：

> **主讀數是位移，錨點是等失真。** … 語意指標（CLIP／SigLIP）可作為並列的
> 主讀數。 … **兩個讀數都要報。** 只報其一會讓「用強度換抗性」或反之的
> 取捨看不出來。

而 `docs/DIRECTION.md` §6.0 判定色彩載體「已被否定」時，用的是**身分讀數
一個**。身分是後來加進來的（理由充分：位移把「劣化」與「重畫」算成同一件
事），但它是**人臉專屬**的，而本專案的目標並不限定在人臉——編輯輸出「不再
認得出是原來那張圖」可以由很多方式達成。

**一個讀數說零，不等於三個讀數都說零。** 這一支不做任何裁定，只把三個讀數
擺在同一張表上，讓「否定」這個判斷有完整的分母。

三個讀數各自量什麼
────────────────────────────────────────────────────────────────────
| 欄 | 量什麼 | 已知的偏差 |
|---|---|---|
| `fid_dists` | `DISTS(原圖, 防禦圖)`——**這個防禦付了多少失真**。不可用 `edit_dists`，那是兩張編輯輸出之間的距離，是效果不是代價 |
| `edit_lpips` | `LPIPS(編輯(原圖), 編輯(防禦圖))` | 把「劣化」與「重畫」算成同一件事（`DEFECTS.md`），在抗淨化上尤其樂觀 |
| `edit_siglip_sim` | 兩張編輯輸出在 SigLIP 影像空間的餘弦 | 語意層級，對純劣化較不敏感；門檻由 `SIGLIP_BLOCKED_THRESHOLD` 給 |
| `id_drop` | `id_orig − id_def`，偵測不到臉記 `id_def = 0` | 只在畫面裡有臉時有意義；多人合照不可靠 |

`edit_lpips` 與 `edit_siglip_sim` 逐列寫在 `results.csv` 裡，`id_drop` 在各批
的 `identity.csv` 裡，用 `label` 欄接起來。

**這一支不扣空白地板、不做等失真內插。** 那兩件事各有既有的工具
（`retention_table.py`、`matched_distortion_table.py`），混進來只會讓這張表
同時回答三個問題。這裡只回答一個：**同一組防禦圖，三個讀數各給什麼數字。**
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def arm_of(dirname: str, categories: List[str]) -> Optional[str]:
    """目錄名 → 臂名。**兩種順序都要認得**。

    `ip2p_content_constraint` 用 `<臂>_<類別>_<影像>`，
    `ip2p_face_defence` 用 `<類別>_<臂>_<影像>`。只認前者的話 `free`
    （在 `ip2p_face_defence` 裡叫 `clothing_plain_*`）會整批被跳過，
    而表上只是少一列、不會報錯——那正是這張表最需要的那一列。
    """
    for cat in categories:
        if dirname.startswith(f"{cat}_"):
            rest = dirname[len(cat) + 1:]
            return rest.split("_task", 1)[0] if "_task" in rest else rest
        key = f"_{cat}_"
        if key in dirname:
            return dirname.split(key, 1)[0]
    return None


def load_identity(run: Path) -> Dict[str, dict]:
    """該批全部的身分讀數 → `{label: 列}`。

    兩種擺法都要收：有的批次是單一個 `identity.csv`，有的是逐臂一個
    `identity_<臂>.csv`（`ip2p_content_constraint`）。只讀前者的話那一批的
    `id_drop` 欄會整欄空白，而表看起來完全正常。
    """
    out: Dict[str, dict] = {}
    for path in sorted(run.glob("identity*.csv")):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "label" in r:
                    out[r["label"]] = r
    return out


def fnum(row: dict, key: str) -> Optional[float]:
    v = (row.get(key) or "").strip()
    if v == "":
        return None
    return float(v)


def med(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def collect(run: Path, categories: List[str]) -> Dict[str, dict]:
    """一個 runs 目錄 → `{臂: 讀數}`。"""
    ident = load_identity(run)
    arms: Dict[str, dict] = {}
    for d in sorted(run.iterdir()):
        res = d / "results.csv"
        if not d.is_dir() or not res.exists():
            continue
        arm = arm_of(d.name, categories)
        if arm is None:
            continue
        a = arms.setdefault(arm, {"lpips": [], "siglip": [], "blocked": 0,
                                  "n": 0, "id_drop": [], "id_readable": 0,
                                  "dists": []})
        with open(res, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a["n"] += 1
                for key, col in (("lpips", "edit_lpips"),
                                 ("siglip", "edit_siglip_sim"),
                                 ("dists", "fid_dists")):
                    v = fnum(row, col)
                    if v is not None:
                        a[key].append(v)
                if (row.get("blocked") or "").strip().lower() == "true":
                    a["blocked"] += 1
        ir = ident.get(d.name)
        if ir is not None:
            # 可解讀列的定義與 `identity_probe` 一致：攻擊自己就已經把身分
            # 打掉的格子沒有分母可談。門檻取 `identity.SAME_PERSON_REFERENCE`。
            io = fnum(ir, "id_orig")
            dd = fnum(ir, "id_drop")
            if io is not None and dd is not None and io >= 0.55:
                a["id_readable"] += 1
                a["id_drop"].append(dd)
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="runs 目錄（相對於 repo 根）")
    ap.add_argument("--categories", nargs="+",
                    default=["clothing", "accessory", "background"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows: List[dict] = []
    for name in args.runs:
        run = ROOT / name
        if not run.is_dir():
            raise FileNotFoundError(f"找不到 {run}")
        for arm, a in sorted(collect(run, args.categories).items()):
            rows.append({
                "run": name.split("/")[-1], "arm": arm, "n": a["n"],
                "dists": med(a["dists"]),
                "edit_lpips": med(a["lpips"]),
                "siglip_sim": med(a["siglip"]),
                "blocked": f"{a['blocked']}/{a['n']}",
                "id_readable": a["id_readable"],
                "id_drop": med(a["id_drop"]),
            })
    if not rows:
        raise RuntimeError("沒有讀到任何一個臂——確認目錄名的形狀是 "
                           "`<臂>_<類別>_<影像>`，或用 --categories 指定類別")

    def fmt(v, w=8, p=4):
        return " " * w if v is None else f"{v:>{w}.{p}f}"

    print(f"{'run':<26}{'arm':<12}{'n':>4}{'DISTS':>9}{'位移':>9}"
          f"{'SigLIP':>9}{'blocked':>9}{'id可解讀':>9}{'id降幅':>9}")
    for r in rows:
        print(f"{r['run']:<26}{r['arm']:<12}{r['n']:>4}"
              f"{fmt(r['dists'],9)}{fmt(r['edit_lpips'],9)}"
              f"{fmt(r['siglip_sim'],9)}{r['blocked']:>9}"
              f"{r['id_readable']:>9}{fmt(r['id_drop'],9)}")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n寫出 {len(rows)} 列到 {out}")


if __name__ == "__main__":
    main()
