"""防禦圖看起來自然嗎。**只讀已存的影像，不重跑任何 GPU 訓練。**

兩組讀數，其中一組已知不可用
────────────────────────────────────────────────────────────────────
**主讀數是 `src/metrics/naturalness.py`** 的譜斜率／區塊熵／色數：它們只在
支撐內量，問的是「放上去的那塊東西像不像雜訊」，而那正是本專案在意的事。

NIMA-AVA 與 CNNIQA 照跑照報，但校準顯示它們在這個判別上**不可用**：把滿版
彩色雜訊與分散圓點餵進去，CNNIQA 只有 4/10 判對（比擲硬幣差），而且兩組的
drop 都是負的——它認為加了雜訊的圖「品質比原圖好」；NIMA 是 7/10 但兩組
分佈重疊得很厲害。留著是為了讓那個失敗有紀錄，**不是**因為可以拿來下判斷。

為什麼需要它
────────────────────────────────────────────────────────────────────
美化旋鈕的代價此前都用位移或身分讀數量，那量的是**效果掉了多少**，
量不到**外觀好了多少**。兩邊各自都有數字，中間那一格是空的：
「同面積下有沒有美化，看起來的差別有多大」目前只有 `runs/*/README.md` 裡的
人眼判讀（「像檔案壞掉」對「像印花布」），沒有任何可以跨批並排的數。

三個數，缺一不可
────────────────────────────────────────────────────────────────────
    nima_orig / cnniqa_orig   原圖自己的分數
    nima_def  / cnniqa_def    防禦圖的分數
    nima_drop / cnniqa_drop   前者減後者

**`*_orig` 不可省。** 這與 `identity.py` 的 `id_orig` 是同一條理由：單張影像的
美學分數逐圖差很多（構圖、光線、是不是插畫都會動它），不扣掉那個基準就分不出
「這個防禦讓照片變難看」與「這張照片本來分數就低」。跨組比較一律看 drop。

另報 `clip_sim`（防禦圖對原圖的 CLIP 影像餘弦），它與前兩個量的是不同的東西：
NIMA／CNNIQA 不看原圖，只問「這張圖本身好不好看」；CLIP 餘弦問的是
「還是不是同一張照片」。美化的目標是**前者高而後者不必高**——花紋本來就會
讓畫面與原圖不同。

一個必須先做的校準
────────────────────────────────────────────────────────────────────
NIMA 與 CNNIQA 是照片美學／品質模型，不保證抓得到本專案在意的那個差別
（「像刻意放上去的圖案」對「像檔案損毀」）。故 `--calibrate` 先跑在**已知答案**
的兩組上：`runs/ip2p_face_defence/` 的鋪滿（拉圖判讀為彩色雜訊）與分散
（判讀為衣服上的圓點）。指標若在這個已知的差別上分不出來，它就不可以拿來
當讀數——那時要如實回報「這個指標在本專案的判別上不可用」，不是硬套。

元件（無論文出處，故逐列寫進 CSV）
────────────────────────────────────────────────────────────────────
NIMA-AVA 與 CNNIQA 透過 `pyiqa` 取官方預訓練權重，CLIP 走
`openai/clip-vit-base-patch32`，三者都由 `src/metrics/aesthetic.py` 提供，
與 `scripts/apa_baseline.py` 用的是同一份實作。

用法：
    python scripts/aesthetic_probe.py --out runs/x/aesthetic.csv \
        --entry clothing_plain=runs/ip2p_face_defence/clothing_plain_task_x
    python scripts/aesthetic_probe.py --calibrate runs/ip2p_face_defence \
        --out runs/x/aesthetic_calibration.csv
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.naturalness import naturalness_row  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

# 校準用的兩組。名稱前綴取自 `runs/ip2p_face_defence/` 的目錄命名。
CALIBRATION_ARMS = ("plain", "pretty")


def _load(p: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def _pair(run: Path, image: str):
    """(原圖, 防禦圖, 條件名)。缺一就拋錯，**不猜**。"""
    orig = run / f"{image}__orig.png"
    hits = sorted(run.glob(f"{image}__*__def.png"))
    if not orig.exists() or not hits:
        raise SystemExit(f"{run} 缺 {image} 的原圖或防禦圖")
    return _load(orig), _load(hits[0]), hits[0].name.split("__")[1]


def _row(suite: AestheticSuite, x, d, **extra) -> dict:
    """一張防禦圖的欄位。drop 一律是「原圖減防禦圖」，正值＝變差／變雜。

    **主讀數是譜斜率那一組**（`src/metrics/naturalness.py`），它只在支撐內量，
    回答的是「放上去的那塊東西像不像雜訊」。NIMA／CNNIQA 是整張圖的分數，
    照跑照報但已知在本專案的判別上不可用，見模組 docstring。
    """
    no, co = suite.nima(x), suite.cnniqa(x)
    nd, cd = suite.nima(d), suite.cnniqa(d)
    r = dict(extra)
    r.update(naturalness_row(x, d))
    r.update({
        "nima_orig": round(no, 5), "nima_def": round(nd, 5),
        "nima_drop": round(no - nd, 5),
        "cnniqa_orig": round(co, 5), "cnniqa_def": round(cd, 5),
        "cnniqa_drop": round(co - cd, 5),
        "clip_sim": round(suite.clip_image_similarity(x, d), 5),
    })
    return r


def _scan(run: Path) -> list:
    names = sorted({p.name.split("__")[0] for p in run.glob("*__orig.png")})
    if not names:
        raise SystemExit(f"{run} 底下沒有 *__orig.png")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", default=[],
                    help="label=目錄。可重複。")
    ap.add_argument("--calibrate", type=Path, default=None, metavar="ROOT",
                    help="先在已知答案的兩組上跑：ROOT 底下所有 *_plain_* 與 "
                         "*_pretty_* 目錄。用來判斷這個指標在本專案的判別上"
                         "可不可用。")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not args.entry and args.calibrate is None:
        raise SystemExit("要嘛給 --entry，要嘛給 --calibrate，至少一個")

    suite = AestheticSuite(torch.device(args.device))
    rows = []

    entries = []
    for spec in args.entry:
        if "=" not in spec:
            raise SystemExit(f"--entry 要寫成 label=目錄，收到 {spec!r}")
        entries.append(tuple(spec.split("=", 1)))
    if args.calibrate is not None:
        for arm in CALIBRATION_ARMS:
            for d in sorted(args.calibrate.glob(f"*_{arm}_*")):
                if d.is_dir():
                    entries.append((arm, str(d)))

    for label, path in entries:
        run = Path(path)
        for image in _scan(run):
            x, d, cond = _pair(run, image)
            r = _row(suite, x, d, image=image, label=label, condition=cond,
                     run=run.name)
            rows.append(r)
            print(f"{image:<32s} {label:<16s} 支撐 {r['support_area']:.3f}  "
                  f"斜率 {r['slope_orig']}→{r['slope_def']}"
                  f"（drop {r['slope_drop']}）  熵 drop {r['entropy_drop']}  "
                  f"色數 ×{r['colours_ratio']}  "
                  f"nima drop {r['nima_drop']:+.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    print(f"\n寫出 {args.out}（{len(rows)} 列）")

    if args.calibrate is not None:
        print("\n校準：兩組已知答案的分佈（拉圖判讀為「彩色雜訊」對「圓點圖案」）")
        for arm in CALIBRATION_ARMS:
            v = [r for r in rows if r["label"] == arm]
            if not v:
                continue
            for k in ("slope_drop", "entropy_drop", "colours_ratio",
                      "nima_drop", "cnniqa_drop", "clip_sim"):
                col = [r[k] for r in v if r[k] != ""]
                if not col:
                    continue
                print(f"  {arm:<8s} {k:<12s} 中位數 {st.median(col):+.4f}"
                      f"  範圍 {min(col):+.4f} … {max(col):+.4f}  n={len(col)}")
        print("\n**分不出來就是分不出來**：兩組的分佈重疊時，這個指標不可以"
              "拿來當美化的讀數，要如實記下而不是換一個門檻硬套。")


if __name__ == "__main__":
    main()
