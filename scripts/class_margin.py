#!/usr/bin/env python
"""類別 margin：把「編輯成功了沒有」量成一個與人眼一致的判定。

    python scripts/class_margin.py --batch runs/v14r_merged \
        --images bird_03 cat_02 dog_03 --out runs/v14r_margin.csv

只對既有的 PNG 做 CLIP／SigLIP 前向，**不跑 SDEdit、不佔 GPU 排程**。

## 為什麼要換讀出量

使用者 2026-08-08 定案：真值在圖上，成敗大部分很直白；`effect_siglip` 只是
代理量，而它的噪聲與訊號同量級（`RESULTS_2026-08-08` §8）正說明它沒有追蹤到
眼睛看到的那件事。這不只影響報告——`L_def` **就是**用它建的，量錯了東西
等於訓練目標也對錯了東西。

現行量：

    effect_siglip = SigLIP(對照側編輯, 目標prompt) − SigLIP(防禦側編輯, 目標prompt)
                    └──────── 兩張不同的圖 ────────┘   └──── 同一個 prompt ────┘

實測 v14r 的 dog_03：未防禦的輸出在 seed 1／3 上變成塑膠與木雕質感，那種
畫質劇變會大幅移動 SigLIP 分數，**而那隻狗根本沒變成貓**。該量把「圖變怪了」
與「類別被改掉了」放進同一個數字。

本腳本的量：

    margin(y) = SigLIP(y, 目標類) − SigLIP(y, 原類)
                └──────── 同一張圖對兩個 prompt ────────┘

- **符號即判定**：`margin > 0` 即被判成目標類，即編輯成功。這正是眼睛做的事。
- 畫質與風格的變化同時影響兩項而抵消，剩下的是類別訊息。
- 攻擊成功率 = 判成目標類的比例；防禦效果 = 成功率的下降。這是文獻慣用的
  attack success rate，且可解讀（「未防禦成功 20%，防禦後 100%」）。
- 可直接當損失：最小化 `margin(y_def)` 即把編輯輸出推回原類，對 SigLIP 可微。

## 原類從哪裡來

逐格 `metrics_*.json` 的 `group`（bird／cat／dog）即原類，`prompt` 即目標類。
兩者都是既有欄位，不新增假設。對照側（`control/`）不寫 `metrics_*.json`，
故其 `group`／`prompt` 由同一張影像的任一個 eval 格取得——那兩個欄位逐影像
固定，與條件無關。

## τ 的來源

防禦側的編輯輸出在 2026-08-08 之前不帶 τ，四個 τ 互相覆寫（見
`executors._run_eval`）。故舊批次每一格只剩一個 τ，是哪一個必須由旁邊的
`metrics_*.json` 讀，不能由檔名猜。本腳本逐格讀出並寫進 CSV 的 `tau` 欄；
讀不到就留空而不填預設值。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment.executors import load_image_tensor  # noqa: E402
from src.metrics.suite import MetricSuite                # noqa: E402

FIELDS = ["batch", "image_id", "group", "condition", "purify_dir",
          "purify_kind", "purify_strength", "seed", "tau", "png",
          "s_target", "s_source", "margin", "success",
          "clip_target", "clip_source", "clip_margin"]

# 摘要與篩選一律用 `purify_dir`（目錄名）而不是 `purify_kind`。
# 對照側（`control/`）不寫 `metrics_*.json`，其 `purify_kind` 只能落回目錄名，
# 於是 `purify_kind == "identity"` 這種比對會**靜默漏掉整個對照側**——而對照側
# 正是分母。目錄名兩側都有且逐字相同。
IDENTITY_DIR = "identity_0"


def edit_pngs(cell_dir: Path) -> List[Tuple[Optional[str], int, Path]]:
    """該格的編輯輸出，回傳 (τ, seed, path)；對照側與舊批次的 τ 為 None。

    2026-08-09 修。before：回傳 `(seed, path)`，且以 `out[seed] = p` 收集。
    帶 τ 的新檔名下同一個目錄有四個 τ × 五個 seed，四個 τ 因此**塌成同一個
    鍵**，只有 glob 排序最後的 τ=0.35 留下來。`meta_for` 反過來取排序的
    **第一個**，拿到的是 τ=0.05——影像與 metrics 於是配對錯開，而 CSV 的
    `tau` 欄看起來完全正常。

    舊批次每格只剩一個 τ（那正是 `e9a35a5c6` 修掉的覆寫），所以這個缺陷在
    v14／v14r 上不顯現；ip3 是第一個四個 τ 都在磁碟上的批次。與 §7 那個
    覆寫是同一族的漏帶鍵。

    after：鍵改為 (τ, seed)，τ 由檔名解析而不是由旁邊的 json 猜。
    """
    out: List[Tuple[Optional[str], int, Path]] = []
    for p in sorted(cell_dir.glob("edit_tau*_seed*.png")):
        tau, _, seed = p.stem[len("edit_tau"):].partition("_seed")
        out.append((tau, int(seed), p))
    if out:
        return sorted(out, key=lambda t: (float(t[0]), t[1]))
    for p in sorted(cell_dir.glob("edit_seed*.png")):
        out.append((None, int(p.stem.rsplit("seed", 1)[1]), p))
    return sorted(out, key=lambda t: t[1])


def meta_for(cell_dir: Path, tau: Optional[str], seed: int) -> Dict:
    """與該 (τ, seed) 對應的 metrics json。τ 為 None 時取不帶 τ 的舊檔名。

    **不接受「隨便一個 seed 相符的檔」**：那正是上面那個缺陷的另一半。
    """
    pats = [] if tau is None else [f"metrics_tau{tau}_seed{seed}.json"]
    pats.append(f"metrics_seed{seed}.json")
    for pat in pats:
        for p in sorted(cell_dir.glob(pat)):
            try:
                return json.load(p.open(encoding="utf-8"))
            except OSError:
                return {}
    return {}


def image_classes(batch: Path, image_id: str,
                  conds: List[str]) -> Optional[Tuple[str, str]]:
    """(原類 prompt, 目標類 prompt)。由任一個 eval 格的 metrics 取得。"""
    for cond in conds:
        if cond == "control":
            continue
        for cdir in sorted((batch / cond / image_id / "purify").glob("*")):
            for tau, seed, _ in edit_pngs(cdir):
                m = meta_for(cdir, tau, seed)
                if m.get("group") and m.get("prompt"):
                    return f"a {m['group']}", str(m["prompt"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+",
                    default=["control", "N1", "N2", "N3", "R",
                             "photoguard_c", "mist", "dia_r"])
    ap.add_argument("--purifiers", nargs="*", default=None,
                    help="淨化算子目錄名，省略即全部")
    ap.add_argument("--device", default="cpu",
                    help="cpu 或 cuda:N。SigLIP／CLIP 很小，cpu 可行但慢")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    suite = MetricSuite(device=torch.device(args.device))
    rows: List[Dict] = []

    for image_id in args.images:
        classes = image_classes(args.batch, image_id, args.conditions)
        if classes is None:
            raise SystemExit(
                f"{image_id}：找不到帶 group／prompt 的 metrics json，"
                "原類與目標類無從決定。不猜，直接停下")
        src_prompt, tgt_prompt = classes
        print(f"{image_id}: 原類 {src_prompt!r} → 目標 {tgt_prompt!r}",
              flush=True)

        for cond in args.conditions:
            root = args.batch / cond / image_id / "purify"
            if not root.is_dir():
                print(f"  {cond}: (無目錄，略過)", flush=True)
                continue
            for cdir in sorted(root.glob("*")):
                if args.purifiers and cdir.name not in args.purifiers:
                    continue
                for tau, seed, png in edit_pngs(cdir):
                    m = meta_for(cdir, tau, seed)
                    x = load_image_tensor(png, suite.device)
                    sc = suite.semantic_multi(x, [tgt_prompt, src_prompt])
                    marg = sc[tgt_prompt]["siglip"] - sc[src_prompt]["siglip"]
                    rows.append({
                        "batch": args.batch.name, "image_id": image_id,
                        "group": m.get("group", ""), "condition": cond,
                        "purify_dir": cdir.name,
                        "purify_kind": m.get("purify_kind", ""),
                        "purify_strength": m.get("purify_strength", ""),
                        # τ 取自檔名（唯一可靠的來源），json 只在舊批次
                        # 沒有帶 τ 的檔名時退而求其次
                        "seed": seed,
                        "tau": tau if tau is not None else m.get("tau", ""),
                        "png": png.relative_to(args.batch).as_posix(),
                        "s_target": sc[tgt_prompt]["siglip"],
                        "s_source": sc[src_prompt]["siglip"],
                        "margin": marg, "success": int(marg > 0),
                        "clip_target": sc[tgt_prompt]["clip"],
                        "clip_source": sc[src_prompt]["clip"],
                        "clip_margin": sc[tgt_prompt]["clip"]
                        - sc[src_prompt]["clip"],
                    })
            print(f"  {cond}: 累計 {len(rows)} 列", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n寫入 {args.out}（{len(rows)} 列）")

    # 摘要：identity（未淨化）上的編輯成功率，**逐 τ** 一張表。
    #
    # 2026-08-09 加上 τ 這一層。before：只按 (條件, 影像) 聚合。四個 τ 都在
    # 磁碟上時（ip3 是第一批）那等於把不同失真預算的成功率平均起來，而整個
    # 實驗的前提就是「在同一個 τ 上比較」。對照側（control）沒有 τ 這個軸，
    # 它是每一張表共用的分母，故在每一張表裡都印一次。
    agg = defaultdict(list)
    for r in rows:
        if r["purify_dir"] == IDENTITY_DIR:
            agg[(str(r["tau"]), r["condition"], r["image_id"])].append(
                r["success"])
    imgs = sorted({k[2] for k in agg})
    taus = sorted({k[0] for k in agg if k[0] != ""},
                  key=lambda s: float(s)) or [""]
    ctl = {k: v for k, v in agg.items() if k[0] == ""}

    for tau in taus:
        print(f"\nidentity（未淨化）上的編輯成功率　τ={tau or '（不分'})")
        print(f"{'條件':<14}" + "".join(f"{i:>12}" for i in imgs)
              + f"{'合計':>10}")
        for cond in args.conditions:
            cells, tot = [], []
            for i in imgs:
                v = agg.get((tau, cond, i)) or ctl.get(("", cond, i), [])
                tot += v
                cells.append(f"{sum(v)}/{len(v)}" if v else "—")
            line = f"{cond:<14}" + "".join(f"{c:>12}" for c in cells)
            line += f"{(f'{100*sum(tot)/len(tot):.0f}%' if tot else '—'):>10}"
            print(line)


if __name__ == "__main__":
    main()
