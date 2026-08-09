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

from src.experiment import grid                         # noqa: E402
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


def edit_pngs(cell_dir: Path) -> List[Tuple[int, str, Path]]:
    """該格的編輯輸出，回傳 `(seed, τ 字串, path)`，**逐 τ 各一列**。

    2026-08-09 改。before：回傳 `(seed, path)`，內部以 `Dict[int, Path]`
    **只以 seed 為鍵**，於是同一個 seed 的五個 τ 互相覆寫，最後留下的是
    字串排序最大的那一個。

    那個寫法是為「四個 τ 寫同一個檔名而互相覆寫」的舊產物設計的
    （`RESULTS_2026-08-08` §10），在 `e9a35a5c6` 讓檔名帶上 τ 之後就不再
    成立——新批次五個 τ 的 PNG 同時存在，而它只會挑一個。

    症狀特別難看出來：挑中的**恰好**是最大的 τ（"0.5" 在 "0.05"、"0.1"、
    "0.2"、"0.35" 之後），也就是主表那一點，所以 margin 的數值是對的；
    錯的是配上去的 metadata（見 `meta_for`），而 `tau` 欄一錯，
    `eval_protocols` 的 ISR 段就按錯的 τ 分組。**數字看起來完全合理。**
    這個「挑中對的那個」純屬字串排序的巧合：τ 若含 0.55 與 0.6 就會挑錯。

    對照側（`control/`）是 φ=0，沒有 τ 這個軸，其檔名不帶 τ，故回傳空字串。
    """
    out: List[Tuple[int, str, Path]] = []
    for p in sorted(cell_dir.glob("edit_tau*_seed*.png")):
        stem = p.stem
        tau = stem.split("_tau", 1)[1].split("_seed")[0]
        out.append((int(stem.rsplit("seed", 1)[1]), tau, p))
    if out:
        return sorted(out, key=lambda t: (t[0], t[1]))
    for p in sorted(cell_dir.glob("edit_seed*.png")):
        out.append((int(p.stem.rsplit("seed", 1)[1]), "", p))
    return sorted(out, key=lambda t: (t[0], t[1]))


def meta_for(cell_dir: Path, seed: int, tau: str = "") -> Dict:
    """該格的 metrics。**τ 給定時只讀對應那一個檔。**

    2026-08-09 改。before：`for p in sorted(glob(f"metrics_tau*_seed{seed}"))`
    後立刻 `return`，即永遠取排序**第一個**（τ 最小的那個），而 PNG 那一側
    取的是最後一個。兩者於是配到不同的 τ：s3a 實測 `photoguard_c` 的圖是
    τ=0.5 而 metadata 來自 τ=0.05。`tau`、`purify_kind`、`prompt` 全部跟著錯。
    """
    pats = ([f"metrics_tau{tau}_seed{seed}.json"] if tau else []) + [
        f"metrics_tau*_seed{seed}.json", f"metrics_seed{seed}.json"]
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
            for seed, tau, _ in edit_pngs(cdir):
                m = meta_for(cdir, seed, tau)
                if m.get("group") and m.get("prompt"):
                    return f"a {m['group']}", str(m["prompt"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    # 預設由格點的登記表導出，不再寫死一份會與它分岔的清單——寫死的症狀是
    # 新條件被靜默漏掉，表格看起來完整只是少了幾列（2026-08-09）。
    ap.add_argument("--conditions", nargs="+",
                    default=["control", *grid.CONDITIONS])
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
                for seed, tau, png in edit_pngs(cdir):
                    m = meta_for(cdir, seed, tau)
                    x = load_image_tensor(png, suite.device)
                    sc = suite.semantic_multi(x, [tgt_prompt, src_prompt])
                    marg = sc[tgt_prompt]["siglip"] - sc[src_prompt]["siglip"]
                    rows.append({
                        "batch": args.batch.name, "image_id": image_id,
                        "group": m.get("group", ""), "condition": cond,
                        "purify_dir": cdir.name,
                        "purify_kind": m.get("purify_kind", ""),
                        "purify_strength": m.get("purify_strength", ""),
                        "seed": seed, "tau": (tau or m.get("tau", "")),
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

    # 摘要：identity 上的編輯成功率，逐條件逐影像，**逐 τ 分開**。
    #
    # 不可跨 τ 混算：那等於拿不同失真預算的條件互比，而整個實驗設計的共同
    # 貨幣就是 τ（`DESIGN` §3.2）。2026-08-09 之前本表確實會混——因為
    # `edit_pngs` 每格只回傳一個 τ，混不起來也看不出來。
    #
    # 對照側（`control/`）是 φ=0，沒有 τ 這個軸，故它在每一個 τ 的表裡都出現，
    # 且逐 τ 相同——它就是全部條件共用的分母。
    taus = sorted({r["tau"] for r in rows
                   if r["purify_dir"] == IDENTITY_DIR and r["tau"]},
                  key=float)
    for tau in taus:
        agg = defaultdict(list)
        for r in rows:
            if r["purify_dir"] != IDENTITY_DIR:
                continue
            if r["tau"] and r["tau"] != tau:
                continue
            agg[(r["condition"], r["image_id"])].append(r["success"])
        imgs = sorted({k[1] for k in agg})
        print(f"\nidentity（未淨化）上的編輯成功率　τ = {tau}")
        print(f"{'條件':<14}" + "".join(f"{i:>12}" for i in imgs)
              + f"{'合計':>10}")
        for cond in args.conditions:
            cells, tot = [], []
            for i in imgs:
                v = agg.get((cond, i), [])
                tot += v
                cells.append(f"{sum(v)}/{len(v)}" if v else "—")
            line = f"{cond:<14}" + "".join(f"{c:>12}" for c in cells)
            line += f"{(f'{100*sum(tot)/len(tot):.0f}%' if tot else '—'):>10}"
            print(line)


if __name__ == "__main__":
    main()
