"""候選讀數對「人眼判定的防禦成功」的預測力。**不跑 GPU 上的擴散模型。**

問題
────────────────────────────────────────────────────────────────────
本專案的主讀數是位移 `LPIPS(編輯(原圖), 編輯(防禦圖))`。25 張的視覺稽核
（`runs/obedience_audit/`）顯示它在同一條件內確實能分辨哪一張被擋下
（AUC 0.88–1.00），但它同時在**未防禦攻擊本來就失敗**的影像上照樣為正，
而且約一半可由單純模糊複製。

所以問題不是「位移有沒有用」，而是**有沒有更貼近人眼判定的讀數**。這支把
候選讀數逐一對 39 格人眼標記算 AUC，讓選擇有依據而不是憑偏好。

候選讀數
────────────────────────────────────────────────────────────────────
    displacement      LPIPS(編輯(原圖), 編輯(防禦圖))       現行主讀數
    drift             LPIPS(原圖, 編輯(防禦圖))             輸出離原始場景多遠
    drift_ratio       drift / LPIPS(原圖, 編輯(原圖))       同上，除掉該張本來
                                                            就會移動多少
    dists_*           上列三者的 DISTS 版
    acutance          編輯(防禦圖) 相對 編輯(原圖) 的銳利度
    clip_sim          CLIP 影像空間裡兩張編輯的餘弦相似度
    siglip_sim        同上，SigLIP

`drift` 這一組的動機來自逐張看圖：baseline 擋下的每一格都是模型**重畫出
另一個場景**而不是溫和劣化。若那是防禦成功的真正型態，「輸出離原始場景
多遠」應該比「兩張輸出彼此差多少」更貼近判定。

語意那一組不需要 caption——`semantic` 量的是影像對一句文字的對齊，而
OmniEdit 給的是指令，那條路徑在服從率驗收上近乎隨機。

AUC 的方向：對每個讀數各報「大者為擋下」與「小者為擋下」兩個方向裡較高的
那個，並標明方向，避免因為符號猜錯而把有判別力的量誤判為無用。
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from defense_purify_gallery import RESOLUTION, discover  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402


def auc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=None,
                    help="人眼判定的 CSV。給定時只算它列出的格子並報 AUC；"
                         "不給時算 `--src` 底下**全部**格子，只寫 CSV 不報 AUC"
                         "（沒有金標準就沒有可算的判別力）")
    ap.add_argument("--images", nargs="+", default=None,
                    help="只算這些影像。用來限制在「未防禦編輯確實執行了指令」"
                         "的子集上")
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    found = discover(args.src)
    if args.labels is not None:
        labels = {(r["image"], r["condition"]): r["verdict"]
                  for r in csv.DictReader(args.labels.open(encoding="utf-8"))}
    else:
        labels = {(img, cond): ""
                  for img, by in found.items() for cond in by
                  if cond != "__orig__"}
    if args.images:
        keep = set(args.images)
        labels = {k: v for k, v in labels.items() if k[0] in keep}
    suite = MetricSuite(device=torch.device("cpu"))
    dev = torch.device("cpu")

    rows: List[Dict] = []
    for (image, cond), verdict in sorted(labels.items()):
        kinds = found.get(image, {}).get(cond, {})
        if not {"edit_orig", "edit_def"} <= set(kinds):
            continue
        orig_path = args.data / image / f"{image}.png"
        if not orig_path.exists():
            continue
        x = load_image_tensor(orig_path, dev, size=RESOLUTION)
        eo = load_image_tensor(kinds["edit_orig"], dev, size=RESOLUTION)
        ed = load_image_tensor(kinds["edit_def"], dev, size=RESOLUTION)

        m_oo = suite.pairwise(x, eo)      # 原圖 → 未防禦編輯
        m_od = suite.pairwise(x, ed)      # 原圖 → 防禦後編輯
        m_dd = suite.pairwise(eo, ed)     # 兩張編輯之間
        sim = suite.image_similarity(eo, ed)

        rows.append({
            "image": image, "condition": cond, "verdict": verdict,
            "displacement": round(m_dd["lpips"], 5),
            "dists_displacement": round(m_dd["dists"], 5),
            "drift": round(m_od["lpips"], 5),
            "dists_drift": round(m_od["dists"], 5),
            "drift_ratio": round(m_od["lpips"] / max(m_oo["lpips"], 1e-6), 4),
            "acutance": round(m_dd["acutance_ratio"], 4),
            "clip_sim": round(sim["clip"], 5),
            "siglip_sim": round(sim["siglip"], 5),
        })
        print(f"  {image[:32]:32s} {cond:14s} {verdict}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)

    keys = [k for k in rows[0] if k not in ("image", "condition", "verdict")]
    pos = [r for r in rows if r["verdict"] == "blocked"]
    neg = [r for r in rows if r["verdict"] == "attack_succeeded"]
    if not pos or not neg:
        # 沒有金標準就不報判別力。此時本檔的用途是產出逐格的讀數表。
        by_cond: Dict[str, List[dict]] = {}
        for r in rows:
            by_cond.setdefault(r["condition"], []).append(r)
        print(f"\n{'條件':18s}{'n':>4s}{'SigLIP 均值':>13s}{'低於 0.837':>12s}")
        for cond, rs in sorted(by_cond.items()):
            b = sum(1 for r in rs if float(r["siglip_sim"]) < 0.837)
            print(f"{cond:18s}{len(rs):4d}"
                  f"{st.fmean(float(r['siglip_sim']) for r in rs):13.3f}"
                  f"{b:>8d}/{len(rs)}")
        return
    print(f"\n全部條件池化（擋下 n={len(pos)}、攻擊成功 n={len(neg)}）")
    print(f"{'讀數':22s}{'AUC':>8s}{'方向':>8s}{'擋下均值':>12s}{'成功均值':>12s}")
    scored = []
    for k in keys:
        up = auc([float(r[k]) for r in pos], [float(r[k]) for r in neg])
        best, direction = (up, "大") if up >= 0.5 else (1 - up, "小")
        scored.append((best, k, direction,
                       st.fmean(float(r[k]) for r in pos),
                       st.fmean(float(r[k]) for r in neg)))
    for a, k, d, mp, mn in sorted(scored, reverse=True):
        print(f"{k:22s}{a:8.3f}{d:>8s}{mp:12.4f}{mn:12.4f}")

    print("\n同一條件內（跨條件的強度差被排除，這才是乾淨的判別力）")
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["condition"]][r["verdict"]].append(r)
    for cond, g in sorted(by.items()):
        p, n = g["blocked"], g["attack_succeeded"]
        if not p or not n:
            print(f"  {cond:16s} 擋下 {len(p)}／成功 {len(n)}：單邊，跳過")
            continue
        line = []
        for k in keys:
            a = auc([float(r[k]) for r in p], [float(r[k]) for r in n])
            line.append((max(a, 1 - a), k))
        line.sort(reverse=True)
        top = "  ".join(f"{k} {a:.2f}" for a, k in line[:4])
        print(f"  {cond:16s} 擋下 {len(p)}／成功 {len(n)}   最佳四項： {top}")


if __name__ == "__main__":
    main()
