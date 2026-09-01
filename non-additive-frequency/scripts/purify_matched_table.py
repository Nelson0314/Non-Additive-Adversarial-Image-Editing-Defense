"""等失真的抗淨化表：把「付了多少失真」與「每個算子上的淨增益」接起來。

為什麼需要這一支
────────────────────────────────────────────────────────────────────
現有兩支各做一半，中間有個缺口：

- `matched_distortion_table.py` 讀訓練批次的 `results.csv`，做的是
  **失真對未淨化位移**的等失真內插——只有 identity 那一欄。
- `retention_table.py` 讀 `phase_retention.py` 的輸出，做的是**逐算子的
  總增益與淨增益**——但每個條件停在自己的失真上，沒有對齊。

於是「同失真下誰的模糊欄比較高」問不出來。低自由度的參數化（明暗場、色彩族）
最容易死在「與同失真隨機對照無法區分」（FND-004 的死法），而那正是這個缺口
裡的問題。本支把兩邊接起來。

協定（DEC-029，逐條照用）
────────────────────────────────────────────────────────────────────
- 曲線點 = 逐強度的平均。x 軸是失真（預設 `fid_dists`），y 軸是**該算子的
  淨增益**。
- 錨點以**線性內插**求得，**落在掃描範圍外一律拒絕外插**並標 `out_of_range`。
- 逐圖相減先做：地板逐圖不同，先平均再相減會把地板的影像組成混進差額。
- **兩個失真軸都要報**：`--x fid_dists` 與 `--x fid_lpips` 各跑一次。
  兩者對同一組影像的判定經常相反，只在單一軸上成立的結論不算數。
- 幾何類算子的地板由構造為 0，扣地板是恆等變換；相減照做不特例化。

輸入
────────────────────────────────────────────────────────────────────
`--train` 是訓練批次的目錄（各含 `results.csv`），提供失真；
`--purify` 是抗淨化批次的目錄（含 `<tag>_all.csv` 與 `floor_all.csv`），
提供逐算子的 effect。兩邊以**目錄名（tag）**對應。

用法：

    python scripts/purify_matched_table.py \\
        --train runs/ip2p_shading/*/ --purify runs/ip2p_shading_purify \\
        --curve opt --curve-of rand --anchor 0.03 --out runs/.../matched.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import write_csv  # noqa: E402

FLOOR_TAG = "floor"
SHARDS = ("color", "scene", "object", "all")


def tag_of(filename: str) -> str:
    """`opt_r010_all.csv` → `opt_r010`。分片名是族群不是流水號。"""
    stem = filename[:-4] if filename.endswith(".csv") else filename
    for shard in SHARDS:
        if stem.endswith("_" + shard):
            return stem[: -len(shard) - 1]
    raise ValueError(f"{filename} 的分片名不在 {SHARDS} 裡，無法還原標籤")


def read_purify(src: Path, field: str
                ) -> Dict[Tuple[str, str, str], float]:
    """→ {(tag, image, purifier): effect}。`usable=False` 與空讀數的列排除。"""
    out: Dict[Tuple[str, str, str], float] = {}
    dropped = 0
    for path in sorted(src.rglob("*_all.csv")):
        tag = tag_of(path.name)
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("usable", "True")).lower() == "false":
                    dropped += 1
                    continue
                if r.get(field, "") in ("", None):
                    dropped += 1
                    continue
                key = (tag, r["image"], r["purifier"])
                if key in out:
                    raise ValueError(f"重複的格：{key}（{path}）")
                out[key] = float(r[field])
    print(f"抗淨化：讀入 {len(out)} 格，usable=False 排除 {dropped} 列")
    return out


def read_train(dirs: List[Path], x_field: str) -> Dict[Tuple[str, str], float]:
    """→ {(tag, image): 失真}。tag 取目錄名。"""
    out: Dict[Tuple[str, str], float] = {}
    for d in dirs:
        csv_path = d / "results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{d} 沒有 results.csv")
        with csv_path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                v = r.get(x_field, "")
                if v in ("", None):
                    raise KeyError(
                        f"{csv_path} 沒有 {x_field} 欄，無法對齊失真")
                out[(d.name, r["image"])] = float(v)
    return out


def net_gains(purify: Dict[Tuple[str, str, str], float], tag: str,
              images: List[str], purifier: str) -> Optional[float]:
    """逐圖扣地板再平均。缺地板或缺格的影像整格排除。"""
    vals = []
    for img in images:
        eff = purify.get((tag, img, purifier))
        flo = purify.get((FLOOR_TAG, img, purifier))
        if eff is None or flo is None:
            continue
        vals.append(eff - flo)
    return statistics.fmean(vals) if vals else None


def interpolate(points: List[Tuple[float, float]], anchor: float
                ) -> Tuple[Optional[float], str, Optional[Tuple], Optional[Tuple]]:
    """在 (x, y) 上線性內插。**範圍外不外插**，回傳 `out_of_range`。"""
    pts = sorted(points)
    if not pts:
        return None, "no_points", None, None
    if anchor < pts[0][0] or anchor > pts[-1][0]:
        return None, "out_of_range", pts[0], pts[-1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= anchor <= x1:
            if x1 == x0:
                return y0, "exact", (x0, y0), (x1, y1)
            w = (anchor - x0) / (x1 - x0)
            return y0 + w * (y1 - y0), "interpolated", (x0, y0), (x1, y1)
    return None, "out_of_range", pts[0], pts[-1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, nargs="+", required=True,
                    help="訓練批次的目錄，各含 results.csv。tag 取目錄名")
    ap.add_argument("--purify", type=Path, nargs="+", required=True,
                    help="抗淨化批次的目錄，含 <tag>_all.csv 與 floor_all.csv")
    ap.add_argument("--x", default="fid_dists",
                    help="失真軸。協定要求 fid_dists 與 fid_lpips 各跑一次")
    ap.add_argument("--anchor", type=float, nargs="+", required=True,
                    help="要對齊的失真值。範圍外一律拒絕外插")
    ap.add_argument("--group", nargs="+", action="append", required=True,
                    metavar="NAME=TAG,TAG",
                    help="一條曲線。例如 opt=opt_r010,opt_r020")
    ap.add_argument("--purifiers", nargs="+", default=None,
                    help="只報這些算子。預設全部")
    ap.add_argument("--effect-field", default="effect_mean",
                    help="用哪一欄當 effect。`effect_mean` 是現行參照，"
                         "`effect_codefense_mean` 是共防禦參照。兩者並列不取代")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    purify: Dict[Tuple[str, str, str], float] = {}
    for d in args.purify:
        purify.update(read_purify(d, args.effect_field))
    train = read_train(list(args.train), args.x)

    curves: Dict[str, List[str]] = {}
    for group in args.group:
        for spec in group:
            name, _, tags = spec.partition("=")
            if not tags:
                raise SystemExit(f"--group 要寫成 NAME=TAG,TAG，收到 {spec!r}")
            curves[name] = tags.split(",")

    # 影像交集：不同曲線的完成張數可能不同，把張數不同的平均並排讀會把
    # 「跑到哪張」讀成方法差異。
    per_tag_images = {}
    for tags in curves.values():
        for t in tags:
            per_tag_images[t] = {img for (tg, img, _) in purify if tg == t}
    floor_images = {img for (tg, img, _) in purify if tg == FLOOR_TAG}
    common = set.intersection(*per_tag_images.values(), floor_images) \
        if per_tag_images else set()
    images = sorted(common)
    print(f"共有影像 {len(images)} 張：{images}")
    if not images:
        raise SystemExit("沒有任何曲線共有的影像")

    purifiers = args.purifiers or sorted(
        {p for (_, _, p) in purify if p != ""})

    rows = []
    for pur in purifiers:
        for name, tags in curves.items():
            pts = []
            for t in tags:
                xs = [train[(t, img)] for img in images if (t, img) in train]
                if len(xs) != len(images):
                    raise KeyError(
                        f"{t} 在訓練批次裡缺影像：有 {len(xs)} 張、"
                        f"共有影像 {len(images)} 張。缺的那幾張要先補齊，"
                        f"否則失真與效果不是同一組影像上的。")
                y = net_gains(purify, t, images, pur)
                if y is None:
                    continue
                pts.append((statistics.fmean(xs), y))
            for a in args.anchor:
                y, status, lo, hi = interpolate(pts, a)
                rows.append({
                    "purifier": pur, "curve": name, "x_field": args.x,
                    "effect_field": args.effect_field,
                    "anchor": a, "n_images": len(images),
                    "n_points": len(pts),
                    "net_gain": "" if y is None else round(y, 5),
                    "status": status,
                    "lo_x": "" if lo is None else round(lo[0], 5),
                    "lo_y": "" if lo is None else round(lo[1], 5),
                    "hi_x": "" if hi is None else round(hi[0], 5),
                    "hi_y": "" if hi is None else round(hi[1], 5),
                })

    write_csv(args.out, rows)
    names = list(curves)
    print(f"\n等失真淨增益（x = {args.x}）")
    head = f"{'算子':<18}{'錨點':>9}" + "".join(f"{n:>16}" for n in names)
    print(head)
    print("-" * len(head))
    for pur in purifiers:
        for a in args.anchor:
            cells = []
            for n in names:
                r = next(r for r in rows
                         if r["purifier"] == pur and r["curve"] == n
                         and r["anchor"] == a)
                cells.append(f"{r['net_gain']:>16}" if r["status"] != "out_of_range"
                             else f"{'out_of_range':>16}")
            print(f"{pur:<18}{a:>9.4f}" + "".join(cells))
    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()
