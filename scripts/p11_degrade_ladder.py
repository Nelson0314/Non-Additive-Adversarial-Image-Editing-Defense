"""E31 判準補完：編輯輸出的感知劣化階梯。

為什麼需要這一支。E25 把防禦成功的判準由 `net_lpips` 改為語意軸是對的，
但 arXiv:2512.14320 定義的 ISR 是一個聯集——「語意不符 prompt」**或**
「明顯的感知劣化」。本專案只取了前半。在 strength=0.5 的全域 SDEdit 下，
輸出主要由 prompt 重新生成，語意不符在原理上幾乎不可達成；文獻上真正被
達成的是後半（PhotoGuard 推向灰色目標，成功的樣子就是輸出糊掉）。

指標與門檻都不得憑聲譽選——本專案已出現過三次（ΔE00、NLPD、VIF）。作法沿用
E20 的四條件等 LPIPS 探針與 E28 的色度階梯：造一道已知強度的劣化階梯，四個
候選無參考指標各評一次，再由人眼比對頁判定哪一級開始「這張圖已經不能用」，
門檻定在那一級。

執行：python scripts/p11_degrade_ladder.py
成本：本機 CPU 或 4 GB GPU，數分鐘。不需要 SD 權重。
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.purify.ops import gaussian_blur, gaussian_noise, jpeg_real, quantize_real

# 候選指標。四個都是無參考的：判準要問的是「這張輸出本身還能不能用」，
# 不是「它離未防禦的輸出多遠」——後者正是 E25／E29 兩次判定失效的量。
CANDIDATES = ("niqe", "musiq", "clipiqa", "topiq_nr")
# 方向：越大越好者為 True。NIQE 是與自然影像統計的距離，越小越好。
HIGHER_IS_BETTER = {"niqe": False, "musiq": True, "clipiqa": True,
                    "topiq_nr": True}


def ladder_arms():
    """四個算子各四級，由弱到強。

    強度取自 `src/purify/ops.py::eval_sweep` 的量級，但四級之間的間距放大：
    那裡要量的是淨化的殘存率（弱擾動即可），此處要跨越「還能用」到「不能用」，
    範圍不夠寬時人眼永遠看不到後者，門檻就定不出來。
    """
    return (
        [("blur", s) for s in (0.5, 1.0, 2.0, 4.0)]
        + [("jpeg", q) for q in (80, 50, 30, 10)]
        + [("noise", s) for s in (0.02, 0.05, 0.10, 0.20)]
        + [("quantize", n) for n in (64, 16, 8, 4)]
    )


def apply_arm(x, kind, strength):
    """階梯一律用真實實作，不用可微代理：這裡沒有梯度要走。"""
    if kind == "blur":
        return gaussian_blur(x, strength)
    if kind == "noise":
        return gaussian_noise(x, strength, seed=0)
    if kind == "jpeg":
        return jpeg_real(x, int(strength))
    if kind == "quantize":
        return quantize_real(x, int(strength))
    raise ValueError(f"未知的階梯算子 {kind!r}")


def load_sources(root: Path, size: int, device):
    """來源取**未防禦的編輯輸出**，不是原圖。

    理由：判準作用的對象是編輯輸出。編輯輸出本身已經是擴散生成的，其無參考
    品質基線與自然照片不同（實測 runs/e29c_P 的 edit_orig NIQE 為 3.00，
    而自然照片一般更低），用原圖判讀會把門檻定在錯的基線上。

    兩處來源：既有 run 的 `edit_orig.png`，以及 `scripts/e31_make_edits.py`
    在本機補產生的。補產生的理由是既有 run 中以正確攻擊設定（w=7.5）跑出來
    的未防禦編輯只有 car_00 一張——`runs/e29c_C_*` 與 `runs/e29c_P_*` 的
    `edit_orig.png` 經雜湊比對是同一個檔（SDEdit 在相同影像／prompt／種子／
    strength 下是決定性的），用一張圖判讀門檻太薄。

    逐檔以位元組雜湊去重，否則同一張圖會在比對頁上出現兩次，看起來像兩個
    獨立樣本。
    """
    import hashlib

    paths = (sorted(root.glob("runs/e29c_*_tau0.10/*/edit_orig.png"))
             + sorted(root.glob("runs/e31_sources/*__edit_orig.png")))
    if not paths:
        raise FileNotFoundError(
            "找不到任何未防禦的編輯輸出。請先確認 runs/e29c_*_tau0.10/ 已入"
            "版控，或跑 scripts/e31_make_edits.py 產生 runs/e31_sources/"
        )
    out, seen = [], set()
    for p in paths:
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen:
            print(f"  [跳過重複] {p.parent.name}/{p.name}", flush=True)
            continue
        seen.add(digest)
        name = (p.stem.replace("__edit_orig", "")
                if p.parent.name == "e31_sources"
                else f"{p.parent.parent.name}/{p.parent.name}")
        img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        out.append((name, T.ToTensor()(img).unsqueeze(0).to(device)))
    return out


def _slug(s):
    return s.replace("/", "__")


def _fmt(r):
    return " ".join(f"{k}={r[k]:.2f}" for k in CANDIDATES)


def _write_compare_page(out: Path, rows):
    """比對頁：每一列一個算子，欄為四級，格內附四個指標值。

    指標之間出現矛盾時以人眼為準。這是本專案第五次用同一個作法
    （E20 的 p1、E25 的 p5、E26 的 p7、E28 的 p10）。
    """
    srcs = sorted({r["source"] for r in rows})
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>E31 劣化階梯</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;"
        "padding:16px}table{border-collapse:collapse}"
        "td{padding:4px;text-align:center;vertical-align:top}"
        "img{width:190px;display:block}small{font-size:11px;color:#aaa}"
        "th{text-align:left;color:#8cf}</style>",
        "<h1>E31 劣化階梯：哪一級開始「這張圖不能用」</h1>",
        "<p>來源是既有 run 的<b>未防禦</b>編輯輸出。請逐列由左至右看，"
        "指出第一個你認為已經不能用的欄位。四個指標的數值只作參考，"
        "判定以人眼為準。</p>",
        "<p>方向：niqe 越小越好；musiq、clipiqa、topiq_nr 越大越好。</p>",
    ]
    for s in srcs:
        html.append(f"<h2>{s}</h2><table>")
        base = [r for r in rows if r["source"] == s and r["kind"] == "none"][0]
        html.append(
            "<tr><th>原始</th><td><img src='img/%s__none.png'>"
            "<small>%s</small></td></tr>" % (_slug(s), _fmt(base)))
        for kind in ("blur", "jpeg", "noise", "quantize"):
            html.append(f"<tr><th>{kind}</th>")
            for r in [r for r in rows
                      if r["source"] == s and r["kind"] == kind]:
                html.append(
                    "<td><img src='img/%s__%s_%s.png'>"
                    "<small>%s = %s<br>%s</small></td>"
                    % (_slug(s), kind, r["strength"], kind, r["strength"],
                       _fmt(r)))
            html.append("</tr>")
        html.append("</table>")
    (out / "compare.html").write_text("\n".join(html), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/p11_degrade_ladder")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    import pyiqa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    (out / "img").mkdir(parents=True, exist_ok=True)

    metrics = {k: pyiqa.create_metric(k, device=device) for k in CANDIDATES}
    sources = load_sources(ROOT, args.size, device)
    print(f"[p11] device={device} 來源 {len(sources)} 張", flush=True)

    rows = []
    for src_name, x in sources:
        x = x.clamp(0, 1)
        base = {k: float(m(x)) for k, m in metrics.items()}
        rows.append({"source": src_name, "kind": "none", "strength": 0.0,
                     "level": 0, **base})
        save_image(x, out / "img" / f"{_slug(src_name)}__none.png")
        print(f"  {src_name} 原始  " + _fmt(base), flush=True)
        for kind, strength in ladder_arms():
            level = [s for k, s in ladder_arms()
                     if k == kind].index(strength) + 1
            y = apply_arm(x, kind, strength).clamp(0, 1)
            row = {"source": src_name, "kind": kind, "strength": strength,
                   "level": level, **{k: float(m(y)) for k, m in metrics.items()}}
            rows.append(row)
            save_image(y, out / "img" / f"{_slug(src_name)}__{kind}_{strength}.png")
            print(f"  {src_name} {kind}={strength}  " + _fmt(row), flush=True)

    with open(out / "ladder.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _write_compare_page(out, rows)
    print(f"[p11] 寫出 {out / 'ladder.csv'} 與 {out / 'compare.html'}")
    print("[p11] 下一步由人眼判讀 compare.html，決定哪一級開始「不能用」，"
          "並選出在該級上可分的指標。門檻寫進 docs/RESULTS_E25-E31.md")


if __name__ == "__main__":
    main()
