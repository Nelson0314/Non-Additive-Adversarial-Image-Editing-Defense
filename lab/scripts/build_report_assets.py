"""把逐格 PNG 併成報告頁用的 WebP 拼圖（sprite sheet）。

為什麼要拼圖
────────────────────────────────────────────────────────────────────
報告頁是 artifact，**整個 artifact 的檔案數上限是 256**，而這一批逐格影像有
六個臂 × 8 影像 × 4 指令 × 2 場景 ×（未淨化 ＋ 淨化）＝ 768 張，加上未防禦
對照臂共 896 張，逐張發佈直接超過上限。把同一個條件的格子併成一張大圖，
頁面再用 CSS `background-position` 切出 512×512 的原尺寸方塊：**檔案數降到
四十幾個，而每一格仍是原解析度**。

有損與無損分開
────────────────────────────────────────────────────────────────────
| 拼圖 | 編碼 | 理由 |
|---|---|---|
| 防禦圖、混合場、SDEdit 原始輸出 | **無損** | 自然度要逐像素判，有損會把接縫與色帶抹掉 |
| 編輯圖、淨化後編輯圖 | 有損 q86 | 這一層判的是「攻擊者拿不拿得到可用結果」，不是像素 |

這個分法要寫在頁面上，不藏。

拼圖的排法
────────────────────────────────────────────────────────────────────
- 防禦類：4 欄 × 2 列，順序是 `man_00..03`、`woman_00..03`。
- 編輯類：4 欄（指令 p0–p3）× 8 列（影像，同上順序）。

用法
    python lab/scripts/build_report_assets.py --out lab/report/img
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

TILE = 512
IMAGES = ["man_00", "man_01", "man_02", "man_03",
          "woman_00", "woman_01", "woman_02", "woman_03"]
ARMS = ["style_random", "style_low", "style_filter", "style_filter_guided",
        "style_affine", "curve_dual_spatial", "curve_dual_spatial_anchored",
        "curve_dual_chroma", "ab_warp", "inpaint_bg", "inpaint_outside_face"]
SCENARIOS = ["ip2p", "inpaint"]
PROMPTS = [0, 1, 2, 3]
PURIFIER = "jpeg30"


def sheet(paths, cols: int, out: Path, lossless: bool, quality: int = 86) -> dict:
    """`paths` 依列優先排好；`None` 留白（缺檔要看得出來，不是靜默跳過）。"""
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * TILE, rows * TILE), (24, 24, 28))
    missing = []
    for i, p in enumerate(paths):
        x, y = (i % cols) * TILE, (i // cols) * TILE
        if p is None or not Path(p).is_file():
            missing.append(str(p))
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (TILE, TILE):
            im = im.resize((TILE, TILE), Image.LANCZOS)
        canvas.paste(im, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    if lossless:
        canvas.save(out, "WEBP", lossless=True, quality=100, method=6)
    else:
        canvas.save(out, "WEBP", quality=quality, method=6)
    return {"file": out.name, "cols": cols, "rows": rows,
            "bytes": out.stat().st_size, "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("lab/runs"))
    ap.add_argument("--out", type=Path, default=Path("lab/report/img"))
    ap.add_argument("--quality", type=int, default=86)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="要收的臂。預設是 ARMS 常數那一串")
    ap.add_argument("--aux", nargs="*", default=None,
                    help="`<臂>:<後綴>`，收該臂的副圖（field、warp、raw…）")
    args = ap.parse_args()

    R = args.root
    global ARMS
    if args.arms:
        ARMS = list(args.arms)
    manifest = {"tile": TILE, "images": IMAGES, "arms": ARMS,
                "scenarios": SCENARIOS, "prompts": PROMPTS,
                "purifier": PURIFIER, "sheets": {}}

    def add(key, info):
        manifest["sheets"][key] = info
        print(f"{key:44s} {info['file']:36s} "
              f"{info['bytes']/1e6:6.2f} MB  缺 {len(info['missing'])}", flush=True)

    # ---- 防禦圖（無損）----
    add("defence/original", sheet(
        [R / "defence" / "style_random" / f"{n}__orig.png" for n in IMAGES],
        4, args.out / "def_original.webp", lossless=True))
    for arm in ARMS:
        add(f"defence/{arm}", sheet(
            [R / "defence" / arm / f"{n}__{arm}__def.png" for n in IMAGES],
            4, args.out / f"def_{arm}.webp", lossless=True))

    # ---- 副圖（無損）：混合場、色度平面、pipeline 原輸出 ----
    aux = args.aux if args.aux is not None else [
        f"{a}:field" for a in ARMS] + ["ab_warp:warp", "style_filter:sdedit_raw",
                                       "inpaint_bg:raw", "inpaint_outside_face:raw"]
    for spec in aux:
        arm, suffix = spec.split(":", 1)
        paths = [R / "defence" / arm / f"{n}__{suffix}.png" for n in IMAGES]
        if all(p.is_file() for p in paths):
            add(f"aux/{arm}/{suffix}",
                sheet(paths, 4, args.out / f"aux_{arm}_{suffix}.webp",
                      lossless=True))

    # ---- 編輯圖（有損）----
    def edit_dir(cond, scenario, purified):
        if cond == "undefended":
            if purified:
                return (R / "edit_purified" / "undefended" / PURIFIER
                        / f"{scenario}_undefended_{PURIFIER}")
            arm = "ip2p_si18" if scenario == "ip2p" else "inpaint_undefended"
            return R / "edit_preflight" / arm
        if purified:
            return (R / "edit_purified" / cond / PURIFIER
                    / f"{scenario}_{cond}_{PURIFIER}")
        return R / "edit_defended" / cond / f"{scenario}_{cond}"

    for cond in ["undefended"] + ARMS:
        for scenario in SCENARIOS:
            for purified in (False, True):
                d = edit_dir(cond, scenario, purified)
                paths = [d / f"{n}__p{p}.png" for n in IMAGES for p in PROMPTS]
                tag = f"{PURIFIER}" if purified else "plain"
                add(f"edit/{cond}/{scenario}/{tag}", sheet(
                    paths, 4,
                    args.out / f"edit_{cond}_{scenario}_{tag}.webp",
                    lossless=False, quality=args.quality))

    total = sum(v["bytes"] for v in manifest["sheets"].values())
    miss = sum(len(v["missing"]) for v in manifest["sheets"].values())
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n拼圖 {len(manifest['sheets'])} 張，合計 {total/1e6:.1f} MB，"
          f"缺檔 {miss} 格 → {args.out}", flush=True)
    if miss:
        for k, v in manifest["sheets"].items():
            for m in v["missing"][:3]:
                print(f"  缺 {k}: {m}", flush=True)


if __name__ == "__main__":
    main()
