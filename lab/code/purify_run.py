"""把淨化算子套在已產好的防禦圖（或未防禦的原圖）上，寫成可直接再編輯的目錄。

為什麼淨化要單獨一支
────────────────────────────────────────────────────────────────────
淨化是**攻擊方**做的事：拿到一張看起來正常的圖，先過一道常見的影像處理，
再送進編輯模型。所以它讀的是已經寫成 8 位元 PNG 的防禦圖，不重跑攻擊——
重跑攻擊只會把 `photoguard_c` 的 6400 秒/張再付一次，換到同一批防禦圖。

輸出的版面刻意與防禦程式相同（`<名稱>__def.png`），
`code/edit_preflight.py --defended <目錄>` 不必改就能吃。

算子與設定
────────────────────────────────────────────────────────────────────
| 標籤 | 算子 | 設定 |
|---|---|---|
| `identity` | 無 | 保留率的分母，不可排除 |
| `crop_resize0.1` | 中心裁切後升回 | 每邊裁 10%，**保留中央 80% 邊長**，bicubic |
| `jpeg30` / `jpeg50` / `jpeg80` | JPEG 再解碼 | 品質因子 30 / 50 / 80 |
| `blur1` / `blur2` | 高斯模糊 | σ = 1.0 / 2.0 |
| `rotate15` | 繞中心旋轉 | 15°，雙線性、邊界補零 |

`rotate` 與 `crop_resize` 是**幾何類**：取景本身被改掉，之後的讀數同時含
「防禦被洗掉多少」與「畫面被移動多少」，兩者在報表上不可混為一談
（`src/purify/ops.py` 的 `GEOMETRIC_KINDS`）。

用法
    python code/purify_run.py --defended images/defence_portraits/mist \\
        --out images/edit_purified/mist
    python code/purify_run.py --out images/edit_purified/undefended
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from src.purify.ops import Purifier  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

#: 逐項 (kind, strength)。標籤與 `phase_retention.label()` 同式，
#: 故兩邊的 CSV 可以直接對起來。
PURIFIERS = (
    ("identity", 0.0),
    ("crop_resize", 0.1),
    ("jpeg", 30),
    ("jpeg", 50),
    ("jpeg", 80),
    ("blur", 1.0),
    ("blur", 2.0),
    ("rotate", 15.0),
)


def label(kind: str, strength: float) -> str:
    return kind if not strength else f"{kind}{strength:g}"


def defended_sources(directory: Path) -> dict:
    """`<名稱>__<條件>__def.png`（條件名在中間），同一張只能對到一個檔案。"""
    out = {}
    for path in sorted(directory.glob("*__def.png")):
        name = path.name.split("__")[0]
        if name in out:
            raise SystemExit(
                f"{directory} 裡 {name} 對到兩個防禦圖："
                f"{out[name].name} 與 {path.name}；目錄混了兩個條件")
        out[name] = path
    if not out:
        raise SystemExit(f"{directory} 裡沒有任何 __def.png")
    return out


def dataset_sources(root: Path) -> dict:
    """未防禦的那一組：每類一個子目錄，檔名即影像名。"""
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name in ("masks", "headmasks", "overview"):
            continue
        for img in sorted(d.glob("*.png")):
            out[img.stem] = img
    if not out:
        raise SystemExit(f"{root} 底下找不到任何影像")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--defended", type=Path,
                        help="防禦圖目錄（防禦程式的輸出）")
    source.add_argument("--data", type=Path,
                        help="資料集根目錄，淨化未防禦的原圖")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", default=None,
                        help="寫進 CSV 的條件名；預設取 --defended 的目錄名，"
                             "或未防禦時的 `undefended`")
    args = parser.parse_args()

    if args.defended is not None:
        sources = defended_sources(args.defended)
        condition = args.condition or args.defended.name
    else:
        sources = dataset_sources(args.data)
        condition = args.condition or "undefended"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, strength in PURIFIERS:
        tag = label(kind, strength)
        purifier = Purifier(kind, strength)
        target = args.out / tag
        target.mkdir(parents=True, exist_ok=True)
        for name, path in sources.items():
            x01 = load_image_tensor(path, device, size=RESOLUTION)
            with torch.no_grad():
                y01 = purifier.evaluate(x01).clamp(0, 1)
            save_image(y01, target / f"{name}__def.png")
            # 這兩個數量的是「淨化把圖動了多少」，不是防禦效果；不需要任何模型。
            mse = float(((y01 - x01) ** 2).mean())
            rows.append({
                "condition": condition, "purifier": tag,
                "kind": kind, "strength": strength,
                "image": name,
                "source_png": path.as_posix(),
                "output_png": (target / f"{name}__def.png").as_posix(),
                "psnr_vs_source": round(float("inf") if mse == 0
                                        else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item(), 4),
                "rms_vs_source": round(mse ** 0.5, 6),
                "geometric": kind in ("crop_resize", "rotate"),
            })
        print(f"[DONE] {condition:16s} {tag:16s} {len(sources)} 張", flush=True)
    write_csv(args.out / "purified.csv", rows)
    print(f"[ALLDONE] {args.out / 'purified.csv'}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
