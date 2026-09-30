"""由模型畫出 inpainting 的逐影像遮罩，並輸出供逐張看圖的對照版面。

遮罩指的是什麼
────────────────────────────────────────────────────────────────────
場景二的攻擊是「保留主體、改掉其餘區域」（`prompts.yaml` 的 `prompts[1]`，
例如 *a dog in the park*）。在這個威脅模型下，防禦方選定要保護的內容 `c_a`，
而 `c_a` 必須落在**重繪區之外**——否則被保護的東西自己就被模型重畫掉了。

於是本腳本產生的 PNG 採 Stable Diffusion inpainting 的標準極性：

    白（255）＝ 要重繪的區域 ＝ 主體以外
    黑（0）  ＝ 保留的區域   ＝ 主體（含外擴的安全邊）

主體遮罩怎麼來
────────────────────────────────────────────────────────────────────
`src/defense/subject_mask.subject_mask`：CLIPSeg（`CIDAS/clipseg-rd64-refined`）
以一句文字指出主體。文字取 `prompts.yaml` 每類的 `content` 欄，也就是 `c_a`
本身——**威脅模型的前提是防護對象已知**，所以用 `c_a` 當提示詞不算作弊。

呼叫時 `feather=0`。羽化是為了合成時的軟過渡；inpainting 的遮罩要二值，
軟邊會在寫成 8-bit PNG 時被門檻切掉，留著只會讓「寫出去的是什麼」變得不明確。
`dilate` 保留模組預設：分割的邊緣切在輪廓上，而重繪會沿輪廓外側動手，
外擴一圈是讓主體與重繪區之間有安全邊。

面積出界時 `subject_mask` 會拋錯而不是靜默接受，本腳本不攔截——一個空的
主體遮罩會讓整張圖都變成重繪區，那正是要避免的失效模式。

看圖用的版面
────────────────────────────────────────────────────────────────────
`{out}/overview/{類別}.png`：每張影像一列，三欄為原圖、主體區塊著色的疊圖、
二值遮罩本身。**遮罩準不準由看圖判定**，本腳本只把圖與數擺出來。
`{out}/provenance.json` 逐張記下模型、參數與面積。

用法
    python scripts/make_masks.py --data data/lo_aligned
    python scripts/make_masks.py --data data/lo_aligned --images cat_00 man_02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import yaml  # noqa: E402
from src.defense.subject_mask import (  # noqa: E402
    CLIPSEG_REPO, DILATE, THRESHOLD, subject_mask,
)
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
# **本專案的 inpainting 遮罩用 4 px，不是 `subject_mask.DILATE` 的 16。**
# 16 px 在主體旁邊留下一圈**原始背景**，那是很強的條件線索，模型會順著它把
# 原本的背景延伸下去——實測同一句指令在 16 px 下什麼都不出現，4 px 下醫院
# 走廊、健身器材、積雪枝幹都畫得出來（`runs/edit_preflight/` 的四臂對照）。
# 不改 `subject_mask.DILATE` 本身：那個預設是給合成用的（寧可多保留一點主體），
# 與 inpainting 的需求相反，而且貼片線也在用它。
MASK_DILATE = 4
# 疊圖的主體著色。紅色，半透明。
TINT = (1.0, 0.25, 0.25)
TINT_ALPHA = 0.45


def load_subjects(root: Path) -> list:
    """只讀「每一類要保護的是什麼」，**不讀編輯指令**。

    原本是借 `baseline_run.load_dataset`，但那一支要求逐類別的 `prompts` 鍵，
    而新的資料集把指令按場景放在 `edits` 底下，借用會 KeyError。更根本的理由是
    **遮罩不該依賴攻擊指令**：遮罩由防禦方選定的 `content`（c_a）決定，
    指令是攻擊方寫的，兩者在威脅模型裡屬於不同的人。
    """
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for cls in sorted(k for k in spec if k != "edits"):
        entry = spec[cls]
        if not isinstance(entry, dict) or "content" not in entry:
            continue
        for img in sorted((root / cls).glob("*.png")):
            out.append({"name": img.stem, "class": cls, "path": img,
                        "content": entry["content"]})
    if not out:
        raise SystemExit(f"{root} 底下找不到任何影像，或 prompts.yaml 缺 content")
    return out


def inpaint_mask(subject: torch.Tensor) -> torch.Tensor:
    """主體遮罩 → inpainting 遮罩。1 = 重繪，即主體的補集。

    輸入必須已經是二值的（`subject_mask(..., feather=0)` 的輸出）；出現中間值
    表示呼叫端羽化了，那會讓寫出去的 PNG 到底重繪哪裡變得不明確，故拋錯。
    """
    uniq = torch.unique(subject)
    if not torch.isin(uniq, torch.tensor([0.0, 1.0], device=uniq.device)).all():
        raise ValueError(
            f"主體遮罩不是二值的（出現 {uniq.numel()} 種值）。"
            "inpainting 的遮罩要二值，請以 feather=0 呼叫 subject_mask。")
    return 1.0 - subject


def _overlay(x01: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
    """把主體區塊著色疊在原圖上，供肉眼比對邊界。"""
    tint = torch.tensor(TINT, device=x01.device,
                        dtype=x01.dtype).view(1, 3, 1, 1)
    a = subject * TINT_ALPHA
    return x01 * (1.0 - a) + tint * a


def _row(x01: torch.Tensor, subject: torch.Tensor,
         mask: torch.Tensor) -> torch.Tensor:
    """一列三欄：原圖、疊圖、二值遮罩。"""
    return torch.cat([x01, _overlay(x01, subject), mask.repeat(1, 3, 1, 1)],
                     dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--out", type=Path, default=None,
                    help="預設是 {data}/masks")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--dilate", type=int, default=MASK_DILATE,
                    help="主體往外擴幾個像素，即主體與重繪區之間的安全邊")
    ap.add_argument("--device", default="cpu",
                    help="CLIPSeg 是 rd64，CPU 即可；GPU 工作一律送遠端")
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--overview-side", type=int, default=256,
                    help="看圖版面每一格的邊長")
    args = ap.parse_args()

    out = args.out or (args.data / "masks")
    out.mkdir(parents=True, exist_ok=True)
    (out / "overview").mkdir(exist_ok=True)

    items = load_subjects(args.data)
    if args.images:
        keep = set(args.images)
        items = [d for d in items if d["name"] in keep]
    if not items:
        raise SystemExit("沒有符合的影像")

    device = torch.device(args.device)
    rows, records = {}, []
    for item in items:
        x01 = load_image_tensor(item["path"], device, size=RESOLUTION)
        subject = subject_mask(x01, item["content"], threshold=args.threshold,
                               dilate=args.dilate, feather=0, device=device)
        mask = inpaint_mask(subject)

        path = out / f"{item['name']}.png"
        _write_png(mask, path)
        records.append({
            "name": item["name"], "image": str(item["path"]).replace("\\", "/"),
            "mask": str(path).replace("\\", "/"), "text": item["content"],
            "model": CLIPSEG_REPO, "threshold": args.threshold,
            "dilate": args.dilate,
            "subject_area": round(float(subject.mean()), 5),
            "repaint_area": round(float(mask.mean()), 5),
        })
        cls = item["name"].rsplit("_", 1)[0]
        side = args.overview_side
        rows.setdefault(cls, []).append(
            F.interpolate(_row(x01, subject, mask), size=(side, 3 * side),
                          mode="bilinear", align_corners=False))
        print(f"{item['name']}: 主體 {subject.mean():.3f} / "
              f"重繪 {mask.mean():.3f} -> {path}")

    for cls, rs in rows.items():
        _write_png(torch.cat(rs, dim=-2), out / "overview" / f"{cls}.png")
    (out / "provenance.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"看圖版面 {out / 'overview'}，出處 {out / 'provenance.json'}")


def _write_png(x: torch.Tensor, path: Path) -> None:
    import torchvision.utils as vutils
    vutils.save_image(x.clamp(0, 1), path)


if __name__ == "__main__":
    main()
