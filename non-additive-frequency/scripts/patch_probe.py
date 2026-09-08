"""補丁的零號實驗：**IP2P 有沒有長程通道**。不最佳化，只放固定內容。

問的是什麼
────────────────────────────────────────────────────────────────────
對抗補丁（arXiv:1712.09665）能贏，是因為分類器只輸出**一個全域決策**，
補丁在那個唯一的輸出上把整張圖的證據壓過去。IP2P 沒有這個東西：它輸出的是
逐位置的 ε，影像條件是逐 token 拼進 UNet 的。補丁要影響**主體那一塊**的
編輯，只能走 self-attention 的長程通道。

所以在寫任何最佳化之前，要先問一個可以證偽的問題：

    在主體之外放一塊高幅度的固定內容，主體那一塊的編輯會不會改變？

- 主體內位移貼著零 → 通道不存在，這一族在「不動主體」的前提下不可能成立。
- 主體內位移明顯抬起來 → 才值得寫 `PatchParam` 去最佳化它。

**本批不做任何最佳化，也不宣稱任何效果。** 依 CLAUDE.md 第八節，這裡只把
數據與影像擺出來。

三種內容、三個面積、三個位置
────────────────────────────────────────────────────────────────────
內容刻意**避開顏色語意**——「高飽和物件撞上顏色指令」是語意誘餌那一族已經
測過的機制（`runs/ip2p_decoy*`），混進來會讓兩件事分不開。三種內容是一個
頻率與自然度的階梯：

| 內容 | 是什麼 | 測的是 |
|---|---|---|
| `noise` | 逐像素均勻噪聲 | 高頻極端，L∞ 最大 |
| `gray` | 純中灰 0.5 | 低頻極端，語意為零 |
| `photo` | 另一張資料集影像的同位置像素 | 自然內容、語意外來 |

**面積不取全圖的固定比例，而是由畫面實際的空位決定。** 兩張目標影像的主體
遮罩都很大且分散（盆栽人核心 44.5%／平均 52.6%，瑪利歐 49.1%／56.4%），
實測連 128×128 的方塊在盆栽人上都找不到與遮罩零重疊的位置。固定比例會讓
大部分格子變成「放不下」，量不到東西。故先**二分搜尋出最大的合法邊長**
`s_max`，再取 `s_max` 的 1.0／0.7／0.5 三個尺寸。`s_max` 逐列寫進 CSV。

位置有三個：

- `far`／`near`：離主體重心最遠與最近的合法方形位置，由**確定性搜尋**選出。
- `complement`：不是方形，而是**整個遮罩補集**。這一格是這條路的上界——
  若把主體以外的每一個像素都換成噪聲，主體那一塊的編輯仍然不動，那麼
  長程通道就是不存在的，不必再談形狀與最佳化。

**補丁一律不碰主體遮罩。** 合法的定義是區域內遮罩逐像素為零——羽化帶也算在
內，所以主體逐位元不動。找不到合法位置時記 `placeable=False` 並跳過那一格，
**這是結果不是錯誤**。

讀數
────────────────────────────────────────────────────────────────────
主讀數是分區位移（`src/metrics/regional.py`）：全圖／主體內／主體外。
全圖那一欄會很大——補丁本身就穿過編輯——**所以全圖位移在這一批不可解讀**，
要看的是主體內那一欄。

同時量 `image_guidance` 的固定評估（`L_ig`），使這一批可以直接放進
`scripts/ig_probe.py` 的同一張圖上。

裁切依協定換參照
────────────────────────────────────────────────────────────────────
`crop_resize` 屬 `GEOMETRIC_KINDS`，故

    effect = LPIPS( 編輯(p(原圖)), 編輯(p(補丁圖)) )      地板 ≡ 0

`reference` 欄逐列記下用的是哪一種參照。**不可與 identity 那幾列直接相減。**

用法：

    python scripts/patch_probe.py --out runs/patch_probe/dev1 --ig-zt diffuse_src
        --images task_attr_mod_color_11699 task_attr_mod_color_6205
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402

from src.defense.image_guidance_loss import ZT_MODES, make_image_guidance_loss  # noqa: E402
# 幾何helper 與 `PatchParam` 共用同一份實作：兩份會在其中一份改動時悄悄分岔，
# 而「補丁碰到主體」這種錯誤在數字上看不出來。
from src.defense.patch_param import (  # noqa: E402
    choose, largest_legal_side, placements, rect_support,
)
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
CONTENTS = ("noise", "gray", "photo")
# 最大合法邊長的倍數。1.0 是畫面上放得下的最大方塊。
SIZE_SCALES = (1.0, 0.7, 0.5)
PLACEMENTS = ("far", "near", "complement")
STRIDE = 16                                 # 候選位置的搜尋步長
MIN_SIDE = 32                               # 比這更小的方塊不值得跑一次編輯
# `photo` 的來源。固定一張與兩個目標無關的影像，逐列記進 CSV。
PHOTO_SOURCE = "task_env_weather_112463"
PATCH_SEED = 20260902


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--catalogue", type=Path,
                    default=Path("data/decoy_catalogue.yaml"))
    ap.add_argument("--out", type=Path, required=True,
                    help="輸出目錄。**每寫一列是整份重寫 CSV**，"
                         "同一個目錄不可有第二個 process")
    ap.add_argument("--contents", nargs="+", default=list(CONTENTS),
                    choices=list(CONTENTS))
    ap.add_argument("--size-scales", nargs="+", type=float,
                    default=list(SIZE_SCALES),
                    help="最大合法邊長的倍數。面積由畫面的空位決定，"
                         "不是全圖的固定比例")
    ap.add_argument("--placements", nargs="+", default=list(PLACEMENTS),
                    choices=list(PLACEMENTS))
    ap.add_argument("--min-side", type=int, default=MIN_SIDE)
    ap.add_argument("--photo-source", default=PHOTO_SOURCE)
    ap.add_argument("--patch-seed", type=int, default=PATCH_SEED)
    ap.add_argument("--crop", type=float, default=0.10,
                    help="裁切算子的比例。0 表示不跑裁切那一半")
    ap.add_argument("--ig-zt", choices=ZT_MODES, default=None,
                    help="**必填**，理由同 scripts/ig_probe.py")
    ap.add_argument("--ig-t-min", type=int, default=1)
    ap.add_argument("--ig-t-max", type=int, default=1000)
    ap.add_argument("--eval-draws", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=99991)
    ap.add_argument("--edit-steps", type=int, default=100)
    ap.add_argument("--text-guidance", type=float, default=7.5)
    ap.add_argument("--image-guidance", type=float, default=1.5)
    ap.add_argument("--edit-seed", type=int, default=20260812)
    ap.add_argument("--subject-mask-threshold", type=float, default=0.30)
    ap.add_argument("--subject-mask-dilate", type=int, default=16)
    ap.add_argument("--subject-mask-feather", type=int, default=24)
    return ap


def check_args(args) -> None:
    if args.ig_zt is None:
        raise SystemExit(
            "--ig-zt 必填：兩個 z_t 的抽法都是近似，填錯不會有症狀，"
            f"只會量到另一條軸。已存的批次全部是 diffuse_src。{ZT_MODES}")
    if not 1 <= args.ig_t_min <= args.ig_t_max:
        raise SystemExit(f"需要 1 <= --ig-t-min <= --ig-t-max，收到 "
                         f"{args.ig_t_min}／{args.ig_t_max}")
    for s in args.size_scales:
        if not 0.0 < s <= 1.0:
            raise SystemExit(
                f"--size-scales 是最大合法邊長的倍數，須在 (0,1]，收到 {s}")
    if not args.catalogue.exists():
        raise SystemExit(f"找不到主體名詞目錄 {args.catalogue}")


def even(s: float) -> int:
    """取整到偶數，使裁切與降取樣不產生半格。"""
    n = int(round(s))
    return max(0, n - (n % 2))


def full_content(kind: str, ref: torch.Tensor, photo: torch.Tensor,
                 seed: int) -> torch.Tensor:
    """整張畫面大小的內容場，[0,1]。支撐由 `paste` 的遮罩挑，不在這裡裁。

    整張產生再挑支撐，使方形格與 `complement` 格看到的是**同一個內容場**，
    兩者的差只來自支撐的形狀。
    """
    if kind == "noise":
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.rand(ref.shape, generator=g).to(
            device=ref.device, dtype=ref.dtype)
    if kind == "gray":
        return torch.full_like(ref, 0.5)
    if kind == "photo":
        if photo.shape != ref.shape:
            raise ValueError(
                f"photo 的形狀 {tuple(photo.shape)} 與目標 "
                f"{tuple(ref.shape)} 不同，逐位置取代會對不齊")
        return photo.to(device=ref.device, dtype=ref.dtype)
    raise ValueError(f"未知的補丁內容 {kind!r}")


def complement_support(mask: torch.Tensor) -> torch.Tensor:
    """遮罩補集的支撐：**逐像素等於零**才算，羽化帶不算。"""
    return mask <= 0.0


def paste(x01: torch.Tensor, content: torch.Tensor,
          support: torch.Tensor) -> torch.Tensor:
    """把內容放進支撐。支撐之外**逐位元**保留原圖。"""
    s = support.to(device=x01.device)
    return torch.where(s, content.to(x01), x01)


def main() -> None:
    args = build_parser().parse_args()
    check_args(args)

    from apa_baseline import load_dataset

    # DEF「防禦圖存在不代表資料集裡還有那張影像」：清單先對照資料集，
    # 再載入任何權重。
    dataset = {d["name"]: d for d in load_dataset(args.data)}
    missing = [n for n in list(args.images) + [args.photo_source]
               if n not in dataset]
    if missing:
        raise SystemExit(f"{args.data} 底下沒有這些影像：{missing}")

    args.out.mkdir(parents=True, exist_ok=True)

    from src.defense.subject_mask import mask_stats, subject_mask
    from src.metrics.regional import RegionalLPIPS, split_displacement
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper
    from src.purify.ops import GEOMETRIC_KINDS, Purifier

    ip2p = IP2PWrapper(dtype=torch.float32)
    dev = ip2p.device
    suite = MetricSuite(device=dev)
    regional = RegionalLPIPS(suite.lpips_module)

    import yaml
    objects = (yaml.safe_load(
        args.catalogue.read_text(encoding="utf-8")) or {}).get("objects") or {}

    def edit(x):
        return ip2p.edit(x, item["prompt"], seed=args.edit_seed,
                         steps=args.edit_steps, s_t=args.text_guidance,
                         s_i=args.image_guidance)

    ops = [("identity", None)]
    if args.crop > 0:
        ops.append((f"crop_resize{args.crop:g}",
                    Purifier("crop_resize", strength=args.crop)))

    photo = load_image_tensor(
        dataset[args.photo_source]["path"], dev, size=RESOLUTION)

    rows = []
    for name in args.images:
        item = dataset[name]
        x = load_image_tensor(item["path"], dev, size=RESOLUTION)
        if name not in objects:
            raise SystemExit(
                f"{args.catalogue} 的 objects 裡沒有 {name} 的主體名詞。"
                "分區讀數需要知道要保護的是誰，不可猜。")
        texts = objects[name]
        texts = [texts] if isinstance(texts, str) else list(texts)
        m = subject_mask(x, texts,
                         threshold=args.subject_mask_threshold,
                         dilate=args.subject_mask_dilate,
                         feather=args.subject_mask_feather)
        stats = mask_stats(m)
        print(f"\n{name}  指令「{item['prompt']}」\n  遮罩 {texts} {stats}",
              flush=True)

        loss = make_image_guidance_loss(
            ip2p, zt_mode=args.ig_zt, x_clean=x,
            t_min=args.ig_t_min, t_max=args.ig_t_max)
        fixed = loss.make_fixed(args.eval_draws, args.eval_seed)
        with torch.no_grad():
            l0 = float(fixed(x))

        vutils.save_image(x, args.out / f"{name}__orig.png")
        # 每個算子的參照編輯各算一次，跨設定共用（幾何類換參照的成本就在這裡）。
        ref_edit = {}
        for op_name, op in ops:
            src = x if op is None else op.evaluate(x)
            ref_edit[op_name] = edit(src)
            vutils.save_image(ref_edit[op_name],
                              args.out / f"{name}__{op_name}__edit_orig.png")
        print(f"  參照編輯完成（{len(ops)} 個算子）", flush=True)

        # 面積由畫面實際的空位決定，不是全圖的固定比例。
        s_max = largest_legal_side(m, min_side=args.min_side)
        print(f"  最大合法邊長 s_max={s_max}"
              f"（佔全圖面積 {s_max * s_max / (RESOLUTION ** 2):.3f}）",
              flush=True)

        # (標籤, 支撐, 位置說明) 的清單。方形格與補集格走同一條路。
        cells = []
        for scale in args.size_scales:
            side = even(s_max * scale)
            cands, centre = placements(m, side) if side >= args.min_side \
                else ([], (0.0, 0.0))
            seen = set()
            for mode in args.placements:
                if mode == "complement":
                    continue
                pos = choose(cands, centre, mode)
                # 該邊長只剩一個合法位置時 far 與 near 是同一格。跑兩次會產生
                # 兩列逐位元相同的結果，讀表的人分不出那是重複還是巧合。
                if pos is not None and pos in seen:
                    print(f"  s{scale:g}_{mode} 與已排入的位置相同 {pos}，不重複跑",
                          flush=True)
                    continue
                if pos is not None:
                    seen.add(pos)
                cells.append((f"s{scale:g}_{mode}", side, pos))
        if "complement" in args.placements:
            cells.append(("complement", 0, None))

        for content in args.contents:
            field = full_content(content, x, photo, args.patch_seed)
            for label, side, pos in cells:
                cond = f"patch_{content}_{label}"
                if label == "complement":
                    support = complement_support(m)
                    top, left = "", ""
                elif pos is None:
                    # 該邊長沒有與遮罩零重疊的位置。這是結果不是錯誤。
                    rows.append({
                        "image": name, "condition": cond, "content": content,
                        "side": side, "placement": label, "s_max": s_max,
                        "placeable": False, "top": "", "left": "",
                        "op": "", "reference": "",
                        "instruction": item["prompt"], **stats,
                    })
                    print(f"  {cond:34s}沒有與遮罩零重疊的位置，跳過",
                          flush=True)
                    write_csv(args.out / "results.csv", rows)
                    continue
                else:
                    top, left = pos
                    support = rect_support(x.shape, top, left, side)
                t0 = time.time()
                xd = paste(x, field, support)
                patch_area = float(support.to(torch.float32).mean())
                vutils.save_image(xd, args.out / f"{name}__{cond}__def.png")
                with torch.no_grad():
                    lv = float(fixed(xd))
                dist = split_displacement(regional, x, xd, m)
                fid = suite.pairwise(x, xd)

                for op_name, op in ops:
                    src = xd if op is None else op.evaluate(xd)
                    ed = edit(src)
                    vutils.save_image(
                        ed, args.out / f"{name}__{cond}__{op_name}__edit_def.png")
                    eo = ref_edit[op_name]
                    disp = split_displacement(regional, eo, ed, m)
                    kind = "identity" if op is None else op.kind
                    rows.append({
                        "image": name, "condition": cond,
                        "content": content, "side": side,
                        "placement": label, "s_max": s_max,
                        "patch_area_fraction": round(patch_area, 5),
                        "placeable": True, "top": top, "left": left,
                        "op": op_name,
                        # 幾何類換參照：編輯(p(原圖)) 對 編輯(p(補丁圖))，
                        # 地板恆為 0。非幾何類不換。判定走 Purifier.kind，
                        # 不比對標籤字串。
                        "reference": ("edit_of_purified_original"
                                      if kind in GEOMETRIC_KINDS
                                      else "edit_of_original"),
                        "instruction": item["prompt"],
                        "photo_source": (args.photo_source
                                         if content == "photo" else ""),
                        "patch_seed": (args.patch_seed
                                       if content == "noise" else ""),
                        **stats,
                        "subject_mask_threshold": args.subject_mask_threshold,
                        "subject_mask_dilate": args.subject_mask_dilate,
                        "subject_mask_feather": args.subject_mask_feather,
                        "ig_zt": args.ig_zt, "ig_t_min": args.ig_t_min,
                        "ig_t_max": args.ig_t_max,
                        "eval_draws": args.eval_draws,
                        "eval_seed": args.eval_seed,
                        "ig_loss_original": round(l0, 8),
                        "ig_loss": round(lv, 8),
                        "ig_ratio_to_original": round(lv / l0, 6),
                        "fid_lpips": round(float(fid["lpips"]), 5),
                        "fid_dists": round(float(fid["dists"]), 5),
                        "fid_psnr": round(float(fid["psnr"]), 4),
                        "dist_subject": round(dist["lpips_subject"], 5),
                        "dist_background": round(dist["lpips_background"], 5),
                        # 全圖位移在這一批**不可解讀**：補丁本身穿過編輯。
                        "disp_full": round(disp["lpips_full"], 5),
                        "disp_subject": round(disp["lpips_subject"], 5),
                        "disp_background": round(disp["lpips_background"], 5),
                        "edit_steps": args.edit_steps,
                        "s_t": args.text_guidance, "s_i": args.image_guidance,
                        "edit_seed": args.edit_seed,
                        "seconds": round(time.time() - t0, 1),
                    })
                    print(f"  {cond:30s}{op_name:16s}"
                          f"主體位移 {disp['lpips_subject']:.4f}  "
                          f"背景 {disp['lpips_background']:.4f}  "
                          f"主體失真 {dist['lpips_subject']:.4f}  "
                          f"L_ig {lv / l0:.3f}×", flush=True)
                    write_csv(args.out / "results.csv", rows)

    write_csv(args.out / "results.csv", rows)
    print(f"\n寫出 {args.out / 'results.csv'}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
