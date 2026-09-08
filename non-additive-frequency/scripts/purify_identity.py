"""抗淨化：攻擊方先把防禦圖淨化一遍，再下指令編輯。**只讀已存的防禦圖。**

現行威脅模型的主讀數是身分（編輯後還認不認得出那個人），而
`scripts/phase_retention.py` 量的是 LPIPS 位移。兩者的協定相同、讀數不同，
故本檔只做「淨化 → 編輯 → 存圖 → 存成對指標」，身分讀數事後由
`scripts/identity_probe.py` 在 CPU 上補算——那一段不佔卡。

輸出的檔名刻意與派工腳本一致，`identity_probe.py` 不必改就讀得到：

    {out}/{arm}_{cat}_{pur}/{image}__orig.png
    {out}/{arm}_{cat}_{pur}/{image}__{cond}__def.png        淨化後的防禦圖
    {out}/{arm}_{cat}_{pur}/{image}__{cond}__edit_orig.png  參照
    {out}/{arm}_{cat}_{pur}/{image}__{cond}__edit_def.png   攻擊輸出

## 參照依算子分兩種（沿用 `docs/EVALUATION.md` 的協定）

    幾何類    edit_orig := 編輯(p(原圖))
    其餘      edit_orig := 編輯(原圖)      ← 直接沿用來源目錄已存的那一張

幾何類就是 `src/purify/ops.py` 的 `GEOMETRIC_KINDS`，**判定用 `Purifier.kind`
不是標籤字串**。理由是取景與像素格點：`crop_resize` 是繞中心的 1.2488× 放大、
`rotate` 讓四角離開畫面，就算完全沒有防禦，`編輯(p(原圖))` 與 `編輯(原圖)`
之間也會差很多，讀數會被取景差異吃光。其餘算子不換——它們的地板反映的是算子
對**影像內容**的破壞，那正是扣地板要扣掉的東西。

非幾何類的 `edit_orig` 直接複製來源目錄裡已存的那一張，不重跑：同一組
`seed`／`steps`／引導強度下編輯是確定性的，重跑只會多花一次前向，而且換了卡
還可能差最後幾個位元。

## `--arm floor`：空白地板

`floor` 這個臂的「防禦圖」就是原圖本身，故量到的是**淨化算子自己造成的位移**，
與有沒有防禦無關。沒有這一格，「淨化之後某個臂的絕對位移比較大」就無法排除
「該算子本來就把編輯推得比較開」這個平庸解釋。

幾何類在 `floor` 下兩側完全相同（同算子、同輸入、同種子），故 `edit_def` 與
`edit_orig` 逐位元相同、成對指標恰為 0。量到非 0 表示編輯不是確定性的或種子
沒對齊——那是量測本身壞了，本檔直接拋錯。

## 為什麼一定要有隨機臂

FaceLock（CVPR 2025）、EditShield（ECCV 2024）、FaceShield（ICCV 2025）報的
抗淨化都是絕對值，既沒有空白地板也沒有等幾何的隨機對照
（`docs/reference/SURVEY_IDENTITY_EDITING.md` §6）。本專案已量到淨化之後
最佳化相對同幾何隨機雜訊的優勢幾乎消失（裁切 1.13、jpeg→resize 1.04）。
沒有 `rand` 臂就分不出「防禦守住了」與「衣服上有一塊很吵的東西」。

用法：
    python scripts/purify_identity.py --arm tint \\
        --cells "runs/ip2p_content_constraint/tint_clothing_task_*" \\
        --purifiers identity jpeg90 jpeg75 blur1.5 rotate10 \\
        --out runs/ip2p_purify_identity
"""

from __future__ import annotations

import argparse
import csv
import glob
import shutil
import sys
import time
from pathlib import Path

import torch
import torchvision.utils as vutils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.ip2p import (  # noqa: E402
    IP2P_IMAGE_GUIDANCE, IP2P_SEED, IP2P_STEPS, IP2P_TEXT_GUIDANCE, IP2PWrapper,
)
from src.purify import ops as purify_ops  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

# 文獻上這一族實際用的算子與強度，逐項出處見
# `docs/reference/SURVEY_IDENTITY_EDITING.md` §5。**強度不自己另立新數字**，
# 沿用論文明載的值；本專案既有格點沒有的兩個（blur σ=1.5、jpeg 80）在此補上。
LITERATURE_SET = (
    "identity",           # 分母，不可排除
    "jpeg90",             # FaceLock
    "jpeg80",             # EditShield 的 JPEG Compression countermeasure
    "jpeg75",             # FaceLock；亦即真實世界串接協定的 C
    "jpeg60",             # FaceLock
    "blur1.5",            # FaceLock「k=5, σ=1.5」的 σ；本專案的核由 σ 決定
    "rotate10",           # FaceLock「random rotation between (-10, 10)」
    "crop_resize0.1",     # DIA 的 10%
    "jpeg_then_resize75",  # arXiv:2604.23688 的 C&R 串接
    "resize_only",        # FaceShield 的降採樣再還原
    "quantize8",          # FaceShield 的 3-bit（8 階）
    "adverse_cleaner",    # 本專案既有，導向濾波
)


#: 旋轉的種子與其餘隨機算子分開。`--purify-seed` 的 0 在 `degrees=10` 上抽到
#: −0.075°，512 px 影像最遠角落位移 0.33 px，整張圖都在次像素量級——那一格
#: 量到的是恆等映射，卻掛著 `rotate10` 的名字進表。1 抽到 5.15°，接近
#: `U(−10,10)` 的 RMS（10/√3 ≈ 5.77），故取 1；其餘算子的種子留在 0，
#: 使既有 CSV 仍逐格可比。角度不是自由參數，抽到什麼算什麼，
#: 只擋退化的那一種，見 `build_purifier` 的守門。
ROTATE_SEED_DEFAULT = 1


def build_purifier(lbl: str, seed: int, rotate_seed: int = ROTATE_SEED_DEFAULT):
    """由標籤建 `Purifier`。標籤格式與 `phase_retention.label()` 相同：
    `kind` 或 `kind{strength}`，故 `kind_of_label` 是唯一的解析入口——
    不用字首猜，`jpeg_then_resize75` 不會被 `jpeg` 吃掉。

    旋轉走 `rotate_seed`，並且**抽到接近零的角度就拋錯**：那樣的一格與
    `identity` 欄無從分辨，兩欄並列時看不出異常，只看得到「旋轉不傷防禦」。
    退化的抽樣要在派工前擋下來，不要花完機時才在表上發現。
    """
    kind = purify_ops.kind_of_label(lbl)
    rest = lbl[len(kind):]
    strength = float(rest) if rest else 0.0
    if kind == "rotate":
        deg = strength if strength else purify_ops.ROTATE_DEGREES_FACELOCK
        angle = purify_ops.rotate_angle(deg, rotate_seed)
        if abs(angle) < purify_ops.ROTATE_DEGENERATE_DEGREES:
            raise SystemExit(
                f"--rotate-seed {rotate_seed} 在 {deg}° 上抽到 {angle:+.3f}°，"
                f"小於 {purify_ops.ROTATE_DEGENERATE_DEGREES}°：512 px 影像各處位移"
                f"皆在次像素量級，這一格量到的是恆等映射而不是旋轉。換一個種子。")
        return purify_ops.Purifier(kind, strength, seed=rotate_seed)
    return purify_ops.Purifier(kind, strength, seed=seed)


def read_cell(cell: Path) -> dict:
    """一個派工格：原圖、防禦圖、已存的 `編輯(原圖)`、指令、類別。

    五者缺一就拋錯，不猜也不跳過——靜默跳過會讓某個算子的分母悄悄少幾張，
    而表上只看得到平均變了，不會拋錯。
    """
    res = cell / "results.csv"
    if not res.exists():
        raise SystemExit(f"{cell} 沒有 results.csv")
    rows = list(csv.DictReader(res.open(encoding="utf-8")))
    if len(rows) != 1:
        raise SystemExit(f"{cell}/results.csv 有 {len(rows)} 列，預期恰好 1 列")
    row = rows[0]
    image, cond = row["image"], row["condition"]
    paths = {
        "orig": cell / f"{image}__orig.png",
        "def": cell / f"{image}__{cond}__def.png",
        "edit_orig": cell / f"{image}__{cond}__edit_orig.png",
    }
    for k, p in paths.items():
        if not p.exists():
            raise SystemExit(f"{cell} 缺 {p.name}")
    return {
        "image": image, "condition": cond,
        "instruction": row["instruction"],
        "category": row.get("attack_category", ""),
        **{k: str(v) for k, v in paths.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True,
                    help="臂名。`floor` 是保留字：防禦圖改用原圖本身，量空白地板")
    ap.add_argument("--cells", required=True,
                    help="派工格目錄的 glob，例如 "
                         "'runs/ip2p_content_constraint/tint_clothing_task_*'")
    ap.add_argument("--out", type=Path, default=Path("runs/ip2p_purify_identity"))
    ap.add_argument("--purifiers", nargs="+", default=list(LITERATURE_SET),
                    help=f"淨化算子標籤。預設是文獻組：{' '.join(LITERATURE_SET)}")
    ap.add_argument("--purify-seed", type=int, default=0,
                    help="隨機類算子（noise、rotate）的種子。**兩側必須同值**，"
                         "否則量到的是兩個不同的取景或兩份不同的噪聲")
    ap.add_argument("--rotate-seed", type=int, default=ROTATE_SEED_DEFAULT,
                    help="旋轉專用的種子，與 --purify-seed 分開。抽到的角度小於 "
                         f"{purify_ops.ROTATE_DEGENERATE_DEGREES}° 會直接拋錯——"
                         "那一格與 identity 欄分不開")
    ap.add_argument("--edit-steps", type=int, default=IP2P_STEPS)
    ap.add_argument("--text-guidance", type=float, default=IP2P_TEXT_GUIDANCE)
    ap.add_argument("--image-guidance", type=float, default=IP2P_IMAGE_GUIDANCE)
    ap.add_argument("--edit-seed", type=int, default=IP2P_SEED)
    ap.add_argument("--skip-existing", action="store_true",
                    help="輸出已有 edit_def 的格直接跳過，供中斷後續跑")
    ap.add_argument("--preflight", action="store_true",
                    help="只走參數與檔案檢查，不載權重、不編輯")
    args = ap.parse_args()

    if "identity" not in args.purifiers:
        raise SystemExit("--purifiers 必須包含 identity，它是保留率的分母")

    # **只取目錄**。派工腳本把每一格的 stdout 存成 `<格名>.log`，就落在格目錄
    # 旁邊，於是 `..._task_*` 這種 glob 會同時匹配到目錄與它的日誌檔。
    cells = sorted(p for p in (Path(q) for q in glob.glob(args.cells)) if p.is_dir())
    if not cells:
        raise SystemExit(f"--cells {args.cells!r} 沒有匹配到任何目錄")
    items = [read_cell(c) for c in cells]
    cats = sorted({it["category"] for it in items})
    if len(cats) != 1:
        raise SystemExit(f"一次只跑一個攻擊類別，這批有 {cats}")
    cat = cats[0]

    purifiers = [(lbl, build_purifier(lbl, args.purify_seed, args.rotate_seed))
                 for lbl in args.purifiers]
    unavailable = [lbl for lbl, p in purifiers if not p.available]
    if unavailable:
        raise SystemExit(f"相依不齊的算子：{unavailable}。"
                         "**不靜默跳過**——少一個算子表上就少一欄")

    print(f"[purify_identity] arm={args.arm} cat={cat} "
          f"{len(items)} 格 × {len(purifiers)} 算子", flush=True)
    n_geo = sum(1 for _, p in purifiers if p.kind in purify_ops.GEOMETRIC_KINDS)
    print(f"  幾何類 {n_geo} 個（參照改用 編輯(p(原圖))），"
          f"其餘 {len(purifiers) - n_geo} 個沿用已存的 編輯(原圖)", flush=True)
    if args.preflight:
        for lbl, p in purifiers:
            print(f"  {lbl:20s} kind={p.kind:18s} strength={p.strength:g} "
                  f"geometric={p.kind in purify_ops.GEOMETRIC_KINDS}")
        for it in items:
            print(f"  {it['image']:34s} {it['instruction']}")
        print("[preflight] 通過，未載入權重")
        return

    ip2p = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=ip2p.device)

    def edit(x01, instruction):
        return ip2p.edit(x01, instruction, seed=args.edit_seed,
                         steps=args.edit_steps, s_t=args.text_guidance,
                         s_i=args.image_guidance)

    rows = []
    t_start = time.time()
    for lbl, pur in purifiers:
        geometric = pur.kind in purify_ops.GEOMETRIC_KINDS
        outdir = args.out / f"{args.arm}_{cat}_{lbl}"
        outdir.mkdir(parents=True, exist_ok=True)
        for it in items:
            image, cond = it["image"], it["condition"]
            dst_def = outdir / f"{image}__{cond}__edit_def.png"
            if args.skip_existing and dst_def.exists():
                continue
            t0 = time.time()
            x01 = load_image_tensor(it["orig"], ip2p.device, size=RESOLUTION)
            # `floor` 臂的「防禦圖」就是原圖本身。
            x_def = (x01 if args.arm == "floor"
                     else load_image_tensor(it["def"], ip2p.device, size=RESOLUTION))
            p_def = pur.evaluate(x_def)
            if geometric:
                e_orig = edit(pur.evaluate(x01), it["instruction"])
            else:
                e_orig = load_image_tensor(it["edit_orig"], ip2p.device,
                                           size=RESOLUTION)
            e_def = edit(p_def, it["instruction"])
            if args.arm == "floor" and geometric:
                # 兩側同算子、同輸入、同種子，必須逐位元相同。不同表示編輯
                # 不是確定性的或種子沒對齊，那是量測壞了，不是一個小數字。
                if not torch.equal(e_orig, e_def):
                    d = float((e_orig - e_def).abs().max())
                    raise SystemExit(
                        f"floor 的幾何類 {lbl} 在 {image} 上兩側不同（最大差 {d:g}）"
                        "——編輯不是確定性的或種子沒對齊")
            shutil.copyfile(it["orig"], outdir / f"{image}__orig.png")
            vutils.save_image(p_def.clamp(0, 1), outdir / f"{image}__{cond}__def.png")
            vutils.save_image(e_orig.clamp(0, 1),
                              outdir / f"{image}__{cond}__edit_orig.png")
            vutils.save_image(e_def.clamp(0, 1), dst_def)
            prot = suite.pairwise(e_orig, e_def)
            fid = suite.pairwise(x01, p_def)
            sim = suite.image_similarity(e_orig, e_def)
            rows.append({
                "arm": args.arm, "category": cat, "image": image,
                "condition": cond, "purifier": lbl, "purifier_kind": pur.kind,
                "purifier_strength": pur.strength,
                "reference": "purified_orig" if geometric else "orig",
                "geometric": geometric,
                "purify_seed": args.purify_seed,
                # 旋轉是唯一「同一個標籤可以對到不同算子」的欄：角度由種子抽出。
                # 把角度寫進 CSV，出表的人不必回頭重算就知道那一列轉了幾度。
                "rotate_seed": args.rotate_seed,
                "rotate_degrees": (
                    round(purify_ops.rotate_angle(
                        pur.strength or purify_ops.ROTATE_DEGREES_FACELOCK,
                        args.rotate_seed), 4)
                    if pur.kind == "rotate" else ""),
                "instruction": it["instruction"],
                # 失真半邊：原圖對**淨化後的**防禦圖，即攻擊方手上那一張。
                **{f"fid_{k}": round(v, 5) for k, v in fid.items()},
                # 效果半邊：兩張編輯輸出之間。FaceLock 的六個指標裡的四個
                # （LPIPS/PSNR/SSIM 與 CLIP-I）就是這裡的 edit_lpips／
                # edit_psnr／edit_ssim／edit_clip_sim。
                **{f"edit_{k}": round(v, 5) for k, v in prot.items()},
                "edit_clip_sim": round(sim["clip"], 5),
                "edit_siglip_sim": round(sim["siglip"], 5),
                "edit_steps": args.edit_steps, "edit_seed": args.edit_seed,
                "s_t": args.text_guidance, "s_i": args.image_guidance,
                "seconds": round(time.time() - t0, 1),
            })
            write_csv(args.out / f"purify_{args.arm}_{cat}.csv", rows)
            print(f"  {lbl:20s} {image:34s} "
                  f"edit_lpips={rows[-1]['edit_lpips']:.4f} "
                  f"({rows[-1]['seconds']:.0f}s)", flush=True)
    print(f"[purify_identity] {args.arm}／{cat} 收工，{len(rows)} 列，"
          f"{(time.time() - t_start) / 60:.1f} 分", flush=True)


if __name__ == "__main__":
    main()
