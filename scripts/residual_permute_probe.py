"""候選一 步驟 1：把防禦殘差的區塊打亂，看效果還剩多少。

判定的是 `docs/reference/SURVEY_ARCHITECTURE.md` 第三節的 **H1**：

    DCT-Shield 的效果是**統計驅動**的，本方法的是**實現驅動**的。

H1 是「它裁切保留 98.2%、我們 13%」目前唯一的機制解釋，也是候選一（統計驅動
的擾動）的前提。第三節給的判定實驗是：

    取已存的 `*__def.png` 抽出殘差，做一次 32×32 區塊隨機置換後加回原圖，
    量效果。**預期 DCT-Shield 幾乎不掉、本方法掉光。** 若兩者都掉光，H1 錯。

區塊置換保留的東西與破壞的東西分得很乾淨：**每個區塊的內容逐位元不變**（功率
譜、頻帶分布、區塊內的相位關係全部保留），只有**區塊之間的相對位置**被打亂。
所以效果若只由「畫面上到處都有這種統計的能量」決定，置換後不該掉；效果若由
「這一塊的擾動正好對準這一塊的內容」決定，置換後應該掉光。

**這不是淨化算子**，不得進入頭對頭的淨化器清單：它需要原圖才能抽出殘差，而
淨化算子只拿得到防禦圖。同理它也不放進 `src/purify/ops.py`（那裡的 `Purifier`
介面只收一張影像）。

為什麼不重用抗淨化表裡的 `identity` 那一列當分母：那一列與這裡的置換列若不在
同一個 process 裡用同一組種子算出來，兩者的分母就不是同一個量。分母塌陷時
比值不可解讀（`FND-037`／`FND-039`，相關係數 −0.83／−0.900），故 `identity`
在這裡重算。

用法（分片同抗淨化，色／景／物三片）：

    python scripts/residual_permute_probe.py \\
        --run runs/ip2p_axis_necessity/b_pg_r20 --tag ours_add \\
        --images <該分片的影像> --out runs/residual_permute/ours_add_color.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def permute_blocks(r: torch.Tensor, block: int, seed: int) -> torch.Tensor:
    """把 (1,C,H,W) 的殘差切成 `block×block` 的方格並隨機重排方格的位置。

    方格**內容**完全不動（含通道間的關係），只有方格的座標被換。三個通道
    共用同一個排列，否則被破壞的就不只是空間對位，還有色彩結構，歸因會混掉。
    """
    if r.dim() != 4 or r.shape[0] != 1:
        raise ValueError(f"需要 (1,C,H,W) 張量，收到 {tuple(r.shape)}")
    c, h, w = r.shape[1:]
    if h % block or w % block:
        raise ValueError(f"{h}×{w} 不是 {block} 的整數倍，方格會切不齊")
    nh, nw = h // block, w // block
    tiles = (r.view(1, c, nh, block, nw, block)
              .permute(0, 2, 4, 1, 3, 5)              # (1, nh, nw, C, b, b)
              .reshape(1, nh * nw, c, block, block))
    g = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(nh * nw, generator=g).to(r.device)
    tiles = tiles[:, order]
    return (tiles.view(1, nh, nw, c, block, block)
                 .permute(0, 3, 1, 4, 2, 5)
                 .reshape(1, c, h, w))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="防禦圖所在的目錄（讀 results.csv 決定條件與檔名）")
    ap.add_argument("--tag", required=True,
                    help="寫進 CSV 的條件標籤，例如 ours_add / dct_e18")
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", nargs="+", default=None, help="分片用")
    ap.add_argument("--block", type=int, default=32,
                    help="置換的方格邊長。32 是 SURVEY_ARCHITECTURE 指定的值")
    ap.add_argument("--permute-seed", type=int, default=0,
                    help="方格排列的種子。與編輯的 seed 無關，逐列記下")
    ap.add_argument("--seeds", type=int, default=3, help="編輯的種子數")
    ap.add_argument("--gallery", type=Path, default=None,
                    help="存第一個種子的防禦圖／置換圖與三張編輯輸出。"
                         "判「掉光」還是「還在」要人眼看過，而編輯本來"
                         "就算過，存圖不多花運算。影像不入版控")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import csv
    with (args.run / "results.csv").open(encoding="utf-8") as fh:
        rows_in = list(csv.DictReader(fh))
    if not rows_in:
        raise SystemExit(f"{args.run / 'results.csv'} 是空的")
    cells = [{"image": r["image"], "condition": r["condition"]} for r in rows_in]
    if args.images:
        keep = set(args.images)
        cells = [c for c in cells if c["image"] in keep]
    if not cells:
        raise SystemExit(f"{args.run / 'results.csv'} 沒有符合的影像")

    from src.models.ip2p import (
        IP2P_IMAGE_GUIDANCE, IP2P_SEED, IP2P_STEPS, IP2P_TEXT_GUIDANCE,
        IP2PWrapper,
    )

    sd = IP2PWrapper(dtype=torch.float32)
    kw = {"steps": IP2P_STEPS, "s_t": IP2P_TEXT_GUIDANCE,
          "s_i": IP2P_IMAGE_GUIDANCE}
    seeds = [IP2P_SEED + k for k in range(args.seeds)]
    suite = MetricSuite(device=sd.device)
    dataset = {d["name"]: d for d in load_dataset(args.data)}

    rows = []
    for cell in cells:
        name, cond = cell["image"], cell["condition"]
        item = dataset[name]
        x = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        def_png = args.run / f"{name}__{cond}__def.png"
        if not def_png.exists():
            raise FileNotFoundError(f"缺少防禦圖 {def_png}")
        x_def = load_image_tensor(def_png, sd.device, size=RESOLUTION)

        r = x_def - x
        r_perm = permute_blocks(r, args.block, args.permute_seed)
        x_perm = (x + r_perm).clamp(0, 1)

        t0 = time.time()
        e_orig = [sd.edit(x.clamp(0, 1), item["prompt"], seed=s, **kw) for s in seeds]
        eff_def, eff_perm = [], []
        for k, s in enumerate(seeds):
            ed = sd.edit(x_def.clamp(0, 1), item["prompt"], seed=s, **kw)
            ep = sd.edit(x_perm, item["prompt"], seed=s, **kw)
            eff_def.append(float(suite.pairwise(e_orig[k], ed)["lpips"]))
            eff_perm.append(float(suite.pairwise(e_orig[k], ep)["lpips"]))
            # 只存第一個種子的五張圖。**位移是連續讀數，判「掉光」還是「還在」
            # 要人眼看過**（跨方法的擋下率一律人眼定案），而編輯本來就算過，
            # 存圖不多花任何運算。影像不入版控，用完即可刪。
            if args.gallery is not None and s == seeds[0]:
                args.gallery.mkdir(parents=True, exist_ok=True)
                for sub, img in (("def", x_def), ("perm", x_perm),
                                 ("edit_orig", e_orig[k]), ("edit_def", ed),
                                 ("edit_perm", ep)):
                    save_image(img.clamp(0, 1),
                               args.gallery / f"{name}__{args.tag}__{sub}.png")
        m_def = suite.pairwise(x, x_def)
        m_perm = suite.pairwise(x, x_perm)

        base = statistics.fmean(eff_def)
        base_sd = statistics.stdev(eff_def) if len(eff_def) > 1 else float("nan")
        # 分母塌陷時比值不可解讀，逐列標明，不靜默照算。
        usable = bool(base_sd == base_sd and base >= 3.0 * base_sd)
        rows.append({
            "image": name, "tag": args.tag, "condition": cond,
            "block": args.block, "permute_seed": args.permute_seed,
            "effect_def_mean": round(base, 5),
            "effect_def_sd": round(base_sd, 5) if base_sd == base_sd else "",
            "effect_perm_mean": round(statistics.fmean(eff_perm), 5),
            "effect_perm_sd": round(statistics.stdev(eff_perm), 5)
                              if len(eff_perm) > 1 else "",
            "retention": round(statistics.fmean(eff_perm) / base, 5)
                         if base > 0 else "",
            # 置換不改殘差的能量，只改它落在哪裡；兩個 RMS 應該幾乎相同，
            # 差異只來自 clamp。不同就是置換寫錯了，故逐列記下當守門。
            "rms_def": round(m_def["rms"], 6),
            "rms_perm": round(m_perm["rms"], 6),
            "dists_def": round(m_def["dists"], 6),
            "dists_perm": round(m_perm["dists"], 6),
            "psnr_def": round(m_def["psnr"], 4),
            "psnr_perm": round(m_perm["psnr"], 4),
            "usable": usable,
            "seconds": round(time.time() - t0, 1),
        })
        write_csv(args.out, rows)
        print(f"{name:34s} 原 {base:.4f} → 置換 {rows[-1]['effect_perm_mean']:.4f} "
              f"（保留 {rows[-1]['retention']}）  RMS {m_def['rms']:.4f}/"
              f"{m_perm['rms']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    ok = [r for r in rows if r["usable"]]
    if ok:
        print(f"\n{args.tag}：{len(ok)}/{len(rows)} 張可用，"
              f"置換後的效果保留率平均 "
              f"{statistics.fmean(r['retention'] for r in ok):.4f}")
    print(f"表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
