"""主讀數：防禦後的編輯離未防禦的編輯有多遠。

位移是什麼
────────────────────────────────────────────────────────────────────
`位移 = LPIPS( 編輯(原圖), 編輯(防禦圖) )`。防禦的問題不是「防禦圖看起來像不
像原圖」——那是保真那一欄——而是**攻擊方拿到的編輯結果有沒有被推開**。
兩張圖都在編輯之後，所以同一條管線、同一顆種子、同一句指令，唯一的差別
是輸入有沒有被免疫過。

全圖位移會被灌水
────────────────────────────────────────────────────────────────────
防禦端改掉的顏色或貼上的補丁會原封不動穿過編輯，被 LPIPS 算成「編輯被推開
了」，而受保護的主體可能完全沒被保住。故同時報**主體內**與**主體外**兩塊
（`src/metrics/regional.py`，遮罩為空間權重，全 1 時逐位元退回全圖 LPIPS）。

其餘欄位照報
────────────────────────────────────────────────────────────────────
`disp_*` 五項是定案的標準欄位（LPIPS／SSIM／PSNR／VIFp／DISTS），方向見
`src/metrics/standard.py`：防禦那一半「位移大」為強。CLIP 與 SigLIP 照報
**不作判準**（本專案量過語意讀數會與看圖相反）。`siglip_pair` 是兩張編輯
結果在 SigLIP 影像空間的餘弦，低於 `SIGLIP_BLOCKED_THRESHOLD` 記為
`blocked`；門檻逐列寫進 CSV，因為門檻改了之後舊列仍要讀得懂。

**判準不由本腳本下。** 它只把數擺出來，成立與否看圖。

用法
    python scripts/edit_displacement.py --defended-root runs/edit_defended \\
        --preflight runs/edit_preflight --data data/portraits \\
        --out runs/edit_defended/displacement.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.metrics.regional import RegionalLPIPS, split_displacement  # noqa: E402
from src.metrics.standard import SIGLIP_BLOCKED_THRESHOLD, standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

#: 未防禦的對照臂。**設定必須與防禦臂逐項相同**，否則位移裡混進了設定差異。
DEFAULT_ARMS = {"ip2p": "ip2p_si18", "inpaint": "inpaint_undefended"}


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def undefended_png(preflight: Path, arm: str, name: str, index: str) -> Path:
    path = preflight / arm / f"{name}__p{index}.png"
    if not path.is_file():
        raise SystemExit(f"找不到未防禦的對照圖：{path}。"
                         "分母缺一張就不能只算其餘的，那會讓條件之間不可比")
    return path


def subject_mask(repaint: torch.Tensor) -> torch.Tensor:
    """由 inpainting 遮罩換算主體遮罩。

    **極性要翻**：`data/*/masks` 是白＝重繪，而重繪區正是主體以外的地方；
    `split_displacement` 要的是主體遮罩（1 = 主體）。不翻的話主體內與主體外
    兩欄會整組對調，而且不會報錯。
    """
    return 1.0 - (repaint >= 0.5).float()


def defended_png(condition_dir: Path, arm: str, name: str, index: str) -> Path:
    path = condition_dir / arm / f"{name}__p{index}.png"
    if not path.is_file():
        raise SystemExit(f"找不到防禦後的編輯圖：{path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--defended-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/portraits"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ip2p-arm", default=DEFAULT_ARMS["ip2p"])
    parser.add_argument("--inpaint-arm", default=DEFAULT_ARMS["inpaint"])
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()
    arms = {"ip2p": args.ip2p_arm, "inpaint": args.inpaint_arm}

    directories = sorted(d for d in args.defended_root.iterdir()
                         if d.is_dir() and not d.name.startswith("_")
                         and (d / "preflight.csv").is_file())
    if args.conditions:
        keep = set(args.conditions)
        directories = [d for d in directories if d.name in keep]
    if not directories:
        raise SystemExit(f"{args.defended_root} 底下沒有任何帶 preflight.csv 的條件目錄")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)

    rows = []
    for directory in directories:
        condition = directory.name
        for row in read_csv(directory / "preflight.csv"):
            scenario, name, index = row["scenario"], row["image"], row["prompt_index"]
            if scenario not in arms:
                raise SystemExit(f"沒有為場景 {scenario} 指定未防禦的對照臂")
            a = load_image_tensor(undefended_png(args.preflight, arms[scenario], name, index),
                                  device, size=RESOLUTION)
            b = load_image_tensor(defended_png(directory, row["arm"], name, index),
                                  device, size=RESOLUTION)
            mask_path = args.data / "masks" / f"{name}.png"
            mask = subject_mask(
                load_image_tensor(mask_path, device, size=RESOLUTION)[:, :1])
            with torch.no_grad():
                pair = suite.pairwise(a, b)
                split = split_displacement(regional, a, b, mask)
                similar = suite.image_similarity(a, b)
            siglip = float(similar["siglip"])
            rows.append({
                "condition": condition, "scenario": scenario, "arm": row["arm"],
                "image": name, "prompt_index": index, "prompt": row["prompt"],
                "undefended_arm": arms[scenario],
                **standard_row("disp_", pair),
                "disp_rms": round(float(pair["rms"]), 6),
                "disp_linf": round(float(pair["linf"]), 6),
                **{f"disp_{k}": round(float(v), 5) for k, v in split.items()},
                "clip_pair": round(float(similar["clip"]), 5),
                "siglip_pair": round(siglip, 5),
                "blocked": siglip < SIGLIP_BLOCKED_THRESHOLD,
                "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
                "defended_png": defended_png(directory, row["arm"], name, index).as_posix(),
                "undefended_png": undefended_png(args.preflight, arms[scenario],
                                                 name, index).as_posix(),
            })
            write_csv(args.out, rows)
        print(f"[DONE] {condition:16s} 累計 {len(rows)} 列", flush=True)
    print(f"[ALLDONE] {args.out}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
