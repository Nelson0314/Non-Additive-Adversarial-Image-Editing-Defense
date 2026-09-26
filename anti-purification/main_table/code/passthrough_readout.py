"""把主表各條件的位移拆成「穿透」與「扣掉穿透之後」兩部分。

現行的位移 `D = LPIPS(edit(x), edit(x_def))` 把兩件事算在一起：防禦圖的改動
**原樣穿過編輯器**所造成的差異，以及編輯器真的被推離原本輸出的部分。對全域色調
這類改動，ip2p 近乎等變，前者可以佔絕大部分。

三個讀數（皆分全圖／主體／背景，遮罩與 `edit_displacement.py` 同源）：

    D   = LPIPS( edit(x),    edit(x_def) )     現行位移
    P   = LPIPS( edit(x),    T̂(edit(x)) )      完全等變時的預測位移
    D_T = LPIPS( T̂(edit(x)), edit(x_def) )     扣掉穿透後的位移

`T̂` 是把防禦端的改動套到編輯輸出上的預測。**本檔只實作加性族**：

    T̂(y) = clip( y + (x_def − x), 0, 1 )，再量化到 8 bit

主表十二個條件裡有十一個是加性擾動（像素域或 DCT 係數域的 PGD、前饋式的 DiffVax），
這個 `T̂` 對它們是該方法自己的擾動，不是近似。**`colour_curve_ours` 是全域色調映射**，
它的 `T̂` 應該由參數族回推；加性版本在這裡照樣算出來，是為了讓十二列在同一個 `T̂`
定義下可比，引用時要與由映射回推的值分開講——後者在 `lab/results/passthrough/`
（另一批的協定，回推誤差 ΔE00 0.04）。

另報 `siglip_pair_T = SigLIP(T̂(edit(x)), edit(x_def))` 與 `blocked_T`（同一個 0.837 門檻）。

只讀既有 PNG，不呼叫擴散。用法：

    python code/passthrough_readout.py --out results/passthrough.csv
    python code/passthrough_readout.py --conditions mist dia_r --out /tmp/probe.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from src.metrics.regional import RegionalLPIPS, split_displacement  # noqa: E402
from src.metrics.standard import SIGLIP_BLOCKED_THRESHOLD  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

from edit_displacement import subject_mask  # noqa: E402

RESOLUTION = 512
SCENARIO = "ip2p"

#: 防禦圖的產出目錄，兩者的檔名式樣相同（`<影像>__<條件>__def.png`）。
#: 原生條件在 `main_table/images/defence_portraits/`，但那份影像沒有同步到遠端
#: （`.gitignore` 排除、scp 時也沒帶），遠端只有求解當時寫出的
#: `runs/defence_portraits/`，所以兩個位置都要找。等失真臂只在 `runs/eps_aligned/`。
def _first_dir(*candidates: Path) -> Path:
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit("找不到防禦圖目錄，找過：" + "、".join(str(c) for c in candidates))


NATIVE_DEF = _first_dir(paths.IMAGES / "defence_portraits",
                        paths.SOURCE_HOME / "runs" / "defence_portraits")
ALIGNED_DEF = _first_dir(paths.SOURCE_HOME / "runs" / "eps_aligned")


def load(path: Path, device) -> torch.Tensor:
    return load_image_tensor(path, device, size=RESOLUTION)


def quantise(y: torch.Tensor) -> torch.Tensor:
    """量化到 8 bit。編輯輸出是從 PNG 讀進來的，預測也要走同一個量化階，
    否則 `P` 會含一層只存在於浮點的差異。"""
    return (y.detach().clamp(0, 1) * 255).round() / 255


def defence_png(condition: str, image: str, arm: str) -> Path:
    root = ALIGNED_DEF if arm == "aligned" else NATIVE_DEF
    return root / condition / f"{image}__{condition}__def.png"


def original_png(image: str) -> Path:
    """原圖。每個條件目錄都存了一份 `__orig.png`，內容同一組八張。"""
    for root in (NATIVE_DEF, ALIGNED_DEF):
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            p = d / f"{image}__orig.png"
            if p.is_file():
                return p
    raise SystemExit(f"找不到 {image} 的原圖（找過 {NATIVE_DEF} 與 {ALIGNED_DEF}）")


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve(recorded: str) -> Path:
    """CSV 記的影像路徑。兩份 CSV 的寫法不同——`displacement.csv` 是相對於主線
    目錄的相對路徑，`displacement_aligned.csv` 是絕對路徑——所以相對的那一種要
    補上主線目錄，不能直接當 cwd 相對路徑用。"""
    p = Path(recorded)
    return p if p.is_absolute() else paths.SOURCE_HOME / p


def sources(args) -> list:
    """回傳 (arm, csv 列) 的清單，只取 ip2p 場景。"""
    out = []
    for arm, rel in (("native", "displacement.csv"),
                     ("aligned", "aligned/displacement_aligned.csv")):
        if args.arms and arm not in args.arms:
            continue
        rows = read_csv(paths.RESULTS / rel)
        for r in rows:
            if r["scenario"] != SCENARIO:
                continue
            if args.conditions and r["condition"] not in args.conditions:
                continue
            out.append((arm, r))
    if not out:
        raise SystemExit("沒有符合條件的列")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.IMAGES,
                    help="只用底下的 masks/")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=("native", "aligned"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)

    items = sources(args)
    cache = {}          # (arm, condition, image) -> 防禦端的加性改動
    masks = {}
    originals = {}
    rows = []
    for arm, r in items:
        cond, name, k = r["condition"], r["image"], r["prompt_index"]
        if name not in originals:
            originals[name] = load(original_png(name), device)
            repaint = load(args.data / "masks" / f"{name}.png", device)[:, :1]
            masks[name] = subject_mask(repaint)
        key = (arm, cond, name)
        if key not in cache:
            dp = defence_png(cond, name, arm)
            if not dp.is_file():
                raise SystemExit(f"找不到防禦圖：{dp}")
            cache[key] = load(dp, device) - originals[name]
        a = load(resolve(r["undefended_png"]), device)
        b = load(resolve(r["defended_png"]), device)
        with torch.no_grad():
            ta = quantise(a + cache[key])
            d = split_displacement(regional, a, b, masks[name])
            p = split_displacement(regional, a, ta, masks[name])
            dt = split_displacement(regional, ta, b, masks[name])
            sig = float(suite.image_similarity(ta, b)["siglip"])
        rows.append({
            "arm": arm, "condition": cond, "scenario": SCENARIO,
            "image": name, "prompt_index": k, "prompt": r["prompt"],
            "t_hat": "additive",
            **{f"D_{key2}": round(float(v), 5) for key2, v in d.items()},
            **{f"P_{key2}": round(float(v), 5) for key2, v in p.items()},
            **{f"DT_{key2}": round(float(v), 5) for key2, v in dt.items()},
            "D_csv": r["disp_lpips_full"],
            "siglip_pair": r["siglip_pair"],
            "siglip_pair_T": round(sig, 5),
            "blocked_T": sig < SIGLIP_BLOCKED_THRESHOLD,
            "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
            "defence_png": defence_png(cond, name, arm).as_posix(),
        })
        write_csv(args.out, rows)
    print(f"[ALLDONE] {args.out}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
