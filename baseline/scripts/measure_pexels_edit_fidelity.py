"""Pexels 人像編輯端的 SSIM 與 PSNR：未防禦編輯 E_u 對防禦後編輯 E_d，兩者經同一淨化。

配對與讀圖同 `pipelines.displacement`／`pipelines.retention`：兩張 PNG 以 `load_image_tensor(size=512)` 讀入，
`piq.ssim`、`piq.psnr`（data_range 1.0，即 `MetricSuite.pairwise` 的算法），整張影像。小數位同 `metrics.standard.ROUNDING`。
格的來源為後續量測的語意表：每列的 `edit_png` 為 E_d，同表 `undefended` 列中相同（影像、句、淨化）的 `edit_png` 為 E_u。
    七道（與未淨化，用以核對）：followup/<根>/results/<aligned|color3d>/semantic_identity.csv → edit_fidelity.csv
    VAE-Trans：               followup/<根>/results/<aligned|color3d>/semantic_identity_vae_trans.csv → edit_fidelity_vae_trans.csv
未淨化格另與位移表的 `disp_ssim`、`disp_psnr` 逐格核對，不一致即失敗。缺 E_u 或 PNG 時失敗。

用法（遠端 baseline 專案根，CPU）：python scripts/measure_pexels_edit_fidelity.py <seven|vae_trans> [--workers N]
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RESOLUTION = 512
UNDEFENDED = "undefended"
FIELDS = ("condition", "scenario", "image", "prompt_index", "purifier", "disp_ssim", "disp_psnr",
          "defended_png", "undefended_png", "device")
CT = Path("../color/trials/lut3d_defense_and_naturalness")


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def measure(pairs):
    import piq
    import torch
    from immunization_core.io import load_image_tensor
    from immunization_core.metrics.standard import DEFAULT_ROUNDING, ROUNDING
    torch.set_num_threads(1)
    out = []
    with torch.no_grad():
        for a, b in pairs:
            xa = load_image_tensor(Path(a), "cpu", size=RESOLUTION)
            xb = load_image_tensor(Path(b), "cpu", size=RESOLUTION)
            out.append((round(float(piq.ssim(xa, xb, data_range=1.0)), ROUNDING.get("ssim", DEFAULT_ROUNDING)),
                        round(float(piq.psnr(xa, xb, data_range=1.0)), ROUNDING.get("psnr", DEFAULT_ROUNDING))))
    return out


def plan(semantic_csv):
    rows = read(semantic_csv)
    reference = {(r["scenario"], r["image"], r["prompt_index"], r["purifier"]): r["edit_png"]
                 for r in rows if r["condition"] == UNDEFENDED}
    cells = []
    for r in rows:
        if r["condition"] == UNDEFENDED:
            continue
        key = (r["scenario"], r["image"], r["prompt_index"], r["purifier"])
        if key not in reference:
            raise SystemExit(f"{semantic_csv} 缺 {key} 的未防禦編輯")
        for p in (r["edit_png"], reference[key]):
            if not Path(p).is_file():
                raise SystemExit(f"找不到 {p}")
        cells.append({"condition": r["condition"], "scenario": r["scenario"], "image": r["image"],
                      "prompt_index": r["prompt_index"], "purifier": r["purifier"],
                      "defended_png": r["edit_png"], "undefended_png": reference[key]})
    return cells


def displacement_reference(sub, group):
    """未淨化格的位移表值：(條件, 影像, 句) -> (disp_ssim, disp_psnr)。"""
    if group == "color3d":
        paths = [CT / "results/color3d_pexels/displacement.csv"]
    else:
        R = Path("runtime/pexels_portraits/aligned") / sub / "results"
        paths = sorted(R.glob("displacement_*.csv"))
    table = {}
    for p in paths:
        for r in read(p):
            if r["scenario"] == "ip2p":
                table[(r["condition"], r["image"], r["prompt_index"])] = (float(r["disp_ssim"]), float(r["disp_psnr"]))
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("seven", "vae_trans"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    source = "semantic_identity.csv" if args.part == "seven" else "semantic_identity_vae_trans.csv"
    target = "edit_fidelity.csv" if args.part == "seven" else "edit_fidelity_vae_trans.csv"
    sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
    t0 = time.time()
    total = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for s in sets:
            sub = Path(s["root"]) if s["root"] else Path()
            F = Path("runtime/pexels_portraits/followup") / sub / "results"
            for group in ("aligned", "color3d"):
                cells = plan(F / group / source)
                pairs = [(c["undefended_png"], c["defended_png"]) for c in cells]
                chunks = [pairs[i::args.workers] for i in range(args.workers)]
                results = list(pool.map(measure, chunks))
                for i, c in enumerate(cells):
                    c["disp_ssim"], c["disp_psnr"] = results[i % args.workers][i // args.workers]
                    c["device"] = "cpu"
                if args.part == "seven":
                    base = displacement_reference(sub, group)
                    plain = [c for c in cells if c["purifier"] == "none"]
                    if not plain:
                        raise SystemExit(f"{F / group / source} 沒有未淨化格可核對")
                    diff_ssim = max(abs(c["disp_ssim"] - base[(c["condition"], c["image"], c["prompt_index"])][0]) for c in plain)
                    diff_psnr = max(abs(c["disp_psnr"] - base[(c["condition"], c["image"], c["prompt_index"])][1]) for c in plain)
                    print(f"[CHECK] {F / group}：未淨化 {len(plain)} 格對位移表的最大差 SSIM {diff_ssim:.6f}、PSNR {diff_psnr:.4f} dB", flush=True)
                    # 位移表可能以 CUDA 計算：容許浮點差，超過即視為配對或協定不同
                    if diff_ssim > 1e-4 or diff_psnr > 1e-2:
                        raise SystemExit("未淨化格與位移表的差超過 SSIM 1e-4 或 PSNR 0.01 dB，配對或協定不一致")
                with open(F / group / target, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
                    w.writeheader()
                    w.writerows(cells)
                total += len(cells)
                print(f"[DONE] {F / group / target}（{len(cells)} 格，累計 {total}，{time.time() - t0:.0f}s）", flush=True)
    print(f"[ALLDONE] {args.part} {total} 格，{time.time() - t0:.0f}s，{args.workers} 個 CPU 行程", flush=True)


if __name__ == "__main__":
    main()
