"""teaser 案例的防禦、淨化與 ip2p 編輯（使用者給影像與編輯指令）。

條件（--conditions）：
  undefended          原圖
  photoguard_encoder  等失真 PhotoGuard（encoder 攻擊）
  mist                等失真 Mist
  dia_pt              等失真 DIA-PT
  dct_shield          等失真 DCT-Shield
      以上四臂以 immunization_baseline.cli.generate_defenses 求解，--eps-scale 讀自 results/aligned/defense_<條件>.csv
      （主表等失真臂的同一倍率），種子 0。
  blurguard           immunization_baseline.cli.generate_blurguard_defenses（官方 pgd_freq，sigma_weighting 0），
                      再以 scripts/apply_blurguard_scale.py 乘 BLURGUARD_SCALE（主表的校準倍率，見 docs/HANDOVER.md）
  immulut             ImmuLUT 防禦圖，由 color 專案的求解產生；從 --immulut-dir 讀 <影像>.png 或 <影像>__*__def.png（每張恰一個）
淨化（--purification）：none、jpeg50、blur2（高斯 σ=2）、vae_trans（color trial 的 VAE-Trans encoder）。
編輯：timbrooks/instruct-pix2pix（IP2PWrapper，fp32），預設 text guidance 7.5、image guidance 1.8、50 步、種子 20260812（主實驗協定）。

輸出（--run-dir）：
  data/man/<影像>.png                    輸入影像副本（generate_defenses 的讀檔版面）
  defenses/<條件>/<影像>__<條件>__def.png   防禦圖（已存在則不重解）
  inputs/<條件>/<淨化>/<影像>.png          送入 ip2p 的影像
  edits/<條件>/<淨化>/<影像>__p<k>.png     編輯結果；prompts.json 記 k 對應的指令全文，edits.csv 逐次紀錄
同一 --run-dir 再次執行時沿用已存在的檔案；--edit-prompt 與既有 prompts.json 不同時失敗（換一個 --run-dir）。

用法（遠端 baseline 專案根，需一張卡；取卡見 vendor/scripts/run_with_gpu_lease.sh）：
    python scripts/run_teaser_edits.py --images artifacts/synthetic_portraits/<名稱> --run-dir artifacts/teaser/<名稱> \\
        --edit-prompt "<指令一>" --edit-prompt "<指令二>" --immulut-dir <ImmuLUT 防禦圖目錄>
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONDITIONS = ["undefended", "photoguard_encoder", "mist", "dia_pt", "dct_shield", "blurguard", "immulut"]
ALIGNED = ["photoguard_encoder", "mist", "dia_pt", "dct_shield"]
PURIFICATIONS = ["none", "jpeg50", "blur2", "vae_trans"]
BLURGUARD_SCALE = 0.6071
VAE_TRANS_ENCODER = Path.home() / "image-immunization/color/trials/lut3d_defense_and_naturalness/artifacts/purification/vae_trans/model/encoder.pt"
LAYOUT_CLASS = "man"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, type=Path,
                    help="影像來源：目錄（遞迴讀 *.png）、清單檔（.txt，每行一個 PNG 路徑）或 PNG 路徑")
    ap.add_argument("--edit-prompt", action="append", required=True, help="編輯指令；可重複給多句，依序編號 p0、p1…")
    ap.add_argument("--run-dir", type=Path, required=True, help="輸出目錄")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    ap.add_argument("--purification", nargs="+", choices=PURIFICATIONS, default=PURIFICATIONS)
    ap.add_argument("--immulut-dir", type=Path, default=None, help="ImmuLUT 防禦圖目錄（--conditions 含 immulut 時必填）")
    ap.add_argument("--text-guidance", type=float, default=7.5)
    ap.add_argument("--image-guidance", type=float, default=1.8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260812)
    args = ap.parse_args(argv)
    if "immulut" in args.conditions and args.immulut_dir is None:
        ap.error("--conditions 含 immulut 時需要 --immulut-dir")
    if args.immulut_dir is not None and not args.immulut_dir.is_dir():
        ap.error(f"--immulut-dir 不存在：{args.immulut_dir}")
    return args


def collect_images(sources):
    paths = []
    for s in sources:
        if s.is_dir():
            paths += sorted(s.rglob("*.png"))
        elif s.suffix == ".txt" and s.is_file():
            paths += [Path(line.strip()) for line in s.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif s.suffix == ".png" and s.is_file():
            paths.append(s)
        else:
            raise SystemExit(f"--images 無法讀取：{s}")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"缺影像：{missing[:3]}")
    if not paths:
        raise SystemExit("--images 沒有任何 PNG")
    stems = [p.stem for p in paths]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise SystemExit(f"影像檔名（不含副檔名）重複：{dup[:3]}")
    return paths


def immulut_defense(directory, stem):
    found = sorted({*directory.glob(f"{stem}.png"), *directory.glob(f"{stem}__*__def.png")})
    if len(found) != 1:
        raise SystemExit(f"--immulut-dir 中 {stem} 的防禦圖應恰有一個，找到 {[str(p) for p in found]}")
    return found[0]


def aligned_eps_scale(cond):
    with open(f"results/aligned/defense_{cond}.csv", encoding="utf-8", newline="") as f:
        values = {r["eps_scale"] for r in csv.DictReader(f)}
    if len(values) != 1:
        raise SystemExit(f"results/aligned/defense_{cond}.csv 的 eps_scale 不唯一：{values}")
    return values.pop()


def run(cmd):
    print("[RUN]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def solve_defenses(args, stems):
    data, D = args.run_dir / "data", args.run_dir / "defenses"
    out = {}
    for cond in args.conditions:
        if cond == "undefended":
            out[cond] = {s: data / LAYOUT_CLASS / f"{s}.png" for s in stems}
            continue
        if cond == "immulut":
            out[cond] = {s: immulut_defense(args.immulut_dir, s) for s in stems}
            continue
        target = {s: D / cond / f"{s}__{cond}__def.png" for s in stems}
        todo = [s for s in stems if not target[s].exists()]
        if todo:
            if cond in ALIGNED:
                run([sys.executable, "-m", "immunization_baseline.cli.generate_defenses", "--output-dir", D / cond,
                     "--data-root", data, "--images", *todo, "--conditions", cond, "--eps-scale", aligned_eps_scale(cond),
                     "--seed", "0", "--tag", f"teaser_{int(time.time())}"])
            else:
                raw = D / "blurguard_unscaled"
                run([sys.executable, "-m", "immunization_baseline.cli.generate_blurguard_defenses", "--data-root", data,
                     "--output-dir", raw, "--images", *todo, "--sigma-weighting", "0", "--tag", f"teaser_{int(time.time())}"])
                run([sys.executable, "scripts/apply_blurguard_scale.py", "apply", raw, D / cond, str(BLURGUARD_SCALE)])
        missing = [str(p) for p in target.values() if not p.exists()]
        if missing:
            raise SystemExit(f"{cond} 求解後仍缺防禦圖：{missing[:3]}")
        out[cond] = target
    return out


def main(argv=None):
    args = parse_args(argv)
    paths = collect_images(args.images)
    stems = [p.stem for p in paths]
    run_dir = args.run_dir
    prompts_file = run_dir / "prompts.json"
    if prompts_file.exists() and json.loads(prompts_file.read_text(encoding="utf-8")) != args.edit_prompt:
        raise SystemExit(f"{prompts_file} 的指令與本次 --edit-prompt 不同；換一個 --run-dir")
    (run_dir / "data" / LAYOUT_CLASS).mkdir(parents=True, exist_ok=True)
    for p in paths:
        dst = run_dir / "data" / LAYOUT_CLASS / f"{p.stem}.png"
        if not dst.exists():
            shutil.copyfile(p, dst)
    prompts_file.write_text(json.dumps(args.edit_prompt, ensure_ascii=False, indent=1), encoding="utf-8")

    defenses = solve_defenses(args, stems)

    import torch
    from immunization_baseline.vae_trans import VAETrans
    from immunization_core.artifacts.images import save_image
    from immunization_core.editors.instruct_pix2pix import IP2PWrapper
    from immunization_core.io import load_image_tensor
    from immunization_core.purifiers.operators import Purifier
    victim = IP2PWrapper(dtype=torch.float32)
    dev = victim.device
    purify = {"none": lambda x: x, "jpeg50": Purifier("jpeg", 50).evaluate, "blur2": Purifier("blur", 2.0).evaluate}
    if "vae_trans" in args.purification:
        purify["vae_trans"] = VAETrans(VAE_TRANS_ENCODER, dev)
    log = run_dir / "edits.csv"
    new_log = not log.exists()
    with open(log, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new_log:
            w.writerow(["condition", "purification", "image", "prompt_index", "prompt", "defense_png", "input_png", "edit_png",
                        "text_guidance", "image_guidance", "steps", "seed", "seconds"])
        for cond in args.conditions:
            for p in args.purification:
                for s in stems:
                    inp = run_dir / "inputs" / cond / p / f"{s}.png"
                    if not inp.exists():
                        inp.parent.mkdir(parents=True, exist_ok=True)
                        with torch.no_grad():
                            save_image(purify[p](load_image_tensor(defenses[cond][s], dev, size=512)).clamp(0, 1), inp)
                    x = None
                    for k, prompt in enumerate(args.edit_prompt):
                        out = run_dir / "edits" / cond / p / f"{s}__p{k}.png"
                        if out.exists():
                            continue
                        out.parent.mkdir(parents=True, exist_ok=True)
                        x = load_image_tensor(inp, dev, size=512) if x is None else x
                        t = time.time()
                        with torch.no_grad():
                            save_image(victim.edit(x, prompt, seed=args.seed, steps=args.steps, s_t=args.text_guidance,
                                                   s_i=args.image_guidance), out)
                        w.writerow([cond, p, s, k, prompt, defenses[cond][s], inp, out, args.text_guidance, args.image_guidance,
                                    args.steps, args.seed, round(time.time() - t, 2)])
                        f.flush()
                        print(f"[EDIT] {cond} {p} {s} p{k}", flush=True)
    print(f"[ALLDONE] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
