"""未防禦預檢：候選指令在**沒有防禦**的情況下會做出什麼。

HANDOFF 第 13 條：新的指令格一定要先跑這一支再逐張看圖。未防禦的編輯自己就
換人時，那一格的身分分母是零，防禦有沒有效讀不出來——「make it look older」
在 IP2P 上就是這樣，年輕男子被換成白髮老婦。

只跑原圖，不碰任何防禦圖。輸出 `{image}__{key}__s{seed}__preflight.png`
與一份 CSV（身分餘弦 ＋ 指令對齊），兩者都要看：**數字只作初篩，成不成立看圖。**
"""
import argparse, csv, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=17001)
    args = ap.parse_args()
    from src.defense.assets import load_image, save_png
    from src.metrics.identity import identity_row
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper
    CAND = {
      "hat": "put a hat on it",
      "glasses": "put a pair of glasses on it",
      "snow_bg": "change the background to a snowy street",
      "beach_bg": "change the background to a beach",
      "smile": "make it smile",
      "older": "make it look older",
      "red_shirt": "make the shirt red",
    }
    ip2p = IP2PWrapper(dtype=torch.bfloat16); dev = ip2p.device
    suite = MetricSuite(device=dev)
    origs = {p.stem: p for p in sorted(args.data.glob("*/*.png")) if p.parent.name not in ("headmasks","masks")}
    args.out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name in args.images:
        x = load_image(origs[name], dev)
        save_png(x, args.out / f"{name}__orig.png")
        for k, instr in CAND.items():
            t0=time.time()
            with torch.no_grad():
                e = ip2p.edit(x, instr, seed=args.seed, steps=50, s_t=7.5, s_i=1.5)
            save_png(e, args.out / f"{name}__{k}__s{args.seed}__preflight.png")
            idr = identity_row(x, e, e)
            sem = suite.semantic(e, instr)
            sem0 = suite.semantic(x, instr)
            rows.append({"image":name,"key":k,"instruction":instr,
                         "id_orig_vs_edit": idr["id_orig"],
                         "clip_edit": round(sem["clip"],4), "clip_input": round(sem0["clip"],4),
                         "clip_gain": round(sem["clip"]-sem0["clip"],4),
                         "siglip_gain": round(sem["siglip"]-sem0["siglip"],4),
                         "lpips_vs_orig": round(float(suite.pairwise(x,e)["lpips"]),4),
                         "seconds": round(time.time()-t0,1)})
            print(rows[-1], flush=True)
            with (args.out/"preflight.csv").open("w",newline="",encoding="utf-8") as fh:
                w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__ == "__main__":
    main()
