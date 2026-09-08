"""衣服底色 vs 色度族的施力空間。**CPU、不佔卡。**

問的是什麼
────────────────────────────────────────────────────────────────────
`runs/ip2p_patch_tint_colour/cgrid_rand`（隨機色彩場、半徑 0.60、未最佳化）
在三張影像上量到 `edit_lpips` 0.6219／0.1318／0.0644，而載體面積分別是
0.368／0.190／0.329——**面積幾乎一樣的兩張差了將近十倍**。

仿射色彩重映射的輸出是 `a·x + b`。輸入接近零時乘性項沒有施力點，只剩偏移
項能動。若這個機制是對的，那麼「這一族在某張影像上能不能用」應該**事先**
就由載體區的亮度分布預測得出來，不必等它跑完。

怎麼量
────────────────────────────────────────────────────────────────────
施力空間不用理論推導，直接實測：對同一張影像抽 S 組**盒角**的隨機仿射場
（`draw="corner"` 抽 ±radius，是該半徑下的極值），量 `render − x` 在載體區
的 RMS，取 S 組的平均。那就是「這一族在這張影像上最多能推動多少」。

對照組是補丁族：它直接換掉像素，施力空間與底色無關，故其 RMS 恆為
「支撐內的可用動態範圍」本身。
"""
import csv, sys
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.defense.carrier_mask import carrier_mask
from src.defense.color_param import ColorGridRandomParam
from src.defense.subject_mask import subject_mask

SEEDS = 8
RADIUS = 0.60

def main() -> None:
    """**整支腳本原本在 module 層級執行**，於是 `import` 它就會跑完整個實驗
    並覆寫 `runs/carrier_survey/colour_headroom.csv`。`tests/test_ip2p_run_columns.py`
    的 `test_every_driver_script_imports` 會 import 每一支 `scripts/*.py`，
    所以每跑一次測試就重寫一次結果檔；在缺那些資料檔的機器上則直接
    `FileNotFoundError`。工作一律放在這裡。"""
    cat = yaml.safe_load((ROOT/"data/carrier_catalogue.yaml").read_text(encoding="utf-8"))["objects"]
    IMGS = [k for k in cat if k != "task_attr_mod_color_6205"]

    rows = []
    for name in sorted(IMGS):
        p = ROOT/f"data/omniedit150/{name}/{name}.png"
        arr = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255
        x = torch.from_numpy(arr).permute(2, 0, 1)[None]
        c = carrier_mask(x, "clothes")
        m = subject_mask(x, cat[name])
        w = ((c > 0.5) & (m <= 0.0)).to(x.dtype)
        sel = w[0, 0].bool().numpy()
        if sel.sum() < 100:
            print(f"{name}: 載體太小，跳過"); continue

        luma = (0.299*arr[..., 0] + 0.587*arr[..., 1] + 0.114*arr[..., 2])[sel]
        sat = (arr.max(-1) - arr.min(-1))[sel]

        ds = []
        for s in range(SEEDS):
            prm = ColorGridRandomParam(radius=RADIUS, grid=16, luma_bins=16,
                                       apply_where=w)
            prm.reset(x, seed=1000 + s)
            d = (prm.render(x) - x)[0].permute(1, 2, 0).detach().numpy()
            ds.append(float(np.sqrt((d[sel]**2).mean())))
        rows.append({
            "image": name, "carrier_area": round(float(w.mean()), 5),
            "luma_mean": round(float(luma.mean()), 4),
            "luma_p10": round(float(np.percentile(luma, 10)), 4),
            "luma_p50": round(float(np.percentile(luma, 50)), 4),
            "luma_p90": round(float(np.percentile(luma, 90)), 4),
            "luma_std": round(float(luma.std()), 4),
            "sat_mean": round(float(sat.mean()), 4),
            "headroom_rms": round(float(np.mean(ds)), 5),
        })
        r = rows[-1]
        print(f"{name:<32s} 面積 {r['carrier_area']:.3f}  亮度 {r['luma_mean']:.3f} "
              f"(p10 {r['luma_p10']:.3f}/p90 {r['luma_p90']:.3f})  飽和 {r['sat_mean']:.3f} "
              f"→ 施力空間 {r['headroom_rms']:.4f}", flush=True)

    out = ROOT/"runs/carrier_survey"/"colour_headroom.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)

    xs = np.array([r["luma_mean"] for r in rows]); ys = np.array([r["headroom_rms"] for r in rows])
    ar = np.array([r["carrier_area"] for r in rows])
    def pear(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
    print(f"\n施力空間 vs 載體平均亮度   Pearson r = {pear(xs, ys):+.3f}  (n={len(rows)})")
    print(f"施力空間 vs 載體面積       Pearson r = {pear(ar, ys):+.3f}")
    print(f"施力空間 vs 載體平均飽和度 Pearson r = {pear(np.array([r['sat_mean'] for r in rows]), ys):+.3f}")
    print(f"\n寫出 {out}")


if __name__ == "__main__":
    main()
