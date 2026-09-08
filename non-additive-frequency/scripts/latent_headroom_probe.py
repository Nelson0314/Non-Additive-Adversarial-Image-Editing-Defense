"""色度族在衣服上推得動 latent 嗎——尤其是**受保護主體那些 token**。

為什麼要量 latent 而不是 RGB
────────────────────────────────────────────────────────────────────
`colour_headroom.csv` 量到載體區的 RGB 施力空間由衣服亮度幾乎完全決定
（r = +0.968），但最暗與最亮之間只差 32%，而實測 `edit_lpips` 差近十倍。
**RGB 上的施力空間解釋不了那個差距。** IP2P 看到的不是 RGB 是 latent，
而且防禦要推動的是**主體那些 token**（花紋碰不到主體，只能靠 UNet 的長程
作用間接影響）——與 `runs/ig_probe/by_region_*.csv` 問的是同一件事。

用攻擊模型自己的 VAE（`timbrooks/instruct-pix2pix` 的 `vae` 子資料夾），
不是本機快取裡的 SD 1.4——換一個 VAE 就不是同一條軸。
"""
import csv, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.defense.carrier_mask import carrier_mask
from src.defense.color_param import ColorGridRandomParam
from src.defense.subject_mask import subject_mask

# **權重延遲載入。** 原本這一行在 module 層級，於是 `import` 這支腳本就會
# 把 IP2P 的 VAE 拉下來——`tests/test_ip2p_run_columns.py` 的
# `test_every_driver_script_imports` 會 import 每一支 `scripts/*.py`。
_VAE = None


def _vae():
    global _VAE
    if _VAE is None:
        from diffusers import AutoencoderKL
        _VAE = AutoencoderKL.from_pretrained(
            "timbrooks/instruct-pix2pix", subfolder="vae").eval()
        print("VAE 載入完成", flush=True)
    return _VAE


@torch.no_grad()
def enc(x01):
    return _vae().encode(x01 * 2 - 1).latent_dist.mean


SEEDS = 4
def main() -> None:
    """**這一段原本在 module 層級執行**，於是 `import` 這支腳本就會跑完整個
    實驗並覆寫結果檔。`test_every_driver_script_imports` 會 import 每一支
    `scripts/*.py`，所以每跑一次測試就重寫一次結果；在缺那些資料檔的機器上
    則直接 `FileNotFoundError`。工作一律放在這裡。"""
    cat = yaml.safe_load((ROOT/"data/carrier_catalogue.yaml").read_text(encoding="utf-8"))["objects"]
    IMGS = sorted(k for k in cat if k != "task_attr_mod_color_6205")

    rows = []
    for name in IMGS:
        arr = np.asarray(Image.open(ROOT/f"data/omniedit150/{name}/{name}.png"
                                    ).convert("RGB")).astype(np.float32)/255
        x = torch.from_numpy(arr).permute(2, 0, 1)[None]
        c = carrier_mask(x, "clothes"); m = subject_mask(x, cat[name])
        w = ((c > 0.5) & (m <= 0.0)).to(x.dtype)
        z = enc(x)
        # 主體遮罩降到 latent 解析度：token 只要沾到主體就算
        zm = F.max_pool2d(m, kernel_size=8, stride=8)
        subj = (zm > 0.0).expand_as(z)
        full, sub = [], []
        for s in range(SEEDS):
            prm = ColorGridRandomParam(radius=0.60, grid=16, luma_bins=16, apply_where=w)
            prm.reset(x, seed=1000 + s)
            dz = enc(prm.render(x).clamp(0, 1)) - z
            full.append(float(dz.norm() / z.norm()))
            sub.append(float(dz[subj].norm() / z[subj].norm()))
        luma = (0.299*arr[..., 0]+0.587*arr[..., 1]+0.114*arr[..., 2])[w[0,0].bool().numpy()]
        rows.append({"image": name, "carrier_area": round(float(w.mean()), 5),
                     "luma_mean": round(float(luma.mean()), 4),
                     "dz_full": round(float(np.mean(full)), 5),
                     "dz_subject": round(float(np.mean(sub)), 5)})
        r = rows[-1]
        print(f"{name:<32s} 面積 {r['carrier_area']:.3f} 亮度 {r['luma_mean']:.3f}"
              f"  Δz 全圖 {r['dz_full']:.4f}  **Δz 主體 token {r['dz_subject']:.4f}**", flush=True)

    out = ROOT/"runs/carrier_survey"/"latent_headroom.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)
    def pear(a, b):
        a, b = np.array(a), np.array(b); a, b = a-a.mean(), b-b.mean()
        return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
    L=[r["luma_mean"] for r in rows]; A=[r["carrier_area"] for r in rows]
    print(f"\nΔz 主體 vs 亮度  r = {pear(L,[r['dz_subject'] for r in rows]):+.3f}")
    print(f"Δz 主體 vs 面積  r = {pear(A,[r['dz_subject'] for r in rows]):+.3f}")
    print(f"Δz 全圖 vs 亮度  r = {pear(L,[r['dz_full'] for r in rows]):+.3f}")
    print(f"寫出 {out}")


if __name__ == "__main__":
    main()
