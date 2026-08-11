"""實驗一：原生 APA 階段一 LoRA + 自訂 budget，測保真。

回答指導者的問題「第一階段為何沒用 APA 原生方法，LoRA 去哪了」——
FND-016／FND-022（`docs/FINDINGS.md`）的「原文階段一在本管線上是空操作」
是在 `optimize.align()` 誤用本專案自己的保真 hinge 當階段一損失下量到的，
不是 APA 官方訓練目標（Eq.6 的 denoising MSE）本身。本腳本換回官方目標
（`optimize.align_apa_native`，逐字對照 `docs/reference/dia_apa.md` §1.2），
與 DEC-016 現行做法（latent 對齊 + 解碼器逐圖微調）並列比較。

budget 的定義（使用者 2026-08-11 裁決，採用推薦方案 b）：不限制步數，
跑滿官方 200 步固定超參數（rank=8/alpha=8、AdamW lr=1e-4、noise_offset=0.1，
`src/residual/site_apa.py` 的 `APA_STAGE1_*` 常數），只看它把 φ=0 的重建
下限壓到哪裡，對照 `DESIGN.md §3.2` 既有的 τ ∈ {0.20, 0.35}。

影像（使用者 2026-08-11 裁決，決策點 1）：APA 官方 repo
（github.com/deep-kaixun/APA）在 `images_un/` 附了 5 張真實影像與
`data.json` 的逐圖 ImageNet 標籤，故不用本專案自己的 PIE-Bench 子集，改用
官方影像與官方標籤（`data/apa_native/provenance.json` 記錄來源與
sha256）。文字條件 `c` 直接取官方的 class 欄位，不做模板化——這正是
`dia_apa.md` §1.2 查證到的官方作法（`visual_alignment.py:146,151`）。

比較臂：
    floor        現行路徑 φ=0（BDIA 反演 + 去噪 + decode，模組停用）
    lora_native  APA 官方階段一目標訓練出的 LoRA，φ 啟用後跑同一條路徑
    recon        DEC-016 現行做法（latent 對齊 + 解碼器逐圖微調），沿用
                 `scripts/recon_floor_ab.py` 的預設超參數以便與既有
                 EXP-s3t20_r 批次的數字對照

指標：LPIPS／SSIM／DISTS／PSNR（`MetricSuite.pairwise`，另存 L∞／VIFp／
FSIM 等既有欄位）、CLIP 影像-影像餘弦、NIMA-AVA、CNNIQA
（`src/metrics/aesthetic.AestheticSuite`，對照 APA 官方 Table 3 的欄位，
`dia_apa.md` §5.1）。

用法：
    python scripts/apa_native_lora_probe.py --out runs/apa_native_probe
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.defense import optimize, recon  # noqa: E402
from src.defense.generator import DefenseGenerator  # noqa: E402
from src.experiment import executors  # noqa: E402
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.residual.site_apa import (  # noqa: E402
    APA_LORA_ALPHA,
    APA_LORA_RANK,
    APA_NOISE_OFFSET,
    APA_STAGE1_LR,
    APA_STAGE1_STEPS,
)
from src.residual.site_weight import APA_BLOCKS, WeightResidual  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "apa_native"
# 與 EXP-s3t20／EXP-s3t20_r 同一條軌（SD v1.4／512²），使 `recon` 臂與既有
# DEC-016 批次的數字可比；APA 官方原生設定同為 SD v1.x 世代的 UNet 架構。
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
RESOLUTION = 512
# BDIA 精確反演的既有校準值（`src/defense/optimize.py` 模組 docstring：
# tiny-SD 實測 k=20, t_max=500 時 latent 來回誤差降到 1.37e-04）。
K_INV, T_MAX = 20, 500

# recon 臂（DEC-016 A 段）沿用 `scripts/recon_floor_ab.py` 的預設值。
RECON_KEY = "lpips"
RECON_W_PIXEL = 0.5
RECON_A1_STEPS, RECON_A1_LR = 300, 0.02
RECON_A2_STEPS, RECON_A2_LR = 300, 2e-3
RECON_GAMMA_ACUT, RECON_ACUT_BAND = 1.0, 0.05
RECON_FLOOR_RATIO = 0.50

ARMS = ("floor", "lora_native", "recon")
METRIC_KEYS = ("lpips", "ssim", "dists", "psnr", "clip_img", "nima", "cnniqa")


def load_dataset() -> list:
    """讀 `data/apa_native/provenance.json`，回傳 `[{name, class, path}]`。"""
    provenance = json.loads(
        (DATA_DIR / "provenance.json").read_text(encoding="utf-8"))
    return [
        {"name": p["output"][:-4], "class": p["apa_class"],
         "path": DATA_DIR / p["output"]}
        for p in provenance
    ]


def run_one(sd: SDWrapper, suite: MetricSuite, aes: AestheticSuite,
           item: dict, seed: int, out: Path) -> dict:
    name, class_name, path = item["name"], item["class"], item["path"]
    x01 = executors.load_image_tensor(path, sd.device, size=RESOLUTION)

    def pair(y: torch.Tensor) -> dict:
        return suite.pairwise(x01, y)

    def full_metrics(y: torch.Tensor) -> dict:
        m = pair(y)
        a = aes.measure(y)
        return {"lpips": m["lpips"], "ssim": m["ssim"], "dists": m["dists"],
               "psnr": m["psnr"], "clip_img": aes.clip_image_similarity(x01, y),
               "nima": a["nima"], "cnniqa": a["cnniqa"]}

    lora = WeightResidual(sd.unet, rank=APA_LORA_RANK, alpha=APA_LORA_ALPHA,
                          blocks=APA_BLOCKS, seed=seed).to(sd.device)
    gen = DefenseGenerator(sd, lora, k_inv=K_INV, t_max=T_MAX,
                           exact_inversion=True)

    # ---- 1. floor：φ=0，現行路徑的原點 ----
    lora.disable()
    with torch.no_grad():
        y_floor = gen.generate(x01, gen.prepare(x01)).detach()
    m_floor_full = full_metrics(y_floor)
    m_floor_raw = pair(y_floor)  # 供 recon 臂的 target/band 計算取原始值

    # ---- 2. lora_native：APA 官方階段一目標 ----
    t0 = time.time()
    history = optimize.align_apa_native(
        sd, lora, x01, class_name,
        steps=APA_STAGE1_STEPS, lr=APA_STAGE1_LR,
        noise_offset=APA_NOISE_OFFSET, seed=seed)
    train_seconds = time.time() - t0
    diverged = bool(history and history[-1].get("diverged"))
    lora.enable()
    with torch.no_grad():
        y_lora = gen.generate(x01, gen.prepare(x01)).detach()
    m_lora_full = full_metrics(y_lora)
    lora.remove()

    # ---- 3. recon：DEC-016 現行做法（A1 latent 對齊 + A2 解碼器微調）----
    tunable = recon.decoder_tunable(sd.vae.decoder)
    params = [p for _, p in tunable]
    band = recon.acutance_band(m_floor_raw["acutance_ratio"], RECON_ACUT_BAND)
    target = RECON_FLOOR_RATIO * m_floor_raw[RECON_KEY]
    z1, _h1, _s1 = recon.align_latent(
        sd, x01, suite.lpips_module, pair,
        steps=RECON_A1_STEPS, lr=RECON_A1_LR, key=RECON_KEY,
        w_pixel=RECON_W_PIXEL, gamma_acut=RECON_GAMMA_ACUT, band=band)
    with recon.restored(params):
        _h2, s2 = recon.finetune_decoder(
            sd, x01, z1, params, suite.lpips_module, pair,
            steps=RECON_A2_STEPS, lr=RECON_A2_LR, target=target,
            key=RECON_KEY, w_pixel=RECON_W_PIXEL,
            gamma_acut=RECON_GAMMA_ACUT, band=band)
        with torch.no_grad():
            y_recon = sd.decode_latent(z1)
        m_recon_full = full_metrics(y_recon)

    # ---- 4. 落盤 ----
    images = {"orig": x01, "floor": y_floor, "lora_native": y_lora,
             "recon": y_recon}
    for tag, img in images.items():
        save_image(img, out / f"{name}__{tag}.png")

    row = {"image": name, "class": class_name,
          "lora_train_seconds": round(train_seconds, 1),
          "lora_diverged": diverged,
          "recon_a2_reached": s2["reached"],
          "recon_a2_stop_step": s2["stop_step"]}
    for arm, m in (("floor", m_floor_full), ("lora_native", m_lora_full),
                  ("recon", m_recon_full)):
        for k in METRIC_KEYS:
            row[f"{arm}_{k}"] = round(m[k], 4)
    return row


def b64(path: Path) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii"))


CSS = """body{background:#16171b;color:#e8e8ea;font:13px/1.5 system-ui,
sans-serif;margin:24px}h1{font-size:18px}h2{font-size:15px;margin-top:28px}
table{border-collapse:collapse}td,th{border:1px solid #2e3038;padding:6px;
text-align:center;vertical-align:top}th{background:#1e2027}
img{display:block;width:220px;height:220px;object-fit:contain;background:#000}
.m{font:11px ui-monospace,monospace;color:#9aa0aa}.hint{color:#9aa0aa;
max-width:64em}"""


def cell(out: Path, name: str, arm: str, row: dict) -> str:
    m = "<br>".join(f"{k} {row[f'{arm}_{k}']}" for k in METRIC_KEYS)
    return (f"<td><img src='{b64(out / f'{name}__{arm}.png')}'>"
           f"<span class=m>{m}</span></td>")


def build_page(rows: list, out: Path) -> Path:
    h = ["<!-- 由 scripts/apa_native_lora_probe.py 產生 -->",
        "<meta charset='utf-8'>", f"<style>{CSS}</style>",
        "<h1>實驗一：原生 APA 階段一 LoRA vs. DEC-016 latent+解碼器對齊</h1>",
        "<p class=hint>影像取自 APA 官方 repo（github.com/deep-kaixun/APA "
        "的 images_un/，見 data/apa_native/provenance.json）；文字條件 "
        "為官方 data.json 的 class 欄。第 2 欄是現行路徑 φ=0 的重建下限"
        "（BDIA 反演＋去噪＋decode，模組停用）。第 3 欄是 APA 官方階段一"
        "目標（denoising MSE，200 步固定超參數）訓練出的 LoRA，啟用後跑同一"
        "條路徑。第 4 欄是 DEC-016 現行做法（latent 對齊＋解碼器逐圖微調）。"
        "LPIPS／DISTS 愈小愈好，SSIM／PSNR／CLIP-img/NIMA/CNNIQA 愈大愈好。"
        "</p>",
        "<table><tr><th>影像</th><th>原圖</th><th>floor<br>"
        "<span class=m>φ=0</span></th><th>lora_native<br>"
        "<span class=m>APA 官方階段一</span></th><th>recon<br>"
        "<span class=m>DEC-016</span></th><th>訓練秒數／備註</th></tr>"]
    for r in rows:
        name = r["image"]
        h.append(f"<tr><td>{name}<br><span class=m>{r['class']}</span></td>")
        h.append(f"<td><img src='{b64(out / f'{name}__orig.png')}'></td>")
        for arm in ARMS:
            h.append(cell(out, name, arm, r))
        h.append(f"<td class=m>LoRA 訓練 {r['lora_train_seconds']} 秒"
                 f"{'（發散）' if r['lora_diverged'] else ''}<br>"
                 f"recon A2 {'達到' if r['recon_a2_reached'] else '未達'}"
                 f"目標於第 {r['recon_a2_stop_step']} 步</td>")
        h.append("</tr>")
    h.append("</table>")
    page = out / "compare.html"
    page.write_text("\n".join(h), encoding="utf-8")
    return page


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32",
                    help="計算精度。APA 官方以 fp32 訓練 LoRA，預設沿用")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=dtype)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)

    rows = []
    for item in load_dataset():
        print(f"=== {item['name']} ({item['class']}) ===", flush=True)
        row = run_one(sd, suite, aes, item, args.seed, args.out)
        rows.append(row)
        print(row, flush=True)

    executors.write_csv(args.out / "apa_native_probe.csv", rows)
    page = build_page(rows, args.out)
    print(f"\n表：{args.out / 'apa_native_probe.csv'}\n比對頁：{page}")


if __name__ == "__main__":
    main()
