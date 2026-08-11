"""對照實驗：APA 官方 LoRA 在「真的有 DDIM 反演誤差」的條件下有沒有用。

動機（使用者質疑 `apa_native_lora_probe.py` 的負向結果，要求逐字核對原文與
原始碼）：重讀 APA 論文 §3.3（`arXiv:2506.01511` p.3-4）後發現兩件事：

1. 論文原文："without perturbations to the zT obtained via DDIM Inversion,
   the model **nearly reconstructs** the clean image after T denoising
   steps"——VCA 的敘事起點就是「未擾動時 DDIM 反演已經『幾乎』能重建」，
   VCA 的目的寫在下一句："enabling the model to generate visually
   consistent outputs **whether regular noise or adversarial noise is
   applied to zT**"（p.4，Eq.6 之後）。VCA 修的是**對 z_T 擾動的穩健性**
   （Figure 3：橫軸是 noise scale 0→0.4/0.5，w/VCA 全程掉得比 w/o VCA 慢），
   不是「零擾動時的重建誤差」本身。

2. 但指導者提供的 rebuttal 表（DDIM 0.19 → DDIM+VCA 0.05，"under a
   non-attacking setting"）量的正是零擾動起點，說明 VCA 在該點也確實把
   **DDIM 反演本身的誤差**（非精確反演造成的、與擾動無關的殘留誤差）
   壓低了。這與本專案 `bdia_inversion` docstring 的量測一致：DDIM 反演
   「反演時 ε 在 z_i 評估、去噪時在 z_{i+1} 評估，兩者不同，誤差逐步累積」，
   BDIA 存在的理由正是消掉這個誤差（`bdia_inversion` 的算法背景）。

**本腳本要驗證的假設**：`apa_native_lora_probe.py` 量到「lora_native 比
floor 差」，會不會是我方 `align_apa_native` 的重現本身有問題？驗法是在
**真的有 DDIM 反演誤差**的條件下（即走 `exact_inversion=False` 的 DDIM
路徑，而非 BDIA），用同一個 `align_apa_native` 訓練出的 LoRA 是否確實
把誤差壓低——若壓得下來，代表訓練程式碼本身沒有問題，`apa_native_lora_
probe.py` 的負向結果是 BDIA 已消掉目標誤差的**真結果**，不是 bug；
若壓不下來，代表訓練程式碼有問題，須另外除錯。

同時比較 BDIA 路徑（重用同一組訓練好的 φ，不重新訓練——`align_apa_native`
訓練時不接觸生成鏈，φ 與反演方式無關），確認同一組 LoRA 在兩種反演方式下
的效果方向是否相反（DDIM 下改善、BDIA 下惡化），這才是「BDIA 已經沒有
反演誤差可修」這個因果解釋的直接證據。

DDIM 路徑取 `k_inv=50, t_max=None`——對應論文 §4.1 "APA-SG adopts the
entire inversion step of T=50"，全範圍反演，不像 `apa_native_lora_probe.
py` 的 BDIA 對照組取 `t_max=500`（那是本專案為了在 k_inv=10~20 這種較少
步數下也有可用的 φ=0 基準而做的縮限，見 `generator.py` 的 t_max 註解，
不是 APA 原文的設定）。

用法：
    python scripts/apa_native_lora_ddim_control.py --out runs/apa_native_ddim_control
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

from src.defense import optimize  # noqa: E402
from src.defense.generator import DefenseGenerator  # noqa: E402
from src.experiment import executors  # noqa: E402
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
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
RESOLUTION = 512

# DDIM：論文 §4.1 的 APA-SG 設定，全範圍反演。
DDIM_K_INV, DDIM_T_MAX = 50, None
# BDIA：與 `apa_native_lora_probe.py` 同一組，供方向對照。
BDIA_K_INV, BDIA_T_MAX = 20, 500


def load_dataset() -> list:
    provenance = json.loads(
        (DATA_DIR / "provenance.json").read_text(encoding="utf-8"))
    return [
        {"name": p["output"][:-4], "class": p["apa_class"],
         "path": DATA_DIR / p["output"]}
        for p in provenance
    ]


def run_one(sd: SDWrapper, suite: MetricSuite, item: dict, seed: int,
           out: Path) -> dict:
    name, class_name, path = item["name"], item["class"], item["path"]
    x01 = executors.load_image_tensor(path, sd.device, size=RESOLUTION)

    def pair(y: torch.Tensor) -> dict:
        m = suite.pairwise(x01, y)
        return {"lpips": m["lpips"], "ssim": m["ssim"], "dists": m["dists"],
               "psnr": m["psnr"]}

    lora = WeightResidual(sd.unet, rank=APA_LORA_RANK, alpha=APA_LORA_ALPHA,
                          blocks=APA_BLOCKS, seed=seed).to(sd.device)
    gen_ddim = DefenseGenerator(sd, lora, k_inv=DDIM_K_INV, t_max=DDIM_T_MAX,
                                exact_inversion=False)
    gen_bdia = DefenseGenerator(sd, lora, k_inv=BDIA_K_INV, t_max=BDIA_T_MAX,
                                exact_inversion=True)

    # ---- floor：φ=0，兩種反演各一次 ----
    lora.disable()
    with torch.no_grad():
        y_floor_ddim = gen_ddim.generate(x01, gen_ddim.prepare(x01)).detach()
        y_floor_bdia = gen_bdia.generate(x01, gen_bdia.prepare(x01)).detach()
    m_floor_ddim = pair(y_floor_ddim)
    m_floor_bdia = pair(y_floor_bdia)

    # ---- 訓練 APA 官方階段一目標（與反演方式無關，訓練時不碰生成鏈）----
    t0 = time.time()
    history = optimize.align_apa_native(
        sd, lora, x01, class_name,
        steps=APA_STAGE1_STEPS, lr=APA_STAGE1_LR,
        noise_offset=APA_NOISE_OFFSET, seed=seed)
    train_seconds = time.time() - t0
    diverged = bool(history and history[-1].get("diverged"))

    # ---- lora_native：同一組訓練好的 φ，兩種反演各跑一次 ----
    lora.enable()
    with torch.no_grad():
        y_lora_ddim = gen_ddim.generate(x01, gen_ddim.prepare(x01)).detach()
        y_lora_bdia = gen_bdia.generate(x01, gen_bdia.prepare(x01)).detach()
    m_lora_ddim = pair(y_lora_ddim)
    m_lora_bdia = pair(y_lora_bdia)
    lora.remove()

    for tag, img in (("orig", x01), ("floor_ddim", y_floor_ddim),
                     ("lora_ddim", y_lora_ddim), ("floor_bdia", y_floor_bdia),
                     ("lora_bdia", y_lora_bdia)):
        save_image(img, out / f"{name}__{tag}.png")

    row = {"image": name, "class": class_name,
          "lora_train_seconds": round(train_seconds, 1),
          "lora_diverged": diverged}
    for tag, m in (("floor_ddim", m_floor_ddim), ("lora_ddim", m_lora_ddim),
                  ("floor_bdia", m_floor_bdia), ("lora_bdia", m_lora_bdia)):
        for k, v in m.items():
            row[f"{tag}_{k}"] = round(v, 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)

    rows = []
    for item in load_dataset():
        print(f"=== {item['name']} ({item['class']}) ===", flush=True)
        row = run_one(sd, suite, item, args.seed, args.out)
        rows.append(row)
        print(row, flush=True)

    executors.write_csv(args.out / "ddim_control.csv", rows)
    print(f"\n表：{args.out / 'ddim_control.csv'}")


if __name__ == "__main__":
    main()
