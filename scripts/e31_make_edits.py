"""在本機產生未防禦的編輯輸出，供 E31 的劣化階梯定錨使用。

為什麼需要。p11 的階梯要以**編輯輸出**為來源（見該檔 `load_sources` 的說明），
而既有 `runs/` 中唯一以正確攻擊設定（w=7.5）產生的未防禦編輯只有 car_00 一張
——`runs/e29c_C_*` 與 `runs/e29c_P_*` 的 `edit_orig.png` 經雜湊比對是同一個檔，
因為 SDEdit 在相同影像／prompt／種子／strength 下是決定性的。用一張圖定錨門檻
太薄，而 E20 與 E28 兩次定錨都用了多個臂與多張圖。

本機實測（`runs/logs/e31_local_probe.log`）：RTX 2050 4 GB 跑得動無梯度的
512² SDEdit，峰值 4873 MB（超過實體記憶體，靠 Windows 的共享記憶體外溢），
單次 222.5 s。慢，但這是零雲端成本的，且只需跑數張。

種子取 `OptimConfig.seed + EVAL_SEED_OFFSET`，與 `run_defense.py::evaluate` 的
held-out 分支一致，故本腳本產生的圖與那裡的 `edit_orig.png` 同分佈。

執行：python scripts/e31_make_edits.py --limit 6
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.sd import SDWrapper
from src.utils.device import get_device

# 與 scripts/run_defense.py 的常數一致：評測用未見過的種子
EVAL_SEED_OFFSET = 10000
DEFAULT_SEED = 20260728


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--data", default="data/dayn_testset")
    ap.add_argument("--out", default="runs/e31_sources")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    from scripts.run_defense import load_images

    device = get_device()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    sd = SDWrapper(args.model)
    emb_u = sd.encode_text("")
    lat = sd.latent_shape(args.size, args.size)
    images = load_images(ROOT / args.data, args.size, device, args.limit)
    print(f"[edits] device={device} 影像 {[n for n, _, _ in images]}", flush=True)

    for name, x01, plist in images:
        prompt = plist[0]          # 訓練一律用第 [0] 個，與 run_defense 一致
        emb = sd.encode_text(prompt)
        t0 = time.perf_counter()
        with torch.no_grad():
            n = sd.sample_edit_noise(torch.empty(lat, device=device),
                                     seed=args.seed + EVAL_SEED_OFFSET)
            y = sd.sdedit(x01, emb, n, args.n_edit, strength=args.strength,
                          guidance_scale=args.guidance_scale, emb_uncond=emb_u)
        save_image(y.clamp(0, 1), out / f"{name}__edit_orig.png")
        save_image(x01.clamp(0, 1), out / f"{name}__orig.png")
        print(f"  {name} {prompt!r}  {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"[edits] 寫出 {out}")


if __name__ == "__main__":
    main()
