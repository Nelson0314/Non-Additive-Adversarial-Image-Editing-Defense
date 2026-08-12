"""掃描未防禦編輯的設定，找出「攻擊確實成功」的操作點。

動機：`apa_native_full_pipeline` 首輪的 `edit_orig`（未防禦的編輯結果）在
butterfly 上與原圖幾乎沒有差別——攻擊本身沒有發生。防禦有沒有效是拿
「防禦後的編輯」對「未防禦的編輯」比的，分母不成立時整組數字都沒有意義。
這與 `data/lo_aligned/prompts.yaml` 第 34–38 行記的是同一個失效模式
（「跳太大的編輯連未防禦的那一側都難說成功」）。

本腳本只跑攻擊、不跑防禦：對每張圖掃 strength × prompt，把結果排成一頁
供人眼判「這個編輯有沒有真的發生」。判準是人眼（`DESIGN` §1.1），
CLIP-T 與 SigLIP 只作為輔助欄位。

用法：
    python scripts/apa_native_edit_sweep.py --out runs/apa_edit_sweep
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.experiment import executors  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "apa_native"
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
RESOLUTION = 512
EDIT_GUIDANCE = 7.5
EDIT_STEPS = 30
EDIT_SEED = 20260812

STRENGTHS = [0.4, 0.55, 0.7, 0.85]

# 逐圖的候選 prompt。第一個是現行的，其餘是「改動幅度較小、但仍看得出來」
# 的替代——原文的教訓是跳太大時連未防禦那一側都不成功。
CANDIDATES = {
    "butterfly": [
        "a group of monarch butterflies resting on a red rose",
        "a butterfly resting on a bright red rose",
        "an orange monarch butterfly on a dried thistle",
    ],
    "coot": [
        "a white swan gliding on a golden lake at sunset",
        "a white swan on a lake",
    ],
    "panda": [
        "a giant panda astronaut floating in outer space",
        "a brown grizzly bear resting on grass",
    ],
}


def b64(path: Path, max_side=260, q=85) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=["butterfly", "coot"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    emb_uncond = sd.uncond_prompt()

    rows = []
    for name in args.images:
        path = DATA_DIR / f"{name}.png"
        x01 = executors.load_image_tensor(path, sd.device, size=RESOLUTION)
        save_image(x01, args.out / f"{name}__orig.png")
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=EDIT_SEED)
        for pi, prompt in enumerate(CANDIDATES[name]):
            emb = sd.encode_text(prompt)
            for st in STRENGTHS:
                with torch.no_grad():
                    y = sd.sdedit(x01, emb, noise, EDIT_STEPS, strength=st,
                                  guidance_scale=EDIT_GUIDANCE,
                                  emb_uncond=emb_uncond).detach()
                tag = f"{name}__p{pi}__s{st:g}"
                save_image(y, args.out / f"{tag}.png")
                m = suite.pairwise(x01, y)
                sem = suite.semantic(y, prompt)
                sem_o = suite.semantic(x01, prompt)
                row = {"image": name, "prompt_index": pi, "prompt": prompt,
                       "strength": st, "tag": tag,
                       "lpips_vs_orig": round(m["lpips"], 4),
                       "clip_orig": round(sem_o["clip"], 4),
                       "clip_edit": round(sem["clip"], 4),
                       "clip_gain": round(sem["clip"] - sem_o["clip"], 4),
                       "siglip_gain": round(sem["siglip"] - sem_o["siglip"], 4)}
                rows.append(row)
                print(row, flush=True)
    executors.write_csv(args.out / "edit_sweep.csv", rows)

    # 比對頁
    h = ["<meta charset='utf-8'>", "<title>編輯強度掃描</title>", """<style>
body{background:#12151b;color:#e7eaef;font:14px/1.6 system-ui,sans-serif;margin:24px}
h2{font-size:17px;margin:28px 0 4px}h3{font-size:13px;color:#98a1ae;margin:14px 0 6px;
font-family:ui-monospace,monospace;font-weight:400}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start}
figure{margin:0;width:220px;background:#191d25;border:1px solid #2b323e;border-radius:8px;
overflow:hidden}figure img{display:block;width:100%;aspect-ratio:1;object-fit:cover}
figcaption{padding:6px 8px;font:11px ui-monospace,monospace;color:#98a1ae}
b{color:#e7eaef}.hint{color:#98a1ae;max-width:70em}
</style>"""]
    h.append("<h1>未防禦編輯的強度掃描</h1><p class=hint>判準是人眼：這個編輯有沒有"
             "真的發生。CLIP 增益只是輔助——它可以在畫面幾乎沒變時仍為正。"
             "要挑的是「明顯照著 prompt 改了、但構圖還認得出是同一張照片」的那一格。</p>")
    for name in args.images:
        h.append(f"<h2>{name}</h2>")
        h.append(f"<div class=row><figure><img src='{b64(args.out / f'{name}__orig.png')}'>"
                 f"<figcaption><b>原圖</b></figcaption></figure></div>")
        for pi, prompt in enumerate(CANDIDATES[name]):
            h.append(f"<h3>p{pi} · {prompt}</h3><div class=row>")
            for st in STRENGTHS:
                tag = f"{name}__p{pi}__s{st:g}"
                r = next(x for x in rows if x["tag"] == tag)
                h.append(f"<figure><img src='{b64(args.out / f'{tag}.png')}'>"
                         f"<figcaption><b>strength {st:g}</b><br>"
                         f"LPIPS {r['lpips_vs_orig']}<br>"
                         f"CLIP {r['clip_orig']}→{r['clip_edit']} ({r['clip_gain']:+})"
                         f"</figcaption></figure>")
            h.append("</div>")
    page = args.out / "edit_sweep.html"
    page.write_text("\n".join(h), encoding="utf-8")
    print(f"\n表：{args.out / 'edit_sweep.csv'}\n比對頁：{page}")


if __name__ == "__main__":
    main()
