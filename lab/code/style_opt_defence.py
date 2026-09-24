"""風格由最佳化選，但交付的仍然只是一張色彩濾鏡：身分由參數化保證。

這一臂在回答什麼
────────────────────────────────────────────────────────────────────
生成載體的三個臂到目前為止**都沒有最佳化**——風格句是防禦方隨手選的，
所以「現成的生成模型能推動多少」與「選對風格能推動多少」這兩件事還沒分開。
這一臂把風格**選**出來，其餘逐項與 `style_affine` 相同，兩臂因此可以直接相減。

為什麼不是最佳化整個 77×768 的 embedding
────────────────────────────────────────────────────────────────────
自由的 contextual embedding 可以離開 CLIP 的訓練分布，而本專案已經量到
**三道自然度門檻都被最佳化鑽過**，事後加門檻守不住。所以變數被限制在一個
**風格字典的凸包**上：

    a = softmax(logits)                    a_j ≥ 0、Σ a_j = 1
    emb = Σ_j a_j · CLIP(style_j)

十個字典項都是真的攝影調性句，混合仍在它們張成的凸包內。這是「自然度進
參數化」的具體形式，不是再加一道正則項。**凸包不是自然度的證書**——混合
embedding 仍可能落在沒有訓練樣本的地方，所以逐張看圖照樣是判準。

交付的東西與 `style_affine` 完全相同
────────────────────────────────────────────────────────────────────
    raw   = SDEdit(x ; emb(a), noise(seed), num_steps, strength, w)
    q     = localAffine(guide=x, src=raw, radius, eps)      三通道局部仿射
    α(p)  = a_bg·(1 − w(p)) + a_face·w(p)
    x_def = clamp( x + α(p)·(q − x) , 0, 1 )

`q` 被限制成 `x` 的局部仿射函數，所以邊只能是原圖的邊乘一個係數——不會長出
新結構，身分不會被換掉。生成模型只決定**顏色往哪走**。

三組變數，同一個學習率
────────────────────────────────────────────────────────────────────
| 變數 | 形狀 | 可行域 | 作用 |
|---|---|---|---|
| `logits` | (D,) | 無界（softmax 之後在單體上） | 選風格 |
| `alpha` | (2,) | `[0, --max-scale]` | 背景與臉的振幅 |

`alpha` 是**變數**而不是像 `style_affine` 那樣用二分搜尋定出來的：二分搜尋
不可微，把它留在迴圈裡會讓梯度看不到「振幅可以換分數」這件事。改成變數之後
兩道 ΔE00 上限就走與 `curve_*` 相同的增廣 Lagrange，**約束進損失、不做事後
投影**，與本專案其餘的臂同一套。

約束
────────────────────────────────────────────────────────────────────
`frame` ΔE00 ≤ 16、`face_box` ≤ 8、`skin_colour` ≤ 8（與原膚色同色的像素，
不限位置）、`chroma_p95` ≤ 原圖 p95 × 1.15。後兩道是本專案量到的「色偏只有
兩種情形難看」的直接實作。

求解端不含任何文字條件：目標是 `FreeObjective`，三個項都不經過 text encoder。
風格字典是**防禦方自己選的設定**，與攻擊指令屬於不同的人。

成本
────────────────────────────────────────────────────────────────────
每一步要跑一次可微分的 SDEdit（`--num-steps` 步 × 2 條 CFG 分支，UNet 與 VAE
都開 gradient checkpointing）加上 `FreeObjective` 自己的鏈，比 `curve_*` 的
單步貴一個量級。`--steps` 預設 60 而不是 900，**這是成本決定的，不是收斂
判定**；逐步的 `free_curve` 照樣寫進 CSV，收斂與否由那條曲線自己說。

產出（版面與 `defence_run.py` 相同）
────────────────────────────────────────────────────────────────────
    {out}/{image}__orig.png
    {out}/{image}__{arm}__def.png
    {out}/{image}__raw.png        SDEdit 的原始輸出（未轉移）
    {out}/{image}__field.png      α(p)
    {out}/results.csv             含 `weights` 欄：字典的混合係數
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from colour_support import chroma_p95, skin_colour_support  # noqa: E402
from curve_budget_defence import (  # noqa: E402
    box_support, expanded_box, load_images, smooth_field, write_rows,
)
from style_filter_defence import guided_filter_colour  # noqa: E402
from src.defense.color_amplitude import delta_e00  # noqa: E402
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.defense.immunise import Cap, optimise_carrier, quantise  # noqa: E402
from src.defense.instruction_free import FreeObjective  # noqa: E402
from src.defense.uniformity import lab_offset, tv_offset  # noqa: E402
from src.metrics.identity import embed_box, face_boxes  # noqa: E402
from src.models.ip2p import IP2PWrapper  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
CARRIER_MODEL = "runwayml/stable-diffusion-v1-5"
SOLVER_PROMPT = ("風格字典（防禦方設定）",
                 "求解端不含攻擊指令；FreeObjective 的三個項不經過 text encoder")

#: 風格字典。**這裡是 SDEdit 不是 inpainting**，所以媒材詞（film、grain）
#: 會變成色調而不是物件——`inpaint_region_defence.py` 那邊相反，不要互抄。
STYLE_DICT = [
    "a vintage cross-processed film photograph, heavy grain, teal and orange "
    "colour grading, faded highlights",
    "a photograph with a strong cyan-magenta colour filter, high contrast, "
    "cold cinematic grade",
    "a warm golden-hour photograph, amber light, soft lifted shadows",
    "a cold overcast photograph, desaturated blue-grey tones, flat contrast",
    "a bleach-bypass photograph, silvery desaturated colour, crushed blacks",
    "a sepia-toned photograph, warm brown monochrome cast",
    "an infrared-style photograph, magenta foliage, pale sky",
    "a high-key photograph, lifted blacks, pastel washed colour",
    "a deep green-tinted photograph, forest cast, heavy shadows",
    "a sunset-graded photograph, orange highlights and violet shadows",
]


class StyleSimplexAffine:
    """風格字典凸包上的 SDEdit ＋ 三通道局部仿射轉移。

    `logits` 與 `alpha` 的自然尺度接近（一個是 softmax 前的分數、一個是
    `[0, max_scale]` 的振幅），所以共用一個 Adam 學習率沒有問題；
    `ab_warp` 那邊要另外做無量綱化是因為它的兩組差兩個數量級。
    """

    name = "style_simplex_affine"

    def __init__(self, sd, embs, uncond, noise, field, args):
        self.sd = sd
        self.embs = embs                  # (D,1,77,768)，常數
        self.uncond = uncond
        self.noise = noise
        self.field = field
        self.args = args
        self.logits = None
        self.alpha = None

    # ---- 介面 ----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        d = self.embs.shape[0]
        dev = x01.device
        self.logits = torch.zeros(d, device=dev, dtype=torch.float32,
                                  requires_grad=True)
        self.alpha = torch.tensor([1.0, 0.6], device=dev, dtype=torch.float32,
                                  requires_grad=True)

    def params(self):
        return [self.logits, self.alpha]

    @torch.no_grad()
    def project(self) -> None:
        self.alpha.clamp_(0.0, self.args.max_scale)

    def state_dict(self):
        return {"logits": self.logits.detach().clone(),
                "alpha": self.alpha.detach().clone()}

    def load_state_dict(self, state):
        self.logits = state["logits"].clone().requires_grad_(True)
        self.alpha = state["alpha"].clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, a):
        if isinstance(a, (list, tuple)):
            a = a[0]
        self.amplitude = float(a)

    def step_scale(self) -> float:
        return float(self.args.max_scale)

    def set_radius(self, r: float) -> None:
        self.args.max_scale = float(r)

    # ---- 構造 ----

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, 0)

    def embedding(self) -> torch.Tensor:
        a = self.weights.view(-1, 1, 1, 1)
        return (a * self.embs).sum(0)

    def raw(self, x01: torch.Tensor) -> torch.Tensor:
        return self.sd.sdedit(
            x01, self.embedding(), self.noise,
            num_steps=self.args.num_steps, strength=self.args.strength,
            guidance_scale=self.args.guidance, emb_uncond=self.uncond,
            use_ckpt=True, vae_ckpt=True).clamp(0, 1)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        raw = self.raw(x01).to(x01.dtype)
        q = guided_filter_colour(x01, raw, self.args.radius, self.args.eps)
        w = self.field.to(device=x01.device, dtype=x01.dtype)
        a = self.alpha.to(x01.dtype)
        alpha = a[0] * (1.0 - w) + a[1] * w
        return (x01 + alpha * (q - x01)).clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--num-steps", type=int, default=10,
                    help="可微分 SDEdit 的步數。每一個最佳化步要跑一次它的"
                         "前向與反向，所以這個數直接乘在總成本上")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--radius", type=int, default=48)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--max-scale", type=float, default=3.0)
    ap.add_argument("--feather", type=float, default=0.35)
    ap.add_argument("--frame-cap", type=float, default=16.0)
    ap.add_argument("--face-cap", type=float, default=8.0)
    ap.add_argument("--skin-radius", type=float, default=12.0)
    ap.add_argument("--chroma-gain", type=float, default=1.15)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--lr-final-ratio", type=float, default=0.25)
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--lam-every", type=int, default=5)
    ap.add_argument("--check-every", type=int, default=5)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--attack-steps", type=int, default=50)
    ap.add_argument("--s-i", type=float, default=1.5)
    ap.add_argument("--dtype", default="float16", choices=("float32", "float16"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sd = SDWrapper(CARRIER_MODEL, dtype=getattr(torch, args.dtype))
    ip2p = IP2PWrapper(dtype=torch.float32)
    device = ip2p.device

    with torch.no_grad():
        embs = torch.stack([sd.encode_text(p)[0] for p in STYLE_DICT]).unsqueeze(1)
        uncond = sd.uncond_prompt().detach()
    print(f"風格字典 {embs.shape[0]} 項，embedding {tuple(embs.shape)}", flush=True)

    items = load_images(args.data, set(args.images) if args.images else None)
    rows = []
    for item in items:
        started = time.time()
        x = load_image_tensor(item["path"], device, size=RESOLUTION)
        boxes = face_boxes(x, device)
        if not boxes:
            raise SystemExit(f"{item['name']} 偵測不到臉；膚色群心與身分讀數"
                             "都沒有錨點，不要靜默退回單一預算")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        frame = torch.ones_like(x[:, :1])
        face = box_support(x, box)
        skin = skin_colour_support(x, face, args.skin_radius)
        field = smooth_field(x, box, args.feather)
        c95 = float(chroma_p95(x))
        chroma_cap = c95 * args.chroma_gain

        with torch.no_grad():
            noise = sd.sample_edit_noise(sd.encode_image(x), args.seed)
        carrier = StyleSimplexAffine(sd, embs, uncond, noise, field, args)
        carrier.reset(x, args.seed)

        caps = [
            Cap("frame", lambda y: delta_e00_torch(x, y, frame),
                lambda y: delta_e00(x, y, frame), args.frame_cap),
            Cap("face_box", lambda y: delta_e00_torch(x, y, face),
                lambda y: delta_e00(x, y, face), args.face_cap),
            Cap("skin_colour", lambda y: delta_e00_torch(x, y, skin),
                lambda y: delta_e00(x, y, skin), args.face_cap),
            Cap("chroma_p95", lambda y: chroma_p95(y),
                lambda y: float(chroma_p95(y)), chroma_cap),
        ]
        objective = FreeObjective(
            ip2p, x, box=box, k=4, steps=args.attack_steps, seed=args.seed,
            weights={"id": 1.0, "enc": 0.5, "cond": 1.0},
            chain_steps=6, grad_steps=1, s_i=args.s_i, resample=True)

        stats = optimise_carrier(
            carrier, x, objective, steps=args.steps, lr=args.lr, caps=caps,
            rho=args.rho, lam_every=args.lam_every, check_every=args.check_every,
            log_every=args.log_every, lr_final_ratio=args.lr_final_ratio,
            probe_every=args.probe_every)

        with torch.no_grad():
            y = quantise(carrier.render(x))
            raw = carrier.raw(x).to(x.dtype)
            off = lab_offset(x, y)
            w = carrier.weights.detach()
            id0 = embed_box(x, box, device).reshape(1, -1)
            cos = torch.nn.functional.cosine_similarity
            row = {
                "image": item["name"], "class": item["class"], "arm": args.arm,
                "carrier": carrier.name, "carrier_model": CARRIER_MODEL,
                "strength": args.strength, "num_steps": args.num_steps,
                "guidance": args.guidance, "seed": args.seed,
                "radius": args.radius, "eps": args.eps,
                "max_scale": args.max_scale, "feather": args.feather,
                "frame_cap": args.frame_cap, "face_cap": args.face_cap,
                "skin_radius": args.skin_radius, "chroma_gain": args.chroma_gain,
                "dict_size": len(STYLE_DICT),
                "weights": "|".join(f"{v:.4f}" for v in w.tolist()),
                "weight_top": int(w.argmax()),
                "weight_top_prompt": STYLE_DICT[int(w.argmax())],
                "weight_entropy": round(float(-(w * w.clamp_min(1e-9).log()).sum()), 4),
                "a_bg": round(float(carrier.alpha[0]), 5),
                "a_face": round(float(carrier.alpha[1]), 5),
                "solver_prompt": SOLVER_PROMPT[0],
                "solver_prompt_source": SOLVER_PROMPT[1],
                "deltaE00_frame": round(float(delta_e00(x, y, frame)), 4),
                "deltaE00_face_box": round(float(delta_e00(x, y, face)), 4),
                "deltaE00_skin_colour": round(float(delta_e00(x, y, skin)), 4),
                "chroma_p95_orig": round(c95, 4),
                "chroma_p95_cap": round(chroma_cap, 4),
                "chroma_p95_out": round(float(chroma_p95(y)), 4),
                "tv_frame": round(float(tv_offset(off, frame)), 5),
                "psnr": round(float(10 * torch.log10(1.0 / (y - x).pow(2).mean())), 4),
                "linf": round(float((y - x).abs().max()), 5),
                "id_cos": round(float(cos(id0, embed_box(y, box, device).reshape(1, -1))), 5),
                "id_cos_sdedit_raw": round(
                    float(cos(id0, embed_box(raw, box, device).reshape(1, -1))), 5),
                "seconds": round(time.time() - started, 1),
                **{k: v for k, v in stats.items()},
            }
        save_image(x, args.out / f"{item['name']}__orig.png")
        save_image(y, args.out / f"{item['name']}__{args.arm}__def.png")
        save_image(raw, args.out / f"{item['name']}__raw.png")
        save_image((carrier.alpha.detach().max().clamp_min(1e-6).reciprocal()
                    * (carrier.alpha[0] * (1 - field) + carrier.alpha[1] * field)
                    ).detach().repeat(1, 3, 1, 1).clamp(0, 1),
                   args.out / f"{item['name']}__field.png")
        rows.append(row)
        write_rows(args.out / "results.csv", rows)
        print(f"[{args.arm}] {item['name']}  ΔE frame {row['deltaE00_frame']} "
              f"face {row['deltaE00_face_box']}  id {row['id_cos']} "
              f"(raw {row['id_cos_sdedit_raw']})  頂點 {row['weight_top']} "
              f"熵 {row['weight_entropy']}  違反 {row['free_cap_violations']}  "
              f"{row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
