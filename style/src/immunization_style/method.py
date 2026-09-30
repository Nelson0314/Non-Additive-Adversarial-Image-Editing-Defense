"""ip2p 風格編輯當載體（SPA，Wang et al., Neurocomputing 2026 的移植）。

x_ref = G(x, e_txt)、x_def = G(x, e_adv; z_off)。G 與攻擊端同一個 ip2p 權重，DDIM `steps` 步：
前 `warm` 步 eta=1 且不回傳梯度，其後 eta=0 並回傳梯度；兩條軌跡共用起始噪聲與暖身噪聲。
可調的是指令 token（含第一個 EOS）的 embedding 與暖身結束時的 latent 位移。

訓練目標一律不使用任何攻擊指令（評估指令或自選指令皆不用），文字只用空字串或類別詞：
`free`（空指令 ip2p 代理，錨在 x_ref）、`enc_grey`（攻擊端影像條件 E(y) 推向灰圖 latent）、
`attn`（攻擊端 UNet 自注意力圖相對 x_ref 的偏離，最大化）、`xattn`（攻擊端對類別詞的交叉注意力，
最小化）、`chaos`（攻擊端以類別詞為指令的短程編輯輸出，最大化其與輸入的結構差異）、
`classifier`（SPA 原損失：ResNet-50 對防禦圖的指數邊界損失，標籤取原圖 top-1）。
`--sampler ddpm --prompt-scope full --select last` 為 SPA 論文設定（附錄 B、整段 77 token、取最後一步）。
身分以 FaceNet 餘弦下限 τ = max(id_floor, cos(x_ref, x) − id_margin) 約束；`--struct-cap` 以灰階
LPIPS 限制防禦圖相對 x_ref 的結構改動。`--freeze-warm` 讓暖身段固定用原風格指令，梯度不再被截斷。
`--updates 0` 只產參考圖，即風格指令的無最佳化預覽。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint as ckpt  # noqa: E402

from color_defence import channel_shift_p95, load_images, write_rows  # noqa: E402
from src.defense.immunise import quantise  # noqa: E402
from src.defense.instruction_free import FreeObjective  # noqa: E402
from src.defense.ncf_param import rgb_to_lab  # noqa: E402
from src.metrics.identity import embed_box, embed_box_differentiable, face_boxes  # noqa: E402
from src.models.ip2p import IP2PWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
STYLES = {
    "noedit": "make no edit",
    "cool_grade": "apply subtle cool cinematic grading",
    "overcast": "use soft cool overcast lighting",
    "bluegreen": "apply a muted blue-green photographic look",
    "winter": "make it winter",
    "winter_day": "make it look like a cold winter day",
    # SPA 論文 §4.2 的五句風格指令
    "p_noedit": "Make no edit.",
    "p_snow": "Add some snow.",
    "p_light": "Add some light.",
    "p_night": "Make it at night.",
    "p_fog": "Make it in fog.",
}


def encode_text(ip2p, prompt):
    tok = ip2p.pipe.tokenizer(prompt, padding="max_length",
                              max_length=ip2p.pipe.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
    ids = tok.input_ids.to(ip2p.device)
    with torch.no_grad():
        emb = ip2p.text_encoder(ids)[0]
    return ids[0], emb


class StyleEditor:
    """`sampler="ddim"`：暖身 eta=1、其後 eta=0。`sampler="ddpm"`：SPA 附錄 B 的 (B.5)／(B.6)，
    暖身段為 DDPM 後驗平均加 σ_t δ（σ_t² = 1 − ᾱ_t／ᾱ_prev），其後只取後驗平均。"""

    def __init__(self, ip2p, x, *, steps, warm, s_t, s_i, seed, sampler="ddim"):
        from diffusers import DDIMScheduler

        self.ip2p, self.s_t, self.s_i, self.warm, self.sampler = ip2p, s_t, s_i, warm, sampler
        dev = ip2p.device
        self.sched = DDIMScheduler.from_config(
            ip2p.pipe.scheduler.config, clip_sample=False, thresholding=False,
            set_alpha_to_one=False, timestep_spacing="leading")
        self.sched.set_timesteps(steps, device=dev)
        if not 0 <= warm < len(self.sched.timesteps):
            raise ValueError(f"warm={warm} 必須小於步數 {len(self.sched.timesteps)}")
        dtype = ip2p.unet.dtype
        with torch.no_grad():
            self.cond = ip2p.image_latents(x).to(dtype)
            self.null = encode_text(ip2p, "")[1].to(dtype)
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        self.z_T = torch.randn(self.cond.shape, generator=gen, device=dev,
                               dtype=dtype) * self.sched.init_noise_sigma
        self.warm_noise = [torch.randn(self.cond.shape, generator=gen, device=dev, dtype=dtype)
                           for _ in range(warm)]
        self.abar = self.sched.alphas_cumprod.to(dev)
        self.stride = ip2p.pipe.scheduler.config.num_train_timesteps // steps
        self.z_warm = None

    def _step(self, eps, t, z, warming, i):
        if self.sampler == "ddim":
            return self.sched.step(eps, t, z, eta=1.0 if warming else 0.0,
                                   variance_noise=self.warm_noise[i] if warming else None,
                                   return_dict=False)[0]
        prev = int(t) - self.stride
        ab = self.abar[t]
        ab_prev = self.abar[prev] if prev >= 0 else torch.ones_like(ab)
        alpha = ab / ab_prev
        z = (z - (1 - alpha) / (1 - ab).sqrt() * eps) / alpha.sqrt()
        return z + (1 - alpha).sqrt() * self.warm_noise[i] if warming else z

    def freeze_warm(self, text):
        """用固定文字跑完暖身段並快取結果，之後 run() 從暖身結束處開始。"""
        z = self.z_T
        with torch.no_grad():
            for i, t in enumerate(self.sched.timesteps[:self.warm]):
                eps = self._guided(z, t, text.to(self.null.dtype), False)
                z = self._step(eps, t, z, True, i)
        self.z_warm = z

    def _unet(self, a, t, text):
        return self.ip2p.unet(a, t, encoder_hidden_states=text, return_dict=False)[0]

    def _guided(self, z, t, text, active):
        zin = self.sched.scale_model_input(z, t)
        with_img = torch.cat([zin, self.cond], 1)
        no_img = torch.cat([zin, torch.zeros_like(self.cond)], 1)
        calls = ((with_img, text), (with_img, self.null), (no_img, self.null))
        e_text, e_img, e_un = (
            ckpt.checkpoint(self._unet, a, t, e, use_reentrant=False) if active
            else self._unet(a, t, e) for a, e in calls)
        return e_un + self.s_t * (e_text - e_img) + self.s_i * (e_img - e_un)

    def run(self, text, grad, z_offset=None):
        text = text.to(self.null.dtype)
        start = 0 if self.z_warm is None else self.warm
        z = self.z_T if self.z_warm is None else self.z_warm
        x0s = []
        for i, t in enumerate(self.sched.timesteps):
            if i < start:
                continue
            if i == self.warm and z_offset is not None:
                z = z + z_offset.to(z.dtype)
            active = grad and i >= self.warm
            with torch.set_grad_enabled(active):
                eps = self._guided(z, t, text if active else text.detach(), active)
                if i >= self.warm:
                    ab = self.abar[t]
                    x0s.append(((z - (1 - ab).sqrt() * eps) / ab.sqrt()).float())
                z = self._step(eps, t, z, i < self.warm, i)
        with torch.set_grad_enabled(grad):
            y = self.ip2p.decode_latent(z, use_ckpt=grad)
        return y.float(), x0s


class AnchoredFree(FreeObjective):
    """FreeObjective 的 enc／cond 兩項，原圖側換成 x_ref；VAE 後驗只算一次並 checkpoint。"""

    def terms(self, x_def):
        post = self.ip2p.posterior_mean(x_def, use_ckpt=True)
        z = (post * self.ip2p.scaling_factor).float()
        cond = post.to(self.text.dtype)
        gaps = [self._wnorm(self._eps(cond, t).float() - self.eps0[int(t)].float())
                / self.eps0_norm[int(t)] for t in self.timesteps]
        return {"enc": self._wnorm(z - self.z0) / self.z0_norm,
                "cond": torch.stack(gaps).mean()}


class EncoderTarget:
    """攻擊端影像條件 E(y)（VAE 後驗平均）對灰圖 latent 的均方差，最小化（PhotoGuard 編碼器攻擊）。

    除以 x_ref 上的起點值，使目標從 1 開始、與身分／色偏懲罰同一量級。
    """

    stochastic = False

    def __init__(self, ip2p, x_ref, args):
        self.ip2p, self.a = ip2p, args
        with torch.no_grad():
            self.target = ip2p.posterior_mean(torch.full_like(x_ref, 0.5)).float()
            self.v0 = float((ip2p.posterior_mean(x_ref).float() - self.target).pow(2).mean())

    def value(self, y, backward=False, validation=False):
        with torch.set_grad_enabled(backward):
            v = (self.ip2p.posterior_mean(y, use_ckpt=True).float() - self.target).pow(2).mean() / self.v0
            if backward:
                (self.a.a_adv * v).backward()
        return float(v)


class AttnObjective:
    """攻擊端 UNet 自注意力圖相對 x_ref 的偏離，最大化（回傳值取負號，最小化）。

    文字條件為空字串；UNet 輸入的噪聲通道是 y 自己的 latent 加噪、影像條件是 E(y)，與 x_ref
    共用同一組 (t, ε)。只取 32×32 以下解析度的自注意力層（`attn1`），注意力機率由
    `get_attention_scores` 明確算出。每層取相對平方 Frobenius 距離後平均。
    """

    stochastic = True

    def __init__(self, ip2p, x_ref, args):
        self._setup(ip2p, x_ref, args, "", lambda n: n.endswith("attn1")
                    and not n.startswith(("down_blocks.0", "up_blocks.3")))

    def _setup(self, ip2p, x_ref, args, text, keep):
        from diffusers.models.attention_processor import AttnProcessor

        self.ip2p, self.a, self.dev = ip2p, args, ip2p.device
        self.dtype = ip2p.unet.dtype
        self.text = encode_text(ip2p, text)[1].to(self.dtype)
        self.abar = ip2p.pipe.scheduler.alphas_cumprod.to(self.dev).float()
        self.maps, self.capturing, self.layers = [], False, []
        for name, m in ip2p.unet.named_modules():
            if keep(name):
                m.set_processor(AttnProcessor())
                m.get_attention_scores = self._wrap(m.get_attention_scores)
                self.layers.append(name)
        if not self.layers:
            raise SystemExit("UNet 裡找不到指定的注意力層")
        with torch.no_grad():
            post = ip2p.posterior_mean(x_ref).float()
            self.z_ref, self.c_ref = post * ip2p.scaling_factor, post
        g = torch.Generator(device=self.dev).manual_seed(int(args.val_seed))
        self.val_draws = [self._draw(g) for _ in range(args.attn_val_k)]
        self.gen = torch.Generator(device=self.dev).manual_seed(int(args.noise_seed))

    def _wrap(self, orig):
        def scores(query, key, attention_mask=None):
            p = orig(query, key, attention_mask)
            if self.capturing:
                self.maps.append(p)
            return p
        return scores

    def _draw(self, g):
        t = torch.randint(self.a.attn_tmin, self.a.attn_tmax, (1,), generator=g, device=self.dev)
        eps = torch.randn(self.z_ref.shape, generator=g, device=self.dev)
        return t, eps

    def _maps(self, z, c, t, eps):
        ab = self.abar[t].view(1, 1, 1, 1)
        zt = ab.sqrt() * z + (1 - ab).sqrt() * eps
        self.maps, self.capturing = [], True
        try:
            self.ip2p.unet(torch.cat([zt, c], 1).to(self.dtype), t, encoder_hidden_states=self.text,
                           return_dict=False)
        finally:
            self.capturing = False
        return self.maps

    def value(self, y, backward=False, validation=False):
        draws = self.val_draws if validation else [self._draw(self.gen) for _ in range(self.a.attn_k)]
        total = 0.0
        for t, eps in draws:
            with torch.no_grad():
                ref = [m.detach() for m in self._maps(self.z_ref, self.c_ref, t, eps)]
            with torch.set_grad_enabled(backward):
                post = self.ip2p.posterior_mean(y, use_ckpt=True).float()
                cur = self._maps(post * self.ip2p.scaling_factor, post, t, eps)
                d = torch.stack([(a.float() - b.float()).pow(2).sum() / b.float().pow(2).sum()
                                 for a, b in zip(cur, ref)]).mean()
                if backward:
                    (-self.a.a_adv * d / len(draws)).backward()
            total += float(d)
            self.maps = []
        return -total / len(draws)


class CrossAttnObjective(AttnObjective):
    """攻擊端 UNet 對類別詞（`--class-word`，如 man／woman）的交叉注意力質量，最小化。

    文字條件只有類別詞；每個交叉注意力層（`attn2`，全部解析度）取該詞 token 的注意力機率在
    所有像素與 head 上的平均，除以 x_ref 在同一組 (t, ε) 下的值後跨層平均。起點為 1，
    0 代表模型不再注意這個詞。
    """

    def __init__(self, ip2p, x_ref, args):
        self._setup(ip2p, x_ref, args, args.class_word, lambda n: n.endswith("attn2"))
        word = ip2p.pipe.tokenizer(args.class_word, add_special_tokens=False).input_ids
        self.tok = list(range(1, 1 + len(word)))

    def _mass(self, z, c, t, eps):
        return [p[..., self.tok].float().sum(-1).mean() for p in self._maps(z, c, t, eps)]

    def value(self, y, backward=False, validation=False):
        draws = self.val_draws if validation else [self._draw(self.gen) for _ in range(self.a.attn_k)]
        total = 0.0
        for t, eps in draws:
            with torch.no_grad():
                ref = [m.detach() for m in self._mass(self.z_ref, self.c_ref, t, eps)]
            with torch.set_grad_enabled(backward):
                post = self.ip2p.posterior_mean(y, use_ckpt=True).float()
                cur = self._mass(post * self.ip2p.scaling_factor, post, t, eps)
                d = torch.stack([a / b for a, b in zip(cur, ref)]).mean()
                if backward:
                    (self.a.a_adv * d / len(draws)).backward()
            total += float(d)
            self.maps = []
        return total / len(draws)


class ClassifierObjective:
    """SPA 論文 §3.4 Eq. 2–3 的統一指數邊界損失，直接施加在防禦圖 y（風格編輯的輸出）上。

    分類器為凍結的 torchvision ImageNet 模型（§4.1；預設 ResNet-50）；y 以可微的雙線性（antialias）
    縮成 224×224，再做 ImageNet mean/std 正規化。人像沒有 ImageNet 標註，真值標籤取分類器在原圖 x
    上的 top-1（非定向攻擊）。m = (l_y − max_{k≠y} l_k)/τ，L = exp(κ·tanh(m/κ))，τ=1、κ=9。
    """

    stochastic = False

    def __init__(self, x_orig, args, dev):
        import torchvision.models as tvm

        self.a = args
        weights = tvm.get_model_weights(args.cls_arch).DEFAULT if args.cls_weights == "default"             else tvm.get_model_weights(args.cls_arch)[args.cls_weights]
        self.weights_name = str(weights)
        self.net = tvm.get_model(args.cls_arch, weights=weights).to(dev).eval()
        self.net.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        with torch.no_grad():
            lx = self.logits(x_orig)[0]
        self.label = int(lx.argmax())
        self.onehot = torch.zeros_like(lx, dtype=torch.bool)
        self.onehot[self.label] = True
        self.margin_x = float(self._margin(lx))
        self.info = {}

    def logits(self, y):
        z = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        return self.net((z - self.mean) / self.std).float()

    def _margin(self, l):
        return l[self.label] - l.masked_fill(self.onehot, float("-inf")).max()

    def value(self, y, backward=False, validation=False):
        with torch.set_grad_enabled(backward):
            l = self.logits(y)[0]
            margin = self._margin(l)
            m = margin / self.a.cls_tau
            loss = torch.exp(self.a.cls_kappa * torch.tanh(m / self.a.cls_kappa))
            if backward:
                (self.a.a_adv * loss).backward()
        self.info = {"cls_margin": round(float(margin.detach()), 4), "cls_top1": int(l.argmax()),
                     "cls_p_true": round(float(l.detach().softmax(-1)[self.label]), 5)}
        return float(loss)


def grey(x01):
    return (0.299 * x01[:, :1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]).expand(-1, 3, -1, -1)


class ChaosObjective:
    """攻擊端以類別詞為指令、`--chaos-steps` 步 DDIM（s_t、s_i 同攻擊端）編輯 y，最大化輸出與 y 的
    灰階 LPIPS（回傳值取負號，最小化）。y 一側不回傳梯度，梯度只經由攻擊端的影像條件 E(y)。

    每步抽一組起始噪聲；驗證用固定的 `--chaos-val-k` 組。
    """

    stochastic = True

    def __init__(self, ip2p, x_ref, args, lp):
        from diffusers import DDIMScheduler

        self.ip2p, self.a, self.lp, self.dev = ip2p, args, lp, ip2p.device
        self.dtype = ip2p.unet.dtype
        self.sched = DDIMScheduler.from_config(
            ip2p.pipe.scheduler.config, clip_sample=False, thresholding=False,
            set_alpha_to_one=False, timestep_spacing="leading")
        self.sched.set_timesteps(args.chaos_steps, device=self.dev)
        self.text = encode_text(ip2p, args.class_word)[1].to(self.dtype)
        self.null = encode_text(ip2p, "")[1].to(self.dtype)
        with torch.no_grad():
            self.shape = ip2p.image_latents(x_ref).shape
        g = torch.Generator(device=self.dev).manual_seed(int(args.val_seed))
        self.val_draws = [self._draw(g) for _ in range(args.chaos_val_k)]
        self.gen = torch.Generator(device=self.dev).manual_seed(int(args.noise_seed))

    def _draw(self, g):
        return torch.randn(self.shape, generator=g, device=self.dev, dtype=self.dtype) \
            * self.sched.init_noise_sigma

    def _unet(self, a, t, text):
        return self.ip2p.unet(a, t, encoder_hidden_states=text, return_dict=False)[0]

    def edit(self, y, z, grad):
        cond = self.ip2p.posterior_mean(y, use_ckpt=grad).to(self.dtype)
        for t in self.sched.timesteps:
            zin = self.sched.scale_model_input(z, t)
            with_img = torch.cat([zin, cond], 1)
            no_img = torch.cat([zin, torch.zeros_like(cond)], 1)
            calls = ((with_img, self.text), (with_img, self.null), (no_img, self.null))
            e_text, e_img, e_un = (
                ckpt.checkpoint(self._unet, a, t, e, use_reentrant=False) if grad
                else self._unet(a, t, e) for a, e in calls)
            eps = e_un + self.a.s_t * (e_text - e_img) + self.a.attack_s_i * (e_img - e_un)
            z = self.sched.step(eps, t, z, eta=0.0, return_dict=False)[0]
        return self.ip2p.decode_latent(z, use_ckpt=grad).float().clamp(0, 1)

    def value(self, y, backward=False, validation=False):
        draws = self.val_draws if validation else [self._draw(self.gen)]
        total = 0.0
        for z in draws:
            with torch.set_grad_enabled(backward):
                d = self.lp(grey(self.edit(y, z, backward)), grey(y.detach())).mean()
                if backward:
                    (-self.a.a_adv * d / len(draws)).backward()
            total += float(d)
        return -total / len(draws)


def warm_shift(x01, y01, channel, q, mask=None):
    """相對 x01 往 a*＋／b*＋ 的逐像素位移分位數；給 mask 時只在遮罩內取。"""
    if mask is None:
        return channel_shift_p95(x01, y01, channel, 1, q)
    d = rgb_to_lab(y01.clamp(0, 1).float()) - rgb_to_lab(x01.clamp(0, 1).float())
    v = torch.relu(d[:, channel])[mask[:, 0] > 0.5]
    return torch.quantile(v, q)


def psnr(a, b):
    return float(10 * torch.log10(1.0 / (a - b).pow(2).mean()))


def optimise(args, ip2p, ed, x_ref, e_txt, ids, x0_ref, id_cos, id_cos_diff, tau, lp, trace, tag,
             bg=None, face=None, x_orig=None):
    dev = ip2p.device
    eos = int((ids == ip2p.pipe.tokenizer.eos_token_id).nonzero()[0])
    extra = {"objective": args.objective}
    free = target = None
    if args.objective == "free":
        free = AnchoredFree(ip2p, x_ref, box=None, k=4, steps=50, seed=args.noise_seed,
                            weights={"id": 0.0, "enc": args.w_enc, "cond": 1.0},
                            chain_steps=6, grad_steps=1, s_i=1.5, resample=False)
    elif args.objective == "enc_grey":
        target = EncoderTarget(ip2p, x_ref, args)
    elif args.objective == "classifier":
        target = ClassifierObjective(x_orig, args, dev)
        extra.update(cls_arch=args.cls_arch, cls_weights=target.weights_name, cls_tau=args.cls_tau,
                     cls_kappa=args.cls_kappa, cls_label=target.label,
                     cls_margin_x=round(target.margin_x, 4))
    elif args.objective == "chaos":
        target = ChaosObjective(ip2p, x_ref, args, lp)
        extra.update(class_word=args.class_word, chaos_steps=args.chaos_steps,
                     chaos_val_k=args.chaos_val_k, attack_s_i=args.attack_s_i)
    else:
        target = (CrossAttnObjective if args.objective == "xattn" else AttnObjective)(ip2p, x_ref, args)
        extra.update(attn_k=args.attn_k, attn_val_k=args.attn_val_k, attn_tmin=args.attn_tmin,
                     attn_tmax=args.attn_tmax, attn_layers=len(target.layers))
        if args.objective == "xattn":
            extra["class_word"] = args.class_word
    extra.update(freeze_warm=int(args.freeze_warm), struct_cap=args.struct_cap)
    gen = torch.Generator(device=dev).manual_seed(int(args.seed) + 1)
    use_prompt = args.carrier in ("prompt", "prompt_latent")
    # instr：指令 token 與第一個 EOS（位置 1..eos）；full：整段 77 個 token（含 BOS 與 padding）
    lo, hi = (1, eos + 1) if args.prompt_scope == "instr" else (0, e_txt.shape[1])
    delta = (torch.randn((hi - lo, e_txt.shape[-1]), generator=gen, device=dev)
             * args.init_rms * float(use_prompt)).requires_grad_(use_prompt)
    extra["prompt_scope"] = args.prompt_scope
    z_off = (torch.zeros_like(ed.z_T, dtype=torch.float32).requires_grad_()
             if args.carrier in ("latent", "prompt_latent") else None)

    def project():
        with torch.no_grad():
            if use_prompt:
                rms = delta.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
                delta.mul_((args.delta_rms_cap / rms).clamp(max=1.0))
            if z_off is not None:
                zr = z_off.pow(2).mean().sqrt().clamp_min(1e-12)
                z_off.mul_((args.latent_rms_cap / zr).clamp(max=1.0))

    project()
    extra["carrier"] = args.carrier
    groups = []
    if use_prompt:
        groups.append({"params": [delta], "lr": args.lr})
    if z_off is not None:
        groups.append({"params": [z_off], "lr": args.latent_lr})
        extra.update(latent_lr=args.latent_lr, latent_rms_cap=args.latent_rms_cap)
    params = [g["params"][0] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    key = "val_l_adv" if target is not None and target.stochastic else "l_adv"
    best, last = None, None
    stale, decayed, best_val, init_val = 0, 0, None, None
    zero = torch.zeros((), device=dev)
    for u in range(args.updates + 1):
        started = time.time()
        e_adv = torch.cat([e_txt[:, :lo], e_txt[:, lo:hi] + delta[None], e_txt[:, hi:]], 1)
        y, x0s = ed.run(e_adv, grad=True, z_offset=z_off)
        l_perc = torch.stack([(a - b).pow(2).mean() / (b.pow(2).mean() + 1e-6)
                              for a, b in zip(x0s, x0_ref)]).mean()
        if args.prompt_scope == "instr":
            tok_cos = F.cosine_similarity(e_adv[0, lo:hi], e_txt[0, lo:hi], dim=-1).mean()
        else:  # 論文 Eq. 6：整段 embedding 攤平後的餘弦
            tok_cos = F.cosine_similarity(e_adv.flatten()[None], e_txt.flatten()[None]).squeeze()
        cos_d = id_cos_diff(y)
        l_id = torch.relu(tau - cos_d)
        reg = args.a_perc * l_perc + args.a_prompt * (1 - tok_cos) + args.a_id * l_id
        if bg is not None and args.bg_cap > 0:
            reg = reg + args.a_bg * torch.relu(lp(y * bg + x_ref * (1 - bg), x_ref).mean() - args.bg_cap)
        if args.struct_cap > 0:
            reg = reg + args.a_struct * torch.relu(lp(grey(y), grey(x_ref)).mean() - args.struct_cap)
        col_terms = []
        if args.col_cap > 0:
            # 相對 x_ref 往 a*＋（紅、洋紅）與 b*＋（黃）的位移：p95 ≤ col_cap、p99 ≤ 2·col_cap
            for c in (1, 2):
                for q, cap in ((0.95, args.col_cap), (0.99, 2 * args.col_cap)):
                    col_terms.append((c, q, cap))
                    reg = reg + args.a_col * torch.relu(warm_shift(x_ref, y, c, q, face) - cap)
        need = u < args.updates
        with torch.no_grad():
            yq = quantise(y.detach())
            id_q = id_cos(yq)
            col_q = {f"{'ab'[c - 1]}_pos_p{int(q * 100)}": float(warm_shift(x_ref, yq, c, q, face))
                     for c, q, _ in col_terms}
            col_ok = all(col_q[f"{'ab'[c - 1]}_pos_p{int(q * 100)}"] <= cap for c, q, cap in col_terms)
            bg_q = float(lp(yq * bg + x_ref * (1 - bg), x_ref).mean()) if bg is not None else 0.0
            struct_q = float(lp(grey(yq), grey(x_ref)).mean())
        if args.snapshot_every > 0 and (u % args.snapshot_every == 0 or u == args.updates):
            (args.out / "snapshots").mkdir(exist_ok=True)
            save_image(yq, args.out / "snapshots" / f"{tag}__u{u:03d}__def.png")
        if free is not None:
            terms = free.terms(y)
            l_adv = -args.w_enc * torch.tanh(terms["enc"]) - torch.tanh(terms["cond"])
            loss = reg + args.a_adv * l_adv
            with torch.no_grad():
                tq = free.terms(yq)
            adv = {"loss": float(loss), "l_adv_train": float(l_adv),
                   "l_adv": float(-args.w_enc * torch.tanh(tq["enc"]) - torch.tanh(tq["cond"])),
                   "enc": float(tq["enc"]), "cond": float(tq["cond"])}
        else:
            leaf = yq.detach().requires_grad_(need)  # 量化對 y 直通
            adv = {"l_adv": target.value(leaf, backward=need)}
            adv["loss"] = float(reg) + args.a_adv * adv["l_adv"]
            if target.stochastic and (u % args.val_every == 0 or u == args.updates):
                adv["val_l_adv"] = target.value(leaf.detach(), validation=True)
        with torch.no_grad():
            rec = {"tag": tag, "update": u, **adv, "l_perc": float(l_perc),
                   "tok_cos": float(tok_cos), "id_cos": round(id_q, 5), "tau": round(tau, 5),
                   "bg_lpips_ref": round(bg_q, 5), "struct_lpips_ref": round(struct_q, 5),
                   **{k: round(v, 3) for k, v in col_q.items()},
                   "feasible": int(id_q >= tau and (args.bg_cap <= 0 or bg_q <= args.bg_cap) and col_ok
                                   and (args.struct_cap <= 0 or struct_q <= args.struct_cap)),
                   "lpips_def_ref": float(lp(yq, x_ref).mean()),
                   **getattr(target, "info", {}),
                   "delta_rms": float(delta.pow(2).mean(-1).sqrt().max()),
                   **({"z_off_rms": float(z_off.pow(2).mean().sqrt())} if z_off is not None else {}),
                   }
        last = (u, yq, rec)
        if rec["feasible"] and key in rec and (best is None or rec[key] < best[2][key]):
            best = (u, yq, rec)
        if key in rec:
            init_val = rec[key] if init_val is None else init_val
            if rec["feasible"] and (best_val is None or rec[key] < best_val - 0.01 * abs(init_val)):
                best_val, stale = rec[key], 0
            else:
                stale += 1
        if stale >= args.patience:
            if decayed >= args.max_decays:
                rec["stop"], need = "plateau", False
            else:
                for g in opt.param_groups:
                    g["lr"] /= 4
                decayed, stale = decayed + 1, 0
                rec["lr_decay"] = decayed
        if need:
            opt.zero_grad(set_to_none=True)
            if free is not None:
                loss.backward()
            else:
                torch.autograd.backward([y, reg], [leaf.grad, None])
            grads = [p.grad for p in params]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise SystemExit(f"{tag}：第 {u} 步梯度缺失或含 NaN／inf")
            gnorm = float(torch.stack([g.norm() for g in grads]).norm())
            if u == 0 and gnorm == 0:
                raise SystemExit(f"{tag}：第一步梯度為零")
            opt.step()
            project()
            rec["grad_norm"] = gnorm
        rec["seconds"] = round(time.time() - started, 1)
        trace.append(rec)
        write_rows(args.out / "trace.csv", trace)
        print(f"  [{tag}] u{u} loss {rec['loss']:.4f} adv {rec['l_adv']:.4f}"
              + (f" val {rec['val_l_adv']:.4f}" if "val_l_adv" in rec else "")
              + f" id {rec['id_cos']:.3f}/{tau:.3f} lpips_ref {rec['lpips_def_ref']:.4f}"
              + f" struct {rec['struct_lpips_ref']:.4f} feas {rec['feasible']} {rec['seconds']}s"
              + (f" margin {rec['cls_margin']:.3f} top1 {rec['cls_top1']}" if "cls_margin" in rec else ""),
              flush=True)
        if rec.get("stop"):
            break
    return best, last, extra


def select_result(best, last, policy):
    """選點政策與限制判定分別記錄；last 不代表符合限制。"""
    selected = last if policy == "last" or best is None else best
    if selected is None:
        raise ValueError("最佳化沒有可選取的更新")
    feasible = int(selected[2]["feasible"])
    return selected, {"selection_policy": policy, "selected_feasible": feasible,
                      "feasible": feasible}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--styles", nargs="+", default=list(STYLES), choices=list(STYLES))
    ap.add_argument("--s-i", type=float, nargs="+", default=[2.0])
    ap.add_argument("--s-t", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=0)
    ap.add_argument("--objective", default="free", choices=("free", "enc_grey", "attn", "xattn", "chaos",
                                                                    "classifier"))
    ap.add_argument("--cls-arch", default="resnet50", help="classifier：torchvision 模型名（論文 §4.1）")
    ap.add_argument("--cls-weights", default="IMAGENET1K_V1", help="classifier：權重名或 default")
    ap.add_argument("--cls-tau", type=float, default=1.0, help="classifier：邊界縮放 τ（論文 1.0）")
    ap.add_argument("--cls-kappa", type=float, default=9.0, help="classifier：飽和常數 κ（論文 9.0）")
    ap.add_argument("--class-word", default="", help="xattn／chaos：攻擊端的文字條件（類別詞，如 man、woman）")
    ap.add_argument("--chaos-steps", type=int, default=10, help="chaos：攻擊端短程編輯的 DDIM 步數")
    ap.add_argument("--chaos-val-k", type=int, default=2, help="chaos：固定驗證用幾組起始噪聲")
    ap.add_argument("--attack-s-i", type=float, default=1.8, help="chaos：攻擊端的影像引導強度")
    ap.add_argument("--freeze-warm", action="store_true", help="暖身段固定用原風格指令跑一次並快取")
    ap.add_argument("--sampler", default="ddim", choices=("ddim", "ddpm"), help="ddpm：SPA 附錄 B 的 (B.5)／(B.6)")
    ap.add_argument("--prompt-scope", default="instr", choices=("instr", "full"))
    ap.add_argument("--select", default="best", choices=("best", "last"),
                    help="best：可行步中目標最佳者；last：最後一步（論文）")
    ap.add_argument("--struct-cap", type=float, default=0.0, help="> 0 時防禦圖對 x_ref 的灰階 LPIPS 上限")
    ap.add_argument("--a-struct", type=float, default=20.0)
    ap.add_argument("--carrier", default="prompt", choices=("prompt", "latent", "prompt_latent"))
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--latent-lr", type=float, default=0.02)
    ap.add_argument("--latent-rms-cap", type=float, default=0.3)
    ap.add_argument("--init-rms", type=float, default=0.01)
    ap.add_argument("--delta-rms-cap", type=float, default=0.10)
    ap.add_argument("--a-perc", type=float, default=25.0)
    ap.add_argument("--a-adv", type=float, default=1.0)
    ap.add_argument("--a-prompt", type=float, default=1.0)
    ap.add_argument("--a-id", type=float, default=20.0)
    ap.add_argument("--w-enc", type=float, default=0.1)
    ap.add_argument("--id-floor", type=float, default=0.80)
    ap.add_argument("--id-margin", type=float, default=0.03)
    ap.add_argument("--bg-cap", type=float, default=0.0, help="> 0 時背景對 x_ref 的 LPIPS 上限")
    ap.add_argument("--a-bg", type=float, default=20.0)
    ap.add_argument("--col-cap", type=float, default=0.0, help="> 0 時限制相對 x_ref 往 a*＋／b*＋ 的位移")
    ap.add_argument("--col-face-only", action="store_true", help="色偏上限只在臉框內量")
    ap.add_argument("--a-col", type=float, default=0.05)
    ap.add_argument("--attn-k", type=int, default=2, help="attn：每步抽幾組 (t, ε)")
    ap.add_argument("--attn-val-k", type=int, default=4, help="attn：固定驗證用幾組 (t, ε)")
    ap.add_argument("--attn-tmin", type=int, default=100)
    ap.add_argument("--attn-tmax", type=int, default=900)
    ap.add_argument("--val-seed", type=int, default=777)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=15, help="停滯判定：連續幾次評估沒有改善 1%")
    ap.add_argument("--max-decays", type=int, default=2, help="停滯時最多降 lr 幾次（每次 /4）後停止")
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--tf32", action="store_true")
    args = ap.parse_args()
    if args.objective in ("xattn", "chaos") and not args.class_word:
        raise SystemExit(f"--objective {args.objective} 需要 --class-word")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ip2p = IP2PWrapper(dtype=torch.float32)
    dev = ip2p.device
    import piq
    lp = piq.LPIPS().to(dev).eval()
    lp.requires_grad_(False)
    items = load_images(args.data, set(args.images) if args.images else None)

    rows, trace = [], []
    for item in items:
        name = item["name"]
        x = load_image_tensor(item["path"], dev, size=RESOLUTION)
        boxes = face_boxes(x, dev)
        if not boxes:
            raise SystemExit(f"{name} 偵測不到臉，身分下限無從定義")
        box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        e0 = embed_box(x, box, dev).float()
        mask_png = args.data / "masks" / f"{name}.png"
        if not mask_png.is_file():
            raise SystemExit(f"找不到背景遮罩 {mask_png}")
        bg = (load_image_tensor(mask_png, dev, size=RESOLUTION)[:, :1] >= 0.5).float()
        face = torch.zeros_like(bg)
        fx0, fy0, fx1, fy1 = (int(round(v)) for v in box)
        face[..., max(0, fy0):fy1, max(0, fx0):fx1] = 1.0

        def id_cos(y):
            return float(F.cosine_similarity(embed_box(y, box, dev).float()[None], e0[None]))

        def id_cos_diff(y):
            return F.cosine_similarity(embed_box_differentiable(y, box, dev).float()[None],
                                       e0[None]).squeeze()

        save_image(x, args.out / f"{name}__orig.png")
        with torch.no_grad():
            v = quantise(ip2p.decode_latent(ip2p.encode_image(x)).float())
        save_image(v, args.out / f"{name}__vae.png")
        rows.append({"image": name, "style": "vae_roundtrip", "prompt": "", "s_i": "",
                     "id_ref": round(id_cos(v), 5),
                     "lpips_ref_x": round(float(lp(v, x).mean()), 5),
                     "psnr_ref_x": round(psnr(v, x), 3)})

        for style in args.styles:
            ids, e_txt = encode_text(ip2p, STYLES[style])
            for s_i in args.s_i:
                started = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                tag = f"{name}__{style}_si{s_i:g}".replace(".", "p")
                ed = StyleEditor(ip2p, x, steps=args.steps, warm=args.warm,
                                 s_t=args.s_t, s_i=s_i, seed=args.seed, sampler=args.sampler)
                if args.freeze_warm:
                    ed.freeze_warm(e_txt)
                with torch.no_grad():
                    y_ref, x0_ref = ed.run(e_txt, grad=False)
                    x_ref = quantise(y_ref)
                save_image(x_ref, args.out / f"{tag}__ref.png")
                id_ref = id_cos(x_ref)
                row = {"image": name, "style": style, "prompt": STYLES[style], "s_i": s_i,
                       "s_t": args.s_t, "steps": args.steps, "warm": args.warm, "seed": args.seed,
                       "id_ref": round(id_ref, 5),
                       "lpips_ref_x": round(float(lp(x_ref, x).mean()), 5),
                       "psnr_ref_x": round(psnr(x_ref, x), 3)}
                if args.updates > 0:
                    tau = max(args.id_floor, id_ref - args.id_margin)
                    best, last, extra = optimise(
                        args, ip2p, ed, x_ref, e_txt, ids, x0_ref, id_cos, id_cos_diff, tau, lp,
                        trace, tag, bg=bg, face=face if args.col_face_only else None, x_orig=x)
                    (u, yd, rec), selection = select_result(best, last, args.select)
                    # last 政策仍輸出所選的最後一步，限制判定由 selection 記錄。
                    suffix = "def" if args.select == "last" or best is not None else "def_infeasible"
                    save_image(yd, args.out / f"{tag}__{suffix}.png")
                    row.update(extra)
                    row.update({
                        "updates": args.updates, "lr": args.lr, "delta_rms_cap": args.delta_rms_cap,
                        "a_perc": args.a_perc, "a_adv": args.a_adv, "a_prompt": args.a_prompt,
                        "a_id": args.a_id, "w_enc": args.w_enc, "tau": round(tau, 5),
                        **selection, "chosen_update": u,
                        "last_update": last[0], "stop": last[2].get("stop", ""),
                        "id_def": rec["id_cos"],
                        "lpips_def_x": round(float(lp(yd, x).mean()), 5),
                        "lpips_def_ref": round(rec["lpips_def_ref"], 5),
                        "psnr_def_x": round(psnr(yd, x), 3),
                        "psnr_def_ref": round(psnr(yd, x_ref), 3),
                        "l_adv": round(rec["l_adv"], 5), "l_perc": round(rec["l_perc"], 6),
                        **{k: round(rec[k], 5) for k in ("val_l_adv", "enc", "cond") if k in rec},
                        **{k: rec[k] for k in ("cls_margin", "cls_top1", "cls_p_true") if k in rec},
                        "tok_cos": round(rec["tok_cos"], 5), "delta_rms": round(rec["delta_rms"], 5)})
                row["seconds"] = round(time.time() - started, 1)
                if torch.cuda.is_available():
                    row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                rows.append(row)
                write_rows(args.out / "results.csv", rows)
                print(f"[{tag}] id_ref {row['id_ref']} lpips_ref_x {row['lpips_ref_x']}"
                      + (f"  id_def {row['id_def']} lpips_def_ref {row['lpips_def_ref']} "
                         f"feasible {row['feasible']}" if args.updates > 0 else "")
                      + f"  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 列 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
