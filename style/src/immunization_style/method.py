"""style_prompt 方法：以 ip2p 風格編輯為載體的防禦（SPA 的移植），目標與選點。

命令列入口為 `immunization_style.cli.generate_style_prompt_defenses`，方法說明見其模組 docstring。
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
import yaml
import torch.utils.checkpoint as ckpt

from immunization_core.color.shift import channel_shift_p95
from immunization_core.color.space import rgb_to_lab
from immunization_core.artifacts.images import save_image
from immunization_core.io import write_sorted_csv
from immunization_core.optimization.attention import (
    AttnObjective, CrossAttnObjective, encode_text,
)
from immunization_core.optimization.carrier import quantize
from immunization_core.optimization.instruction_free import FreeObjective
from immunization_style import layout

STYLES_FILE = layout.PROJECT / "configs" / "styles.yaml"

RESOLUTION = 512
#: 風格名 → ip2p 指令；正本為 configs/styles.yaml。
STYLES = {name: str(prompt) for name, prompt in
          yaml.safe_load(STYLES_FILE.read_text(encoding="utf-8"))["styles"].items()}


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


def optimize(args, ip2p, ed, x_ref, e_txt, ids, x0_ref, id_cos, id_cos_diff, tau, lp, trace, tag,
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
            yq = quantize(y.detach())
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
        write_sorted_csv(args.out / "trace.csv", trace)
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


