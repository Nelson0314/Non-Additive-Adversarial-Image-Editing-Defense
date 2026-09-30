"""攻擊端 IP2P UNet 的注意力圖目標：自注意力偏離與類別詞交叉注意力質量。

UNet 輸入的噪聲通道為 y 的 latent 加噪、影像條件為 E(y)，與參照影像 x_ref 共用同一組
(t, ε)；注意力機率由 `get_attention_scores` 明確取出。設定由 `args` 提供：
`attn_tmin`、`attn_tmax`、`attn_k`、`attn_val_k`、`val_seed`、`noise_seed`、`a_adv`，
`CrossAttnObjective` 另需 `class_word`。
"""
from __future__ import annotations

import torch


def encode_text(ip2p, prompt):
    tok = ip2p.pipe.tokenizer(prompt, padding="max_length",
                              max_length=ip2p.pipe.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
    ids = tok.input_ids.to(ip2p.device)
    with torch.no_grad():
        emb = ip2p.text_encoder(ids)[0]
    return ids[0], emb


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
