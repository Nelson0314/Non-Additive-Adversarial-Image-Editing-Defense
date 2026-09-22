"""`src/baselines/danp.py` 的驗收 —— 用替身 SD，不載入真模型。

DANP 沒有官方程式可比對，整個實作的正確性只能靠「論文寫的性質有沒有被
釘住」。這裡釘四組：

1. **動態門檻**（Eq. 7–10）：常數輸入、單格輸入這兩個退化情形的行為，
   以及雙峰輸入下 Kapur 有沒有把高的那一群切出來。
2. **兩個方向的符號**（Eq. 11）：最小化 `L_DAA` 必須**壓低** `M=1` 處、
   **抬高** `M=0` 處。符號寫反不會有症狀——輸出仍是一張合理的防禦圖，
   只是方法變成了 SA 的反面。
3. **預算換算**：`γ=0.03` 在 `[0,1]` 值域，`eps_pixel01` 必須等於它。
4. **`BaselineSpec` 一致性**：Algorithm 1 的每一行對應到哪一個欄位。

另有兩組把「注意力擷取拿到的是不是 softmax 機率」與「loss_fn 端到端可微」
釘住，因為那兩件事錯了同樣沒有症狀。
"""

import math

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from src.baselines import danp
from src.baselines.pgd import run_pgd, step_size_at


# ===========================================================================
# 一、動態門檻（Eq. 7–10）
# ===========================================================================


def test_正規化把一般輸入映到零與一之間():
    a = torch.tensor([[-3.0, 0.0, 1.0, 5.0]])
    n = danp.normalise_attention(a)
    assert float(n.min()) == 0.0
    assert float(n.max()) == 1.0
    # min-max 是仿射映射，順序不變
    assert torch.equal(n.argsort(), a.argsort())


def test_常數輸入的正規化回傳全零而不是除以零():
    """`max == min` 時 `(a−a)/(a−a)` 沒有值。此處寫死成全 0，
    否則會得到 nan 而後續的直方圖、argmax 全部靜默失效。"""
    for value in (0.0, 0.25, 7.0, -2.0):
        a = torch.full((3, 4), value)
        n = danp.normalise_attention(a)
        assert torch.equal(n, torch.zeros_like(a))
        assert torch.isfinite(n).all()


def test_常數輸入的門檻讓遮罩全為零():
    """退化輸入下沒有任何合法的 τ（class 1 永遠是空的），
    規則是回傳 `bins−1`，即「沒有區域與文字相關」。"""
    bins = 128
    att = torch.full((1, 16, 5), 0.3)
    mask, tau_idx, tau_val = danp.kapur_mask(att, bins)
    assert tau_idx == bins - 1
    assert tau_val == pytest.approx(1.0)
    assert float(mask.sum()) == 0.0


def test_質量全落在最高一格時同樣不切():
    """另一個退化方向：所有值都在直方圖的最後一格。class 0 為空，
    同樣沒有合法的 τ。"""
    v = torch.ones(64)
    assert danp.kapur_threshold_index(v, 32) == 31


def test_級距數小於二直接拋出():
    with pytest.raises(ValueError):
        danp.kapur_threshold_index(torch.rand(10), 1)


def test_空張量直接拋出而不是回傳一個數():
    with pytest.raises(ValueError):
        danp.kapur_threshold_index(torch.zeros(0), 16)


def test_雙峰輸入下Kapur把高的那一群切出來():
    """一半的值在 0.05 附近、一半在 0.95 附近。門檻必須落在兩群之間，
    遮罩剛好取到高的那一半。"""
    g = torch.Generator().manual_seed(0)
    low = 0.05 + 0.02 * torch.rand(500, generator=g)
    high = 0.95 + 0.02 * torch.rand(500, generator=g)
    att = torch.cat([low, high]).reshape(1, 100, 10)
    mask, tau_idx, tau_val = danp.kapur_mask(att, danp.PAPER_KAPUR_BINS)
    # `tau_val` 是**正規化之後**的門檻：兩群在 min-max 之後分別落在 0 與 1
    # 附近，門檻必須落在兩群中間的空白帶裡。
    norm = danp.normalise_attention(att).reshape(-1)
    low_top = float(norm[:500].max())
    high_bottom = float(norm[500:].min())
    assert low_top < tau_val < high_bottom
    assert float(mask.sum()) == pytest.approx(500.0)
    # 被選中的正是高的那一群
    assert bool((att.reshape(-1)[mask.reshape(-1) > 0] > 0.5).all())


def test_遮罩不帶梯度():
    att = torch.rand(1, 16, 4, requires_grad=True)
    mask, _, _ = danp.kapur_mask(att, 32)
    assert not mask.requires_grad


def test_遮罩與注意力圖同形():
    att = torch.rand(2, 36, 7)
    mask, _, _ = danp.kapur_mask(att, 64)
    assert mask.shape == att.shape


def test_門檻值等於級距的上邊界():
    v = torch.linspace(0.0, 1.0, 1000)
    bins = 50
    idx = danp.kapur_threshold_index(v, bins)
    mask, tau_idx, tau_val = danp.kapur_mask(v.reshape(1, -1, 1), bins)
    assert tau_idx == idx
    assert tau_val == pytest.approx((idx + 1) / bins)


# ===========================================================================
# 二、兩個方向的符號（Eq. 11）
# ===========================================================================


def test_DAA對相關區的梯度為正對不相關區為負():
    """Eq. 11 的解析梯度是 `2·Att⊙M − 2λ·Att⊙(1−M)`。
    符號決定了「壓低相關區、抬高不相關區」這句話成不成立。"""
    att = torch.rand(1, 9, 3, dtype=torch.float64) + 0.5
    att.requires_grad_(True)
    mask = torch.zeros_like(att)
    mask[:, :4] = 1.0
    lam = 1.0
    loss = danp.daa_loss(att, mask, lam)
    g = torch.autograd.grad(loss, att)[0]
    expect = 2.0 * att.detach() * mask - 2.0 * lam * att.detach() * (1 - mask)
    assert torch.allclose(g, expect)
    assert bool((g[:, :4] > 0).all())      # 相關區：梯度為正
    assert bool((g[:, 4:] < 0).all())      # 不相關區：梯度為負


def test_沿負梯度走一步會壓低相關區並抬高不相關區():
    """把符號寫成「它對最佳化做了什麼」，而不是只比對解析式。
    `run_pgd` 的 `objective="minimize"` 走的就是 `−grad`。"""
    att = (torch.rand(1, 16, 4, dtype=torch.float64) + 0.5).requires_grad_(True)
    mask = torch.zeros_like(att)
    mask[:, :8] = 1.0
    loss = danp.daa_loss(att, mask, danp.PAPER_LAMBDA_DAA)
    g = torch.autograd.grad(loss, att)[0]
    stepped = att.detach() - 0.01 * g

    rel_before = (att.detach() * mask).pow(2).sum()
    rel_after = (stepped * mask).pow(2).sum()
    irr_before = (att.detach() * (1 - mask)).pow(2).sum()
    irr_after = (stepped * (1 - mask)).pow(2).sum()

    assert float(rel_after) < float(rel_before)     # 相關區被壓低
    assert float(irr_after) > float(irr_before)     # 不相關區被抬高


def test_lambda為零時只剩壓低那一項():
    att = (torch.rand(1, 9, 3, dtype=torch.float64) + 0.5).requires_grad_(True)
    mask = torch.zeros_like(att)
    mask[:, :4] = 1.0
    g = torch.autograd.grad(danp.daa_loss(att, mask, 0.0), att)[0]
    assert bool((g[:, 4:] == 0).all())


def test_遮罩全為一時DAA退化成SA式的單向壓低():
    att = torch.rand(1, 9, 3, dtype=torch.float64).requires_grad_(True)
    mask = torch.ones_like(att)
    loss = danp.daa_loss(att, mask, 1.0)
    assert float(loss.detach()) == pytest.approx(float(att.detach().pow(2).sum()))


def test_遮罩全為零時DAA退化成整張抬高():
    """常數注意力圖會走到這一格（見上面的退化測試）。"""
    att = torch.rand(1, 9, 3, dtype=torch.float64).requires_grad_(True)
    loss = danp.daa_loss(att, torch.zeros_like(att), 1.0)
    assert float(loss.detach()) == pytest.approx(-float(att.detach().pow(2).sum()))


def test_NBA是負的平方距離所以最小化會拉大距離():
    a = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    b = (a + 0.1).requires_grad_(True)
    loss = danp.nba_loss(a, b)
    assert float(loss.detach()) < 0.0
    assert float(loss.detach()) == pytest.approx(-float((a - b.detach()).pow(2).sum()))
    g = torch.autograd.grad(loss, b)[0]
    stepped = b.detach() - 0.01 * g
    assert float((a - stepped).pow(2).sum()) > float((a - b.detach()).pow(2).sum())


def test_NBA在兩者相同時為零且梯度為零():
    a = torch.randn(1, 4, 4, 4, dtype=torch.float64)
    b = a.clone().requires_grad_(True)
    loss = danp.nba_loss(a, b)
    assert float(loss.detach()) == 0.0
    assert float(torch.autograd.grad(loss, b)[0].abs().max()) == 0.0


# ===========================================================================
# 三、預算換算
# ===========================================================================


def test_值域是零到一因此預算不需換算():
    spec = danp.SPEC_PAPER
    assert (spec.value_range.lo, spec.value_range.hi) == (0.0, 1.0)
    assert spec.value_range.scale == 1.0
    assert spec.eps == pytest.approx(danp.PAPER_GAMMA)
    assert spec.eps_pixel01 == pytest.approx(danp.PAPER_GAMMA)
    assert spec.step_size_pixel01 == pytest.approx(danp.RECONSTRUCTED_STEP_SIZE)


def test_值域的反推與論文的PSNR一致():
    """`DANP_RANGE` 的理由是 Table IV 的 PSNR 34.76 dB 反推的 rms 大於
    `[-1,1]` 讀法的上界。這個算術如果錯了，值域的選擇就沒有依據。"""
    rms = 10 ** (-34.76 / 20)
    assert rms == pytest.approx(0.018281, abs=1e-6)
    assert rms < danp.PAPER_GAMMA                 # [0,1] 讀法：相容
    assert rms > danp.PAPER_GAMMA / 2.0           # [-1,1] 讀法：矛盾


def test_一百步的sign更新走得滿預算():
    """α=1/255、N=100 下 `100·α = 0.392 ≫ γ = 0.03`，預算可達。
    若步長小到走不滿，`eps` 這一欄就不再描述實際的擾動大小。"""
    spec = danp.SPEC_PAPER
    assert spec.steps * spec.step_size > spec.eps


def test_步長不衰減():
    spec = danp.SPEC_PAPER
    for it in (0, 17, spec.steps - 1):
        assert step_size_at(spec, it) == pytest.approx(spec.step_size)


def test_linf投影把擾動限制在gamma內():
    spec = danp.SPEC_PAPER
    x = torch.rand(1, 3, 8, 8, dtype=torch.float64)
    far = (x + 0.5).clamp(0.0, 1.0)
    from src.baselines.pgd import project
    out = project(far, x, spec)
    assert float((out - x).abs().max()) <= spec.eps + 1e-12
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ===========================================================================
# 四、BaselineSpec 一致性
# ===========================================================================


def test_spec逐欄對應Algorithm1():
    s = danp.SPEC_PAPER
    assert s.name == "danp"
    assert s.norm == "linf"                    # §IV-D ‖x_imu − x₀‖∞ ≤ γ
    assert s.objective == "minimize"           # Algorithm 1 第 13 行是減號
    assert s.update_rule == "sign"             # Algorithm 1 第 13 行 sign(g)
    assert s.init_rule == "none"               # Algorithm 1 第 1 行 δ ← 0
    assert s.step_schedule == "constant"       # α 不隨 n 改變
    assert s.steps == 100                      # §V-A，N = 100
    assert s.grad_reps == 10                   # Algorithm 1 第 5–12 行，|𝒯| = 10
    assert s.needs_target_image is False
    assert s.needs_mask is False               # 遮罩由 Eq. 10 自己算
    assert s.grad_outside_mask is False


def test_論文給的超參數逐項等於常數():
    e = danp.SPEC_PAPER.extras
    assert e["lambda_daa"] == danp.PAPER_LAMBDA_DAA == 1.0
    assert e["lambda_nba"] == danp.PAPER_LAMBDA_NBA == 1.0
    assert e["kapur_bins"] == danp.PAPER_KAPUR_BINS == 128
    assert e["num_timesteps"] == danp.PAPER_NUM_TIMESTEPS == 10
    assert danp.SPEC_PAPER.grad_reps == e["num_timesteps"]


def test_重建過就必須寫明重建了什麼():
    s = danp.SPEC_PAPER
    assert s.modified_from_paper is True
    assert s.modification_note.strip()
    assert s.discrepancy_note.strip()
    # 論文未給的三項必須出現在註記裡，否則報表會把它們讀成論文設定
    for token in ("步長", "值域", "head"):
        assert token in s.modification_note


def test_source註明沒有官方程式():
    assert "2512.14333" in danp.SPEC_PAPER.source
    assert "無官方程式" in danp.SPEC_PAPER.source


def test_spec不可變():
    with pytest.raises(Exception):
        danp.SPEC_PAPER.eps = 0.1


def test_NBA的縮放係數是本檔的決定且為一():
    """§V-F 說縮放過但沒給係數。填 1.0 等於照 Eq. 13 字面，
    這一格改動時必須同時改 `modification_note`。"""
    assert danp.RECONSTRUCTED_NBA_SCALE == 1.0
    assert danp.SPEC_PAPER.extras["nba_scale"] == 1.0
    assert "NBA" in danp.SPEC_PAPER.modification_note


# ===========================================================================
# 五、注意力擷取：拿到的必須是 softmax 機率
# ===========================================================================


def _attn_layer(query_dim=8, cross_dim=6, heads=2, dim_head=4):
    return Attention(
        query_dim=query_dim,
        cross_attention_dim=cross_dim,
        heads=heads,
        dim_head=dim_head,
    ).to(torch.float64)


def test_擷取到的是後softmax機率而不是注意力輸出():
    """每一列必須對 token 軸加總為 1。若誤取 `A·V` 或 `A·V·W_out`
    （PromptFlare 記的那個量），這個性質不會成立。"""
    torch.manual_seed(0)
    attn = _attn_layer()
    ctrl = danp.DANPAttnController()
    attn.set_processor(danp.DANPAttnProcessor(ctrl, "blocks.0.attn2"))
    hidden = torch.randn(1, 16, 8, dtype=torch.float64)
    enc = torch.randn(1, 5, 6, dtype=torch.float64)
    attn(hidden, encoder_hidden_states=enc)

    assert len(ctrl.maps) == 1
    m = ctrl.maps[0]
    assert m.shape == (1, 16, 5)               # head 已被平均掉
    assert torch.allclose(m.sum(-1), torch.ones(1, 16, dtype=torch.float64))
    assert bool((m >= 0).all())


def test_self_attention不被記錄():
    """Eq. 3 的 K_l 來自文字嵌入，attn1 不在定義內。"""
    torch.manual_seed(0)
    attn = _attn_layer(cross_dim=8)
    ctrl = danp.DANPAttnController()
    attn.set_processor(danp.DANPAttnProcessor(ctrl, "blocks.0.attn1"))
    attn(torch.randn(1, 16, 8, dtype=torch.float64))
    assert ctrl.maps == []


def test_processor收到未預期的引數直接中止():
    ctrl = danp.DANPAttnController()
    proc = danp.DANPAttnProcessor(ctrl, "blocks.0.attn2")
    with pytest.raises(RuntimeError):
        proc(None, None, foo=1)


def test_沒有記到任何層時聚合直接拋出():
    with pytest.raises(RuntimeError):
        danp.DANPAttnController().aggregate()


def test_聚合上採樣到最大網格並對層取平均():
    """Eq. 4：`(1/L)·Σ Upsample(A_l)`。粗的那一層被放大，
    常數圖放大後仍是同一個常數，故平均可以手算。"""
    ctrl = danp.DANPAttnController()
    fine = torch.full((1, 16, 3), 0.2, dtype=torch.float64)     # 4×4
    coarse = torch.full((1, 4, 3), 0.6, dtype=torch.float64)    # 2×2
    ctrl.maps = [fine, coarse]
    out = ctrl.aggregate()
    assert out.shape == (1, 16, 3)
    assert torch.allclose(out, torch.full_like(out, 0.4))


def test_非正方形的token數直接拋出():
    ctrl = danp.DANPAttnController()
    ctrl.maps = [torch.rand(1, 12, 3)]
    with pytest.raises(NotImplementedError):
        ctrl.aggregate()


def test_聚合保留梯度():
    ctrl = danp.DANPAttnController()
    a = torch.rand(1, 4, 3, dtype=torch.float64, requires_grad=True)
    ctrl.maps = [a]
    ctrl.aggregate().sum().backward()
    assert a.grad is not None


# ===========================================================================
# 六、端到端：替身 SD 上的 prepare / loss_fn / run_pgd
# ===========================================================================


class _Block(torch.nn.Module):
    """一個帶 cross-attention 的區塊，模組名以 `attn2` 結尾。"""

    def __init__(self, grid: int, channels: int, cross_dim: int):
        super().__init__()
        self.grid = grid
        self.attn2 = _attn_layer(
            query_dim=channels, cross_dim=cross_dim, heads=2, dim_head=4
        )


class _StubUNet(torch.nn.Module):
    """兩個 cross-attention 層級（4×4 與 2×2），輸出與輸入同形。"""

    def __init__(self, channels=3, cross_dim=6):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [_Block(4, channels, cross_dim), _Block(2, channels, cross_dim)]
        )

    def forward(self, z, emb):
        out = z * 2.0
        for blk in self.blocks:
            h = F.adaptive_avg_pool2d(z, blk.grid).flatten(2).transpose(1, 2)
            a = blk.attn2(h, encoder_hidden_states=emb)
            out = out + a.mean() * torch.ones_like(z)
        return out


class _StubSD:
    """把 VAE 換成平均池化、UNet 換成上面那個小模型。

    `encode_image` 取 8×8 區塊平均（32² 影像 → 4² latent，與最粗的注意力
    層級對得上），`alphas_cumprod` 由 1 單調遞減。
    """

    num_train_timesteps = 1000

    def __init__(self, device="cpu", dtype=torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.unet = _StubUNet().to(dtype)
        self.forward_ts = []

    def alphas_cumprod(self, device=None):
        t = torch.arange(self.num_train_timesteps, dtype=self.dtype)
        return (1.0 - t / self.num_train_timesteps).clamp(min=1e-6)

    def encode_text(self, prompt):
        g = torch.Generator().manual_seed(len(prompt))
        return torch.randn(1, 5, 6, generator=g, dtype=self.dtype)

    def encode_image(self, x01):
        return F.avg_pool2d(x01, 8)

    def latent_shape(self, h, w):
        return (1, 3, h // 8, w // 8)

    def conditioning_for(self, x01, mask=None, vae_ckpt=False):
        import contextlib
        return contextlib.nullcontext()

    def unet_forward(self, z, t, emb, **kw):
        self.forward_ts.append(int(t))
        return self.unet(z, emb)


def _img(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g, dtype=torch.float64)


def test_沒有給prompt就拒絕執行():
    """DAA 的整個機制以編輯指令為條件，論文用的是逐張不同的原生指令，
    沒有可沿用的預設值。"""
    with pytest.raises(NotImplementedError):
        danp.prepare(_StubSD(), _img(), danp.SPEC_PAPER)


def test_timesteps是零到T減一的等距格點():
    ts = danp.paper_timesteps(_StubSD(), 10)
    assert len(ts) == 10
    assert int(ts[0]) == 0 and int(ts[-1]) == 999
    diffs = (ts[1:] - ts[:-1]).tolist()
    assert max(diffs) - min(diffs) <= 1


def test_一次迭代走訪每個timestep恰好一次():
    """Algorithm 1 第 5–12 行：一次迭代對 𝒯 的每個元素各求一次梯度。
    骨幹以 `grad_reps` 次呼叫 loss_fn 實現，故游標必須剛好繞完一圈。"""
    sd = _StubSD()
    ctx = danp.prepare(sd, _img(), danp.SPEC_PAPER, prompt="make it a dragon")
    seen = [int(ctx.next_timestep()) for _ in range(danp.SPEC_PAPER.grad_reps)]
    assert sorted(seen) == sorted(ctx.timesteps.tolist())
    assert int(ctx.next_timestep()) == seen[0]     # 下一輪從頭開始
    ctx.close()


def test_grad_reps與timestep數不一致時直接拋出():
    import dataclasses
    bad = dataclasses.replace(danp.SPEC_PAPER, grad_reps=3)
    with pytest.raises(ValueError):
        danp.prepare(_StubSD(), _img(), bad, prompt="x")


def test_loss_fn回傳純量且對防禦圖可微():
    sd = _StubSD()
    x = _img()
    ctx = danp.prepare(sd, x, danp.SPEC_PAPER, prompt="make it a dragon")
    probe = x.clone().requires_grad_(True)
    loss = danp.loss_fn(sd, probe, ctx)
    assert loss.dim() == 0 and torch.isfinite(loss)
    g = torch.autograd.grad(loss, probe)[0]
    assert float(g.abs().max()) > 0.0
    ctx.close()


def test_loss_fn兩條分支共用同一個timestep():
    """Algorithm 1 第 7、8 行用的是同一個 t。分開會讓 NBA 比較兩個
    不同時刻的噪聲預測，而那不是 Eq. 12。"""
    sd = _StubSD()
    ctx = danp.prepare(sd, _img(), danp.SPEC_PAPER, prompt="x")
    sd.forward_ts.clear()
    danp.loss_fn(sd, _img().requires_grad_(True), ctx)
    assert len(sd.forward_ts) == 2
    assert sd.forward_ts[0] == sd.forward_ts[1]
    ctx.close()


def test_loss_fn記下門檻供報表引用():
    sd = _StubSD()
    ctx = danp.prepare(sd, _img(), danp.SPEC_PAPER, prompt="x")
    danp.loss_fn(sd, _img().requires_grad_(True), ctx)
    assert ctx.last_tau_index is not None
    assert 0.0 <= ctx.last_tau_value <= 1.0
    assert 0.0 <= ctx.last_mask_fraction <= 1.0
    ctx.close()


def test_close還原原本的processor():
    """不還原會污染共用同一個 SDWrapper 的後續實驗。"""
    sd = _StubSD()
    before = [m.get_processor() for n, m in sd.unet.named_modules()
              if n.endswith("attn2")]
    ctx = danp.prepare(sd, _img(), danp.SPEC_PAPER, prompt="x")
    during = [m.get_processor() for n, m in sd.unet.named_modules()
              if n.endswith("attn2")]
    assert all(isinstance(p, danp.DANPAttnProcessor) for p in during)
    ctx.close()
    after = [m.get_processor() for n, m in sd.unet.named_modules()
             if n.endswith("attn2")]
    assert after == before


def test_找不到attn2層直接拋出():
    sd = _StubSD()
    sd.unet = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError):
        danp.prepare(sd, _img(), danp.SPEC_PAPER, prompt="x")


def test_run_pgd在替身上跑得完且擾動落在預算內():
    """骨幹與 spec 的接線：起點是原圖、更新是 sign、投影是 L∞ γ。
    步數縮短只是為了讓 CPU 測試跑得完，其餘欄位照 SPEC_PAPER。"""
    import dataclasses
    spec = dataclasses.replace(danp.SPEC_PAPER, steps=3)
    sd = _StubSD()
    x = _img()
    res = run_pgd(sd, x, spec, seed=0, verbose=False,
                  prompt="make the girl a giant dragon")
    assert res.x_adv01.shape == x.shape
    assert float(res.delta01.abs().max()) <= spec.eps + 1e-12
    assert float(res.delta01.abs().max()) > 0.0
    assert len(res.history) == 3
    assert all(math.isfinite(r["loss"]) for r in res.history)
    # 每一步走訪 |𝒯| 次，每次兩條分支
    assert len(sd.forward_ts) == 3 * spec.grad_reps * 2
