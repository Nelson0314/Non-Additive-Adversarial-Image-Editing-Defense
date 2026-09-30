"""`src/baselines/dayn.py` 的驗收 —— 用替身 SD，不載入真模型。

DAYN（Lo et al., CVPR 2024，論文自稱 semantic attack）沒有官方程式可比對，
整個實作的正確性只能靠「論文寫的性質有沒有被釘住」。這裡釘六組：

1. **`c_a` 的 token 定位**（Eq. 2）：定位正確，以及**定位失敗時必須拋錯**。
   退回「取全部 token」或「取第 1 格」會讓攻擊壓到別的東西而沒有症狀。
2. **注意力擷取拿到的是 softmax 機率**：每一列對 token 軸加總為 1。
   若誤取 `A·V` 或 `A·V·W_out`，這個性質不會成立。
3. **Eq. 3 的聚合是相加、不是平均**，且插值用 bicubic。
4. **損失方向的符號**（Eq. 5）：最小化必須**壓低** `c_a` 在其區域的注意力。
   符號寫反不會有症狀，只是方法變成把注意力加強在該區域。
5. **timestep-universal 更新的算術**（Algorithm 1 第 5–12 行）：一次 PGD
   迭代走訪 `𝒯` 的每個元素恰好一次，位移等於 `−s·sign(平均梯度)`。
6. **預算換算與值域**，以及 `BaselineSpec` 逐欄對 Algorithm 1。
"""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from src.baselines import dayn
from src.baselines.pgd import project, run_pgd, step_size_at


# ===========================================================================
# 一、c_a 的 token 定位（Eq. 2）
# ===========================================================================


class _StubTokenizer:
    """字詞層級的替身 tokenizer，介面與 `CLIPTokenizer` 相同的那幾項。

    `[BOS]=0`、`[EOS]=1`、`[PAD]=2`，其餘字詞依出現順序配號。釘的是
    `locate_content_tokens` 的演算法（去掉 BOS/EOS 後找連續子序列），
    不是 CLIP 的分詞結果。
    """

    model_max_length = 12
    BOS, EOS, PAD = 0, 1, 2

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        out = [self.BOS]
        for word in text.split():
            self.vocab.setdefault(word, 3 + len(self.vocab))
            out.append(self.vocab[word])
        out.append(self.EOS)
        return out

    def __call__(self, text, padding=None, max_length=None, truncation=False,
                 return_tensors=None):
        ids = self._ids(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            ids = ids + [self.PAD] * (max_length - len(ids))
        return {"input_ids": ids}


def test_只給內容時token落在BOS之後():
    tok = _StubTokenizer()
    assert dayn.locate_content_tokens(tok, "dog") == [1]
    assert dayn.locate_content_tokens(tok, "old woman") == [1, 2]


def test_在較長的序列裡定位到正確的位移():
    """`c_a` 的欄選錯不會有症狀——攻擊照跑，只是壓到別的詞。"""
    tok = _StubTokenizer()
    idx = dayn.locate_content_tokens(tok, "woman", prompt="a photo of a woman")
    ids = tok("a photo of a woman", padding="max_length",
              max_length=tok.model_max_length, truncation=True)["input_ids"]
    core = tok("woman")["input_ids"][1:-1]
    assert [ids[i] for i in idx] == core
    assert idx == [5]


def test_多詞的內容回傳連續索引():
    tok = _StubTokenizer()
    idx = dayn.locate_content_tokens(
        tok, "old woman", prompt="a photo of an old woman in a street"
    )
    assert idx == [5, 6]
    assert idx[1] - idx[0] == 1


def test_定位失敗直接拋錯而不是退回預設():
    tok = _StubTokenizer()
    with pytest.raises(ValueError):
        dayn.locate_content_tokens(tok, "cat", prompt="a dog in the park")


def test_空的內容直接拋錯():
    tok = _StubTokenizer()
    with pytest.raises(ValueError):
        dayn.locate_content_tokens(tok, "   ")


def test_只有標點之類沒有字詞的內容也拋錯():
    """去掉 BOS/EOS 之後沒有任何 token 的情形。"""
    tok = _StubTokenizer()
    with pytest.raises(ValueError):
        dayn.locate_content_tokens(tok, "\t\n ")


# ===========================================================================
# 二、注意力擷取：拿到的必須是 softmax 機率
# ===========================================================================


def _attn_layer(query_dim=8, cross_dim=6, heads=2, dim_head=4):
    return Attention(
        query_dim=query_dim,
        cross_attention_dim=cross_dim,
        heads=heads,
        dim_head=dim_head,
    ).to(torch.float64)


def test_擷取到的是後softmax機率而不是注意力輸出():
    torch.manual_seed(0)
    attn = _attn_layer()
    ctrl = dayn.DAYNAttnController()
    attn.set_processor(dayn.DAYNAttnProcessor(ctrl, "blocks.0.attn2"))
    hidden = torch.randn(1, 16, 8, dtype=torch.float64)
    enc = torch.randn(1, 5, 6, dtype=torch.float64)
    attn(hidden, encoder_hidden_states=enc)

    assert len(ctrl.maps) == 1
    m = ctrl.maps[0]
    assert m.shape == (1, 16, 5)               # head 已被平均掉
    assert torch.allclose(m.sum(-1), torch.ones(1, 16, dtype=torch.float64))
    assert bool((m >= 0).all())


def test_self_attention不被記錄():
    """Eq. 2 的 K 來自 c_a 的文字嵌入，attn1 不在定義內。"""
    torch.manual_seed(0)
    attn = _attn_layer(cross_dim=8)
    ctrl = dayn.DAYNAttnController()
    attn.set_processor(dayn.DAYNAttnProcessor(ctrl, "blocks.0.attn1"))
    attn(torch.randn(1, 16, 8, dtype=torch.float64))
    assert ctrl.maps == []


def test_processor收到未預期的引數直接中止():
    ctrl = dayn.DAYNAttnController()
    proc = dayn.DAYNAttnProcessor(ctrl, "blocks.0.attn2")
    with pytest.raises(RuntimeError):
        proc(None, None, foo=1)


def test_沒有記到任何層時聚合直接拋出():
    with pytest.raises(RuntimeError):
        dayn.DAYNAttnController().aggregate()


def test_聚合是相加不是平均():
    """Eq. 3：「We sum the attention maps pixel by pixel」。
    常數圖經 bicubic 放大後仍是同一個常數，故總和可以手算。
    寫成平均會得到 0.4，那是 DANP 的 Eq. 4，不是本篇。"""
    ctrl = dayn.DAYNAttnController()
    fine = torch.full((1, 16, 3), 0.2, dtype=torch.float64)     # 4×4
    coarse = torch.full((1, 4, 3), 0.6, dtype=torch.float64)    # 2×2
    ctrl.maps = [fine, coarse]
    out = ctrl.aggregate()
    assert out.shape == (1, 16, 3)
    assert torch.allclose(out, torch.full_like(out, 0.8))


def test_上採樣用bicubic():
    """§3.2 明寫 bicubic。雙線性在非常數圖上會給出另一張圖。"""
    ctrl = dayn.DAYNAttnController()
    coarse = torch.tensor(
        [[[0.1], [0.4], [0.6], [0.9]]], dtype=torch.float64
    )                                                           # (1, 4, 1)，2×2
    ctrl.maps = [torch.zeros(1, 16, 1, dtype=torch.float64), coarse]
    out = ctrl.aggregate()
    expect = F.interpolate(
        coarse.transpose(1, 2).reshape(1, 1, 2, 2),
        size=(4, 4), mode="bicubic", align_corners=False,
    ).reshape(1, 1, 16).transpose(1, 2)
    assert torch.allclose(out, expect)


def test_非正方形的token數直接拋出():
    ctrl = dayn.DAYNAttnController()
    ctrl.maps = [torch.rand(1, 12, 3)]
    with pytest.raises(NotImplementedError):
        ctrl.aggregate()


def test_聚合保留梯度():
    ctrl = dayn.DAYNAttnController()
    a = torch.rand(1, 4, 3, dtype=torch.float64, requires_grad=True)
    ctrl.maps = [a]
    ctrl.aggregate().sum().backward()
    assert a.grad is not None


# ===========================================================================
# 三、c_a 的欄選取、Eq. 4 的遮罩
# ===========================================================================


def test_內容注意力是所選欄的和():
    att = torch.arange(24, dtype=torch.float64).reshape(1, 4, 6)
    out = dayn.content_attention(att, [1, 3])
    assert out.shape == (1, 4)
    assert torch.allclose(out, att[..., 1] + att[..., 3])


def test_沒有欄可選直接拋錯():
    with pytest.raises(ValueError):
        dayn.content_attention(torch.rand(1, 4, 6), [])


def test_欄索引超出範圍直接拋錯():
    with pytest.raises(IndexError):
        dayn.content_attention(torch.rand(1, 4, 6), [6])


def test_門檻預設是空間平均():
    m = torch.tensor([[0.0, 1.0, 2.0, 5.0]], dtype=torch.float64)
    assert float(dayn.default_tau(m)) == pytest.approx(2.0)


def test_遮罩取嚴格大於門檻的位置():
    m = torch.tensor([[0.0, 1.0, 2.0, 5.0]], dtype=torch.float64)
    mask = dayn.content_mask(m, dayn.default_tau(m))
    assert mask.tolist() == [[0.0, 0.0, 0.0, 1.0]]
    assert mask.shape == m.shape


def test_遮罩不帶梯度():
    m = torch.rand(1, 16, dtype=torch.float64, requires_grad=True)
    mask = dayn.content_mask(m, dayn.default_tau(m))
    assert not mask.requires_grad


def test_空遮罩直接拋錯而不是靜靜地什麼都不做():
    """常數注意力圖在「> 平均」之下選不到任何位置，此時 Eq. 5 的損失恆為
    0、梯度恆為零，PGD 會輸出與原圖逐位元相同的「防禦圖」。"""
    m = torch.full((1, 16), 0.3, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        dayn.content_mask(m, dayn.default_tau(m))


def test_可以指定絕對門檻():
    m = torch.tensor([[0.0, 1.0, 2.0, 5.0]], dtype=torch.float64)
    mask = dayn.content_mask(m, torch.tensor(0.5, dtype=torch.float64))
    assert mask.tolist() == [[0.0, 1.0, 1.0, 1.0]]


# ===========================================================================
# 四、損失方向的符號（Eq. 5）
# ===========================================================================


def test_損失等於遮罩內的注意力總和():
    cmap = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    loss = dayn.attention_suppressing_loss(cmap, mask)
    assert float(loss) == pytest.approx(4.0)


def test_梯度在遮罩內為正在遮罩外為零():
    """`∂L/∂Att = M ⊙ sign(Att)`。遮罩外必須完全沒有梯度——Eq. 5 只作用在
    含 c_a 的區域。"""
    cmap = (torch.rand(1, 8, dtype=torch.float64) + 0.5).requires_grad_(True)
    mask = torch.zeros_like(cmap)
    mask[:, :4] = 1.0
    g = torch.autograd.grad(dayn.attention_suppressing_loss(cmap, mask), cmap)[0]
    assert bool((g[:, :4] > 0).all())
    assert bool((g[:, 4:] == 0).all())


def test_沿負梯度走一步會壓低遮罩內的注意力():
    """把符號寫成「它對最佳化做了什麼」。`run_pgd` 的 `objective="minimize"`
    走的就是 `−grad`；寫反會變成把注意力加強在該區域。"""
    cmap = (torch.rand(1, 16, dtype=torch.float64) + 0.5).requires_grad_(True)
    mask = torch.zeros_like(cmap)
    mask[:, :8] = 1.0
    g = torch.autograd.grad(dayn.attention_suppressing_loss(cmap, mask), cmap)[0]
    stepped = cmap.detach() - 0.01 * g
    before = float((cmap.detach() * mask).sum())
    after = float((stepped * mask).sum())
    assert after < before
    # 遮罩外原封不動
    assert torch.allclose(stepped * (1 - mask), cmap.detach() * (1 - mask))


def test_形狀不合直接拋錯():
    with pytest.raises(ValueError):
        dayn.attention_suppressing_loss(torch.rand(1, 8), torch.rand(1, 4))


# ===========================================================================
# 五、預算換算與值域
# ===========================================================================


def test_值域是負一到一且預算換算成一半():
    spec = dayn.SPEC_PAPER
    assert (spec.value_range.lo, spec.value_range.hi) == (-1.0, 1.0)
    assert spec.value_range.scale == 2.0
    assert spec.eps == pytest.approx(dayn.PAPER_KAPPA)
    assert spec.eps_pixel01 == pytest.approx(dayn.PAPER_KAPPA / 2.0)
    assert spec.eps_pixel01 == pytest.approx(0.03)
    assert spec.step_size_pixel01 == pytest.approx(1.0 / 255.0)


def test_一百步的sign更新走得滿預算():
    """s=2/255、N=100 下 `100·s = 0.78 ≫ κ = 0.06`。若步長小到走不滿，
    `eps` 這一欄就不再描述實際的擾動大小。"""
    spec = dayn.SPEC_PAPER
    assert spec.steps * spec.step_size > spec.eps


def test_步長不衰減():
    spec = dayn.SPEC_PAPER
    for it in (0, 17, spec.steps - 1):
        assert step_size_at(spec, it) == pytest.approx(spec.step_size)


def test_linf投影把擾動限制在kappa內並夾回值域():
    spec = dayn.SPEC_PAPER
    x = torch.rand(1, 3, 8, 8, dtype=torch.float64) * 2.0 - 1.0
    far = x + 0.5
    out = project(far, x, spec)
    assert float((out - x).abs().max()) <= spec.eps + 1e-12
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0


# ===========================================================================
# 六、BaselineSpec 逐欄對 Algorithm 1
# ===========================================================================


def test_spec逐欄對應Algorithm1():
    s = dayn.SPEC_PAPER
    assert s.name == "dayn"
    assert s.norm == "linf"                 # Algorithm 1 第 12 行 clip(δ, −κ, κ)
    assert s.objective == "minimize"        # Eq. 5 是要被壓低的注意力
    assert s.update_rule == "sign"          # Algorithm 1 第 12 行 sign(all_grad)
    assert s.init_rule == "none"            # Algorithm 1 第 2 行 δ ← 0
    assert s.step_schedule == "constant"    # s 不隨 n 改變
    assert s.steps == 100                   # §4.1，N = 100
    assert s.grad_reps == 10                # Algorithm 1 第 5–11 行，|𝒯| = 10
    assert s.needs_target_image is False
    assert s.needs_mask is False            # 遮罩由 Eq. 4 自己算
    assert s.grad_outside_mask is False


def test_論文給的超參數逐項等於常數():
    e = dayn.SPEC_PAPER.extras
    assert dayn.PAPER_KAPPA == 0.06
    assert dayn.PAPER_ITERS == 100
    assert dayn.PAPER_NUM_TIMESTEPS == 10
    assert e["kappa"] == dayn.PAPER_KAPPA
    assert e["num_timesteps"] == dayn.PAPER_NUM_TIMESTEPS
    assert dayn.SPEC_PAPER.grad_reps == e["num_timesteps"]
    assert "bicubic" in e["upsample"]
    assert "相加" in e["aggregation"]


def test_重建過就必須寫明重建了什麼():
    s = dayn.SPEC_PAPER
    assert s.modified_from_paper is True
    assert s.modification_note.strip()
    assert s.discrepancy_note.strip()
    # 論文未給的四項必須出現在註記裡，否則報表會把它們讀成論文設定
    for token in ("步長", "值域", "τ", "head"):
        assert token in s.modification_note


def test_source註明沒有官方程式():
    assert "CVPR 2024" in dayn.SPEC_PAPER.source
    assert "無官方程式" in dayn.SPEC_PAPER.source


def test_spec不可變():
    with pytest.raises(Exception):
        dayn.SPEC_PAPER.eps = 0.1


def test_沒有註冊進REGISTRY():
    """`tests/test_baselines.py` 以 `AUDIT == REGISTRY` 稽核，
    加進去會打到既有測試（danp／sifm／tdae 也都沒註冊）。"""
    from src.baselines import REGISTRY
    assert "dayn" not in REGISTRY


# ===========================================================================
# 七、替身 SD 上的 prepare / loss_fn / run_pgd
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
    """把 VAE 換成平均池化、UNet 換成上面那個小模型。"""

    num_train_timesteps = 1000

    # 替身的權重必須是**固定**的：有兩項測試拿兩個 _StubSD 互相比對
    # （同 seed 的 prepare 要給同一張遮罩、run_pgd 要等於手算的一步），
    # 而 `_StubUNet()` 預設從全域 RNG 抽權重，兩個實例的注意力圖不同，
    # 症狀會被誤讀成 prepare 不可重現。建構時借用固定種子再把全域狀態還原，
    # 避免影響同一輪的其他測試。
    WEIGHT_SEED = 0

    def __init__(self, device="cpu", dtype=torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = _StubTokenizer()
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.WEIGHT_SEED)
            self.unet = _StubUNet().to(dtype)
        finally:
            torch.random.set_rng_state(state)
        self.forward_ts = []

    def alphas_cumprod(self, device=None):
        t = torch.arange(self.num_train_timesteps, dtype=self.dtype)
        return (1.0 - t / self.num_train_timesteps).clamp(min=1e-6)

    def encode_text(self, prompt):
        g = torch.Generator().manual_seed(len(prompt))
        return torch.randn(
            1, self.tokenizer.model_max_length, 6, generator=g, dtype=self.dtype
        )

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
    """值域壓在 `[0.2, 0.8]`，讓 `[-1,1]` 上的一步不會撞到夾邊。"""
    g = torch.Generator().manual_seed(seed)
    return 0.2 + 0.6 * torch.rand(1, 3, h, w, generator=g, dtype=torch.float64)


CONTENT = "dog"


def test_沒有給內容就拒絕執行():
    with pytest.raises(NotImplementedError):
        dayn.prepare(_StubSD(), _img(), dayn.SPEC_PAPER)


def test_傳入編輯prompt直接拒絕():
    """威脅模型：c_a 由防禦方選、prompt 由攻擊方寫，不是同一個人。
    從 prompt 推 c_a 會壓到攻擊方指定的東西，且沒有症狀。"""
    with pytest.raises(RuntimeError):
        dayn.prepare(
            _StubSD(), _img(), dayn.SPEC_PAPER,
            content=CONTENT, prompt="a cat in the park",
        )


def test_grad_reps與timestep數不一致時直接拋出():
    bad = dataclasses.replace(dayn.SPEC_PAPER, grad_reps=3)
    with pytest.raises(ValueError):
        dayn.prepare(_StubSD(), _img(), bad, content=CONTENT)


def test_timesteps是零到T減一的等距格點():
    ts = dayn.paper_timesteps(_StubSD(), 10)
    assert len(ts) == 10
    assert int(ts[0]) == 0 and int(ts[-1]) == 999
    diffs = (ts[1:] - ts[:-1]).tolist()
    assert max(diffs) - min(diffs) <= 1


def test_一次迭代走訪每個timestep恰好一次():
    """Algorithm 1 第 5–11 行：一次迭代對 𝒯 的每個元素各求一次梯度。
    骨幹以 `grad_reps` 次呼叫 loss_fn 實現，故游標必須剛好繞完一圈。"""
    sd = _StubSD()
    ctx = dayn.prepare(sd, _img(), dayn.SPEC_PAPER, content=CONTENT)
    seen = [int(ctx.next_timestep()) for _ in range(dayn.SPEC_PAPER.grad_reps)]
    assert sorted(seen) == sorted(ctx.timesteps.tolist())
    assert int(ctx.next_timestep()) == seen[0]     # 下一輪從頭開始
    ctx.close()


def test_遮罩由乾淨影像算且在攻擊過程中固定():
    """Eq. 4 的 M 取自 `Att(x, c_a)`，與 δ 無關。若誤用 x_adv，遮罩會隨
    攻擊漂移，被壓低的區域跟著縮小，而曲線照樣會動。"""
    sd = _StubSD()
    x = _img()
    ctx = dayn.prepare(sd, x, dayn.SPEC_PAPER, content=CONTENT)
    mask0 = ctx.mask.clone()
    assert not ctx.mask.requires_grad
    assert 0.0 < float(mask0.mean()) < 1.0
    far = dayn.DAYN_RANGE.from01((x + 0.2).clamp(0, 1)).requires_grad_(True)
    dayn.loss_fn(sd, far, ctx)
    assert torch.equal(ctx.mask, mask0)
    ctx.close()


def test_遮罩與門檻由同一張乾淨圖決定且可複現():
    sd_a, sd_b = _StubSD(), _StubSD()
    x = _img()
    a = dayn.prepare(sd_a, x, dayn.SPEC_PAPER, content=CONTENT, seed=3)
    b = dayn.prepare(sd_b, x, dayn.SPEC_PAPER, content=CONTENT, seed=3)
    assert torch.equal(a.mask, b.mask)
    assert float(a.tau) == pytest.approx(float(b.tau))
    assert float(a.tau) == pytest.approx(float(dayn.default_tau(a.clean_content_map)))
    a.close()
    b.close()


def test_loss_fn回傳純量且對防禦圖可微():
    sd = _StubSD()
    x = _img()
    ctx = dayn.prepare(sd, x, dayn.SPEC_PAPER, content=CONTENT)
    probe = dayn.DAYN_RANGE.from01(x).requires_grad_(True)
    loss = dayn.loss_fn(sd, probe, ctx)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert float(loss) > 0.0          # Att 非負，遮罩非空
    g = torch.autograd.grad(loss, probe)[0]
    assert float(g.abs().max()) > 0.0
    ctx.close()


def test_loss_fn一次前向只走一條分支():
    """Eq. 5 只有免疫圖那一條（沒有乾淨影像的對照項）。"""
    sd = _StubSD()
    x = _img()
    ctx = dayn.prepare(sd, x, dayn.SPEC_PAPER, content=CONTENT)
    sd.forward_ts.clear()
    dayn.loss_fn(sd, dayn.DAYN_RANGE.from01(x).requires_grad_(True), ctx)
    assert len(sd.forward_ts) == 1
    ctx.close()


def test_找不到attn2層直接拋出():
    sd = _StubSD()
    sd.unet = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError):
        dayn.prepare(sd, _img(), dayn.SPEC_PAPER, content=CONTENT)


def test_close還原原本的processor():
    """不還原會污染共用同一個 SDWrapper 的後續實驗。"""
    sd = _StubSD()
    before = [m.get_processor() for n, m in sd.unet.named_modules()
              if n.endswith("attn2")]
    ctx = dayn.prepare(sd, _img(), dayn.SPEC_PAPER, content=CONTENT)
    during = [m.get_processor() for n, m in sd.unet.named_modules()
              if n.endswith("attn2")]
    assert all(isinstance(p, dayn.DAYNAttnProcessor) for p in during)
    ctx.close()
    after = [m.get_processor() for n, m in sd.unet.named_modules()
             if n.endswith("attn2")]
    assert after == before


def test_run_pgd在替身上跑得完且擾動落在預算內():
    spec = dataclasses.replace(dayn.SPEC_PAPER, steps=3)
    sd = _StubSD()
    x = _img()
    res = run_pgd(sd, x, spec, seed=0, verbose=False, content=CONTENT)
    assert res.x_adv01.shape == x.shape
    assert float(res.delta01.abs().max()) <= spec.eps_pixel01 + 1e-12
    assert float(res.delta01.abs().max()) > 0.0
    assert len(res.history) == 3
    assert all(math.isfinite(r["loss"]) for r in res.history)
    # 每一步走訪 |𝒯| 次，每次一條分支；另加 prepare 算遮罩的 |𝒯| 次前向
    assert len(sd.forward_ts) == spec.grad_reps * (1 + 3)


def test_timestep_universal的更新等於負步長乘平均梯度的sign():
    """Algorithm 1 第 9–12 行的算術：`|𝒯|` 個時刻的梯度先取平均，
    再取 sign 乘步長。逐時刻更新（sign 在平均之前）會得到另一個解，
    而兩者的損失曲線看起來一樣。"""
    spec = dataclasses.replace(dayn.SPEC_PAPER, steps=1)
    x = _img(seed=5)

    sd = _StubSD()
    res = run_pgd(sd, x, spec, seed=11, verbose=False, content=CONTENT)

    # 用同一個 seed 重建一次 ctx，逐 t 取梯度再平均。prepare 消耗的隨機
    # 抽樣次數固定，故噪聲序列與上面那次逐位元相同。
    sd2 = _StubSD()
    ctx = dayn.prepare(sd2, x, spec, content=CONTENT, seed=11)
    x_ref = dayn.DAYN_RANGE.from01(x)
    grads = []
    for _ in range(spec.grad_reps):
        probe = x_ref.clone().requires_grad_(True)
        grads.append(torch.autograd.grad(dayn.loss_fn(sd2, probe, ctx), probe)[0])
    ctx.close()
    mean_grad = torch.stack(grads).mean(0)

    expect_paper = x_ref - spec.step_size * mean_grad.sign()
    expect01 = dayn.DAYN_RANGE.to01(expect_paper)
    assert torch.allclose(res.x_adv01, expect01, atol=1e-12)
    # 位移剛好是一步，且沒有被 κ 或值域夾到
    assert float(res.delta01.abs().max()) == pytest.approx(
        spec.step_size / 2.0, abs=1e-12
    )
