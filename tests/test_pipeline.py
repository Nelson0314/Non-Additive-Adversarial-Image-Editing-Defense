"""E1 正確性測試 — spec §7.2。

設計為模型無關，本機以 tiny-SD 在 CPU 執行；TWCC 上換真實 SD 跑同一組
測試，一旦數字不對可立即分辨是邏輯錯誤還是規模問題。

**T1 的敘述已相對 spec v1 修正。** 原文為「φ=0 ⟹ d(y_def, y_orig) ≈ 0」，
該敘述僅對 site P 成立（Δ=0 使 x_def = x 恆等）。site L 即使殘差為零，
x_def 仍是 DDIM inversion + 去噪 + VAE 來回的重建，本身即帶誤差，
y_def 不可能等於 y_orig。正確的不變量是「模塊停用時其存在不改變任何
計算結果」，本檔依此撰寫，並保留 site P 的恆等檢查作為 T1-P。
"""

import pytest
import torch

from src.defense.generator import DefenseGenerator
from src.defense.objective import LossConfig
from src.defense.optimize import OptimConfig, align, optimize
from src.models.sd import SDWrapper
from src.purify.ops import default_train_set
from src.residual.site_embedding import EmbeddingResidual
from src.residual.site_latent import LatentResidual
from src.residual.base import ResidualModule
from src.residual.site_pixel import PixelResidual
from src.residual.site_pixel_full import FullRankPixelResidual
from src.utils.device import get_device

TINY = "hf-internal-testing/tiny-stable-diffusion-pipe"
SIZE = 64
SEED = 20260728

# SDWrapper 把模型放到 get_device()，測試自造的張量必須放到同一裝置。
# 本機 CPU-only 時兩者都是 cpu，裝置不符不會顯現；此檔在 GPU 上跑才會抓到。
DEV = get_device()


@pytest.fixture(scope="module")
def sd():
    return SDWrapper(TINY)


@pytest.fixture(scope="module")
def x01():
    # generator 綁定 CPU，故先在 CPU 生成再搬移，保證跨裝置取到同一組亂數
    g = torch.Generator().manual_seed(SEED)
    return torch.rand(1, 3, SIZE, SIZE, generator=g).to(DEV)


def _latent(sd):
    return sd.latent_shape(SIZE, SIZE)


def _noise(sd, lat, seed=SEED):
    return sd.sample_edit_noise(torch.empty(lat, device=DEV), seed=seed)


# ------------------------------------------------------------------ T1-P


def test_T1_P_殘差為零時防禦圖與原圖恆等(sd, x01):
    """site P：Δ=0 ⟹ x_def = x 逐元素相等，無任何重建誤差。"""
    lat = _latent(sd)
    mod = PixelResidual(size=SIZE, max_rank=8, const_rank=8).to(DEV)
    gen = DefenseGenerator(sd, mod, k_inv=2)
    ctx = gen.prepare(x01)
    x_def = gen.generate(x01, ctx)
    assert torch.equal(x_def, x01), "V 初始為零，x_def 必須與 x 完全相同"
    assert lat[1] >= 1


def test_T1_P_編輯結果在殘差為零時完全相同(sd, x01):
    lat = _latent(sd)
    emb = sd.encode_text("a photo").detach()
    noise = _noise(sd, lat)

    mod = PixelResidual(size=SIZE, max_rank=8, const_rank=8).to(DEV)
    gen = DefenseGenerator(sd, mod, k_inv=2)
    x_def = gen.generate(x01, gen.prepare(x01))

    with torch.no_grad():
        y_orig = sd.sdedit(x01, emb, noise, num_steps=3)
        y_def = sd.sdedit(x_def, emb, noise, num_steps=3)
    assert torch.equal(y_def, y_orig)


# ------------------------------------------------------------------ T1-L


def test_T1_L_停用模塊不改變去噪結果(sd, x01):
    """site L 的正確不變量：模塊停用時，其存在不得改變任何計算結果。"""
    k = 2
    lat = _latent(sd)
    emb = sd.encode_text("").detach()
    ts = sd.timesteps(k)

    with torch.no_grad():
        z0 = sd.encode_image(x01)
        z_inv = sd.ddim_inversion(z0, emb, ts, k)
        baseline, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)

    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4).to(DEV)
    mod.disable()
    with torch.no_grad():
        withmod, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=mod.eps_hook(ts, k))

    assert torch.equal(baseline, withmod), "停用的模塊改變了去噪結果"


def test_T1_L_初始參數下殘差為零且不改變結果(sd, x01):
    """V=0 使 Δ=0，即使模塊啟用，結果也必須與無模塊一致。"""
    k = 2
    lat = _latent(sd)
    emb = sd.encode_text("").detach()
    ts = sd.timesteps(k)

    with torch.no_grad():
        z_inv = sd.ddim_inversion(sd.encode_image(x01), emb, ts, k)
        baseline, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)

    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4).to(DEV)
    assert mod.enabled
    with torch.no_grad():
        out, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=mod.eps_hook(ts, k))

    assert torch.allclose(baseline, out, atol=0, rtol=0)


def test_T1_L_參數非零時結果必須改變(sd, x01):
    """反向檢查：V 填入非零值後，去噪結果必須不同，否則注入根本沒生效。"""
    k = 2
    lat = _latent(sd)
    emb = sd.encode_text("").detach()
    ts = sd.timesteps(k)

    with torch.no_grad():
        z_inv = sd.ddim_inversion(sd.encode_image(x01), emb, ts, k)
        baseline, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)

    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4).to(DEV)
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.5)
        out, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=mod.eps_hook(ts, k))

    assert not torch.allclose(baseline, out, atol=1e-6)


# -------------------------------------------------------------------- T2


@pytest.mark.parametrize("r", [1, 2, 4])
def test_T2_注入的latent殘差秩精確等於設定值(sd, r):
    """秩約束精確作用於注入的 latent 殘差（spec §4.3）。"""
    lat = _latent(sd)
    n = lat[-1]
    mod = LatentResidual(steps=3, channels=lat[1], size=n, max_rank=4, const_rank=r).to(DEV)
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.02)

    delta = mod.tensor(step=0, rank=r)
    for c in range(delta.shape[0]):
        sv = torch.linalg.svdvals(delta[c].double())
        assert sv[r - 1] / sv[0] > 1e-6
        if r < n:
            assert (sv[r] / sv[0]).item() < 1e-6


def test_T2_像素殘差的秩是湧現的須實測不可假設(sd, x01):
    """site L 的像素秩沒有理論保證，本測試只記錄實測值、不對數值下斷言。

    唯一的斷言是「它確實被量到了」，避免此診斷在重構中被靜默移除。
    """
    k = 2
    lat = _latent(sd)
    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=2).to(DEV)
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.3)

    gen = DefenseGenerator(sd, mod, k_inv=k)
    with torch.no_grad():
        ctx = gen.prepare(x01, prompt_def="")
        x_def = gen.generate(x01, ctx)
        delta = (x_def - x01)[0]

    ranks = [
        torch.linalg.matrix_rank(delta[c].double(), rtol=1e-6).item()
        for c in range(delta.shape[0])
    ]
    print(f"\n[T2] 注入 latent 秩=2 → 像素殘差實測秩（逐通道）: {ranks}")
    assert len(ranks) == delta.shape[0]
    assert all(isinstance(v, int) for v in ranks)


# -------------------------------------------------------------------- T3


def test_T3_兩分支噪聲逐元素相同(sd):
    lat = _latent(sd)
    a = _noise(sd, lat)
    b = _noise(sd, lat)
    assert torch.equal(a, b)

    c = _noise(sd, lat, SEED + 1)
    assert not torch.equal(a, c), "不同 seed 必須給出不同噪聲"


def test_T3_相同噪聲下編輯管線為決定性(sd, x01):
    lat = _latent(sd)
    emb = sd.encode_text("a photo").detach()
    noise = _noise(sd, lat)
    with torch.no_grad():
        y1 = sd.sdedit(x01, emb, noise, num_steps=3)
        y2 = sd.sdedit(x01, emb, noise, num_steps=3)
    assert torch.equal(y1, y2)


def test_T3_不同噪聲下編輯結果不同(sd, x01):
    """確認噪聲確實影響輸出——否則 T3 的共用檢查沒有意義。"""
    lat = _latent(sd)
    emb = sd.encode_text("a photo").detach()
    n1 = _noise(sd, lat)
    n2 = _noise(sd, lat, SEED + 99)
    with torch.no_grad():
        y1 = sd.sdedit(x01, emb, n1, num_steps=3)
        y2 = sd.sdedit(x01, emb, n2, num_steps=3)
    assert not torch.allclose(y1, y2, atol=1e-6)


# ------------------------------------------------------------ 梯度與快取


def test_梯度抵達phi且SD保持凍結(sd, x01):
    """spec §5.3：φ 是唯一可訓練參數，凍結參數不阻斷梯度。"""
    k, n = 2, 2
    lat = _latent(sd)
    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4).to(DEV)
    gen = DefenseGenerator(sd, mod, k_inv=k)
    emb = sd.encode_text("a photo").detach()
    noise = _noise(sd, lat)

    with torch.no_grad():
        y_orig = sd.sdedit(x01, emb, noise, n)

    ctx = gen.prepare(x01)
    x_def = gen.generate(x01, ctx)
    y_def = sd.sdedit(x_def, emb, noise, n)
    (-(y_def - y_orig).pow(2).mean()).backward()

    assert mod.tensor.V.grad is not None
    assert mod.tensor.V.grad.abs().sum().item() > 0, "梯度未抵達 φ"
    assert all(not p.requires_grad for p in sd.unet.parameters())
    assert all(not p.requires_grad for p in sd.vae.parameters())


def test_inversion快取不依賴phi(sd, x01):
    """spec §4.3 效率設計的前提：z_inv 與 φ 無關，改變 φ 不應改變 z_inv。"""
    k = 2
    lat = _latent(sd)
    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4).to(DEV)
    gen = DefenseGenerator(sd, mod, k_inv=k)

    ctx_a = gen.prepare(x01)
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.5)
    ctx_b = gen.prepare(x01)

    assert torch.equal(ctx_a.z_inv, ctx_b.z_inv), "z_inv 受到了 φ 影響，快取前提不成立"
    assert mod.enabled, "prepare() 必須還原模塊的啟用狀態"


def test_中間圖可留存(sd, x01):
    """spec §8.3：去噪每步的 x̂₀ 估計必須可取得。"""
    k = 3
    lat = _latent(sd)
    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=2).to(DEV)
    gen = DefenseGenerator(sd, mod, k_inv=k)
    with torch.no_grad():
        ctx = gen.prepare(x01)
        gen.generate(x01, ctx, collect_x0=True)
    assert len(ctx.x0_trace) == k
    assert all(t.shape == tuple(lat) for t in ctx.x0_trace)


def test_秩排程沿去噪步序的方向(sd):
    """排程名稱的遞增／遞減是對 timestep t 而言，去噪走訪 t 遞減，故方向相反。

    LD = ceil((1 − t/T)·R_m)：t=0（乾淨）秩最高、t=T（高噪）秩最低。
    去噪由高噪走向乾淨 ⟹ 沿 step_idx **遞增**。LI 則相反。
    """
    lat = _latent(sd)
    ts = sd.timesteps(4)

    ld = LatentResidual(
        steps=4, channels=lat[1], size=lat[-1], max_rank=4, schedule="LD"
    ).to(DEV).rank_trace(ts, 4)
    li = LatentResidual(
        steps=4, channels=lat[1], size=lat[-1], max_rank=4, schedule="LI"
    ).to(DEV).rank_trace(ts, 4)

    assert len(ld) == len(li) == 4
    assert ld == sorted(ld), f"LD 沿去噪步序應遞增：{ld}"
    assert li == sorted(li, reverse=True), f"LI 沿去噪步序應遞減：{li}"
    # 去噪走訪的是 ts[steps..1]，**t=0 那一格永遠不會被評估**（最後一步以
    # ts[1] 的 ε 更新至 ts[0]）。故排程只碰得到 t=t_max 這一端的端點值，
    # 碰不到 t=0 端。斷言只能對 t=t_max 端下。
    assert ld[0] == 0, f"LD 在 t=t_max 應為 0：{ld}"
    assert li[0] == 4, f"LI 在 t=t_max 應為 max_rank：{li}"


def test_checkpoint不改變數值結果(sd, x01):
    """UNet 與 VAE 的 gradient checkpointing 只影響記憶體，不得影響數值。

    checkpoint 在反向時重算前向，若前向含有非決定性成分（dropout、
    非決定性 kernel），重算結果會與原本不同而導致梯度錯誤。此測試同時
    檢查前向數值與 φ 的梯度，兩者都必須一致。

    **四組必須共用同一個模塊實例。** 每組各自 `LatentResidual(...)` 會讓
    U 取到不同的亂數（U 是唯一的隨機來源），量到的差異其實來自初始化而非
    checkpoint。此陷阱曾使本測試誤報：單一 UNet 步與單次 VAE decode 的
    checkpoint 前向差為 0.0，完整路徑卻差 5e-4，正是由此而來。
    """
    k, n = 2, 2
    lat = _latent(sd)
    emb = sd.encode_text("a photo").detach()
    noise = _noise(sd, lat)

    mod = LatentResidual(
        steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4, seed=SEED
    ).to(DEV)
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.1, generator=torch.Generator(DEV).manual_seed(SEED))
    gen = DefenseGenerator(sd, mod, k_inv=k)

    grads, outs = [], []
    for unet_ck, vae_ck in [(False, False), (True, False), (False, True), (True, True)]:
        mod.zero_grad(set_to_none=True)
        ctx = gen.prepare(x01)
        x_def = gen.generate(x01, ctx, use_ckpt=unet_ck, vae_ckpt=vae_ck)
        y_def = sd.sdedit(x_def, emb, noise, n, use_ckpt=unet_ck, vae_ckpt=vae_ck)
        y_def.pow(2).mean().backward()

        outs.append(y_def.detach().clone())
        grads.append(mod.tensor.V.grad.clone())

    for i in range(1, 4):
        assert torch.equal(outs[0], outs[i]), f"組合 {i} 的前向數值不同"
        assert torch.allclose(grads[0], grads[i], rtol=1e-5, atol=1e-8), (
            f"組合 {i} 的梯度不同，"
            f"相對差 {(grads[0] - grads[i]).norm() / grads[0].norm():.3e}"
        )


def test_相同seed的模塊初始化完全相同(sd):
    """U 是唯一的隨機來源，未固定 seed 會讓跨組比較混入初始化差異。"""
    lat = _latent(sd)
    kw = dict(steps=2, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4)

    a = LatentResidual(seed=SEED, **kw)
    b = LatentResidual(seed=SEED, **kw)
    c = LatentResidual(seed=SEED + 1, **kw)
    assert torch.equal(a.tensor.U, b.tensor.U), "同 seed 必須得到相同的 U"
    assert not torch.equal(a.tensor.U, c.tensor.U), "不同 seed 必須得到不同的 U"


def test_const排程每步相同(sd):
    lat = _latent(sd)
    ts = sd.timesteps(4)
    trace = LatentResidual(
        steps=4, channels=lat[1], size=lat[-1], max_rank=4,
        schedule="const", const_rank=3,
    ).to(DEV).rank_trace(ts, 4)
    assert trace == [3, 3, 3, 3]


# ------------------------------------------------------- 階段一：保真對齊


def _cfg(**kw):
    base = dict(steps=1, k_inv=2, n_edit=2, n_eot=1, lr=0.01,
                align_steps=2, align_lr=0.05, log_every=100, seed=SEED)
    base.update(kw)
    return OptimConfig(**base)


def _latent_module(sd, steps):
    lat = _latent(sd)
    return LatentResidual(
        steps=steps, channels=lat[1], size=lat[-1], max_rank=4,
        const_rank=4, seed=SEED,
    ).to(DEV)


def test_階段一會改變phi且記錄每一步(sd, x01):
    """階段一是一段真的優化，不是佔位。

    只驗結構性質——步數、φ 有動、每步都有記錄。不驗「重建誤差一定下降」：
    tiny-SD 是隨機初始化的測試模型，其上的收斂行為不能代表真實 SD，
    在這裡斷言收斂會得到一個看起來會過但沒有意義的測試。
    """
    cfg = _cfg()
    mod = _latent_module(sd, cfg.k_inv)
    gen = DefenseGenerator(sd, mod, k_inv=cfg.k_inv)
    before = mod.tensor.V.detach().clone()

    x_align, hist = align(sd, mod, x01, cfg, LossConfig(), gen)

    assert len(hist) == cfg.align_steps, "每一步都必須留下記錄"
    assert all("fid_lpips" in h and "fid_psnr_total" in h for h in hist), \
        "必須記錄重建品質，呼叫端據此判斷容量是否足夠"
    assert not torch.equal(mod.tensor.V, before), "φ 必須真的被更新"
    assert x_align.shape == x01.shape
    assert torch.isfinite(x_align).all()


def test_階段一後保真基準改為對齊結果(sd, x01):
    """x_base 必須是 G(x; φ_align) 而非 G(x; φ=0)。

    若仍用 φ=0 的重建當基準，階段一好不容易吸收掉的重建誤差會被當成
    「防禦可以自由使用的預算」，保真度約束等於被放寬了。
    """
    cfg = _cfg()
    mod = _latent_module(sd, cfg.k_inv)
    res = optimize(sd, mod, x01, cfg, LossConfig(), default_train_set())

    assert res.x_base0 is not None and res.x_base is not None
    assert len(res.align_history) == cfg.align_steps
    assert not torch.equal(res.x_base, res.x_base0), \
        "階段一改動了 φ，保真基準必須跟著改為對齊後的重建"


def test_無重建誤差時略過階段一(sd, x01):
    """site P 的 G(x; φ=0) 逐元素等於 x，沒有東西可對齊。

    判準是數值上「這個位置有沒有重建誤差」，不是 site 名稱，故以
    x_base0 與 x 是否相等來決定，將來新增注入位置不需改動這段邏輯。
    """
    cfg = _cfg()
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4, seed=SEED).to(DEV)
    res = optimize(sd, mod, x01, cfg, LossConfig(), default_train_set())

    assert torch.equal(res.x_base0, x01), "site P 的 φ=0 輸出必須等於原圖"
    assert res.align_history == [], "沒有重建誤差時階段一必須被略過"
    assert res.align_seconds == 0.0


# ------------------------------------------- generator 依能力而非 site 分派


def test_像素側模塊不因site名稱而走錯路徑(sd, x01):
    """generator 必須依「提供哪種能力」分派，不得比對 site 名稱。

    原本的條件是 `pixel is not None and self.module.site == "P"`。新增全秩
    對照（site "PF"）時該條件失效：模塊確實提供 pixel_residual，卻因名稱
    不是 "P" 而被送進 inversion + 去噪路徑，取到的 eps_hook 為 None，φ 完全
    沒有進入計算圖。症狀只在 backward 出現，訊息是 "does not require grad"，
    完全看不出真正原因。
    """
    mod = FullRankPixelResidual(size=SIZE).to(DEV)
    gen = DefenseGenerator(sd, mod, k_inv=2)

    ctx = gen.prepare(x01)
    assert ctx.z_inv is None, "像素側模塊不需要 inversion 快取"

    with torch.no_grad():
        mod.delta_param.copy_(torch.full_like(mod.delta_param, 0.01))
    x_def = gen.generate(x01, ctx)

    assert x_def.requires_grad, "φ 必須在計算圖上"
    assert not torch.equal(x_def, x01), "φ 非零時防禦圖必須改變"
    x_def.pow(2).sum().backward()
    assert mod.delta_param.grad is not None
    assert float(mod.delta_param.grad.abs().max()) > 0.0


def test_兩種能力都不提供時明確報錯(sd, x01):
    """φ 進不了計算圖要當場講，不能拖到 backward 才以無關的訊息出現。"""
    class _Empty(ResidualModule):
        site = "X"

    gen = DefenseGenerator(sd, _Empty().to(DEV), k_inv=2)
    ctx = gen.prepare(x01)
    with pytest.raises(ValueError, match="無法進入計算圖"):
        gen.generate(x01, ctx)


# ------------------------------------------------------------------ site E


def _emb_module(sd, r=4, prompt=""):
    emb = sd.encode_text(prompt)
    return EmbeddingResidual(
        tokens=emb.shape[-2], dim=emb.shape[-1],
        max_rank=r, const_rank=r, seed=SEED,
    ).to(DEV), emb


def test_site_E_初始殘差為零且不改變去噪結果(sd, x01):
    """φ=0 對照必須從第一天就成立。site L 白跑了 36 格才發現 φ 貢獻為零。"""
    mod, emb = _emb_module(sd)
    assert torch.equal(mod.delta(), torch.zeros_like(mod.delta())), "V=0 ⟹ Δ=0"

    gen = DefenseGenerator(sd, mod, k_inv=2)
    ctx = gen.prepare(x01)
    with torch.no_grad():
        out_zero = gen.generate(x01, ctx)
        mod.disable()
        out_off = gen.generate(x01, gen.prepare(x01))
    assert torch.equal(out_zero, out_off), "φ=0 與停用必須給出完全相同的結果"


def test_site_E_參數非零時結果必須改變(sd, x01):
    """反向檢查：確認上一個測試不是把注入整個關掉了。"""
    mod, _ = _emb_module(sd)
    gen = DefenseGenerator(sd, mod, k_inv=2)
    ctx = gen.prepare(x01)
    with torch.no_grad():
        before = gen.generate(x01, ctx)
        mod.tensor.V.normal_(0, 0.5)
        after = gen.generate(x01, gen.prepare(x01))
    assert not torch.allclose(before, after, atol=1e-6), "嵌入擾動未生效"


def test_site_E_梯度抵達phi(sd, x01):
    mod, _ = _emb_module(sd)
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.1)
    gen = DefenseGenerator(sd, mod, k_inv=2)
    gen.generate(x01, gen.prepare(x01)).pow(2).sum().backward()
    assert mod.tensor.V.grad is not None
    assert float(mod.tensor.V.grad.abs().max()) > 0.0


def test_site_E_inversion快取不依賴phi(sd, x01):
    """快取不變量：DDIM inversion 必須用未擾動的嵌入。

    若 inversion 也吃了擾動，z_inv 會依賴 φ，prepare() 的快取失效——那個
    快取每個 iteration 省下一條 k_inv 步的 UNet 前向，是本迴圈最大的節省。
    """
    mod, _ = _emb_module(sd)
    gen = DefenseGenerator(sd, mod, k_inv=2)
    z_before = gen.prepare(x01).z_inv.clone()

    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.5)
    z_after = gen.prepare(x01).z_inv

    assert torch.equal(z_before, z_after), "z_inv 不得依賴 φ"


def test_site_E_嵌入形狀不符時報錯(sd):
    """靜默廣播會讓「模塊建錯了」延後到數值階段才顯現。"""
    mod = EmbeddingResidual(tokens=8, dim=16, max_rank=2, const_rank=2).to(DEV)
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.1)
    with pytest.raises(ValueError, match="不符"):
        mod.emb_residual(torch.zeros(1, 77, 768, device=DEV))
