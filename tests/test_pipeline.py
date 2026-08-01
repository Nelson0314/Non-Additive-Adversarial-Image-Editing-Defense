"""E1 正確性測試 — spec §7.2。

設計為模型無關，本機以 tiny-SD 在 CPU 執行；TWCC 上換真實 SD 跑同一組
測試，一旦數字不對可立即分辨是邏輯錯誤還是規模問題。

T1 的敘述已相對 spec v1 修正。原文為「φ=0 ⟹ d(y_def, y_orig) ≈ 0」，
該敘述僅對 site P 成立（Δ=0 使 x_def = x 恆等）。site L 即使殘差為零，
x_def 仍是 DDIM inversion + 去噪 + VAE 來回的重建，本身即帶誤差，
y_def 不可能等於 y_orig。正確的不變量是「模塊停用時其存在不改變任何
計算結果」，本檔依此撰寫，並保留 site P 的恆等檢查作為 T1-P。
"""

import pytest
import torch

from src.defense.generator import DefenseGenerator
from src.defense.objective import DefenseObjective, LossConfig
from src.defense.optimize import (
    OptimConfig, align, optimize, optimize_encoder,
)
from src.models.sd import SDWrapper
from src.purify.ops import default_train_set
from src.residual.site_embedding import EmbeddingResidual
from src.residual.site_latent import LatentResidual
from src.residual.base import ResidualModule
from src.residual.site_pixel import PixelResidual
from src.residual.site_pixel_full import FullRankPixelResidual
from src.residual.site_warp import WarpResidual
from src.residual.site_weight import WeightResidual
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


def x01_plain():
    """與 `x01` fixture 同一張圖，但不需要 `sd`。

    site S 不經生成路徑，其單元測試不該為了拿一張圖而載入 tiny-SD。
    """
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
    去噪由高噪走向乾淨 ⟹ 沿 step_idx 遞增。LI 則相反。
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
    # 去噪走訪的是 ts[steps..1]，t=0 那一格永遠不會被評估（最後一步以
    # ts[1] 的 ε 更新至 ts[0]）。故排程只碰得到 t=t_max 這一端的端點值，
    # 碰不到 t=0 端。斷言只能對 t=t_max 端下。
    assert ld[0] == 0, f"LD 在 t=t_max 應為 0：{ld}"
    assert li[0] == 4, f"LI 在 t=t_max 應為 max_rank：{li}"


def test_checkpoint不改變數值結果(sd, x01):
    """UNet 與 VAE 的 gradient checkpointing 只影響記憶體，不得影響數值。

    checkpoint 在反向時重算前向，若前向含有非決定性成分（dropout、
    非決定性 kernel），重算結果會與原本不同而導致梯度錯誤。此測試同時
    檢查前向數值與 φ 的梯度，兩者都必須一致。

    四組必須共用同一個模塊實例。每組各自 `LatentResidual(...)` 會讓
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


# ------------------------------------------------------------------ site W


def test_site_W_初始B為零時完全不改變模型行為(sd, x01):
    """W' = W + (α/r)·B·A，B 初始為零 ⟹ W' 逐元素等於 W。

    這是「模塊停用時其存在不改變任何計算結果」在權重空間的形式。site W
    是唯一直接改動模型的位置，這個不變量若不成立，之後所有結果都無法歸因。
    """
    k = 2
    lat = _latent(sd)
    emb = sd.encode_text("").detach()
    ts = sd.timesteps(k)
    with torch.no_grad():
        z_inv = sd.ddim_inversion(sd.encode_image(x01), emb, ts, k)
        baseline, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)

    mod = WeightResidual(sd.unet, rank=4, seed=SEED).to(DEV)
    try:
        assert mod.n_layers > 0
        with torch.no_grad():
            out, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)
        assert torch.equal(baseline, out), "B=0 時去噪結果必須逐元素相同"

        mod.disable()
        with torch.no_grad():
            off, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)
        assert torch.equal(baseline, off), "停用時亦須逐元素相同"
    finally:
        mod.remove()


def test_site_W_B非零時結果必須改變且梯度抵達phi(sd, x01):
    """反向檢查：確認上一個測試不是把整個注入關掉了。"""
    k = 2
    lat = _latent(sd)
    emb = sd.encode_text("").detach()
    ts = sd.timesteps(k)
    with torch.no_grad():
        z_inv = sd.ddim_inversion(sd.encode_image(x01), emb, ts, k)
        baseline, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)

    mod = WeightResidual(sd.unet, rank=4, seed=SEED).to(DEV)
    try:
        with torch.no_grad():
            for h in mod.hooks.values():
                h.B.normal_(0, 0.5)
            out, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)
        assert not torch.allclose(baseline, out, atol=1e-6), "LoRA 未生效"

        z, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)
        z.pow(2).sum().backward()
        grads = [h.B.grad for h in mod.hooks.values() if h.B.grad is not None]
        assert grads, "梯度未抵達任何一層的 B"
        assert max(float(g.abs().max()) for g in grads) > 0.0
    finally:
        mod.remove()


def test_site_W_remove後模型完全還原(sd, x01):
    """hook 註冊在 SD 的模組上，模塊被回收不會移除它們。

    殘留的 hook 會污染同一個 SDWrapper 的後續實驗，而且症狀是「另一個
    site 的結果莫名其妙被改動」，極難追。
    """
    k = 2
    emb = sd.encode_text("").detach()
    ts = sd.timesteps(k)
    with torch.no_grad():
        z_inv = sd.ddim_inversion(sd.encode_image(x01), emb, ts, k)
        baseline, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)

    mod = WeightResidual(sd.unet, rank=4, seed=SEED).to(DEV)
    with torch.no_grad():
        for h in mod.hooks.values():
            h.B.normal_(0, 0.5)
        changed, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)
    assert not torch.allclose(baseline, changed, atol=1e-6)

    mod.remove()
    with torch.no_grad():
        restored, _ = sd.denoise(z_inv, emb, ts, k, eps_hook=None)
    assert torch.equal(baseline, restored), "remove() 後必須逐元素還原"


@pytest.mark.parametrize("r", [2, 4])
def test_site_W_參數量等於逐層r乘以進出維度之和(sd, r):
    """site W 的存在理由是容量，故參數量必須符合設計而非碰巧。

    不在 tiny-SD 上斷言「比低秩 eps 注入多」：那是模型規模的性質而非
    設計的性質。tiny-SD 的 cross-attention 只有 32–64 維、6 個 block，
    算出來反而比 latent 少；真實 SD v1.4 的 16 個 block、768 維 context
    在 r=16 時是 1,591,296，為低秩 eps 注入（163,840）的 9.7 倍。
    此處驗的是逐層公式，該公式在兩個規模上都成立。
    """
    mod = WeightResidual(sd.unet, rank=r, seed=SEED).to(DEV)
    try:
        expected = sum(h.A.numel() + h.B.numel() for h in mod.hooks.values())
        assert mod.num_trainable() == expected, "不得有 A、B 以外的可訓練參數"
        for h in mod.hooks.values():
            assert h.A.shape[0] == r and h.B.shape[1] == r, "秩必須等於設定值"
            assert h.A.numel() + h.B.numel() == r * (h.A.shape[1] + h.B.shape[0])
    finally:
        mod.remove()


def test_site_W_generator不誤判phi進不了計算圖(sd, x01):
    """site W 不提供三種殘差能力中的任何一種，靠 patches_model() 表明。"""
    mod = WeightResidual(sd.unet, rank=4, seed=SEED).to(DEV)
    try:
        assert mod.patches_model() is True
        assert mod.pixel_residual(x01) is None
        assert mod.eps_hook(sd.timesteps(2), 2) is None
        assert mod.emb_residual(sd.encode_text("")) is None

        gen = DefenseGenerator(sd, mod, k_inv=2)
        with torch.no_grad():
            for h in mod.hooks.values():
                h.B.normal_(0, 0.3)
            x_def = gen.generate(x01, gen.prepare(x01))   # 不得拋出
        assert x_def.shape == x01.shape
    finally:
        mod.remove()


# ------------------------------------------------- 編碼器目標與有目標模式


def test_編碼器目標不走去噪鏈且梯度抵達phi(sd, x01):
    """optimize_encoder 完全不呼叫 sdedit：這正是它比 optimize 便宜的原因。

    以計數器包住 sd.sdedit 驗證呼叫次數為零，而不是只看它跑得完——後者
    在 sdedit 被誤加回去時仍然會過。
    """
    calls = {"n": 0}
    orig = sd.sdedit

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    sd.sdedit = counting
    try:
        mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4, seed=SEED).to(DEV)
        cfg = OptimConfig(steps=2, k_inv=2, n_edit=2, lr=0.05, log_every=100)
        res = optimize_encoder(sd, mod, x01, cfg, LossConfig(), default_train_set())
    finally:
        sd.sdedit = orig

    assert calls["n"] == 0, f"編碼器目標不得呼叫 sdedit（實際 {calls['n']} 次）"
    assert len(res.history) == 2
    assert res.x_def is not None
    assert not torch.equal(res.x_def, x01), "φ 必須真的被更新"


def test_編碼器目標的損失朝目標下降(sd, x01):
    """z_target 預設為零：把 x_def 推向「VAE 看不到內容」的退化點。

    兩個設計上的必要條件，缺一則這個測試沒有鑑別力：

    1. `purify_mode="all"`——預設的 rotate 每步輪替不同算子，相鄰步量到的
       是不同條件下的損失，本來就會震盪。E0d 的學習率判準踩過同一個坑。
    2. `lam_fid=0`——保真項的作用正是把 x_def 拉回 x，與編碼器目標直接對抗。
       兩者並存時「損失有沒有下降」量到的是兩股力的淨結果，不是編碼器目標
       本身可不可優化。
    """
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4, seed=SEED).to(DEV)
    cfg = OptimConfig(steps=4, k_inv=2, lr=0.1, log_every=100, purify_mode="all")
    res = optimize_encoder(sd, mod, x01, cfg, LossConfig(lam_fid=0.0),
                           default_train_set())
    l = [h["L_def"] for h in res.history]
    assert l[-1] < l[0], f"編碼器損失未下降：{l[0]:.5f} -> {l[-1]:.5f}"


def test_有目標模式需要y_target(sd, x01):
    """缺少目標時必須在此報錯，不得靜默退回無目標。"""
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4, seed=SEED).to(DEV)
    cfg = OptimConfig(steps=1, k_inv=2, n_edit=2, log_every=100)
    with pytest.raises(ValueError, match="y_target"):
        optimize(sd, mod, x01, cfg, LossConfig(defense_mode="targeted"),
                 default_train_set())


def test_CFG_權重為一時與單分支逐位元相同(sd, x01):
    """既有 53 個 run 的可重現性靠這條。

    E26 為 `sdedit` 加上 classifier-free guidance。w = 1.0 必須精確等於原本的
    單次前向，否則所有既有數字都不再可重現，連「舊結果錯在哪裡」都無從對照。
    此處驗的是逐位元相同，不是近似相同：`_eps_cfg` 在 w == 1.0 時直接
    `return self._eps(...)`，不走 eps_u + 1.0 * (eps_c − eps_u) 的算式——
    後者在 float32 上不保證還原成 eps_c。
    """
    z = sd.encode_image(x01)
    emb = sd.encode_text("a photo")
    t = torch.tensor(200)
    a = sd._eps(z, t, emb)
    b = sd._eps_cfg(z, t, emb, 1.0)
    assert torch.equal(a, b), "w=1.0 必須逐位元等於單分支前向"


def test_CFG_缺少無條件嵌入時拒絕(sd, x01):
    """靜默退回單分支等於把 w 悄悄變回 1，而那正是 E26 找到的缺陷。"""
    z = sd.encode_image(x01)
    emb = sd.encode_text("a photo")
    with pytest.raises(ValueError, match="emb_uncond"):
        sd._eps_cfg(z, torch.tensor(200), emb, 7.5)


def test_CFG_算式正確且改變輸出(sd, x01):
    """ε = ε(∅) + w·[ε(c) − ε(∅)]。以逐項重算驗證，不只驗「有變」。"""
    z = sd.encode_image(x01)
    emb = sd.encode_text("a wrecked car")
    emb_u = sd.encode_text("")
    t = torch.tensor(200)
    w = 7.5

    got = sd._eps_cfg(z, t, emb, w, emb_u)
    eu, ec = sd._eps(z, t, emb_u), sd._eps(z, t, emb)
    assert torch.allclose(got, eu + w * (ec - eu), atol=1e-6)
    assert not torch.allclose(got, ec, atol=1e-4), "w=7.5 必須與單分支不同"


def test_CFG_經sdedit後仍可微(sd, x01):
    """CFG 走兩次 UNet 前向，梯度必須從兩條都回得來——防禦訓練要用它。"""
    lat = sd.latent_shape(SIZE, SIZE)
    noise = sd.sample_edit_noise(torch.empty(lat, device=DEV), seed=SEED)
    emb = sd.encode_text("a wrecked car")
    emb_u = sd.encode_text("")
    x = x01.clone().requires_grad_(True)

    y = sd.sdedit(x, emb, noise, 2, strength=0.5,
                  guidance_scale=7.5, emb_uncond=emb_u)
    y.pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_有目標模式端到端可訓練(sd, x01):
    """這條路徑至今從未被跑過。`runs/` 全部 4882 列 `results.csv` 的
    `defense_mode` 都是 `untargeted`（E25 清點）。而 `objective.py` 自己的
    註解就寫了「無目標最大化在文獻上一貫比有目標脆弱」，並引用本專案實測的
    3.3 倍噪聲過擬合。既然要用它，就必須先有一條釘住「它真的會動」的測試，
    否則第一次在 GPU 上跑才發現不動，代價是整批機時。

    斷言取結構性質而非收斂：tiny-SD 是隨機初始化的，四步之內的收斂行為不能
    代表真實 SD（同 `test_低秩注入使編輯偏移增加` 的理由）。
    """
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4, seed=SEED).to(DEV)
    before = mod.tensor.V.detach().clone()
    cfg = OptimConfig(steps=3, k_inv=2, n_edit=2, lr=0.1, log_every=100)
    y_target = torch.rand(1, 3, SIZE, SIZE, device=DEV)

    res = optimize(sd, mod, x01, cfg, LossConfig(defense_mode="targeted"),
                   default_train_set()[:1], y_target=y_target)

    assert res.history[0]["defense_mode"] == "targeted"
    # 有目標的 L_def 是一個距離而非 hinge，故不會在起點就是 0——那正是它
    # 比無目標穩的原因：損失地形有盆地，不是只有一個「往外走」的方向。
    assert res.history[0]["L_def"] > 0.0
    assert res.history[-1]["grad_norm"] > 0.0, "梯度未抵達 φ"
    assert not torch.equal(before, mod.tensor.V), "φ 未被更新"


def test_階段一還原軌跡最佳的phi而非最後一步(sd, x01):
    """E12 實測 8 個組合中 7 個的最後一步比自己的最佳值差，且劣化幅度隨
    參數量遞增（163,840 參數 +0.0316；1,591,296 參數 +0.0526）。拿最後一步
    會系統性低報每個載體的能力，且低報程度與參數量相關——那正好讓「容量」
    與「優化穩定性」兩個變因無法分離。

    以刻意過大的學習率製造後段發散，再驗證回傳的 φ 對應的是最佳步而非
    最後一步。若模型恰好沒有發散（最佳步就是最後一步），則跳過斷言而非
    誤判失敗。
    """
    cfg = _cfg(align_steps=8, align_lr=5.0)   # 刻意過大，逼出發散
    mod = _latent_module(sd, cfg.k_inv)
    gen = DefenseGenerator(sd, mod, k_inv=cfg.k_inv)
    _, hist = align(sd, mod, x01, cfg, LossConfig(), gen)

    losses = [h["align_loss"] for h in hist]
    best_step = hist[0]["align_best_step"]
    assert best_step == losses.index(min(losses)), "記錄的最佳步必須是損失最小者"

    if best_step == len(losses) - 1:
        pytest.skip("此設定下未發生後段發散，本測試沒有可驗證的還原行為")

    # 還原後重新前向，其損失必須等於最佳步的損失而非最後一步的
    obj = DefenseObjective(LossConfig(gamma_psnr=cfg.align_gamma_psnr), DEV)
    with torch.no_grad():
        x_gen = gen.generate(x01, gen.prepare(x01))
        again, _ = obj.fidelity_term(x_gen, x01, x_base=None)
    assert float(again) == pytest.approx(min(losses), rel=1e-4), (
        f"還原後的損失 {float(again):.4f} 應等於最佳步 {min(losses):.4f}，"
        f"而非最後一步 {losses[-1]:.4f}"
    )


# ------------------------------------------------------------ site S（空間變形）


def test_site_S_零位移時防禦圖與原圖逐位元相等(sd, x01):
    """本位置存在的理由就是這條：φ=0 時沒有重建誤差。

    site L / E / W 都經過 VAE 解碼，φ=0 時已與原圖相差 LPIPS 0.194，保真度
    預算在 φ 起作用前就用光。空間變形停在像素空間，此不變量必須是逐位元的，
    否則本位置相對那三個就沒有優勢可言。

    在 `torch.no_grad()` 下檢查：這正是短路生效的條件，也正是所有評測、
    留存影像與 x_base0 = G(x;0) 的計算條件。建圖時另有下一條測試。
    """
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.0).to(DEV)
    gen = DefenseGenerator(sd, mod, k_inv=2)
    with torch.no_grad():
        x_def = gen.generate(x01, gen.prepare(x01))
    assert torch.equal(x_def, x01), "位移為零時 x_def 必須與 x 完全相同"


def test_site_S_零位移下梯度仍抵達phi(sd, x01):
    """回歸測試：零位移短路曾切斷計算圖，訓練第一步即以
    `element 0 of tensors does not require grad` 失敗（tiny-SD 端對端實測）。

    位移在初始化時恰為零，若此時 φ 進不了計算圖，本位置永遠無法開始訓練。
    """
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.0).to(DEV)
    out = mod.pixel_residual(x01)
    assert out.requires_grad, "建圖模式下輸出必須連著 φ"
    out.pow(2).sum().backward()
    assert mod.flow.grad is not None and mod.flow.grad.abs().sum() > 0
    # 建圖模式下多承受的偏差就是重取樣的數值底線，量級須維持在可忽略範圍
    assert (out - x01).abs().max().item() < 1e-4


def test_site_S_未知的resample模式必須報錯():
    """靜默退回 bilinear 會讓整批實驗跑完才發現設定沒生效。"""
    with pytest.raises(ValueError, match="resample"):
        WarpResidual(size=SIZE, grid_size=8, resample="lanczos")


def test_site_S_預設維持bilinear():
    """E13–E19 的既有數字全部是 bilinear 產生的。改預設會讓它們無法重現，
    故 bicubic 只能是明確選項，不能是新預設。"""
    assert WarpResidual(size=SIZE, grid_size=8).resample == "bilinear"


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_site_S_兩種重取樣都可微且零位移為恆等(mode):
    # 影像必須與模組同裝置。先前 DEV 恰為 CPU 故沒發作；本機裝上 CUDA build
    # 之後 DEV 變成 cuda，grid_sample 立刻以「兩個裝置」報錯。
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.0, resample=mode).to(DEV)
    out = mod.pixel_residual(x01_plain().to(DEV))
    assert out.requires_grad
    out.pow(2).sum().backward()
    assert mod.flow.grad is not None and mod.flow.grad.abs().sum() > 0
    assert (out - x01_plain().to(DEV)).abs().max().item() < 1e-3


def test_site_S_bicubic保留較多高頻():
    """本專案改用 bicubic 的唯一理由，必須有測試釘住。

    E20 §5.2 在 512² 真實影像上量到：同一 LPIPS 下 bilinear 保留 85.0%、
    bicubic 99.9%。此處在小尺寸合成圖上驗同一方向——絕對值不會相同，
    要釘的是「bicubic 的銳利度損失明顯較小」這個順序。
    """
    from src.metrics.acutance import acutance

    x = x01_plain().to(DEV)
    ratios = {}
    for mode in ("bilinear", "bicubic"):
        mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.0,
                           max_disp=50.0, resample=mode).to(DEV)
        # 同一個位移場，只換重取樣，否則比較的不是重取樣本身
        g = torch.Generator().manual_seed(SEED)
        mod.flow.data = torch.randn(mod.flow.shape, generator=g).to(DEV) * 0.5
        with torch.no_grad():
            ratios[mode] = acutance(x, mod.pixel_residual(x))["acutance_ratio"]

    assert abs(ratios["bicubic"] - 1.0) < abs(ratios["bilinear"] - 1.0), (
        f"bicubic 應比 bilinear 更接近原銳利度，實得 {ratios}")


def test_site_S_bicubic輸出仍在值域內():
    """bicubic 會過衝到 [0,1] 之外（bilinear 是凸組合故不會）。不夾回的話
    後續的 LPIPS 與銳利度都在量一張不存在的影像。"""
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.0,
                       max_disp=50.0, resample="bicubic").to(DEV)
    g = torch.Generator().manual_seed(SEED)
    mod.flow.data = torch.randn(mod.flow.shape, generator=g).to(DEV) * 0.8
    with torch.no_grad():
        out = mod.pixel_residual(x01_plain().to(DEV))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_site_S_停用模塊不改變結果(sd, x01):
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.3, seed=SEED).to(DEV)
    mod.disable()
    gen = DefenseGenerator(sd, mod, k_inv=2)
    assert torch.equal(gen.generate(x01, gen.prepare(x01)), x01)


def test_site_S_恆等網格經grid_sample的數值誤差有界(sd, x01):
    """量測（不是假設）零位移短路所迴避的那個數值底線。

    實測 512²、float32 下最大差 5.78e-05。根因是 align_corners=True 的座標
    −1 + 2i/(N−1) 在 float32 無法精確表示（偏離整數像素位置最多 3.05e-05），
    雙線性插值再依比例混入鄰居。此測試把該量級釘住：若某次 torch 升級後
    它變大一個數量級，`force_resample` 對照組的解讀就得跟著改。
    """
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.0,
                       force_resample=True).to(DEV)
    out = mod.pixel_residual(x01)
    err = (out - x01).abs().max().item()
    assert not torch.equal(out, x01), "force_resample 必須真的跑一次重取樣"
    assert err < 1e-4, f"重取樣數值誤差 {err:.3e} 超出預期量級"


def test_site_S_位移非零時結果必須改變且梯度抵達phi(sd, x01):
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.5, seed=SEED).to(DEV)
    x_def = mod.pixel_residual(x01)
    assert not torch.equal(x_def, x01), "位移非零時必須改變影像"

    x_def.pow(2).sum().backward()
    assert mod.flow.grad is not None and mod.flow.grad.abs().sum() > 0, (
        "梯度必須抵達位移場本身"
    )


def test_site_S_位移上界確實不被超過(sd):
    """max_disp 是硬上界而非懲罰項：本位置的保真度預算就是位移量，
    量到超界表示預算宣告與實際不符，整張比較表都會失效。"""
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=5.0,
                       max_disp=1.5, seed=SEED).to(DEV)
    d = mod.displacement(SIZE, SIZE)
    assert d.abs().max().item() <= 1.5 + 1e-6
    st = mod.disp_stats()
    # 逐像素位移長度是兩個分量的平方和開根號，故上界是 sqrt(2)·max_disp
    assert st["disp_max_px"] <= 1.5 * (2 ** 0.5) + 1e-5
    assert st["grid_size"] == 8


def test_site_S_粗網格上採樣後仍是平滑場(sd):
    """平滑性由構造（粗網格 + 雙線性上採樣）保證，不靠懲罰項。

    以相鄰像素的位移差為度量：上採樣後的逐像素變化量，不得超過控制點之間
    的變化量除以放大倍率的合理範圍。這裡直接比較粗網格與細網格兩種設定，
    前者必須明顯更平滑。
    """
    coarse = WarpResidual(size=SIZE, grid_size=4, init_std=0.5, seed=SEED).to(DEV)
    fine = WarpResidual(size=SIZE, grid_size=None, init_std=0.5, seed=SEED).to(DEV)

    def step(m):
        d = m.displacement(SIZE, SIZE)
        return (d[..., 1:, :] - d[..., :-1, :]).abs().mean().item()

    assert step(coarse) < step(fine) / 5, (
        "粗網格上採樣的位移場必須遠比逐像素自由位移平滑"
    )


def test_site_S_是非加性的(sd, x01):
    """與 site P 的區別必須是可量測的，不能只寫在文件裡。

    加性方法滿足 f(x) − x 與 x 無關：同一組 φ 施加在不同影像上，得到的
    像素差值完全相同。空間變形不滿足——差值由影像本身的梯度決定。
    """
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.5, seed=SEED).to(DEV)
    g = torch.Generator().manual_seed(SEED + 1)
    x2 = torch.rand(1, 3, SIZE, SIZE, generator=g).to(DEV)

    with torch.no_grad():
        d1 = mod.pixel_residual(x01) - x01
        d2 = mod.pixel_residual(x2) - x2
    rel = (d1 - d2).abs().mean() / d1.abs().mean().clamp_min(1e-12)
    assert float(rel) > 0.5, (
        f"同一 φ 在兩張影像上的像素差值相對差 {float(rel):.3f} 太小，"
        "行為接近加性——那樣本位置就失去存在意義"
    )


def test_site_S_輸出恆在有效像素範圍內(sd, x01):
    """重新取樣是原像素的凸組合，故不需要 clamp 就必定落在 [0,1]。

    這是空間變形相對加性注入的一個結構性差異：site P 需要 clamp（因而
    殘差的數值秩不等於設定值），本位置不需要。若某天實測不成立，表示
    padding_mode 或插值模式被改過，須立即察覺。
    """
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=1.0,
                       max_disp=3.0, seed=SEED).to(DEV)
    out = mod.pixel_residual(x01)
    assert float(out.min()) >= float(x01.min()) - 1e-6
    assert float(out.max()) <= float(x01.max()) + 1e-6


def test_site_S_不回報像素殘差以免被誤讀為加性(sd):
    mod = WarpResidual(size=SIZE, grid_size=8, init_std=0.5, seed=SEED).to(DEV)
    assert mod.raw_residual() is None
    assert mod.rank_trace() == [8]


# ------------------------------------------------- cross-attention 目標


def test_注意力擷取不改變UNet的前向結果(sd, x01):
    """hook 只讀取輸入、另算一次注意力，UNet 自己的計算路徑不得受影響。

    若改用替換 attention processor 的做法，SDPA 融合核心會換成展開的 QKᵀ，
    數值與記憶體行為都會變，「有沒有開這個目標」就不再是單一變因。
    """
    from src.models.attention import CrossAttentionRecorder

    emb = sd.encode_text("a photo").detach()
    z = sd.encode_image(x01).detach()
    t = torch.tensor(200)

    with torch.no_grad():
        base = sd._eps(z, t, emb)
        rec = CrossAttentionRecorder(sd.unet)
        with rec:
            withhook = sd._eps(z, t, emb)

    assert torch.equal(base, withhook), "hook 改變了 UNet 的前向結果"
    assert len(rec.maps) > 0, "沒有記錄到任何 cross-attention 層"


def test_注意力分佈是合法機率分佈(sd, x01):
    from src.models.attention import CrossAttentionRecorder

    emb = sd.encode_text("a photo").detach()
    z = sd.encode_image(x01).detach()
    rec = CrossAttentionRecorder(sd.unet)
    with torch.no_grad(), rec:
        sd._eps(z, torch.tensor(200), emb)

    for m in rec.maps:
        assert m.min() >= 0
        s = m.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-3)
        assert m.shape[-1] == sd.tokenizer.model_max_length


def test_hook移除後不再累積(sd, x01):
    """context manager 離開後仍留著 hook 的話，後續每一次前向都會偷偷
    多算一次注意力並累積張量——症狀是記憶體緩慢上升且與本目標無關的
    實驗被拖慢，極難追。"""
    from src.models.attention import CrossAttentionRecorder

    emb = sd.encode_text("a photo").detach()
    z = sd.encode_image(x01).detach()
    rec = CrossAttentionRecorder(sd.unet)
    with torch.no_grad():
        with rec:
            sd._eps(z, torch.tensor(200), emb)
        n = len(rec.maps)
        sd._eps(z, torch.tensor(200), emb)
    assert len(rec.maps) == n, "離開 context 後 hook 仍在記錄"


def test_內容token區間排除BOS與EOS(sd):
    from src.models.attention import token_span

    s, e = token_span(sd.tokenizer, "a wrecked car")
    assert s == 1, "起點必須跳過 BOS"
    assert e > s
    ids = sd.tokenizer("a wrecked car", padding=False).input_ids
    assert e == len(ids) - 1, "終點必須排除 EOS"
    # 空 prompt 沒有內容 token，須回傳空區間讓呼叫端明確處理
    s0, e0 = token_span(sd.tokenizer, "")
    assert e0 <= s0


def test_散度對相同分佈為零對不同分佈為正(sd, x01):
    from src.models.attention import CrossAttentionRecorder, attention_divergence

    emb = sd.encode_text("a photo").detach()
    rec = CrossAttentionRecorder(sd.unet)

    def maps_of(z):
        with torch.no_grad(), rec:
            sd._eps(z, torch.tensor(200), emb)
        out = [m.clone() for m in rec.maps]
        rec.clear()
        return out

    z = sd.encode_image(x01).detach()
    a = maps_of(z)
    b = maps_of(z)
    assert float(attention_divergence(a, b)) < 1e-6, "同一組分佈的散度必須為零"

    c = maps_of(z + torch.randn_like(z) * 0.5)
    assert float(attention_divergence(a, c)) > 1e-4, "不同分佈的散度必須為正"


def test_層數不同時報錯而非安靜取前幾層(sd, x01):
    from src.models.attention import CrossAttentionRecorder, attention_divergence

    emb = sd.encode_text("a photo").detach()
    rec = CrossAttentionRecorder(sd.unet)
    with torch.no_grad(), rec:
        sd._eps(sd.encode_image(x01).detach(), torch.tensor(200), emb)
    with pytest.raises(ValueError):
        attention_divergence(rec.maps, rec.maps[:-1])


def test_均勻分佈的熵最大(sd):
    from src.models.attention import attention_entropy

    n = 16
    uniform = [torch.full((1, 4, n), 1.0 / n)]
    peaked = [torch.zeros(1, 4, n)]
    peaked[0][..., 0] = 1.0
    assert float(attention_entropy(uniform)) == pytest.approx(
        float(torch.tensor(float(n)).log()), rel=1e-4)
    assert float(attention_entropy(peaked)) < float(attention_entropy(uniform))


def test_內容質量抑制的值域與方向(sd):
    """suppress 量的是 1 − 內容 token 的注意力質量，故值域 [0,1]、越大綁定越弱。"""
    from src.models.attention import attention_content_suppression

    n, span = 16, (2, 6)          # 4 個內容 token，共 16 格
    uniform = [torch.full((1, 4, n), 1.0 / n)]
    on_content = [torch.zeros(1, 4, n)]
    on_content[0][..., span[0]:span[1]] = 1.0 / (span[1] - span[0])
    off_content = [torch.zeros(1, 4, n)]
    off_content[0][..., 0] = 1.0

    assert float(attention_content_suppression(uniform, span)) == pytest.approx(
        1.0 - (span[1] - span[0]) / n, rel=1e-5)
    assert float(attention_content_suppression(on_content, span)) == pytest.approx(
        0.0, abs=1e-6), "質量全在內容 token 上時抑制為 0"
    assert float(attention_content_suppression(off_content, span)) == pytest.approx(
        1.0, abs=1e-6), "質量完全不在內容 token 上時抑制為 1"


def test_內容質量抑制缺少span時拒絕(sd):
    """落回全域不是「比較粗糙」而是「恆等於 0」，故必須拒絕而非靜默退回。"""
    from src.models.attention import attention_content_suppression

    maps = [torch.full((1, 4, 16), 1.0 / 16)]
    with pytest.raises(ValueError):
        attention_content_suppression(maps, None)
    with pytest.raises(ValueError):
        attention_content_suppression(maps, (3, 3))


def test_suppress模式在phi等於零時梯度非零(sd, x01):
    """這是 `test_divergence模式在phi等於零時無梯度` 的對照。

    divergence 從 φ=0 起不了步，原因是 KL 在 φ=0 恰為最小值。suppress 換成一個
    最佳點不在 φ=0 的量（內容 token 的注意力質量），故起步梯度一般非零。
    兩個測試必須並存：一個釘住缺陷、一個釘住修法確實有效。
    """
    from src.defense.optimize import optimize_crossattn

    cfg = _cfg(steps=2, attn_timesteps=2)
    cfg.unet_ckpt = False
    cfg.attn_mode = "suppress"
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4).to(DEV)
    before = mod.tensor.V.detach().clone()
    res = optimize_crossattn(sd, mod, x01, cfg, LossConfig(),
                             default_train_set()[:1])

    assert res.history[0]["grad_norm"] > 0, "suppress 在 φ=0 必須有梯度"
    assert 0.0 <= res.history[0]["L_def"] * -1.0 <= 1.0, "抑制量的值域為 [0,1]"
    assert not torch.equal(before, mod.tensor.V), "φ 未被更新"


def test_suppress模式在空prompt時提前拒絕(sd, x01):
    """span 為空時 suppress 會退化成常數，必須在跑之前就拒絕。"""
    from src.defense.optimize import optimize_crossattn

    cfg = _cfg(steps=1, attn_timesteps=1)
    cfg.unet_ckpt = False
    cfg.attn_mode = "suppress"
    cfg.prompt_edit = ""
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4).to(DEV)
    with pytest.raises(ValueError, match="suppress"):
        optimize_crossattn(sd, mod, x01, cfg, LossConfig(),
                           default_train_set()[:1])


def test_cross_attention目標的梯度抵達phi(sd, x01):
    """本目標的整條路徑（G → 淨化 → VAE 編碼 → 加噪 → UNet → 注意力）
    都必須可微。特別是 hook 記錄的張量若來自 checkpoint 區塊內部，梯度會
    斷掉而非報錯，故此測試必須存在。

    2026-08-01 改用 entropy 模式。before：不指定 `attn_mode`，即使用
    預設的 `divergence`。after：明確指定 `attn_mode="entropy"`。原因是
    divergence 模式在 φ=0 時 L_def 恆等於 0 且梯度精確為零（見下一個
    測試），此測試先前通過靠的是 `alpha_ssim=1.0` 時 SSIM 在恆等點的浮點
    殘渣 1.22e-10——那不是「梯度抵達 φ」的證據。E20 把 alpha_ssim 改為 0
    之後殘渣消失，測試才暴露出來。要驗的是整條路徑可微，故改用一個
    在 φ=0 有真實梯度的模式。
    """
    from src.defense.optimize import optimize_crossattn

    cfg = _cfg(steps=2, attn_timesteps=2)
    cfg.unet_ckpt = False
    cfg.attn_mode = "entropy"
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4).to(DEV)
    res = optimize_crossattn(sd, mod, x01, cfg, LossConfig(),
                             default_train_set()[:1])

    assert len(res.history) == 2
    assert res.history[-1]["grad_norm"] > 0, "梯度未抵達 φ"
    assert res.x_def is not None


def test_divergence模式在phi等於零時無梯度(sd, x01):
    """釘住一個已知缺陷，不是釘住正確行為。

    `attn_mode="divergence"` 量的是當前注意力圖與未防禦參照的 KL 散度。
    φ=0 時兩者逐元素相同，KL = 0；而 0 是 KL 的最小值，故其梯度也精確為 0。
    最佳化從 φ=0 出發永遠離不開起點——實測兩步的 grad_norm 皆為 0.000e+00，
    L_def 皆為 −0.000e+00。

    這是 `divergence` 的預設模式，且該目標至今一次都沒在 GPU 上跑過
    （見 docs/NEXT_SESSION.md §5）。若直接拿去跑，它會不會產生任何更新。

    此測試存在的目的是讓該缺陷有一個具名的位置：修好之後這個測試會失敗，
    屆時應改成斷言梯度非零，而不是刪掉它。
    """
    from src.defense.optimize import optimize_crossattn

    cfg = _cfg(steps=2, attn_timesteps=2)
    cfg.unet_ckpt = False
    cfg.attn_mode = "divergence"
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4).to(DEV)
    res = optimize_crossattn(sd, mod, x01, cfg, LossConfig(),
                             default_train_set()[:1])

    assert res.history[0]["L_def"] == 0.0, "φ=0 時與參照的散度必為 0"
    assert res.history[0]["grad_norm"] == 0.0, (
        "散度在最小值處的梯度為零，這是 divergence 模式無法從 φ=0 起步的原因")


def test_cross_attention目標下phi確實被改動(sd, x01):
    """同上，改用 entropy 模式：divergence 在 φ=0 沒有梯度，φ 不會被更新。"""
    from src.defense.optimize import optimize_crossattn

    cfg = _cfg(steps=2, attn_timesteps=2)
    cfg.unet_ckpt = False
    cfg.attn_mode = "entropy"
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4).to(DEV)
    before = mod.tensor.V.detach().clone()
    optimize_crossattn(sd, mod, x01, cfg, LossConfig(), default_train_set()[:1])
    assert not torch.equal(before, mod.tensor.V), "φ 未被更新"


def test_未知的attn_mode明確報錯(sd, x01):
    from src.defense.optimize import optimize_crossattn

    cfg = _cfg(steps=1, attn_timesteps=1)
    cfg.unet_ckpt = False
    cfg.attn_mode = "nonsense"
    mod = PixelResidual(size=SIZE, max_rank=4, const_rank=4).to(DEV)
    with pytest.raises(ValueError):
        optimize_crossattn(sd, mod, x01, cfg, LossConfig(),
                           default_train_set()[:1])


def test_注意力擷取與checkpoint不相容須以錯誤呈現(sd, x01):
    """釘住實測到的失敗模式，避免有人「順手」把 use_ckpt 加回這條前向。

    tiny-SD 實測：開 checkpoint 後 backward 以 RuntimeError 中止，因為 hook
    在原前向時掛著、其額外的 to_q/to_k/QKᵀ 進了 checkpoint 區塊的圖，而
    backward 觸發重算時 hook 已卸除，兩次存檔的張量數對不上（477 vs 459）。
    這是硬錯誤而非安靜斷梯度——但仍必須有測試，因為錯誤發生在 backward，
    離肇因很遠。
    """
    from src.models.attention import CrossAttentionRecorder

    emb = sd.encode_text("a photo").detach()
    z = sd.encode_image(x01).detach().requires_grad_(True)
    rec = CrossAttentionRecorder(sd.unet)
    with rec:
        sd._eps(z, torch.tensor(200), emb, use_ckpt=True)
    with pytest.raises(RuntimeError):
        rec.maps[0].sum().backward()


# ------------------------------------------------------- BDIA 精確反演


@pytest.mark.parametrize("k,tmax", [(4, 500), (10, 500), (20, 999)])
def test_BDIA來回誤差遠小於DDIM(sd, x01, k, tmax):
    """本方法唯一的賣點就是這條：反演與去噪互為精確反解。

    tiny-SD 實測（latent 空間最大絕對誤差）：
        k= 4 t_max=500  DDIM 2.53e+00  BDIA 8.06e-06
        k=10 t_max=500  DDIM 1.52e+00  BDIA 1.06e-04
        k=20 t_max=999  DDIM 2.25e+01  BDIA 3.36e-03
    差距為 3~5 個數量級。此處以「BDIA 至少小 100 倍」為門檻，留下模型與
    torch 版本差異的餘裕，同時仍能抓到實作寫錯。
    """
    emb = sd.encode_text("").detach()
    ts = sd.timesteps(k, t_max=tmax)
    with torch.no_grad():
        z0 = sd.encode_image(x01)
        z_ddim, _ = sd.denoise(sd.ddim_inversion(z0, emb, ts, k), emb, ts, k)
        z_bdia, _ = sd.bdia_denoise(sd.bdia_inversion(z0, emb, ts, k),
                                    emb, ts, k)
    e_ddim = (z_ddim - z0).abs().max().item()
    e_bdia = (z_bdia - z0).abs().max().item()
    print(f"\n[BDIA] k={k} t_max={tmax}  DDIM={e_ddim:.3e}  BDIA={e_bdia:.3e}")
    assert e_bdia * 100 < e_ddim


def test_BDIA的gamma為零時明確報錯(sd, x01):
    """γ=0 使上行遞迴退化成無法反解的形式。安靜接受會得到一條看似正常
    但根本不是精確反演的路徑，數字全錯而沒有任何跡象。"""
    emb = sd.encode_text("").detach()
    ts = sd.timesteps(4)
    with torch.no_grad():
        z0 = sd.encode_image(x01)
        with pytest.raises(ValueError):
            sd.bdia_inversion(z0, emb, ts, 4, gamma=0.0)
        pair = sd.bdia_inversion(z0, emb, ts, 4)
        with pytest.raises(ValueError):
            sd.bdia_denoise(pair, emb, ts, 4, gamma=0.0)


def test_generator以exact_inversion切換路徑(sd, x01):
    """開關必須真的改變計算結果，且 context 帶回第二個狀態。"""
    k = 4
    mod = _latent_module(sd, k)
    mod.disable()

    g_ddim = DefenseGenerator(sd, mod, k_inv=k, t_max=500)
    g_bdia = DefenseGenerator(sd, mod, k_inv=k, t_max=500, exact_inversion=True)
    with torch.no_grad():
        c1 = g_ddim.prepare(x01)
        c2 = g_bdia.prepare(x01)
        x1 = g_ddim.generate(x01, c1)
        x2 = g_bdia.generate(x01, c2)

    assert c1.z_prev is None, "DDIM 路徑不應帶回第二個狀態"
    assert c2.z_prev is not None, "BDIA 路徑必須帶回 z_{K-1}"
    assert not torch.equal(x1, x2), "切換反演方式卻得到相同結果"


def test_BDIA下零殘差的重建等於VAE的來回(sd, x01):
    """精確反演的正確結論不是「G(x;0) 更接近原圖」，而是擴散那一段的誤差
    被消掉，只剩 VAE 的來回，即 G(x;0) ≈ decode(encode(x))。

    起初寫的是「BDIA 的 G(x;0) 比 DDIM 的更接近原圖」，實測在 tiny-SD 上
    不成立（DDIM 0.2728 vs BDIA 0.2744）。原因是 tiny-SD 的 VAE 為隨機
    權重，其來回誤差 0.27 完全蓋過 latent 的差異，兩者離原圖的距離都由
    VAE 決定，孰近孰遠是雜訊。那個斷言的前提本身就是錯的，故改成這一條：
    它由構造成立，在任何模型上都可驗證。

    真實 SD 上這條的意義是：重建地板由 LPIPS 0.194 降到 VAE 來回的 0.143，
    不是降到零。像素側加性位置實際運作在 0.063，故此路徑仍未解封。
    """
    k = 4
    mod = _latent_module(sd, k)
    mod.disable()
    with torch.no_grad():
        vae_only = sd.decode_latent(sd.encode_image(x01))
        g1 = DefenseGenerator(sd, mod, k_inv=k, t_max=500)
        g2 = DefenseGenerator(sd, mod, k_inv=k, t_max=500, exact_inversion=True)
        x_ddim = g1.generate(x01, g1.prepare(x01))
        x_bdia = g2.generate(x01, g2.prepare(x01))
    e1 = (x_ddim - vae_only).abs().max().item()
    e2 = (x_bdia - vae_only).abs().max().item()
    # ASCII 輸出：Windows 主控台為 cp950，U+2212 之類的字元會讓 print 本身
    # 丟 UnicodeEncodeError，測試因此失敗而與被測邏輯無關
    print(f"\n[BDIA] max |G(x;0)-VAE roundtrip|: DDIM={e1:.4e}  BDIA={e2:.4e}")
    assert e2 * 100 < e1, "BDIA 的 G(x;0) 應與純 VAE 來回幾乎相同"


def test_BDIA下site_L的注入仍生效且梯度抵達phi(sd, x01):
    """精確反演不得讓注入失效。BDIA 的去噪比 DDIM 少一個注入點
    （K−1 vs K），若索引寫錯，症狀會是「有些步的 φ 從未被用到」。"""
    k = 4
    mod = _latent_module(sd, k)
    gen = DefenseGenerator(sd, mod, k_inv=k, t_max=500, exact_inversion=True)
    ctx = gen.prepare(x01)

    with torch.no_grad():
        mod.disable()
        base = gen.generate(x01, ctx)
        mod.enable()
        mod.tensor.V.normal_(0, 0.5)
        out = gen.generate(x01, ctx)
    assert not torch.allclose(base, out, atol=1e-6), "注入未生效"

    x_def = gen.generate(x01, ctx)
    x_def.pow(2).sum().backward()
    assert mod.tensor.V.grad is not None and mod.tensor.V.grad.abs().sum() > 0
