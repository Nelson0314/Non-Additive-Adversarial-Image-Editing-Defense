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
from src.models.sd import SDWrapper
from src.residual.site_latent import LatentResidual
from src.residual.site_pixel import PixelResidual

TINY = "hf-internal-testing/tiny-stable-diffusion-pipe"
SIZE = 64
SEED = 20260728


@pytest.fixture(scope="module")
def sd():
    return SDWrapper(TINY)


@pytest.fixture(scope="module")
def x01():
    g = torch.Generator().manual_seed(SEED)
    return torch.rand(1, 3, SIZE, SIZE, generator=g)


def _latent(sd):
    return sd.latent_shape(SIZE, SIZE)


# ------------------------------------------------------------------ T1-P


def test_T1_P_殘差為零時防禦圖與原圖恆等(sd, x01):
    """site P：Δ=0 ⟹ x_def = x 逐元素相等，無任何重建誤差。"""
    lat = _latent(sd)
    mod = PixelResidual(size=SIZE, max_rank=8, const_rank=8)
    gen = DefenseGenerator(sd, mod, k_inv=2)
    ctx = gen.prepare(x01)
    x_def = gen.generate(x01, ctx)
    assert torch.equal(x_def, x01), "V 初始為零，x_def 必須與 x 完全相同"
    assert lat[1] >= 1


def test_T1_P_編輯結果在殘差為零時完全相同(sd, x01):
    lat = _latent(sd)
    emb = sd.encode_text("a photo").detach()
    noise = sd.sample_edit_noise(torch.empty(lat), seed=SEED)

    mod = PixelResidual(size=SIZE, max_rank=8, const_rank=8)
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

    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4)
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

    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4)
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

    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4)
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
    mod = LatentResidual(steps=3, channels=lat[1], size=n, max_rank=4, const_rank=r)
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
    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=2)
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
    a = sd.sample_edit_noise(torch.empty(lat), seed=SEED)
    b = sd.sample_edit_noise(torch.empty(lat), seed=SEED)
    assert torch.equal(a, b)

    c = sd.sample_edit_noise(torch.empty(lat), seed=SEED + 1)
    assert not torch.equal(a, c), "不同 seed 必須給出不同噪聲"


def test_T3_相同噪聲下編輯管線為決定性(sd, x01):
    lat = _latent(sd)
    emb = sd.encode_text("a photo").detach()
    noise = sd.sample_edit_noise(torch.empty(lat), seed=SEED)
    with torch.no_grad():
        y1 = sd.sdedit(x01, emb, noise, num_steps=3)
        y2 = sd.sdedit(x01, emb, noise, num_steps=3)
    assert torch.equal(y1, y2)


def test_T3_不同噪聲下編輯結果不同(sd, x01):
    """確認噪聲確實影響輸出——否則 T3 的共用檢查沒有意義。"""
    lat = _latent(sd)
    emb = sd.encode_text("a photo").detach()
    n1 = sd.sample_edit_noise(torch.empty(lat), seed=SEED)
    n2 = sd.sample_edit_noise(torch.empty(lat), seed=SEED + 99)
    with torch.no_grad():
        y1 = sd.sdedit(x01, emb, n1, num_steps=3)
        y2 = sd.sdedit(x01, emb, n2, num_steps=3)
    assert not torch.allclose(y1, y2, atol=1e-6)


# ------------------------------------------------------------ 梯度與快取


def test_梯度抵達phi且SD保持凍結(sd, x01):
    """spec §5.3：φ 是唯一可訓練參數，凍結參數不阻斷梯度。"""
    k, n = 2, 2
    lat = _latent(sd)
    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4)
    gen = DefenseGenerator(sd, mod, k_inv=k)
    emb = sd.encode_text("a photo").detach()
    noise = sd.sample_edit_noise(torch.empty(lat), seed=SEED)

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
    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4)
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
    mod = LatentResidual(steps=k, channels=lat[1], size=lat[-1], max_rank=4, const_rank=2)
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
    ).rank_trace(ts, 4)
    li = LatentResidual(
        steps=4, channels=lat[1], size=lat[-1], max_rank=4, schedule="LI"
    ).rank_trace(ts, 4)

    assert len(ld) == len(li) == 4
    assert ld == sorted(ld), f"LD 沿去噪步序應遞增：{ld}"
    assert li == sorted(li, reverse=True), f"LI 沿去噪步序應遞減：{li}"
    # 去噪走訪的是 ts[steps..1]，**t=0 那一格永遠不會被評估**（最後一步以
    # ts[1] 的 ε 更新至 ts[0]）。故排程只碰得到 t=t_max 這一端的端點值，
    # 碰不到 t=0 端。斷言只能對 t=t_max 端下。
    assert ld[0] == 0, f"LD 在 t=t_max 應為 0：{ld}"
    assert li[0] == 4, f"LI 在 t=t_max 應為 max_rank：{li}"


def test_const排程每步相同(sd):
    lat = _latent(sd)
    ts = sd.timesteps(4)
    trace = LatentResidual(
        steps=4, channels=lat[1], size=lat[-1], max_rank=4,
        schedule="const", const_rank=3,
    ).rank_trace(ts, 4)
    assert trace == [3, 3, 3, 3]
