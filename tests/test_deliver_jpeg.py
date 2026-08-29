"""`--deliver-jpeg`：把 JPEG 量化放進迴圈**並且交付壓縮後的圖**。

與已否決的 `--purify-aware` 的差別只有一個，那個差別是這一支存在的全部理由：
**交付什麼**。

| | 最佳化時看到 JPEG | 交付什麼 |
|---|---|---|
| `--purify-aware fixed75`（已否決） | ✓ | 未壓縮的圖 |
| DCT-Shield（§6.3 抗 JPEG） | ✓ | 壓縮的圖 |
| `--deliver-jpeg QD` | ✓ | 壓縮的圖 |

交付壓縮的圖等於把輸出約束在 QD 的量化格點上，攻擊方以同品質或更高品質重壓
時近似恆等。`--purify-aware` 那三個變體把 JPEG 放進迴圈了卻交付連續值的圖，
擾動一離開迴圈就不在格點上，於是白做。

本檔釘住三件事：關閉時逐位元等於加旗標之前、交付的圖確實落在格點上、
迴圈看到的與交付的是同一個量化。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ip2p_run  # noqa: E402
from src.baselines.jpeg_codec import (  # noqa: E402
    jpeg_roundtrip, jpeg_roundtrip_ste,
)
from src.defense.param_pgd import run_param_pgd  # noqa: E402
from phase_ablation import build  # noqa: E402


def _x(n: int = 128) -> torch.Tensor:
    """平滑的彩色測試圖。**不用均勻雜訊**：雜訊在 JPEG 下大量觸底／觸頂，
    夾到 [0,1] 會把值推離量化格點，量到的「格點鎖定」會比真實影像悲觀得多。"""
    g = torch.Generator().manual_seed(0)
    base = torch.rand(1, 3, n // 16, n // 16, generator=g)
    return (0.15 + 0.7 * torch.nn.functional.interpolate(
        base, size=(n, n), mode="bilinear", align_corners=False)).clamp(0, 1)


def _gray(n: int = 128) -> torch.Tensor:
    """灰階版。**4:2:0 的色度次取樣對常數色度是恆等的**，所以這一張上
    「交付的圖重壓一次不變」是可以逐位元成立的等式，而不是一個比值。
    彩色圖上色度的 下取樣→上取樣 本身就不是恆等，那是 JPEG 的性質、
    與交付這一步無關，故拿灰階圖釘等式、拿彩色圖釘量級。"""
    g = torch.Generator().manual_seed(0)
    base = torch.rand(1, 1, n // 16, n // 16, generator=g)
    y = torch.nn.functional.interpolate(
        base, size=(n, n), mode="bilinear", align_corners=False)
    return (0.15 + 0.7 * y).expand(1, 3, n, n).contiguous()


def _loss(z: torch.Tensor) -> torch.Tensor:
    return z.pow(2).sum()


def _args(*extra: str):
    return ip2p_run.build_parser().parse_args(
        ["--out", "x", "--radius", "0.03", "--steps", "4", *extra])


# ---------------------------------------------------------------------------
# 一、關閉時逐位元等於加這個旗標之前
# ---------------------------------------------------------------------------

def test_預設是關閉的():
    assert ip2p_run.build_parser().parse_args(["--out", "x"]).deliver_jpeg == 0.0


def test_關閉時前向變換維持_None():
    """加旗標之前，`--purify-aware none` 就是傳 `transform=None` 給 PGD。"""
    assert ip2p_run.deliver_quality(_args()) is None
    assert ip2p_run._forward_transform(_args()) is None


def test_關閉時防禦圖逐位元等於加旗標之前的路徑():
    """對照組直接呼叫 `run_param_pgd(transform=None)`，那正是加旗標之前
    `defend()` 相位路徑做的事。"""
    x = _x()
    args = _args()
    got, radius, unreachable, modified, extras = ip2p_run.defend(
        None, None, "add", x, args, _loss)

    param, _, _ = build("add", args.seed, block=args.block, r_min=args.r_min,
                        hop=args.hop, r_max=args.r_max, quantile=args.quantile,
                        gl_iters=args.gl_iters,
                        pixel_gate_sigma=args.pixel_gate_sigma,
                        gain_ratio=args.gain_ratio,
                        gate_edge_power=args.gate_edge_power,
                        freq_weight=args.freq_weight,
                        freq_weight_power=args.freq_weight_power,
                        gain_weight=args.gain_weight,
                        channels=args.phase_channels,
                        spectral_floor=args.spectral_floor,
                        floor_gate=args.floor_gate,
                        theta_budget=args.theta_budget,
                        warp_grid=args.warp_grid)
    param.set_radius(args.radius)
    want = run_param_pgd(x, param, _loss, steps=args.steps, seed=args.seed,
                         transform=None)

    assert torch.equal(got, want.x_def)
    # 收斂欄位（停止原因／實走步數／最佳評估）一律寫出來，關著 --eval-every
    # 時 best_eval 是空字串。**沒有這三欄就分不出「跑滿」與「早停」。**
    assert extras == {"resumed": 0, "stop_reason": want.stop_reason,
                      "stopped_at": want.stopped_at, "best_eval": ""}
    assert (radius, unreachable, modified) == (want.radius, False, False)


# ---------------------------------------------------------------------------
# 二、交付的圖落在量化格點上
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", (95, 85, 75, 50))
def test_交付的圖重壓一次逐位元不變_未交付的會變(q):
    """這一格就是整個實驗的假說：交付壓縮的圖 = 把輸出釘在 QD 的量化格點上，
    攻擊方以同品質重壓時是**恆等**。灰階圖上這是等式而不是比值（見 `_gray`）。"""
    x = _gray()
    delivered, *_ = ip2p_run.defend(
        None, None, "add", x, _args("--deliver-jpeg", str(q)), _loss)
    plain, *_ = ip2p_run.defend(None, None, "add", x, _args(), _loss)

    assert float((jpeg_roundtrip(delivered, q) - delivered).abs().max()) == 0.0
    # 對照：沒有交付這一步時，同一個品質的重壓確實會動到圖。
    assert float((jpeg_roundtrip(plain, q) - plain).pow(2).mean().sqrt()) > 1e-4


@pytest.mark.parametrize("q", (85, 75))
def test_存成八位元之後仍然落在格點上(q):
    """防禦圖是以 PNG 存下來再由抗淨化那一輪讀回去的，中間過一次 uint8。
    八位元的四捨五入本身就會推離格點一點點，判準因此取「殘留量在八位元的
    量化尺度以內」（1.5/255），而不是零。"""
    x = _x()
    delivered, *_ = ip2p_run.defend(
        None, None, "add", x, _args("--deliver-jpeg", str(q)), _loss)
    plain, *_ = ip2p_run.defend(None, None, "add", x, _args(), _loss)

    def regrid(t):
        u = torch.round(t.clamp(0, 1) * 255) / 255
        return float((jpeg_roundtrip(u, q) - u).pow(2).mean().sqrt())

    assert regrid(delivered) < 1.5 / 255
    assert regrid(delivered) * 2 < regrid(plain)


# ---------------------------------------------------------------------------
# 三、迴圈看到的與交付的是同一個量化
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", (95, 85, 75, 50))
def test_迴圈前向的_STE_值逐位元等於交付用的真實往返(q):
    """兩者分岔的話，最佳化找到的位置與交付出去的圖就是兩件事。"""
    x = _x()
    transform = ip2p_run._forward_transform(_args("--deliver-jpeg", str(q)))
    assert torch.equal(transform(x, 0).detach(), jpeg_roundtrip(x, q))
    assert torch.equal(jpeg_roundtrip_ste(x, q).detach(), jpeg_roundtrip(x, q))


def test_小數與整數的品質寫法指同一張量化表():
    """`--q-alg` 用論文式的 0.85，這一支必須跟著同一條換算路徑。"""
    assert ip2p_run.deliver_quality(_args("--deliver-jpeg", "0.85")) == 85
    assert ip2p_run.deliver_quality(_args("--deliver-jpeg", "85")) == 85


def test_前向的順序是先自壓再讓攻擊方淨化():
    x = _x()
    args = _args("--deliver-jpeg", "75", "--purify-aware", "fixed75")
    got = ip2p_run._forward_transform(args)(x, 0).detach()
    assert torch.equal(got, jpeg_roundtrip_ste(jpeg_roundtrip_ste(x, 75), 75))


def test_STE_的梯度通得過交付這一步():
    """交付若切斷梯度，迴圈就退化成沒有 JPEG 的普通 PGD。"""
    x = _x().requires_grad_(True)
    ip2p_run._forward_transform(_args("--deliver-jpeg", "75"))(x, 0).pow(2) \
        .sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# 四、欄位與守門
# ---------------------------------------------------------------------------

def test_保留率與品質都逐列寫進_CSV():
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert '"deliver_jpeg": args.deliver_jpeg,' in src
    assert "**extras," in src

    x = _x()
    _, _, _, _, extras = ip2p_run.defend(
        None, None, "add", x, _args("--deliver-jpeg", "75"), _loss)
    assert set(extras) == {"resumed", "stop_reason", "stopped_at", "best_eval",
                           "deliver_retention", "deliver_cosine",
                           "deliver_retention_base", "deliver_rms_raw",
                           "deliver_rms_out"}
    # 兩個基準（拿原圖／拿壓過的原圖當底）在真實影像上差得極小，不會因為選錯
    # 一個而讀出不同的結論。
    assert abs(extras["deliver_retention"]
               - extras["deliver_retention_base"]) < 0.05


def test_套到別人的方法上直接拒絕():
    """交付自壓是接在本方法的參數化後面的一步。DCT-Shield 自己就把 δ 加在
    量化係數上，再套一層等於把它的方法改掉一半。"""
    x = _x()
    for cond in ("dct_shield", "advdrop", "dct_wm"):
        with pytest.raises(SystemExit, match="deliver-jpeg"):
            ip2p_run.defend(None, None, cond, x,
                            _args("--deliver-jpeg", "75"), _loss)


def test_關閉時別的條件照樣跑得過守門():
    """守門只在旗標開著時作用，否則既有的 baseline 批次會全部啟動失敗。
    `advdrop_max` 不載入模型，正好用來驗守門那一行沒有誤擋。"""
    x = _x()
    out, radius, unreachable, modified, extras = ip2p_run.defend(
        None, None, "advdrop_max", x, _args(), _loss)
    assert out.shape == x.shape and extras == {} and modified is True
