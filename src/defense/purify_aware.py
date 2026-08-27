"""針對淨化最佳化：把可微分的淨化算子放進防禦的前向路徑。

為什麼要有這個
────────────────────────────────────────────────────────────────────
本專案至今的抗淨化都是**事後量測**——先把防禦做好，再看它過了淨化還剩多少。
擾動從來沒有「知道」自己會被壓縮過。

FND-061 給了一個很說明問題的對照：DCT-Shield 在 JPEG 下大勝紋理重相位
（淨增益 +0.519 對 +0.135），而它並沒有做任何 min-max 最佳化——它只是把
擾動**長在 JPEG 的量化格點上**。也就是「與淨化算子對齊」這件事本身就足以
換到抗性。反過來，同一個方法在高斯模糊下淨增益只剩 +0.010，因為模糊不吃
那套格點。

所以最直接的強化路徑是：**把淨化算子放進最佳化迴圈，讓梯度自己去找活得下來
的位置。** 作法取自 MetaCloak-JPEG（arXiv:2604.18537）：把可微分 JPEG 放進
前向，並讓品質因子沿著一條 curriculum 由高走低。

與 min-max 的差別
────────────────────────────────────────────────────────────────────
這**不是** min-max。攻擊方的淨化算子在此是固定的、已知的、可微的，我們只是
在它的複合函數上做一般的 PGD。真正的 min-max 需要對淨化參數也做內層最佳化，
成本高一個數量級，且 2026-08-13 已否決過（見 CLAUDE.md 的否決清單）。

課程排程為什麼由高品質走向低品質
────────────────────────────────────────────────────────────────────
高品質的 JPEG 量化步長細，幾乎所有頻率都留得住，梯度看到的地形接近沒有淨化
的情形；低品質步長粗，只有落在保留頻帶上的擾動才活得下來。先易後難讓擾動
先找到有效的方向、再被逼進活得下來的頻帶。直接從低品質起步時，早期的梯度
幾乎全被量化吃掉，最佳化沒有方向可循。

**排程的端點是本專案指定的**：MetaCloak-JPEG 的摘要只說「由 95 降到 50」，
未載明衰減形狀。此處取線性，並把它寫成參數。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

from src.baselines.jpeg_codec import jpeg_roundtrip_ste
from src.purify.ops import (
    CROP_ANTIALIAS, CROP_FRACTION_DIA, CROP_INTERPOLATION, crop_resize,
    gaussian_blur,
)

# MetaCloak-JPEG 摘要載明的端點；衰減形狀為本專案指定（線性）
CURRICULUM_Q_HI = 95
CURRICULUM_Q_LO = 50


def jpeg_quality_at(step: int, steps: int, *, q_hi: int = CURRICULUM_Q_HI,
                    q_lo: int = CURRICULUM_Q_LO) -> int:
    """第 `step` 步該用的 JPEG 品質。線性由 `q_hi` 降到 `q_lo`。

    端點都取得到：`step = 0` 給 `q_hi`，`step = steps - 1` 給 `q_lo`。
    `steps = 1` 時退化為 `q_hi`（沒有可衰減的區間）。
    """
    if steps <= 0:
        raise ValueError(f"steps 必須為正，收到 {steps}")
    if not 0 <= step < steps:
        raise ValueError(f"step={step} 超出 [0, {steps})")
    if q_lo > q_hi:
        raise ValueError(f"q_lo={q_lo} 高於 q_hi={q_hi}，課程應由高走低")
    if steps == 1:
        return q_hi
    frac = step / (steps - 1)
    return int(round(q_hi + (q_lo - q_hi) * frac))


def make_jpeg_transform(steps: int, *, q_hi: int = CURRICULUM_Q_HI,
                        q_lo: int = CURRICULUM_Q_LO
                        ) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """回傳 `transform(x01, step)`，把影像過一次該步品質的可微分 JPEG 往返。

    前向值逐位元等於真實的 JPEG 往返（`jpeg_codec.jpeg_roundtrip`），只有
    `round()` 的反向被當成恆等，故**量到的失真與最終存檔的一致**。
    """

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        return jpeg_roundtrip_ste(x01, jpeg_quality_at(step, steps,
                                                       q_hi=q_hi, q_lo=q_lo))

    return transform


def make_fixed_jpeg_transform(quality: int
                              ) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """固定品質的版本，用來把「課程」與「有沒有放 JPEG」兩個變因分開。"""

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        return jpeg_roundtrip_ste(x01, quality)

    return transform


def make_eot_jpeg_transform(qualities: Sequence[int], seed: int = 0
                            ) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """每一步從 `qualities` 裡**隨機抽一個**品質。

    這是 expectation-over-transformation 的最省事版本：不對每步的多個品質求
    平均（那樣每步要多算好幾次 VAE），而是讓抽樣在步與步之間攤平。**與
    MetaCloak-JPEG 的課程排程是兩種不同的做法**，不要混在同一列報表上。
    """
    if not qualities:
        raise ValueError("qualities 不可為空")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        i = int(torch.randint(len(qualities), (1,), generator=gen))
        return jpeg_roundtrip_ste(x01, qualities[i])

    return transform


def make_eot_ops_transform(qualities: Sequence[int] = (75,),
                           blur_sigma: float = 1.0,
                           crop_frac: float = CROP_FRACTION_DIA,
                           seed: int = 0,
                           ) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """**多算子**的 expectation-over-transformation：每步隨機抽一個淨化算子。

    為什麼要它：2026-08-21 的縮小版量到紋理重相位在 JPEG 上淨增益 +0.1704
    （勝 DCT-Shield 2.77 倍、逐圖 5/5），但在高斯模糊上打平、在裁切縮放上輸
    （+0.0360 對 +0.1083、0/5）。「贏一個算子、輸兩個」在文獻上的標準解就是
    把整組算子放進期望值裡。

    **算子直接取自 `src/purify/ops.py`，不另寫一份。** `gaussian_blur` 與
    `crop_resize` 本來就原生可微（`ops._DIFFERENTIABLE` 列著），複製一份只會
    多一個「最佳化的對象和評測的對象悄悄不同」的錯誤來源。JPEG 不可微，走
    `jpeg_roundtrip_ste` 的直通估計。

    抽樣而非每步求平均：求平均要每步多算三次完整前向（VAE 編碼是主要成本），
    抽樣把成本攤到步與步之間。**這與 `make_eot_jpeg_transform` 只抽品質不同，
    也與 MetaCloak-JPEG 的課程排程不同，三者不可混在同一列報表上。**

    `identity` 一定在候選裡——沒有它，最佳化會為了抗淨化而放棄未淨化時的
    效果，那不是我們要的取捨。
    """
    if not qualities:
        raise ValueError("qualities 不可為空")
    ops: list = [
        lambda z: z,
        lambda z: gaussian_blur(z, blur_sigma),
        lambda z: crop_resize(z, crop_frac),
    ]
    ops += [(lambda q: (lambda z: jpeg_roundtrip_ste(z, q)))(q) for q in qualities]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        i = int(torch.randint(len(ops), (1,), generator=gen))
        return ops[i](x01)

    return transform


# 隨機化幾何 EOT 的預設族。比例含 0（即 identity），使最佳化不必為了抗淨化
# 而放棄未淨化時的效果；評測用的 0.10 落在族內，這是 EOT 成立的前提。
# **這四個值與「偏移隨機而非置中」都是本專案指定的**，沒有出處。
GEOMETRY_FRACTIONS = (0.0, 0.05, 0.10, 0.15)


def random_crop_resize(x01: torch.Tensor, frac: float,
                       gen: torch.Generator) -> torch.Tensor:
    """隨機位置的裁切再縮放回原尺寸。插值設定與 `ops.crop_resize` 同一組常數。

    與 `ops.crop_resize` 的差別只有**裁切窗的位置**：那一支是中心裁切，這裡
    在合法範圍內均勻抽。位置對輸出不可微（抽的是整數索引），但那個梯度本來
    就不需要——要學的是擾動，不是裁切窗。
    """
    if x01.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W) 張量，收到 {tuple(x01.shape)}")
    h, w = x01.shape[-2:]
    dh, dw = int(round(h * frac)), int(round(w * frac))
    if dh == 0 and dw == 0:
        return x01
    if 2 * dh >= h or 2 * dw >= w:
        raise ValueError(f"裁切比例 {frac} 對 {h}×{w} 過大")
    top = int(torch.randint(2 * dh + 1, (1,), generator=gen))
    left = int(torch.randint(2 * dw + 1, (1,), generator=gen))
    cropped = x01[..., top:top + h - 2 * dh, left:left + w - 2 * dw]
    return torch.nn.functional.interpolate(
        cropped, size=(h, w), mode=CROP_INTERPOLATION,
        antialias=CROP_ANTIALIAS).clamp(0, 1)


def make_eot_geometry_transform(fractions: Sequence[float] = GEOMETRY_FRACTIONS,
                                seed: int = 0, jitter: bool = False,
                                ) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """對**一族**裁切與縮放取期望值，而不是對一個固定的幾何。

    為什麼這一支和 `make_eot_ops_transform` 不一樣
    ────────────────────────────────────────────────────────────────
    後者裡本來就有 `crop_resize`，但放進去的是**一個固定的幾何**（中心裁切、
    比例恆為 0.10）。固定的變換可以被 co-adapt——最佳化只要學會那一個特定的
    位移就行，不會產生對一族裁切的不變性。

    量測支持這個區分（`runs/ip2p_residual_signature/band_transfer.csv`）：
    裁切縮放留下 51–99% 的殘差能量，對**原網格**的餘弦是 0.000，但對**算子
    自己搬過的同一擾動**是 0.995–0.996。擾動幾乎原封不動地通過了，只是被搬到
    別的位置與尺度上。要對付的因此是**對位**，不是能量，也不是頻帶——沒有任何
    一帶的方向存活率高於 0.02。

    `0.0` 留在族內即 identity，理由與 `make_eot_ops_transform` 相同：沒有它，
    最佳化會為了抗淨化而放棄未淨化時的效果。

    `jitter` 決定裁切窗置中還是隨機
    ────────────────────────────────────────────────────────────────
    **預設 False（置中），因為評測算子是置中的。** `ops.crop_resize` 的
    `CROP_MODE = "center"`，而中心裁切再縮放的幾何效果是**以中心為不動點的
    純放大**，淨平移為零——實測亮點 64 → 16、448 → 496、而 **256 不動**
    （`runs/ip2p_residual_signature/crop_decomposition.csv`）。

    所以這一族要撐開的是**放大倍率**，那由 `fractions` 提供；隨機化裁切偏移
    等於再要求一個評測不會考的平移不變性，是把 EOT 的容量花在 nuisance 參數
    上。`jitter=True` 保留給「攻擊方不置中裁切」那個更強的威脅模型，**它是
    嚴格更難的目標，兩者不可混在同一列報表上**。
    """
    if not fractions:
        raise ValueError("fractions 不可為空")
    for f in fractions:
        if not 0.0 <= f < 0.5:
            raise ValueError(f"裁切比例必須落在 [0, 0.5)，收到 {f}")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        i = int(torch.randint(len(fractions), (1,), generator=gen))
        frac = float(fractions[i])
        if jitter:
            return random_crop_resize(x01, frac, gen)
        return crop_resize(x01, frac) if frac > 0 else x01

    return transform


def describe(transform: Optional[Callable]) -> str:
    """報表用的一行說明。`None` 代表沒有針對淨化最佳化。"""
    return "none" if transform is None else getattr(
        transform, "__qualname__", "custom").split(".")[0]


# ---------------------------------------------------------------------------
# 分階段訓練用的「依序輪替」算子序列
# ---------------------------------------------------------------------------

# 名稱 → (算子, 強度階)。**強度階只用於 `ramp`，不影響 `shuffle`／`cycle`。**
# 算子一律取自 `src/purify/ops.py`，與評測用的是同一份程式——訓練時打的靶
# 與評測時考的題若各寫一份，會悄悄變成兩件事。
STAGE2_OPS: Dict[str, tuple] = {
    "identity": (lambda z: z, 0),
    "blur1": (lambda z: gaussian_blur(z, 1.0), 1),
    "blur2": (lambda z: gaussian_blur(z, 2.0), 2),
    "crop05": (lambda z: crop_resize(z, 0.05), 1),
    "crop10": (lambda z: crop_resize(z, 0.10), 1),
    "crop15": (lambda z: crop_resize(z, 0.15), 2),
}

STAGE2_ORDERS = ("shuffle", "cycle", "random")


def resolve_stage2_ops(names: Sequence[str]) -> List[str]:
    """檢查名稱合法並保持給定順序。**未知的名字直接拋錯，不靜默略過**。"""
    if not names:
        raise ValueError("算子清單不可為空")
    unknown = [n for n in names if n not in STAGE2_OPS]
    if unknown:
        raise ValueError(
            f"未知的淨化算子 {unknown}，可用的是 {sorted(STAGE2_OPS)}")
    return list(names)


def stage2_schedule(names: Sequence[str], steps: int, *, order: str = "shuffle",
                    seed: int = 0, ramp: bool = False) -> List[str]:
    """逐步要用哪一個算子，回傳長度為 `steps` 的名稱串。

    **先把整條排程算出來再跑**，理由是它要能被測試逐項檢查，也要能寫進 log
    ——「第幾步餵了什麼」若只存在於迴圈裡，事後就無法重建。

    三種順序
    ────────────────────────────────────────────────────────────────
    | `random` | 每一步獨立抽一個。這是 `make_eot_ops_transform` 的作法 |
    | `cycle` | 固定順序輪替，每一輪每個算子各一次 |
    | `shuffle` | 每一輪把清單洗牌再依序走完（**預設**） |

    `random` 的問題是短期內分布不均：只跑幾百步時，某個算子連著被抽到好幾次
    而另一個一次都沒抽到，會直接把方向偏掉。`cycle` 修掉了覆蓋不均，但每一輪
    的最後一個算子恆是「最後說話的那個」，配上 sign 更新（只看梯度方向、不看
    大小）容易走成週期性的來回。`shuffle` 保留均勻覆蓋、拿掉固定的相位關係。

    `ramp` 由弱到強
    ────────────────────────────────────────────────────────────────
    開啟時前半段只用強度階 ≤ 1 的算子，後半段用全部。動機是重模糊之下梯度
    幾乎沒有訊號（`runs/phase_drift_diagnosis`：σ=2 時殘差能量只剩 3.3%），
    先給輕的讓它有方向可循。**預設關閉**，關閉時與只有 `order` 的版本逐項相同。
    """
    if steps < 0:
        raise ValueError(f"steps 不可為負，收到 {steps}")
    if order not in STAGE2_ORDERS:
        raise ValueError(f"未知的順序 {order!r}，可用的是 {STAGE2_ORDERS}")
    pool_all = resolve_stage2_ops(names)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def pool_at(step: int) -> List[str]:
        if not ramp or steps == 0:
            return pool_all
        if step * 2 >= steps:
            return pool_all
        weak = [n for n in pool_all if STAGE2_OPS[n][1] <= 1]
        # 全部都是重算子時不能把池子清空，退回全池並照實記在 log。
        return weak if weak else pool_all

    out: List[str] = []
    bag: List[str] = []
    bag_pool: List[str] = []
    for i in range(steps):
        pool = pool_at(i)
        if order == "random":
            out.append(pool[int(torch.randint(len(pool), (1,), generator=gen))])
            continue
        if not bag or bag_pool != pool:
            bag_pool = list(pool)
            if order == "shuffle":
                perm = torch.randperm(len(pool), generator=gen).tolist()
                bag = [pool[j] for j in perm]
            else:
                bag = list(pool)
        out.append(bag.pop(0))
    return out


def make_sequenced_ops_transform(names: Sequence[str], steps: int, *,
                                 order: str = "shuffle", seed: int = 0,
                                 ramp: bool = False):
    """把 `stage2_schedule` 包成 `transform(x01, step)`。

    `step` 是**階段二內部的步序**（由 0 起算），不是整體迭代數——階段一不套
    任何算子，兩者的步序若混用會讓排程整段錯位。
    """
    plan = stage2_schedule(names, steps, order=order, seed=seed, ramp=ramp)

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        if not 0 <= step < len(plan):
            raise ValueError(f"step={step} 超出階段二的 [0, {len(plan)})")
        return STAGE2_OPS[plan[step]][0](x01)

    transform.plan = plan          # 讓派工端能把排程寫進 log
    return transform
