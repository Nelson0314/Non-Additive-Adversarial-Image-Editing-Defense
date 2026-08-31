"""stAdv（Xiao et al., ICLR 2018, arXiv:1801.02612）的流場正則項 `L_flow`。

移植的範圍
────────────────────────────────────────────────────────────────────
該篇的方法由三個零件組成，本模組只放**第三個**：

1. 逐像素稠密流場 `f_i = (Δu^(i), Δv^(i))`，共 `2·H·W` 個變數。
   在本專案裡由 `WarpParam(grid=H)` 提供——粗網格邊長等於影像邊長時
   雙三次上採樣退化成恆等（`F.interpolate` 同尺寸時逐位元不動，由
   `tests/test_stadv_flow.py` 釘住），此時係數本身就是稠密場。
2. 雙線性四鄰居取樣。`WarpParam._sample` 用的是
   `grid_sample(mode="bilinear", padding_mode="border", align_corners=False)`，
   取樣式與原文的
   `x_adv^(i) = Σ_{q∈N(u,v)} x^(q)·(1−|u^(i)−u^(q)|)·(1−|v^(i)−v^(q)|)`
   相同（`N` 是左上／右上／左下／右下四格）。
3. **流場正則 `L_flow`，即本模組。** 原文第 3.2 節的原貌：

       L_flow(f) = Σ_p Σ_{q∈N(p)} sqrt( ‖Δu^(p) − Δu^(q)‖₂²
                                        + ‖Δv^(p) − Δv^(q)‖₂² )

   **根號在鄰居和的裡面。** 寫成 `sqrt(Σ_q ...)`（根號提到外面）不會拋錯，
   也不會有任何症狀，只是變成另一個正則項——那一版是逐像素的鄰域梯度模長，
   對「單一鄰居的大跳變」的懲罰比原式輕。兩者的數值差由
   `tests/test_stadv_flow.py` 用一個具體的場釘住。

原文的總目標是 `argmin_f L_adv + τ·L_flow`，`τ = 0.05`（ImageNet 上由
0.0005–0.05 網格搜尋得到）。**本專案的 `L_adv` 不是原文的那一個**：原文是
分類器 logits 上的 Carlini–Wagner 式 `max(max_{i≠t} g(x)_i − g(x)_t, κ)`、
`κ = 0`，本專案用的是擴散模型上的 encoder／latent／image-guidance 損失
（`--loss`）。因此 τ 的取值不可沿用原文，它是 CSV 的欄位 `flow_tau`。

與原文的差異（modified_from_paper）
────────────────────────────────────────────────────────────────────
**一、根號內的 `eps`（原文的式子裡沒有這一項）**

    原貌  sqrt( ‖Δu^(p) − Δu^(q)‖₂² + ‖Δv^(p) − Δv^(q)‖₂² )
    改法  sqrt( ‖Δu^(p) − Δu^(q)‖₂² + ‖Δv^(p) − Δv^(q)‖₂² + eps )
    理由  `f ≡ 0` 時每一項都是 `sqrt(0)`。`sqrt` 在 0 的導數是 +∞，鏈式
          規則再乘上「平方和對 `f` 的導數為 0」，得到 `inf × 0 = NaN`，
          整個梯度全毀（本機驗證：`torch.sqrt((x*x).sum())` 在 `x = 0` 處
          `x.grad` 是 `nan`）。而 `WarpParam.reset` 在 `init_std = 0` 時的
          起點**正是全零**，也就是說不加這一項，流場臂第一步就拿到 NaN。
          公開的重建實作同樣是在根號內加一個小常數繞開。
    代價  每一項因此有一個下限 `sqrt(eps)`，總和多出一個常數
          `鄰居對數 × sqrt(eps)`。常數的梯度是零、不影響最佳化方向，但
          **報表上的 `L_flow` 讀數含這個偏移**，跨 `eps` 比較數值時要扣掉。
    參數  `eps` 是**必填的關鍵字參數，沒有預設值**（按 CLAUDE.md：查不到的
          參數設為必填或標 `modified_from_paper`，不要填一個看起來合理的
          預設）。它是 CSV 的欄位 `flow_eps`。

**二、鄰域 `N(p)` 的定義（原文未載明）**

    原文只寫 `q ∈ N(p)`，沒有寫 `N` 是四鄰域、八鄰域，還是只取右與下以免
    同一對被算兩次。三種選法會讓 `L_flow` 差一個倍率（四鄰域恰為「右與下」
    的兩倍）與一個方向偏好（八鄰域把對角線也算進去），因此**由本專案指定**
    並列為 CSV 的欄位 `flow_neighbourhood`：

        "right_down"  N(p) = {右, 下}。每一對相鄰像素只算一次。
        "four"        N(p) = {上, 下, 左, 右}。每一對算兩次，恰為前者的 2 倍。
        "eight"       四鄰域再加四條對角線。

    `flow_tv_loss` 的 `neighbourhood` 同樣是**必填**，沒有預設值。

**三、邊界**

    影像外的鄰居**不計入** `N(p)`，不做 padding。padding 會憑空造出一圈
    位移差為零的項，那些項在加了 `eps` 之後仍然貢獻 `sqrt(eps)`，等於把
    邊界長度混進正則項的常數裡。

**四、歸約方式**

    照原文取**和**（`Σ_p Σ_q`）不取平均。τ 的量級與這個選擇綁在一起。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

# 鄰域 `N(p)` 的三種選法，每一項是 (dy, dx) 的位移集合。原文未載明用哪一種，
# 故三種都提供、由呼叫端明給，見模組 docstring 的差異二。
NEIGHBOURHOODS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "right_down": ((0, 1), (1, 0)),
    "four": ((0, 1), (0, -1), (1, 0), (-1, 0)),
    "eight": ((0, 1), (0, -1), (1, 0), (-1, 0),
              (1, 1), (1, -1), (-1, 1), (-1, -1)),
}


def _aligned_pair(field: torch.Tensor, dy: int, dx: int):
    """回傳 `(f[p], f[q])`，`q = p + (dy, dx)`，只取兩者都落在影像內的位置。"""
    h, w = field.shape[-2:]
    ys_p = slice(max(0, -dy), h - max(0, dy))
    ys_q = slice(max(0, dy), h - max(0, -dy))
    xs_p = slice(max(0, -dx), w - max(0, dx))
    xs_q = slice(max(0, dx), w - max(0, -dx))
    return field[..., ys_p, xs_p], field[..., ys_q, xs_q]


def flow_tv_loss(field: torch.Tensor, *, eps: float,
                 neighbourhood: str) -> torch.Tensor:
    """stAdv 的 `L_flow`。`field` 是 (B, 2, H, W)：通道 0 = Δu、通道 1 = Δv。

        L_flow(f) = Σ_p Σ_{q∈N(p)} sqrt( ‖Δu^(p) − Δu^(q)‖₂²
                                        + ‖Δv^(p) − Δv^(q)‖₂² + eps )

    **根號在鄰居和的裡面**，`eps` 也在根號裡面。兩個關鍵字參數都是必填，
    理由與代價見模組 docstring 的「與原文的差異」。回傳一個純量張量，
    梯度回得到 `field`。
    """
    if field.dim() != 4 or field.shape[1] != 2:
        raise ValueError(
            f"flow 必須是 (B, 2, H, W) 的位移場，收到 {tuple(field.shape)}")
    if neighbourhood not in NEIGHBOURHOODS:
        raise ValueError(
            f"未知的鄰域 {neighbourhood!r}，可用的是 "
            f"{'／'.join(sorted(NEIGHBOURHOODS))}")
    if not eps > 0:
        raise ValueError(
            f"eps 必須為正，收到 {eps}。它的存在理由是 f ≡ 0 時 sqrt(0) 的"
            "梯度是 NaN；填 0 等於把那個 NaN 放回來")

    total = field.new_zeros(())
    for dy, dx in NEIGHBOURHOODS[neighbourhood]:
        a, b = _aligned_pair(field, dy, dx)
        d = a - b
        # `sum(dim=1)` 併掉的是 Δu 與 Δv 兩個通道，也就是式子裡的兩個
        # ‖·‖₂²；根號套在這個和上，再對所有 (p, q) 求和。
        total = total + torch.sqrt(d.pow(2).sum(dim=1) + eps).sum()
    return total


def flow_pair_count(height: int, width: int, neighbourhood: str) -> int:
    """`L_flow` 裡有幾項。乘上 `sqrt(eps)` 就是 `eps` 帶進來的常數偏移。"""
    if neighbourhood not in NEIGHBOURHOODS:
        raise ValueError(
            f"未知的鄰域 {neighbourhood!r}，可用的是 "
            f"{'／'.join(sorted(NEIGHBOURHOODS))}")
    n = 0
    for dy, dx in NEIGHBOURHOODS[neighbourhood]:
        n += max(0, height - abs(dy)) * max(0, width - abs(dx))
    return n
