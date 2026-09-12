"""可微內層：用梯度解顏色場，外層的無導數搜尋只挑結構。

為什麼現在才接
────────────────────────────────────────────────────────────────────
`runs/color_scaleup_search/` 的診斷量到可微代理在**全域仿射**的可達集合上只變動
1–6%，把 latent 推遠 5.8 倍仍然指令完成、身分保住——在那個參數化上梯度不帶訊息。
但判準修好之後（固定框身分、連言軟極小、搜尋與回報種子分離），
`runs/carrier_search_fixed/` 的 720 個候選點上量到 VAE latent 位移與判準的秩相關
是 **−0.63／−0.58**；壞掉的判準上量到的是 +0.04。代理這次帶訊息。

分工也變了。`ColorFieldParam` 在 `grid = 16` 時有上千個場參數，無導數搜尋在
240 次評估內碰不動它們，只能靠一個亂數種子抽整片場。內層用梯度解那些參數，
外層繼續挑控制點密度、帶寬、配色、幅度這些結構性的選擇。

目標
────────────────────────────────────────────────────────────────────
最大化防禦圖與原圖的 VAE latent 距離（PhotoGuard 的 encoder attack），因為它
每步只要一次 VAE 前向，比反傳穿過取樣鏈便宜兩個數量級，而上面的秩相關說它
與判準同向。**這是代理，不是判準**：外層仍然用真編輯評分，內層只負責把場推到
一個代理上好的點。

約束在每一步之後投影，不進損失
────────────────────────────────────────────────────────────────────
色差上限（整圖、臉上）是使用者對產物自然度的要求，不是可以用權重換的一項。
每步之後把幅度整體縮回到上限內——幅度是空間常數，縮放它不改變場的形狀，
所以投影不會把梯度找到的結構破壞掉。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch


def _scale_into_cap(carrier, x01, measure: Callable[[torch.Tensor], float],
                    cap: float, tries: int = 12) -> float:
    """把各段的幅度同時縮到 `measure(render) <= cap`，回傳用掉的縮放係數。

    二分而不是解析解：`measure` 是 CIEDE2000 加上軟裁，對幅度不是線性的。
    幅度已經在上限內時回傳 1.0，不做任何事。
    """
    base = [getattr(s, 'amplitude', 1.0) for s in carrier.stages]
    if measure(carrier.render(x01)) <= cap:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(tries):
        mid = 0.5 * (lo + hi)
        carrier.set_amplitude([b * mid for b in base])
        if measure(carrier.render(x01)) <= cap:
            lo = mid
        else:
            hi = mid
    carrier.set_amplitude([b * lo for b in base])
    return lo


def optimise_field(carrier, x01, ip2p, *, steps: int = 60, lr: float = 0.05,
                   caps: Optional[List[tuple]] = None,
                   log_every: int = 0) -> Dict[str, float]:
    """用 Adam 最大化 `‖E(render(x)) − E(x)‖₂`，每步之後投影回色差上限。

    `caps` 是 [(measure, cap), ...]，`measure` 收一張渲染圖回傳純量。全部都要
    滿足，所以逐條縮放、取最緊的那一條。

    回傳起點與終點的 latent 距離、實際用掉的幅度縮放，以及步數——這些要寫進
    CSV，內層有沒有在動是可查的。
    """
    caps = caps or []
    params = [p for p in carrier.params() if p.requires_grad]
    if not params:
        raise ValueError('這個載體沒有可學參數，內層無事可做')
    with torch.no_grad():
        z0 = ip2p.encode_image(x01).float()

    def latent_gap() -> torch.Tensor:
        z = ip2p.encode_image(carrier.render(x01)).float()
        return (z - z0).flatten().norm()

    with torch.no_grad():
        start = float(latent_gap())
    opt = torch.optim.Adam(params, lr=lr)
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = -latent_gap()
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
            raise FloatingPointError(f'第 {k} 步的梯度不是有限值；不要靜默跳過')
        opt.step()
        carrier.project()
        if log_every and k % log_every == 0:
            with torch.no_grad():
                print(f'    內層 {k} latent {float(latent_gap()):.2f}', flush=True)

    shrink = 1.0
    with torch.no_grad():
        for measure, cap in caps:
            if cap and cap > 0:
                shrink = min(shrink, _scale_into_cap(carrier, x01, measure, cap))
        end = float(latent_gap())
    return {'inner_steps': steps, 'inner_lr': lr,
            'inner_latent_start': round(start, 4),
            'inner_latent_end': round(end, 4),
            'inner_amplitude_shrink': round(shrink, 5)}
