"""等變殘差目標：編輯器對防禦映射 `T` 不等變的程度（`docs/NEXT_PLAN.md` B0）。

    comm(θ; ξ) = LPIPS( N_ξ(T_θ(x)),  T_θ(N_ξ(x)) )
    score(θ)   = − w_comm · comm(θ; ξ) / c0

`N_ξ` 是 `FreeObjective.null_edit`：無文字的 ip2p 短鏈（`chain_steps`、`grad_steps`、
`s_i` 同現行），噪聲由 `seed + draws` 決定。若編輯器對 `T` 等變，`N(T(x)) = T(N(x))`，
防禦圖的色調只是原樣穿過編輯；`comm` 量的是穿過之後剩下的差，對準
`D_T = LPIPS(T̂(edit(x)), edit(x_def))` 這個讀數。

- `N_ξ(x)` 不帶梯度；`T_θ(N_ξ(x))` 只經過 render 對 θ 帶梯度；`N_ξ(T_θ(x))` 經鏈末
  `grad_steps` 步反傳，與現行 `id` 項同一套截斷。
- 不加 `tanh`。`c0` 為起點在固定驗證抽樣上的 `comm`，使該項起點為 1。
- `w_comm` 由呼叫端以起點處的梯度範數對齊現行 `FreeObjective` 後設定（`align_weight`）。
- 恆等映射下 `comm` 逐位元為 0、梯度亦為 0：呼叫端必須用非恆等起點，並以
  `assert_nonzero_grad` 在實際起點上斷言。
- 驗證抽樣固定為 `VAL_DRAWS` 個噪聲（與訓練抽樣不重疊），供 checkpoint 選擇與收斂紀錄。
"""

from __future__ import annotations

import torch

VAL_DRAWS = (90001, 90002)


class CommObjective:
    def __init__(self, free, carrier, x01, lpips, resample=True):
        self.free = free                 # FreeObjective：提供 ip2p、null_edit、噪聲計數
        self.carrier = carrier
        self.x = x01
        self.lpips = lpips               # piq.LPIPS()，凍結
        self.resample = bool(resample)
        self.w_comm = 1.0
        self.c0 = 1.0

    def _comm_at(self, y, draw):
        keep = self.free.draws
        self.free.draws = draw
        try:
            with torch.no_grad():
                nx = self.free.null_edit(self.x).float().clamp(0, 1)
            ny = self.free.null_edit(y).float().clamp(0, 1)
            tn = self.carrier.render(nx).float()
            return self.lpips(ny, tn).mean()
        finally:
            self.free.draws = keep

    def terms(self, y):
        if self.resample:
            self.free.draws += 1
        return {"comm": self._comm_at(y, self.free.draws)}

    def score(self, y):
        return -self.w_comm * self.terms(y)["comm"] / self.c0

    def eval_terms(self, y):
        return {"comm": torch.stack([self._comm_at(y, d) for d in VAL_DRAWS]).mean()}

    def eval_score(self, y):
        return -self.w_comm * self.eval_terms(y)["comm"] / self.c0


def grad_norm(carrier, fn):
    """`fn()` 對載體參數的梯度範數；不改變參數。"""
    for p in carrier.params():
        p.grad = None
    fn().backward()
    g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in carrier.params() if p.grad is not None))
    for p in carrier.params():
        p.grad = None
    return float(g)


def assert_nonzero_grad(carrier, fn):
    for p in carrier.params():
        p.grad = None
    fn().backward()
    for i, p in enumerate(carrier.params()):
        if p.grad is None or float(p.grad.abs().max()) <= 0:
            raise RuntimeError(f"起點處第 {i} 組參數的梯度為零；comm 在恆等附近沒有梯度")
    for p in carrier.params():
        p.grad = None


def align_weight(comm, free, carrier, x01):
    """c0 取固定驗證抽樣上的起點值；w_comm 使 comm 項的起點梯度範數等於 FreeObjective。"""
    with torch.no_grad():
        comm.c0 = float(comm.eval_terms(carrier.render(x01))["comm"])
    if not comm.c0 > 0:
        raise RuntimeError(f"起點 comm = {comm.c0}；非恆等起點仍為 0，不能正規化")
    comm.w_comm = 1.0
    g_comm = grad_norm(carrier, lambda: comm.eval_score(carrier.render(x01)))
    g_free = grad_norm(carrier, lambda: free.eval_score(carrier.render(x01)))
    if not (g_comm > 0 and g_free > 0):
        raise RuntimeError(f"梯度範數 comm={g_comm} free={g_free}，無法對齊")
    comm.w_comm = g_free / g_comm
    return {"comm_c0": comm.c0, "comm_grad_norm_start": g_comm,
            "free_grad_norm_start": g_free, "comm_weight": comm.w_comm}
