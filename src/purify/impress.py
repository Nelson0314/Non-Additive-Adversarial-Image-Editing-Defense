"""IMPRESS 淨化算子（Cao et al., NeurIPS 2023，官方 repo `AAAAAAsuka/Impress`）。

出處與參數查證見 `docs/_audit_purify.md` §2。摘要：

- 損失：`MSE(D(E(x_pur)), x_pur) + α · max(LPIPS(x_pur, x_ptb) − L, 0)`
  （論文 Eq. 4.1；程式 `impress.py` 第 13–17 行）。
- 官方 `impress.py` 有三組「預設」（函式簽名／Glaze 情境／PhotoGuard 情境）。
  本專案的威脅模型是編輯而非風格微調，依 `SOURCE_AUDIT_2026-08-05.md` §8 取
  **PhotoGuard 情境**那一組：eps(L) 0.1、iters 1000、lr 0.005、α 0.01、σ 0.05、
  值域 `[-1, 1]`。該組同時見於 README、`scripts/new/pg_mask_diff_test.sh`
  與論文附錄 B，三者一致。
- `model` 就是 SD 的 VAE（`glaze_pur.py` 第 52–60 行），且呼叫前已凍結參數。
  故本模組要求傳入 `SDWrapper`，缺席時明確拋出。

**與原始碼的差異（逐項列出，不得隱藏）**：

1. 原始碼全程 `.half()`（fp16）。本專案的 `SDWrapper` 預設 fp32，本模組跟隨傳入
   VAE 的 dtype，不強制轉型。Adam 的 `eps=1e-5`（非 torch 預設 1e-8）照抄原碼保留。
2. 原始碼固定 batch 為 1（`preprocess` 產生 `(1,3,H,W)`）。本模組對 batch 逐張
   獨立執行，使每張圖的 LPIPS hinge 與原碼語意相同（不跨圖平均）。
3. 原始碼的初始化雜訊無種子。本模組接受 `seed`，使 `proxy_gap` 可量測、
   評測可重現。`seed=None` 時行為與原碼相同。
4. LPIPS 後端：原碼用 `lpips.LPIPS(net='vgg')`。本模組預設即該後端；若環境未安裝
   `lpips` 套件，**不會靜默改用其他實作**，而是拋出並說明。`backend="piq"` 是
   我方替代（本專案他處使用的 `piq.LPIPS`，同為 VGG16 骨幹但非同一份實作），
   採用時須在報表標註。

**可微性**：IMPRESS 本身是對 `x_pur` 的梯度下降，但作為淨化算子放進我們的訓練
迴圈時，梯度需穿過 1000 步 Adam 更新。官方實作以 `optimizer.step()` 與
`X_p.data = ...` 就地更新葉張量，該路徑對輸入不保留 autograd 圖；要展開必須改寫
其優化迴圈，屬重新設計別人的方法，本專案不做。故訓練側走直通估計
（`straight_through`），`differentiable = False`，此限制須在報告中載明。
"""

from __future__ import annotations

import torch
import torch.nn as nn

# PhotoGuard 情境的預設值（`_audit_purify.md` §2.4 (C) 欄）
PHOTOGUARD_PRESET = {
    "eps": 0.1,      # 論文的 L：LPIPS margin
    "iters": 1000,
    "lr": 0.005,
    "alpha": 0.01,   # 論文的 α，程式的 pur_alpha
    "noise": 0.05,   # 論文的 σ，程式的 pur_noise
}
CLAMP_MIN = -1.0
CLAMP_MAX = 1.0
ADAM_EPS = 1e-5              # 原碼 `torch.optim.Adam([X_p], lr=lr, eps=1e-5)`
SCHEDULER_ETA_MIN = 1e-5     # 原碼 `CosineAnnealingLR(optimizer, iters, eta_min=1e-5)`


def has_impress_deps(sd, backend: str = "lpips") -> bool:
    """VAE 與 LPIPS 後端是否都到位。`Purifier.available` 用它。

    兩者都要檢查。只檢查 `sd` 的話，缺 `lpips` 套件會在**跑到那一格時**
    才由 `_make_lpips` 拋出——而 IMPRESS 那 285 格排在數小時機時之後，
    且連續失敗會在第 10 格中止整段。`has_diffpure_weights` 檢查檢查點與
    `guided_diffusion` 兩者是同一個理由。

    2026-08-06 新增。before：`ops.Purifier.available` 對 `impress` 只回傳
    `self.options.get("sd") is not None`，漏掉了預設後端所需的 `lpips` 套件。
    """
    if sd is None:
        return False
    if backend == "lpips":
        from importlib.util import find_spec

        return find_spec("lpips") is not None
    return True


def _make_lpips(backend: str, device, dtype):
    """建立 LPIPS 函式，輸入為 `[-1, 1]` 的張量（與原碼一致）。"""
    if backend == "lpips":
        try:
            import lpips as lpips_pkg
        except ImportError as e:
            raise NotImplementedError(
                "IMPRESS 原始碼使用 `lpips.LPIPS(net='vgg')`，但環境未安裝 `lpips` 套件。"
                "請 `pip install lpips`（首次執行會下載 VGG16 與 LPIPS 線性層權重）。"
                "若確定要改用本專案既有的 `piq.LPIPS`，須顯式指定 backend='piq'，"
                "並在報表標註「LPIPS 後端為我方替代」。"
            ) from e
        net = lpips_pkg.LPIPS(net="vgg").to(device=device, dtype=dtype)
        for p in net.parameters():
            p.requires_grad_(False)
        return lambda a, b: net(a, b)
    if backend == "piq":
        import piq

        net = piq.LPIPS().to(device=device, dtype=dtype)
        # piq.LPIPS 的輸入約定為 [0,1]，原碼傳的是 [-1,1]，故此處換算
        return lambda a, b: net(a / 2 + 0.5, b / 2 + 0.5)
    raise ValueError(f"未知的 LPIPS 後端 {backend!r}（可用：'lpips'、'piq'）")


def _resolve_vae(sd):
    """從 SDWrapper 取出 VAE。缺席或型別不符時明確拋出。"""
    if sd is None:
        raise ValueError(
            "IMPRESS 需要 SD 的 VAE：請以 Purifier('impress', sd=<SDWrapper>) 傳入。"
            "官方實作的 `model` 引數即 `pipe.vae`（glaze_pur.py 第 53 行）。"
        )
    vae = getattr(sd, "vae", None)
    if vae is None:
        raise ValueError(
            f"傳入的物件 {type(sd).__name__} 沒有 `.vae`，無法作為 SDWrapper 使用。"
        )
    return vae


def impress_real(
    x01: torch.Tensor,
    sd,
    *,
    eps: float = PHOTOGUARD_PRESET["eps"],
    iters: int = PHOTOGUARD_PRESET["iters"],
    lr: float = PHOTOGUARD_PRESET["lr"],
    alpha: float = PHOTOGUARD_PRESET["alpha"],
    noise: float = PHOTOGUARD_PRESET["noise"],
    seed: int = None,
    backend: str = "lpips",
) -> torch.Tensor:
    """真實 IMPRESS。輸入輸出皆為 `(B,3,H,W)`、RGB、`[0,1]`。

    內部在 `[-1,1]` 上運作（原碼的 `clamp_min=-1, clamp_max=1`），回傳前以
    `(x/2 + 0.5).clamp(0,1)` 還原，與 `glaze_pur.py` 第 60 行相同。
    """
    vae = _resolve_vae(sd)
    param = next(vae.parameters())
    device, dtype = param.device, param.dtype
    lpips_fn = _make_lpips(backend, device, dtype)
    criterion = nn.MSELoss()

    x = x01.detach().to(device=device, dtype=dtype).clamp(0, 1)
    x_adv_all = 2.0 * x - 1.0
    g = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))

    out = []
    # `evaluate` 帶 no_grad，而 IMPRESS 本身就是梯度下降，故必須顯式開啟
    with torch.enable_grad():
        for i in range(x_adv_all.shape[0]):
            X_adv = x_adv_all[i : i + 1]
            init = torch.randn(X_adv.shape, generator=g, device=device, dtype=dtype)
            X_p = (X_adv.clone().detach() + init * noise).requires_grad_(True)
            optimizer = torch.optim.Adam([X_p], lr=lr, eps=ADAM_EPS)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, iters, eta_min=SCHEDULER_ETA_MIN
            )
            for _ in range(iters):
                _X_p = vae(X_p).sample
                optimizer.zero_grad()
                lnorm_loss = criterion(_X_p, X_p)
                d = lpips_fn(X_p, X_adv)
                lpips_loss = max(d - eps, 0)  # 原碼即用 Python 內建 max（hinge）
                loss = lnorm_loss + alpha * lpips_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                X_p.data = torch.clamp(X_p, min=CLAMP_MIN, max=CLAMP_MAX)
            out.append((X_p.detach() / 2 + 0.5).clamp(0, 1))
    return torch.cat(out, dim=0).to(x01.device, x01.dtype)
