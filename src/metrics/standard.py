"""使用者 2026-08-19 定案的統一指標清單，以及它的兩個半邊。

    LPIPS  FID  SSIM  PSNR  VIFp  HEval  CLIP  SigLIP  DISTS

**同一組指標量兩次，方向相反**，這是這份清單唯一需要小心的地方：

| 半邊 | 量什麼 | 方向 | 欄位前綴 |
|---|---|---|---|
| 失真（noise perception） | `def` 對 `orig` | LPIPS↓ FID↓ SSIM↑ PSNR↑ VIFp↑ | `fid_` |
| 防禦效果（edit protection） | `edit(def)` 對 `edit(orig)` | LPIPS↑ FID↑ SSIM↓ PSNR↓ VIFp↓ | `edit_` |

這正是 DCT-Shield（arXiv:2504.17894）Table 1 的兩個半邊，照這份清單報，
本專案的數字就與該表逐欄對得上。對齊協定見
`docs/reference/BASELINE_ALIGNMENT.md`。

**`fid_` 這個前綴是「fidelity」不是「Fréchet Inception Distance」。** 它早於
FID 進入本專案（2026-07 起的全部 run 都用它），改名會讓既有 CSV 與全部分析
腳本斷掉，故不動。新加入的 FID 一律用 `frechet` 當欄名，兩者不會撞。

HEval 是人眼判定，由 `compare.html` 產生，不在本模組。CLIP／SigLIP 由
`MetricSuite.semantic_multi` 產生，欄名沿用既有的 `edit_clip_*`／
`edit_siglip_*`，本模組不重複定義。
"""

from typing import Dict

# 報表的欄序，與論文 Table 1 相同。`frechet` 即 FID（欄名理由見模組 docstring）。
STANDARD_METRICS = ("lpips", "frechet", "ssim", "psnr", "vif_p", "dists")

# `MetricSuite.pairwise` 實際回傳的鍵。FID 不在其中——它是分布指標，
# 由 `MetricSuite.fid` 另算，故本表比 `STANDARD_PAIRWISE` 少一項。
PAIRWISE_KEYS = ("lpips", "ssim", "psnr", "vif_p", "dists")

# 小數位數。PSNR 是 dB、量級 20–50，其餘落在 [0,1]。
ROUNDING = {"psnr": 3, "frechet": 3}
DEFAULT_ROUNDING = 5

# 方向表。True = 該半邊裡「越高代表越好」。
# 失真半邊的「好」是失真小；防禦半邊的「好」是位移大。
FIDELITY_HIGHER_IS_BETTER = {
    "lpips": False, "frechet": False, "ssim": True, "psnr": True,
    "vif_p": True, "dists": False,
}
PROTECTION_HIGHER_IS_BETTER = {
    "lpips": True, "frechet": True, "ssim": False, "psnr": False,
    "vif_p": False, "dists": True,
}


def standard_row(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    """把 `MetricSuite.pairwise` 的輸出轉成帶前綴的標準欄位。

    只取 `PAIRWISE_KEYS` 的五項，**缺任何一項就拋錯**——這份清單是定案的，
    悄悄少一欄會讓該批次無法與其他批次並排，而那正是統一清單要解決的問題。
    `pairwise` 回傳的其他欄位（`linf`／`rms`／`fsim`／`acutance_ratio`）
    不在此列，呼叫端要留就自己加。
    """
    missing = [k for k in PAIRWISE_KEYS if k not in metrics]
    if missing:
        raise KeyError(f"標準指標缺欄位：{missing}（有的是 {sorted(metrics)}）")
    return {f"{prefix}{k}": round(float(metrics[k]),
                                  ROUNDING.get(k, DEFAULT_ROUNDING))
            for k in PAIRWISE_KEYS}
