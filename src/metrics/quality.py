"""指標計算 — SPEC.md §2.6、STRUCTURE.md §2.5。

PSNR/SSIM/VIFp/FSIM/LPIPS 一律用 piq，不可自行實作、不可換套件
（APA 用 IQA-PyTorch，不同套件，勿混用）。FID、CLIP score 為補充。

衡量的是兩個「編輯結果」之間的差異，差異越大代表保護越成功。
"""

# 方向定義：True 表示「數值越高、防禦越成功」
# 採 DAYN 慣例（SPEC §2.6）。PhotoGuard 用相反慣例，勿混淆。
METRIC_HIGHER_IS_BETTER = {
    "psnr":  False,
    "ssim":  False,
    "vifp":  False,
    "fsim":  False,
    "lpips": True,
    "fid":   True,
    "clip":  False,
}
