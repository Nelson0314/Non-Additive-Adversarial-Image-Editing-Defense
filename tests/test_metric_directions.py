"""指標方向驗證（preflight §2.6）— 以人工構造之極端案例確認方向定義。

情境 A「編輯完全成功」（防禦失敗）：edited_prot ≈ edited_orig（僅微小噪聲）。
情境 B「編輯完全失敗」（防禦成功）：edited_prot 與 edited_orig 差異極大。
依 METRIC_HIGHER_IS_BETTER（DAYN 慣例，True=越高防禦越成功），每項指標須滿足：
higher-is-better 指標於情境 B 較高；lower-is-better 指標於情境 A 較高。
FID 以合成特徵驗證方向（免下載 InceptionV3）；CLIP 需外部模型，
方向由定義保證（與惡意 prompt 對齊度，越低防禦越成功），於 TWCC 抽查。
"""

import torch

from src.metrics.quality import METRIC_HIGHER_IS_BETTER, compute_all, compute_fid


def _cases():
    torch.manual_seed(0)
    edited_orig = torch.rand(1, 3, 64, 64)
    similar = (edited_orig + 0.01 * torch.randn_like(edited_orig)).clamp(0, 1)  # 情境 A
    different = (1.0 - edited_orig.roll(shifts=32, dims=-1)).clamp(0, 1)        # 情境 B
    return edited_orig, similar, different


def test_piq_metric_directions():
    edited_orig, similar, different = _cases()
    m_fail = compute_all(similar, edited_orig)      # 防禦失敗
    m_success = compute_all(different, edited_orig)  # 防禦成功
    for k in m_fail:
        higher_better = METRIC_HIGHER_IS_BETTER[k]
        if higher_better:
            assert m_success[k] > m_fail[k], f"{k}: 防禦成功時應較高"
        else:
            assert m_fail[k] > m_success[k], f"{k}: 防禦失敗時應較高"


def test_fid_direction():
    torch.manual_seed(0)
    base = torch.randn(64, 128)
    near = base + 0.01 * torch.randn(64, 128)     # 分布幾乎相同（防禦失敗）
    far = base * 3.0 + 5.0                        # 分布差異大（防禦成功）
    fid_fail = compute_fid(near, base)
    fid_success = compute_fid(far, base)
    assert METRIC_HIGHER_IS_BETTER["fid"] is True
    assert fid_success > fid_fail
