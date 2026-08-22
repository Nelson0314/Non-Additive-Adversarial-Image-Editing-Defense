"""擋下率的欄位：`edit_siglip_sim`、`blocked`、`siglip_blocked_threshold`。

存在理由
────────────────────────────────────────────────────────────────────
主讀數是 SigLIP 影像相似度判定的擋下率（低於門檻即擋下），但
`scripts/ip2p_run.py` 寫出的 `results.csv` 裡沒有任何語意欄位——擋下率此前
必須另跑 `scripts/defense_outcome_metrics.py` 二次讀圖才拿得到，而那支依賴
防禦圖還留在磁碟上。影像不入版控，故那條路徑對任何已清理過的批次都失效。

門檻 0.837 此前只是 `defense_outcome_metrics.py` 一句 print 裡的字面量。
本專案的規則是「未載的參數要成為欄位，不是註解」，故它成為具名常數並逐列
寫進 CSV，讓門檻改動之後舊列仍可解讀。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.metrics.standard import SIGLIP_BLOCKED_THRESHOLD, blocked_by_siglip  # noqa: E402


REQUIRED = ("edit_clip_sim", "edit_siglip_sim", "blocked",
            "siglip_blocked_threshold")


def test_block_rate_columns_are_written_literally():
    """欄位名必須以字面字串出現在組列的地方。

    與 `test_ip2p_run_columns.py` 同型：這裡不載入 IP2P（需要 GPU 與權重），
    而漏掉欄位的失效方式正是「那一行不存在」。
    """
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in REQUIRED:
        assert f'"{col}":' in src, col


def test_threshold_is_the_calibrated_value():
    """39 格人眼標記上取最高正確率定出的門檻，正確率 93.5%、AUC 0.974。"""
    assert SIGLIP_BLOCKED_THRESHOLD == 0.837


def test_blocked_is_strictly_below_the_threshold():
    """低於門檻為擋下。等於門檻不算擋下——門檻是在候選值上取到的，
    把等號算進去會讓校準那一格自己翻面。"""
    assert blocked_by_siglip(0.836) is True
    assert blocked_by_siglip(SIGLIP_BLOCKED_THRESHOLD) is False
    assert blocked_by_siglip(0.99) is False


def test_outcome_script_uses_the_shared_constant_not_a_literal():
    """`defense_outcome_metrics.py` 不得再有自己的字面量門檻，
    否則兩支腳本會靜默分岔。"""
    src = (ROOT / "scripts" / "defense_outcome_metrics.py").read_text(
        encoding="utf-8")
    assert "SIGLIP_BLOCKED_THRESHOLD" in src


DCT_REQUIRED = ("dct_q_alg", "dct_gamma")


def test_dct_shield_quality_factor_is_a_column():
    """`q_alg` 決定 DCT-Shield 的量化表，base 用 0.95、Y-only 用 0.85。

    此前它只活在 CLI 預設值裡，於是兩個品質因子跑出的列在報表上長得一模
    一樣——RESULTS 的 FND-058「Y-only 的比較混淆了品質因子」就是這個缺陷的
    症狀。`gamma` 同理：它是論文 §5.4 的步長係數，換了不會有任何欄位改變。
    """
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in DCT_REQUIRED:
        assert f'"{col}":' in src, col
