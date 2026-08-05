"""四軸格點：驗「實際會跑哪些格子」與設計文件相符。

格點定義錯誤是事後看不出來的——少跑一個條件或漏掉半個淨化掃描，
產出的表格看起來仍然完整，只是少了幾列。故它必須有測試。
"""

import pytest

from src.experiment import grid

IMGS = ("pie_0001", "pie_0002", "pie_0003")


# ---------------------------------------------------------------------------
# 軸的定義
# ---------------------------------------------------------------------------

def test_九個訓練條件():
    """三個非加性 + 五篇 baseline + 一條隨機對照。

    DiffVax 不在內：它的免疫器吃 masked image、只支援 inpainting，
    在無 mask 的 SDEdit 下忠實重現結構上不可能（SOURCE_AUDIT §9 第 1 項）。
    """
    assert len(grid.NONADDITIVE) == 3
    assert "diffvax" not in grid.CONDITIONS
    assert grid.RANDOM_CONTROL in grid.CONDITIONS


def test_隨機對照不是選配():
    """先驗實測：同一可辨失真上，隨機高斯雜訊即取得最佳化解 60–74% 的
    語意失效。沒有這條對照，任何正結果都不可解讀。"""
    assert grid.RANDOM_CONTROL in grid.CONDITIONS


def test_主表在tau零點二():
    """τ = 0.10 低於 N3 的 VAE 重建下限，主表設在那裡會只剩位移場一組
    非加性方法（LOGIC_CHECK C1，使用者 2026-08-05 定案）。"""
    assert grid.MAIN_TAU == 0.20
    assert grid.MAIN_TAU in grid.TAUS


def test_兩個完整淨化點都在N3可達的區間內():
    """曲線兩端必須可比。若其中一端 N3 不存在，那一端的比較只有位移場。"""
    for t in grid.FULL_PURIFY_TAUS:
        assert not grid.generative_floor_skip("N3", t), f"τ={t} 對 N3 不適用"


def test_訓練只在最大tau上做一次():
    """逐 τ 重訓是 ×4 成本，且「匹配失真」的前提已四次被證偽。"""
    assert grid.TRAIN_TAU == max(grid.TAUS)


# ---------------------------------------------------------------------------
# N3 的不適用格
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau,expect_skip", [
    (0.05, True), (0.10, True), (0.20, False), (0.35, False),
])
def test_N3在低tau上不適用(tau, expect_skip):
    """走生成路徑的條件在 VAE 重建下限（LPIPS 0.1434）以下結構上不可能達成。"""
    assert bool(grid.generative_floor_skip("N3", tau)) is expect_skip


def test_不適用的理由字串只有一種說法():
    """報表與測試必須取到同一個字串，否則同一件事會出現兩種說法。"""
    r = grid.generative_floor_skip("N3", 0.05)
    assert str(grid.GENERATIVE_LPIPS_FLOOR) in r and "0.05" in r


def test_像素路徑的條件不受下限影響():
    for c in ("N1", "N2", "mist", "R"):
        assert grid.generative_floor_skip(c, 0.05) == ""


def test_不適用的格仍被列出而非消失():
    """列出並標記 skipped，與 failed 分開。整段消失的話，
    報表看不出「這裡本來應該有東西」。"""
    cells = grid.rayscale_cells(IMGS, conditions=("N3",), taus=(0.05, 0.20))
    assert len(cells) == 2 * len(IMGS)
    assert sum(1 for c in cells if c.skipped) == len(IMGS)


# ---------------------------------------------------------------------------
# 淨化的分配
# ---------------------------------------------------------------------------

def test_完整淨化只在兩個tau上():
    assert len(grid.purifiers_for(0.05)) == 1
    assert len(grid.purifiers_for(0.10)) == 1
    assert len(grid.purifiers_for(0.20)) > 1
    assert len(grid.purifiers_for(0.35)) > 1


def test_掃描組只在主表的tau上():
    """掃描組用於機制分析（確認強模糊與強量化下的反轉是否重現），
    不需要在每個 τ 上跑。"""
    main = grid.purifiers_for(grid.MAIN_TAU)
    other = grid.purifiers_for(0.35)
    assert set(grid.SWEEP_PURIFIERS) <= set(main)
    assert not (set(grid.SWEEP_PURIFIERS) & set(other))


def test_每個完整點都含identity():
    """identity 是未淨化的基準，抗淨化的絕對效果要對它相減。"""
    for t in grid.TAUS:
        assert grid.IDENTITY in grid.purifiers_for(t), t


def test_主組是文獻共識的六個算子加identity():
    kinds = {k for k, _ in grid.MAIN_PURIFIERS}
    assert kinds == {"identity", "jpeg", "crop_resize", "adverse_cleaner",
                     "cnn_denoise_substitute", "impress", "diffpure"}


def test_CNN去噪的命名帶substitute():
    """NTIRE 2023 冠軍的程式碼與權重皆未公開，此為我方替代。
    命名不帶 substitute 會讓報表看起來像重現了 DiffVax 的該項評測。"""
    kinds = {k for k, _ in grid.MAIN_PURIFIERS}
    assert "cnn_denoise_substitute" in kinds
    assert "ntire2023" not in kinds


# ---------------------------------------------------------------------------
# 種子與樣本
# ---------------------------------------------------------------------------

def test_種子數低於二即拋出():
    """n=1 時樣本標準差恆為 0，`|mean| > sd` 對任何負值自動成立——
    先驗實驗曾因此產生 24 格假陽性。"""
    with pytest.raises(ValueError, match="假陽性"):
        grid.eval_cells(IMGS, n_seeds=1)


def test_預設五個種子():
    assert grid.N_SEEDS == 5


# ---------------------------------------------------------------------------
# φ=0 對照的共用
# ---------------------------------------------------------------------------

def test_對照不依賴條件():
    """它只依賴 (影像, 淨化, 種子)。每個條件各算一次就是 9 倍重複計算。"""
    cells = grid.control_cells(IMGS)
    keys = [(c.image_id, c.purify, c.seed) for c in cells]
    assert len(keys) == len(set(keys)), "對照有重複"
    assert {c.condition for c in cells} == {"phi0"}


def test_對照涵蓋全部會用到的淨化設定():
    """漏掉任何一個，該格的絕對效果就沒有基準可減。"""
    needed = {(img, p, s)
              for img in IMGS
              for t in grid.TAUS
              for p in grid.purifiers_for(t)
              for s in range(grid.N_SEEDS)}
    have = {(c.image_id, c.purify, c.seed) for c in grid.control_cells(IMGS)}
    assert needed == have


def test_對照的數量遠少於評測():
    ev = len(grid.eval_cells(IMGS))
    ct = len(grid.control_cells(IMGS))
    assert ct * 5 < ev, f"共用未生效：對照 {ct}、評測 {ev}"


# ---------------------------------------------------------------------------
# 規模與可擴充性
# ---------------------------------------------------------------------------

def test_格點數與設計相符():
    p = grid.plan(IMGS)
    s = grid.summarize(p)
    n_cond, n_img = len(grid.CONDITIONS), len(IMGS)

    assert s["train"]["total"] == n_cond * n_img
    assert s["rayscale"]["total"] == n_cond * n_img * len(grid.TAUS)
    # N3 在兩個低 τ 上不適用
    assert s["rayscale"]["skipped"] == n_img * 2
    assert s["eval"]["total"] == sum(
        len(grid.purifiers_for(t)) for t in grid.TAUS
    ) * n_cond * n_img * grid.N_SEEDS


def test_N由三擴到一百五十只需改影像清單():
    """設計的硬性約束：不得有任何寫死樣本數的地方。"""
    small = grid.summarize(grid.plan(IMGS))
    big = grid.summarize(grid.plan([f"img{i}" for i in range(150)]))
    assert big["train"]["total"] == small["train"]["total"] // 3 * 150


def test_cell_id可讀且不含雜湊():
    c = grid.Cell("eval", "N1", "pie_0007", tau=0.20, purify=("jpeg", 30), seed=3)
    cid = c.cell_id()
    assert "eval" in cid and "N1" in cid and "pie_0007" in cid
    assert "jpeg30" in cid and "seed3" in cid and "tau0.2" in cid


def test_identity的cell_id不帶強度():
    """`identity0` 會讓人以為有一個強度為 0 的參數可調。"""
    c = grid.Cell("eval", "N1", "img", tau=0.2, purify=grid.IDENTITY, seed=0)
    assert "purifyidentity" in c.cell_id()


def test_cell不可變():
    c = grid.Cell("train", "N1", "img")
    with pytest.raises(Exception):
        c.condition = "N2"


# ---------------------------------------------------------------------------
# 未納入的方法（使用者 2026-08-05 裁決）
# ---------------------------------------------------------------------------

def test_未納入的方法都有記錄理由():
    """「某篇為何不在表上」是審稿人一定會問的。移出而不留記錄，
    等於讓那個問題無從查考。"""
    assert set(grid.EXCLUDED) == {"dia_pt", "diffvax"}
    for name, reason in grid.EXCLUDED.items():
        assert len(reason) > 80, f"{name} 的理由過於簡略"


def test_未納入的方法確實不在條件清單內():
    for name in grid.EXCLUDED:
        assert name not in grid.CONDITIONS


def test_dia_pt的程式碼仍保留():
    """裁決是「保留但不實作」——spec 逐字忠於原始碼、不加投影，
    改 CONDITIONS 一行即可納入。刪掉程式碼就回不去了。"""
    from src.baselines import REGISTRY

    assert "dia_pt" in REGISTRY
    assert REGISTRY["dia_pt"].init_rule == "l1_ball", "不得為了規避缺陷而改它"
    assert REGISTRY["dia_pt"].modified_from_paper is False


def test_DIA仍有一個忠實的代表():
    """整篇 DIA 不能因為一個變體有缺陷就整篇消失。"""
    assert "dia_r" in grid.CONDITIONS


def test_五篇baseline():
    assert len(grid.BASELINES) == 5
    assert len(grid.CONDITIONS) == 9
