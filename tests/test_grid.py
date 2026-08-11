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

def test_登記表的條件組成():
    """六個非加性 + 三篇 baseline + 兩條隨機對照。

    DiffVax 不在內：它的免疫器吃 masked image、只支援 inpainting，
    在無 mask 的 SDEdit 下忠實重現結構上不可能（SOURCE_AUDIT §9 第 1 項）。

    2026-08-09：apa（apa + Lo 式 5）與 Ra（apa 上的隨機對照）加入登記表。
    **`CONDITIONS` 是登記表不是某一批的清單**——第三階段的批次由
    `--conditions` 選出五個，位移場的三個移出格點但原始碼與登記表都保留。
    """
    # 2026-08-11：4 → 5，`apa_rd` 加入（改良 5b）。它與 `apa` 同參數化、
    # 同監看量，只有 L_def 的分母不同（`‖Att ⊙ M‖₁` 對「它佔全圖的比例」），
    # 故列在非加性而不是另開一個軸——兩者的對照就是「注意力這個著力點該
    # 怎麼用」。學習率鍵另立 `lr.N5_stage2`，理由見 `ConditionSpec`。
    # 2026-08-11：4 → 6。`apa_rd` 是損失變因、`apa_pj` 是約束變因，兩者都
    # 與 `apa` 同參數化。分成兩個條件而不是一個旗標，是因為它們**不可同時
    # 開**：一起改的話結果變好也分不出是哪一個造成的。
    assert len(grid.NONADDITIVE) == 7
    assert {"apa_rd", "apa_pj", "apa_tj"} <= set(grid.NONADDITIVE)
    assert "diffvax" not in grid.CONDITIONS
    assert set(grid.RANDOM_CONTROLS) <= set(grid.CONDITIONS)


def test_隨機對照不是選配且逐參數化各一個():
    """先驗實測：同一可辨失真上，隨機高斯雜訊即取得最佳化解 60–74% 的
    語意失效。沒有這條對照，任何正結果都不可解讀。

    對照必須與被比較的條件走**同一個參數化**（`DESIGN` §6.3 (b)）。位移場
    的對照是 R、site apa 的是 Ra，兩者不可互相代替：拿位移場的隨機對照去比
    apa 的方法，量到的差異裡混著參數化本身的效果。
    """
    assert grid.RANDOM_CONTROL in grid.CONDITIONS
    assert grid.RANDOM_CONTROL_APA in grid.CONDITIONS
    from src.experiment.executors import condition_spec
    assert condition_spec(grid.RANDOM_CONTROL).site == "warp"
    assert condition_spec(grid.RANDOM_CONTROL_APA).site == "apa"
    # 每一個非加性條件都要有同參數化的隨機對照，否則它的結果不可解讀
    sites = {condition_spec(c).site for c in grid.NONADDITIVE}
    control_sites = {condition_spec(c).site for c in grid.RANDOM_CONTROLS}
    assert sites <= control_sites


def test_主表在tau零點二():
    """τ = 0.10 低於 N3 的 VAE 重建下限，主表設在那裡會只剩位移場一組
    非加性方法（LOGIC_CHECK C1，使用者 2026-08-05 定案）。"""
    assert grid.MAIN_TAU == 0.20
    assert grid.MAIN_TAU in grid.TAUS


def test_兩個完整淨化點都在N3可達的區間內():
    """曲線兩端必須可比。若其中一端 N3 不存在，那一端的比較只有位移場。"""
    for t in grid.FULL_PURIFY_TAUS:
        assert not grid.generative_floor_skip("N3", t), f"τ={t} 對 N3 不適用"


def test_訓練只在一個tau上做一次():
    """逐 τ 重訓是 ×4 成本，且「匹配失真」的前提已四次被證偽。

    訓練點必須是 `TAUS` 中的一個，但**不必是最大的那個**：`solve_k` 雙向
    可行，段 2 能往上也能往下縮放。

    2026-08-06 修正。before：`assert grid.TRAIN_TAU == max(grid.TAUS)`。
    該斷言把「訓練在最大預算」寫死成不變量，而實測顯示位移場在 max(TAUS)
    = 0.35 上已頂死 `max_disp` 的硬上界（disp_max 自 τ=0.20 起固定在
    8√2 = 11.31 px）且人眼明顯壞掉，那個點不該是最佳化發生的地方。
    理由與數據見 `grid.TRAIN_TAU` 的註解。
    """
    assert grid.TRAIN_TAU in grid.TAUS


def test_訓練點是人眼可接受的那個點():
    """訓練點與主表同點。

    主表是全部主張的宣告位置，在別的點上最佳化等於報告一個沒有被最佳化過
    的工作點。這一條與上一條分開：上一條管「是不是格點之一」，這一條管
    「是哪一個」。
    """
    assert grid.TRAIN_TAU == grid.MAIN_TAU


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


def test_逐影像的下限覆蓋名目常數():
    """下限是**該影像**的性質，不是一個對全體成立的常數。

    2026-08-07 加入。SD v1.4／512² 實測 bird_03 0.1330、dog_03 0.1403、
    cat_02 **0.2398**，而名目常數是 0.1434。cat_02 在 τ=0.20 上因此結構上
    不可達，卻被名目值判為可跑，送進 `solve_k` 之後以 `ValueError` 中止、
    該格記為 failed，整個分片停住。
    """
    # 名目值判為可跑，實測值判為不可跑——兩者必須給出不同答案
    assert grid.generative_floor_skip("N3", 0.20) == ""
    r = grid.generative_floor_skip("N3", 0.20, floor=0.2398)
    assert r and "0.2398" in r

    # 反向：某張圖的下限比名目值低時，名目值不該把可跑的格擋掉
    assert grid.generative_floor_skip("N3", 0.10) != ""
    assert grid.generative_floor_skip("N3", 0.10, floor=0.0889) == ""


def test_實測與名目的理由字串分得出來():
    """報表要看得出這一格是依實測值還是依常數被跳過的。"""
    nominal = grid.generative_floor_skip("N3", 0.05)
    measured = grid.generative_floor_skip("N3", 0.05, floor=0.2398)
    assert "名目" in nominal and "名目" not in measured
    assert "實測" in measured


def test_格點依逐影像下限跳過正確的格():
    """cat_02 只有 τ=0.20 被跳過，bird_03 在同一個 τ 上照跑。"""
    floors = {"bird_03": 0.1330, "cat_02": 0.2398, "dog_03": 0.1403}
    cells = grid.rayscale_cells(
        ["bird_03", "cat_02", "dog_03"], conditions=("N3",),
        taus=(0.20, 0.35), floors=floors)
    got = {(c.image_id, c.tau): c.skipped for c in cells}
    assert got[("cat_02", 0.20)] is True
    assert got[("cat_02", 0.35)] is False, "0.35 高於該圖下限，仍要跑"
    assert got[("bird_03", 0.20)] is False
    assert got[("dog_03", 0.20)] is False


def test_跳過與否不進config_hash():
    """換一個下限不得讓已完成的格重跑。

    `skip_reason` 是排程資訊不是設定：它決定「這一格跑不跑」，不決定
    「跑起來算的是什麼」。若它進了雜湊，2026-08-07 這次修正會讓兩個批次
    已經算完的上千格全部作廢。
    """
    from src.experiment.runner import cell_config
    from src.utils.cellid import config_hash

    base = {"spec_version": 1, "model": "m", "resolution": 512,
            "guidance": 7.5, "steps": 50, "strength": 0.6, "gpu": "g",
            "precision": "fp32", "loss_params": {}, "module_params": {},
            "optim_params": {}, "lr": None}
    a = grid.Cell("rayscale", "N3", "cat_02", tau=0.35, skip_reason="")
    b = grid.Cell("rayscale", "N3", "cat_02", tau=0.35,
                  skip_reason="某個理由")
    assert config_hash(cell_config(a, base)) == config_hash(cell_config(b, base))


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


def test_主組的算子清單():
    """2026-08-06 起不含 `impress`（機時裁決，程式保留、見 EXCLUDED）。"""
    kinds = {k for k, _ in grid.MAIN_PURIFIERS}
    assert kinds == {"identity", "jpeg", "crop_resize", "adverse_cleaner",
                     "cnn_denoise_substitute", "diffpure"}


def test_抗淨化仍有強淨化與強度掃描():
    """主張一是抗淨化。移除 IMPRESS 之後，承擔該主張的是 DiffPure 這個
    強淨化對照，加上 JPEG 與 blur 的強度掃描——三者缺一，主張一就只剩
    單點量測，看不出「效果隨淨化強度怎麼衰減」。"""
    main = {k for k, _ in grid.MAIN_PURIFIERS}
    assert "diffpure" in main
    jpeg = sorted(s for k, s in grid.MAIN_PURIFIERS if k == "jpeg")
    blur = sorted(s for k, s in grid.SWEEP_PURIFIERS if k == "blur")
    assert len(jpeg) >= 2, f"JPEG 至少要兩個強度，實得 {jpeg}"
    # 2026-08-11：由 ≥4 降為 ≥3。強度點數本身不是目的，可讀性才是——
    # blur 1.0 起未防禦的編輯自己就被毀掉（ΔNIQE +1.63），多留一點只是
    # 多一列比不出東西的數字。判準見 `SWEEP_PURIFIERS`。
    assert len(blur) >= 3, f"blur 至少要三個強度，實得 {blur}"
    assert max(blur) <= 1.0, (
        f"blur 掃描不得含毀掉未防禦編輯的強度，實得 {blur}")


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
    # 走生成路徑的條件在兩個低 τ 上不適用（名目下限 0.1434 > 0.05、0.10）
    n_gen = len(grid.GENERATIVE_CONDITIONS)
    assert s["rayscale"]["skipped"] == n_img * 2 * n_gen
    assert s["eval"]["total"] == sum(
        len(grid.purifiers_for(t)) for t in grid.TAUS
    ) * n_cond * n_img * grid.N_SEEDS


# ------------------------------------------------- 逐批次的 τ 計畫（第三階段）


def test_不指定tau_train時v14的格點逐格不變():
    """**本輪最重要的格點回歸測試。**

    `TAUS` / `MAIN_TAU` / `TRAIN_TAU` / `FULL_PURIFY_TAUS` 是模組層級常數，
    就地改會讓 v14／v14r 重跑產出不同的格點，破壞 `runs/` 已入版控的證據。
    故改為逐批次的 `TauPlan`，而預設路徑必須回到**同一個物件**。

    逐格比對而不是只比數量：數量相同但某一格的 τ 或淨化設定換了，
    報表看起來仍然完整。
    """
    assert grid.tau_plan_for() is grid.DEFAULT_TAU_PLAN
    assert grid.tau_plan_for(grid.TRAIN_TAU) is grid.DEFAULT_TAU_PLAN

    before = grid.plan(IMGS)
    after = grid.plan(IMGS, tau_plan=grid.tau_plan_for(grid.TRAIN_TAU))
    for stage in before:
        assert [c.cell_id() for c in before[stage]] == \
               [c.cell_id() for c in after[stage]], f"{stage} 的格點改變了"


def test_換tau_train時四個常數一致地跟著走():
    """只改訓練點而不動其餘三個，訓練點上一個淨化格都不會有——而抗淨化
    （主張一）的分母正是訓練點上的效果。症狀只是「那幾列不在表裡」。"""
    p = grid.tau_plan_for(0.50)
    assert p.train_tau == 0.50
    assert p.main_tau == 0.50, "主表與掃描組必須落在最佳化實際發生的那一點"
    assert 0.50 in p.taus
    assert 0.50 in p.full_purify_taus

    full = grid.purifiers_for(0.50, tau_plan=p)
    assert len(full) == len(grid.MAIN_PURIFIERS) + len(grid.SWEEP_PURIFIERS)
    assert grid.IDENTITY in full
    # 其餘 τ 只補曲線的低端，仍然只跑 identity
    assert grid.purifiers_for(0.20, tau_plan=p) == [grid.IDENTITY]


def test_訓練點不在完整淨化組內即拒絕():
    """這正是 §4a 指出的失效，故在資料結構層就擋下，不等到跑完才發現。"""
    with pytest.raises(ValueError, match="full_purify_taus"):
        grid.TauPlan(taus=(0.20, 0.50), main_tau=0.50, train_tau=0.50,
                     full_purify_taus=(0.20,))


def test_主表或訓練點不在tau軸上即拒絕():
    with pytest.raises(ValueError, match="不在 taus"):
        grid.TauPlan(taus=(0.05, 0.20), main_tau=0.50, train_tau=0.20,
                     full_purify_taus=(0.20,))


def test_批次A的格點規模():
    """五個條件、τ_train=0.50、完整淨化組只在訓練點。

    釘住它是因為機時估計直接由格數導出，而估計錯了會排錯批次。
    """
    conds = ("apa", "Ra", "photoguard_c", "mist", "dia_r")
    p = grid.tau_plan_for(0.50)
    s = grid.summarize(grid.plan(IMGS, conditions=conds, tau_plan=p))
    assert s["train"]["total"] == 5 * 3
    assert s["rayscale"]["total"] == 5 * 3 * 5
    # 2026-08-11：1725 → 1500，掃描組由 12 個強度重訂為 9 個（弱端補三點、
    # 強端移除七點，判準見 `SWEEP_PURIFIERS` 的說明）。差額 −225 = 5 條件 ×
    # 3 影像 × 5 seed × (−3) 個強度；`control` 同理由 285 減到 240。
    assert s["eval"]["total"] == 1500
    assert s["control"]["total"] == 240


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
    assert set(grid.EXCLUDED) == {"dia_pt", "diffvax", "advpaint",
                                  "promptflare", "impress"}
    for name, reason in grid.EXCLUDED.items():
        assert len(reason) > 80, f"{name} 的理由過於簡略"


def test_未納入的方法確實不在條件清單內():
    for name in grid.EXCLUDED:
        assert name not in grid.CONDITIONS
    # impress 是淨化算子不是條件，它要不在的是主組
    assert "impress" not in {k for k, _ in grid.MAIN_PURIFIERS}


def test_機時裁決移除的方法程式碼仍保留():
    """裁決是「保留但不實作」。刪掉程式碼就回不去了，而三者各自的逐行
    原始碼佐證（`SOURCE_AUDIT` §1、§2、§8）是重跑時唯一的依據。"""
    from src.baselines import REGISTRY
    from src.purify.impress import PHOTOGUARD_PRESET, impress_real

    for name in ("advpaint", "promptflare"):
        assert name in REGISTRY, f"{name} 的 spec 被刪掉了"
    assert callable(impress_real)
    assert PHOTOGUARD_PRESET["iters"] == 1000, "保留的必須是原設定"


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


def test_三篇baseline():
    """2026-08-06 的機時裁決後為三篇。三篇都是常被引用的加性對照，
    次要主張的「最佳 baseline」因此仍有公認的比較對象。"""
    assert grid.BASELINES == ("photoguard_c", "mist", "dia_r")
    # 登記表 = 非加性 + baseline + 隨機對照，且三者互不重疊
    assert len(grid.CONDITIONS) == (
        len(grid.NONADDITIVE) + len(grid.BASELINES) + len(grid.RANDOM_CONTROLS)
    )
    assert len(set(grid.CONDITIONS)) == len(grid.CONDITIONS)


# --------------------------------------------------- 條件篩選（--conditions）


def test_未指定條件時涵蓋全部():
    assert grid.resolve_conditions(None) == grid.CONDITIONS
    assert grid.resolve_conditions([]) == grid.CONDITIONS


def test_指定條件時只留下指定的那些():
    got = grid.resolve_conditions(["N2", "N3"])
    assert got == ("N2", "N3")
    cells = grid.plan(["img_a"], conditions=got)["train"]
    assert sorted({c.condition for c in cells}) == ["N2", "N3"]


def test_未知的條件名必須拋出而不是靜默忽略():
    """打錯字若被忽略，整段會跑成空集合而看起來像「全部都已完成」。"""
    with pytest.raises(ValueError, match="N9"):
        grid.resolve_conditions(["N2", "N9"])


def test_條件篩選不影響共用的phi0對照格():
    """control 是 φ=0 的同淨化對照，跨條件共用，故不隨條件篩選而變。

    否則以 `--conditions N2` 續跑時，那 285 格會被算成另一組雜湊而重跑一次。
    """
    full = grid.plan(["img_a"], conditions=grid.CONDITIONS)["control"]
    part = grid.plan(["img_a"], conditions=("N2",))["control"]
    assert [c.cell_id() for c in full] == [c.cell_id() for c in part]


# --------------------------------------------------------- 相對預算軸（Δ 模式）


def test_相對預算軸只有一個點且完整淨化組落在該點():
    plan = grid.budget_tau_plan(0.04)
    assert plan.taus == (0.04,)
    assert plan.main_tau == plan.train_tau == 0.04
    assert plan.full_purify_taus == (0.04,)
    assert plan.relative is True
    assert plan.metric == "dists", "預設指標是 DISTS，見 MET-dists"


def test_相對預算軸不因重建下限跳過生成路徑的格():
    """下限已被減掉，任何正的 Δ 都落在 `build(0)` 之上。

    絕對軸上 τ=0.04 對 apa 是結構上不可達（VAE 來回下限就有 0.14），
    相對軸上同一個數字是「超出下限 0.04」，兩者不是同一件事。
    """
    assert grid.generative_floor_skip("apa", 0.04, floor=0.1581) != ""
    assert grid.generative_floor_skip("apa", 0.04, floor=0.1581,
                                      relative=True) == ""

    cells = grid.plan(["horse_00"], conditions=("apa",),
                      floors={"horse_00": 0.1581},
                      tau_plan=grid.budget_tau_plan(0.04))
    assert [c for c in cells["rayscale"] if c.skipped] == []
    assert [c for c in cells["eval"] if c.skipped] == []


def test_相對預算軸的Δ必須為正():
    with pytest.raises(ValueError, match="delta"):
        grid.budget_tau_plan(0.0)


def test_預算軸的指標只接受lpips與dists():
    with pytest.raises(ValueError, match="metric"):
        grid.budget_tau_plan(0.04, "psnr")


def test_預設τ計畫是絕對軸且量在lpips上():
    """新增模式不得改動既有批次的軸。"""
    assert grid.DEFAULT_TAU_PLAN.relative is False
    assert grid.DEFAULT_TAU_PLAN.metric == "lpips"
