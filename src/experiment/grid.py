"""四軸格點的列舉 — `docs/DESIGN.md` §3。

本模組決定「實際會跑哪些格子」。它是純函數、不碰 GPU，因為
**格點定義錯誤是事後看不出來的**：少跑一個條件、某個 τ 的淨化掃描漏了一半，
產出的表格看起來仍然完整，只是少了幾列，而讀者無從得知。

## 四個軸

| 軸 | 內容 |
|---|---|
| 一＋二：方法 × loss | 9 個訓練條件（N1／N2／N3、B1–B5、R） |
| 三：失真預算 | τ ∈ {0.05, 0.10, **0.20**, 0.35}，由射線縮放取得，不逐 τ 重訓 |
| 四：淨化 | 主組 6 個文獻算子 + identity；掃描組 12 個設定 |

## 兩個不對稱，都是刻意的

**淨化掃描只在 τ ∈ {0.20, 0.35} 完整跑**，另兩點只跑 identity。
兩個完整點都落在 N3 可達的區間內（N3 走生成路徑，其 VAE 重建誤差下限
LPIPS 0.1434 使 τ = 0.05 與 0.10 結構上不可能達成），故曲線兩端可比。

**掃描組只在主表所在的 τ = 0.20 跑。** 它用於機制分析——先驗實驗顯示
強模糊與強量化下非加性的優勢會反轉，需確認該型態在正確的攻擊設定下是否重現。

## N3 的不適用格

N3 在 τ = 0.05、0.10 上標記為 `skipped` 而非 `failed`。
把結構上不適用算成失敗，儀表板會永遠是紅的，然後就沒有人看它了。
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 軸一＋二：訓練條件
# ---------------------------------------------------------------------------

# 三個非加性條件是本專案的方法；五個 baseline 是加性對照（使用者 2026-08-05
# 決定不再自行實作加性方法，由 baseline 擔任該角色）；R 是同失真隨機對照。
#
# 2026-08-09（第三階段）新增 `apa`。它與 N3 同樣走 site apa，差別在損失：
# N3 是 `targeted_output`（推向固定目標影像），apa 是 `suppress_attn_ca`
# （Lo et al. 式 5，壓低防禦方指名的詞 c_a 在其對應區域的注意力）。
# `apa_rd` 是 `apa` 的損失變因（改良 5b，2026-08-11）：同參數化、同學習率鍵，
# L_def 由 `‖Att ⊙ M‖₁` 改為它佔全圖的比例。列在這裡而不是另開一個軸，是因為
# 它與 `apa` 的對照就是「注意力這個著力點該怎麼用」，屬於同一條主張。
# `apa_pj` 是 `apa` 的**約束**變因（改良 1–3）：同一個損失，保真度由 hinge
# 改為每步投影回失真預算的球面。與 `apa_rd`（損失變因）分開列，是因為兩者
# 不可同時開——一起改的話結果變好也分不出是哪一個造成的。
NONADDITIVE = ("N1", "N2", "N3", "apa", "apa_rd", "apa_pj", "apa_tj")

# `dia_pt` 與 `diffvax` 保留在程式中但**不納入本輪實驗**，各有其原因，
# 兩者都記錄在 `EXCLUDED` 並由測試釘住——移出而不留記錄，等於讓
# 「為什麼少了這一篇」變成無從查考的事。
BASELINES = ("photoguard_c", "mist", "dia_r")

# 同失真隨機對照，**逐參數化各一個**。
#
# 隨機對照必須與被比較的條件走**同一個參數化**（`DESIGN` §6.3 (b)：問的是
# 「最佳化取得了多少，超過同樣形狀的隨機擾動」）。`R` 是位移場上的那一個，
# 只能拿來對照 N1／N2；N3／apa 走 site apa，其對照是 `Ra`。
#
# 沒有 `Ra` 的話 apa 上的任何正結果都不可解讀——v14r 正是靠 `R` 才判斷出
# 位移場沒有貢獻（N1 對 R 的 `edit_lpips` 比值 1.046、語意失敗 4/15 對 4/15，
# `RESULTS_2026-08-08` §12.1 與 §11.3）。
RANDOM_CONTROL = "R"
RANDOM_CONTROL_APA = "Ra"
RANDOM_CONTROLS = (RANDOM_CONTROL, RANDOM_CONTROL_APA)

# 未納入的方法與理由。**保留在此而非刪除**：報表與論文都要引用這些理由，
# 而「某篇為何不在表上」是審稿人一定會問的。
EXCLUDED: Dict[str, str] = {
    "dia_pt": (
        "DIA 的 l1_ball 起點在某些輸入下遠超其 eps 球（實測 eps=0.05 下 "
        "‖d‖∞ 達 1.499，30 倍）。根因是其 _l1_projection 取自 AutoAttack，"
        "該演算法的箱型約束假設 x ∈ [0,1] 而 DIA 用在 [-1,1] 上。"
        "加投影可以修好，但那是我方改動別人的攻擊程序；同一篇的 DIA-R "
        "變體不受影響且已納入，故 DIA 仍有忠實的代表。程式碼保留於 "
        "src/baselines/dia.py，改 CONDITIONS 一行即可納入。"
    ),
    "diffvax": (
        "其免疫器實際餵入的是編輯區已歸零的 masked image（完整原圖載入後"
        "從未使用），attack() 硬編碼 9 通道 inpainting 輸入，全 repo 對 "
        "img2img／SDEdit 命中 0 筆。在無 mask 的全圖 SDEdit 威脅模型下"
        "忠實重現結構上不可能，強行改寫等於我方設計一個新方法再冠上它的名字。"
        "另其論文報告的 counter-attack 評測（CNN 去噪、JPEG、IMPRESS）"
        "在 repo 中完全不存在。改列為相關工作引用，不進比較表。"
    ),
    # 以下三項為 2026-08-06 的機時裁決（使用者決定），與上面兩項的「方法本身
    # 有結構性問題」不同類。理由仍寫在此，因為審稿人問的是同一個問題。
    "advpaint": (
        "機時裁決（使用者 2026-08-06）。基準留三篇：PhotoGuard（ICML 2023）、"
        "Mist（ICML 2023 Oral）、DIA（ICCV 2025），三者皆為常被引用的加性"
        "對照，次要主張的「最佳 baseline」仍有公認的比較對象。移除兩篇省下"
        "段 1 約 1 小時與段 3 的 900 格（約 4 小時，3090 實測外推）。"
        "程式碼與逐行佐證完整保留於 src/baselines/advpaint.py，"
        "把名字加回 BASELINES 一行即可納入。"
    ),
    "promptflare": (
        "機時裁決（使用者 2026-08-06），與 advpaint 同一次決定。它是五篇中"
        "訓練最貴的第二名（400 步 × 2 列，約為 PhotoGuard-c 的十分之一而為"
        "其餘三篇的二至八倍）。程式碼保留於 src/baselines/promptflare.py，"
        "把名字加回 BASELINES 一行即可納入。"
    ),
    "impress": (
        "機時裁決（使用者 2026-08-06）：程式保留但本輪不執行。IMPRESS "
        "（Cao et al., NeurIPS 2023）每格是 1000 次 Adam 迭代，每次為一整個"
        "1024² SDXL VAE 的前向加反向，285 格依實測的 VAE 成本外推逾 100 小時，"
        "且在 23.56 GB 的 RTX 3090 上實測 OOM（用掉 23.45 GB 後失敗）。"
        "抗淨化的證據改以 DiffPure 與 JPEG／blur 的強度掃描承擔。"
        "程式碼與參數佐證保留於 src/purify/impress.py，"
        "把它加回 MAIN_PURIFIERS 一行即可納入。"
    ),
}

# 隨機對照不是選配。先驗實測：在同一可辨失真上，隨機高斯雜訊即取得最佳化解
# 60–74% 的語意失效。沒有這條對照，任何正結果都不可解讀。
#
# `CONDITIONS` 是**已定義條件的登記表**，不是某一批要跑的清單。哪些條件進入
# 某一批由 `--conditions` 決定（`resolve_conditions`），批次的選擇記在
# `scripts/shard.sh` 的 profile 裡。
#
# 2026-08-09：第三階段的批次跑 `apa Ra photoguard_c mist dia_r` 五個條件，
# 位移場的三個（N1／N2／R）**移出格點但原始碼與登記表都保留**——`runs/` 有
# 36893 個已入版控的檔案要靠 `site_warp` 由 `.pt` 重建，刪掉它們等於讓那些
# 證據無法還原。放棄位移場的量化依據見 `docs/archive/DECISION_stage3.md`。
CONDITIONS: Tuple[str, ...] = NONADDITIVE + BASELINES + RANDOM_CONTROLS


def resolve_conditions(names) -> Tuple[str, ...]:
    """把 CLI 的 `--conditions` 轉成格點用的條件序列；未指定即全部。

    存在理由是 `purify_mode` 進 `config_hash`（`RunConfig.loss_params`）而它
    **只影響訓練期對淨化算子取期望值的方式**：`src/baselines/` 完全不引用
    `Purifier`，R 不最佳化，故那四個條件的結果與該旗標無關。沒有條件篩選時，
    只為了改 N2／N3 的取樣方式就得整批換旗標，而那會讓 baseline 的雜湊一起
    改變、把已經算完的 `photoguard_c` 判成未完成——那一格是段 1 的主成本
    （200 步 × 10 reps × 4 步去噪，以成本單位計為其餘兩篇的 40 至 80 倍）。

    未知名稱一律拋出。靜默忽略打錯的名字會得到空集合，而空集合在續跑判定
    下看起來與「全部都已完成」完全一樣。
    """
    if not names:
        return CONDITIONS
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"未知的條件 {unknown}；本輪已定義的是 {list(CONDITIONS)}。"
            f"（已排除的方法見 EXCLUDED，加回去要改 NONADDITIVE／BASELINES）"
        )
    # 依 CONDITIONS 的順序輸出，使兩次給定順序不同的呼叫得到相同的格點順序
    return tuple(c for c in CONDITIONS if c in set(names))

# ---------------------------------------------------------------------------
# 軸三：失真預算
# ---------------------------------------------------------------------------

TAUS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.35)
MAIN_TAU = 0.20          # 主表所在點，見模組 docstring

# 訓練點。其餘 τ 一律由段 2 的射線縮放取得（`solve_k` 雙向可行，故訓練點
# 不必是最大的那個）。
#
# 2026-08-06 由 0.35 改為 0.20（使用者裁決）。before：取 max(TAUS)，理由是
# 「訓練在最大預算做一次，其餘縮放取得」。實測推翻了那個理由——**位移場在
# τ=0.35 上已經不是一個可用的工作點**：
#
#   τ      LPIPS    PSNR    銳利度比   disp_max    k
#   0.05   0.0456   33.40   1.023      5.84 px     3.0
#   0.10   0.0992   29.33   1.026      9.73 px     5.5
#   0.20   0.1999   25.99   1.037     11.31 px    10.5   ← 主表，人眼可接受
#   0.35   0.3513   23.34   1.075     11.31 px    26.0   ← 人眼明顯壞掉
#
# （隨機對照 R、bird_03、SDXL/1024²，`runs/b1_bird_03/R/bird_03/`）
#
# 兩個獨立的證據：
#
# 1. `max_disp=8.0` 的每分量上界對應最大位移量 8√2 = 11.31 px，而 τ=0.20 與
#    0.35 量到的**都是 11.31**。從 0.20 往上，k 由 10.5 拉到 26.0 而最大位移
#    一動也不動——射線縮放已不是在縮放射線，只是把越來越多像素壓到上界，
#    訓練學到的方向在此過程中被逐步破壞。
# 2. 人眼：τ=0.35 的圖輪廓整條起伏、喙斷開、腹線呈波浪狀。這不是「匹配的
#    不可辨失真」。**指標與人眼矛盾時以人眼為準**（`CLAUDE.md`）。
#
# 這同時是「LPIPS 低估非加性可辨度」的量化形式：同為 LPIPS 0.35，加性擾動
# 看起來是雜訊，位移場看起來是整張圖被扭曲。
#
# τ=0.35 仍留在 `TAUS` 中照實報告，只是不再是**最佳化**發生的地方。
TRAIN_TAU = 0.20
FULL_PURIFY_TAUS: Tuple[float, ...] = (0.20, 0.35)


# ---------------------------------------------------------------------------
# 逐批次的 τ 計畫（2026-08-09，第三階段）
# ---------------------------------------------------------------------------
#
# 上面四個常數是**模組層級的**，而第三階段的批次 A 要在 τ_train = 0.50 上跑。
# 就地改它們會讓 v14／v14r 重跑時產出不同的格點，破壞 `runs/` 已入版控的
# 160705 個檔案的可重現性——那些檔案是唯一的證據來源，容器已刪、實驗無法重跑。
#
# 故改為：常數不動，另立一個逐批次的計畫物件，由 `--tau-train` 導出。
#
# **為什麼四個常數必須一致地跟著批次走**，而不是只改 `TRAIN_TAU`：
# `purifiers_for` 對不在 `FULL_PURIFY_TAUS` 內的 τ 只回傳 identity，且掃描組
# 只在 `tau == MAIN_TAU` 上跑。照現況把 τ_train 設成 0.50 而不動其餘三個，
# 結果會是**訓練點上一個淨化格都沒有**——而抗淨化是主張一。這個症狀在報表上
# 看起來只是「那幾列不在表裡」，不會有任何錯誤訊息。

@dataclass(frozen=True)
class TauPlan:
    """一個批次的失真預算軸。四個欄位必須一致，故綁在同一個物件上。

    `taus` 是要報告的全部預算點；`train_tau` 是**最佳化實際發生**的那一點，
    其餘由段 2 的射線縮放取得；`main_tau` 是主表與掃描組所在點；
    `full_purify_taus` 是跑完整主組淨化算子的點。
    """

    taus: Tuple[float, ...]
    main_tau: float
    train_tau: float
    full_purify_taus: Tuple[float, ...]
    # 預算軸量在哪個保真指標上，以及 τ 是絕對值還是相對於該格自己的 φ=0
    # 下限的增量。**兩者都有預設，故既有批次的 TauPlan 逐欄不變。**
    #
    # `relative=True` 的意義見 `budget_tau_plan`：τ 不再是「達成的失真」而是
    # 「超出重建下限多少」。生成路徑的下限因此被減掉，任何正的 τ 都可達，
    # `generative_floor_skip` 在該模式下不跳過任何格。
    metric: str = "lpips"
    relative: bool = False

    def __post_init__(self):
        if self.metric not in ("lpips", "dists"):
            raise ValueError(
                f"metric={self.metric!r} 不是 lpips 或 dists。射線縮放要對"
                "哪個指標二分是本批的宣告，不接受靜默回退")
        for name in ("main_tau", "train_tau"):
            if getattr(self, name) not in self.taus:
                raise ValueError(
                    f"{name}={getattr(self, name)} 不在 taus={self.taus} 內。"
                    f"主表或訓練點不在報告的預算軸上，該點的結果不會被列舉出來"
                )
        missing = [t for t in self.full_purify_taus if t not in self.taus]
        if missing:
            raise ValueError(
                f"full_purify_taus 的 {missing} 不在 taus={self.taus} 內"
            )
        if self.train_tau not in self.full_purify_taus:
            raise ValueError(
                f"train_tau={self.train_tau} 不在 full_purify_taus="
                f"{self.full_purify_taus} 內：訓練點上不會有任何淨化格，"
                "而抗淨化（主張一）的分母正是訓練點上的效果。"
                "這正是 2026-08-09 §4a 指出的失效"
            )


DEFAULT_TAU_PLAN = TauPlan(
    taus=TAUS, main_tau=MAIN_TAU, train_tau=TRAIN_TAU,
    full_purify_taus=FULL_PURIFY_TAUS,
)


def budget_tau_plan(delta: float, metric: str = "dists") -> TauPlan:
    """相對預算軸：τ 是**超出該格自己 φ=0 下限**的增量。

    ## 為什麼要有這個模式

    絕對 LPIPS 軸把兩件性質不同的失真加在一起：生成路徑的 VAE 來回下限，
    與最佳化真正加上去的那一份。前者實測**看不出來**——`φ=0` 的重建圖與
    原圖在人眼下無法區分，儘管 LPIPS 已經是 0.128（2026-08-10 逐圖確認）。
    把兩份綁在同一個門檻上，等於讓加性 baseline 拿到全部預算、非加性只拿到
    扣掉下限之後的餘額，而扣掉的那一份並不可見。

    相對模式把下限減掉：**每一格都取自己的 `build(0)` 當原點**，加性位置的
    原點就是原圖（下限 0），故同一條規則對兩類位置都成立，不必特例。

    ## 為什麼指標預設是 DISTS

    LPIPS 是全圖平均，主體只占畫面 15% 時，主體被改寫的代價會被背景稀釋
    ——實測 LPIPS 給「主體毀掉」的 horse_00 0.179、給人眼可接受的 horse_03
    0.198，順序是反的（`docs/METRICS.md` MET-dists）。DISTS、NIQE、VIF_p 與
    銳利度比在同一組樣本上排序正確。

    只給一個 τ 點：相對軸上的低端沒有意義（τ→0 就是重建圖），曲線要補
    低端仍用絕對模式。
    """
    delta = float(delta)
    if not delta > 0:
        raise ValueError(f"delta 須為正數，收到 {delta!r}")
    return TauPlan(taus=(delta,), main_tau=delta, train_tau=delta,
                   full_purify_taus=(delta,), metric=metric, relative=True)


def tau_plan_for(train_tau: Optional[float] = None,
                 full_purify_taus: Optional[Sequence[float]] = None
                 ) -> TauPlan:
    """由 `--tau-train` 導出該批次的 τ 計畫。

    **`train_tau` 為 None 或恰等於模組常數 `TRAIN_TAU` 時，回傳
    `DEFAULT_TAU_PLAN` 本身**，即 v14／v14r 的格點逐格不變。這一條由
    `tests/test_grid.py` 釘住——本函式的存在理由是換批次，不是換既有結果。

    其餘情形的導出規則：

    - `taus` = 既有的四點 ∪ {train_tau}，排序。既有四點照實保留，讓
      失真–效果曲線的低端仍然補得起來，且與 v14／v14r 的報告點對得上。
    - `main_tau` = `train_tau`。主表與掃描組必須落在最佳化實際發生的那一點；
      v14 的主表與訓練點本來就是同一點（都是 0.20），此處只是把該性質寫成規則。
    - `full_purify_taus` = `(train_tau,)`，除非呼叫端明給。

    最後一項是**本輪的選擇，不是既有設計的延伸**，故必須說明：v14 在 0.20 與
    0.35 兩點跑完整淨化組，理由是「兩個完整點都落在 N3 可達的區間內，故曲線
    兩端可比」。批次 A 的 φ 訓練在 0.50，其 0.35 那一點是射線縮放的結果，
    與 v14r 訓練在 0.20 的 0.35 不是同一個東西，並列不構成「可比」。而多一個
    完整點的代價是段 3 的 eval 由 1725 格增為 2175 格（+26%）。故預設只在
    訓練點跑完整淨化組，要加回來用 `--full-purify-taus` 明給並記錄理由。
    """
    if train_tau is None or float(train_tau) == TRAIN_TAU:
        if full_purify_taus is None:
            return DEFAULT_TAU_PLAN
        train_tau = TRAIN_TAU

    train_tau = float(train_tau)
    if not train_tau > 0:
        raise ValueError(f"train_tau 須為正數，收到 {train_tau!r}")
    taus = tuple(sorted(set(TAUS) | {train_tau}))
    full = (tuple(sorted(float(t) for t in full_purify_taus))
            if full_purify_taus is not None else (train_tau,))
    return TauPlan(taus=taus, main_tau=train_tau, train_tau=train_tau,
                   full_purify_taus=full)

# 走生成路徑的條件在此下限以下結構上不可能達成。**這是名目值，只用於
# 乾跑**：真正執行時一律改用該影像自己實測的下限（見下方
# `generative_floor_skip` 的 `floors`）。
#
# 2026-08-07 改。before：本常數是唯一的判準。實測顯示下限**逐影像**差很多
# ——SD v1.4／512² 上 bird_03 為 0.1330、dog_03 0.1403，而 cat_02 是
# **0.2398**，高於 τ=0.20。於是 `rayscale/N3/cat_02/tau0.2` 被送進 `solve_k`，
# 二分 28 次後正確地拒絕（k 已推到 3.7e-09 仍降不到目標）並拋出，該格記為
# `failed`，整個分片就此停住。這與 2026-08-06 第 8 號缺陷（門檻是一個不可達
# 的全域常數）是同一個型態：一個對「平均影像」成立的常數，套到個別影像上
# 就不成立。
GENERATIVE_LPIPS_FLOOR = 0.1434
# 全部走 site apa 的條件都有這個下限——它來自 `decode(encode(x))` 這條來回，
# 與損失是什麼無關。2026-08-09 補入 apa 與 Ra：漏掉的症狀是段 2 的 `solve_k`
# 在達不到的 τ 上二分到極限才拋出，整個分片就此停住（2026-08-07 的 cat_02 事故）。
GENERATIVE_CONDITIONS = ("N3", "apa", "Ra")

# ---------------------------------------------------------------------------
# 軸四：淨化
# ---------------------------------------------------------------------------

IDENTITY = ("identity", 0.0)

# 主組：文獻共識的算子。出處見 `docs/reference/SOURCE_AUDIT.md` §8。
# `cnn_denoise_substitute` 的命名刻意帶 substitute——NTIRE 2023 冠軍的
# 程式碼與權重皆未公開，此為我方替代，不得聲稱重現 DiffVax 的該項評測。
#
# 2026-08-06 由六項降為五項：`impress` 移入 `EXCLUDED`（機時裁決，程式保留）。
# 抗淨化的證據因此由 DiffPure 這個「強淨化」對照，加上 SWEEP_PURIFIERS 的
# JPEG 兩點與 blur 四點強度掃描承擔。
MAIN_PURIFIERS: Tuple[Tuple[str, float], ...] = (
    IDENTITY,
    ("jpeg", 75),
    ("jpeg", 30),
    ("crop_resize", 0.10),
    ("adverse_cleaner", 0.0),
    ("cnn_denoise_substitute", 0.0),
    ("diffpure", 150),
)

# 掃描組：本專案既有的四個算子，用於機制分析。
# 2026-08-11 重訂掃描組（使用者裁決）。**強端整段移除，不只是補弱端。**
#
# 判準是事前宣告的、量得到的：**未防禦的編輯自己被淨化毀掉多少**。
# 淨化算子作用在防禦圖上，而 `edit_lpips` 比的是「未防禦的編輯」對「防禦後
# 的編輯」——若前者本身已被毀掉，這個量比的是糊對糊，任何比值都不再是防禦
# 的強弱。以 `edit_niqe_a`（未防禦編輯的 NIQE，越高越糟）對不淨化的基線
# 3.487 取差，實測（`runs/s3t20_merged`，apa 那一組，3 影像 × 5 seed）：
#
#     blur 0.5  −0.28    jpeg 75   −0.20    quantize 64  −0.02    ← 可用
#     crop 0.1  −0.17    identity   0.00    quantize 32  +0.07
#     jpeg 30   +0.17    quantize 16 +0.27  adverse_cl.  +0.25
#     ---- 以下 ΔNIQE ≥ +1.0，未防禦編輯已被毀掉，整段移除 ----
#     quantize 8 +1.13   blur 1.0  +1.63    noise 0.02   +1.90
#     noise 0.04 +3.80   blur 2.0  +4.59    noise 0.08   +6.44
#     blur 3.0   +6.40
#
# `noise 0.01`（+0.77）留下：它在門檻內，且是雜訊那一軸唯一還可讀的點。
# 移除的那七個設定**已跑出來的格不刪**（`runs/` 是唯一的證據來源），只是
# 不再列入格點、也不進報告的曲線；要引用它們必須連同「未防禦側已毀掉」
# 這件事一起講。
#
# 主組（`MAIN_PURIFIERS`）不動：那是文獻共識的算子清單，`diffpure 150`
# 雖然 ΔNIQE +1.72，但它是重生成而不是劣化，且移掉它等於拿掉唯一的強淨化
# 對照。它的可讀性限制在報告裡以文字交代，不用改格點解決。
# **加點不換點**：既有四個強度的 `config_hash` 逐位不變，故 s3t20 與 ip20
# 已完成的格全部沿用，只跑新增的那些。
#
# 補的理由是分辨力，不是完整性：加性 baseline 的保留率在 blur 0.5 → 1.0
# 之間掉了 54 個百分點（84% → 30%），而非加性維持在 100% 上下，交叉點就落
# 在那一段裡而該段中間沒有取樣點。只有兩端的話，「加性掉到一半時非加性還
# 剩多少」這句話寫不出來——那正是主張一要報的那條曲線。
#
# blur 3.0 的教訓一併記在這裡：該強度下**未防禦的編輯本身就已整張糊掉**，
# 各條件的 `edit_lpips` 都塌到 0.02–0.10，此時的比值比的是殘差不是防禦。
# 補弱端就是為了讓曲線落在「輸入還沒被毀掉」的區間裡。
SWEEP_PURIFIERS: Tuple[Tuple[str, float], ...] = tuple(
    [("blur", s) for s in (0.25, 0.5, 0.75)]
    + [("noise", s) for s in (0.005, 0.01)]
    + [("quantize", n) for n in (128, 64, 32, 16)]
)

# ---------------------------------------------------------------------------
# 種子
# ---------------------------------------------------------------------------

# 兩個條件共用逐元素相同的評測噪聲、逐種子相減，使所需樣本數由約 48 降到約 7。
# n ≥ 2 是硬性下限：n = 1 時樣本標準差恆為 0，`|mean| > sd` 對任何負值自動
# 成立，先驗實驗曾因此產生 24 格假陽性。
N_SEEDS = 5
MIN_SEEDS = 2


@dataclass(frozen=True)
class Cell:
    """一個待執行的格點。`purify` 為 None 表示該段不涉及淨化。"""

    stage: str
    condition: str
    image_id: str
    tau: Optional[float] = None
    purify: Optional[Tuple[str, float]] = None
    seed: Optional[int] = None
    skip_reason: str = ""

    @property
    def skipped(self) -> bool:
        return bool(self.skip_reason)

    def cell_id(self) -> str:
        from src.utils.cellid import cell_id

        parts: Dict[str, object] = {}
        if self.tau is not None:
            parts["tau"] = self.tau
        if self.purify is not None:
            kind, strength = self.purify
            parts["purify"] = kind if kind == "identity" else f"{kind}{strength:g}"
        if self.seed is not None:
            parts["seed"] = self.seed
        return cell_id(self.stage, self.condition, self.image_id, **parts)


def generative_floor_skip(condition: str, tau: float,
                          floor: Optional[float] = None,
                          relative: bool = False) -> str:
    """回傳不適用的理由，適用時回傳空字串。

    抽成函數而非寫在迴圈裡，是為了讓「為什麼這格沒跑」在報表與測試中
    都取得同一個字串，不會出現兩種說法。

    `floor` 是**該影像**實測的 `LPIPS(decode(encode(x)), x)`。給定時用它，
    並在理由中註明是實測值；`None` 時退回 `GENERATIVE_LPIPS_FLOOR` 並註明
    那是名目值。兩者的字串刻意不同：報表上「這格為什麼沒跑」必須看得出
    依據的是量出來的數字還是一個常數。

    `None` 只該出現在乾跑——它要在載入模型之前回答「這次要跑多久」，
    那時沒有任何影像被讀進來。真正執行的路徑由 `run_stage.py` 在
    `build_resources` 之後量測並重建格點。

    `relative=True`（見 `budget_tau_plan`）時恆回傳空字串：該模式下 τ 是
    超出下限的增量，下限本身不再構成不可達。
    """
    if relative:
        # 相對預算軸上 τ 是「超出下限多少」，下限已被減掉，任何正的 τ 都
        # 落在 `build(0)` 之上。此處不是「放寬判準」而是判準不適用。
        return ""
    if condition not in GENERATIVE_CONDITIONS:
        return ""
    if floor is None:
        if tau < GENERATIVE_LPIPS_FLOOR:
            return (f"{condition} 走生成路徑，其 VAE 重建誤差下限名目值為 LPIPS "
                    f"{GENERATIVE_LPIPS_FLOOR}，τ={tau} 結構上不可能達成"
                    "（名目值，未逐影像量測）")
        return ""
    if tau < floor:
        return (f"{condition} 走生成路徑，本影像實測的 VAE 重建誤差下限為 "
                f"LPIPS {floor:.4f}，τ={tau} 結構上不可能達成")
    return ""


def train_cells(images: Sequence[str],
                conditions: Sequence[str] = CONDITIONS) -> List[Cell]:
    """段 1：每條件每影像訓練一次，在最大 τ 上。

    不逐 τ 重訓——那是 ×4 的成本，且「匹配失真」這個前提已四次被證偽
    （同一個 LPIPS 下 PSNR 可差 12 dB）。改由射線縮放一次取得整條曲線。
    """
    return [Cell("train", c, img) for c in conditions for img in images]


def rayscale_cells(images: Sequence[str],
                   conditions: Sequence[str] = CONDITIONS,
                   taus: Sequence[float] = TAUS,
                   floors: Optional[Mapping[str, float]] = None,
                   relative: bool = False) -> List[Cell]:
    """段 2：把訓練好的 φ 沿參數射線縮放到各個 τ。"""
    out = []
    for c in conditions:
        for img in images:
            fl = None if floors is None else floors.get(img)
            for t in taus:
                out.append(Cell("rayscale", c, img, tau=t,
                                skip_reason=generative_floor_skip(
                                    c, t, fl, relative)))
    return out


def purifiers_for(tau: float, include_sweep: bool = True,
                  tau_plan: TauPlan = DEFAULT_TAU_PLAN
                  ) -> List[Tuple[str, float]]:
    """該 τ 要跑哪些淨化設定。

    完整主組只在 `tau_plan.full_purify_taus` 上跑；掃描組只在主表的 τ 上跑。
    其餘 τ 只跑 identity——它們的用途是把失真–效果曲線的低端補起來，
    不是量抗淨化。

    `tau_plan` 預設為 `DEFAULT_TAU_PLAN`，即模組常數；逐批次覆寫見
    `tau_plan_for`。**判準取自 plan 而非模組常數**，否則改了 `--tau-train`
    的批次會在訓練點上一個淨化格都沒有（§4a）。
    """
    if tau not in tau_plan.full_purify_taus:
        return [IDENTITY]
    out = list(MAIN_PURIFIERS)
    if include_sweep and tau == tau_plan.main_tau:
        out += list(SWEEP_PURIFIERS)
    return out


def eval_cells(images: Sequence[str],
               conditions: Sequence[str] = CONDITIONS,
               taus: Sequence[float] = TAUS,
               n_seeds: int = N_SEEDS,
               include_sweep: bool = True,
               floors: Optional[Mapping[str, float]] = None,
               tau_plan: TauPlan = DEFAULT_TAU_PLAN,
               relative: bool = False) -> List[Cell]:
    """段 3：淨化與編輯評測。"""
    if n_seeds < MIN_SEEDS:
        raise ValueError(
            f"n_seeds={n_seeds} 低於下限 {MIN_SEEDS}。n=1 時樣本標準差恆為 0，"
            "`|mean| > sd` 對任何負值自動成立——先驗實驗曾因此產生 24 格假陽性"
        )
    out = []
    for c in conditions:
        for img in images:
            fl = None if floors is None else floors.get(img)
            for t in taus:
                skip = generative_floor_skip(c, t, fl, relative)
                for p in purifiers_for(t, include_sweep, tau_plan):
                    for s in range(n_seeds):
                        out.append(Cell("eval", c, img, tau=t, purify=p,
                                        seed=s, skip_reason=skip))
    return out


def control_cells(images: Sequence[str], taus: Sequence[float] = TAUS,
                  n_seeds: int = N_SEEDS,
                  include_sweep: bool = True,
                  tau_plan: TauPlan = DEFAULT_TAU_PLAN) -> List[Cell]:
    """φ=0 的同淨化對照。**不依賴條件也不依賴 τ**，故跨條件共用。

    每個 `(影像, 淨化, 種子)` 只算一次。若每個條件各算一次，就是 9 倍的
    重複計算——本專案的評測成本主要在這裡。

    τ 只用來決定「哪些淨化設定會被用到」，對照本身與 τ 無關，故最後去重。
    """
    seen = set()
    out = []
    for img in images:
        for t in taus:
            for p in purifiers_for(t, include_sweep, tau_plan):
                for s in range(n_seeds):
                    key = (img, p, s)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Cell("control", "phi0", img, purify=p, seed=s))
    return out


def plan(images: Sequence[str], **kw) -> Dict[str, List[Cell]]:
    """全部四段的格點。供 `run_stage.py` 與儀表板共用同一份定義。

    `floors` 為 `{影像 id: 該圖實測的 VAE 重建下限 LPIPS}`。省略時走名目值，
    只適用於乾跑——理由見 `generative_floor_skip`。

    `tau_plan` 為該批次的預算軸（見 `tau_plan_for`）。**`taus` 由它決定，
    不再各段各自取模組常數**：兩者分開給定時，`rayscale` 與 `eval` 可以列舉
    到不同的 τ 集合，而症狀是段 3 讀不到某個 τ 的 φ，跑到那一格才失敗。
    呼叫端仍可明給 `taus` 覆寫（測試用），但那必須是 `tau_plan.taus` 的子集。
    """
    floors = kw.get("floors")
    tau_plan = kw.get("tau_plan", DEFAULT_TAU_PLAN)
    taus = kw.get("taus", tau_plan.taus)
    extra = [t for t in taus if t not in tau_plan.taus]
    if extra:
        raise ValueError(
            f"taus 的 {extra} 不在 tau_plan.taus={tau_plan.taus} 內。"
            "淨化組的判準取自 tau_plan，多出來的 τ 會只跑 identity 而看不出原因"
        )
    conditions = kw.get("conditions", CONDITIONS)
    n_seeds = kw.get("n_seeds", N_SEEDS)
    include_sweep = kw.get("include_sweep", True)
    return {
        "train": train_cells(images, conditions),
        "rayscale": rayscale_cells(images, conditions, taus, floors,
                                   tau_plan.relative),
        "eval": eval_cells(images, conditions, taus, n_seeds,
                           include_sweep, floors, tau_plan,
                           tau_plan.relative),
        "control": control_cells(images, taus, n_seeds, include_sweep,
                                 tau_plan),
    }


def summarize(plan_dict: Dict[str, List[Cell]]) -> Dict[str, Dict[str, int]]:
    """每段的總數與其中不適用的格數，供排程前確認規模。"""
    return {
        stage: {
            "total": len(cells),
            "skipped": sum(1 for c in cells if c.skipped),
            "runnable": sum(1 for c in cells if not c.skipped),
        }
        for stage, cells in plan_dict.items()
    }
