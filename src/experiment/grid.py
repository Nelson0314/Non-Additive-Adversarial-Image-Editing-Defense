"""四軸格點的列舉 — `docs/DESIGN_2026-08-05.md` §3。

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
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 軸一＋二：訓練條件
# ---------------------------------------------------------------------------

# 三個非加性條件是本專案的方法；五個 baseline 是加性對照（使用者 2026-08-05
# 決定不再自行實作加性方法，由 baseline 擔任該角色）；R 是同失真隨機對照。
NONADDITIVE = ("N1", "N2", "N3")

# `dia_pt` 與 `diffvax` 保留在程式中但**不納入本輪實驗**，各有其原因，
# 兩者都記錄在 `EXCLUDED` 並由測試釘住——移出而不留記錄，等於讓
# 「為什麼少了這一篇」變成無從查考的事。
BASELINES = ("photoguard_c", "mist", "dia_r")
RANDOM_CONTROL = "R"

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

# R 不是選配。先驗實測：在同一可辨失真上，隨機高斯雜訊即取得最佳化解
# 60–74% 的語意失效。沒有這條對照，任何正結果都不可解讀。
CONDITIONS: Tuple[str, ...] = NONADDITIVE + BASELINES + (RANDOM_CONTROL,)

# ---------------------------------------------------------------------------
# 軸三：失真預算
# ---------------------------------------------------------------------------

TAUS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.35)
MAIN_TAU = 0.20          # 主表所在點，見模組 docstring
TRAIN_TAU = 0.35         # 訓練在最大 τ 上做一次，其餘由射線縮放取得
FULL_PURIFY_TAUS: Tuple[float, ...] = (0.20, 0.35)

# 走生成路徑的條件在此下限以下結構上不可能達成（實測 decode(encode(x)) 本身
# 即 LPIPS 0.1434 / PSNR 27.51 dB）。壓縮實驗若把下限降到 0.0716 以下，
# 改這個常數即可讓 τ = 0.10 加入，不需動格點邏輯。
GENERATIVE_LPIPS_FLOOR = 0.1434
GENERATIVE_CONDITIONS = ("N3",)

# ---------------------------------------------------------------------------
# 軸四：淨化
# ---------------------------------------------------------------------------

IDENTITY = ("identity", 0.0)

# 主組：文獻共識的算子。出處見 `docs/SOURCE_AUDIT_2026-08-05.md` §8。
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
SWEEP_PURIFIERS: Tuple[Tuple[str, float], ...] = tuple(
    [("blur", s) for s in (0.5, 1.0, 2.0, 3.0)]
    + [("noise", s) for s in (0.01, 0.02, 0.04, 0.08)]
    + [("quantize", n) for n in (64, 32, 16, 8)]
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


def generative_floor_skip(condition: str, tau: float) -> str:
    """回傳不適用的理由，適用時回傳空字串。

    抽成函數而非寫在迴圈裡，是為了讓「為什麼這格沒跑」在報表與測試中
    都取得同一個字串，不會出現兩種說法。
    """
    if condition in GENERATIVE_CONDITIONS and tau < GENERATIVE_LPIPS_FLOOR:
        return (f"{condition} 走生成路徑，其 VAE 重建誤差下限為 LPIPS "
                f"{GENERATIVE_LPIPS_FLOOR}，τ={tau} 結構上不可能達成")
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
                   taus: Sequence[float] = TAUS) -> List[Cell]:
    """段 2：把訓練好的 φ 沿參數射線縮放到各個 τ。"""
    out = []
    for c in conditions:
        for img in images:
            for t in taus:
                out.append(Cell("rayscale", c, img, tau=t,
                                skip_reason=generative_floor_skip(c, t)))
    return out


def purifiers_for(tau: float, include_sweep: bool = True
                  ) -> List[Tuple[str, float]]:
    """該 τ 要跑哪些淨化設定。

    完整主組只在 `FULL_PURIFY_TAUS` 上跑；掃描組只在主表的 τ 上跑。
    其餘 τ 只跑 identity——它們的用途是把失真–效果曲線的低端補起來，
    不是量抗淨化。
    """
    if tau not in FULL_PURIFY_TAUS:
        return [IDENTITY]
    out = list(MAIN_PURIFIERS)
    if include_sweep and tau == MAIN_TAU:
        out += list(SWEEP_PURIFIERS)
    return out


def eval_cells(images: Sequence[str],
               conditions: Sequence[str] = CONDITIONS,
               taus: Sequence[float] = TAUS,
               n_seeds: int = N_SEEDS,
               include_sweep: bool = True) -> List[Cell]:
    """段 3：淨化與編輯評測。"""
    if n_seeds < MIN_SEEDS:
        raise ValueError(
            f"n_seeds={n_seeds} 低於下限 {MIN_SEEDS}。n=1 時樣本標準差恆為 0，"
            "`|mean| > sd` 對任何負值自動成立——先驗實驗曾因此產生 24 格假陽性"
        )
    out = []
    for c in conditions:
        for img in images:
            for t in taus:
                skip = generative_floor_skip(c, t)
                for p in purifiers_for(t, include_sweep):
                    for s in range(n_seeds):
                        out.append(Cell("eval", c, img, tau=t, purify=p,
                                        seed=s, skip_reason=skip))
    return out


def control_cells(images: Sequence[str], taus: Sequence[float] = TAUS,
                  n_seeds: int = N_SEEDS,
                  include_sweep: bool = True) -> List[Cell]:
    """φ=0 的同淨化對照。**不依賴條件也不依賴 τ**，故跨條件共用。

    每個 `(影像, 淨化, 種子)` 只算一次。若每個條件各算一次，就是 9 倍的
    重複計算——本專案的評測成本主要在這裡。

    τ 只用來決定「哪些淨化設定會被用到」，對照本身與 τ 無關，故最後去重。
    """
    seen = set()
    out = []
    for img in images:
        for t in taus:
            for p in purifiers_for(t, include_sweep):
                for s in range(n_seeds):
                    key = (img, p, s)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Cell("control", "phi0", img, purify=p, seed=s))
    return out


def plan(images: Sequence[str], **kw) -> Dict[str, List[Cell]]:
    """全部四段的格點。供 `run_stage.py` 與儀表板共用同一份定義。"""
    return {
        "train": train_cells(images, kw.get("conditions", CONDITIONS)),
        "rayscale": rayscale_cells(images, kw.get("conditions", CONDITIONS),
                                   kw.get("taus", TAUS)),
        "eval": eval_cells(images, kw.get("conditions", CONDITIONS),
                           kw.get("taus", TAUS), kw.get("n_seeds", N_SEEDS),
                           kw.get("include_sweep", True)),
        "control": control_cells(images, kw.get("taus", TAUS),
                                 kw.get("n_seeds", N_SEEDS),
                                 kw.get("include_sweep", True)),
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
