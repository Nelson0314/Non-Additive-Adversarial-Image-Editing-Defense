# 下一階段：對齊 Lo et al. (CVPR 2024) 的協定

<!-- STATUS-BLOCK -->
| | |
|---|---|
| **狀態** | 現行。2026-08-04 改寫：L1／L4 跑完之後的判定 |
| **設計依據** | `docs/specs/2026-08-03-lo-aligned-protocol.md`（§6 已由 §6.1 取代） |
| **基準論文** | Lo, Yeo, Shuai, Cheng, *Distraction is All You Need*, CVPR 2024 |
| **前一輪** | `docs/RESULTS_E25-E31.md`（本機階段，結果仍有效） |
| **白話說明** | `docs/EXPLAINER.md`（含架構圖與真實影像） |

> 三份索引：主張查 [`docs/LEDGER.md`](LEDGER.md)、檔案查 [`docs/INDEX.md`](INDEX.md)、
> 比對頁查 [`docs/gallery.html`](gallery.html)。本檔只講**現在要做什麼**。

---

## 1. 一句話交代現況

指導者 Ling Lo 是基準論文的第一作者，其約束（L∞ ≤ 0.06）、判準（Table 1
五指標）與 baseline（PhotoGuard 兩變體）已定為本專案的必要對齊項。
**L1 已跑完 72/72 格：兩個 PhotoGuard 變體 達到或超過論文公布值**，即我們的基準
實作不是弱化版。**L4 只跑到 7 格機器時間即用盡，且非加性位置零格可用**——
該條件跑在其校準學習率的 1/12.5（LEDGER 6.16），三道約束一次都沒啟動。
當務之急是套用已修好的三項設定後重跑 L4。

---

## 2. 論證的兩層結構

| 層 | 要回答的 | 現況 |
|---|---|---|
| **第一層：重現** | 在 L∞ ≤ 0.06、N = 100、Table 1 判準下，我們的加性實作是否達到論文水準？ | **兩個 PhotoGuard 變體 已達到**（3.15）。`semantic` 未重現，原因是協定只跑了一半，使用者已決定不追（3.16） |
| **第二層：貢獻** | 非加性在匹配人眼可辨失真下能否勝過該基準？ | **尚未成立。** 加性位置兩格可用、非加性位置零格可用（3.23） |

順序不可顛倒。第一層落後基準時，任何對其判準或設定的質疑都會被讀成
「沒調好就怪尺」。兩個 PhotoGuard 變體 重現之後，這道限制對它們已解除；
`semantic` 那一個仍未重現，涉及它的比較維持不提出。

---

## 3. 進度

### 已完成（2026-08-03，全部在本機，零雲端成本）

| 項 | 產出 |
|---|---|
| 論文本文與補充材料的協定逐項抄錄 | 規格 §2 |
| Table 1 五指標補齊 | `src/metrics/suite.py` 加入 `vif_p`、`fsim` |
| 式 (3)(4)(5) 實作 | `src/models/attention.py` 三個新函式 |
| Algorithm 1 實作 | `src/defense/linf_attack.py` |
| 三個基準方法的損失 | 同上：`semantic`／`pg_encoder`／`pg_diffusion` |
| 驅動腳本 | `scripts/run_lo_baseline.py`（含 20 種子平均） |
| 對照報表 | `scripts/report_table1.py` → `docs/RESULTS_TABLE1.md` |
| 資料集規格與正規化工具 | `data/lo_aligned/prompts.yaml`、`scripts/prepare_dataset.py` |
| 既有資料在該判準下的重判 | 規格 §6（**已由 §6.1 取代**：那批全部是 w = 1） |
| 測試 | `tests/test_lo_protocol.py`，27 項 |

### 已完成（2026-08-03，雲端）

| 項 | 產出 |
|---|---|
| **L1** | `runs/lo_baseline/`，72/72 格，24 張 × 3 攻擊 × 20 種子 |
| **L2** | `docs/RESULTS_TABLE1.md`。兩個 PhotoGuard 變體 重現 |
| **L4（部分）** | `runs/ours_lo/`，7 格。另有兩批設定錯誤的負對照 |

### 已完成（2026-08-04，本機）

| 項 | 產出 |
|---|---|
| 設定缺陷的系統性稽核 | LEDGER 6.16–6.20，六個同型缺陷 |
| 逐 site 學習率、逐預算門檻 | `scripts/run_ours_lo_eval.py`，`tests/test_site_config.py` |
| 對照表加上可用性判定 | `scripts/report_table1.py` 分成「可用」與「不可用」兩張表 |
| 人眼比對頁與大圖 | `scripts/lo_compare_page.py` → `runs/figs/compare.html` |
| 白話說明 | `docs/EXPLAINER.md` |

### 待辦

| 編號 | 內容 | 在哪跑 | 前置 |
|---|---|---|---|
| **L4′** | **以修正後的設定重跑 L4。** 逐 site lr、逐預算 τ_acut／τ_chroma，兩個 site 交錯逐影像跑 | 雲端 | 無。設定已就緒 |
| **L3** | 同一批 x_adv 加測語意軸與劣化軸 | 本機 | L1（已有） |
| **L5** | 匹配失真的掃描：PhotoGuard 降 κ 或本專案升 τ | 雲端 | L4′ |

指令：

```bash
# L4′（雲端）。lr 與兩道副門檻現在都由腳本依 site 與預算自己決定，
# 不要再手動傳 --lr——那正是 runs/ours_lo/ 那批失效的原因。
bash scripts/drivers/ours_l2.sh

# 報表與比對頁（本機，秒級）
python scripts/report_table1.py --out docs/RESULTS_TABLE1.md
python scripts/lo_compare_page.py
```

**L5 之前不做任何其他實驗。** 明確排除的清單見規格 §7。

---

## 4. 重跑 L4 之前必須確認的三件事

這三項是 `runs/ours_lo/` 失效的直接原因，全部已修，重跑前確認沒有被
命令列參數蓋回去：

1. **逐 site 的學習率**（LEDGER 6.16）。`--lr` 留空即取 `SITE_LR`；
   傳單一數值會套用到全部 site，那正是原本的錯誤。
2. **逐預算的 τ_acut／τ_chroma**（6.17）。`--tau_acut`／`--tau_chroma`
   留空即依 `--tau_lpips` 查 `runs/p14_budget_thresholds/thresholds.csv`。
3. **訓練期淨化 EOT**（6.20）。這一項**沒有正確答案，是設計決定**：
   基準的三個攻擊不做淨化、評測也不量淨化，故 `--no_purify_train` 讓兩個條件
   在同一個問題上；不加則本專案的條件多背一個評測不回報的目標。

跑完先看 `summary.csv` 的 `steps_done`：跑滿上限的格子依 6.4 不可用於跨
site 比較，而依 6.18，三道 hinge 全程為零的格子**結構上不可能提早停**，
那種「跑滿」不是未收斂而是準則沒有定義。

---

## 5. 兩套約束不可混用

| | L∞ 球（κ = 0.06） | LPIPS 綁定 + 鈍化 + 色度 |
|---|---|---|
| 角色 | **必要的對照條件** | **我們的貢獻條件** |
| 適用 | 加性方法之間的比較 | 含非加性方法的比較 |
| 為什麼不能只用前者 | `e15_S_tau0.05`（空間變形）的 L∞ 是 0.5654，即 κ 的 9.4 倍，而實際位移不到一個像素——L∞ 量不出非加性方法的可辨失真 | — |

`report_table1.py` 強制併列 `L∞` 與 `×κ` 兩欄，就是要讓「哪幾列可以互相
比較」在表面上看得出來。

---

## 6. 本機與雲端的分工

**線上 GPU 時間只用於必須用它的部分。** 本機是 i5-12500H + RTX 2050 4 GB。

| 工作 | 在哪跑 | 實測成本 |
|---|---|---|
| 含梯度的 512² 訓練（含 PGD） | **只能雲端** | 本機 256² 已是 178 s/step，比 H100 的 512² 慢 75 倍 |
| 無梯度的 512² SDEdit（即 L1 的評測段） | 本機可 | 峰值 4873 MB、單次 222.5 s |
| 指標、判定、報表、比對頁 | 本機 | 秒級到數分鐘 |

**不要並行跑兩個 GPU 工作，也不要讓 CPU 密集工作與 GPU 工作並行**（實測單張
SDEdit 由 222 s 被拉長到 30 分鐘以上）。`scripts/drivers/local_night.sh` 因此
把本機工作串起來跑。

---

## 7. 雲端環境

- Lightning AI Studio，H100 80GB，torch 2.8.0+cu128，conda env `cloudspace`，
  repo 在 `/teamspace/studios/this_studio/WACV`。
- 背景腳本**不是 login shell**，`PY` 用絕對路徑
  `/home/zeus/miniconda3/envs/cloudspace/bin/python3`。
- 環境準備用 `scripts/drivers/colab_setup.sh`，**不是** `remote_setup.sh`
  （後者會 `pip install torch` 而有換版風險）。
- 三個已修掉的坑：numpy 2 與預裝 pandas／matplotlib／scikit-learn 的 ABI
  衝突、`pyiqa` 要 `--no-deps`、缺 `sentencepiece` 會讓 SigLIP 起不來。
- 換機器先跑 `scripts/colab_probe.py`。TF32 開／關的成本差三倍且會改變數值。
- 遠端 `git pull` 常因 `runs/` 未追蹤檔衝突而 abort，先 `mv` 到備份目錄。
- 連線資訊與 token 由使用者提供，**不得寫入任何入庫檔案**。

本機環境於 2026-08-03 補裝：`clip-anytorch`、`ftfy`、`wcwidth`、`facexlib`
（`pyiqa` 的 `clipiqa` 與 `topiq_nr` 需要），`setuptools` 由 83 降到 80。
`piq` 0.8.0 已內含 `vif_p` 與 `fsim`，不需新增相依。

---

## 8. 分支

`claude/e20-fidelity-constraint`，**未併入 main**。
未經明確授權不得併入（`CLAUDE.md`）。

---

## 附：可直接貼進新 session 的交接 prompt

（原 `docs/NEXT_SESSION.md`，2026-08-04 併入本檔——兩者都在回答「接下來要做什麼」，內容大量重複。原文一字未改，只把標題降一級。）

### 交接 prompt（2026-08-04 夜，收斂設計提案之後）

<!-- STATUS-BLOCK -->
| | |
|---|---|
| **狀態** | 現行。取代 2026-08-03 夜的「清理與撰寫」版（該輪已完成） |
| **用途** | 以下橫線之後的內容可直接貼進新 session |
| **前一版** | 2026-08-03 夜那一版的內容已全部落實，見 `docs/NIGHT_LOGS.md` |

---

WACV 專案接手 — 白盒非加性抗文字編輯防禦

#### 先讀三份文件，順序不要換

1. **`docs/CONVERGENCE.md`** — 為什麼主題會發散、最小變因的設計提案。
   這是 2026-08-04 的主要產出，也是目前的行動依據。
2. **`docs/LEDGER.md`** — 主張的索引。§7 是已推翻的，§8 是死路清單。
   2026-08-04 新增 31 條：1.15–1.26、3.23–3.27、5.10–5.14、6.16–6.20、7.10。
3. **`docs/INDEX.md`** — 哪份文件現行、哪個 run 目錄屬於哪個實驗。

`docs/EXPLAINER.md` 是從頭到尾的白話說明（含架構圖與真實影像），
需要全貌時讀它。`docs/NIGHT_LOGS.md` 是過程紀錄。

#### 研究問題（未變）

> 在白盒條件、外掛模組形式下，找出**非加性**方法，在**匹配人眼可辨失真**下，
> 於抵抗文字引導編輯上勝過加性基準。

**使用者 2026-08-04 明確指出的問題：主題常常發散、操作的變因過多。**
`docs/CONVERGENCE.md` 是對這一點的回應，接手時請把它當成硬約束而不是建議：
**任何新實驗在動手前，先數一數它有幾個操作變因。超過兩個就先停下來問為什麼。**

#### 現況

| 層 | 內容 | 狀態 |
|---|---|---|
| **第一層：重現** | 在 L∞ ≤ 0.06、N = 100、Table 1 判準上追平基準 | **兩個 PhotoGuard 變體 已達到或超過**（3.15）。`semantic` 未重現，協定只跑了一半，使用者已決定不追（3.16） |
| **第二層：貢獻** | 非加性在匹配可辨失真下勝過加性 | **尚未成立。** L4 的加性位置兩格可用、非加性位置零格可用（3.23） |

#### 這一輪要做的事

##### 一、把閘門的樣本數補足（最高優先，本機約 40 分鐘）

```bash
python scripts/gate_reeval.py --run runs/gate_suppress --eval_seeds 20
python scripts/gate_reeval.py --run runs/gate_S       --eval_seeds 20
python scripts/gate_compare.py --runs runs/gate_suppress,runs/gate_S
```

**不需要重新訓練**——`x_def` 與隨機對照條件的 PNG 都已存檔。

為什麼：閘門量到最佳化條件對同失真隨機條件的配對差是 −0.01487 ± 0.01400、
**Cohen d = −1.06**（大效果），5 個評測種子中 4 個同向。但依該標準差，
α = 0.05 雙尾、power 80% 需要 **n ≈ 7**（LEDGER 1.23）。現在是 5。
**這一步就能把「傾向通過」變成明確判定，而且只是評測成本。**

##### 二、把論文自己的損失接到殘差模塊上（雲端或本機）

`src/defense/linf_attack.py::make_semantic_attack_loss` 完整實作了 Lo et al.
式 (3)(4)(5)，L1 的 `semantic` 那一個就是它。**它只接在 PGD 路徑
（`x_adv = x + δ`）上，沒有接到 `x_adv = G(x; φ)`。**

為什麼要換掉現在用的 `attention_content_suppression`（LEDGER 1.21、5.7(d)）：

| | 論文的版本 | 本專案的變體 |
|---|---|---|
| 對象 | **c_a**（防禦方選的詞） | 攻擊方的編輯 prompt |
| 威脅模型假設 | 較弱 | **較強**（要知道攻擊 prompt） |
| 實測 Δsiglip（horse_00） | **−0.0722** | −0.0567 |

**它同時效果更強、假設更弱。** 這不是新研究，是把兩段已經存在且各自
驗證過的程式接起來。

##### 三、L4′：以修正後的設定重跑（雲端）

```bash
bash scripts/drivers/ours_l2.sh
```

`--lr`、`--tau_acut`、`--tau_chroma` 一律留空由腳本自己決定。
**手動傳單一 `--lr` 正是 `runs/ours_lo/` 失效的原因**（6.16）。

#### 不要做的事

- **不要重新發散方向。** 研究問題已定死。
- **不要在沒有「同失真的隨機對照」的情況下宣稱任何正結果。** 實測隨機擾動
  就取得最佳化解 60–74% 的語意失效（1.18、1.20）。這是 `e2_phi0` 那一課
  （7.1）的重演。
- **不要用 `untargeted` 目標函數。** 它在起點的梯度精確為零（5.10），
  59 個有記錄的 run 全部是它，而它至今能跑起來只是因為淨化 EOT 剛好打破
  對稱（5.11）。
- **不要在跑 GPU 工作時跑 pytest 或其他 CPU 密集工作**（6.8）。
- 不要用 try/except 或條件跳過掩蓋症狀。
- 不要把分支併入 main（目前在 `claude/e20-fidelity-constraint`）。
- **不要刪 `runs/` 底下任何東西。**

#### 環境

- 本機 Python：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**）。
  指令前加 `PYTHONIOENCODING=utf-8`（6.9）。
- 測試基準：**367 passed / 1 skipped**（2026-08-04 新增 21 項）。
- 本機 RTX 2050 4 GB 的每步成本（256²，`runs/logs/l4_crossattn_probe.log`）：
  `untargeted` 177.7 s、`crossattn/suppress` 65.0 s、`encoder` 11.9 s。
  **512² 的 crossattn 會 OOM。**
- 雲端：Lightning AI Studio。連線資訊由使用者提供，**不得寫入任何入庫檔案**。

#### 2026-08-04 找到的、必須記住的四件事

1. **`untargeted` 在起點梯度精確為零**（5.10–5.12）。兩個 site 的起步機制
   因此不同，差三個數量級——加性與非加性的比較從第一步就不對等。
2. **同失真的隨機擾動取得 60–74% 的效果**（1.18、1.20）。而在 Table 1 的
   距離判準上隨機甚至勝過最佳化。
3. **「匹配失真」第四次被證明是假的**，這次兩個條件連參數化都相同：同一個
   LPIPS 上最佳化是漩渦狀紋理、隨機是均勻顆粒，Δniqe 差 6 倍
   （1.25、1.26，圖在 `runs/figs/2026-08-04_matched_lpips_not_matched.png`）。
   **後果對本專案有利**——最佳化的每單位可辨劣化換到的語意失效遠高於隨機。
4. **語意軸挑得出方法的機制，Table 1 的距離軸挑不出來**（1.15）。
   這是本專案最強的判準論證，且完全在基準論文自己的協定上取得。
