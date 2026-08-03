# 交接 prompt（2026-08-03，對齊基準論文之後）

<!-- STATUS-BLOCK -->
| | |
|---|---|
| **狀態** | 現行。取代同日稍早的「執行 E31 剩餘部分」版——E31 的網格設計已作廢 |
| **用途** | 以下橫線之後的內容可直接貼進新 session |
| **前一版** | `docs/archive/2026-08-01-HANDOFF_PROMPT.md` |

---

WACV 專案接手 — 白盒非加性抗文字編輯防禦

## 這一輪要你做的事

跑完 L1（雲端），判定第一層重現是否通過。**方向已定，設計寫好了，不要重新
發散。** 明確不做的事列在規格 §7，照著守。

## 最重要的一件事先講

**指導者 Ling Lo 是基準論文的第一作者**：Lo, Yeo, Shuai, Cheng,
*Distraction is All You Need: Memory-Efficient Image Immunization against
Diffusion-Based Image Editing*, CVPR 2024。使用者 2026-08-03 指示：他的
**約束、判準、baseline 全部是必要對齊項**；本專案既有的那些一律保留為
**額外改良**，不捨棄。

論證因此是兩層，**順序不可顛倒**：

| 層 | 要回答的 | 現況 |
|---|---|---|
| 第一層：重現 | 在 L∞ ≤ 0.06、N = 100、Table 1 判準下，我們的加性實作是否達到論文水準 | **沒過** |
| 第二層：貢獻 | 非加性在匹配人眼可辨失真下能否勝過該基準 | 第一層過了才有意義 |

第一層沒過的具體數字：落在 κ 上的 `e8_rank_tau0.05`（L∞ 0.0606）是
LPIPS 0.3466／PSNR 24.96，對 PhotoGuard diffusion 的 0.4056／18.26，
PSNR 差 6.7 dB。**在重現落後基準時提出「文獻判準有問題」，會被讀成
「沒調好就怪尺」**，所以那條論證（ρ = −0.207）先壓著，資料留在
`runs/p16_criterion_correlation/`。

## 先讀這些（順序固定）

| # | 檔案 | 為什麼 |
|---|---|---|
| 1 | `docs/NEXT_SESSION.md` | 現況、待辦 L0–L4、本機／雲端分工 |
| 2 | `docs/specs/2026-08-03-lo-aligned-protocol.md` | 設計依據。**§3 必要項與改良項、§7 不做的事**是兩個關鍵 |
| 3 | `CLAUDE.md` | 研究範圍與工作規範 |

查東西用這三份索引，不要從頭讀所有文件：

- `docs/LEDGER.md` —— **主張**的索引。某個結論還算不算數、出處、被什麼推翻。
- `docs/INDEX.md` —— **檔案**的索引。哪份文件現行、哪個 run 屬於哪個實驗。
- `docs/gallery.html` —— **人眼比對頁**的索引。某個門檻當初看哪張圖定的。

文獻脈絡讀 `docs/SURVEY.md`，其 §0 是基準論文的逐項照錄。

## 基準論文的協定（數字是論文寫死的，不可改）

- 保真約束：**只有 L∞ ≤ κ，κ = 0.06**。沒有 LPIPS／鈍化／色度約束。
- PGD sign 更新 + L∞ 硬投影，N = 100 步，timestep T = 10。
- 損失是遮罩內 cross-attention 反應的 L1（式 5）；遮罩由**原圖**的注意力
  二值化（式 4）；注意力是跨層雙三次上採樣後**相加**（式 3）。
- 判準：Table 1 的 PSNR↓ SSIM↓ VIFp↓ FSIM↓ LPIPS↑，全部量「免疫後的編輯
  輸出 vs 原始編輯結果」。沒有語意判準、沒有 ISR。
- baseline：PhotoGuard 的 encoder attack 與 diffusion attack。
- **strength 與 guidance_scale 論文與補充材料都沒公布**——值得直接問指導者，
  比實驗掃描便宜得多。本專案依 E26 一律用 guidance_scale = 7.5。

實作在 `src/defense/linf_attack.py`（Algorithm 1）與 `src/models/attention.py`
（式 3/4/5）。與論文的四處偏離全部寫在那兩個檔的 docstring，包含
Algorithm 1 第 13 行的誤植（字面執行會讓偏移累積成 Σδᵢ 而不受 κ 限制）。

## 已完成

| 項 | 產出 |
|---|---|
| 協定實作 | `linf_attack.py`、`attention.py` 三個新函式、`suite.pairwise` 補 `vif_p`／`fsim` |
| 驅動與報表 | `run_lo_baseline.py`（20 種子平均）、`report_table1.py` |
| 資料集（**L0 完成**） | `data/lo_aligned/`，六類各 4 張共 24 張，全部 CC0 真實照片、512²、出處齊備 |
| 既有資料重判 | `docs/RESULTS_TABLE1.md`，全部 run 在 Table 1 判準下的對照 |
| 本機煙霧測試 | `runs/lo_smoke/`，端到端跑通，抓到一個計算圖共用的 bug |
| κ 的感知代價 | `runs/p17_kappa_visibility/`，含使用者的人眼定錨 |

測試基準：**312 passed / 1 skipped / 0 failed**。

## 待辦（照順序）

| 編號 | 內容 | 在哪跑 |
|---|---|---|
| **L1** | 三個攻擊在 κ = 0.06 上跑完 24 張、20 種子評測 | 雲端，**約 3.3 小時** |
| **L2** | `report_table1.py` 對照 Table 1，判定第一層是否通過 | 本機 |
| **L3** | 同一批 x_adv 加測語意軸與劣化軸（改良判準） | 本機 |
| **L4** | 非加性臂在匹配 LPIPS 下與 L1 的加性解比較 | 雲端 |

```bash
# 上機第一件事：把成本模型的自由度釘死（估時裡的 U ≈ 39 ms 是反推的）
python scripts/colab_probe.py

# L1
python scripts/run_lo_baseline.py --data data/lo_aligned \
    --out runs/lo_baseline --attacks pg_encoder,pg_diffusion,semantic \
    --eval_seeds 20

# L2
python scripts/report_table1.py --out docs/RESULTS_TABLE1.md
```

**L4 之前不做任何其他實驗。**

## 第一層若沒通過，往哪裡查（依可能性排序）

1. **單種子 vs 20 種子**——既有 run 全部單種子。L1 直接消除。
2. **優化器**——我們是 Adam + 平台停止，論文是 PGD sign × 固定 100 步。L1 直接消除。
3. `n_edit = 10` 對論文未公布的 SDEdit 步數。
4. 真實照片 vs 擴散生成影像（論文的資料是生成的）。
5. strength 與 guidance——論文沒公布。

## 兩套約束不可混用

L∞ 球是**必要的對照條件**（加性方法之間可比）；LPIPS 綁定是**貢獻條件**
（含非加性時才可比）。理由：`e15_S_tau0.05`（空間變形）的 L∞ 是 0.5654，
即 κ 的 9.4 倍，而實際位移不到一個像素——**L∞ 量不出非加性方法的可辨失真**，
那正是「匹配人眼可辨失真」這個研究問題的立足點。`report_table1.py` 強制
併列 `L∞` 與 `×κ` 兩欄就是為了讓可比與不可比在表面上看得出來。

## 這個專案最重要的幾條經驗

- **「匹配失真」已經三次被證明是假的**（site S 買模糊、site C 買色調偏移）。
  每引入一個新參數化，先用等 LPIPS 多臂探針量現行約束對它收不收費。
- **不得憑文獻聲譽選指標。** ΔE00、NLPD、VIF 都是被實測推翻的前例。
- **指標矛盾時把影像做成比對頁給人眼判斷。** τ_acut、τ_chroma、E29 的否定
  結果、以及 κ 的可見度都是這樣定或確認的。
- **綁定者診斷是常設步驟**（`scripts/e27_binding_check.py`），已踩過四個假的。
- **判定需要 n ≥ 2。** n = 1 時 `pstdev` 恆為 0，任何判定自動成立；E25 曾
  因此產生 24 格假陽性。
- **推翻自己先前的判斷時，把錯誤的假設與推翻它的資料一起留著**，不要改寫
  成正確版本。`docs/LEDGER.md` §7 就是這麼記的。
- **先量再說。** 本輪三個關鍵發現（τ=0.28 不可達、本機不能跑網格、κ 對應
  LPIPS 0.58）都是零成本的探針量出來的，不是推論出來的。
- **上雲端前先在本機跑一次端到端煙霧測試。** 這一輪就靠它抓到 semantic
  attack 共用 VAE 計算圖的 bug——直接上雲端會浪費一整輪。

## 資料集

24 張全部是 Wikimedia Commons 的 **CC0 真實照片**，六類（man／woman／dog／
cat／horse／bird）各 4 張。**不得用擴散模型生成影像充當資料**——使用者
2026-08-03 明確要求「別用假照片」，理由除立場外還有評測效度：生成影像本身
就在該模型的分布內，拿它評測對真實照片的保護會高估。

需要補圖時：`scripts/fetch_cc0_images.py` 抓候選池，`scripts/prepare_dataset.py`
正規化。注意 Commons 的 User-Agent 必須帶聯絡方式（否則 429），查詢用
`incategory:"CC-Zero"`，且**搜尋排序完全不可信，必須做聯絡表逐張看過再挑**。

## 環境

- 本機：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**），
  i5-12500H + RTX 2050 4 GB。指令前加 `PYTHONIOENCODING=utf-8`——cp950 編
  不了 `²` 這類字元，會在印出結果時才炸，前面的計算全部白做。
- **本機不要並行跑兩個 GPU 工作**，也不要讓 CPU 密集工作與 GPU 工作並行
  （實測單張 SDEdit 由 222 s 拉長到 30 分鐘以上）。
- 本機跑得動**無梯度**的 512² SDEdit（4873 MB、222.5 s），**跑不動任何規模
  可用的訓練**（256² 已是 178 s/step，H100 512² 的 75 倍）。
- 雲端：Lightning AI H100 80GB，環境準備用 `scripts/drivers/colab_setup.sh`
  （**不是** `remote_setup.sh`，後者會 `pip install torch` 而有換版風險）。
- 連線資訊與 token 由使用者提供，**不得寫入任何入庫檔案**。

## 工作要求

- 一律用繁體中文、客觀學術語氣；程式碼關鍵字、函式名、套件名維持英文。
  **commit message 用英文。**
- 動手前先驗證假設（讀檔、跑指令），不要憑記憶猜 API。
- 修改論文方法要記 before/after：具體行號、原貌、原因。
- 架構或實驗設計先提計劃討論再寫程式。
- 宣告完成前必須實際跑過並看到成功輸出；失敗就說失敗。
- **未經明確授權不得併入 main**（目前在 `claude/e20-fidelity-constraint`）。
- 禁止用 try/except 或條件跳過掩蓋症狀，要找根本原因。
- `runs/` 是唯一的證據來源，所有 CSV / JSON / log / PNG / HTML 一律入版控。
  改動 `.gitignore` 的 `runs/` 區塊時必須用 `git status --porcelain --ignored`
  確認沒有結果檔被排除。
