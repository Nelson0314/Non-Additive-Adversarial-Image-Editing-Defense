# 文件與實驗資料索引

最後整理：2026-08-04（L1／L4 跑完並清理設定缺陷之後）。

> **方向已於 2026-08-03 修正。** 指導者 Ling Lo 為 *Distraction is All You
> Need*（CVPR 2024）第一作者，其約束、判準與 baseline 已定為必要對齊項。
> 現行的設計依據是 `docs/specs/2026-08-03-lo-aligned-protocol.md`；
> E31 的 12 格網格設計已作廢，其本機階段的量測結果仍然有效。

本檔的用途是讓接手者在不讀 git log 的情況下，知道**哪一份文件是現行的、
哪一份已被取代、每一個 run 目錄屬於哪個實驗**。

三份索引各司其職，不重複：

| 檔案 | 索引的對象 | 什麼時候查 |
|---|---|---|
| `docs/INDEX.md`（本檔） | **檔案** | 想知道哪份文件現行、哪個 run 目錄屬於哪個實驗 |
| `docs/LEDGER.md` | **主張** | 想知道某個結論還算不算數、出處在哪、被什麼推翻 |
| `docs/gallery.html` | **人眼比對頁** | 想知道某個門檻當初是看哪張圖定出來的 |

每一份 `RESULTS_*.md` 的標題下都有一塊統一的狀態表（狀態／日期／硬體／
承接／後續／資料），可直接 `grep "STATUS-BLOCK" docs/` 列出全部。

---

## 1. 文件

### 現行

| 檔案 | 內容 | 狀態 |
|---|---|---|
| `docs/REPORT_2026-08-04.html` | **對外進度報告**：以「每一組實驗動的是哪一個變因」展開的六節（擾動怎麼產生／加在哪／控制點數／損失讀哪一層／同 LPIPS 下失真的種類／擾動幅度），含兩張逐層畫出的網路結構圖。只收有效批次的數字，設定錯誤與中止的批次不寫進去（那些記在本檔與 LEDGER）。影像已內嵌，可單獨寄出 | **現行。** 由 `docs/REPORT_2026-08-04.src.html` 經 `scripts/build_report_html.py` 產生，**不要直接改產出檔** |
| `docs/APA_fidelity.html` | **APA（arXiv:2506.01511）的防禦圖與原圖差多少**：論文自報的 Table 3、本專案 2026-07-26 在匹配 LPIPS 下實測的 PSNR／L∞，以及兩張六欄比對圖。核心數字是「同一個 LPIPS、PSNR 差 12.70 dB、L∞ 差 28.2 倍」 | 現行。影像取自 `docs/archive/2026-07-26-v2-evidence/figures/` |
| `docs/EVIDENCE_nonadditive.html` | **非加性 vs 加性的證據盤點**：抗淨化的 133 格資料與成對影像、三個損失著力點各跑過什麼、四次「能否打平」比較的完整歷史，以及**缺的實驗與缺的圖**逐項列出 | **現行。** 由 `docs/EVIDENCE_nonadditive.src.html` 產生。性質是盤點不是結論，每一項都標明資料現在還算不算數 |
| `docs/SLIDES_2026-08-04.html` | **簡短版簡報**（10 頁）：架構與 loss 的具體內容、路線一（模組化 SD）擱置的五項發現、路線二（空間變形）的來源文獻與實驗、與 baseline 的對照、統整與後續規劃。**不含加性像素殘差的實驗**，也不收方向不明確的結果 | **現行。** 由 `docs/SLIDES_2026-08-04.src.html` 經 `scripts/build_report_html.py` 產生，**不要直接改產出檔** |
| `docs/EXPLAINER.md` | **從頭到尾的白話說明**：威脅模型、六個注入位置、L1 三個攻擊、L4 兩個位置、評測路徑，全部附架構圖與 `runs/` 的真實影像；後接文獻 survey、方向、困難與規劃 | **現行，接手先讀這份**。只收有效的東西，作廢的實驗只在「已知的坑」一節提一句 |
| `docs/CONVERGENCE.md` | **發散的成因分析與最小變因設計提案**：把三個月來的實驗按「實際在回答什麼」分類（55% 是校準量測裝置）、指出發散是一個十次的遞迴、提出三個收斂動作與一個單變因對照實驗 | **現行。** 性質是**建議不是決定** |
| `docs/LEDGER.md` | **結論總帳**：現行主張、已被推翻的主張（含推翻它的資料）、死路清單、尚未回答的問題 | 現行。每條都附出處 |
| `docs/NEXT_SESSION.md` | 現在在做什麼、下一步是什麼：兩層論證、L0–L4 待辦、兩套約束的分工、H100 成本。**末節是可直接貼進新 session 的交接 prompt**（原 `docs/HANDOFF_PROMPT.md`，2026-08-04 併入） | 現行，接手先讀這份 |
| `docs/SURVEY.md` | 文獻調查，按**問句**組織而非按論文。§0 是**基準論文**（CVPR 2024）的逐項照錄，其餘各節圍繞它 | 現行 |
| `docs/RESULTS_E13-E23.md` | 承接 E13–E23，由四份合併（2026-08-04）：warp vs additive 主網格與 VAE 重建誤差下限拆解（E13–E17）、步數/lr/λ/decoder 掃描與銳利度指標（E18–E19）、保真約束的 paper survey 與四條件等 LPIPS 探針（E20）、新約束下的三個設定重跑與位移上界假設被推翻（E21–E23） | 現行。引用格式不變（「E20 §5.2」指本檔 E20 那一節底下的 5.2） |
| `docs/RESULTS_E25-E31.md` | 承接 E25–E31，由五份合併（2026-08-04）：語意軸重判與**攻擊端缺少 classifier-free guidance 這個根因**（E25–E26）、成本基準與四個誤判的有效約束（E27）、色度偏壓約束與 ΔE 系列全部不合格的判別（E28）、**防禦擋不住編輯**這個否定結果（E29）、本機階段的判準補完與逐預算門檻（E31） | **現行，且是最重要的一份**。E25–E26 那一節使 E2–E23 的每一個 `net_lpips` 失效 |
| `docs/RESULTS_TABLE1.md` | 既有全部 run 在 Table 1 判準下的對照表，由 `scripts/report_table1.py` 生成。分成「可用」與「不可用」兩張表，並附 `ours_lo` 的逐格步數 | 現行。**不要手改** |
| `docs/specs/2026-08-03-lo-aligned-protocol.md` | **對齊基準論文的協定**：方法／約束／判準／baseline 逐項照錄、必要項與改良項的分工、資料集、實驗清單 L0–L4、**明確不做的事** | **現行，且是本階段的設計依據** |
| `docs/specs/2026-07-28-lowrank-residual-defense.md` | 設計規格 v1 | 現行的**設計依據**，但其後的推翻不回頭改寫本文，須與上一份對讀 |
| `docs/OBJECTIVE_HISTORY.md` | `src/defense/objective.py` 五次修訂的 before/after 與各自的實測依據 | 現行。2026-08-04 由該模組的 docstring 原封搬出（原 198 行，佔全檔 39%） |
| `docs/gallery.html` | **人眼比對頁總覽**：把散在 `runs/` 各處的比對頁集中，並記下每一頁當初決定了什麼 | 現行 |
| `docs/NIGHT_LOGS.md` | 兩次夜間自主執行的過程紀錄（2026-07-29 與 2026-08-04），2026-08-04 合併為一檔 | 歷史紀錄，非報告。結論已入 LEDGER；保留的是推導過程 |
| `docs/REPORT.html` | 涵蓋 E0–E12 的報告 | **防禦效果的數字已失效**（E26 的 guidance 缺陷與 E25 的判準問題）。檔案開頭已加註。方法層面的結論仍有效 |
| `docs/architecture.html` | 方法解說：威脅模型、低秩外積參數化、loss 與梯度路徑 | 內容部分過時（注入位置、保真約束、攻擊端的 guidance）。檔案開頭已加註說明，概念部分仍可讀 |

### 已封存（`docs/archive/`）

| 檔案 | 為何封存 |
|---|---|
| `2026-07-26-RESULTS_v2.html` | 前一代研究框架的結果（pg_enc / pg_diff / advdiff / apa / hybrid，以 LPIPS=0.464 為錨）。該路線已整批作廢，改為單一外掛殘差模組。程式碼可由 commit `02cf175` 取回 |
| `2026-07-26-v2-evidence/` | 上一列那份報告的**原始數值來源**（stage0/1/2 的 calibration、fairness、results.csv、manifest 與各階段 config/env 快照，18 檔 0.33 MB）。原本只存在於 TWCC 持久儲存、從未入庫。另有約 152 MB 的逐圖 PNG 留在 `/work`，理由見該目錄的 README |
| `2026-07-29-RESULTS_lowrank.html` | 由 `scripts/make_report.py` 生成，只涵蓋 E0–E3，被 `docs/REPORT.html` 取代。保留是因為 E0 成本表與 E0c 下限表只有這份有 |
| `2026-08-02-e31-positive-control.md` | E31 的設計規格（正對照搜尋、ISR 判準、12 格網格）。**設計部分於 2026-08-03 作廢**，由 `docs/specs/2026-08-03-lo-aligned-protocol.md` 的 L1 取代；2026-08-04 移入封存。§12 的逐預算門檻仍有效，其結論已入 LEDGER 2.9–2.14 與 6.17，量測結果在 `runs/p14_budget_thresholds/` |
| `2026-08-02-e31-plan.md` | E31 的實作計畫（12 個任務，1813 行，原 `docs/plans/`）。Task 1–7 已完成且有效、Task 8–11 隨上一列一起作廢；2026-08-04 移入封存，`docs/plans/` 目錄一併移除 |
| `2026-08-01-HANDOFF_PROMPT.md` | 2026-08-01 交接時給新 session 的 prompt。其描述的狀態早於 E25–E28 |
| `2026-08-03-HANDOFF_PROMPT-L1.md` | 2026-08-03 交接時的版本，L1 跑完前的狀態 |

### 2026-08-04 的文件合併

檔案數由 20 份 `.md` 降到 12 份。**合併一律是原文照搬 + 標題降一級，內容一字未改**，
理由與對應關係如下；引用格式（例如「E20 §5.2」）不變。

| 原檔 | 併入 | 為什麼 |
|---|---|---|
| `RESULTS_E13-E17.md`、`RESULTS_E18-E19.md`、`RESULTS_E20_fidelity.md`、`RESULTS_E21-E22.md` | `RESULTS_E13-E23.md` | 四者處理的是同一件事——在 site S 與 site P 之間建立可用的保真約束並跑主網格，且彼此互相引用 |
| `RESULTS_E25-E26.md`、`RESULTS_E27_calibration.md`、`RESULTS_E28_chroma.md`、`RESULTS_E29_negative.md`、`RESULTS_E31_local.md` | `RESULTS_E25-E31.md` | 五者是同一條線：換掉判準 → 找出攻擊端根因 → 補色度約束 → 得到否定結果 → 在本機補完判準 |
| `NIGHT_RUN_2026-07-29.md`、`NIGHT_2026-08-04.md` | `NIGHT_LOGS.md` | 同一類過程紀錄，結論都已入 LEDGER |
| `HANDOFF_PROMPT.md` | `NEXT_SESSION.md` 末節 | 兩者都在回答「接下來要做什麼」，內容大量重複 |

---

## 2. 實驗資料（`runs/`）

122 個 run 目錄。所有 CSV / JSON / log / PNG 自 commit `1942e38` 起全部入版控
（在那之前 `.gitignore` 有一條規則讓 273 個檔案靜默漏掉，詳見該 commit）。

| run 目錄 | 實驗 | 說明 |
|---|---|---|
| `pilot` | — | 最早的可行性試跑 |
| `e0`, `e0_vae`, `e0b` | E0 / E0b | 成本實測與記憶體歸因。結論：瓶頸是 VAE 激活，非步數 |
| `e0c`, `e0c_tmax` | E0c | site L 的保真度下限。促成 `t_max` 由 999 改為 500 |
| `e0d_L` | E0d | 學習率校準 |
| `e2`, `e2_phi0` | E2 | 主網格 site × rank。`e2_phi0` 是 φ=0 對照，**它推翻了 site L 有防禦這個結論** |
| `e2_la` | E2 | site LA（latent anchored）。該 site 只存在於未合併分支 `claude/anchored-site-l` |
| `e4_scale` | E4 | 注入強度掃描 |
| `e5_la10` | E5 | 三個設定的比較 |
| `e6_stepsP`, `e6_stepsLA` | E6 | 步數掃描 |
| `e7_stepsP_20/100`, `e7_stepsLA_20/100` | E7 | 步數掃描（擴充） |
| `e8_rank_tau0.02/0.05/0.10` | E8 | 全秩 vs 低秩，三個 LPIPS 預算。24 MB，本目錄群最大 |
| `e9_align_probe` | E9 | 對齊容量探測（2 圖、200 步） |
| `e10_eot_all` | E10 | EOT 梯度平均 |
| `e11_wlr_0.001/0.008` | E11 | 權重空間 LoRA 的學習率 |
| `e12_align_L_fixed`, `e12_align_W_r4/r16` | E12 | 對齊階段重跑（修正 PSNR 加權損失後） |
| `e13_slr_0.008/0.03/0.1/0.3` | E13 | site S 學習率。**發現 L∞ hinge 把 site S 完全節流** |
| `e14_P_*`, `e14_S_*` | E14 | 兩個 site 在同一判準下重新校準學習率 |
| `e15_P_tau0.02/0.05/0.10`, `e15_S_*` | E15 | **主網格**。6 圖、未見種子、無淨化、`beta_linf=0` |
| `e16_S_disp3.0/6.0` | E16 | 位移上界。`disp6.0` 只跑完 12 格中的 8 格 |
| `e17_vae_floor` | E17 | VAE 重建誤差下限拆解。四分支：roundtrip / latent_opt / asym_free / asym_leak |
| `e18_lopt_lr0.005`, `e18_lopt_lr0.02` | E18 | latent 最佳化的 lr 與步數。結論：兩者都不是瓶頸 |
| `e19_lam0.1`, `e19_lam1`, `e19_lam10` | E19 | λ × decoder 完全交叉。最佳配置 λ=10 + asym decoder |
| `p1_iso_lpips_probe` | E20 | 四條件等 LPIPS 探針（模糊／雜訊／變形-雙線性／變形-雙三次）。含 `compare.html` 人眼比對頁 |
| `p2_e15_battery` | E20 | 13 項候選指標重判 E15 τ=0.05 的 site S vs P |
| `p3_local_acutance` | E20 | 局部銳利度偏差在四個條件與 E15 上的評分 |
| `p4_constraint_check` | E20 | 新約束的可行域檢查。site S 0/6、site P 6/6 |
| `e21_{Sbic,Sbil,P}_tau*` | E21 | 新約束下的三個設定主網格，9 格 6 圖。site S 的 net 較 E15 掉 67.3% |
| `e21_report` | E21 | 上列的彙整表 |
| `e22_Sbic_d6_tau*` | E22 | bicubic 條件放寬 max_disp 到 6.0 的重跑。**推翻了「被上界綁住」的假設**：解幾乎沒變 |
| `e23_{Sbic,P}_s100_tau0.05` | E23 | 100 步重跑。比值由 1.14× 反轉為 0.85× |
| `e23_Sbic_s100_tau0.10` | E23 | **只完成 3/6 圖**（實驗中止），不可用於比較 |
| `p5_semantic_axis` | E25 | 語意軸重判：33 個 run 的 ΔCLIP/ΔSigLIP、CLIP 未通過對照的紀錄、人眼比對頁 `compare.html` |
| `p6_purify_retention` | E25 | 七對 S/P 配對的逐淨化條件保留率與 S/P 比值 |
| `p7_attack_sanity` | E26 | **guidance scale 掃描**（真實 SD v1.4、CPU）。含 `compare.html`：w=1 與 w=7.5 的未防禦編輯對照 |
| `e27_lrC_*`、`e27b_lrC_*` | E27 | site C 學習率校準的第一、二輪。兩輪都是 `max_dev` 綁住而非 τ |
| `e27c_*` | E27 | 第三輪：放寬 margin。仍未綁住，暴露出原始 `lpips` 項 |
| `e27d_*` | E27 | 第四輪：`alpha_lpips=0`。**τ 終於綁得住**，並定出兩個條件的 lr。含 `compare.html` |
| `e27_evaltiming` | E27 | 單格完整評測的成本量測 |
| `p9_chroma_probe` | E28 | 五條件等 LPIPS 判別：ΔE 系列全部不合格，`local_chroma_bias` 通過 |
| `p10_chroma_ladder` | E28 | 色度偏壓階梯，含 `compare.html`。τ=0.8 由此定出 |
| `p8_site_c_capacity` | E26 | site C 的 `max_dev` 掃描，確認它進得了 τ∈[0.02,0.10] 的運作點 |
| `e29_C_lr0.1/0.3`、`e29_P_lr0.03/0.1` | E29 | 三道約束下的學習率重新校準，τ=0.05、60 步、`--no_eval`。**site C 全部由色度 hinge 綁住** |
| `e29b_C_lr0.15/0.2` | E29 | 補兩個中間學習率。四個值跨 3 倍範圍，有效約束不變 |
| `e29c_C_tau0.10`、`e29c_P_tau0.10` | E29 | 網格會用到的最寬鬆運作點：τ=0.10、上限 150 步、平台停止、含評測。**含 SigLIP 語意軸，即正式判準** |
| `e29_edit_page`、`e29c_edit_page` | E29 | 把防禦圖真的拿去被編輯一次的人眼比對頁。**否定結果的主要證據** |
| `p11_degrade_ladder` | E31 | 感知劣化階梯：四算子各四級 × 四個無參考指標，含人眼比對頁。用以定出 ISR 另一半的門檻 |
| `p12_isr_rejudge` | E31 | 以 ISR 聯集判準重判既有全部 run。828 格語意失敗 0 格；`edit_niqe_*` 從未被讀過這件事在此曝光 |
| `p13_budget_probe` | E31 | 把既有解沿射線放大到各預算，量 acut／chroma／RMS。**含 i.i.d. 白雜訊參照條件**，據此判定 τ=0.28 在原約束集下不可達 |
| `p14_budget_thresholds` | E31 | 逐預算的多條件等 LPIPS 探針，定出每個預算的 τ_acut 與 τ_chroma |
| `e31_strength_sweep` | E31 | 既有防禦圖在 strength 0.2–0.5 下被編輯的響應曲線。**遷移設定不是匹配設定**（防禦圖在 0.5 下訓練），故為下界 |
| `e31_sources` | E31 | 本機產生的六張未防禦編輯輸出（w=7.5、strength=0.5、held-out 種子），供 p11 判讀門檻用。既有 run 只有 car_00 一張 |
| `lo_smoke` | Lo 對齊 | 本機端到端煙霧測試（1 圖、2 步、2 timestep）。**抓到 semantic attack 共用 VAE 計算圖的 bug**，並給出 κ 與感知失真的第一個對照 |
| `p17_kappa_visibility` | Lo 對齊 | κ = 0.06 對應多少感知失真的射線階梯，含 `ladder.png` 人眼比對頁。**κ 對應 LPIPS 0.58**，是本專案跑過最大預算的 5.8 倍 |
| `lo_timing`、`lo_timing_tf32`、`lo_par_1` | L1 | 新機器（RTX PRO 6000 Blackwell 96 GB）上的成本實測與 TF32 取捨。`lo_par_1` 證明並行不是槓桿（吞吐量只增 1.10×）。**5 步的短跑，不是結果** |
| **`lo_baseline`** | **L1** | **本階段的主結果。** 三個攻擊 × 24 張 × 20 種子，72/72 格全部完成。兩個 PhotoGuard 變體 重現（LEDGER 3.15），`semantic` 未重現（3.16）。只跑 `--prompt_index 0` |
| **`ours_lo`** | **L4** | 本專案的 PF 與 S 走 L1 的同一條評測路徑。**只跑到 7 格**（機器時間用盡），且**只有 man_00/PF 與 man_03/PF 兩格可用**——其餘跑滿 150 步上限。site S 的三格三道 hinge 一次都沒啟動（LEDGER 3.23），該條件跑在校準 lr 的 1/12.5 |
| `ours_lo_linfbound` | L4 | **負對照，不可用於任何比較。** 綁定約束錯成 L∞（`beta_linf` 未覆寫，LEDGER 6.12）。site S 停在 `pert_lpips` 0.0034，即預算的 1/29 |
| `ours_lo_earlystop` | L4 | **負對照，不可用於任何比較。** 停止準則在約束仍被違反時觸發（6.13），且 site PF 用了會震盪的 lr 0.03（6.14）。**它的每個數字都比 `ours_lo` 好看**（PF 的編輯 LPIPS 0.4372 對 0.2593），因為它停在 3.3 倍預算的失真上——這正是必須標明的理由 |
| `l3_ours_lo` | L3 | 同一套三類分解套到 `runs/ours_lo/`。產出 3.27：本專案的兩個條件在語意軸上是**負值**（方向正確）、劣化軸也是負值（幾乎無劣化），與兩個 PhotoGuard 變體 的側寫相反 |
| `l3_criterion_axes_working` | L3 | 只留未防禦編輯明顯成功的 18 張後的重判（LEDGER 1.22）。過濾後 `semantic` 的 Δsiglip 由 +0.0001 轉為 −0.0124 |
| `l3_criterion_axes` | **L3** | 三類判準在**有效威脅模型**（w = 7.5）下的重判，n = 1440 列 / 72 格。回答 LEDGER 9.10——先前的 ρ = −0.207 只有 w = 1、n = 217。產出 1.15–1.17 |
| `gate_suppress` | 閘門 | `docs/CONVERGENCE.md` §3 的單變因閘門：`suppress` + site PF，含**同失真的隨機對照條件**。本機 RTX 2050、256²、τ_lpips = 0.55。產出 1.20、1.23 |
| `gate_S` | 閘門 | 同上但跑 site S（非加性）。**本專案第一次在匹配失真上、有隨機對照的加性 vs 非加性比較**。site S 用其校準 lr = 0.1，位移上界放寬到 8.0 px（1.5 px 達不到 τ = 0.55，會讓上界而非 LPIPS 成為有效約束） |
| `gate_S_g128` | **中止** | 容量掃描的第一次嘗試（`grid_size` 128）。啟動後才發現它同時把重取樣核從 `bilinear` 換成 `bicubic`（LEDGER 6.24），對 `gate_S` 而言是兩個變因，05:41 中止於第 10–20 步。**保留為中止紀錄**：它的 `protocol.json` 是 6.24 的證據。不要拿它與 `gate_S` 比較 |
| `gate_S_g128b` | 閘門 | 重跑的容量掃描：與 `gate_S` **只差 `grid_size`**（32 → 128），重取樣核同為 `bilinear`。回答 3.29／3.32 留下的問題——site S 的劣勢是參數化的天花板，還是控制點密度不足 |
| `_smoke_gate`、`_smoke_gate2` | 閘門 | 閘門腳本的煙霧測試（horse_00、2 步、2 種子）。**`_smoke_gate2` 不是可丟棄的暫存**——LEDGER 1.18 的「同 LPIPS 的隨機擾動取得 60% 的語意失效」就是它量到的，是第一次發現該現象的資料 |
| `random_baseline` | (B) | 隨機擾動在各失真上「免費」取得多少免疫效果，512²、不需訓練。**2026-08-04 04:02 為了讓路給 site S 而中止**，只完成第一級的擾動圖，`summary.csv` 尚未產生。可隨時重跑 |
| `figs` | L1／L4／閘門 | 比對頁與大圖。`2026-08-04_matched_lpips_not_matched.png` 是「匹配失真第四次被證明是假的」那張（同 site、同參數化、同 LPIPS，觀感完全不同，LEDGER 1.25）。`compare.html` 是人眼判讀的入口，`2026-08-04_*.png` 是可貼進報告的版本（2026-08-03 那三張的標籤是豆腐字，由 `lo_compare_page.py` 重畫） |
| `logs` | — | 各 run 的 driver 與 stdout 紀錄。含 E29 的 `tf32_probe.log`（TF32 開／關的成本實測）與 E31 的 `e31_local_probe.log`（本機無梯度 SDEdit 的可行性）、`e31_train_probe.log`（本機含梯度訓練的解析度上限），以及 L4 三批的完整訓練日誌 `ours_l2*.log` |

### 涉及已作廢方向的資料

`e8_rank_*` 系列（24 MB）與低秩本身的探討有關，而低秩已於 2026-07-30 明確
列為不深究。資料保留不刪：它是 E8 結論的唯一證據。

`e2_la`、`e5_la10`、`e6_stepsLA`、`e7_stepsLA_*` 使用 site LA，該 site 的
程式碼只存在於分支 `claude/anchored-site-l`，未併入 main。若要重跑須先取回該分支。

---

## 3. 未合併的分支

**未經明確授權不得併入 main。**

| 分支 | 未合併 commit 數 | 內容 |
|---|---|---|
| `claude/anchored-site-l` | 11 | site LA（latent anchored）的實作與 E4–E6 資料 |
| `claude/net-defense-objective` | 1 | 提議在防禦 hinge 內扣除未防禦對照 |

---

## 4. 重跑報告的指令

```bash
python scripts/e13_report.py      # E13 site S 學習率
python scripts/e14_report.py      # E14 兩 site 重新校準
python scripts/e15_report.py      # E15 主網格 + 淨化殘存 + 泛化
python scripts/e16_report.py      # E16 位移上界
python scripts/e18_report.py      # E18/E19 步數、lr、λ 與三指標把關
python scripts/e19_acutance.py    # E19 各條件的銳利度保留率
python scripts/e15_acutance.py    # E15 site S vs site P 的銳利度重判
python scripts/p1_iso_lpips_probe.py  # E20 等 LPIPS 的模糊 vs 雜訊探針（CPU 約 50 分）
python scripts/p1_summary.py      # E20 P1 的判定（只讀 CSV，秒級）
python scripts/p1b_warp_arm.py    # E20 加上兩個空間空間變形位置（CPU 約 20 分）
python scripts/p1_compare_page.py # E20 人眼比對頁
python scripts/p2_e15_battery.py  # E20 候選指標重判 E15（CPU 約 6 分）
python scripts/p3_local_acutance.py   # E20 局部銳利度偏差（秒級）
python scripts/p4_constraint_check.py # E20 新約束的可行域檢查（約 1 分）
python scripts/e21_report.py          # E21 三個設定的比較 + 與 E15 對照
python scripts/e21_disp_saturation.py # E21/E22 位移飽和度診斷（秒級）
python scripts/e22_binding_check.py   # E15/E21/E22 逐格檢查哪道 hinge 真的啟動過（秒級）
python scripts/p5_semantic_axis.py    # E25 語意軸重判（CPU，載 CLIP/SigLIP，約 10 分）
python scripts/p6_purify_retention.py # E25 淨化保留率（只讀 CSV，秒級）
python scripts/p5_compare_page.py     # E25 人眼比對頁：編輯有沒有被擋下來
python scripts/p7_attack_sanity.py    # E26 guidance 掃描（真實 SD v1.4，CPU 約 25 分）
python scripts/p7_compare_page.py     # E26 人眼比對頁：w=1 vs w=7.5
python scripts/p8_site_c_capacity.py  # E26 site C 容量檢查（CPU，約 1 分）
python scripts/e27_binding_check.py runs/e27d_*   # 逐格判定哪道約束綁住（秒級）
python scripts/e27_compare_page.py runs/e27d_C_lr0.3 runs/e27d_P_lr0.03  # E27 防禦圖比對頁
python scripts/e27_report.py --prefix e30   # 主網格彙整（網格跑完後才有資料）
python scripts/p9_chroma_probe.py     # E28 色度約束的候選判別（GPU 約 2 分／CPU 約 25 分）
python scripts/p10_chroma_ladder.py   # E28 色度偏壓階梯 + 人眼比對頁（約 1 分）
python scripts/make_report_figures.py   # docs/figures/*.png（E2/E8/E9/E10/E12）
python scripts/make_report.py     # docs/archive/2026-07-29-RESULTS_lowrank.html（E0–E3）
python scripts/colab_probe.py     # 實測本機器每步／每格評測成本，推算時間
python scripts/e29_edit_page.py runs/<run> --montage_only  # 只重畫比對圖，不需 GPU
# --- E31（全部可在本機跑）---
python scripts/p11_degrade_ladder.py      # 劣化階梯 + 人眼比對頁（GPU 數分鐘）
python scripts/p12_isr_rejudge.py --degrade_tau <值>   # ISR 重判既有 run（秒級）
python scripts/p13_budget_probe.py        # 預算探針 + 白雜訊參照條件（CPU 約 2 分）
python scripts/p14_budget_thresholds.py   # 逐預算門檻（GPU 數分鐘 / CPU 約 1 小時）
python scripts/make_target_gray.py        # 產生 targeted 模式的灰色目標影像
python scripts/e31_make_edits.py --limit 6  # 本機產生未防禦編輯（GPU 約 22 分）
python scripts/e31_local_probe.py         # 本機能否跑無梯度 512² SDEdit
python scripts/e31_train_probe.py         # 本機含梯度訓練的解析度上限
python scripts/e31_report.py --degrade_tau <值>   # E31 網格彙整（該網格已作廢）
bash   scripts/drivers/local_night.sh     # 上列三支 GPU 工作串起來跑，避免互搶顯存
# --- 對齊基準論文（2026-08-03）---
python scripts/fetch_cc0_images.py --out data/_raw --per_class 20     # 動物類 CC0 候選池
python scripts/prepare_dataset.py --src <來源> --dst data/lo_aligned  # 資料集正規化
python scripts/prepare_dataset.py --check     # 只驗證資料集完整性（秒級）
python scripts/run_lo_baseline.py --out runs/lo_baseline --eval_seeds 20  # L1，需雲端
python scripts/report_table1.py               # L2，Table 1 對照表（只讀 CSV，秒級）
bash   scripts/drivers/lo_l1.sh 0             # L1 驅動，0/1 是編輯 prompt 索引
python scripts/run_ours_lo_eval.py --out runs/ours_lo --sites PF,S  # L4，需雲端
bash   scripts/drivers/ours_l2.sh             # L4 驅動
python scripts/lo_compare_page.py             # L1／L4 比對頁與大圖（只讀 PNG，秒級）
# --- 2026-08-04 新增（全部可在本機跑）---
python scripts/l3_criterion_axes.py           # L3 三類判準重判（只讀 CSV，秒級）
python scripts/l4_crossattn_probe.py          # 三個目標函數的本機每步成本（GPU 約 15 分）
python scripts/gate_suppress.py --steps 60    # CONVERGENCE §3 的閘門（GPU 約 1.5 小時/張）
python scripts/random_baseline_curve.py       # 隨機擾動的免費基線曲線（不需訓練）
python scripts/l3_edit_success.py             # 哪些影像的未防禦編輯真的成功（數秒）
python scripts/gate_reeval.py --eval_seeds 20 # 加評測種子重評已存檔的防禦圖（不需訓練）
python scripts/gate_compare.py                # 兩個 site 的「相對隨機優勢」比較（只讀 CSV）
python scripts/ray_curve.py --run runs/gate_suppress --site PF  # 代價-效果曲線（不需訓練）
python scripts/build_report_html.py docs/REPORT_2026-08-04.src.html \
       docs/REPORT_2026-08-04.html      # 把 runs/ 的影像內嵌成單一 HTML（秒級）
```

`random_baseline_curve.py` 補的是一個文獻上沒有的對照。**PhotoGuard 的
Table 6 確實有隨機雜訊對照，但它匹配的是振幅**（「of the same intensity」），
結論是隨機「not effective」；**匹配感知失真的版本沒有人做過**，而兩者差很多
——同振幅時隨機的可辨失真只有最佳化解的 1/2 到 1/3（LEDGER 2.19、1.24）。
本專案匹配 LPIPS 後量到隨機取得最佳化解 60–74% 的語意失效（1.18、1.20）。

`ray_curve.py` 把既有的解沿射線縮放，一次訓練就得到整條「可辨代價 vs
防禦效果」曲線。**兩條曲線只要在某個區間重疊就能比較，不需要任何一次執行
剛好落在指定的點上**——這繞開了「匹配失真」四次被證明是假的那個問題
（LEDGER 1.25、`docs/CONVERGENCE.md` 動作二之二）。

`gate_compare.py` 的比較量是**相對同失真隨機的優勢**
（`Δsiglip(opt) − Δsiglip(rand)`）。這個定義在構造上已對各 site 自己達到的
失真正規化，故**兩個 site 即使停在不同的 LPIPS 上仍可比較**——那是本專案
三個月來反覆撞上的那道牆的一個繞法（見 `docs/CONVERGENCE.md` §3）。

`run_ours_lo_eval.py` 的 `--lr`、`--tau_acut`、`--tau_chroma` 留空即由腳本
依 site 與 LPIPS 預算自己決定。**手動傳單一 `--lr` 就是 `runs/ours_lo/`
失效的原因**（LEDGER 6.16）。

`scripts/e17_vae_floor.py` 需要 GPU 與 SD 權重，不能在本機重跑。
`scripts/e29_edit_page.py` 不加 `--montage_only` 時需要 GPU 與 SD 權重。

實驗驅動在 `scripts/drivers/`（見該目錄的 `README.md`）。其中
`e30_grid.sh` 對應的主網格計畫已由 E29 中止，檔案保留但頂端已加註。
Colab 的完整流程另有 notebook：`notebooks/colab_e29_e30.ipynb`，
其環境與推送的部分仍可用，第 5–7 節對應的計畫同樣已中止。

---

## 5. 待執行的實驗

設計的出處是 `docs/specs/2026-08-03-lo-aligned-protocol.md` §5，
現在要做什麼看 `docs/NEXT_SESSION.md`。

| 編號 | 內容 | 在哪跑 | 前置 | 狀態 |
|---|---|---|---|---|
| **L0** | 備齊資料集影像（人物與動物六類），過 `prepare_dataset.py --check` | 本機 | — | **已完成** |
| **L1** | 三個攻擊在 κ = 0.06 上跑完，20 種子評測 → `runs/lo_baseline/` | 雲端 | L0 | **已完成**，72/72 格。PhotoGuard 兩根重現（LEDGER 3.15），semantic 未重現（3.16） |
| **L2** | `report_table1.py` 對照 Table 1，判定第一層是否通過 | 本機 | L1 | **已完成** → `docs/RESULTS_TABLE1.md` |
| **L3** | 同一批 x_adv 加測語意軸與劣化軸 | 本機 | L1 | 未做 |
| **L4** | 本專案的條件走 L1 的同一條評測路徑 → `runs/ours_lo/` | 雲端 | L2 | **部分且大半無效**。7 格，只有 man_00/PF（48 步）與 man_03/PF（122 步）可用；非加性位置零格可用（LEDGER 3.23） |
| **L4′** | 以修正後的設定重跑 L4：逐 site lr、逐預算 τ_acut／τ_chroma | 雲端 | 無 | **未做，最優先。** 設定已就緒（LEDGER 6.16／6.17） |
| **L5** | 匹配失真的掃描：PhotoGuard 降 κ 或本專案升 τ，讓兩條曲線在同一失真上取值 | 雲端 | L4′ | 未做。使用者 2026-08-03 指示「先不要著重失真匹配」 |

L1 只跑了 `--prompt_index 0`。補充材料 §A 的 Table 1 是兩個編輯 prompt 一起
平均，另一半用 `bash scripts/drivers/lo_l1.sh 1`（約 2–3 小時）。使用者已決定
semantic 改引用論文原數據，故該半邊不是必要項。

### 已中止（保留為證據）

| 腳本 | 為何中止 |
|---|---|
| `drivers/e29_calibration.sh` | 已執行。判準未通過：site C 六個學習率全部由色度 hinge 綁住（E29 §3） |
| `drivers/e30_grid.sh` | 未執行。E29 的否定結果使該網格只會得到「兩個都無效」的 36 格版本（E29 §8） |
| `drivers/e31_calibration.sh`、`drivers/e31_grid.sh` | 未執行。2026-08-03 由 L1 取代——基準論文 Table 1 提供了現成的座標，不需要盲搜運作點 |

### 本機可跑（零雲端成本）

見 §4 的 E31 區塊。`drivers/local_night.sh` 把三支 GPU 工作串起來跑——
本機是 RTX 2050 4 GB，兩個 GPU 工作並行必然互搶，CPU 工作與 GPU 工作並行時
CPU 那側的 LPIPS 會把 GPU 工作的 Python 執行緒餓住（實測單張耗時由 222 s
拉長到 30 分鐘以上）。
