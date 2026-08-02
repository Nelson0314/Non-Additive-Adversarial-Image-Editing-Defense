# 文件與實驗資料索引

最後整理：2026-08-01。

本檔的用途是讓接手者在不讀 git log 的情況下，知道**哪一份文件是現行的、
哪一份已被取代、每一個 run 目錄屬於哪個實驗**。

---

## 1. 文件

### 現行

| 檔案 | 內容 | 狀態 |
|---|---|---|
| `docs/REPORT.html` | 涵蓋 E0–E12 的報告 | **防禦效果的數字已失效**（E26 的 guidance 缺陷與 E25 的判準問題）。檔案開頭已加註。方法層面的結論仍有效 |
| `docs/RESULTS_E13-E17.md` | 承接 E13–E17：warp vs additive 主網格、VAE 地板拆解 | 現行 |
| `docs/RESULTS_E18-E19.md` | 承接 E18–E19：步數/lr/λ/decoder 掃描，以及銳利度這項新指標 | 現行 |
| `docs/RESULTS_E21-E22.md` | 承接 E21–E23：新約束下的三臂重跑、位移上界假設被推翻、步數受限的方法問題 | 現行 |
| `docs/RESULTS_E20_fidelity.md` | 承接 E20：保真約束的 paper survey、四臂等 LPIPS 探針、局部銳利度偏差 | 現行 |
| `docs/RESULTS_E25-E26.md` | 承接 E25–E26：語意軸重判（726 格語意失敗 0 格）、淨化保留率、**攻擊端缺少 classifier-free guidance 這個根因**、cross-attention 與 targeted 兩個目標、site C（色度矩陣場） | **現行，且是最重要的一份**。它使 E2–E23 的每一個 `net_lpips` 失效 |
| `docs/RESULTS_E27_calibration.md` | 承接 E27：H100 的成本基準、四個假的綁定者、兩臂的學習率重新校準、**匹配失真第三次被證明是假的（site C 買色調偏移）** | 現行。主網格**未開跑**，開跑前的待辦列在 §5 |
| `docs/RESULTS_E28_chroma.md` | 承接 E28：色度偏壓約束（第三道）、ΔE 全族不合格的判別、τ=0.8 的人眼定錨、TF32 跨機器精度陷阱 | 現行。主網格的設定就緒，列在 §5 |
| `docs/HANDOFF_PROMPT.md` | 可直接貼進新 session 的交接 prompt | 現行 |
| `docs/NEXT_SESSION.md` | 2026-08-01 改寫（E28 之後）：失效的與存活的結論、主網格的完整設定與指令、死路清單 | 現行，接手先讀這份 |
| `docs/specs/2026-07-28-lowrank-residual-defense.md` | 設計規格 v1 | 現行的**設計依據**，但其後的推翻不回頭改寫本文，須與上兩份對讀 |
| `docs/archive/2026-08-01-HANDOFF_PROMPT.md` | 2026-08-01 上一次交接時給新 session 的 prompt | 已封存。其描述的狀態早於 E25–E28 |
| `docs/NIGHT_RUN_2026-07-29.md` | 2026-07-29 夜間自主執行的完整紀錄 | 歷史紀錄，非報告。記錄了 site L 防禦不存在這個否定結果的推導過程 |
| `docs/architecture.html` | 方法解說：威脅模型、低秩外積參數化、loss 與梯度路徑 | 內容部分過時（注入位置、保真約束、攻擊端的 guidance）。檔案開頭已加註說明，概念部分仍可讀 |

### 已封存（`docs/archive/`）

| 檔案 | 為何封存 |
|---|---|
| `2026-07-26-RESULTS_v2.html` | 前一代研究框架的結果（pg_enc / pg_diff / advdiff / apa / hybrid，以 LPIPS=0.464 為錨）。該路線已整批作廢，改為單一外掛殘差模組。程式碼可由 commit `02cf175` 取回 |
| `2026-07-26-v2-evidence/` | 上一列那份報告的**原始數值來源**（stage0/1/2 的 calibration、fairness、results.csv、manifest 與各階段 config/env 快照，18 檔 0.33 MB）。原本只存在於 TWCC 持久儲存、從未入庫。另有約 152 MB 的逐圖 PNG 留在 `/work`，理由見該目錄的 README |
| `2026-07-29-RESULTS_lowrank.html` | 由 `scripts/make_report.py` 生成，只涵蓋 E0–E3，被 `docs/REPORT.html` 取代。保留是因為 E0 成本表與 E0c 地板表只有這份有 |

---

## 2. 實驗資料（`runs/`）

93 個 run 目錄。所有 CSV / JSON / log / PNG 自 commit `1942e38` 起全部入版控
（在那之前 `.gitignore` 有一條規則讓 273 個檔案靜默漏掉，詳見該 commit）。

| run 目錄 | 實驗 | 說明 |
|---|---|---|
| `pilot` | — | 最早的可行性試跑 |
| `e0`, `e0_vae`, `e0b` | E0 / E0b | 成本實測與記憶體歸因。結論：瓶頸是 VAE 激活，非步數 |
| `e0c`, `e0c_tmax` | E0c | site L 的保真地板。促成 `t_max` 由 999 改為 500 |
| `e0d_L` | E0d | 學習率校準 |
| `e2`, `e2_phi0` | E2 | 主網格 site × rank。`e2_phi0` 是 φ=0 對照，**它推翻了 site L 有防禦這個結論** |
| `e2_la` | E2 | site LA（latent anchored）。該 site 只存在於未合併分支 `claude/anchored-site-l` |
| `e4_scale` | E4 | 注入強度掃描 |
| `e5_la10` | E5 | 三臂比較 |
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
| `e17_vae_floor` | E17 | VAE 重建地板拆解。四分支：roundtrip / latent_opt / asym_free / asym_leak |
| `e18_lopt_lr0.005`, `e18_lopt_lr0.02` | E18 | latent 最佳化的 lr 與步數。結論：兩者都不是瓶頸 |
| `e19_lam0.1`, `e19_lam1`, `e19_lam10` | E19 | λ × decoder 完全交叉。最佳配置 λ=10 + asym decoder |
| `p1_iso_lpips_probe` | E20 | 四臂等 LPIPS 探針（模糊／雜訊／變形-雙線性／變形-雙三次）。含 `compare.html` 人眼比對頁 |
| `p2_e15_battery` | E20 | 13 項候選指標重判 E15 τ=0.05 的 site S vs P |
| `p3_local_acutance` | E20 | 局部銳利度偏差在四臂與 E15 上的評分 |
| `p4_constraint_check` | E20 | 新約束的可行域檢查。site S 0/6、site P 6/6 |
| `e21_{Sbic,Sbil,P}_tau*` | E21 | 新約束下的三臂主網格，9 格 6 圖。site S 的 net 較 E15 掉 67.3% |
| `e21_report` | E21 | 上列的彙整表 |
| `e22_Sbic_d6_tau*` | E22 | bicubic 臂放寬 max_disp 到 6.0 的重跑。**推翻了「被上界綁住」的假設**：解幾乎沒變 |
| `e23_{Sbic,P}_s100_tau0.05` | E23 | 100 步重跑。比值由 1.14× 反轉為 0.85× |
| `e23_Sbic_s100_tau0.10` | E23 | **只完成 3/6 圖**（實驗中止），不可用於比較 |
| `p5_semantic_axis` | E25 | 語意軸重判：33 個 run 的 ΔCLIP/ΔSigLIP、CLIP 未通過對照的紀錄、人眼比對頁 `compare.html` |
| `p6_purify_retention` | E25 | 七對 S/P 配對的逐淨化臂保留率與 S/P 比值 |
| `p7_attack_sanity` | E26 | **guidance scale 掃描**（真實 SD v1.4、CPU）。含 `compare.html`：w=1 與 w=7.5 的未防禦編輯對照 |
| `e27_lrC_*`、`e27b_lrC_*` | E27 | site C 學習率校準的第一、二輪。兩輪都是 `max_dev` 綁住而非 τ |
| `e27c_*` | E27 | 第三輪：放寬 margin。仍未綁住，暴露出原始 `lpips` 項 |
| `e27d_*` | E27 | 第四輪：`alpha_lpips=0`。**τ 終於綁得住**，並定出兩臂的 lr。含 `compare.html` |
| `e27_evaltiming` | E27 | 單格完整評測的成本量測 |
| `p9_chroma_probe` | E28 | 五臂等 LPIPS 判別：ΔE 全族不合格，`local_chroma_bias` 通過 |
| `p10_chroma_ladder` | E28 | 色度偏壓階梯，含 `compare.html`。τ=0.8 由此定錨 |
| `p8_site_c_capacity` | E26 | site C 的 `max_dev` 掃描，確認它進得了 τ∈[0.02,0.10] 的運作點 |
| `logs` | — | 各 run 的 driver 與 stdout 紀錄 |

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
python scripts/e19_acutance.py    # E19 各臂的銳利度保留率
python scripts/e15_acutance.py    # E15 site S vs site P 的銳利度重判
python scripts/p1_iso_lpips_probe.py  # E20 等 LPIPS 的模糊 vs 雜訊探針（CPU 約 50 分）
python scripts/p1_summary.py      # E20 P1 的判定（只讀 CSV，秒級）
python scripts/p1b_warp_arm.py    # E20 加上兩個空間變形臂（CPU 約 20 分）
python scripts/p1_compare_page.py # E20 人眼比對頁
python scripts/p2_e15_battery.py  # E20 候選指標重判 E15（CPU 約 6 分）
python scripts/p3_local_acutance.py   # E20 局部銳利度偏差（秒級）
python scripts/p4_constraint_check.py # E20 新約束的可行域檢查（約 1 分）
python scripts/e21_report.py          # E21 三臂比較 + 與 E15 對照
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
python scripts/colab_probe.py     # 實測本機器每步／每格評測成本，推算 E29/E30 時間
```

`scripts/e17_vae_floor.py` 需要 GPU 與 SD 權重，不能在本機重跑。

待執行的實驗驅動在 `scripts/drivers/`（見該目錄的 `README.md`）。Colab 的
完整流程另有 notebook：`notebooks/colab_e29_e30.ipynb`。

---

## 5. 待執行的實驗（需雲端 GPU）

驅動腳本在 `scripts/drivers/`，說明見該目錄的 README，設定的出處見
`docs/NEXT_SESSION.md` §4–§6。順序固定，E29 沒通過綁定者判準就不要開 E30。

| 腳本 | 產生的 run | 成本 |
|---|---|---|
| `drivers/remote_setup.sh` | 無（環境準備） | 約 5 分鐘 |
| `drivers/e29_calibration.sh` | `e29_{C,P}_lr*`（8 格） | 約 20 分鐘 |
| `drivers/e30_grid.sh` | `e30_{C,P}_tau*`（36 格） | 2–4.2 小時 |
