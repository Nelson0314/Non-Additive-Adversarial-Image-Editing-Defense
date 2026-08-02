# 下一階段：E31 正對照搜尋

<!-- STATUS-BLOCK -->
| | |
|---|---|
| **狀態** | 現行。2026-08-03 改寫，取代 2026-08-02 的「方向需要重新決定」版——方向已定 |
| **設計依據** | `docs/specs/2026-08-02-e31-positive-control.md` |
| **逐步作法** | `docs/plans/2026-08-02-e31.md` |
| **前一輪** | `docs/RESULTS_E29_negative.md` |

> 三份索引：主張查 [`docs/LEDGER.md`](LEDGER.md)、檔案查 [`docs/INDEX.md`](INDEX.md)、
> 比對頁查 [`docs/gallery.html`](gallery.html)。本檔只講**現在要做什麼**，
> 不重複那三份的內容。

---

## 1. 一句話交代現況

E29 在修好量測與校準之後做了第一次實測，**在試過的每一個運作點上防禦都沒有
阻止文字編輯達成 prompt**，加性與非加性皆然。E31 因此不再比較兩臂——兩個零
之間的比較沒有內容——改為在加性基準上尋找**任何一個擋得下編輯的運作點**，
為本專案從未驗證過的量測裝置建立正對照。

---

## 2. E31 在做什麼

沿三個文獻指認的軸掃描 site P（加性）單臂，12 格 × 2 圖：

| 軸 | 值 | 為什麼是這一軸 |
|---|---|---|
| 目標函數 | `untargeted` / `targeted`（灰圖）/ `crossattn:suppress` | `untargeted` 最大化的正是已被判定不對應防禦成功的 `edit_shift`（LEDGER 5.1）。後兩者從未在真實 SD 上產生過資料（LEDGER 5.2） |
| τ_lpips | 0.10 / 0.28 | 現有資料無法分辨「方法無效」與「預算太小」，因為從未在文獻的預算區間量過 |
| strength | 0.5 / 0.3 | 0.5 的全域編輯下輸出主要由 prompt 重新生成，「語意不符」在原理上幾乎不可達成。這很可能是 E29 為負的最大單一原因 |

判準改為 ISR 式的聯集：**語意不符 prompt 或明顯的感知劣化**。E25 之後只取了
前半（LEDGER 1.5）。

---

## 3. 進度

### 已完成（全部在本機，零雲端成本）

| 項 | 產出 | 結果 |
|---|---|---|
| ISR 重判既有 run | `runs/p12_isr_rejudge/` | 828 格語意失敗 0 格。`edit_niqe_*` 自 E2 起就在 CSV 裡而從未被讀過 |
| 預算探針 | `runs/p13_budget_probe/` | **τ=0.28 在原約束集下不可達**，且與防禦方法無關 |
| 逐預算門檻 | `runs/p14_budget_thresholds/` | 兩道次要門檻改為隨預算而定（規格 §12） |
| 劣化階梯 | `runs/p11_degrade_ladder/` | 待使用者判讀 `compare.html` 定出門檻 |
| 本機補產生的編輯來源 | `runs/e31_sources/` | 六張未防禦編輯（既有 run 只有 car_00 一張） |
| 程式改動 | `src/defense/optimize.py`、`src/metrics/suite.py`、`scripts/run_defense.py` | 三個 `defense_mode` 都可跑；擾動 RMS 與尖峰比例進 CSV；`--tau_acut` 可由命令列指定 |

測試：276 passed / 1 skipped、0 failed（基準 253 + 新增 23）。

### 待辦

1. **使用者判讀** `runs/p11_degrade_ladder/compare.html`，定出感知劣化的門檻。
2. **R1**（雲端，約 20 分）。指令可直接複製：

   ```bash
   TA_028=0.1648 TC_028=3.4009 bash scripts/drivers/e31_calibration.sh
   ```

   數值出自 `runs/p14_budget_thresholds/thresholds.csv` 的 budget=0.28 那一列。
   R2 需要的另一組是 budget=0.10 那一列：`TA_010=0.0594 TC_010=1.2861`。
3. **Gate**：R1 的綁定者判定必須全部是 LPIPS hinge，否則不開 R2
   （處置見規格 §8，**不得以放寬約束草率繞過**）。
4. **R2**（雲端，約 1.5–2 小時）：

   ```bash
   LR_028=<R1 定出的值> TA_010=0.0594 TC_010=1.2861 \
     TA_028=0.1648 TC_028=3.4009 bash scripts/drivers/e31_grid.sh
   ```

5. **判定與報告**（本機）：`scripts/e31_report.py --degrade_tau <第 1 項定出的值>`、
   `docs/RESULTS_E31.md`。報告須逐項對照規格 §9 事先寫下的四種預期否定結果。

### 跑之前先知道：strength 這一軸的預期要調整

E31 規格 §5 把 strength 納入網格的理由是「0.5 的全域編輯下語意不符幾乎不可
達成，這很可能是 E29 為負的最大單一原因」。**本機的強度掃描與該推論相反**
（`docs/RESULTS_E31_local.md` §5.2）：site P 的 Δsiglip 在四個 strength 上
都是正的，且在文獻慣用的 0.3 上最大（+0.0524，是 0.5 下 +0.0049 的十倍）。

那是遷移設定（防禦在 0.5 下訓練）、單一影像，不是結論。但網格的 strength=0.3
那幾格若也是負的，就足以**排除**「E29 為負主要是 strength 造成的」這個解釋，
而不是留著當未檢驗的替代說法。這一軸仍然要跑，只是預期不同。

### 一個尚未決定的事項

acut 軸的分離度隨預算塌掉：0.05 時擋下側是通過側的 4.41 倍，0.28 時只剩
**1.18 倍**（`runs/p14_budget_thresholds/separation.csv`）。原因是 τ=0.28 時
`noise`、`warp_bilinear`、`warp_bicubic` 三者的 acut 全部收斂到約 0.16，
只有真正的高斯模糊（0.61）還分得開——位移大到那個量級時插值核心已不重要。

處置的兩個選項與取捨寫在規格 §12.4，**須與使用者討論後再改**。在那之前
driver 用的是現行歸屬的值（較保守的一組），gate 仍然過得了（site P 的解在
τ=0.28 是 acut 0.1302 < 0.1648、chroma 2.8275 < 3.4009）。

---

## 4. 本機與雲端的分工

**線上 GPU 時間只用於必須用它的部分。** 本機是 i5-12500H + RTX 2050 4 GB
（sm 86、torch 2.13.0+cu126），SD v1.4／SigLIP／tiny-SD 權重都在本機 HF 快取。

| 工作 | 在哪跑 | 實測成本 |
|---|---|---|
| 含梯度的 512² 訓練 | **只能雲端** | H100 每步 2.36 s、峰值 10.3 GB（TF32 開） |
| 無梯度的 512² SDEdit | 本機可 | 峰值 4873 MB（靠 Windows 共享記憶體外溢）、單次 222.5 s |
| 指標、判定、報表、比對頁 | 本機 | 秒級到數分鐘 |
| 多臂等 LPIPS 探針 | 本機 GPU | CPU 上約一小時，GPU 上數分鐘 |

`scripts/drivers/local_night.sh` 把本機的 GPU 工作串起來跑。**不要並行**：
兩個 GPU 工作必然互搶 4 GB；CPU 工作與 GPU 工作並行時，CPU 那側的 LPIPS 會把
GPU 工作的 Python 執行緒餓住（實測單張耗時由 222 s 拉長到 30 分鐘以上）。

---

## 5. 雲端環境

- Lightning AI Studio，H100 80GB，torch 2.8.0+cu128，conda env `cloudspace`，
  repo 在 `/teamspace/studios/this_studio/WACV`。
- 背景腳本**不是 login shell**，`PY` 用絕對路徑
  `/home/zeus/miniconda3/envs/cloudspace/bin/python3`。
- 環境準備用 `scripts/drivers/colab_setup.sh`，**不是** `remote_setup.sh`
  （後者會 `pip install torch` 而有換版風險）。
- 該環境的三個坑已在 E29 修掉並寫進 `colab_setup.sh`：numpy 2 與預裝
  pandas／matplotlib／scikit-learn 的 ABI 衝突、`pyiqa` 要 `--no-deps`、
  缺 `sentencepiece` 會讓 SigLIP 起不來（而 SigLIP 是判準）。
- 換機器先跑 `scripts/colab_probe.py`。TF32 開／關的成本差三倍，且會改變
  數值（LEDGER 6.5）。
- 遠端 `git pull` 常因 `runs/` 未追蹤檔衝突而 abort。先把它們 `mv` 到備份
  目錄再 pull。
- 連線資訊與 token 由使用者提供，**不得寫入任何入庫檔案**。

本機環境於 2026-08-03 補裝：`clip-anytorch`、`ftfy`、`wcwidth`、`facexlib`
（`pyiqa` 的 `clipiqa` 與 `topiq_nr` 需要），並把 `setuptools` 由 83 降到 80
以取回 `pkg_resources.packaging`。

---

## 6. 分支

`claude/e20-fidelity-constraint`，**未併入 main**。
未經明確授權不得併入（`CLAUDE.md`）。
