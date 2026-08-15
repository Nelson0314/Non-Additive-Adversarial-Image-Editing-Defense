# WACV — 白盒非加性抗文字編輯防禦

## 研究範圍

唯一必要目標（使用者於 2026-07-30 明確界定，先前的發散方向已作廢）：

> 在白盒條件（已知攻擊方使用 stock Stable Diffusion）、外掛模組形式下，
> 找出非加性方法，在**匹配人眼可辨失真**下，於**抵抗文字引導編輯**上
> 勝過加性基準。

明確不是重點：

- **low rank 完全不深究。**
- 架構不設限，只需維持外掛模組的形態。

要多做 paper survey；不要在會失敗的方向持續深究。

**2026-08-13 主張階層改版**（使用者定案）：

| 層級 | 主張 | 讀數 |
|---|---|---|
| **主** | 非加性**更**抗淨化<br>**讀數已知有缺陷**：`retention` 比值被分母支配（r=−0.83，FND-037）。<br>DISTS 對齊下勝 9/10（FND-033，分母只差 7.7% 故可用）；人眼門檻下分母差 43%，比值不可解讀。<br>**改報淨化後的絕對位移量：勝加性 10/10**（FND-037） | `effect(purify(·))` |
| **並列** | 防禦效果本身不輸：未淨化時的位移量不低於加性 baseline<br>**已成立**：人眼門檻上 1.55×、逐圖 22/24（FND-035）；五張圖 1.41×（FND-036）。<br>**但與原生預算的 `photoguard_c` 打平**（0.99×，FND-036） | `effect(·, identity)` |
| 三 | 保真受控，報全部指標不挑選 | LPIPS／DISTS／PSNR／SSIM／NIQE／銳利度 |

**不再追求語意抵抗。** CLIP-T 對齊掉幅仍照報，但不作為成敗判準——四個軸全部
否證（FND-024／029／030），且 arXiv:2506.04394（ICIP 2025）獨立測到同一現象。

**兩個讀數都要報**：`retention`（主）與 `effect(·, identity)`（並列）。對照組 R
的定義是**同失真的加性隨機**；site F 另有 `phase_rand`（同失真隨機相位）。
`retention` 的分母塌陷時不可解讀——`phase_rand` 的 1.4–3.4 是分母只有三分之一
造成的假象，不是它比較強（FND-033）。

**判準以人眼為主、數值指標為輔。** `compare.html` 是主要產出物，每一格都必須
有影像可看；指標與人眼矛盾時以人眼為準並記錄。

## 正式方向：site F（紋理重相位）

2026-08-13 使用者裁定。**專案範圍收斂到三個條件，沒有第四個**：

| 條件 | 是什麼 |
|---|---|
| **弱 baseline** | 完全原生 APA，只把 reward 換成 targeted output（DEC-023） |
| **強 baseline** | `photoguard_c`／`mist`／`dia_r` 三個已發表的加性方法 |
| **site F** | **紋理重相位**，`src/residual/site_phase.py` |

`add`（加性 δ 走同一個 encoder-targeted 損失）與 `phase_rand`（同失真隨機相位，
即 RPN）是 site F 消融的內部對照，不是獨立條件。

site F 的構造與定案參數見 `docs/MAINLINE.md` §4 與
`docs/superpowers/specs/2026-08-13-texture-rephasing-design.md`。一句話：把影像
切成重疊區塊做加窗 FFT，**只轉相位、幅度譜逐位保留**，再重疊相加回去；`θ=0`
時輸出逐位元等於原圖。文獻依據是 Random Phase Noise（Galerne et al., TIP 2011）。

**舊主線已刪除。** 2026-08-13 移除 `legacy/`（33 支腳本）、`docs/archive/`、
`src/experiment/`、`src/data/`、`src/defense/` 的四支舊模組、`src/metrics/` 的五支、
`src/residual/site_warp.py`／`site_embedding.py`、`src/models/attention.py`、
`src/utils/` 的三支，以及 21 支對應測試。`runs/` **全部保留**——它是唯一的證據
來源，實驗無法重跑。取回：`git checkout 6bb656280 -- <path>`。

已測過並否決的方向（注意力抑制／分類器 CE／latent／CLIP 四種 reward、DISTS 進
loss 的軟約束、Adam 更新規則、位移場、cross-attention 注入、分階段注入、
amortized generator、顏色通道、site F 搬進 latent）結論留在 FND-004、
FND-023…034，不要重試。

## 程式位置

**完整清單見 `docs/MAINLINE.md` §3。** 六支腳本、35 支 src。最常用的：

| 用途 | 路徑 |
|---|---|
| **site F 算子** | **`src/residual/site_phase.py`** |
| 參數化 PGD ＋ 預算對齊 | `src/defense/param_pgd.py` |
| A 臂消融驅動 | `scripts/phase_ablation.py` |
| 失真掃描（定門檻） | `scripts/phase_distortion_sweep.py` |
| 抗淨化 retention | `scripts/phase_retention.py` |
| 弱／強 baseline 驅動 | `scripts/apa_baseline.py` |
| 階段一／階段二 | `src/defense/apa_stage1.py`、`apa_native_stage2.py` |
| 指標 | `src/metrics/suite.py`、`aesthetic.py` |
| 淨化算子 | `src/purify/ops.py`（含 C&R 串接 `jpeg_then_resize`） |

`src/residual/base.py` 以「能力」而非型別對外表達：像素側實作 `pixel_residual`,
去噪側實作 `eps_hook`。新增位置時提供其一即可，**不要依 site 名稱寫分支**。

## 文件

新 session 先讀 `docs/START_HERE.txt`，接著三份：

| 檔 | 回答什麼 |
|---|---|
| `docs/MAINLINE.md` | 主線是什麼、程式在哪、已知什麼 |
| `docs/FINDINGS.md`／`DECISIONS.md` | 測得的事實與裁決，**判準一律以這兩份為準** |

外部論文的查證紀錄在 `docs/reference/`：`BIBLIOGRAPHY.md` 是全部文獻與網址的
分類索引，`ROBUSTNESS_TESTS.md` 是三份抗淨化檢定協定的精確設定。
**`docs/archive/` 已於 2026-08-13 刪除**，仍然載重的六條舊 FND
（004／008／013／018／019／020）已逐字升到 `FINDINGS.md` 末段。
編碼（`FND-`／`DEC-`／`MET-`／`DEF-`）每一筆自足、可單獨讀完，只用來互相
指認，**不代表先後或依賴**。

## 環境

- 本機 Python：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**，base 沒有 pytest）。
- 測試：`python -m pytest -q`，基準為 **196 passed / 1 xfailed**（2026-08-15 起）。
  xfailed 是刻意釘住的 DIA-PT L1 起點缺陷（原始碼自身的問題，`strict=True`）。
- **GPU 工作一律在 NYCU BASIC lab 跑**（兩台各 8 張 RTX 3090，home 目錄跨機同步）：

      ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-1
      ssh -p 10102 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-2
      source ~/env.sh        # PATH／venv／HF_HOME，並 cd 到 repo

  repo 在 `/nfs/home/nelson0314/WACV-s3`。金鑰認證已設好，**密碼與 token
  不得寫入任何入庫檔案**。卡是多人共用，跑之前先看 `nvidia-smi`。
- 本機 RTX 2050 4GB 跑不動本專案的 GPU 工作，只用於寫程式、跑 pytest、看報表。
- 遠端也用 sparse-checkout。`git pull` 之後若某個頂層目錄沒出現，
  先 `git sparse-checkout add <目錄>`。
- 遠端 `git pull` 常因 `runs/` 未追蹤檔衝突而 abort，先把它們 `mv` 到
  暫存目錄再 pull。

## 工作要求

- 一律用繁體中文回答，客觀學術語氣；程式碼關鍵字、函式名、套件名維持英文。
  **commit message 用英文。**
- 動手前先驗證假設（讀檔、跑指令），不要憑記憶猜 API。
- 修改論文方法要記 before/after：具體行號、原貌、原因。
- 架構或實驗設計需先提計劃討論再寫程式。環境問題直接修掉，不用寫進報告。
- 宣告完成前必須實際跑過並看到成功輸出；失敗就直說失敗。
- **未經明確授權不得把分支併入 main。**
- 禁止用 try/except 或條件跳過來掩蓋症狀，要找根本原因。

## 資料保全

`runs/` 是唯一的證據來源，遠端機器不保證保留，實驗無法重跑。所有 CSV / JSON /
log / PNG 一律入版控。`.gitignore` 的 `runs/` 區塊曾有一條 `runs/*/**` 讓 git
停止遞迴而靜默漏掉 273 個檔案（見 commit `1942e38`）；改動該區塊時必須用
`git status --porcelain --ignored` 確認沒有結果檔被排除。
