# WACV — 白盒非加性抗文字編輯防禦

在白盒條件（攻擊方使用 stock Stable Diffusion）、外掛模組形式下，尋找非加性
方法，在**匹配人眼可辨失真**下，於**抵抗文字引導編輯**上勝過加性基準。

威脅模型只有一種：**img2img（SDEdit，strength 0.55）**。

正式方向是**紋理重相位**——把影像切成重疊區塊做加窗 FFT，只轉相位、幅度譜
逐位保留，再重疊相加回去。

---

## 現況（2026-08-16）

| 層 | 主張 | 現況 |
|---|---|---|
| **主** | 非加性在**抗淨化**上勝過加性 | **成立**。扣掉空白地板的淨增益上勝加性 9/9、勝 PhotoGuard-c 8/9、勝 DIA-R 9/9、勝隨機相位 9/9；輸 Mist 0/9（Mist 的失真是 3.3 倍）。FND-043 |
| **並列** | 防禦效果本身不輸加性 | **成立**，但**對 PhotoGuard-c 只是打平**（聚合 0.994、逐圖 3/11）。勝 APA 弱 baseline 22/22、勝 DIA-R 20/22、輸 Mist 1/22。FND-045 |
| **三** | 失真受控 | 全部指標照報不挑選。沒有任何單一指標可以當加性與非加性的共用預算軸（FND-035） |

**不再追求語意抵抗。** 「把編輯推遠」做得到，「讓編輯不服從 prompt」做不到
（FND-024／029／030，且 arXiv:2506.04394 獨立測到同一現象）。

基準論文是 Lo, Yeo, Shuai, Cheng, *Distraction is All You Need*, CVPR 2024；
其第一作者是本專案的指導者，故其約束、判準與 baseline 為必要對齊項。

---

## 從哪裡開始讀

| 你想知道 | 讀這個 |
|---|---|
| **新 session 的第一份** | **`docs/START_HERE.txt`**（純文字，十分鐘） |
| 方法是什麼、量到什麼、還缺什麼 | **`docs/PHASE_METHOD.md`**（自足，可獨立讀完） |
| 測得的事實 | `docs/FINDINGS.md`（FND-。末段 FND-037 至 FND-048 是現行結論） |
| 裁決 | `docs/DECISIONS.md`（DEC-） |
| 主線是什麼、程式在哪 | `docs/MAINLINE.md` |
| 上機怎麼跑 | `docs/RUNBOOK.md`（2026-08-16 重寫） |
| 指標的定義與陷阱 | `docs/METRICS.md` |
| 犯過的錯 | `docs/DEFECTS.md` |
| 文獻 | `docs/reference/BIBLIOGRAPHY.md`、`SURVEY_2026-08-16.md` |
| 最新一輪的結果 | `reports/2026-08-16/00_summary.md` |
| 工作規範 | `CLAUDE.md` |

編碼（`FND-`／`DEC-`／`MET-`／`DEF-`）每一筆自足、可單獨讀完，只用來互相指認，
**不代表先後或依賴**。

---

## 程式在哪裡

| 用途 | 路徑 |
|---|---|
| **紋理重相位算子** | **`src/residual/texture_rephase.py`** |
| 參數化 PGD ＋ 預算對齊 | `src/defense/param_pgd.py` |
| 共用損失（encoder-targeted） | `src/baselines/encoder_target.py` |
| baseline 攻擊 | `src/baselines/`（`pgd.py` 為共用骨幹，各篇一檔） |
| APA 弱 baseline 的兩階段 | `src/defense/apa_stage1.py`、`apa_native_stage2.py` |
| 殘差模塊介面 | `src/residual/base.py` |
| 指標 | `src/metrics/suite.py`、`aesthetic.py`、`acutance.py` |
| 淨化算子 | `src/purify/ops.py`（含 C&R 串接） |
| SD 包裝 | `src/models/sd.py` |

`src/residual/base.py` 以「能力」而非型別對外表達：像素側實作 `pixel_residual`，
去噪側實作 `eps_hook`。新增注入位置時提供其一即可，**不要依位置的名稱寫分支**。

### 六支腳本

| 腳本 | 做什麼 |
|---|---|
| `scripts/phase_ablation.py` | 像素臂：`add`／`phase`／`phase_rand` |
| `scripts/apa_baseline.py` | 弱 baseline ＋ 三個加性 baseline |
| `scripts/phase_retention.py` | 抗淨化，含 `--floor` 空白地板 |
| `scripts/phase_distortion_sweep.py` | 失真掃描，供人眼定門檻 |
| `scripts/merge_runs.py` | 併分片 |
| `scripts/report_0816.py` | 產報告 |

---

## 執行

```bash
# 測試（CPU，不需要 SD 權重）。基準：196 passed / 1 xfailed
python -m pytest -q

# 像素臂三條件，人眼門檻
python scripts/phase_ablation.py --out runs/<批次> --data data/lo_aligned \
    --human-threshold

# 外部 baseline
python scripts/apa_baseline.py --out runs/<批次> --data data/lo_aligned \
    --conditions apa_weak mist dia_r photoguard_c

# 抗淨化（跑在已存的防禦圖上，不重跑攻擊）＋ 空白地板
python scripts/phase_retention.py --run runs/<批次> --seeds 3
python scripts/phase_retention.py --run runs/<批次> --seeds 3 --floor \
    --out runs/<批次>/retention_floor.csv

# 報告
python scripts/report_0816.py --out reports/<日期>
```

**上機前先讀 `docs/RUNBOOK.md`**——含機器、成本、分片、git 與四個踩過的坑。

**GPU 工作一律在 NYCU BASIC lab 跑**。本機 RTX 2050 4 GB 跑不動，
只用於寫程式、跑 pytest、看報表。本機直譯器是
`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**）；
指令前加 `PYTHONIOENCODING=utf-8`，否則印中文會炸。

**不要讓 CPU 密集工作與 GPU 工作並行**：實測會把 GPU 工作的 Python 執行緒
餓住，單張 SDEdit 由 222 s 拉長到 30 分鐘以上。

---

## 資料保全

`runs/` 是唯一的證據來源，遠端機器不保證保留，實驗無法重跑。
所有 CSV / JSON / log / PNG / HTML 一律入版控。

`.gitignore` 的 `runs/` 區塊曾有一條 `runs/*/**` 讓 git 停止遞迴而靜默漏掉
273 個檔案（commit `1942e38`）；改動該區塊時必須用

```bash
git status --porcelain --ignored runs | grep "^!!"
```

確認沒有結果檔被排除（輸出為空才對）。

本 repo 用 sparse-checkout（cone mode）。新增頂層目錄或新的 `runs/` 子目錄前
要先 `git sparse-checkout add <路徑>`。

---

## 已刪除的東西

2026-08-13 移除舊主線（`legacy/` 的 33 支腳本、`src/experiment/`、`src/data/`、
`docs/archive/` 等），2026-08-15 移除 inpainting 方向的全部產物。
`runs/` **全部保留**。

取回：`git checkout 6bb656280 -- <path>`，或 `git log` 找刪除前的 commit。

---

## 分支

目前在 `claude/stage3-apa-attn`，**未併入 main**。未經明確授權不得併入。
