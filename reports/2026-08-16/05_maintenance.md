# 命名、文件與資料維護紀錄

2026-08-16。本檔記錄本輪對程式與文件所做的非實驗性改動，逐項附上原貌、
新貌與理由。

---

## 1. 命名：字母代號改成看得懂的名字

### 為什麼

原本的注入位置用單一字母代號（site F／L／W／P／E／S／C），以及「A 臂／B 臂」。
這些代號對沒有背過對照表的讀者不傳達任何資訊，而對照表本身散在三份文件裡。

### 改了什麼

**檔名**（`git mv`，歷史保留）：

| 原貌 | 新貌 | 這個模組其實在做什麼 |
|---|---|---|
| `src/residual/site_phase.py` | `src/residual/texture_rephase.py` | 紋理重相位 |
| `src/residual/site_weight.py` | `src/residual/lora_weights.py` | LoRA 權重 |
| `src/residual/site_latent.py` | `src/residual/latent_inject.py` | latent 逐步注入 |
| `src/residual/site_apa.py` | `src/residual/apa_port.py` | APA 的移植 |
| `tests/test_site_phase.py` | `tests/test_texture_rephase.py` | — |
| `docs/SITE_F_RECORD.md` | `docs/PHASE_METHOD.md` | — |

**屬性**：`ResidualModule.site: str = "?"` → `ResidualModule.name: str = "?"`。
子類別的值由字母改成字串：

| 原貌 | 新貌 |
|---|---|
| `site = "F"` | `name = "texture_rephase"` |
| `site = "W"` | `name = "lora_weights"` |
| `site = "L"` | `name = "latent_inject"` |
| `site = "composite"` | `name = "composite"`（不變） |

`.site` 在整個 repo 只有一處被讀（`composite.py:167` 的 repr），故改名安全。

**散文**（文件與註解，共 31 個檔）：

| 原貌 | 新貌 |
|---|---|
| site F | 紋理重相位 |
| site L | latent 逐步注入 |
| site W | LoRA 權重 |
| site E | 文字嵌入 |
| site P | 像素加性 |
| site S | 位移場 |
| site C | 對照條件 |
| A 臂 | 像素臂 |
| B 臂 | latent 臂 |

**沒有改的**：條件字串 `phase`／`add`／`phase_rand`／`photoguard_c` 等。
它們已經是可讀的名字，而且寫在既有的 `runs/*/results.csv` 裡——改了會讓
所有既有資料無法對上。報告層用 `LABEL` 字典把它們顯示成中文。

### 驗證

`python -m pytest -q` → **196 passed / 1 xfailed**，與改名前相同。

### 相關 commit

- `8de62acc9` Name the injection points for what they do
- `9dc83c9ba` Clean up the spacing left by the mechanical rename

---

## 2. 文件：刪掉一份會誤導人的手冊

### `docs/RUNBOOK.md`（729 行 → 258 行）

舊版寫的是 **2026-08-13 已刪除的舊主線**：五段流程（calib／train／rayscale／
eval／report）、SDXL 1024²、attention 擷取、τ 掃描。

它指名的**每一支腳本都已經不存在**：

| 舊版指名 | 現況 |
|---|---|
| `scripts/run_stage.py` | 已刪 |
| `scripts/shard.sh` | 已刪 |
| `scripts/dashboard.py` | 已刪 |
| `scripts/fetch_diffpure.py` | 已刪 |
| `scripts/verify_gpu_env.py` | 已刪 |
| `src/utils/progress.py` | 已刪 |

也就是說，一個新 session 照著它做，得到的只會是一連串 `FileNotFoundError`。
這比沒有手冊更糟。

新版只寫現在真的能跑的六支腳本，並補上本輪實測的成本與四個新踩到的坑
（見 §4）。舊版可用 `git show cce89f4ac:docs/RUNBOOK.md` 取回。

### 沒有刪的

- **`runs/` 一個都沒刪。** CLAUDE.md 明定它是唯一的證據來源、實驗無法重跑。
  61 個批次目錄中有相當多來自已放棄的方向（`apa_*`、`s3*`、`ip*`、`ca_probe`
  等），但那是判準，不是我能自行裁決的事。
- `docs/reference/` 的六份查證紀錄（合計約 4,000 行）全部保留。它們是外部
  原始碼與論文的查證憑據，重查的成本遠高於保存的成本。

---

## 3. 資料保全：找回 625 個從未入版控的證據檔

### 發現

遠端機器上 `git status` 有 **1,892 個未追蹤檔**。分類後：

| 目錄 | 檔數 | 處置 |
|---|---|---|
| `runs/hb5` | 625 | **必須保留**——十算子 retention 表背後的淨化圖、編輯圖與逐圖 log |
| `runs/ip2`／`ip3`／`ip5`／`ip5_param`／`ip_pre` | 2,040 | inpainting，2026-08-15 已明確放棄，留在暫存區不入版控 |

本機 git 只有 80 個 `runs/hb5/purified/` 檔，遠端有 625 個。
**差額 545 個檔案從來沒有進過版控。**

### 處置

1. pull 之前把未追蹤檔 `mv`（不是 `rm`）到 `/nfs/home/nelson0314/_stash_pull_0816`
2. pull 完把 `runs/hb5/` 的部分 `cp -rn` 回去
3. 在機器上 commit（838 個檔，含本輪新產生的批次）
4. 機器沒有 GitHub 認證，故從本機 `git fetch ssh://…` 抓下來再 push

現在 `git ls-files runs/hb5 | wc -l` = **907**（原本 282）。

### 順帶確認

`git status --porcelain --ignored runs/hb5 | grep "^!!"` 輸出為空——
沒有結果檔被 `.gitignore` 靜默排除（那是 commit `1942e38` 踩過的坑）。

### 相關 commit

- `d80e95635` Bring the hb5 purification outputs under version control

---

## 4. 本輪新踩到的四個坑（已寫進新版 RUNBOOK §6–7）

### 4.1 `pkill -f` 會殺到自己

透過 ssh 送 `pkill -f "runs/pidx1"` 時，**執行這條指令的 shell 自己的命令列
就含有 `runs/pidx1`**，於是 pkill 把自己殺掉，連線直接斷、看不到任何輸出。
發生了兩次。處置：用 PID 殺，或在 pattern 上加排除。

### 4.2 等待腳本會等自己

波次二的等待迴圈原本是

```bash
while pgrep -f "$WAIT_FOR" > /dev/null; do sleep 60; done
```

而 `$WAIT_FOR` 是這支腳本自己的第一個參數，於是 `pgrep -f` 匹配到自己，
**永遠等下去**。處置：

```bash
while pgrep -af "$WAIT_FOR" | grep -v "wave2.sh" | grep -q .; do sleep 60; done
```

### 4.3 pattern 要匹配得到才有意義

另一個版本用了 `"apa_baseline.py --out runs/ext24/g6"`，但實際的命令列是
`python scripts/apa_baseline.py --data … --out runs/ext24/g6`——`--out` 不在
`apa_baseline.py` 後面。pattern 匹配不到，等待迴圈**立刻放行**，兩個行程
擠上同一張卡。處置：用不含順序假設的子字串（`runs/ext24/g6`）。

### 4.4 worktree 的 detached HEAD

遠端 repo 是 `/nfs/home/nelson0314/WACV/.git` 的一個 worktree。
`git pull --rebase` 因為缺 git identity 而中斷，留下 **detached HEAD**；
此時 `git branch -f` 會拒絕（「branch used by worktree」）。

不要用 `git checkout` 硬推（會動到工作區的 900 多個檔）。用不碰工作區的兩條：

```bash
git update-ref refs/heads/<branch> <commit>
git symbolic-ref HEAD refs/heads/<branch>
```

事前在機器上設好 `git config user.name` / `user.email` 可以避免整件事。

---

## 5. 程式：三處新增的能力

| 檔 | 新增 | 為什麼 |
|---|---|---|
| `scripts/apa_baseline.py` | `--prompt-index` | `prompts.yaml` 每類有兩個編輯 prompt，**第二個從來沒有被評測過** |
| `scripts/phase_ablation.py` | `--prompt-index`／`--block`／`--r-min`／`--quantile`／`--phase-radius`／`--tag-suffix` | 三個閘設定原本寫死在 `build()`，不改原始碼就掃不了 |
| `scripts/phase_retention.py` | `--floor` | 量淨化算子**自己**造成的位移。這是專案文件列的第一優先缺口 |
| `scripts/report_0816.py` | 新檔 | 產四份 HTML 報告 |

`results.csv` 因此多了六欄：`prompt_index`、`prompt`、`block`、`r_min`、
`quantile`、`target_image`。既有的批次沒有這些欄，報告端用
`num(row, key, default=None)` 讀，缺欄回傳 `None` 而不是丟例外——
合併不同時期的批次時欄位本來就不齊。

### 相關 commit

- `ab15b2ffa` Expose prompt choice and gate settings on the drivers
- `0669023ba` Measure the displacement floor a purifier causes on its own
- `cce89f4ac` Generate the 2026-08-16 batch reports

---

## 6. 文獻：索引檔漏收了本方法最依賴的兩篇

`docs/reference/BIBLIOGRAPHY.md` 自稱收錄「引用過的**全部**文獻」，
但以下五筆在程式與文件中被實際引用、卻不在索引裡：

| 論文 | 在本專案的角色 |
|---|---|
| Galerne, Gousseau, Morel（IEEE TIP 2011） | **紋理重相位的構造來源** |
| Ding et al.（DISTS, TPAMI 2021） | 預算軸與半個機制假設 |
| Madry et al.（PGD, ICLR 2018） | `param_pgd.py` 的更新式 |
| Oppenheim & Lim（Proc. IEEE 1981） | 本方法必須處理的矛盾 |
| 結構張量（Förstner／Harris／Bigün） | 紋理閘的 coherence |

已補進新的 §2b，另加本輪查到的五筆。查證細節與新穎性主張的收窄見
`docs/reference/SURVEY_2026-08-16.md`。

---

## 7. 波次中途又踩到的兩件事（2026-08-16 稍晚）

### 7.1 `git stash -u -- runs` 會把跑到一半的結果吞掉

為了讓遠端能 `git pull`，用了

```bash
git stash -q -u -- runs && git pull --ff-only && git stash pop
```

但那次 pull 因為分支已經分岔而 **abort**，於是 `stash pop` 沒有執行，
`runs/theta/` 六個批次目錄（30 列結果、120 張 PNG）就留在 stash 裡、
從工作區消失。後來 `git stash pop` 把它們救回來了，但 pop 本身也部分失敗
（幾個 log 檔已重新存在），stash entry 至今保留著沒有丟棄。

**處置**：遠端不要再用 `git stash` 同步。要更新遠端的程式而不動 HEAD，用

```bash
git fetch origin
git checkout origin/<branch> -- scripts docs
```

這條只覆寫指定路徑，完全不碰 `runs/`，也不移動 HEAD。

### 7.2 遠端與 origin 分岔之後不能 ff

遠端自己 commit 了批次產物（機器上沒有 GitHub 認證，推不上去），
而本機又繼續往 origin 推，於是兩邊分岔，`git pull --ff-only` 一律 abort。

**處置**：遠端的 commit 用 §4.4 的方式從本機 fetch 下來再推；
遠端本身只用 `git checkout origin/<branch> -- <路徑>` 取程式，不做 merge。

---

## 8. 一個刻意沒有做的事：沒有新增影像

使用者提到「再找稍微多一點的圖片」。本輪**沒有**新增影像，理由是：

1. `data/lo_aligned` 的 24 張與 `data/_selected` 的 24 張候選是 **1:1 的**，
   沒有現成的備用影像。新增就要重新從 Wikimedia Commons 找 CC0 真實照片、
   逐張查證授權、寫 `provenance.json`。
2. 更關鍵的是**新影像跑不動外部比較**。`photoguard_c` 是 1.9 小時一張圖，
   八張卡一整晚也只能加十幾張。新影像只會拿到便宜的三個條件，
   對「跟外部方法比較」這個目標幫助有限。

改成把**既有 24 張的另一個編輯 prompt** 跑完（`prompts.yaml` 每類都有兩個，
第二個從來沒被評測過），等於在不新增影像的前提下把評測情境從一種變成兩種。
結果見 FND-044：兩個 prompt 上的結論一致，第二個的 margin 略寬。

要新增影像的話，建議的順序是：先補完 `photoguard_c` 在既有 24 張上的覆蓋
（目前 11 張），再談擴充資料集。
