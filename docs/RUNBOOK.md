# 上機執行手冊

2026-08-16 重寫。

**舊版（729 行）描述的是 2026-08-13 已刪除的舊主線**——五段流程（calib／train／
rayscale／eval／report）、`scripts/run_stage.py`、`scripts/shard.sh`、
`src/utils/progress.py`、SDXL 1024²、attention 擷取。**它指名的每一支腳本都已經
不存在**，照著做只會得到一連串 FileNotFoundError。取回舊版：

    git show cce89f4ac:docs/RUNBOOK.md

本版只寫**現在真的能跑的東西**：六支腳本、SD v1.4、512²。

---

## 1. 機器

NYCU BASIC lab，兩台各 8 張 RTX 3090（24 GB），home 目錄跨機同步：

    ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-1
    ssh -p 10102 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-2

需要校內網路（VPN）。連不上時兩個 port 都會 timeout。

repo 在 `/nfs/home/nelson0314/WACV-s3`。**它是 `/nfs/home/nelson0314/WACV/.git`
的一個 git worktree**（`.git` 是指標檔不是目錄），這一點在處理 git 狀態時很重要
——見 §6。

### 環境變數要自己給

`~/env.sh` 的 `PYTHONPATH` 指向 `$HOME/WACV`（另一個舊 repo）。**每一支腳本
都要自己明給**：

```bash
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3
export HF_HOME=$HOME/hf_cache
export PYTHONIOENCODING=utf-8
PY=$HOME/venvs/wacv/bin/python
```

DiffPure 權重在 `$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt`。

### 卡是多人共用

跑之前先看：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

`memory.used` 只有 1 MiB 的才算空。本專案每個行程約佔 **10–13 GB**，
一張卡放一個行程。

### 背景執行

```bash
( setsid nohup $PY scripts/xxx.py ... > runs/xxx.log 2>&1 < /dev/null & )
```

`setsid` 與 `< /dev/null` 都不能省，否則 ssh 一斷線就整批帶走。外層的括號
讓 ssh 不等它。

---

## 2. 六支腳本

| 腳本 | 做什麼 | 成本（實測 2026-08-16） |
|---|---|---|
| `phase_distortion_sweep.py` | 把相位與加性的半徑掃一整排，供人眼定門檻 | 分鐘量級 |
| `phase_ablation.py` | 像素臂：`add`／`phase`／`phase_rand` 三條件 | **人眼門檻 45–60 s/圖·條件**；DISTS 對齊要 ×9（二分搜尋） |
| `apa_baseline.py` | 弱 baseline ＋ 三個加性 baseline | 見下表 |
| `phase_retention.py` | 抗淨化：十個算子 × 多 seed，跑在已存的防禦圖上 | **約 660 s/格**（10 算子 × 3 seed） |
| `merge_runs.py` | 把分片的批次併成一個 | CPU，秒 |
| `report_0816.py` | 產四份 HTML 報告 | CPU，分鐘 |

`apa_baseline.py` 逐條件的實測（含兩次 SDEdit 評測）：

| 條件 | 秒/圖 | 備註 |
|---|---|---|
| `mist` | **85** | 必須開 `use_ckpt` 與 `vae_ckpt`，否則 OOM |
| `apa_weak` | **130–150** | 階段一 LoRA 200 步 ＋ 階段二 |
| `dia_r` | **150** | 同上，要開 ckpt |
| `photoguard_c` | **6700–7000** | 200 步／圖，約 **33 s/步**。全批次的瓶頸 |

**`photoguard_c` 佔一批七條件實驗 94% 的機時。** 排程時把它單獨拆出去。

---

## 3. 一個完整批次怎麼跑

### 3.1 像素臂三條件（人眼門檻）

```bash
$PY scripts/phase_ablation.py --out runs/<批次> --data data/lo_aligned \
    --human-threshold
```

24 張圖 × 3 條件約 **45 分鐘一張卡**。

可調的軸（2026-08-16 新增）：

| 旗標 | 意義 |
|---|---|
| `--prompt-index {0,1}` | 用 `prompts.yaml` 的哪個編輯 prompt。0 改主體、1 改場景 |
| `--r-min` | 徑向頻率閘下限。定案 0.12 |
| `--block` | 重疊區塊邊長。定案 32 |
| `--quantile` | 紋理閘的能量參考分位數。定案 0.5 |
| `--phase-radius` | 覆寫人眼門檻的相位半徑 |
| `--target` | 損失的目標影像。定案 `data/targets/gray.png` |
| `--tag-suffix` | 讓同一個 `--out` 下的多組設定不互相覆寫檔名 |

### 3.2 外部 baseline

便宜的三個放一張卡，`photoguard_c` 每張卡分兩張圖：

```bash
# 一張卡，十二張圖，約 75 分鐘
$PY scripts/apa_baseline.py --out runs/<批次>/cheap --data data/lo_aligned \
    --conditions apa_weak mist dia_r --images <12 張>

# 每張卡兩張圖，約 3.7 小時
for g in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$g ( setsid nohup $PY scripts/apa_baseline.py \
      --out runs/<批次>/g$g --data data/lo_aligned \
      --conditions photoguard_c --images <兩張> > runs/pg$g.log 2>&1 < /dev/null & )
done
```

### 3.3 抗淨化

跑在**已存的防禦圖**上，不重跑攻擊：

```bash
$PY scripts/phase_retention.py --run runs/<批次> --data data/lo_aligned --seeds 3
```

空白地板（算子自己造成的位移，2026-08-16 新增）：

```bash
$PY scripts/phase_retention.py --run runs/<批次> --seeds 3 --floor \
    --images <五張> --out runs/<批次>/retention_floor.csv
```

五張圖約 **55 分鐘一張卡**。

### 3.4 報告

```bash
$PY scripts/report_0816.py --out reports/<日期>
```

---

## 4. 分片：同一個目錄只能有一個寫入者

**這不是鎖，是覆寫語意。** `write_csv` 每次呼叫都**整份重寫** `results.csv`。
兩個行程寫同一個 `--out` 會互相蓋掉，而且**不會報錯**。

處置：每張卡給自己的 `--out`，最後用 `merge_runs.py` 併起來。

可分片的軸只有**影像**（`--images`）。條件也可以分（`--conditions`），但要記得
每個分片都會重跑一次評測用的未防禦編輯。

---

## 5. 排程一個晚上的實例（2026-08-16 實跑）

8 張卡、6 小時，實際排法：

| 卡 | 波次一 | 波次二 |
|---|---|---|
| 0–5 | `photoguard_c`，每卡兩張圖（3.7 h） | — |
| 6 | `apa_weak`／`mist`／`dia_r`，十二張圖（75 min） | `--prompt-index 1`，24 張 × 3 條件 |
| 7 | 目標影像消融 ＋ 閘設定掃描（~2 h） | 空白地板（55 min） |

波次二用一支等待腳本接手：

```bash
while pgrep -af "<前一支的識別字串>" | grep -v "本腳本的檔名" | grep -q .; do
  sleep 60
done
```

**`grep -v` 那一段不能省。** 等待腳本的命令列參數裡就含有要等的字串，
直接 `pgrep -f` 會匹配到自己而永遠等下去。

---

## 6. git 與資料保全

### 6.1 runs/ 全部入版控

`runs/` 是唯一的證據來源，遠端機器不保證保留，實驗無法重跑。
所有 CSV／JSON／log／PNG 一律入版控。

`.gitignore` 的 `runs/` 區塊曾有一條 `runs/*/**` 讓 git 停止遞迴而靜默漏掉
273 個檔案（commit `1942e38`）。改動該區塊後必須用

```bash
git status --porcelain --ignored runs | grep "^!!"
```

確認沒有結果檔被排除（輸出為空才對）。

### 6.2 sparse-checkout

本 repo 用 cone mode。新增頂層目錄或新的 `runs/` 子目錄前要先

```bash
git sparse-checkout add runs/<新目錄>
```

否則 `git add` 會被拒絕。

### 6.3 遠端的 pull 常因未追蹤檔而 abort

先把未追蹤的結果檔移到暫存目錄，pull 完再移回：

```bash
git status --porcelain | sed "s/^?? //" > /tmp/untracked.txt
while read -r f; do d="$STASH/$(dirname "$f")"; mkdir -p "$d"; mv "$f" "$d/"; done \
    < /tmp/untracked.txt
git pull --ff-only
cp -rn "$STASH"/. .
```

**用 `mv` 不要用 `rm`。**

### 6.4 遠端推不上 GitHub

機器上沒有 GitHub 認證，`git push` 會停在 `could not read Username`。
在機器上 commit 之後，**從本機把它抓下來再推**：

```bash
git fetch "ssh://nelson0314@server.basiclab.lab.nycu.edu.tw:10102/nfs/home/nelson0314/WACV-s3" \
    claude/stage3-apa-attn
git merge --ff-only FETCH_HEAD
git push origin claude/stage3-apa-attn
```

### 6.5 worktree 的 detached HEAD

repo 是 worktree，`git pull --rebase` 若因缺 identity 而中斷，會留下
**detached HEAD**，而 `git branch -f` 會拒絕（「branch used by worktree」）。
不要用 checkout 硬推（會動到工作區）。用：

```bash
git update-ref refs/heads/<branch> <你要的 commit>
git symbolic-ref HEAD refs/heads/<branch>
```

兩條都不碰工作區。先在機器上設好 identity 可以避免整件事：

```bash
git config user.name "Nelson0314"
git config user.email "nelson.weng20@gmail.com"
```

---

## 7. 其他坑

- **`pkill -f "<字串>"` 會殺到自己。** ssh 送過去的命令列本身含有那個字串，
  pkill 會匹配到執行它的 shell，於是連線直接斷、你也看不到輸出。
  用 PID 殺，或在 pattern 上加排除。
- **測試跑的期間不要改原始碼。** 症狀是隨機幾個測試失敗，重跑就好。
- **本機 RTX 2050 4 GB 跑不動本專案的 GPU 工作**，只用於寫程式、跑 pytest、看報表。
  指令前加 `PYTHONIOENCODING=utf-8`，否則印中文會炸。
- Python 用 `C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**）。
- 測試基準：**196 passed / 1 xfailed**。

---

## 8. 怎麼判讀結果

**判準以人眼為主、數值指標為輔。** 報告的每一格都必須有影像可看；
指標與人眼矛盾時以人眼為準並記錄。

順序：

1. **先確認未防禦的編輯真的成功**（DEC-022）。沒成功的話，抗編輯那一欄的
   分母不成立，該影像整列作廢。
2. 看 `edit_lpips`（防禦後的編輯 對 未防禦的編輯）。越大代表推得越遠。
3. **比值一律報「平均比平均」與逐圖勝場，不報逐圖比值的平均。**
   後者會被分母支配——FND-037（`retention`，r = −0.83）與 FND-039
   （主結果，r = −0.900）是同一個缺陷。
4. 保真度全部照報，不挑選。沒有任何單一指標可以當加性與非加性的
   共用預算軸（FND-035）。
5. 抗淨化要扣掉**空白地板**（§3.3）。地板佔比接近 1 的算子，該列不具鑑別力。
