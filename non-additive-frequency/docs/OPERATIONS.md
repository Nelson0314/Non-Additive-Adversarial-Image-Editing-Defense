# 環境與操作

## 本機

Python：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（**不是 base**，
base 沒有 pytest）。

```
python -m pytest -q
```

基準 **1141 passed / 2 skipped / 1 xfailed**。

- 裝了 `lpips` 套件的機器會少一項（`test_impress_未安裝_lpips_套件時不得靜默
  改用他者` 只能在缺該套件時驗證）。遠端因 IMPRESS 的預設後端就是它而必定裝有。
- xfailed 是刻意釘住的 DIA-PT L1 起點缺陷（原始碼自身的問題，`strict=True`）。

本機 GPU（RTX 2050 4 GB）跑不動本專案的 GPU 工作，只用於寫程式、跑測試、
看報表。

**Git Bash 裡不要用裸 `python`**，會卡到逾時；用 conda 的完整路徑。

## 遠端

GPU 工作一律在 NYCU BASIC lab 跑，兩台各 8 張 RTX 3090，home 目錄跨機同步。

```
ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-1
ssh -p 10102 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-2
```

金鑰認證已設好。**密碼與 token 不得寫入任何入庫檔案。**

Repo 在 `/nfs/home/nelson0314/WACV-s3`。虛擬環境由 `~/env.sh` 提供
（`$PY` 指向 `~/venvs/wacv/bin/python`，另設 `HF_HOME`、`PYTHONPATH`）。
套件安裝走 `uv`：`VIRTUAL_ENV=$VENV uv pip install <pkg>`（venv 裡沒有 pip）。

### 五個必須知道的坑

1. **先 `source ~/env.sh`，再 `cd` 到 repo。** `env.sh` 最後一行會把工作目錄
   換到舊的 `~/WACV`，順序相反時所有相對路徑指向錯的樹，而且不會報錯。
2. **卡是多人共用的，不可以把工作送到別人正在用的卡上。**
   派工前一律跑 `bash scripts/free_cards.sh` 取空卡號：

   ```
   bash scripts/free_cards.sh --verbose     # 看每張卡是誰的
   DEVS=$(bash scripts/free_cards.sh)       # 直接拿去餵派工腳本
   ```

   **不要自己讀 `nvidia-smi --query-gpu=memory.used`**——它只給總量，看不出
   那些記憶體是誰的。實際踩過一次：只看了輸出的前三行、把別人 16 GB 的工作
   當成自己的殘留，把兩個 process 疊到別人正在用的卡上，那張卡變成四個
   process。要看誰佔的得用 `--query-compute-apps=gpu_uuid,pid` 再跟自己的
   pid 比對，那正是 `free_cards.sh` 在做的事。

   判定「空」要兩個條件同時成立：卡上沒有別人的 compute app，且已用記憶體
   低於約 1 GB。沒有空卡就等，不要擠。

   **這一條已由程式強制，不只是規定。** 所有派工腳本（16 支，凡是有
   `require_slots` 的都有）在啟動前會呼叫

   ```
   bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
   ```

   指定的卡只要有一張是別人的就**拒絕啟動**，回傳 3。**列印了卻不擋等於沒擋**
   ——這一道是踩了第二次才加的：第一次是只讀 `nvidia-smi` 的前三行，第二次是
   等待器**印了**空卡清單、清單是空的，但派工沒有依它決定要不要送，於是兩張
   卡上各疊到一個別人的 process。手動送單一 process 時也要自己先 `--assert`。
3. **`git pull` 常因 `runs/` 的未追蹤檔衝突而 abort**，先把它們 `mv` 到暫存
   目錄再 pull。
4. **遠端也用 sparse-checkout**，而且它的模式表是**白名單**（`/*` 之後
   `!/runs/`，再逐一列出允許的 `runs/` 子目錄）。兩件事會因此發生：

   - `git pull` 之後某個頂層目錄沒出現 → `git sparse-checkout add <目錄>`。
   - **更危險的一種**：某個 `runs/` 子目錄原本是未追蹤的本機檔案，一旦它被
     commit 進來，`git pull` 會把它變成「已追蹤但被 sparse 排除」，於是
     **git 直接把那些檔案從工作區刪掉**。實測 `runs/ip2p_dct_band_extend/*/results.csv`
     就是這樣消失的（PNG 未入版控故倖存），而後續讀那份 CSV 的批次才報錯。
     檔案本身在 git 裡沒丟，`git sparse-checkout add` 就會回來。

   **每次新增 `runs/` 子目錄並 commit 之後，先在遠端 `git sparse-checkout add`
   那個目錄，再 pull。**
5. **`pkill -f` 的樣式要用中括號寫法**（`[i]p2p_run`），否則會匹配到自己的
   ssh 連線並截斷輸出。

### 判斷批次是否結束

比對**腳本路徑**，不要比對關鍵字：

```
ps -u $USER -o cmd | grep -c "[i]p2p_run"
```

## 跑實驗

### 主線驅動

| 腳本 | 用途 |
|---|---|
| `scripts/ip2p_run.py` | IP2P 線的防禦與評測。所有條件與旗標的單一入口 |
| `scripts/phase_retention.py` | 抗淨化與空白地板。**只讀已存的防禦圖，不重跑攻擊** |
| `scripts/phase_ablation.py` | 相位參數化消融（SDEdit 線） |
| `scripts/apa_baseline.py` | 弱／強 baseline（SDEdit 線） |
| `scripts/tradeoff_curve.py` | 取捨曲線與錨點內插。**不跑 GPU** |
| `scripts/fid_batch.py` | 批次 FID，樣本數不足時拒絕輸出 |
| `scripts/advdrop_repro.py`／`djsma_repro.py` | 對照組在自己威脅模型上的重現 |
| `scripts/fetch_omniedit.py`／`fetch_imagenet_val.py` | 取資料並寫 provenance |

### 抗淨化的派工

一律讀**已存的**防禦圖，不重跑防禦。分片名只能是 `color`／`scene`／`object`
——`retention_table.py` 的 `tag_of()` 由檔名還原條件標籤。全部吃「卡號」當第一
個參數並有 `require_slots`（點數超過「卡數 × 2」直接拒絕）。

| 腳本 | 跑什麼 | 輸出 |
|---|---|---|
| `purify_headtohead.sh "<卡>" [all\|conditions\|floor] ["<分片>"]` | 五個主條件與空白地板 | `runs/ip2p_purify_headtohead/` |
| `purify_antijpeg.sh "<卡>" "<tag>" ["<分片>"]` | DCT-Shield §6.3 抗 JPEG 設定的工作點 | 同上 `antijpeg/` |
| `purify_eot_geometry.sh "<卡>" "<tag>" ["<分片>"]` | 隨機化幾何 EOT 的工作點 | 同上 `eot_geometry/` |
| `purify_gridpure.sh "<卡>"` | GrIDPure 那一欄的補跑（條件與地板一起） | 同上 `gridpure/` |
| `residual_permute.sh "<卡>" ["<分片>"]` | 殘差區塊置換（H1 的判定實驗） | `runs/residual_permute/` |

出表一律 `retention_table.py --src runs/ip2p_purify_headtohead`，它會遞迴走進
子目錄、逐圖扣地板、缺地板的格子排除並回報。

### 機制探針（**不跑 GPU**，可與批次並行）

| 腳本 | 問什麼 | 輸出 |
|---|---|---|
| `free_cards.sh` | **派工前必跑**：列出真正空著的卡號 | 印到 stdout |
| `residual_signature.py` | 殘差的指紋：集中度／頻帶／格點鎖定 | `runs/ip2p_residual_signature/` |
| `residual_texture_alignment.py` | 殘差有沒有對準影像內容（等第相關） | `runs/residual_texture_alignment/` |
| `purifier_band_transfer.py` | 淨化算子用哪一種方式破壞擾動 | `runs/ip2p_residual_signature/` |
| `shading_field_cost.py` | 乘性明暗場對等 RMS 加性場的失真 | `runs/shading_field_cost/` |

### 分片與並行

一個 process 負責「一個工作點 × 一整批影像」，不要一個 process 跑一張。
載入模型的固定成本約 320 秒（權重在 NFS 上），分片越細被重複越多次。

八張卡上**每卡 2 個** process 是已知不會 OOM 的密度（每個約 6–9 GB，3090 是
24 GB，而**卡是多人共用的**，別人的 process 也在上面）。

**每卡 3 個會 OOM。** 實測：`floor_only_sweep.sh` 的卡號公式
`dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}` 在 6 個工作點配 2 張卡時會繞回去，
把 4 個 process 疊到同一張卡上，六點掛掉四點，全部是 `CUDA out of memory`
——而且錯誤出現在跑了十幾分鐘之後，不是啟動時。

**所有掃描腳本現在都有 `require_slots`**：工作點數超過「卡數 × 2」時直接拒絕
啟動並說要幾張卡，不再靜默疊加。

### 實測單張成本（RTX 3090，8 卡滿載）

| 工作 | 秒／張 |
|---|---|
| 相位防禦 ＋ 一次編輯 | 約 122 |
| DCT-Shield（1000 步 PGD）＋ 一次編輯 | 約 575 |
| 抗淨化一格（4 算子 × 3 seed = 12 次編輯） | 約 868 |
| AdvDrop 一個設定（200 張整批） | 約 26（整批） |

## 資料保全

`runs/` 保存**數值記錄**：CSV / JSON / log / txt 一律入版控。

**影像與 HTML 不入版控。** 防禦圖能由已記錄的參數與種子重跑出來，屬於可重新
產生的中間產物；量測結果（CSV）不可重現，一律保留。

改動 `.gitignore` 的 `runs/` 區塊時，必須用 `git status --porcelain --ignored`
確認沒有結果檔被排除。

## 命名

目錄與檔名一律描述性，不含日期、流水號或順序詞。規則見
[INDEX.md](INDEX.md#命名規則)。

`runs/` 現有的目錄群組：

| 前綴 | 內容 |
|---|---|
| `ip2p_*` | InstructPix2Pix 主線 |
| `sdedit_*` | SDEdit 線（凍結） |
| `phase_*` | 本方法的參數與消融 |
| `*_reproduction`／`*_repro` | 對照組在自己威脅模型上的重現 |
| `baseline_apa/` | APA 弱 baseline |
| `dct_shield_*`／`advdrop_*`／`djsma_*` | 各對照組 |
| `execution_logs/` | 執行日誌 |
