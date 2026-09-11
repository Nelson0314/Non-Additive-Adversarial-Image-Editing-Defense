# 交接

程式與數值在 `non-additive-frequency/`。工作規則見該目錄的 `CLAUDE.md`。

## 現況

遠端 `basic-2` 上有三個行程在跑 `scripts/color_ceiling.py`
（`--config configs/color_scaleup.json`，分片 1/3、2/3、3/3，卡 4、5、6），
輸出在 `/nfs/home/nelson0314/WACV-s4/runs/color_scaleup_search/shard{i}of3`。
每片 40 格、預期 440 列。本機沒有背景工作。

測試基準：`python -m pytest -q --ignore=runs` → **193 passed / 1 skipped**。
`--ignore=runs` 是必要的，`runs/ncf_cpu_test_tmp/pytest-of-nelso` 被提權沙箱的
ACL 鎖住，會讓收集階段直接失敗。

本機 Python `C:/Users/nelso/miniconda3/envs/wacv/python.exe`。
遠端工作目錄 `/nfs/home/nelson0314/WACV-s4`（basic-1 埠 10101、basic-2 埠 10102，
home 跨機同步）。**遠端那棵樹是整棵同步過去的，不是 git 工作區**；改完本機要
自己 `scp`，而且批次跑到一半不要同步（跑著的行程已把模組讀進記憶體，但下一次
啟動就會換版本，兩批因此不可並列）。

## 遠端的三棵樹

`WACV`（最舊）、`WACV-s3`、`WACV-s4`（現行）。`~/env.sh` 的 `PYTHONPATH` 指向
`~/WACV`，派工腳本自己把 repo 根目錄插在 `sys.path[0]`，所以兩棵樹同時在路徑上。
`src/defense/ncf_search.py` 只存在於 `WACV-s3`。

## 跑過什麼

`runs/color_scaleup/`：20 圖 × 6 指令類的放大批次，**只有 2 片完成（528 列）**，
其餘分片崩在 `collision_region` 臂的 NaN 上。那個 NaN 的成因是前向做 SVD 會把
奇異值夾成完全相等的重根，反向就是除以零；修正是把奇異值約束搬出前向
（見 `_clamp_singular_values`）。**這 528 列是 PGD 路徑的產物**
（`collision_steps=300`、表頭沒有 `search_*` 欄），與現行程式不可並列，
不要併進新的批次。

`runs/color_ceiling/`（遠端 450 列，5 圖）、`runs/collision_arm/`（180 列）、
`runs/color_retention/`（450 列）、`runs/gradient_diagnostics/`（45 列，修正版在
`fixed2`）、`runs/step_calibration/`、`runs/throughput/`、`runs/precision_check/`：
都還在遠端，尚未收進專案的 `runs/`，也還沒寫 README。

`runs/objective_pilot/`、`runs/subject_anchored_readout/`、`runs/lowfreq/`：
見各自的 README。`lowfreq` 那 1120 列等於恆等映射，幅度那一軸沒有取樣過。

## 已知的污染

1. **淨化清單漏掉四個色彩算子。** `src/purify/ops.py` 有 `grayscale`、
   `gray_world`、`auto_levels`、`clahe`，註解明說那四個是為色彩重映射防禦而加，
   而跑過的淨化批次一個都沒用。`grayscale` 會把色度整個丟掉。
   **這一項可能推翻「抗淨化成立」這個目前唯一拿到手的結論。** 使用者裁定淨化
   的處理順序另外決定，故未動。
2. **`soft_gamut(knee=0.06)` 在色域內就偏離恆等。** 量到的形狀寫在該函式的
   說明裡：`f(0)=0.041589`、`f(1)=0.958411`，`[0,1]` 上平均偏離 0.005919，
   中段 `[0.2,0.8]` 只有 0.002103；整圖濾鏡把 `amplitude` 設成 0 之後 dE00
   仍有 0.0776 與 0.5149。改 `knee` 會讓既有批次不可並列。
3. **`project()` 不是幂等，收尾的 `delta` 可以超出 `radius`。** 實測
   `collision_region` 臂跑完搜尋後 `|delta|` 最大 0.214925，而 `radius` 是 0.2。
4. **`radius=0.2` 與 `epsilon_lab=5` 來自 NCF，約束的是 `delta`。** 顏色位移
   幾乎全部來自 `T0` 與 `amplitude`：整圖濾鏡在 `amplitude=1`、`delta=0` 時
   dE00 已經是 8.2659 與 12.7213，把 `delta` 推到投影後的盒角只多 0.0136 與
   0.6133（0.2% 與 4.6%）。
5. **「主體不動」只在衣物載體上成立。** 整圖濾鏡的支撐是整張照片，人臉在支撐
   內，該側要看的是身分讀數與 `gate_*` 欄位，不是逐位元相同。

## 三件寫成測試釘住的性質

1. `ReColorAdvParam.project()` **不是幂等的**：`clamp(0,1)` → CIELUV↔RGB 色域
   往返 → L∞ 盒，最後一步會把點推回色域外。與原文 `project_params` 同序，不要修。
2. **盒角初始化到不了可達集合的極端**：色域投影把格點邊緣拉回來，可達集合不是
   盒子。`carrier_objectives.box_corner_init` 產生的是「盒角經投影」的點，
   引用 `runs/objective_pilot/` 的 0.3626 時要照這個定義寫。
3. 訓練前做的初始化會被 `reset` 抹掉，且不拋錯。

## 兩個會靜默失效的環境事實

- `IP2PWrapper(dtype=fp16)` 下 `resolve_precision` 讓 backbone 半精度而 VAE
  留在 fp32，官方 pipeline 的 `edit()` 會在 `vae.encode` 拋 Half/float 不符；
  `edit_differentiable` 自行處理故不受影響。攻擊評測用 fp32 或 bf16。
- `pkill -f <樣式>` 會匹配到 ssh 遠端命令自己的 shell，把連線殺掉並回傳 127。
  改用 `ps -u $USER -o pid=,comm=,args= | awk '$2=="python" && /<樣式>/ {print $1}'`
  取 pid 再殺，殺完回頭用 `nvidia-smi` 複驗。
- 派工用的 shell：`cd dir && cmd &` 會把 `cd` 一起背景化，後面的指令在錯的目錄
  執行，重導向會失敗而且看起來像成功。一次 ssh 只派一片。

## 使用者的裁定

1. 可見的防禦是允許的，只要不動主體、產物自然。
2. 只接受兩種載體：整圖顏色濾鏡、衣物換色。
3. 分工：衣物指令由衣物載體處理，頭部與背景指令由整圖濾鏡處理。
4. **高頻禁令已撤銷。** 載體可以往空間高頻加東西，方法不預設限制，文獻有的
   都可以試。`hf_ratio_rgb_total` 仍逐列量、逐列報，但它是讀數不是門檻；
   加高頻付出的抗淨化代價要用 `src/purify/ops.py` 量出來，不用先驗論證擋。
5. 畫面崩壞也算防禦成功——判準是攻擊者有沒有拿到一張可用的、認得出是同一個人
   的、已完成指令的照片。
6. 助手不做成敗判斷，只把數據與圖擺出來。
