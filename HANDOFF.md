# 交接

程式與數值在 `non-additive-frequency/`。工作規則見該目錄的 `CLAUDE.md`。

## 一句話現況

顏色載體的天花板已經量清楚；換成**低頻平滑的取樣位移場**（幾何）之後，
「自然」與「有效」不再對立，而且找到了兩個各自有效的機制——**粗控制網格**
（局部形變在結構上做不出來）與**過渡帶**（讓背景的剛性約束只管背景）。
六批實驗中前兩批已完成，其餘四批在遠端佇列上依序跑。

## 為什麼離開顏色

三個互相獨立的量測指向同一個根因：**顏色載體的防禦力全部住在空間不均勻裡，
而空間不均勻正是人眼判定不自然的那個東西。**

| 來源 | 載體 | 自由度 | 色差 | 編輯輸出的身分降幅 |
|---|---|---|---|---|
| `runs/color_scaleup_search/` | 全域 Lab 仿射 | 12 | ΔE 6.3 | −0.002 ～ +0.0043 |
| `runs/objective_pilot/` | CIELUV 3D LUT + 平滑正則 | 16×32×32 | ΔE ≈ 6 | +0.0005 ～ +0.0077 |
| `runs/immunise_tv*/` | 仿射場，掃位移場 TV 上限 | — | ΔE 16 | TV 0.5 → +0.943；TV 2.0 → −0.440（不自然） |

前兩列就是三篇參考論文共同的載體家族（AdvCF 的分段線性 tone curve、
NCF 的 Monge–Kantorovich 仿射、AdvColorFilm 的物理色片）。它們驗證的對象是
ImageNet 分類器，跨一條決策邊界就算贏；對 50 步的 IP2P 編輯量到的是零。

新做的 `lab_field`（逐通道低頻 Lab 位移場，與顏色線同 ΔE 16 預算）四道淨化
全部 0–3/25，再次確認這個上限。

## 現行的做法

| 腳本 | 角色 |
|---|---|
| `scripts/immunise.py` | 顏色載體的防禦圖。遞迴檢查設定檔，出現 `instruction`／`prompt` 鍵就拒絕啟動。 |
| `scripts/immunise_field.py` | **位移場載體**的防禦圖。載入上一支的 `assert_no_instructions`，共用同一個檢查。 |
| `scripts/evaluate_defence.py` | 唯讀評估。指令與淨化只在這裡。只載入 `arms` 裡列到的臂。 |
| `scripts/field_readout.py` | 唯讀，**防禦圖本身**的身分與失真。不含指令。 |
| `scripts/summarise_screen.py` | 彙整成可並列的表，含門檻掃描。 |
| `scripts/build_field_report.py` | 產生看圖用的 `data.js` 與影像複本。 |

**載體**（`configs/immunise_field*.json`）：

- `FlowFieldParam`（`src/defense/geometry_field.py`）——`grid×grid×2` 控制點
  雙三次上採樣成取樣座標的位移場，邊界 taper 漸縮為零。四道約束：
  `flow_face`（臉框內位移 CVaR99）、`flow_rigid`（臉框外對最佳全域仿射的殘差，
  支撐可用 `rigid_margin` 向外膨脹）、`flow_face_rigid`（臉框內同樣的殘差）、
  `flow_fold`（`det J` 低於下界的部分）。
- `LabOffsetFieldParam`（`src/defense/lab_offset_field.py`）——逐通道低頻 Lab
  位移場。`delta` 直接就是位移，均勻性是參數化的性質。

**目標函數**：`src/defense/instruction_free.py` 的 `enc`／`cond`／`id` 為基準；
`src/defense/mainstream_terms.py` 另有 PhotoGuard 的 targeted encoder attack
與 AdvDM／Mist 的擴散訓練損失；`src/defense/eot.py` 把攻擊者的前處理放進期望值。
三者都不含文字、不含指令。

## 量到什麼

### 判準（過半規則、門檻 0.5、25 格）

| 載體 | identity | jpeg | blur | crop | 防禦圖是自然照片？ |
|---|---|---|---|---|---|
| 顏色線（`runs/purify_heldout/`） | 18 | 18 | 15 | 16 | **否**（21% 像素 ΔE00 > 25） |
| `flow_16` | 11 | 6 | 11 | 9 | 四張是、`114555` 否（鼻子局部拉長） |
| `flow_08` | 5 | 5 | 6 | 2 | 是 |
| `lab_field` | 0 | 0 | 1 | 1 | 是 |
| `face_rigid_16` | 0 | — | — | — | 是 |
| `face_rigid_32` | 0 | — | — | — | 是 |

**顏色線的 18/25 用的是過半規則**，`summarise_screen.py` 的 `majority` 欄與它
逐格相同，並列時看那一欄。門檻掃描（0.3–0.7）顯示排序穩定。

### 防禦圖本身的身分（`scripts/field_readout.py`）

錨定在原圖主體框的固定座標，偵測器不參與。這一層回答「這張照片還是不是
這個人」，同時是**錨定裁切錯位的上界**。

| 變體 | 範圍 |
|---|---|
| `flow_02` | 0.974–0.989 |
| `flow_08` | 0.504–0.905 |
| `flow_16` | 0.295–0.874 |
| `face_rigid_*` | 0.977–0.997 |
| `coarse_32` | 0.561–0.984（`126577` 是 0.561） |

### 四個機制，兩個有效

1. **局部形變才是不自然的來源**——`flow_16` 在 `114555` 上把鼻子局部拉長；
   加上臉框內的仿射約束之後鼻子修回去了（圖上驗證）。假設成立。
2. **用仿射約束達成它會買掉全部的防禦**——`face_rigid_16` 1/25、
   `face_rigid_32` 0/25。
3. **用粗網格達成它則保住強度**——`coarse_32` 在 `126577` 上位移 29.5 px、
   代理 id 0.528，圖仍自然。兩者都消除局部形變，但粗網格限制的是位移的
   **空間頻率**，不限制臉能怎麼動。
4. **綁住解的是背景剛性，不是臉框預算**——`face_rigid_16` 允許 16 px，
   實際只走到 1.42–2.01，而 `flow_rigid` 貼死在 1.98–1.998 / 2。擋掉多少取決於
   **臉佔畫面的比例**：`coarse_32` 上大頭照走 29.5/32，半身照只走 5.4/32。
   修法是 `rigid_margin` 把臉框向外膨脹再取補集。

## 遠端佇列（`scripts/batch_queue.sh`）

一支總排程，五張卡一次只服務一批，順序就是優先序：

| 批次 | 設定 | 問什麼 |
|---|---|---|
| 補跑 | — | `coarse_16`／`coarse_32` 被 OOM 打斷的 shard 1、4 |
| `margin` | `immunise_field_margin.json` | 過渡帶能不能解開臉框預算 |
| `coarse_margin` | `immunise_field_coarse_margin.json` | 粗網格 × 過渡帶（兩個瓶頸不同，預期相乘） |
| `objective` | `immunise_field_objective.json` | targeted encoder、擴散訓練損失、長鏈 |
| `eot` | `immunise_field_eot.json` | 前處理進期望值（跑完整四道淨化） |

進度看 `runs/batch_queue.log`。每批評估後檢查列數是不是 150，不是就標出來。

## 已經量掉、不要重做的路

1. 全域仿射、全域 3D LUT、逐通道 Lab 位移場都沒有防禦效果。
2. 指令進迴圈會高估防禦（舊路徑 2/15，拿掉後 1/25）。
3. 「解完再縮」不是解受約束問題（1/25 → 18/25）。
4. NIQE 不能當自然度門檻（逐張比值中位 1.054，有一張 0.837）。
5. CVaR95 尾端上限解決不了自然度。
6. TV 單獨不夠（`medium` 的 TV 比 `global` 低 21%，U16 反而更高、圖更糟）。
7. 端點與 raw 越界的約束有效，飽和已不是不自然的來源。
8. **臉框內的仿射約束買到自然度但買掉防禦**（見上）。

## 還沒試過的槓桿

| 槓桿 | 說明 |
|---|---|
| 指令完成那條腿 | `criterion.py` 明寫三種寫法都在圖上被推翻，`dir_norm` 因此不進目標函數。要一個量得準的 VQA 讀數。 |
| 可用性讀數 | `use_norm` 用 NIQE，幾乎不綁。需要有參考的一組，且逐張對圖校準過才准進 `TERMS`。 |
| 幾何 × 顏色串接 | `CompositeParam` 接得上，先各自量到底再談。 |
| 受成像限制的局部 PSF | 空間變化的小幅散焦，人眼讀成景深。 |
| 更大的樣本 | 見下。 |

## 使用者要判斷的取捨

`coarse_32` 在 `126577` 上的防禦圖身分是 **0.5606**——嵌入認為那已經是另一個人，
而圖看起來仍然自然。硬約束寫的是「防禦後的照片本身必須仍是一張自然的照片」，
沒有要求「還是同一個人」。這一點要由使用者看圖決定接不接受。

## 實驗規模的限制

5 張影像（其中 `195273` 是插畫）、3 顆種子、單一初始化。零次攻擊成功時三顆
種子的單側 95% 上界仍有 63%。要支持 ±10 個百分點的成功率敘述需要約 97 個
獨立身分，±5 點約 385。現在的數字只夠當 pilot。

## 環境與陷阱

- 遠端 `ssh -p 10102 nelson0314@server.basiclab.lab.nycu.edu.tw`（basic-2）、
  `-p 10101`（basic-1）。Repo 在 `/nfs/home/nelson0314/WACV-s4`，
  **整棵同步過去的，不是 git 工作區**，改完本機要自己 `scp`。
- **先 `source ~/env.sh`，再 `cd` 到 repo**：`env.sh` 最後一行會換工作目錄。
- 本機 Python `C:/Users/nelso/miniconda3/envs/wacv/python.exe`，
  測試基準 `python -m pytest -q --ignore=runs` → **322 passed / 1 skipped**。
- **殺一支排隊中的腳本不會帶走它已經啟動的子行程。** 踩過一次：兩個孤兒各佔
  19.8 GiB 跑了 26 分鐘，把同卡上的評估擠到 CUDA OOM，兩個變體各只寫出 90 列
  而不是 150，而批次腳本照樣印「篩選完成」。殺完要查
  `ps -u $(whoami) -o pid,ppid,cmd | awk '$2==1'` 找孤兒，並用
  `nvidia-smi --query-compute-apps` 確認卡真的空了。
- **ssh 連線很不穩**，`Connection closed/reset`、`kex_exchange_identification`
  常見，所有遠端呼叫都要寫重試迴圈。
- 用 heredoc 在遠端寫 `.sh` 之後要 `sed -i 's/\r$//'` 再 `bash -n`。
  巢狀 heredoc 裡的 `\\` 會被吃掉，續行會被折成一行——功能不受影響但要知道。

## 報告

- 兩種色差預算：https://claude.ai/code/artifact/95164c77-d3ad-478b-a502-64262285f686
- 未見指令的防禦成績：https://claude.ai/code/artifact/1bbb2517-19d8-437a-acfb-0ab266aa5da2
- 受約束求解的全部產物：https://claude.ai/code/artifact/7a747d0b-0793-4e14-a234-e5af70d5f600
- 尾端上限的四檔：https://claude.ai/code/artifact/735dbe54-1967-41fc-93d8-289c3a3ddbbc
- 均勻度與防禦強度：https://claude.ai/code/artifact/76350b7c-07b6-4a1a-8186-dcbf98a4e552
- 色彩載體防禦講義（背景知識）：https://claude.ai/code/artifact/afb95ad3-d0ca-4e3b-952f-4afd3e0f4b8a
