# 交接

程式與數值在 `non-additive-frequency/`。工作規則見該目錄的 `CLAUDE.md`。

## 一句話現況

**判準上的「防禦成功」有很大一部分是發布前就已經換了人，不是攻擊者的編輯失敗。**
逐張拆開：`coarse_eot_48_crop` 的 6 個過半穿透格裡 **5 格**來自
`task_env_weather_114555`，而那張的防禦圖身分是 **0.336**（編輯前已損失 0.664）；
`grid_05` 的 9 格裡 5 格同樣來自它（0.389）。`criterion` 的軟極小幾乎總是取到
`id_norm`，而發布圖本身已經換了人時，那個比值不需要編輯失敗就會掉下來。

**所以「唯一自然且四道淨化全部 ≥ 6 的可交付點」這個說法已收回。**

其次，**「產物自然」與「防禦有效」在兩個載體上都是同一個取捨的兩面**：
顏色線均勻（自然）→ 0/25、不均勻（18/25）→ 兩成像素 ΔE00 > 25；
幾何線形變集中在臉框（11–21/25）→ 一眼看得出，攤到整個畫面（自然）→ 0–1/25。
但這是**已測載體與現行評分下的經驗取捨**，還不能認定為威脅模型的真實極限——
判準實際只有 `id_norm` 與 `use_norm` 在作用，指令完成度（`dir_norm`）因為量不準
而退出，所以「阻止指令完成但保留自然輸入」這條路根本還沒被評估過。

報告頁：
[位移場](https://claude.ai/code/artifact/cde9d5c2-f2af-48c9-815e-639adec4a344)
· [顏色](https://claude.ai/code/artifact/1a67851c-9871-43cf-8485-77abcbffd59a)
· 本機 `report_field.html`、`report_colour.html`

## 下一步的優先序（Codex 診斷，我已逐項驗證數字）

1. **先補評估，再補防禦。** 四項分開的盲評——發布圖自然度、發布圖是否同一人、
   編輯是否完成、編輯結果是否可用。現有影像就夠做，不需要 GPU。
   並且要多報一欄：**編輯後相對 clean edit 的「額外」身分損失**，
   與編輯前已有的損失分開。
2. **掃整幅仿射殘差**（固定 `all_px=32`，`affine_all_px ∈ {3,6,9,12,18,24,32}`
   加一個拿掉該 cap 的診斷臂）。它是目前最明確又未經人眼校準的約束，
   而且現行的 `whole_08/16/32` 同時動了三個變數，沒有隔離它。
   **注意**：`affine_residual` 量的不只是背景彎曲，也量邊界接合——
   `flow()` 把整個場乘上邊界為零的 taper，而非零的全域仿射位移不可能在整圈
   邊界為零，所以「接回固定邊界」本身就會產生殘差。
3. **條件式目標**：現行的 `id` 只壓低編輯輸出的餘弦，沒有要求發布圖仍保有身分，
   所以「直接改變輸入的臉」是一條捷徑——上面的逐張拆解就是它被走通的證據。
   改成「輸入身分約束 ＋ 編輯後損傷 margin」。
4. **獨立全域變換 ＋ 受局部伸縮約束的殘差場**：目前把「可接受的整體變換」、
   「邊界接合」、「真正的局部變形」混成一個量。
5. **依表面形狀的 adversarial relighting**：物理一致的陰影是「空間不均勻但可能
   自然」的自由度，而現行的 Lab field 是任意低頻色彩位移。

## 衣物支撐做過了，結論是結構性的

`ncf_support(x, 'clothes')`：ATR 衣物類別經導向濾波貼回衣緣、扣掉 MTCNN 臉框、
向內侵蝕與羽化。每一步只移走權重，**支撐外逐位元照抄原圖**
（`protected_max_abs = 0.0`，189219 像素驗過）。

`runs/color_scaleup_search/` 的 480 列，六個臂在支撐加權 ΔE00 6.3–8.6 上，
身分降幅中位 **0.0004–0.0092**；`runs/color_ceiling/` 把它推到 ΔE00 18–20
（整件衣服完全換色）仍在同一量級。

兩個理由，都不是實作問題：**臉不在支撐裡**（那是它保護得乾淨的原因，
也是它動不到身分的原因）；**衣物類指令本身就是改色**，攻擊直接覆蓋防禦——
那一批裡最無效的正是衣物類，防禦後與未防禦的編輯圖位移中位只有 0.02。

## 新的資料集（尚未跑過任何實驗）

`data/portrait_manifest.json` 六張主體鮮明的人像，由
`scripts/select_subjects.py` 排序、逐張看圖選出。**偵測器會誤判、插畫旗標會漏**
（`231482` 是一盤香蕉而 MTCNN 給了 0.67 的臉框；`175843` 是 3D 卡通而兩個旗標
都沒越界），所以那支腳本只出排序。先前的插畫與藝妓兩張已排除。
`configs/evaluate_portrait.json` 四類指令，**restyle 已移除**——它把整張圖重畫，
防禦有沒有作用看不出來。

**現有的全部數字都還是舊的五張。**

## 全部設定的排名（過半規則、門檻 0.5、25 格）

| 變體 | identity | jpeg | blur | crop | 產物 |
|---|---|---|---|---|---|
| `margin_64_norigid` | **21** | — | — | — | 臉毀掉，一張偵測不到臉 |
| 顏色線（`runs/purify_heldout/`） | 18 | 18 | 15 | 16 | 不是照片，21% 像素 ΔE00 > 25 |
| `eot_crop_16` | **17** | 13 | 11 | 14 | `114555` 鼻嘴壞掉 |
| `coarse_margin_96` ／ `grid_06` | 13 ／ 13 | — | — | — | **四張自然，一張邊緣** |
| `grid_08` | 12 | — | — | — | 兩張明顯壞掉 |
| `flow_16` | 11 | 6 | 11 | 9 | 四張自然 |
| `grid_05` | 9 | — | — | — | 自然 |
| `obj_longchain` | 7 | — | — | — | 自然 |
| `eot_both` | 5 | 5 | 6 | 6 | 自然 |
| `flow_08` ／ `eot_none` ／ `obj_baseline`（同設定三次） | 5 ／ 4 ／ 1 | 5／1／— | 6／6／— | 2／1／— | 自然 |
| `eot_crop` | 4 | 4 | 4 | 4 | 自然 |
| `coarse_margin_48` | 3 | — | — | — | 自然 |
| 其餘（含全部臉內仿射約束的） | 0–2 | — | — | — | 自然 |

## 五個機制，一條前緣

每一步都是前一步的**圖**逼出來的，不是先想好的清單。

1. **顏色載體的防禦力全部住在空間不均勻裡**，而那正是人眼判定不自然的東西。
2. **換成位移場**，防禦出現（`flow_16` 11/25），但預算推大會出現局部 warp。
3. **臉框內的仿射約束修好了局部 warp，也一起關掉了防禦**——它是開關不是旋鈕：
   過渡帶 0→64 px 只讓臉框位移從 1.8 動到 2.0；上限 1.0→2.0 讓位移剛好加倍；
   整個拿掉直接跳到 17–21 px。一個平滑的場要移動臉又要接回背景，那個接合
   發生在框**內**，框外的過渡帶救不了。
4. **粗網格用結構取代約束**（有效）：間距 85 px 時整張臉落在兩三個控制點之間，
   被整體改形而不是被撕開。同一張照片上 32 px 間距走 21.5 px 五官就散了，
   85 px 間距走 26.8 px 仍是自然照片。**網格解析度有最佳值**——`grid_08`
   （64 px 間距）代理更強、真編輯沒有更強（12 對 13），而圖壞得多。
5. **EOT 抬最弱的那一道淨化**（有效）：8 px 上 crop 1→4／6、jpeg 1→4／5，
   而 identity 與 blur 幾乎不動。16 px 上是 +6／+7／0／+5，但那一組的對照來自
   別批，而同設定跨批的差可達 1 對 5，所以有一部分可能是批次差。

**主流目標函數沒有勝出**：PhotoGuard targeted encoder、AdvDM／Mist 擴散訓練
損失、長鏈，全部落在重跑雜訊內（`runs/field_objective/README.md`）。

## 第七批：兩個機制相加（已完成）

`configs/immunise_field_coarse_eot.json`，四道淨化，四個變體同批因此彼此可比：

| 變體 | EOT | identity | jpeg | blur | crop | 產物 |
|---|---|---|---|---|---|---|
| `coarse_eot_96_none` | 無（批內對照） | 1 | 2 | 2 | 3 | `114555` 邊緣 |
| **`coarse_eot_48_crop`** | 裁切 | **6** | **6** | **7** | **7** | **看過的三張都自然** |
| `coarse_eot_96_both` | 裁切＋模糊，兩倍成本 | 7 | 7 | 7 | 7 | `114555` 壞 |
| `coarse_eot_96_crop` | 裁切 | 9 | 9 | 10 | **15** | `114555` 壞 |

**EOT 抽的算子族越窄，峰值越高且落在它訓練過的那一道**（`_crop` 在 crop 上 15、
其餘 9–10）；**族越寬，曲線越平但整體較低**（`_both` 齊平 7，四道都不超過
`_crop`——多花一倍買不到東西）。

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

## 遠端佇列（`scripts/batch_queue.sh`／`scripts/queue_rest.sh`）

一支總排程，五張卡一次只服務一批，順序就是優先序。已跑完：補跑 → `margin`
→ `coarse_margin` → `grid`。`objective` 在跑，`eot` 排最後。
進度看 `runs/batch_queue.log` 與 `runs/queue_rest.log`。每批評估後檢查列數是不是
150，不是就在 log 裡標出來——先前那次 OOM 失敗是靜默的，只在列數上看得出來。

**改一支正在跑的 `.sh` 沒有用**：`sed -i` 換的是新 inode，執行中的 bash 仍握著
舊檔（`/proc/<pid>/fd/255` 會指向 NFS silly-rename 過的名字）。要換順序就得
另寫一支等待前一支結束的腳本。

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

**一、防禦圖的臉確實變了。** `coarse_margin` 一族的防禦圖身分餘弦是 0.46–0.98。
硬約束寫的是「防禦後的照片本身必須仍是一張自然的照片」，沒有要求「還是同一
個人」。這一條要由使用者看圖確認。

**二、邊緣的兩張。** `grid_06` 的 `114555`（鼻頰有一片拉扯）與
`coarse_margin_96` 的 `126577`（下半臉開始拉扯）決定 13/25 算不算數。

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

## 檔案在哪

| 東西 | 位置 |
|---|---|
| 兩份報告（可直接開） | `report_field.html`、`report_colour.html` |
| 報告用的影像 | `_rep/field/`、`_rep/colour/` |
| 全部防禦圖與編輯圖的鏡像 | `_rep/defended/`（214 張）、`_rep/edits/`（1125 張，identity・s17001） |
| 每一批的數值記錄 | `non-additive-frequency/runs/field_*/`、`runs/field_*_eval/` 的 CSV |
| 每一批的設計理由 | 各目錄的 `README.md` |

`_rep/` 與 `report_*.html` 都在 `.gitignore` 裡——影像由已記錄的參數與種子重跑
得出，CSV 不可重現所以一律入庫。**本 session 產生的 PNG 已從 `runs/` 刪除**
（6.3 GB → 11 MB，399 個記錄檔全部保留），遠端 `/nfs/home/nelson0314/WACV-s4`
仍有完整副本。先前 session 的影像沒有動。

## 產生報告

用 `.claude/skills/defence-report/`（skill）。它記了六塊結構、兩層讀數、
建置指令、以及發佈的硬限制（總檔數 ≤ 256、單版 ≤ 64 MB、`files` 要 map 形式）。

## 報告

- 兩種色差預算：https://claude.ai/code/artifact/95164c77-d3ad-478b-a502-64262285f686
- 未見指令的防禦成績：https://claude.ai/code/artifact/1bbb2517-19d8-437a-acfb-0ab266aa5da2
- 受約束求解的全部產物：https://claude.ai/code/artifact/7a747d0b-0793-4e14-a234-e5af70d5f600
- 尾端上限的四檔：https://claude.ai/code/artifact/735dbe54-1967-41fc-93d8-289c3a3ddbbc
- 均勻度與防禦強度：https://claude.ai/code/artifact/76350b7c-07b6-4a1a-8186-dcbf98a4e552
- 色彩載體防禦講義（背景知識）：https://claude.ai/code/artifact/afb95ad3-d0ca-4e3b-952f-4afd3e0f4b8a
