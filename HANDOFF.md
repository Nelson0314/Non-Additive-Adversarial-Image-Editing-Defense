# 交接

程式與數值在 `non-additive-frequency/`。工作規則見該目錄的 `CLAUDE.md`。

## 一句話現況

顏色載體上「自然」與「有效」直接對立，三個獨立量測都指向同一個根因；換成
**低頻平滑的取樣位移場**（幾何）之後，臉框內 2 px 的位移就把代理身分項從
1.0 壓到 0.76–0.97，而**防禦圖本身的身分餘弦仍有 0.974–0.989**——身分的
損失是編輯造成的，不是防禦圖自己丟掉的。held-out 真編輯的評估正在跑。

## 為什麼顏色線停在這裡

三個量測，同一個結論：**顏色載體的防禦力全部住在空間不均勻裡，而空間不均勻
正是人眼判定不自然的那個東西。**

| 來源 | 載體 | 色差 | 編輯輸出的身分降幅 |
|---|---|---|---|
| `runs/color_scaleup_search/` | 全域仿射（12 自由度） | ΔE 6.3 | −0.002 ～ +0.0043 |
| `runs/objective_pilot/` | CIELUV 3D LUT `16×32×32` + 平滑正則 | ΔE ≈ 6 | +0.0005 ～ +0.0077 |
| `runs/immunise_tv*/` | 仿射場，只改位移場 TV 上限 | ΔE 16 | TV 0.5 → id +0.943；TV 2.0 → −0.440（但不自然） |

前兩列就是「空間均勻、高自由度的顏色映射」——也就是三篇參考論文
（AdvCF 的分段線性 tone curve、NCF 的 Monge–Kantorovich 仿射、AdvColorFilm 的
物理色片）共同的載體家族。它們驗證的對象是 ImageNet 分類器，跨一條決策邊界
就算贏；對 50 步的 IP2P 編輯，量到的是零。

結構上的理由：全域顏色映射可逆、保住每一條邊與每一道梯度，人臉幾何原封不動，
而身分嵌入主要吃幾何與紋理；而且這類映射對 SD 的 VAE 完全是 in-distribution。

## 現行的做法

**三支腳本，指令只在評估那一支出現。**

| 腳本 | 角色 |
|---|---|
| `scripts/immunise.py` | 顏色載體的防禦圖。遞迴檢查設定檔，出現 `instruction`／`prompt` 鍵就拒絕啟動。 |
| `scripts/immunise_field.py` | **位移場載體**的防禦圖。載入上一支的 `assert_no_instructions`，共用同一個檢查。 |
| `scripts/evaluate_defence.py` | 唯讀評估。指令與淨化只在這裡，防禦圖不會被這裡的讀數調整。 |
| `scripts/field_readout.py` | 唯讀，**防禦圖本身**的身分與失真。不含指令。 |

**兩個位移場載體**（`configs/immunise_field.json`）：

- `FlowFieldParam`（`src/defense/geometry_field.py`）——`grid×grid×2` 控制點
  雙三次上採樣成取樣座標的位移場，邊界 16 px 以 smoothstep 漸縮為零。
  三道約束：臉框內位移 CVaR99、臉框外對最佳全域仿射的殘差、`det J` 的摺疊量。
- `LabOffsetFieldParam`（`src/defense/lab_offset_field.py`）——**逐通道**的低頻
  Lab 位移場，`L`／`a`／`b` 各一道 CVaR99 上限。`delta` 直接就是位移，
  `d(p)` 不含 `c(p)`，均勻性因此是參數化的性質而不是要對抗的約束。

目標函數、augmented Lagrangian 的求解、評估路徑全部與顏色線逐字相同，
一次只換一個因素。

## 量到什麼

### 位移場載體（`runs/field_ladder/`，代理身分項，五張）

| 變體 | 臉框預算 | 代理 id 中位 | 貼住的約束 |
|---|---|---|---|
| `flow_02` | 2 px | 0.799（0.762–0.972） | 三道都貼 |
| `flow_04` | 4 px | 0.790（0.633–0.914） | 多數貼 `flow_rigid` |
| `flow_08` | 8 px | 待補 | — |

高檔位貼住的是 `flow_rigid`（背景直線不能彎）而不是臉框預算——綁住解的是
自然度那一道約束，不是幅度。

### 防禦圖本身（`scripts/field_readout.py`，`flow_02`）

| 影像 | 錨定身分 | 偵測身分 | PSNR | LPIPS | ΔE00 |
|---|---|---|---|---|---|
| `123744` | 0.9838 | 0.9863 | 32.90 | 0.0253 | 1.17 |
| `114555` | 0.9839 | 0.9760 | 27.50 | 0.0226 | 1.73 |
| `126577` | 0.9887 | 0.9844 | 26.73 | 0.0536 | 2.24 |
| `195273` | 0.9819 | 0.9635 | 22.42 | 0.0330 | 2.73 |
| `509477` | 0.9743 | 0.9606 | 29.43 | 0.0323 | 1.80 |

**這一列同時是錨定裁切錯位的上界**：防禦圖被 warp 之後臉會移動，而錨定讀數
用原圖座標，量到的有一部分是錯位。同一個錯位的裁切在防禦圖上只值 0.03，
所以編輯輸出上的下降不是錯位造成的。

### 顏色線的主要結果（`runs/purify_heldout/`，600 列）

| 淨化 | criterion 中位 | 穿透格 |
|---|---|---|
| identity | 0.144 | 18/25 |
| jpeg q75 | 0.098 | 18/25 |
| blur σ1.0 | 0.151 | 15/25 |
| crop 10% | 0.386 | 16/25 |

未防禦臂四種淨化都是 1.000、0/25。**這 75 列裡 75 列的 argmin 都是 `id_norm`**，
`use_norm` 中位 0.974、從來沒有當過最小項，所以那 18/25 是真的打在身分上。
同一批的 `dir_norm` 中位 0.9145——**指令那條腿完全沒被碰過**。
**但顏色線的防禦圖不是自然照片**，那個數字不算防禦成功率。

## 正在跑的

遠端 basic-2 的卡 2–6，兩條鏈式腳本，都 `setsid nohup` 過：

| 腳本 | 做什麼 | 輸出 |
|---|---|---|
| `scripts/screen_variants.sh` | 五個變體逐一跑 `configs/evaluate_screen.json`（五類指令 × 三種子 × 兩臂，只有 identity 淨化） | `runs/field_screen/<variant>/shard*/` |
| `scripts/purify_variants.sh` | 接在後面，換成 `configs/evaluate_heldout.json` 的四道淨化 | `runs/field_purify/<variant>/shard*/` |

順序是 `flow_02 flow_04 flow_08 lab_field flow_16`（優先序，位移最小的先跑）。
一個變體吃滿五張卡再換下一個：時間不夠時損失的是排在後面的變體，不是每個
變體都只跑了一半。進度看 `runs/screen_chain.log` 與 `runs/purify_chain.log`。

**這兩條鏈會佔住五張卡很久**（完整淨化那一支五個變體約 14 小時）。選定操作點
之後應該把不需要的變體殺掉。

## 已經量掉、不要重做的路

1. **全域仿射、全域 3D LUT 都沒有防禦效果**（見上表）。
2. **指令進迴圈會高估防禦。** 舊路徑 2/15；拿掉指令後 1/25。
3. **「解完再縮」不是解受約束問題。** 同樣預算，約束進損失讓穿透率 1/25 → 18/25。
4. **NIQE 不能當自然度門檻。** 逐張比值中位 1.054，`114555` 是 0.837。
5. **CVaR95 尾端上限解決不了自然度。** 掃 25／30／35，產物依然奇怪。
6. **TV 單獨不夠。** `medium` 的 TV 比 `global` 低 21%，U16 反而更高、圖更糟。
7. **端點與 raw 越界的約束有效**，飽和已不是不自然的來源。

## 還沒試過的槓桿

| 槓桿 | 說明 |
|---|---|
| 指令完成那條腿 | `criterion.py` 明寫三種寫法都在圖上被推翻，`dir_norm` 因此不進目標函數。要一個量得準的 VQA 讀數。 |
| 可用性讀數 | `use_norm` 用 NIQE，從來不綁。需要有參考的一組，且逐張對圖校準過才准進 `TERMS`。 |
| 主流免疫法的目標 | PhotoGuard 的 **targeted** encoder attack、AdvDM／Mist 的擴散訓練損失。目前的 `enc` 是無目標的相對位移。 |
| 幾何 × 顏色串接 | `CompositeParam` 接得上，先各自量到底再談。 |
| EOT、更長的鏈 | 目前固定 `noise_seed=0`、`chain_steps=6`、`grad_steps=1`，真編輯是 50 步。 |
| 受成像限制的局部 PSF | 空間變化的小幅散焦，人眼讀成景深。 |

## 實驗規模的限制

5 張影像（其中 `195273` 是插畫）、3 顆種子、單一初始化。零次攻擊成功時三顆
種子的單側 95% 上界仍有 63%。要支持 ±10 個百分點的成功率敘述需要約 97 個
獨立身分，±5 點約 385。現在的數字只夠當 pilot。

## 環境與陷阱

- 遠端 `ssh -p 10102 nelson0314@server.basiclab.lab.nycu.edu.tw`（basic-2）、
  `-p 10101`（basic-1）。Repo 在 `/nfs/home/nelson0314/WACV-s4`，
  **整棵同步過去的，不是 git 工作區**，改完本機要自己 `scp`。
- **先 `source ~/env.sh`，再 `cd` 到 repo**：`env.sh` 最後一行會把工作目錄
  換到舊的 `~/WACV`。
- 本機 Python `C:/Users/nelso/miniconda3/envs/wacv/python.exe`，
  測試基準 `python -m pytest -q --ignore=runs` → **288 passed / 1 skipped**。
- 派工：`ssh host 'cd repo && bash scripts/free_cards.sh --assert N && { nohup ... & }'`。
  **一定要用 `&&`**；`&` 要在遠端 shell 內。派工後自己用
  `nvidia-smi --query-compute-apps=gpu_uuid,pid` 複驗。
- **ssh 連線很不穩**，`Connection closed/reset`、`kex_exchange_identification`
  常見，要寫重試迴圈；同時開太多連線會讓失敗率上升。
- `evaluate_defence.py` 只載入 `arms` 裡列到的臂，`start` 不在清單時不要求
  `*__start.png` 存在。
- 用 heredoc 在遠端寫 `.sh` 之後要 `sed -i 's/\r$//'` 再 `bash -n`。

## 報告

- 兩種色差預算：https://claude.ai/code/artifact/95164c77-d3ad-478b-a502-64262285f686
- 未見指令的防禦成績：https://claude.ai/code/artifact/1bbb2517-19d8-437a-acfb-0ab266aa5da2
- 受約束求解的全部產物：https://claude.ai/code/artifact/7a747d0b-0793-4e14-a234-e5af70d5f600
- 尾端上限的四檔：https://claude.ai/code/artifact/735dbe54-1967-41fc-93d8-289c3a3ddbbc
- 均勻度與防禦強度：https://claude.ai/code/artifact/76350b7c-07b6-4a1a-8186-dcbf98a4e552
- 色彩載體防禦講義（背景知識）：https://claude.ai/code/artifact/afb95ad3-d0ca-4e3b-952f-4afd3e0f4b8a
