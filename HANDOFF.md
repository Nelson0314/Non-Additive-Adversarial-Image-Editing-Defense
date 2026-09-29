# 交接

程式與數值在 `anti-purification/`，工作規則見該目錄的 `CLAUDE.md`。

**兩條線要分清楚：**

| | 主表（本檔的主題） | 顏色載體的改良實驗 |
|---|---|---|
| 問題 | 十二個免疫方法在同一條管線上誰推得動編輯 | 本專案自己的顏色方法要怎麼變強 |
| 位置 | `anti-purification/`（遠端 `image-immunization`） | **已刪除**，紀錄留在 `COLOUR_LINE.md` |
| 狀態 | **已完成，12 條件 × 64 格全齊** | **已停止並移除**（使用者裁定） |
| 交集 | 主表的顏色列已換成 lab 的現行方法 `color`（使用者指示；舊列 `colour_curve_ours` 留在 commit `0dd243b`） | 見 `anti-purification/main_table/STATUS.md` |

改良實驗**不得寫入主目錄**；主表的數字只在使用者指示時更換（顏色列已依指示換成 `color`）。詳見「顏色載體的改良實驗」一節。

**貼片線的檔案不要動**：`src/defense/{material_patch,patch_canvas,print_patch,outside_terms}.py`、
`scripts/{immunise_patch,patch_probe,print_probe}*`、`configs/immunise_patch*.json`、
`runs/{patch_*,print_*,smiley_*,off_*,ok_*,universal}`。那兩個 session 已離線，
但離線不等於放棄。

---

## 現在的位置：主表已完成

十二個條件在同一組影像、同一條編輯管線上，防禦圖 → 防禦後編輯 → 七道淨化
→ 淨化後編輯全部齊全，**零失敗，沒有破洞**。

| 階段 | 狀態 |
|---|---|
| 資料集 | 人像 8 張完成；動物 12 張缺 `prompts.yaml`，跑不了 |
| 編輯管線 | 完成並定案 |
| 防禦圖 | 完成。12 條件 × 8 張 |
| 防禦後編輯 | 完成。12 條件 × 2 場景 × 32 格 |
| 淨化與淨化後編輯 | 完成。12 條件 × 7 算子 × 2 場景 |
| 位移讀數 | 完成。`runs/edit_defended/displacement.csv`，**768 列**（12 × 64） |
| 淨增益 | 完成。`runs/edit_purified/retention.csv`，**5,376 列** |
| **逐格看圖** | **還沒做。** 位移是數字，成立與否本專案的規矩是看圖 |

兩張表的條件集合完全一致，沒有「有位移卻缺淨增益」的條件。

### 幾何類的淨增益是用哪一種遮罩算的

`crop_resize 0.1` 與 `rotate 15` 會把影像變形。**主體內／外的分區需要同樣被變形
過的遮罩**，否則遮罩與它要切的影像不對位。

`edit_retention.py` 的 `purified_mask()` 做這件事，**進版於 commit `199b7de`**。
判準因此可以自己查：**一份 `retention.csv` 若沒有在 `199b7de` 之後重跑過，
它的 `crop_resize 0.1` 與 `rotate 15` 共 1,536 列的
`disp_purified_subject` 與 `disp_purified_background` 就是用未變換的遮罩算的。**
全圖的那幾欄不受遮罩影響。

`anti-purification/main_table/results/retention.csv` 中 `color` 以外的 11 個條件尚未重跑；
`color` 列與 UltraEdit 的保留率是在 `199b7de` 之後算的。使用者裁定之後再跑，不急著送遠端。

---

## 編輯管線（定案，不要再改）

`scripts/edit_preflight.py`。兩個場景共用種子 20260812、50 步、512²。

| | ip2p | inpainting |
|---|---|---|
| 受害模型 | `timbrooks/instruct-pix2pix` | `runwayml/stable-diffusion-inpainting` |
| 呼叫 | `IP2PWrapper.edit` | 官方 `StableDiffusionInpaintPipeline` |
| guidance | `s_t` 7.5、**`s_i` 1.8** | 7.5 |
| 遮罩 | 無 | dilate 4，白＝重繪 |

**`s_i` 是 1.8 不是封裝預設的 1.5**，出處是 helmet 指令的 11 組設定掃描
（`runs/ip2p_helmet_sweep/`，88 格逐格看圖）：1.5 在八張裡有三張多出第二個人、
四張換臉；1.8 在三個 `s_t` 上都把這兩件事歸零；2.2 開始畫不出帽體。
常數在 `scripts/edit_preflight.py` 的 `IP2P_EDIT_IMAGE_GUIDANCE`，
**不要改 `src/models/ip2p.py` 的 1.5**（貼片線也在用）。

分母是未防禦的同設定臂：ip2p 用 `ip2p_si18`、inpaint 用 `inpaint_undefended`。
`runs/edit_preflight/` 裡另外四個臂設定不同，**不可以當這一批的分母**。

---

## 主表的十二個條件

「本專案」一列是 `colour_curve_ours`（AdvCF 那條單調分段線性 RGB tone curve，
`pieces 64`、`radius 5.0`、900 步、`deltae_cap 16.0`），其餘十一列是外部方法。

### 失真（防禦圖對原圖，中位數）

| 條件 | PSNR | LPIPS | L∞ | 束縛種類 |
|---|---|---|---|---|
| `colour_curve_ours` | 16.67 | 0.3367 | 0.341 | ΔE00 ≤ 16（整圖） |
| `photoguard_c` | 40.90 | 0.4471 | 0.140 | L2 = 8 |
| `dia_r` | 40.62 | 0.3539 | 0.025 | L∞ |
| `dia_pt` | 37.47 | 0.4417 | 0.025 | L∞ |
| `dayn` | 33.67 | 0.4810 | 0.030 | L∞（κ=0.06 於 `[-1,1]`） |
| `dct_shield` | 29.84 | 0.5815 | 0.247 | 量化域 |
| `mist` | 26.66 | 0.6574 | 0.063 | L∞ 16/255 |
| `photoguard_linf` | 26.49 | 0.6746 | 0.063 | L∞ 16/255 |
| `dct_shield_y` | 24.45 | 0.6584 | 0.408 | 量化域（Y） |
| `sifm`／`danp` | 逐欄見 CSV | | | L∞ 0.03（`[0,1]`） |
| `diffvax` | 34.71–38.00 | 0.030–0.056 | — | **無硬性預算** |

**束縛種類不同的列不可以直接比大小。** `eps` 欄也不可跨列比——`sifm`／`danp`
的值域是 `[0,1]`、`dayn` 是 `[-1,1]`，`eps` 0.03 與 0.06 其實是同一個像素幅度。
比預算一律看 `eps_pixel01`。

### 位移（主讀數，中位數）

`位移 = LPIPS(編輯(原圖), 編輯(防禦圖))`。

| 條件 | ip2p 全圖／主體內 | inpaint 全圖／主體內 | blocked |
|---|---|---|---|
| `dct_shield_y` | 0.6411／0.6714 | 0.6287／0.6166 | 31/32、2/32 |
| `mist` | 0.6378／0.6157 | 0.6406／0.6211 | 32/32、21/32 |
| `dct_shield` | 0.6242／0.6543 | 0.5723／0.5342 | 29/32、5/32 |
| `photoguard_linf` | 0.5444／0.5358 | 0.6281／0.6233 | 27/32、16/32 |
| `danp` | 0.5419／0.4856 | 0.5070／0.4705 | 19/32、6/32 |
| `sifm` | 0.5309／0.5006 | 0.5127／0.4651 | 19/32、9/32 |
| `dayn` | 0.5101／0.4745 | 0.4915／0.4227 | 18/32、8/32 |
| `dia_pt` | 0.5014／0.4547 | 0.4407／0.3838 | 19/32、4/32 |
| `dia_r` | 0.3955／0.3213 | 0.3498／0.2246 | 7/32、1/32 |
| `photoguard_c` | 0.3881／0.3446 | 0.4341／0.3664 | 7/32、1/32 |
| `colour_curve_ours` | 0.3821／0.3867 | 0.4431／0.3317 | 3/32、4/32 |
| `diffvax` | 0.0723／0.0629 | 0.3380／0.0348 | 0/32、8/32 |

### 淨增益（保留率＝淨化後位移 ÷ 未淨化位移）

| 條件 | jpeg80 | jpeg50 | jpeg30 | blur1 | blur2 | crop | rotate15 | 非幾何均 |
|---|---|---|---|---|---|---|---|---|
| `colour_curve_ours` | 1.018 | 1.045 | 1.065 | 0.980 | 0.996 | 1.053 | 0.906 | **1.021** |
| `mist` | 0.920 | 0.768 | 0.668 | 0.734 | 0.579 | 0.969 | 0.872 | 0.734 |
| `photoguard_linf` | 0.912 | 0.758 | 0.694 | 0.688 | 0.541 | 1.010 | 0.928 | 0.719 |
| `dayn` | 0.833 | 0.717 | 0.648 | 0.681 | 0.507 | 0.912 | 0.850 | 0.677 |
| `dia_pt` | 0.721 | 0.554 | 0.482 | 0.562 | 0.334 | 0.896 | 0.799 | 0.531 |
| `dct_shield_y` | 1.016 | 0.476 | 0.363 | 0.408 | 0.197 | 0.911 | 0.822 | 0.492 |
| `dia_r` | 0.674 | 0.522 | 0.453 | 0.478 | 0.224 | 0.864 | 0.753 | 0.470 |
| `photoguard_c` | 0.677 | 0.517 | 0.470 | 0.429 | 0.252 | 0.964 | 0.859 | 0.469 |
| `dct_shield` | 0.414 | 0.243 | 0.231 | 0.419 | 0.220 | 0.902 | 0.772 | 0.306 |
| `diffvax` | 0.927 | 0.902 | 0.862 | 0.880 | 0.734 | 0.597 | 0.689 | 0.861 |

`crop_resize0.1` 與 `rotate15` 是**幾何類**，取景本身被改掉，讀數同時含
「防禦被洗掉」與「畫面被移動」，不與其餘五道混著平均。`sifm`／`danp` 的
保留率在最後一次重算裡，逐欄見 CSV。

**`diffvax` 的兩件事報表上分開標**：它是十二個條件裡**唯一需要遮罩**的
（擾動只存在於重繪區之外，而 inpainting 的重繪區正好是主體之外，所以它的
主體內／外分區與其他條件不是同一個幾何關係），而且**沒有硬性 `L∞` 預算**
（輸出層是 1×1 Conv、無 activation），掛不上其餘條件的失真錨點——它的保真
LPIPS 是 0.030–0.056，其他條件是 0.34–0.67，差 6 到 20 倍。它是前饋式免疫器，
每張防禦圖不到 1 秒（`colour_curve_ours` 1740 秒、`photoguard_linf` 6481 秒）。
接法見 `src/baselines/diffvax.py` 的「怎麼接進本專案的評測」與
`scripts/defence_run.py` 的 `FEEDFORWARD_CONDITIONS`。

**顏色曲線是唯一淨化後 `blocked` 反而上升的條件**（未淨化 7，七道之後 14–19），
其餘全部下降。照報，不作判準。

---

## 已經量到的，不要重做

1. **遮罩膨脹決定 inpainting 的成敗。** 16 px 在主體旁邊留一圈原始背景，
   模型會順著它延伸；同一句指令在 16 px 下什麼都不出現，4 px 下場景畫得出來。
2. **9 通道 inpainting 不可以在取樣迴圈裡逐步把遮罩外貼回。**
   `tests/test_inpaint_chain.py` 釘住。
3. **SD v1.4 的 img2img 不能當受害模型**（人被換掉，身分失去分母）。
4. **遮住臉的配件指令不能用身分當讀數。**
5. **CLIP 對齊增益會與看圖相反**，只當解釋。
6. **CLIPSeg 遮罩中位 IoU 0.929 但 p10 只有 0.058**，失手時沒有症狀，
   只有看圖攔得下來。主體內／外分區用的就是它的補集。
7. **成本推估會低估。** 實測秒/圖：`colour_curve_ours` 約 1740（900 步）、
   `photoguard_c` 6406、`photoguard_linf` 6481、`dayn` 485、`dct_shield` 159、
   `mist` 78、`dia_pt` 69、`tdae` 13350。

---

## 這一輪新增的三個發現

### 一、TDAE 在本專案的威脅模型下，照論文起點跑會交付原圖

`L = ‖edit(x_adv) − y₀‖₂`，而 `y₀ = edit(x₀)` 用固定的編輯噪聲。論文
Algorithm 1 第 1 行是 `δ_v ← 0`，於是 δ=0 時兩者逐位元相同、`L ≡ 0`；
`‖v‖₂` 在零點的梯度是 0，`s`、`z`、`k` 跟著全為零，`g_FDM` 精確為零。
CPU 實測：恆等起點 grad absmax **0.0**，離開恆等之後 0.132。遠端實跑 15 分鐘
仍停在 `step 0  L=0.000000e+00`。

改用 `uniform_linf` 隨機起點可以跑（四張 PSNR 26.12–27.07），但**使用者裁定
不接受這個偏離**，整個條件連同結果移除。完整理由在
`docs/reference/AUDIT_TDAE.md` §8。**這不是對該篇的評價**，是本專案的威脅模型
（防禦方看不到指令 → 空 prompt → `y₀` 是無指令重建）與論文起點互相作用的結果。

### 二、求解端在 900 步下也不可重現

原本的紀錄是「求解端可重現、評估端不可」。顏色線的把關診斷量到：**同一份設定、
同一顆種子、只是跑在不同卡上**，同一張 `man_00` 的全圖 ΔE00 是 15.93／15.70／
12.51，LPIPS 是 0.3856／0.2403／0.2789。900 步、bf16 下這個結論要改寫。
後果是所有配對比較的訊噪比被壓低，**批內對照臂比以前更必要**。

### 三、濾鏡的色差與「整圖一致」都成立

交付的八張 ΔE00 是 14.15–16.00，**0/8 超過上限**，六張貼著 16。映射的全域性
直接從 PNG 反查驗證過：把「原圖某通道值」當鍵、「防禦圖同位置的該通道值」
當值，八張 × 三通道 × 全部像素，**同一輸入色階的輸出寬度最大值是 0**。
看起來局部劇烈是因為 64 段裡某幾段斜率很陡，而落在那個輸入區間的像素在畫面上
通常聚在一起（膚色、頭髮、某件衣服）——這是全域曲線的正常後果。

---

## 不在主表裡的方法，以及為什麼

| 方法 | 狀態 |
|---|---|
| `uap_semantic` | **已徹底移除。** 它是通用擾動，要 10,000 組 image-prompt 訓練集（替代方案是已刪除的 `data/lo_aligned` 24 張，差 417 倍），且 `prepare`／`loss_fn` 刻意拋 `NotImplementedError`。模組、測試、腳本、審計文件四個檔已刪，報表腳本只留一行註解記錄移除理由 |
| `tdae` | 裁定不進比較。依論文重建的模組仍在 `src/baselines/tdae.py`，理由見 `docs/reference/AUDIT_TDAE.md` §8：本專案的威脅模型（防禦方看不到指令 → 空 prompt → `y₀` 是無指令重建）與論文的 `δ_v ← 0` 起點相互作用，使 `L ≡ 0`、梯度精確為零。**那不是對該篇的評價** |
| AdvCF | 裁定不當 baseline（分類器場景，與本專案的威脅模型無關）。它的 tone curve 參數化本身是本專案顏色線的載體，兩件事要分開 |
| `colour_field` | 裁定整組移除，產物已刪。程式仍在 `src/defense/color_field.py`，因為 `carrier_search.py`、`lab_offset_field.py`、`tests/test_composite.py` 仍 import 它 |

---

## 顏色載體的改良實驗：**已停止並移除**

主表是「十二個條件的外部比較」，那一條線是「本專案自己的顏色方法要怎麼變強」。
兩者共用編輯管線與判定門檻，但產物、目錄、結論完全分開。

**該線已由使用者裁定停止，工作目錄（本機 `colour-lab/`、遠端
`/nfs/home/nelson0314/WACV-colour-lab`）已整個刪除。** 完整紀錄——做過的每一批、
所有數字、硬約束、被修正的說法、參考文獻、未完成的東西——在根目錄的
**`COLOUR_LINE.md`**，另有四個報告頁網址列在該檔開頭。

主表裡的 `colour_curve_ours` 是那條線**最後交付的操作點**（AdvCF tone curve、
`pieces 64`、`radius 5.0`、900 步、`deltae_cap 16.0`），基準載體
`src/defense/color_param.py` 與目標函數、讀數腳本在本目錄都有各自的一份，
**主表不受影響，照現狀引用**。

做完九批，沒有一批被裁定為可交付的改良。

---

## 環境

- 遠端兩台：`ssh -p 10101`（basic-1，8 張卡）／`-p 10102`（basic-2，7 張卡），
  `nelson0314@server.basiclab.lab.nycu.edu.tw`。**兩台都要查**，早先只查
  basic-2 漏掉過 basic-1。
- repo 在 `/nfs/home/nelson0314/image-immunization`。
  home 跨機同步，兩台都看得到。
- **先 `source ~/env.sh` 再 `cd`**（env.sh 會把工作目錄切走）。
- `HF_HOME=/var/cache/huggingface`（機器本地，不是 NFS 那份）。
- **所有運算送遠端。** 卡是多人共用：`bingo`、`chhsu0924`、`briankuo93`、
  `cylin` 都在用。判定「空」要兩個條件同時成立：沒有別人的 compute app、
  已用記憶體 < 1 GB。**全局卡數由使用者逐次授權；未說明或說明不清時預設 6 張，所有 session、主機與排程合計。**
- 本機 Python `C:/Users/nelso/miniconda3/envs/wacv/python.exe`。

### 這一輪踩到的坑

1. **`nohup setsid ... &` 要把三個 fd 都導掉**，否則 ssh 掛著不返回，
   重試迴圈會把同一個工作啟動多次。
2. **CSV 的合併鍵是 `arm`**，每個臂一定要給不同的 `--suffix`。
3. **改名時漏掉「結尾剛好是 `_colour`」這種寫法**，`arm` 欄指到不存在的目錄，
   位移只算了一個條件就停住——而前一個條件已經寫進 CSV、版面照樣產出、
   鏈也照樣印完成。**靜默失效的典型形狀。**
4. **`pgrep -f` 會匹配到自己的 ssh 指令字串**，判定「有沒有在跑」要看真正的
   python 行程（`ps -o args=` 配 `venvs/wacv/bin/python`）。
5. **`--out` 目錄存在就拒絕啟動**（`paper_baseline.py`／`ip2p_helmet_sweep.py`），
   所以 log 不要寫進輸出目錄，否則自己的 mkdir 會把自己擋掉。
6. **報告頁單一版本上限 64 MB**。68 張版面在 256 px／品質 84 下是 63 MB，
   已降到 208 px／76，約 36 MB。

---

## 待辦（依優先序）

1. **逐格看圖。** 主表的 768 格從來沒有人逐格看過。位移與保留率都是數字，
   而本專案的規矩是「成立與否看圖」。68 張版面已經產好（`runs/report/sheets/`），
   掛在報告頁上。**這是主表最大的缺口。**
2. **等失真錨點怎麼處理。** 十二個條件掛在四種不同的預算上——ΔE00（本專案）、
   L2（`photoguard_c`）、L∞（多數）、量化域（`dct_shield`），而 `diffvax` 沒有硬性預算。
   **排名不可直接比大小**，`eps` 欄也不可跨列比（值域不同）。比預算一律看 `eps_pixel01`。
   使用者說先不管，但這是投稿時一定會被問的。
3. **動物那一組的 `prompts.yaml`。** 12 張影像已備好，缺這個檔就跑不了 inpainting。
   動物的 `content` 與編輯指令怎麼定是協定層決定，留給使用者。
4. **報告頁要用現行帳號重發。** 現有連結是更早的帳號發布的，換帳號後開不了。
   資料與版面都在遠端 `runs/report/`，重發即可。

### 清理狀態

`__pycache__` 已全部清除。pytest 的暫存目錄（`anti-purification/.tmp/`、
`runs/ncf_cpu_test_tmp/`、根目錄的 `.tmp/` 與 `colour-lab/.pytest_tmp_*`）
Windows ACL 拒絕存取，`takeown` 需要管理員權限才改得動擁有權，目前仍刪不掉。
不影響任何測試或產出。

---

## 檔案在哪

| 東西 | 路徑 |
|---|---|
| 編輯管線 | `scripts/edit_preflight.py` |
| 防禦圖求解 | `scripts/defence_run.py` |
| 位移讀數 | `scripts/edit_displacement.py` |
| 淨化 | `scripts/purify_run.py` |
| 淨增益 | `scripts/edit_retention.py` |
| 版面與報告頁 | `scripts/build_main_report.py`、`build_main_report_page.py` |
| helmet 掃描 | `scripts/ip2p_helmet_sweep.py`、`ip2p_helmet_contactsheet.py`、`ip2p_helmet_review.py` |
| 顏色線整理成條件 | `scripts/immunise_as_condition.py` |
| baseline 出處總表 | `docs/reference/BASELINE_PROVENANCE.md` |
| 各 baseline 逐行查證 | `docs/reference/AUDIT_*.md` |
| 指標與淨化的比較方式 | `docs/EVALUATION.md` |

## 報告頁

- **主表（現行）** <https://claude.ai/artifact/8DF1HCnyHucxu6y9xJKK9p>
- 四臂對照（inpainting 的決定性實驗）<https://claude.ai/artifact/15aEdYNof2jmrzpWNfrXkS>
- 未防禦編輯預檢 <https://claude.ai/artifact/DKEP8fKM6EDY73tQDm7KJ7>
- 提案資料集 <https://claude.ai/artifact/G9VdG9gbNqnrT7LzoUS3kb>
- helmet 指令掃描 <https://claude.ai/artifact/K2vxzLCh57CMWNiiNRtFFC>
- 顏色載體防禦讀數（舊 trio 資料，只當協定參照）<https://claude.ai/artifact/Xt1MofiTiodn4qWcpJSik5>
