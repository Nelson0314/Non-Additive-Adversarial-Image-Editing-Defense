# 主表：十二個免疫方法在同一條編輯管線上的比較

這個目錄是主表的交付面：**讀數、逐格影像、產生它們的程式、以及每個數字的出處**。
現況、各資料組的關係與接續指引在 `STATUS.md`。

## 目錄

| 路徑 | 內容 |
|---|---|
| `results/` | 主讀數、`metrics_*_union` 聯集、`aligned/` 等失真臂、跨編輯器的讀數 |
| `code/` | 管線腳本與共用的路徑解析 `paths.py`（詳見「程式」一節） |
| `docs/` | 指向主線 `../docs/` 的查閱表（出處文件已合併，正本在那裡） |
| `images/` | 逐格影像的本機部分鏡像（不入版控；遠端位置見 STATUS.md「遠端」） |
| `tests/` | 十二個條件的規格釘樁（`pytest tests/`，不需 GPU） |
| `STATUS.md` | 現況、資料組的關係、接續指引 |

## 讀數

主讀數(12 條件、ip2p/inpaint 兩場景，原生預算)：

| 檔 | 列數 | 內容 |
|---|---|---|
| `results/displacement.csv` | 768 | 位移＝`LPIPS(編輯(原圖), 編輯(防禦圖))`，含主體內／外分區 |
| `results/retention.csv` | 5,376 | 淨化後位移、淨增益、保留率，七道算子 |
| `results/metrics_fidelity_union.csv` | 96 | 保真的 FSIM／ΔE00／MSE |
| `results/metrics_displacement_union.csv` | 768 | 位移的 FSIM／MSE |
| `results/metrics_retention_union.csv` | 5,376 | 淨化後位移的 FSIM |
| `results/metrics_aesthetic_union.csv` | 104 | 七項無參考美學指標，含八張原圖的參照列 |
| `results/metrics_vmaf_union.csv` | 6,240 | VMAF(`libvmaf`，單幀餵法，`pairing` 欄分 fidelity/displacement/retention 三種配對) |
| `results/defence_<方法>.csv` | 8 / 檔 | 防禦圖對原圖的失真與該方法的求解設定（各篇原生預算） |
| `results/aligned/defence_<方法>_aligned.csv` | 8 / 檔 | 同十個方法縮到同一個 LPIPS 錨點（0.3344）的求解結果，見該目錄的 `README.md` |

跨編輯器（協定與主讀數不同，見 STATUS.md「跨編輯器」）：

| 檔 | 列數 | 內容 |
|---|---|---|
| `results/flux_full_<arm>.csv` | 32 / 檔，13 檔(分母 + 12 條件) | FLUX.1-Kontext 編輯輸出的 id_orig／arcface_orig，guidance 3.5、1024×1024 |
| `results/displacement_flux.csv` | 384 | FLUX 全表的位移，欄位與 `displacement.csv` 同組 |
| `results/sdedit_preview*.csv`、`sdedit_preview_stable-diffusion-*.csv` | 各數格到數十格 | SD 1.5／2.1-base 在不同 strength／guidance 下的小樣本探索，SDEdit 這條線最終被放棄，原因與資料見 STATUS.md |
| `results/ultraedit_full/<arm>.csv` | 256 / 檔，13 檔 | UltraEdit（SD3）全表的逐格編輯，未淨化 + 7 道算子，id_orig／arcface_orig |
| `results/displacement_ultraedit.csv`、`retention_ultraedit.csv` | 384、2,688 | UltraEdit 全表的位移與保留率，由 `edit_displacement.py`／`edit_retention.py` 原樣算出 |
| `results/sd_family_*.csv`、`sd_family_offtarget_*.csv` | 數十至兩百格 | SDXL-IP2P 與 UltraEdit 的參數／句型掃描（未防禦影像），及指令以外改動的讀數（背景／主體 ΔE00、LPIPS）；`sd_family_offtarget_ip2p_si18_reference.csv` 是主表 ip2p 的對照 |
| `results/passthrough.csv` | 704 | 穿透拆解（ip2p，原生 12 條件 + 等失真 10 條件），見 `results/PASSTHROUGH.md` |
| `results/flux_preview*.csv` | 各 4 格 | FLUX guidance／true-CFG 小樣本探索 |

十二個條件：`dct_shield_y`、`mist`、`dct_shield`、`photoguard_linf`、`danp`、
`sifm`、`dayn`、`dia_pt`、`dia_r`、`photoguard_c`、`color`、`diffvax`。
顏色那一列原為 `colour_curve_ours`（數值留在 commit `0dd243b`），現為 lab 的現行方法 `color`；
等失真臂仍以舊顏色列的 0.3344 為錨點（見 STATUS.md）。
每個條件 8 影像 × 4 指令 × 2 場景 ＝ 64 格。

## 協定

| | ip2p | inpainting |
|---|---|---|
| 受害模型 | `timbrooks/instruct-pix2pix` | `runwayml/stable-diffusion-inpainting` |
| guidance | `s_t` 7.5、`s_i` 1.8 | 7.5 |
| 未防禦對照臂 | `ip2p_si18` | `inpaint_undefended` |

兩個場景共用種子 20260812、50 步、512²。LPIPS 走 `piq.LPIPS()`（VGG）。

**束縛種類不同的列不可直接比大小**，`eps` 與 `eps_pixel01` 兩欄都不可跨列比：
十二列分成五種束縛（`linf`、`l2`、`dct_coeff_linf`、`delta_e00_cap`、`none`），
逐列的單位見 `../docs/reference/BASELINE_PROVENANCE.md` §「`eps_pixel01` 欄的
單位，逐列」，值域換算的逐篇出處見 `../docs/reference/SOURCE_AUDIT.md` §10。
同檔的規則 1：引用一個數字
就要連它的協定一起引用。

`rotate15` 的角度、以及幾何類分區讀數的一個已知限制，見 `../docs/EVALUATION.md`。

## 程式

`code/` 的腳本執行時需要主線目錄的 `src/` 套件與資料集。
`code/paths.py` 把搬動切斷的兩件事接回去：影像與 CSV 走 `main_table/` 內部
路徑，`src.*` 與 `data/portraits`、`data/targets` 走主線目錄。主線目錄依序找
上一層（併進主線之後主線就是上一層）、`../anti-purification` 與曾用名
`../non-additive-frequency`，也可用環境變數
`IMMUNISATION_SOURCE_HOME` 指定；找不到就中止並列出找過哪些路徑，不回退到
猜測值。依賴的模組（路徑相對於主線目錄）：

| 類別 | 模組 |
|---|---|
| baseline 攻擊 | `src/baselines/{pgd,photoguard,mist,dia,dayn,sifm,danp,diffvax,dct_shield,jpeg_codec}.py` |
| 指標 | `src/metrics/{suite,standard,regional,identity,acutance}.py` |
| 淨化 | `src/purify/{ops,diffpure,impress,adverse_cleaner,freq_grid}.py` |
| 受害模型 | `src/models/{ip2p,sd}.py` |
| 工具 | `src/utils/{io,artifacts,device}.py` |

流程順序：`edit_preflight` → `defence_run`（外部十一條件）與
`immunise_as_condition`（顏色那一列）→ `edit_displacement` → `purify_run` →
`edit_retention` → `metrics_union`（五個 stage，含 VMAF）。

跨編輯器探索另外幾支，讀寫的目錄版面跟上面這條主線不同，互相獨立：

| 腳本 | 做什麼 |
|---|---|
| `edit_sdedit_preview.py` | SD 1.x／2.x 的 SDEdit 小樣本，`--strengths`／`--guidances`／`--model` 三個維度都能單獨掃 |
| `edit_sd_family_preview.py` | SD 1.4／1.5 以外的 SD 系列編輯器（`--editor sdxl-ip2p`／`sd3-ultraedit`／`sdxl-img2img`／`sd3-img2img`）的參數與句型（`--prompt-sets`）網格，CSV 可續跑，另產一張對照圖 |
| `ultraedit_sd3_pipeline.py` | UltraEdit 的 SD3 指令式編輯管線，移植自作者的 diffusers fork |
| `sd_family_offtarget_readout.py` | 編輯結果對原圖的背景／主體 ΔE00 與整張 LPIPS（指令以外的改動） |
| `edit_ultraedit_full.py` | UltraEdit 全表（13 arm × 未淨化與 7 道算子），輸出沿用主表版面 |
| `color_row_chain.sh` | `color` 取代顏色列的整條鏈：`main` 模式跑匯入、編輯、淨化、淨化後重編與 UltraEdit，`flux` 模式跑 FLUX |
| `edit_flux_preview.py` | FLUX.1-Kontext 全表(`--arm undefended`／`--arm <條件>`，每個 arm 32 格，CSV 逐格 append 可續跑) |
| `edit_displacement_flux.py` | FLUX 全表的位移，欄位對齊 `displacement.csv` |
| `flux_full_queue_a.sh` | 依序跑一串 arm 用的排隊腳本，一張卡一份 |
| `passthrough_readout.py` | 穿透拆解，見 `results/PASSTHROUGH.md` |

## 影像原檔

主表的逐格原檔是 **512×512 RGB PNG**（FLUX 為 1024×1024），版面如下（相對於本機 `images/`；
遠端對應主線 `runs/` 的同名目錄）。

| 用途 | 格數 | 路徑式樣 |
|---|---|---|
| 未防禦編輯（分母） | 64 | `images/edit_preflight/{ip2p_si18｜inpaint_undefended}/<圖>__p<N>.png` |
| 防禦後編輯 | 768 | `images/edit_defended/<方法>/<場景>_<方法>/<圖>__p<N>.png` |
| 淨化後·未防禦分母 | 448 | `images/edit_purified/undefended/<算子>/<場景>_undefended_<算子>/<圖>__p<N>.png` |
| 淨化後·防禦 | 5,376 | `images/edit_purified/<方法>/<算子>/<場景>_<方法>_<算子>/<圖>__p<N>.png` |
| 防禦圖 | 96 | `images/defence_portraits/<方法>/<圖>__<方法>__def.png` |
| 原圖 | 8 | `images/defence_portraits/mist/<圖>__orig.png` |
| 重繪遮罩 | 8 | `images/masks/<圖>.png` |

原圖與遮罩在主線目錄另有一份：`../data/portraits/` 底下的
`man/`、`woman/`、`masks/`。`images/masks/` 與該處的 `masks/` 八張逐位元相同。
