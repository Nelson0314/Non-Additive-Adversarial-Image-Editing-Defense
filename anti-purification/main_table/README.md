# 主表：十二個免疫方法在同一條編輯管線上的比較

這個目錄是主表的交付面：**報告頁、讀數、逐格影像、產生它們的程式、以及每個
數字的出處**。建立進度、三組資料的關係與接續指引在 `STATUS.md`。

## 目錄

| 路徑 | 內容 |
|---|---|
| `report/` | 五份報告頁的原始檔，每份自足（見下方「報告頁」一節逐份列出） |
| `results/` | 主讀數、`metrics_*_union` 聯集、`aligned/` 等失真臂、跨編輯器探索的讀數，共 40 張 CSV |
| `code/` | 管線腳本與共用的路徑解析 `paths.py`（詳見「程式」一節） |
| `docs/` | 指向主線 `../docs/` 的查閱表（出處文件已合併，正本在那裡） |
| `images/` | 逐格影像（版控範圍見 `.gitignore`，只有 CSV 進版控） |
| `tests/` | 十二個條件的規格釘樁（`pytest tests/`，不需 GPU） |
| `STATUS.md` | 建立進度、三組資料的關係、接續指引 |

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

跨編輯器探索(FLUX 全表 + SDEdit 診斷，協定與主讀數不同，見 STATUS.md「跨編輯器遷移」)：

| 檔 | 列數 | 內容 |
|---|---|---|
| `results/flux_full_<arm>.csv` | 32 / 檔，13 檔(分母 + 12 條件) | FLUX.1-Kontext 編輯輸出的 id_orig／arcface_orig，guidance 3.5、1024×1024 |
| `results/displacement_flux.csv` | 384 | FLUX 全表的位移，欄位與 `displacement.csv` 同組 |
| `results/sdedit_preview*.csv`、`sdedit_preview_stable-diffusion-*.csv` | 各數格到數十格 | SD 1.5／2.1-base 在不同 strength／guidance 下的小樣本探索，SDEdit 這條線最終被放棄，原因與資料見 STATUS.md |
| `results/flux_preview*.csv` | 各 4 格 | FLUX guidance／true-CFG 小樣本探索 |

十二個條件：`dct_shield_y`、`mist`、`dct_shield`、`photoguard_linf`、`danp`、
`sifm`、`dayn`、`dia_pt`、`dia_r`、`photoguard_c`、`colour_curve_ours`、`diffvax`。
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
| `edit_flux_preview.py` | FLUX.1-Kontext 全表(`--arm undefended`／`--arm <條件>`，每個 arm 32 格，CSV 逐格 append 可續跑) |
| `edit_displacement_flux.py` | FLUX 全表的位移，欄位對齊 `displacement.csv` |
| `flux_full_queue_a.sh` | 依序跑一串 arm 用的排隊腳本，一張卡一份 |
| `report_main_data.py` | 產生 `report/main/` 的 `data.js` 與縮圖(原生 12 條件主表 + FLUX 一節) |
| `report_matrix_data.py` | 產生 `report/aligned_matrix.html` 的 `data.js`(等失真臂) |
| `passthrough_readout.py` | 穿透拆解，見 `results/PASSTHROUGH.md` |

## 影像原檔

逐格原檔是 **512×512 RGB PNG**，全部在 `images/`，共 11,848 張。主表用到的是
其中 6,768 格，其餘是其他臂與逐條件各存一份的 `__orig` 副本（96 張，內容同一
組 8 張原圖）。

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

## 報告頁

`report/` 底下五個資料夾各是一份**自足**的報告：資料與影像都在同一個目錄裡
（相對路徑引用 `data.js` 與 `img/`，或直接內嵌），本機直接雙擊 `index.html`
就能看，不像舊版那樣依賴已發布 artifact 才能解析影像 URL。

| 目錄 | 內容 | 產生方式 |
|---|---|---|
| `report/main/` | **主表**，原生預算 12 條件：防禦圖矩陣、編輯矩陣、VMAF 圖表、fidelity／displacement／retention 三張指標表、美術指標真圖對照、FLUX 獨立一節 | `code/report_main_data.py --out report/main` |
| `report/aligned_matrix.html`(+`data.js`、`img/`，與 `report/main/` 同一層) | **等失真臂**，10 條件縮到同一 LPIPS 錨點 + colour_curve_ours | `code/report_matrix_data.py --out report/data.js`(影像另外手動轉，見該檔 docstring) |
| `report/flux_full/` | FLUX 全表樣張，4 影像 × 4 指令 × 13 arm，每格標 id_orig | 手動組的縮圖 + inline data，來源見 `images/flux_full/` |
| `report/editor_check/` | SD 1.5／SD 2.1(含修正前的 v-prediction 版本，留作對照)／FLUX 在同一組指令下的實際輸出，判斷編輯器本身有沒有站得住 | 手動組 |
| `report/sd_samples/` | SD 1.5(strength 0.3–0.6)與 SD 2.1-base(0.2／0.3／0.5)並排，判斷指令有沒有被執行 | 手動組 |

五份的協定各不相同（原生預算 vs 等失真 vs FLUX 自己的 guidance/解析度），
**引用數字時連同來源報告與協定一起引用**，不要跨報告直接比大小。
