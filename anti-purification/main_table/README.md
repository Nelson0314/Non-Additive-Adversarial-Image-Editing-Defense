# 主表：十二個免疫方法在同一條編輯管線上的比較

這個目錄是主表的交付面：**報告頁、讀數、逐格影像、產生它們的程式、以及每個
數字的出處**。

## 目錄

| 路徑 | 內容 |
|---|---|
| `report/` | 已發布報告頁的原始檔 `index.html`，自足（內嵌資料、資產對照、CSS、JS） |
| `results/` | 六張讀數 CSV 與十二張求解設定 CSV |
| `code/` | 七支管線腳本與共用的路徑解析 `paths.py` |
| `docs/` | 指向主線 `../docs/` 的查閱表（出處文件已合併，正本在那裡） |
| `images/` | 逐格影像 11,848 張 |
| `tests/` | 十二個條件的規格釘樁（`pytest tests/`，不需 GPU） |

## 讀數

| 檔 | 列數 | 內容 |
|---|---|---|
| `results/displacement.csv` | 768 | 位移＝`LPIPS(編輯(原圖), 編輯(防禦圖))`，含主體內／外分區 |
| `results/retention.csv` | 5,376 | 淨化後位移、淨增益、保留率，七道算子 |
| `results/metrics_fidelity_union.csv` | 96 | 保真的 FSIM／ΔE00／MSE |
| `results/metrics_displacement_union.csv` | 768 | 位移的 FSIM／MSE |
| `results/metrics_retention_union.csv` | 5,376 | 淨化後位移的 FSIM |
| `results/metrics_aesthetic_union.csv` | 104 | 七項無參考美學指標，含八張原圖的參照列 |
| `results/defence_<方法>.csv` | 8 / 檔 | 防禦圖對原圖的失真與該方法的求解設定 |

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
逐列的單位見 `docs/BASELINE_PROVENANCE.md` §「`eps_pixel01` 欄的單位，逐列」，
值域換算的逐篇出處見 `docs/SOURCE_AUDIT.md` §10。同檔的規則 1：引用一個數字
就要連它的協定一起引用。

`rotate15` 的角度、以及幾何類分區讀數的一個已知限制，見 `docs/EVALUATION.md`。

## 程式

`code/` 的七支腳本是複本，執行時需要主線目錄的 `src/` 套件與資料集。
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
`edit_retention` → `metrics_union`（四個 stage）。報告頁的排版腳本
（`report_figures`、`report_data`、`report_page`）不在 `code/`，見「報告頁」一節。

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

`report/index.html` 是已發布頁面的原始檔，**自足**：完整資料在
`<script id="payload">`、136 筆資產對照在 `<script id="assets">`，CSS 與 JS 全部
內嵌。頁面的影像走資產 URL（`/_blob/<id>`），那些 URL **只在已發布的 artifact
內解析**，所以本機直接開 `index.html` 看不到圖。

頁面用的 136 張載體圖（128 張逐格圖 ＋ 8 張防禦圖條）存在 artifact 的資產庫，
本機沒有副本；排出它們的 `report_figures.py` 已刪，要重造得重寫排版腳本，
來源影像則都在 `images/`。
