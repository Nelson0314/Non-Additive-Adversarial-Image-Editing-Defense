# 主表的建立進度與接續指引

接手這個目錄先讀這一份，再依需要查 `README.md`（協定與檔案位置）與
`results/aligned/README.md`、`results/PASSTHROUGH.md`（兩組進階量測）。

## 主讀數沒有換

**主表的比較仍以常見的位移讀數為準**：`results/displacement.csv` 的
`disp_lpips_full`（與主體／背景分區），配 `results/retention.csv` 的保留率、
`results/defence_<方法>.csv` 的防禦圖失真，以及四張 `metrics_*_union.csv`。
十二個條件、每條件 8 影像 × 4 指令 × 2 場景 ＝ 64 格，協定見 `README.md`。

底下兩組是**附加的**，回答主讀數回答不了的問題，不取代它。引用時各自標明協定。

## 三組資料的關係

| 資料 | 回答什麼 | 位置 |
|---|---|---|
| 主表 | 各方法在**各自原生預算**下推得動多少 | `results/*.csv` |
| 等失真臂 | 把擾動預算縮到**同一個失真水準**之後，還推得動多少 | `results/aligned/` |
| 穿透拆解 | 位移裡有多少是防禦端的改動**原樣穿過**編輯器 | `results/passthrough.csv`、`results/PASSTHROUGH.md` |

主表的位移排名與失真排名幾乎同序（原生失真 LPIPS 從 0.0426 到 0.6661，差十五倍），
所以一列的位移比另一列高，單看主表讀不出是防禦較有效還是擾動較大——等失真臂是為了
把這兩件事分開。穿透拆解則是把位移本身再拆成兩個來源。

## 已完成

### 一、路徑與可執行性
`code/paths.py` 集中解析路徑：影像與 CSV 走 `main_table/` 內部，`src.*` 套件與
資料集走主線目錄（依序找 `../`、`../../anti-purification`、曾用名
`../../non-additive-frequency`，也吃環境變數 `IMMUNISATION_SOURCE_HOME`，
找不到就中止並列出找過哪些路徑）。七支腳本的 `sys.path` 與路徑字串都走它。

### 二、規格釘樁
`tests/test_conditions.py`（39 項，`pytest tests/`，不需 GPU）釘住十二個條件記在
`results/defence_*.csv` 的求解設定、CSV 規模與鍵的唯一性、協定欄位，以及
`docs/reference/SOURCE_AUDIT.md` §10 的值域對照與 CSV 一致。主線的
`tests/test_baselines.py` 只覆蓋共用 PGD 骨幹的六個 spec，與主表的交集僅四個條件。

### 三、等失真臂
`code/defence_run.py --eps-scale <倍率>` 把各篇的原生 eps 乘上倍率，**`step_size`
同倍率縮放**以維持 `step_size / eps`；倍率不等於 1 時強制標 `modified_from_paper`，
原生值留在 `eps_native` 欄。`diffvax` 是前饋式、沒有這個旋鈕，給了會直接停住。

十個條件縮到防禦圖失真 LPIPS 對上 `colour_curve_ours` 的 0.3344，落點在 −7.8% 到
+2.4%；倍率跨十三倍（0.073–0.969）。編輯端重跑 640 格位移與 4,480 格淨化後編輯，
**分母沒有動**（未防禦側仍是 `ip2p_si18`／`inpaint_undefended`）。
結果、倍率怎麼定出來的、以及兩個沒有對齊點的條件，見 `results/aligned/README.md`。

### 四、穿透拆解
`code/passthrough_readout.py` 對 ip2p 主種子的 704 格算 `D`／`P`／`D_T` 三分區與
`siglip_pair_T`、`blocked_T`。`T̂` 是加性族，對十一個加性條件就是該方法自己的擾動。
另量了 `δ = x_def − x` 與 `Δ = edit(x_def) − edit(x)` 的相關、殘差比與放大倍數。
見 `results/PASSTHROUGH.md`。

### 五、幾何淨化的遮罩
`code/edit_retention.py` 的 `purified_mask()` 把主體遮罩送過與影像同一個
`Purifier`。在它進版（commit `199b7de`）之前跑出來的 `retention.csv`，
`crop_resize0.1` 與 `rotate15` 那 1,536 列的分區兩欄是用未變換的遮罩算的；
判準與重算方式見 `../docs/EVALUATION.md`。

### 六、出處文件
`docs/reference/AUDIT_DCT_SHIELD.md`（新增，主表位移排名前段的兩列先前沒有審計
文件）、`BASELINE_PROVENANCE.md` 的 `eps_pixel01` 逐列單位與「已實作但不在主表」
一節、`EVALUATION.md` 的 `rotate15` 角度三來源不一致。
正本都在主線目錄的 `../docs/`，本目錄的 `docs/README.md` 是指標。

## 報告頁

<https://claude.ai/artifact/NcyZMLGorwD5Pgrc3sYCqc>（私人；要給別人看要從頁面的
Share 選單開分享）。內容是編輯輸出矩陣（4 照片 × 4 指令 × 12 欄）、防禦圖矩陣
（4 照片 × 11 方法，每張標 LPIPS）、失真與位移指標表、保留率表。
格式是使用者指定的：不放說明文字，只有圖與數字；每欄最佳值粗體、欄名以 ↑／↓ 標
方向；與十個對齊 baseline 中位數差兩倍以上的值標紅並附倍率。

頁面原始檔是 `report/aligned_matrix.html`（自足，CSS 與 JS 內嵌，只外部引用
`data.js` 與 `img/`）。**影像不入版控**，重建方式與檔名式樣寫在
`code/report_matrix_data.py` 的 docstring；那一支也負責產生 `data.js`，
跑過一次確認它逐欄重現已發佈的版本。單一版本上限 64 MB、整個 artifact
上限 256 個檔；更新時只傳改動的檔，沒傳的會保留。

另有一份較早的版本 <https://claude.ai/artifact/3XXgCxBMtrjH62VEkXuEBH>，
版面與資料都被上面那一份取代，兩者的影像檔名不同，未刪除。

## 接下來

使用者列的三件事，順序與理由在 `../../HANDOFF.md` 之外另記於助理記憶
（`main-table-next-three-jobs`）：

1. **VMAF 進指標聯集。** 不重跑任何求解或編輯，吃已落地的 PNG，接進
   `code/metrics_union.py` 當第五個 stage。成本在 `libvmaf`／ffmpeg 這個新外部相依，
   以及決定單張圖怎麼餵（VMAF 原生吃序列與時序特徵，複製成靜止序列與只取
   frame-level 兩種選法的數字不一樣，協定要寫明）。
2. **跨編輯器遷移**（SDEdit on SD 1.x／2.x，之後 FLUX）。`src/models/sd.py` 的
   `SDWrapper` 已有需要的元件，缺一個 img2img 取樣迴圈與一條新的未防禦對照臂；
   strength 是新的自由度，要先定再跑全表。FLUX 的門檻是硬體（權重載入超過 24 GB）。

## 規矩

- **GPU 一律送遠端**，上限兩個 session 合計五張，派工前看 `~/lab_leases/`
  （格式 `<主機> <pid> <名稱>`，兩台共用）並自行複驗卡上沒有別人的 compute app。
- **不設判準**：數據與圖擺出來為止，不下「成立／不成立」「值得／不值得」的結論。
  這一條也適用於指標本身。
- **引用一個數字就要連它的協定一起引用**（`docs/reference/BASELINE_PROVENANCE.md`
  規則 1）。等失真臂的每一列都標了 `modified_from_paper`，論文的保證不適用。
- 命名不含日期、流水號或順序詞；文件內容不得有時間相依性。commit message 用英文。
