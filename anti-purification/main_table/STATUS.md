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

### 七、VMAF 進指標聯集
`code/metrics_union.py --stage vmaf`，對 fidelity／displacement／retention 三種
既有配對(6,240 格)各補一欄，`results/metrics_vmaf_union.csv` 的 `pairing` 欄
分三種。餵法：單幀直接餵給 `libvmaf`——在 5 對影像上驗證過，跟「複製成 5 幀
的靜止序列」逐位元同分(`integer_motion` 兩種餵法每一幀都是 0)，故不用另外
拼影片。同一輪順手修掉一個既有 bug：`displacement.csv` 記的影像路徑是搬動
前的相對路徑(開頭 `runs/`)，`stage_displacement`／`stage_vmaf` 現在都用
`resolve_png()` 轉。

### 八、跨編輯器遷移：FLUX 全表完成，SDEdit(SD 1.x／2.x)收線
**FLUX.1-Kontext-dev**：`code/edit_flux_preview.py --arm <undefended|條件>`，
13 個 arm(分母 + 12 條件)× 32 格 = 416 格全部跑完，4-bit NF4 量化
(transformer + T5)單卡 3090 可跑，載入後 VRAM 11.4GB、峰值 14.1GB。
`FluxKontextPipeline` 會把 512×512 的請求蓋成 1024×1024(`_auto_resize`)，
故這一支跟 ip2p/inpaint 不是同一個解析度，數字不可直接比。位移(384 格，
`code/edit_displacement_flux.py`)已補進 `report/main/` 的 FLUX 一節。

**SDEdit(SD 1.x／2.x)：已收線，不再繼續**。過程：
1. SD 2.x 一度整批身分讀數壞掉(`identity_row` 連臉都偵測不到)——根因是
   `sd2-community/stable-diffusion-2-1` 復刻的是 768-v(v-prediction)
   checkpoint，但 `SDWrapper` 的 DDIM 遞迴全部假設 ε-prediction。換成
   `sd2-community/stable-diffusion-2-1-base`(epsilon-prediction、原生 512)
   後恢復正常，跟 SD 1.5 同一個量級。
2. 恢復正常後，strength(0.2–0.6)、guidance_scale(3.0–10.0，獨立掃過)、
   checkpoint 三個維度都掃過：身分保留與指令服從**沒有交集**——能保住身分
   的值指令視覺上沒有被執行(墨鏡／警察制服都沒出現)，指令被執行的值身分
   已經崩了。查過主表引用的兩篇人臉專門論文(FaceLock、DiffusionGuard)，
   兩篇的受害模型分別是 IP2P 與 SD inpainting，**都不是純 SDEdit**——這是
   兩篇論文存在的理由之一，沒有「不加遮罩的 SDEdit 打人像、效果正常」這種
   文獻可以借。
3. 使用者裁定：不加臉部遮罩、不改指令措辭(要跟 ip2p 場景逐字一致)。三個
   參數維度都撐不住之後，SDEdit 這條線收掉，不再嘗試。

資料與圖見 `report/editor_check/`、`report/sd_samples/`，過程記在
commit 記錄(`git log --oneline -- main_table/code/edit_sdedit_preview.py`)。

### 九、SDXL-InstructPix2Pix（SD 系列的指令式編輯器）
`code/edit_sd_family_preview.py --editor sdxl-ip2p`，權重
`diffusers/sdxl-instructpix2pix-768`（SDXL 骨幹，以 InstructPix2Pix 的資料重新
訓練，原生 768×768），30 步、種子 20260812，指令逐字取 `edits.ip2p`、不加遮罩，
與 ip2p 場景相同。只跑未防禦原圖。

- 2 張影像（每類第一張）× 4 指令 × guidance {3, 5, 7.5} × image guidance
  {1.2, 1.5, 2.0}：`results/sd_family_sdxl_ip2p_grid.csv`。
- 8 張影像 × 4 指令 × guidance {5, 7.5} × image guidance {1.5, 1.8, 2.0}：
  `results/sd_family_sdxl_ip2p_all8.csv`。1.8 是主表 ip2p 的 image guidance。

`g7.5_ig1.8` 的 id_orig 中位數（8 張，FaceNet）：墨鏡 0.67、警察制服 0.91、
安全帽 0.91、領結 0.91；≥ 0.55 的格數依序 4、8、7、8／8。墨鏡一欄偏低的格子，
對照圖上眼睛被鏡片遮住，man_01、man_02 兩格臉型也被改動。同一組的對照圖目視：
墨鏡 8/8、領結 8/8 畫出配件；警察制服約 5/8、安全帽約 5/8（未畫出的多為只改衣服
顏色，或畫成耳機），man_02 的安全帽那格整張臉被重繪（id −0.089）。所有格子都有
共同的全域改動：背景轉灰、衣服色調偏移、主體周圍有一圈亮暈。

對照圖在遠端 `main_table/images/sd_family/<批次>/`（`sheet*.jpg`），不入版控。
這一支與 ip2p（512×512、SD 1.5 骨幹）解析度與骨幹都不同，數字不可與主表直接比。

**指令以外的改動**（`code/sd_family_offtarget_readout.py`：背景區與主體區對原圖的
平均 ΔE00、整張 LPIPS，背景區 = `data/portraits/masks/`）。主表 ip2p 未防禦編輯
（g7.5、ig1.8、512×512、50 步）背景 2.90、主體 8.39、LPIPS 0.196；SDXL-IP2P
g7.5／ig1.8 背景 25.8、主體 26.3、LPIPS 0.507；image guidance 推到 4.0（g3）背景
仍 21.1、主體降到 12.8。空指令、g3／ig4 的輸出同樣把白背景換成灰色或紋理、加
亮暈；VAE 編碼再解碼的背景 ΔE00 為 0.5–0.8。背景改動因此來自 checkpoint 本身
的輸出分佈，不隨 guidance 消失。讀數：`results/sd_family_offtarget_*.csv`，
參照列 `results/sd_family_offtarget_ip2p_si18_reference.csv`。

### 十、UltraEdit（SD3）：參數與句型的選定，及全表
`code/edit_sd_family_preview.py --editor sd3-ultraedit`，權重
`BleachNick/SD3_UltraEdit_freeform`（UltraEdit，arXiv:2407.05282，SD3-medium 骨幹、
不需遮罩的版本，512×512）。官方 diffusers 沒有它的管線，`code/ultraedit_sd3_pipeline.py`
移植自作者的 fork，只改 import。作者 README 的範例值是 g7.5、ig1.5、50 步、
`negative_prompt=""`，此外沒有指令寫法的說明；論文列的指令是祈使句
（Add／Change／Replace／Remove／Transform 等九類）。

選定過程（2–4 張影像，讀數見 `results/sd_family_*ultraedit*.csv` 與對應的
`sd_family_offtarget_*.csv`）：
1. g7.5：領結乾淨；墨鏡那格 man_00 整張臉被換（id 約 0.01），woman_00 嘴唇與頭頂
   的花被改。
2. g 2.5／4／5.5 × ig 1.5／2.5：g 越低指令以外的改動越少；ig 2.5 反而提高對比。
3. g2.5／ig1.5 下六種句型（原文 `Let the person wear …`、`Add … to the person`、
   `What if the person wears …?`、`Make …`、`Put … on …`、原文加
   `, keep the face unchanged`，`code/prompt_sets_ultraedit.json`）：背景 ΔE00
   2.21–2.34、LPIPS 0.169–0.180，差距小於影像間的差距；四條指令的執行情況相同。
   `police suit` 在六種句型下都沒有被執行。
4. 換名詞（`police uniform`、`bow tie`）與指明部位的句型，g2.5 與 g4
   （`code/prompt_sets_ultraedit_round2.json`）：g4 下制服只在兩張女性影像局部
   出現，墨鏡那格的換臉加重（man_00 id 0.09–0.13），背景與主體改動增加。

全表採 `add` 句型（`Add sunglasses／a police suit／a helmet／a bowtie to the person`）、
g2.5、ig1.5、50 步：g2.5 下各句型之中它的背景 ΔE00、主體 ΔE00、LPIPS 三項都最低
（2.21／4.80／0.169，4 張影像）。這一組的背景改動低於主表 ip2p 的 2.90。

全表：`code/edit_ultraedit_full.py`，分母 + 12 條件 × （未淨化 + 7 道淨化算子）
× 8 影像 × 4 指令 ＝ 3,328 格，單卡每格 7.4 秒（含身分讀數）。輸入沿用主表已產好
的防禦圖與淨化後輸入（`runs/edit_purified/.../<圖>__orig.png`，即主表 ip2p 實際送進
編輯器的那張），不重跑防禦與淨化。位移與保留率由 `edit_displacement.py`、
`edit_retention.py` 原樣算出（逐條件算，最後合併）：`results/ultraedit_full/<arm>.csv`
（逐格編輯與身分讀數）、`results/displacement_ultraedit.csv`、
`results/retention_ultraedit.csv`。報告頁 `report/main/` 有獨立一節。

## 報告頁

五份報告的清單與各自協定見 `README.md`「報告頁」一節，這裡只記已發布的
artifact 連結(全部私人，要給別人看從頁面 Share 選單開分享)：

| 報告 | 連結 |
|---|---|
| `report/main/` 主表 | <https://claude.ai/artifact/Qdde7nZt2uqewfTtrTTuLg> |
| `report/aligned_matrix.html` 等失真臂 | <https://claude.ai/artifact/NcyZMLGorwD5Pgrc3sYCqc> |
| `report/flux_full/` FLUX 全表樣張 | <https://claude.ai/artifact/4qhUdyt7wXV2eLLs115wFr> |
| `report/editor_check/` 編輯器樣張 | <https://claude.ai/artifact/1ryd73pLkmaiKgtaxYNcSz> |
| `report/sd_samples/` SD 樣張 | <https://claude.ai/artifact/XBaQ9pzrXhhrCsVDHjs1QQ> |

單一版本上限 64 MB、整個 artifact 上限 256 個檔；更新時只傳改動的檔，
沒傳的會保留。**影像不入版控**，各報告的縮圖怎麼產生見對應腳本的 docstring
或 `README.md`。

**`report/aligned_matrix.html` 的 `data.js` 本機原本沒有**(產生它的
`cf62540` commit本身就沒帶，只有已發佈的 artifact 有)，已在這一輪補上
(`code/report_matrix_data.py --out report/data.js`)。**`img/` 仍然沒有本機
副本**——重建需要連遠端抓 `runs/defence_portraits`／`runs/eps_aligned`
等來源影像並轉 WebP，步驟見該腳本 docstring 的「重建整個報告頁」，這一輪
沒有做。也就是說這份報告目前**本機只看得到表格，看不到圖**，跟其餘四份
不同；要看圖仍要開已發布的 artifact 連結。

等失真臂另有一份更早的版本 <https://claude.ai/artifact/3XXgCxBMtrjH62VEkXuEBH>，
版面與資料都被上面那一份取代，兩者的影像檔名不同，未刪除。

## 接下來

原先列的三件事(VMAF、跨編輯器遷移)都已經做到能做的邊界，沒有指定中的
待辦。以下是還沒做、但沒人要求要做的可能方向，僅供參考，不代表應該做：

- 等失真臂、穿透拆解目前都只涵蓋 ip2p/inpaint，沒有 FLUX 對應版本。
- 資料集擴充的成本已經量過：每多一張影像，主表本體(12 條件的防禦＋編輯＋
  淨化重編)約 6.4 GPU 小時(4.08 小時是 `defence_<條件>.csv` 的實測值，
  2.31 小時是編輯生成的估算)，FLUX 全表再加約 1.2 GPU 小時(52 格 ×
  83.76 秒，實測)。換算方式見助理記憶 `main-table-per-image-gpu-cost`。

## 規矩

- **GPU 一律送遠端**，上限兩個 session 合計五張，派工前看 `~/lab_leases/`
  （格式 `<主機> <pid> <名稱>`，兩台共用）並自行複驗卡上沒有別人的 compute app。
- **不設判準**：數據與圖擺出來為止，不下「成立／不成立」「值得／不值得」的結論。
  這一條也適用於指標本身。
- **引用一個數字就要連它的協定一起引用**（`docs/reference/BASELINE_PROVENANCE.md`
  規則 1）。等失真臂的每一列都標了 `modified_from_paper`，論文的保證不適用。
- 命名不含日期、流水號或順序詞；文件內容不得有時間相依性。commit message 用英文。
