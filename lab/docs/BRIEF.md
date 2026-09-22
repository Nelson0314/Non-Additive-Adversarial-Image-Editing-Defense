# 實驗室任務書：以現成生成編輯模型作為免疫載體

本檔是 `lab/` 這條線的任務界定。所有工作只在 `lab/` 內完成，需要的檔案從
`anti-purification/` 複製，不寫回主線目錄。

## 一、現行方法與它的位置

主線已完成的比較表（`anti-purification/main_table/`）把十二個免疫方法放在
同一條編輯管線上。本專案自己的那一列是 `colour_curve_ours`：AdvCF 形式的
單調分段線性 RGB tone curve（`pieces 64`、`radius 5.0`、900 步、
`deltae_cap 16.0`），對整張影像求解一條全域色調映射。

它的兩個讀數位置如下。

| 讀數 | `colour_curve_ours` | 十一個外部條件的範圍 |
|---|---|---|
| 位移（ip2p 全圖／主體內） | 0.3821／0.3867 | 0.0723–0.6411 |
| 位移（inpaint 全圖／主體內） | 0.4431／0.3317 | 0.3380–0.6406 |
| 非幾何淨化保留率均值 | **1.021** | 0.306–0.861 |
| blocked（ip2p、inpaint） | 3/32、4/32 | 0/32–32/32、1/32–21/32 |

即：**抗淨化是十二個條件裡唯一大於 1 的（七道淨化之後防禦不減反增），但
未淨化的位移落在後段。** 這條線要解的就是這個落差。

失真束縛是 ΔE00 ≤ 16（整圖），PSNR 中位數 16.67、LPIPS 0.3367。束縛種類與
其他條件不同，排名不可直接比大小。

### 為什麼位移上不去（已經量到的）

1. 載體的容量本身受限。全域單調 tone curve 的自由度是 64 段斜率 × 3 通道，
   空間上是常數映射，影像內容不能進到參數裡。
2. 顏色線的振幅唯一來源是色度，而人像的色度主要落在膚色上；把振幅推高等於
   把膚色移走，均勻度與防禦強度直接對立，轉折點已量在位移場 TV 1.0–2.0 之間。
3. 三道自然度門檻（NIQE、平均色差、CVaR 尾端）都被最佳化鑽過。結論是
   自然度要進參數化，不是進正則項。
4. 代理讀數推得動不代表編輯端會動：四個代理各推 0.003–3.8 倍，編輯端全部
   沒垮。**任何新設計的判定一律以編輯端讀數與逐格影像為準。**

完整紀錄在 `../COLOUR_LINE.md`。

## 二、新方向

把免疫載體從「解一條低維色調曲線」換成「**用現成的生成編輯模型（Stable
Diffusion 系）對原圖做一次風格／色彩濾鏡轉換**」，轉換後的影像即為防禦圖。

動機有三點，都對應上一節的限制：

- **容量。** 生成式重繪可以同時改動色彩、局部對比、紋理與粒子，自由度遠高於
  全域曲線，而且改動量與影像內容耦合。
- **自然度進參數化。** 產物是生成模型的樣本，落在自然影像流形上，不需要事後
  加門檻去壓；這正是第 1.3 點的結論所指的方向。
- **抗淨化的機制不變。** 風格轉換不是可被移除的加性擾動：JPEG、模糊、
  DiffPure 這類算子的設計目標是還原「乾淨影像」，而防禦圖本身就是一張乾淨
  影像，沒有可還原的參考。`colour_curve_ours` 保留率 1.021 的機制在此保留。

### 參考文獻（起點）

使用者指定的一篇：

- Yibo Wang, Yong Zhou, Bing Liu, Rui Yao, **"Style-controllable adversarial
  example generation via image editing and prompt embedding optimization"**,
  *Neurocomputing*, 2026. DOI `10.1016/j.neucom.2026.134591`
  （PII `S0925231226019892`，cover date 2026-11-14）。
  **全文未取得**：ScienceDirect 對非機構 IP 回 403，使用者提供的簽章 URL 已
  逾期（`X-Amz-Expires=300`），Elsevier API 無金鑰只回 coredata，
  OpenAlex／Semantic Scholar 皆無摘要，非開放取用。已取得的 metadata 存在
  `docs/paper/neucom_134591.json`。標題本身給出兩個元件：**image editing** 作為
  載體、**prompt embedding optimization** 作為求解變數。

其餘要查證的文獻由本任務的第一步補齊，見第四節。

## 三、威脅模型與評測協定（沿用主表，不得更改）

- **防禦方看不到攻擊指令。** 求解端不得讀 `prompts.yaml` 的 `edits.*`。
  防禦方可用的文字條件只有該影像類別的 `content`（`man` / `woman`）。
- **資料集**：`lab/data/portraits/`，8 張人像（`man_00`–`man_03`、
  `woman_00`–`woman_03`），512×512 RGB PNG，附主體遮罩。
- **編輯管線**：`lab/code/edit_preflight.py`，兩個場景共用種子 20260812、
  50 步、512²。

| | ip2p | inpainting |
|---|---|---|
| 受害模型 | `timbrooks/instruct-pix2pix` | `runwayml/stable-diffusion-inpainting` |
| guidance | `s_t` 7.5、`s_i` **1.8** | 7.5 |
| 遮罩 | 無 | dilate 4，白＝重繪 |
| 未防禦對照臂 | `ip2p_si18` | `inpaint_undefended` |

- **指令**：每個場景四條，取自 `data/portraits/prompts.yaml` 的
  `edits.ip2p` 與 `edits.inpaint`。每個條件 8 影像 × 4 指令 × 2 場景 ＝ 64 格。
- **位移**（主讀數）＝ `LPIPS(編輯(原圖), 編輯(防禦圖))`，含主體內／外分區。
- **淨化**：七道算子 `identity`、`crop_resize0.1`、`jpeg30/50/80`、
  `blur1/blur2`、`rotate15`。`crop_resize` 與 `rotate` 是幾何類，讀數不與
  其餘五道混著平均。
- **保留率** ＝ 淨化後位移 ÷ 未淨化位移。

程式已複製到 `lab/code/`（`edit_preflight.py`、`purify_run.py`、
`edit_displacement.py`、`edit_retention.py`、`metrics_union.py`、`paths.py`、
`defence_run.py`、`immunise_as_condition.py`）。`paths.py` 會自己找到主線目錄
取得 `src.*` 套件；必要時用環境變數 `IMMUNISATION_SOURCE_HOME` 指定。

## 四、要交付的東西

### 步驟一：文獻查證（Codex）

針對「以生成模型的重繪／風格轉換本身作為對抗樣本載體」這一族，找出並逐篇
記錄下列資訊到 `lab/docs/LITERATURE.md`（**上限 120 行**）：

每篇一列，欄位為：出處（作者、會議／期刊、年、arXiv 或 DOI）、載體是什麼、
求解變數是什麼、約束是什麼、威脅模型（分類器／編輯模型／個人化）、以及
**與本任務的關係一句話**。

必須涵蓋但不限於：unrestricted / semantic adversarial examples、diffusion 系的
對抗樣本生成（SDEdit 為載體者）、style cloaking（Glaze 一族）、
prompt / text-embedding 空間的最佳化、以及既有的編輯免疫方法中有無用生成模型
產生防禦圖者。**已在主表裡的十二個方法不用重寫**，只需指出哪一個與本設計最近。

同時回答一個具體問題：**已發表工作中，有沒有人把「生成式風格轉換」本身
（而非加性擾動）當成 image editing immunisation 的載體？** 有的話列出；沒有的
話說明最接近的是什麼、差在哪。

### 步驟二：設計審查（Codex）

審查下列草案並提出修正，寫到 `lab/docs/DESIGN_REVIEW.md`（**上限 80 行**）。
重點放在「哪裡會靜默失效」與「哪個環節的成本估錯了」。

**草案：`style_edit` 載體**

```
x_def = SDEdit(x ; e_s, s, r)
```

- 骨幹：`runwayml/stable-diffusion-v1-5` 的 img2img（SDEdit），512²。
- `e_s`：風格文字條件的 embedding（77×768），**求解變數**。
- `s`：denoising strength，控制偏離原圖的幅度。
- `r`：取樣種子，固定。
- 結構保持：低 strength（0.2–0.45）本身就保留構圖；另加保真項。

求解目標（防禦方看不到指令，故不能用攻擊指令）：

```
max_{e_s}   L_def(x_def)  -  λ · L_fid(x_def, x)
```

- `L_def`：對受害模型的代理。候選兩種——(a) PhotoGuard 形式的 VAE encoder
  攻擊 `‖E(x_def) − E(x)‖²`；(b) 空 prompt 下 UNet 的 ε-prediction 誤差。
- `L_fid`：保真約束。候選——LPIPS 上界、主體遮罩內的 ΔE00 上界、或 DISTS。
- 梯度要穿過 SDEdit 的取樣鏈，成本是本設計最大的不確定性。若全鏈不可行，
  退路是只對最後 k 步反傳，或改用 score-distillation 形式。

**要跑的臂**（每一臂都跑完整的防禦 → 編輯 → 淨化 → 淨化後編輯）：

| 臂 | 內容 | 這一臂要回答什麼 |
|---|---|---|
| `style_random` | 固定的一組風格 prompt，**不做最佳化** | 現成模型的風格轉換本身能推動多少 |
| `style_opt` | prompt embedding 最佳化 | 最佳化相對於隨機買到了什麼 |

`style_random` 是必要的對照：本專案已經量到過「最佳化贏隨機但臂間打平」
與「代理推得動換不到效果」兩種情形，缺這一臂時整組讀數無法解讀。

未防禦分母直接複製主表已產好的 `ip2p_si18` 與 `inpaint_undefended`，不重跑。

### 步驟三：實作與執行（本 session）

依審查後的設計實作，送遠端跑完整條管線，產出：

- 防禦圖（每臂 8 張）
- 防禦後編輯（每臂 64 格）
- 淨化圖（每臂 7 × 8）
- 淨化後編輯（每臂 7 × 64）
- `displacement.csv`、`retention.csv`、失真與美學指標

## 五、工作規則

- **不設判準。** 把數據與圖擺出來為止，不代使用者下「成立／不成立」的結論。
  所有量測欄位照報，不挑選。
- **GPU 一律送遠端**（NYCU BASIC lab，`ssh -p 10101` basic-1／`-p 10102`
  basic-2）。**全域上限五張卡**，判定卡空要同時滿足「沒有別人的 compute app」
  與「已用記憶體 < 1 GB」。
- 目錄、檔案、實驗組名稱不含日期、流水號或順序詞。
- commit message 用英文。
- 密碼與 token 不得寫入任何入庫檔案。
