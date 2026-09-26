# lab/ 的設計、協定與已量到的結論

本檔記載協定、各臂的設計理由與量測結論。現況、執行方式與未完成事項見 `../HANDOFF.md`。

---

## 1. 問題與威脅模型

防禦方在發布人像前處理影像；攻擊方以 InstructPix2Pix（`timbrooks/instruct-pix2pix`）依文字
指令編輯，並可先做淨化（JPEG、模糊、裁切、旋轉）。防禦目標：攻擊方的編輯結果偏離
「未防禦時的編輯結果」，同時防禦圖外觀自然、撐得過淨化。

- **防禦方看不到攻擊指令。** 求解端不得讀 `data/portraits/prompts.yaml` 的 `edits.*`；
  可用的文字只有類別名（`man`／`woman`）。
- **資料**：`data/portraits/`，8 張人像（`man_00..03`、`woman_00..03`），512² RGB，附主體遮罩。
- **攻擊端只跑 ip2p**：`s_t` 7.5、`s_i` 1.8、50 步、主種子 20260812，四句指令（加墨鏡、
  警察制服、安全帽、領結），每臂 8 × 4 ＝ 32 格。inpaint 場景的攻擊遮罩取自原圖、
  攻擊者拿不到，且幾何淨化後不跟著變換；使用者裁定不再跑，既有 inpaint 讀數只當參考。
- **分母不重跑**：未防禦編輯 `runs/edit_preflight/ip2p_si18` 與淨化後的
  `runs/edit_purified/undefended` 是主表既有檔（`lab/runs/` 下以符號連結指過去）。
- **淨化**：`jpeg30/50/80`、`blur1/2`（非幾何）與 `crop_resize0.1`、`rotate15`（幾何），兩類分開報。

## 2. 讀數

| 代號 | 定義 | 檔案 |
|---|---|---|
| `D` | `LPIPS(edit(x), edit(x_def))`，全圖／主體／背景 | `results/displacement.csv` |
| 保留率 | 淨化後 `D` ÷ 未淨化 `D` | `results/retention.csv` |
| 失真 | 防禦圖對原圖的 LPIPS、ΔE00、PSNR、L∞、rms | `results/fidelity.csv` |
| `P` | `LPIPS(edit(x), T̂(edit(x)))`：編輯器完全等變時的位移預測 | `results/passthrough/passthrough*.csv` |
| `D_T` | `LPIPS(T̂(edit(x)), edit(x_def))`：扣除色調穿透後的位移 | 同上 |
| `D_seed` | 同一未防禦輸入、不同評估種子兩次編輯之間的 LPIPS | `results/passthrough/seed_spread.csv` |
| `vqa` | Qwen2.5-VL-7B 判定指令物件是否出現（原圖負例 32/32 答 no） | `results/passthrough/vqa_*.csv` |
| `id_edit` | 編輯輸出與原圖的 ArcFace 餘弦 | 同上 |

`T̂` 是由 `(x, x_def)` 回推的全域映射（該臂所屬參數族，Lab 均方差，回推誤差 ΔE00 0.02–0.07），
只對全域映射類的臂有定義。評估種子 20260813–16 的編輯在 `runs/edit_seeds/<臂>/seed<S>/`，
不寫入主種子分母的目錄。

**不設判準。** 數據與圖並列，不下「成立／不成立」「有效／無效」的判定。

## 3. 外觀的硬條件（使用者看圖後定）

- 不使用由臉框長出的空間權重場或任何空間相依的色彩場：產物出現邊界、色塊與臉框圈
  （`style_filter*`、`curve_dual_spatial*` 已刪除；`style_affine`、`style_opt` 被否決）。
- 直接交付 SDEdit 輸出會換人（strength 0.35 時 FaceNet 0.26）；換背景不是目標方向。
- 偏好 `ab_warp` 的全域映射外觀。粉紅、紅、洋紅、黃、暖黃的色調難看；青、藍、綠可接受。
  文獻：Winkler et al. 2015（*Current Biology*，偏藍變化較易被歸給光源）、Pearce et al. 2014
  （*PLoS One*，光源辨別門檻 ΔE*uv 藍 25.7、黃 18.0、紅 17.9、綠 10.7）、Zeng & Luo（偏好膚色
  中心 a*b* ≈ (21, 24)，色相容忍窄）。
- 同色（膚色）ΔE00 放寬到 12–16 可接受（見報告頁 `report_budget/` 的無最佳化預覽）。

## 4. 現行的臂

全部是全域映射（輸出只依賴該像素的顏色），約束走增廣 Lagrange（`optimise_carrier`），
目標為 `FreeObjective`（`id 1.0`／`enc 0.5`／`cond 1.0`，無文字 ip2p 代理），900 步。
參數只定義在 `scripts/defence_cmd.sh`。

| 臂 | 載體 | 上限 |
|---|---|---|
| `curve_dual_chroma` | 單一全域 RGB 曲線 | 整圖 16、臉框 8 |
| `ab_warp` | CIELAB `(a,b)` 7×7 RBF 位移 ＋ 單調亮度曲線 | 整圖 16、臉框／同色 8、彩度 p95 1.15× |
| `ab_warp_s12`／`_s16` | 同上，位移半徑 80 | 整圖 32、臉框／同色 12／16、彩度 2.0× |
| `ab_warp_ch` | 同 `_s16` | 另加分通道 Lab 位移 p95：a*＋ ≤ 4、a*－ ≤ 15、b*＋ ≤ 4、b*－ ≤ 25、\|ΔL*\| ≤ 15 |
| `ab_warp_ch_free` | 同 `ab_warp_ch`，非恆等起點 | 另加逐張輸入 LPIPS ≤ `ab_warp_ch` ＋ 0.0025 |
| `ab_warp_ch_comm` | 同上，目標改為 `comm`（§6） | 同上 |
| `ab_prism` | 色度相依亮度曲線 ＋ 亮度相依色度相似變換（112 參數） | 同 `ab_warp`，輸入 LPIPS ≤ `ab_warp` |
| `style_warp` | SDEdit 風格圖擬合成 `ab_warp` 族映射（不對編輯器最佳化） | 同 `ab_warp` |
| `inpaint_bg`／`inpaint_outside_face` | 重繪主體外／臉外，受保護區逐位元保留 | 無預算（只當參考） |

## 5. 已量到的結論（ip2p，主種子，32 格；另一批的數字標明）

**5.1 位移跟著輸入失真走。** 十個全域顏色臂 73 個「臂 × 影像」點：
`D ≈ 0.084 + 0.932 × 輸入 LPIPS`，r = 0.944。換載體（`ab_prism`）、換預算、加上限都沿著同一條線移動。

**5.2 位移的大部分是色調穿透。** ip2p 近乎原樣保留全域色調，指令物件照樣畫出。

| 臂 | 輸入 LPIPS | D | P | D_T [95% 區間] |
|---|---|---|---|---|
| `colour_curve_ours`（主表那一批） | 0.337 | 0.386 | 0.304 | 0.223 [0.177, 0.276] |
| `ab_warp` | 0.301 | 0.375 | 0.291 | 0.195 [0.144, 0.260] |
| `ab_warp_ch` | 0.299 | 0.358 | 0.277 | 0.165 [0.142, 0.189] |
| `ab_warp_s12` | 0.466 | 0.514 | 0.457 | 0.248 [0.200, 0.311] |
| `ab_warp_s16` | 0.503 | 0.539 | 0.495 | 0.245 [0.195, 0.311] |
| `ab_prism` | 0.239 | 0.295 | 0.222 | 0.132 [0.113, 0.154] |
| `style_warp` | 0.062 | 0.117 | 0.063 | 0.107 [0.089, 0.129] |

區間為影像群集 bootstrap（種子 1729、10,000 次）。五個評估種子的平均與主種子一致
（`ab_warp` D_T 0.186–0.211，`ab_warp_s16` 0.235–0.264）。

**5.3 底線。**
- 取樣本身的離散 `D_seed`：全圖 0.109、主體 0.155、背景 0.058（未防禦，320 對）。
- 非防禦性改動（淨化後的未防禦編輯）：`jpeg30` 輸入 LPIPS 0.301 → D 0.320；`blur2` 0.273 → 0.289；
  `jpeg50` 0.224 → 0.244。`ab_warp`（0.301 → 0.375）比同失真的 JPEG 高 0.055。

**5.4 編輯照樣完成。** VQA 物件出現格數（5 種子 × 32 格 ＝ 160）：未防禦 157、`ab_warp` 157、
`ab_warp_ch` 155、`ab_warp_s16` 157、`ab_prism` 151。`id_edit` 由未防禦 0.833 降到 0.68–0.78
（未扣穿透；色調本身也會影響 ArcFace）。

**5.5 以 `D_T` 重做的配對差**（全圖；括號內為主體）：
- `ab_warp_s16` − `ab_warp`：+0.050 [+0.020, +0.079]（+0.086）；`_s12` 相近。
- `ab_warp_s16` − `colour_curve_ours`：+0.023 [−0.037, +0.086]（+0.057 [−0.014, +0.117]）。
- `ab_warp_ch` − `ab_warp`：−0.030 [−0.075, +0.010]（−0.038 [−0.082, −0.001]）。
- `ab_prism` − `ab_warp`：−0.063 [−0.108, −0.022]。

**5.6 最佳化解落在隨機抽樣碰不到的區域。** 同族隨機候選（每張 4,096 個）在全部上限內、
LPIPS 對齊最佳化解 ±0.0025 的，`ab_prism` 8 張中 2 張為 0、`ab_warp_ch_free` 8 張合計只有 2 個候選。
`ab_prism` 在對齊得上的 6 張輸給同族隨機（D_T −0.029 [−0.044, −0.016]）。使用者裁定不再做隨機對照。

**5.7 等失真的對抗擾動（主表那一批，`../anti-purification/main_table/results/aligned/`）。**
十個 baseline 的 eps 縮到 LPIPS ≈ 0.31–0.34 後，ip2p 的 D：`dia_r` 0.390、`dayn` 0.386、
`dia_pt` 0.386、`sifm` 0.380、`danp` 0.370、`dct_shield` 0.322、`photoguard_c` 0.322、
`dct_shield_y` 0.290、`mist` 0.264、`photoguard_linf` 0.242；`colour_curve_ours` 0.386。
這些 D 尚未拆穿透（PSNR 38–47 dB，改動為高頻擾動而非全域色調）。保留率（七道平均）
0.42–0.63，`colour_curve_ours` 1.018。

**5.8 其他已量到的事。**
- LPIPS 雙邊帶在 `ab_prism` 族不可行（色彩上限內可達 LPIPS 7/8 張低於 `ab_warp`），改為上限。
- blocked（SigLIP < 0.837）隨輸入失真上升（輸入 LPIPS > 0.4 時 28/60），不作為「編輯被擋下」。
- 錨住一個失真指標不會錨住其餘指標；等失真比較要指名錨點並並報其餘指標。
- 淨化洗不掉顏色臂（非幾何保留率 1.00–1.14）。

## 6. 等變殘差目標 `comm`（`code/comm_objective.py`）

```text
comm(θ; ξ) = LPIPS( N_ξ(T_θ(x)), T_θ(N_ξ(x)) )     N：無文字 ip2p 短鏈（FreeObjective.null_edit）
score(θ)   = − w_comm · comm / c0
```

若編輯器對 `T` 等變，`comm` = 0；它對準 `D_T`。`c0` 為起點在固定驗證抽樣上的值；`w_comm` 使起點
梯度範數等於 `FreeObjective`；恆等起點梯度為 0，故 `_comm`／`_free` 共用非恆等起點
（`w_raw ~ N(0, 0.1²)` 投影、`th_raw ~ U(−0.1, 0.1)`，種子 0）並斷言梯度非零。
試跑（`man_02`，300 步）：`comm` 驗證值由 0.004 升到 0.210（`_free` 臂 0.172），輸入 LPIPS 0.261／0.244，
每步約 3.5 秒；此試跑不作為繼續與否的條件。顯存約 18 GB，與別人的行程疊在同一張卡時會 OOM。

## 7. 已刪除或退役的方向（程式已移除，理由摘要）

| 方向 | 理由 |
|---|---|
| 直接交付 SDEdit（`style_random`／`style_low`） | 換人 |
| 生成低頻色彩場／局部仿射（`style_filter*`、`style_affine`、`style_opt`） | 色塊、臉框圈（使用者否決） |
| 臉框分區求解（`curve_dual_spatial*`） | 邊界可見 |
| 同族隨機對照 | 候選幾乎不可行，使用者裁定不做 |
| inpaint 攻擊場景 | 攻擊遮罩取自原圖，威脅模型不成立 |

`style_warp` 用到的 SDEdit 風格圖在遠端 `lab/runs/defence/style_affine/*__sdedit_raw.png`，資料保留。
