# Pilot：主流加性防禦的目標函數搬到 AdvCF 顏色載體上

`configs/advcf_pilot.json`，`scripts/paper_baseline.py --objective free`。
評估 `configs/evaluate_pilot.json`。

**規模：兩張人像 × 兩類指令 × 一顆種子 = 4 格，每臂 32 列。**
這是 pilot，只能看方向，不能下結論。

## 為什麼要跑

`scripts/paper_baseline.py` 先前**完全沒有接 `MainstreamTerms`**——PhotoGuard 的
`enc_target` 與 AdvDM／Mist 的 `diffusion` 只在位移場那條線
HANDOFF 裡「主流目標函數落在重跑雜訊內」
那句話的成立範圍只到位移場，**在 AdvCF 上一次都沒試過**。

## 資料集的兩個選擇

**不用 `garment_colour`。** 它問「衣服主要是什麼顏色」，而本專案的防禦本身就是
整體調色：防禦圖已經把衣服推向目標顏色，VQA 不管編輯有沒有做成都答對。實測
`anchor_de16_free` 在該類是防禦後 26/26、同格未防禦只有 17/26——防禦不可能讓
指令更容易完成。加物件的 `add_hat`／`add_scarf` 不受影響。

**不用 `task_obj_remove_410264`。** 它的 `add_scarf` 整格九顆種子全部換人、
`add_hat` 九顆剔掉七顆，見 `data/undefended_audit.json`。

## 未防禦預檢

`configs/evaluate_pilot_preflight.json` 把 `arms` 設成只有 `undefended`，
先確認攻擊本身成立再派正式批次。四格逐張看過：帽子與圍巾都長出來、眼鏡保留、
四張都還是同一個人。**攻擊在這四格全部有效。**

## 結果（identity 那一道）

| 臂 | LPIPS↑ | DISTS↑ | CLIP↓ | ΔE00 | PSNR | 發布圖身分 |
|---|---|---|---|---|---|---|
| **`pilot_out_fixed`** | **0.4090** | **0.2299** | 0.9124 | 15.81 | 18.48 | 0.9358 |
| **`pilot_photoguard`** | 0.4088 | 0.2141 | **0.8906** | 15.36 | 20.30 | 0.9141 |
| `pilot_free`（分母） | 0.3942 | 0.2178 | 0.9217 | 15.73 | 19.02 | 0.9524 |
| `pilot_advdm` | 0.3736 | 0.2178 | 0.9277 | 15.74 | 17.84 | 0.8996 |
| `pilot_mainstream_both` | 0.3699 | 0.1948 | 0.8978 | 13.89 | 20.28 | 0.8898 |
| **`pilot_mist_target`** | **0.3394** | **0.1862** | 0.9295 | 15.46 | 18.31 | 0.9264 |

四道淨化上的 LPIPS（前三名）：

| 臂 | identity | jpeg | blur | crop |
|---|---|---|---|---|
| `pilot_free` | 0.3942 | 0.4082 | 0.3811 | 0.4335 |
| `pilot_photoguard` | 0.4088 | 0.4006 | 0.4019 | 0.4602 |
| `pilot_out_fixed` | 0.4090 | 0.4164 | 0.3945 | 0.4485 |

## 四件事

**一、`out` 修好選點之後翻盤了。** `runs/advcf_anchor/` 的四個 `out`／`out_tone`
臂全輸，是因為 `CompositeObjective.eval_terms` 先呼叫 `base.eval_terms()`，而
base 在返回前已還原訓練用的 `draws`，`OutputDisplacement` 因此落在**訓練抽樣**
而不是固定驗證抽樣上——checkpoint 是用帶訓練端隨機性的分數挑的。修好之後
`pilot_out_fixed` 的 LPIPS 與 DISTS 都是最高。**先前寫的「out 沒贏過 free」作廢。**

**二、PhotoGuard 的 targeted encoder attack 在這個載體上有效。** CLIP 0.8906 最低、
LPIPS 0.4088，而且代價更低（ΔE00 15.36、PSNR 20.30 都優於分母）。代價是發布圖
身分 0.9141 對 0.9524。

**三、AdvDM 的無目標 diffusion 沒用，兩個主流項疊加更差。** `advdm` 0.3736 輸給
分母；`mainstream_both` 0.3699 且身分最低 0.8898——兩項疊加互相抵銷。

**四、Mist 的 targeted diffusion 是全場最差，根因是目標選錯。**
`diffusion_target` 用的目標沿用 `grey_target`（中性灰）。`enc_target` 沒被這個
害到，因為 VAE latent 是**靜態編碼**，拉過去就算數；`diffusion_target` 拉的是
**去噪行為**，而中性灰的去噪行為是低變異、接近平庸的——把行為拉向平庸等於要求
模型**更容易**還原這張圖，與「讓模型還原不了」是反方向。

**Mist 原論文的目標不是平坦的灰，是一張高頻紋理圖。** 這一項要救，改的是目標
不是權重。

## 實作上踩到的坑

`diffusion_target` 的目標側 ε **不能只算一次就快取**：`FreeObjective` 的
`resample` 每一次求值都重抽時刻**與噪聲**，舊時刻直接 `KeyError`，而查得到的
也配在錯誤的噪聲上。快取要跟著 `obj.draws` 這個抽樣版本走。

## 求解耗時（實測，每張圖）

| 臂 | 秒／圖 |
|---|---|
| `pilot_free` | 506 |
| `pilot_photoguard` | 523 |
| `pilot_advdm` | 709 |
| `pilot_mist_target` | 772 |
| `pilot_out_fixed` | 836 |

評估每臂 32 列約 3.8 分鐘（單卡）。

## 誠實區

- **4 格、單種子。** `out_fixed` 與 `photoguard` 的 +3.7% 領先在這個規模下撐不住
  任何結論。先前批內重跑帶要到九顆種子才穩下來。
- **沒有隨機對照。** 不知道這兩個的領先有沒有超過「隨便一條平滑曲線」。
- **指令完成度沒算**（VQA 不在這條鏈裡）。
- **編輯輸出的色偏很重，逐張看過**：`free` 整片螢光綠、`photoguard` 紫丁香或
  橘褐、`advdm` 橘紫、`out_fixed` 藍綠。人都還認得出來，但這些不是自然照片——
  「可用」那條腿在圖上是垮的，NIQE 之類的自動讀數抓不到。
- **一個單張單類的例外值得記**：`task_env_weather_121086` 的 `add_hat`，
  未防禦版有明顯的卡其色圓盤帽，而 `free`／`photoguard`／`advdm`／`out_fixed`
  四個臂**頭上是光的**，只有 `mainstream_both` 長出一頂粉色小帽。這是本專案
  第一次在圖上看到指令腿真的被擋。但 `79926` 的帽子五臂全部長出來、圍巾兩張圖
  五臂全中，所以是單張單類現象，不是通則。

## 檔案

| 東西 | 路徑 |
|---|---|
| 防禦圖與逐臂 CSV | `runs/advcf_pilot/<臂>/` |
| 防禦圖讀數 | `runs/advcf_pilot/defended_readout.csv` |
| 評估與編輯圖 | `runs/advcf_pilot_eval/<臂>/` |
| 位移三聯 | `runs/advcf_pilot_distance/` |
| 未防禦預檢 | `runs/pilot_preflight/`（遠端） |
