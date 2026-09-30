# EOT：把「攻擊者會先做一道前處理」放進最佳化的期望值

`configs/advcf_eot.json`，`scripts/paper_baseline.py --objective free`。
評估 `configs/evaluate_pilot.json`，與 `runs/advcf_pilot_eval/` 同一份。

**規模：兩張人像 × 兩類指令 × 一顆種子 = 4 格，每臂 32 列。**
分母是 `runs/advcf_pilot/pilot_free`（同設定、同影像、同種子，不重解）。
兩個臂的目標函數與分母**逐欄相同**，差別只有 `eot` 區塊。

## 為什麼跑

`src/defense/eot.py` 寫好很久，但**顏色線從來沒用過**——它只在位移場那條線
`runs/paper_purify_distance/` 量到 JPEG 與
blur 都把防禦推開的距離收回去，**blur 收得最多**（LPIPS gain −0.020 ～ −0.032），
所以 blur 是這個載體最弱的一道。

## 一個先前沒有的選點錯誤

`EOTObjective` 原本只有 `score`／`terms`，沒有 `eval_score`／`eval_terms`。
`optimise_carrier` 走的是 `getattr(objective, 'eval_score', objective.score)`，
所以它會退回 `score`，而 `score` **每一次求值都重抽變換**——checkpoint 是用
「這一步抽到哪一組裁切」挑的。這與 `CompositeObjective.eval_terms` 記的是同一類
失效，那一次讓 `runs/advcf_anchor/` 的四個 `out` 臂全輸、pilot 重跑後翻盤。

已補上：`_val_views` 每次呼叫把生成器重新播種，所以驗證用的那組變換是 `x_def`
的確定性函數，再乘上底層目標自己的固定驗證抽樣。訓練樣本數 1、驗證樣本數 2。
舊批次的數字不受影響。

## 誠實性：算子的族是看過的，參數不是

EOT 抽的是隨機參數（裁切比例在 2–15% 均勻抽、模糊 σ 在 0.4–1.6 抽），而評估
用的是固定的那幾組（裁切 10%、σ=1.0）。**不可以宣稱評估的淨化是完全未見的。**
JPEG 不進 EOT：它不可微，直通估計會讓梯度與真實算子脫節。
`include_identity` 一律開著——少了它，解可以押在「只有被處理過才成立」的地方，
而攻擊者不處理也照樣編輯得動。

## 結果（中位數）

| 臂 | identity | jpeg | blur | crop | ΔE00 | PSNR | 發布圖身分 | 秒／圖 |
|---|---|---|---|---|---|---|---|---|
| **`eot_blur`** | **0.4011** | **0.4168** | 0.3829 | 0.4362 | 15.94 | 18.62 | 0.9480 | 1034 |
| `pilot_free`（分母） | 0.3942 | 0.4082 | 0.3811 | 0.4335 | 15.73 | 19.03 | 0.9524 | 506 |
| `eot_both` | 0.3693 | 0.3698 | 0.3535 | 0.4143 | 15.39 | 18.63 | 0.9137 | 989 |

語意讀數（identity 那一道）：`eot_blur` CLIP 0.9086／SigLIP 0.8691、
`eot_both` 0.9023／0.8574、分母 0.9217／0.8806。

## 三件事

**一、`eot_blur` 在四道上全部小贏分母，但贏得最少的正好是它針對的那一道。**
identity +1.8%、jpeg +2.1%、crop +0.6%，而 **blur 只有 +0.5%**
（0.3829 對 0.3811）。它針對 blur 做期望，換到的是全域的一點點抬升，
不是 blur 那一道的抬升。

**二、`eot_both` 全面輸給分母**（identity −6.3%、blur −7.2%）。
每一步只抽一個算子，所以與 `eot_blur` 的每步成本相同，差別只有抽樣的族從
一個變成兩個。兩個算子分掉同樣的樣本預算之後，兩邊都沒學好。

**三、代價是兩倍的求解時間**（1034／989 秒對 506 秒），因為每一步要對
`1 + samples` 個視圖各跑一次完整的目標。

## 誠實區

- **4 格、單種子、沒有隨機對照。** +1.8% 在這個規模下撐不住任何結論。
- **`eot_blur` 與分母的差距（0.4011 對 0.3942）小於 `runs/advcf_objective/`
  量到的批內重跑帶。** 那一批同一份設定解兩次的 ΔE00 差到 0.1–1.9，而評估端的
  50 步擴散取樣本身就有 ±1 格的抖動。
- **指令完成度沒有走 VQA。** 逐張看圖：`task_env_weather_121086` 的
  `add_hat` 在 `eot_blur` 與 `eot_both` 上**頭都是光的**（未防禦長出卡其色圓盤帽），
  `add_scarf` 兩臂都長出圍巾。
- **發布圖不是自然照片**：`eot_blur` 整片綠、`eot_both` 青綠。要由使用者裁決。

## 檔案

| 東西 | 路徑 |
|---|---|
| 防禦圖與逐臂 CSV | `runs/advcf_eot/<臂>/` |
| 防禦圖讀數 | `runs/advcf_eot/defended_readout.csv` |
| 評估與編輯圖 | `runs/advcf_eot_eval/<臂>/` |
| 位移三聯 | `runs/advcf_eot_distance/` |
