# AdvCF 載體固定，只改目標函數、訓練方法與預算

`configs/advcf_objective.json`，`scripts/paper_baseline.py --objective free`。
三張人像（`data/portrait_trio_manifest.json`）× 十九個臂，評估
`configs/evaluate_instruction.json`（每臂 216 列＝9 格 × 3 種子 × 2 臂 × 4 道淨化），
指令完成度 `scripts/instruction_vqa.py --edits`（identity 那一道，每臂 54 列）。

載體在所有臂都是 `ColorCurveParam(bound_mode='advcf', pieces=64)`——AdvCF 的單調
分段線性 tone curve。**載體家族一次都沒有換過。**

## 批內對照與雜訊底

跨批對照在本專案不成立，所以這一批自帶兩組對照：

- `advcf_control`：`runs/paper_trio/` 的 `trio_advcf_paper` 逐欄複製
  （半徑 1.0、150 步、lr 0.05、凍結抽樣）。
- `advcf_control_repeat`、`advcf_expect_repeat`：同一份設定的第二次求解。

**求解本身幾乎完全可重現**，變異在評估那一端：

| 量 | `advcf_control` | `advcf_control_repeat` |
|---|---|---|
| 防禦圖 ΔE00（三張） | 24.66／8.53／16.38 | 24.68／10.45／16.28 |
| 過半穿透 identity | 2/9 | 1/9 |
| 過半穿透 jpeg | 0/9 | 1/9 |

`advcf_expect` 對 `advcf_expect_repeat` 同樣：ΔE00 20.40／12.91／16.73 對
20.40／12.89／16.39，過半穿透四道全部 0 對 0。

**所以過半這一欄的雜訊底是 ±1 格。** 任何臂要說比對照好，差距要超過這個。

## 一、目標函數與訓練方法的十二個臂

| 臂 | 改了什麼 |
|---|---|
| `advcf_expect` | 每一次求值重抽噪聲與四個時刻，300 步，學習率餘弦降到五分之一 |
| `advcf_expect_deep` | 同上，另把取樣鏈由 6 步加到 10 步、反傳段由 1 加到 3 |
| `advcf_expect_face` | 同 `advcf_expect`，`enc` 與 `cond` 的範數在臉框的潛座標上加權 9 倍 |
| `advcf_expect_face_restart` | 同上，另加三次隨機起點重啟，取固定驗證抽樣上最好的一輪 |
| `advcf_expect_cond` | 同 `advcf_expect`，`cond` 權重由 1.0 拉到 20.0 |
| `advcf_expect_noid` | 同上，`id` 權重歸零 |
| `advcf_expect_noid_jitter` | 同上，起點加 `init_jitter 0.5` |
| `advcf_expect_jitter` | 同 `advcf_expect`，只多一個非恆等起點 |
| `advcf_expect_face_cond` | 臉框加權與 `cond` 重配權疊加 |

### 過半穿透（每格三顆種子過半，四道淨化）

| 臂 | identity | jpeg | blur | crop |
|---|---|---|---|---|
| `advcf_control` | 2 | 0 | 1 | 1 |
| `advcf_control_repeat` | 1 | 1 | 1 | 2 |
| `advcf_expect` | 0 | 0 | 0 | 1 |
| `advcf_expect_repeat` | 0 | 0 | 0 | 0 |
| `advcf_expect_deep` | 2 | 1 | 1 | 2 |
| `advcf_expect_face` | 1 | 1 | 0 | 1 |
| `advcf_expect_face_restart` | 1 | 0 | 1 | 1 |
| `advcf_expect_cond` | 1 | 1 | 0 | 1 |
| `advcf_expect_face_cond` | 2 | 1 | 0 | 1 |
| `advcf_expect_jitter` | 0 | 0 | 1 | 2 |
| `advcf_expect_noid` | 0 | 0 | 0 | 0 |
| `advcf_expect_noid_jitter` | 0 | 0 | 1 | 1 |

**十二個臂全部落在對照的 ±1 格之內**（對照兩次是 identity 2 與 1、jpeg 0 與 1、blur 1 與 1、crop 1 與 2）。`advcf_expect_noid` 的四個零是求解沒有離開起點，不是防禦失敗，見下。 完整的四道淨化 × 三個門檻欄
（任一種子／過半／全種子）在 `summary.csv`。

### 重抽確實壓低了代理的泛化落差

`scripts/free_proxy_readout.py` 把**所有臂的防禦圖**丟進**同一組**抽樣量三項，
`train` 是 `noise_seed = 0` 的凍結抽樣（凍結那一臂就是對著它解的），
`probe` 是另一組與訓練無關的抽樣：

| 臂 | train | probe | 落差 |
|---|---|---|---|
| `advcf_control` | +0.6396 | +0.7010 | **+0.0614** |
| `advcf_control_repeat` | +0.6438 | +0.7002 | **+0.0564** |
| `advcf_expect` | +0.6791 | +0.7121 | **+0.0330** |
| `advcf_expect_repeat` | +0.6801 | +0.7096 | **+0.0296** |
| `advcf_expect_face` | +0.7046 | +0.7354 | +0.0307 |
| `advcf_expect_deep` | +0.6163 | **+0.6753** | +0.0590 |
| `advcf_expect_face_restart` | +0.6154 | +0.6898 | +0.0745 |

（`proxy_readout.csv` 有全部十六個臂的逐張值。）

**落差縮小只在部分重抽臂成立**：`advcf_expect` +0.0330、`advcf_expect_repeat` +0.0296、`advcf_expect_face` +0.0307 確實低於對照的 +0.0614／+0.0564，但 `advcf_expect_deep` 是 +0.0590、`advcf_expect_face_restart` 是 +0.0745，並沒有縮小。probe 的絕對分數也不是都變差——`advcf_expect_deep` 的 +0.6753 低於對照的 +0.7010（較好），`advcf_expect` 的 +0.7121 則較差。`advcf_expect` 的過半穿透是 identity 0、jpeg 0、blur 0、**crop 1**，不是四道全零。

### 恆等起點是零梯度點

`advcf_expect_noid` 交付的是**原圖**：ΔE00 0.00、LPIPS 0.0000、PSNR 80、
criterion 四道全部 1.0000。

原因在 `src/defense/instruction_free.py`：`enc` 與 `cond` 都是差向量的範數，
而載體在恆等起點上 `render(x) == x` 逐位元成立，那兩個差是零向量，
`torch.norm` 在零點回傳零次梯度。`id` 權重歸零之後第 0 步的總梯度精確為零，
Adam 一步都不動。

同一組設定加上 `ColorCurveParam(init_jitter=0.5)` 的非恆等起點就會動
（`advcf_expect_noid_jitter`，ΔE00 24.61／11.28／13.23）。
`advcf_expect_jitter` 把「起點」與「拿掉 id 項」分開。

**這也表示恆等起點的多次重啟沒有意義**：`reset` 不帶亂數時每一次重啟都回到
同一點。`ColorCurveParam` 因此新增 `init_jitter`。

## 二、預算階梯（半徑掃描）

其餘逐欄與 `advcf_control` 相同，只改 AdvCF 自己的半徑。**半徑不換參數化**：
單調分段線性曲線的構造在任何半徑下都做不出逐色格跳變。

| 臂 | 半徑 | 防禦圖 ΔE00（三張） | identity 過半 | jpeg | blur | crop | criterion 中位（identity） |
|---|---|---|---|---|---|---|---|
| `advcf_radius_r05` | 0.5 | 18.24／6.71／10.98 | 1 | 0 | 0 | 1 | 0.9652 |
| `advcf_control` | 1.0 | 24.66／8.53／16.38 | 2 | 0 | 1 | 1 | 0.9423 |
| `advcf_radius_r20` | 2.0 | 30.85／15.46／21.85 | **3** | **3** | 1 | **4** | 0.8558 |
| `advcf_radius_r30` | 3.0 | 34.40／18.32／24.72 | **3** | 2 | 1 | **5** | **0.8316** |
| `advcf_radius_r50` | 5.0 | 37.99／25.63／27.70 | 1 | 2 | 1 | **4** | 0.8954 |

半徑 2.0 與 3.0 在 identity、jpeg、crop 三道上同時超出對照的 ±1 格。
半徑 5.0 的 criterion 中位仍低於對照，但格數回落——九格三種子的規模下格數
對切點敏感，兩個讀數照實並列。


## 二之二、九顆種子的複驗（`runs/advcf_objective_seeds/`）

上表的比較只有三顆種子，而同一份設定解兩次的過半格數就差 ±1 格。
`configs/evaluate_instruction_seeds.json` 把種子由三顆加到九顆（前三顆不變），
每臂 648 列，對五個臂重跑：

| 臂 | identity 任一 | 過半 | 全種子 | criterion 中位 | id 中位 | 指令完成 |
|---|---|---|---|---|---|---|
| `advcf_control` | 5/9 | 0 | 0 | 0.9435 | 0.9442 | 69/81 |
| `advcf_control_repeat` | 5/9 | 0 | 0 | **0.9444** | 0.9548 | 70/81 |
| `advcf_radius_r20` | 6/9 | **3** | 1 | **0.8765** | 0.8862 | 72/81 |
| `advcf_radius_r30` | 5/9 | 1 | 1 | 0.8861 | 0.8820 | 67/81 |
| `advcf_radius_r50` | 6/9 | 2 | 1 | 0.8709 | 0.8658 | **63/81** |

未防禦的指令完成是 70/81。

**兩個對照的 criterion 中位在九顆種子下對到小數第三位**（0.9435 對 0.9444），
而半徑 2.0 以上全部落在 0.87–0.89。**criterion 中位是這一批最穩的讀數**，
格數欄仍然抖（`advcf_radius_r30` 的過半只有 1）。

四道淨化的完整表在 `summary_seeds.csv`／`summary_seeds.txt`。四道上的
criterion 中位：

| 臂 | identity | jpeg | blur | crop |
|---|---|---|---|---|
| `advcf_control` | 0.9435 | 0.9508 | 0.9342 | 0.9381 |
| `advcf_control_repeat` | 0.9444 | 0.9388 | 0.9268 | 0.9506 |
| `advcf_radius_r20` | 0.8765 | 0.9141 | 0.8969 | 0.8824 |
| `advcf_radius_r30` | 0.8861 | 0.8834 | 0.8716 | **0.7965** |
| `advcf_radius_r50` | 0.8709 | 0.8615 | 0.8389 | **0.7662** |

**種子數換了就不能跟三顆種子的表並列**：任一種子那一欄會機械性上升
（抽的次數變多），過半那一欄會下降（九顆裡要五顆比三顆裡要兩顆嚴）。

**指令完成度在半徑 5.0 才明顯掉**：63/81 對未防禦的 70/81。
半徑 2.0 的 72/81 比未防禦還高一格，那是 VQA 讀數自己的抖動，不是防禦讓
指令更容易完成。


## 二之三、兩個槓桿的疊加（九顆種子）

目標函數那一族在半徑 1.0 上全部落在重跑帶內，半徑那一族在固定目標函數上把
criterion 中位由 0.944 壓到 0.87–0.89。這三個臂把兩者放在一起：
`advcf_deep_r20`／`advcf_deep_r30` 用 `advcf_expect_deep` 的目標函數與訓練方法
（每步重抽、鏈長 10、反傳 3、300 步、餘弦學習率），`advcf_face_r20` 用
`advcf_expect_face` 的臉框加權。

| 臂 | identity criterion | jpeg | blur | crop | 發布圖身分（三張） | 指令完成 |
|---|---|---|---|---|---|---|
| `advcf_control` | 0.9435 | 0.9508 | 0.9342 | 0.9381 | 0.958／0.973／0.963 | 69/81 |
| `advcf_radius_r20` | 0.8765 | 0.9141 | 0.8969 | 0.8824 | 0.934／0.972／0.902 | 72/81 |
| `advcf_deep_r20` | 0.8999 | 0.9282 | 0.9127 | 0.9444 | 0.933／0.983／0.927 | 69/81 |
| `advcf_face_r20` | 0.9539 | 0.9508 | 0.9709 | 0.9630 | 0.975／0.883／0.949 | 68/81 |
| `advcf_radius_r30` | 0.8861 | 0.8834 | 0.8716 | 0.7965 | 0.903／0.967／0.867 | 67/81 |
| **`advcf_deep_r30`** | **0.7315** | 0.8289 | 0.8360 | 0.8251 | 0.891／0.971／0.831 | 69/81 |
| `advcf_radius_r50` | 0.8709 | 0.8615 | 0.8389 | 0.7662 | 0.844／0.925／0.806 | 63/81 |

**疊加在半徑 2.0 上不成立，在 3.0 上成立。** `advcf_deep_r20` 的 criterion
中位在四道上都高於（較差）純半徑的 `advcf_radius_r20`；`advcf_deep_r30` 的
identity criterion 是 0.7315，是全部十九個臂裡最低的，比同半徑的
`advcf_radius_r30`（0.8861）低 0.155。`advcf_face_r20` 幾乎回到對照。

**這 0.155 不是拿發布圖身分換來的。** `advcf_deep_r30` 與 `advcf_radius_r30`
的發布圖身分只差 0.036（最低一格 0.831 對 0.867），而與 `advcf_radius_r50`
相比，`advcf_deep_r30` 的發布圖身分**更高**（最低一格 0.831 對 0.806）而
criterion **更低**（0.7315 對 0.8709）。同樣的比較在指令完成度上也成立：
`advcf_deep_r30` 是 69/81（未防禦 70/81），`advcf_radius_r50` 是 63/81。

**單一臂，未重跑。** 對照兩次的 criterion 中位對到小數第三位，但那是半徑 1.0
上的雜訊估計；半徑 3.0 上沒有對應的重跑對照，這個 0.155 還沒有自己的誤差帶。

## 三、四個讀數並列

**發布圖的身分沒有被當成代價付掉。** 十二個目標函數臂的
`subject_id_defended` 全部落在 0.93–0.99（`defended_readout.csv`）。半徑往上才
開始付：r2.0 最低 0.9024、r3.0 最低 0.8669、r5.0 最低 0.8063，三格都在
`task_obj_swap_rand_mask_79926`。



**指令完成度只有在半徑 5.0 才動。** 未防禦 23/27；防禦後十二個目標函數臂與
r0.5／r2.0／r3.0 都是 22–23/27，**r5.0 是 20/27**。判準的第三條腿在此之前
一次都沒有被擋下來。

**`use_norm` 不是主導項，但不是從不當最小項。** 九顆種子、identity 那一道的`argmin_use` 逐臂是 4 到 28（`advcf_radius_r30`／`r50` 各 4、`advcf_control` 20、`advcf_face_r20` 28），`argmin_id` 是 53 到 77。先前寫的「從來不是最小項」被 CSV 否證。

**位移指標三聯**（`before`／`after`／`gain`）在 `runs/advcf_objective_distance/`，
1728 組，由 `scripts/edit_distance_panel.py` 讀已存的編輯圖算出，不重跑擴散。

## 誠實區

- 三張人像、三顆種子＝九格，是 pilot 規模。零次成功時三顆種子的單側 95% 上界
  仍有 63%。
- 半徑 2.0 以上的產物自然度**沒有自動讀數可以判**。NIQE 比（`niqe_defended / niqe_orig`）**分得開，但方向與自然度相反**：半徑越大比值越低（r50 在 `task_obj_remove_410264` 是 0.760、r30 是 0.843），而那幾張正是調色最重的。NIQE 低代表「更像自然影像」，與看圖的判斷矛盾，所以它仍然不能當自然度門檻——但「全部落在 0.99–1.01」這句話是錯的，實測跨 0.760 到 1.027。 要看圖，`_report_metrics/report.html` 是單一自足檔案。
- 半徑階梯的格數不是單調的：r3.0 的 criterion 中位最低（identity 0.8316），
  但 r5.0 的過半格數在 identity 上回落到 1。九格三種子下格數對切點與種子都
  敏感，兩個讀數照實並列，不代為裁定哪一個才算。
- `advcf_expect_deep` 的取樣鏈設定與其他臂不同，但
  `scripts/free_proxy_readout.py` 的共同量測用的是設定檔的 `free` 區塊
  （鏈長 6、反傳 1），所以它那一列量的不是它自己訓練時的鏈。
- VQA 讀數仍未進 `src/defense/criterion.py` 的 `TERMS`，也未進損失。

## 檔案

| 東西 | 路徑 |
|---|---|
| 防禦圖（全解析度 PNG，16 臂 × 3 張）與逐臂 CSV | `runs/advcf_objective/<臂>/` |
| 彙整表 | `runs/advcf_objective/summary.csv`、`summary.txt` |
| 防禦圖讀數 | `runs/advcf_objective/defended_readout.csv` |
| 共同抽樣的代理讀數 | `runs/advcf_objective/proxy_readout.csv` |
| 評估與編輯圖 | `runs/advcf_objective_eval/<臂>/shard*/` |
| 指令完成度 | `runs/advcf_objective_vqa/<臂>/vqa_answers.csv` |
| 位移三聯 | `runs/advcf_objective_distance/shard*/` |
| 九顆種子的評估 | `runs/advcf_objective_seeds/<臂>/shard*/`、`summary_seeds.csv` |
| 九顆種子的指令完成度 | `runs/advcf_objective_seeds_vqa/<臂>/vqa_answers.csv` |
| 報告：十六個臂、三顆種子 | `_report_advcf/report.html`（1728 組編輯、3507 張影像） |
| 報告：八個臂、九顆種子 | `_report_advcf_seeds/report.html`（2592 組編輯、5244 張影像） |

判自然度要看全解析度的 PNG：`runs/advcf_objective/<臂>/<影像>__<臂>__defended.png`。
報告裡的是 q86 的 webp，壓縮本身會改變對「自不自然」的判斷。
編輯圖與淨化圖留在遠端 `runs/advcf_objective_eval/`（980 MB），報告已把全部
1728 組收進單一 HTML。
