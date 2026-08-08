# 文獻指標中本批算不出來的，以及為什麼

不算不等於忽略。逐項寫明理由，否則讀者會以為這些欄位是被跳過的。

| 指標 | 出自 | 為什麼算不出來 |
|---|---|---|
| FID、Precision／Recall | PhotoGuard-c、Mist、AdvPaint 的主指標 | 分布層級指標，需要數百張才穩定。本批 3 張影像 × 5 seed = 15 個樣本，算出來沒有意義（`SURVEY_2026-08-05` §4.2） |
| Aesthetic Score、PickScore | PromptFlare | 需要額外模型權重，未下載 |
| 背景保留（mask 隔離的 PSNR／LPIPS／MSE／SSIM） | DIA | 依賴 PIE-Bench 的編輯 mask。img2img 沒有 mask，該隔離程序無對應物（`SURVEY` §4.2） |
| ISR（原文形式） | SIFM | 判定由 MLLM 做，本專案沒有該 judge。已改為 ΔNIQE 門檻掃描的代理，**不是重現** |
| 人類排名 | DiffVax（67 名受試者） | 需要受試者招募 |

另有兩項是**可算但本批不具鑑別力**的：

- `cnn_denoise_substitute` 淨化算子缺權重，五個算子中有一個從頭到尾沒有資料
  （`RESULTS_2026-08-08` §7.1）。
- `cat_02` 在全部條件下的攻擊都成功（5/5），該影像不提供訊息
  （`RESULTS_2026-08-08` §9.6）。
