# 評測

## 指標

統一清單，**每一份報表都報全部，不逐輪挑選**。程式在 `src/metrics/suite.py`，
欄位名與方向在 `src/metrics/standard.py`（缺欄位時直接拋 `KeyError`）。

| 指標 | 保真（防禦圖 vs 原圖） | 防禦（編輯後 vs 未防禦編輯） |
|---|---|---|
| LPIPS | ↓ 越低越好 | ↑ 越高越好 |
| FID | ↓ | ↑ |
| SSIM | ↑ | ↓ |
| PSNR | ↑ | ↓ |
| VIFp | ↑ | ↓ |
| DISTS | ↓ | ↑ |
| CLIP / SigLIP | 語意對齊，**照報但不作判準** | |
| HEval | 人眼判定，`compare.html` | |

### 三個必須知道的陷阱

**LPIPS 的 backbone 必須寫明。** `piq.LPIPS` 是 **VGG16**（已驗證與官方
`lpips(net='vgg')` 逐位相同）。AlexNet 給出的值低很多，且**比值隨方法而變**
（相位 5.31、DCT-Shield 2.41、加性 17.98），換 backbone 會改變勝負。

**FID 低於 150 張不可信。** `MetricSuite.FID_MIN_TRUSTED = 150`。Inception
pool3 是 2048 維，樣本數低於維度時協方差矩陣退化。`fid_batch.py` 會拒絕輸出，
除非加 `--allow-small`（該列會標 `trusted=False`）。小樣本的 FID 只能用於同一
批內的組間排序，絕對值不可與任何論文對照。

**VIFp 是內容主導的，不可跨資料集比。** 實測：同一張圖上 PSNR 差 3 dB 而 VIFp
幾乎不動（0.4824 vs 0.4817），但不同影像之間從 0.226 跳到 0.587。

## 攻擊模型

### InstructPix2Pix（主線）

`src/models/ip2p.py`。UNet 第一層卷積開 8 個輸入通道（4 噪聲 ＋ 4 影像），
**未加噪的 `E(x)` 直接拼在噪聲 latent 旁**，生成由純噪聲起步。

三個推論參數**論文未載，是本專案指定**，改動會讓新舊批次不可比：
`steps=100`、`s_T=7.5`（文字）、`s_I=1.5`（影像）、`seed=20260812`。

兩個容易補錯又不會報錯的地方，已由測試釘住：
- 拼進 UNet 的影像 latent **不乘** `scaling_factor`（`encode_image` 有乘，
  `image_latents` 沒乘，差一個 0.18 的倍率）。
- 載成一般 SD 1.5（UNet 4 通道）時影像條件會**靜默失效**，編輯退化成純文生圖。
  `_check_channels()` 會拒絕繼續。

### SDEdit（凍結）

`src/models/sd.py`。`z_t = √ᾱ_t·E(x) + √(1−ᾱ_t)·ε`，strength 0.7 對應
√ᾱ = 0.2873。這條線的結果保留在 `runs/sdedit_*`，不再新增。

**兩者在運算上不是介面差異。** SDEdit 的原圖只以「被噪聲稀釋的殘影」進入，
IP2P 是直接拼接。依賴前者那條通道的機制解釋在 IP2P 上不成立。

## 資料集

| 資料 | 內容 | 用途 |
|---|---|---|
| `data/omniedit150/` | OmniEdit dev split 的 150 張 `src_img`，五類任務各 30 張 | IP2P 主線 |
| `data/lo_aligned/` | 六類真實照片各 4 張 | SDEdit 線（凍結） |
| `data/apa_native/` | APA 官方三張 | APA 弱 baseline |
| `data/imagenet_advdrop/` | ImageNet 驗證 200 張，只留已被正確分類者 | AdvDrop／DJSMA 的原生威脅模型 |
| `data/targets/` | `gray.png` 等 targeted 目標 | 損失的目標影像 |

**OmniEdit 必須分層取樣，不可取前 N 張**——dev split 依任務排序，取前 N 會整批
落在同一類。`scripts/fetch_omniedit.py` 做這件事並寫出 `provenance.json`。

**影像清單是巢狀的**：25 ⊂ 75 ⊂ 150，存在 `runs/ip2p_fair_comparison/images*.txt`。
校準、正式表、抗淨化三個階段共用同一份，淨增益才能逐圖相減。

## 淨化算子

`src/purify/ops.py`。`identity` 不可排除，它是保留率的分母。

| 算子 | 設定 | 可微 |
|---|---|---|
| `identity` | — | — |
| `blur` | 高斯 σ=1.0 | 原生可微 |
| `jpeg` | 品質 75 / 30 | 直通估計 |
| `crop_resize` | 每邊裁 10%，bicubic 升回 | 原生可微 |
| `jpeg_then_resize` | C&R 串接 | 不可微 |
| `noise`／`quantize` | — | — |
| `adverse_cleaner` | 導向濾波 | 直通估計 |
| `impress` | 1000 步 Adam，佔一格約 82% 的時間 | 直通估計 |
| `gridpure`／`fdpure` | 超參數**論文未載，本專案指定** | 直通估計 |

尚未對現行條件測過的：`noise`、`quantize`、`jpeg30`、`jpeg_then_resize`、
`adverse_cleaner`、`impress`、`fdpure`。其中 **C&R 串接與 FD-Pure 針對性最強**。

## 工作點對齊

兩個方法的強度參數單位不同（DCT-Shield 的 `ε` 是量化階、本方法的 `θ` 是相位
半徑），**沒有任何一個軸天然對齊**。作法是掃描曲線 ＋ 內插錨點：

1. 各自掃三到四個強度，得到 (失真, 防禦效果) 的取捨曲線。
2. 在對方曲線上**線性內插**出失真相同之處，比防禦效果（等失真錨點）。
3. 反過來內插出效果相同之處，比失真（等效果錨點）。
4. **落在掃描範圍之外一律拒絕外插**，回報 `out_of_range`。

程式：`scripts/tradeoff_curve.py`。回傳值會把所依據的兩個端點一併報出。

**必須同時報兩個軸**（等 DISTS 與等 LPIPS）。實測 DISTS 與 LPIPS 對同一組
影像的判定經常相反，只在單一軸上成立的結論不算數。

## 防禦成功的讀數

**主讀數是人眼判定**：攻擊方拿不到可用的輸出即為「擋下」（內容被換成別的
場景，或劣化到不能用）；指令達成且輸出仍可用即為「攻擊成功」。逐圖判定與
理由記在 `runs/obedience_audit/defense_success_visual.csv` 的 `note` 欄。

**代理讀數是 SigLIP 影像相似度**：兩張編輯輸出在 SigLIP 影像空間裡的餘弦
相似度，低者為擋下。`MetricSuite.image_similarity`。

| 門檻 0.837 | 39 格人眼標記上正確率 93.5%，零誤報，AUC 0.974 |
|---|---|

它**不需要 caption**，這繞過了既有的障礙——`semantic` 量的是影像對一句
**描述**的相符度，而 OmniEdit 給的是**指令**，那條路徑在服從率驗收上近乎
隨機（25 張上 15/25 為正）。

三個使用上的限制：

1. **它是代理，金標準仍是人眼。** 用途是讓新工作點不必每一格都重看圖。
2. **門檻在高失真端偏保守**：高增益那組人眼判 12/13 非成功，門檻只標 8/13。
   門檻是在全部 39 格取最高正確率定出的，不是逐條件校準。
3. **位移續報但不作判準。** 同一條件內位移確實能分辨哪張被擋下
   （AUC 0.88–1.00），但約一半可由單純模糊複製，且在裁切縮放那格會把均勻
   色偏算成防禦。

## 未防禦攻擊的服從率

**位移只有在未防禦編輯確實執行了指令時才有意義。** 主線 25 張裡只有 13 張
成立，判定記在 `runs/obedience_audit/undefended_obedience.csv`。依任務型態
極不均：改顏色 5/5、換物件 3/5、換場景 3/5、**加物件 1/5**、**移除物件 1/5**。

在那 8 張完全沒動的影像上，位移照樣量得到而且**比真有攻擊的 13 張更高**
（本方法 0.2883 對 0.2556）。任何跨全批的平均都混進三分之一沒有攻擊可防的
格子。

## 抗淨化的讀數

驅動：`scripts/phase_retention.py`，只讀已存下的防禦圖，不重跑攻擊。

```
effect(p)   = LPIPS( 編輯(未防禦), 編輯(p(防禦圖)) )
保留率      = effect(p) / effect(identity)
空白地板    = LPIPS( 編輯(原圖), 編輯(p(原圖)) )        --floor
淨增益      = effect(p) − 空白地板                      ← 主讀數
```

`--floor` 那一格的「防禦圖」就是原圖本身，量到的是**算子自己造成的位移**。
它的 `effect(identity)` 由構造為 0，故保留率欄留空——**空白地板只看絕對值**。

`effect(identity)` 的多 seed 平均低於三倍標準差時該列標 `usable=False`，
排除在任何統計之外。
