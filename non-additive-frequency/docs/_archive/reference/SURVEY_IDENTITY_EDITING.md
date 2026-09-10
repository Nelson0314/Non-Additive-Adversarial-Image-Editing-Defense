# 身分保護 × 指令式編輯：這一族的論文用什麼指標

本檔回答一個問題：**與本專案現行威脅模型同型的論文（保護對象是人的身分、
攻擊方是擴散編輯），它們拿什麼數字說「有效」與「不明顯」。**

現行威脅模型是「臉逐位元不動、擾動全放在衣物上、攻擊方用 InstructPix2Pix
下指令編輯、讀數是編輯後還認不認得出那個人」。既有的量測協定（等失真對齊、
扣空白地板、兩個失真軸）在 `../EVALUATION.md`；本檔只記**別人怎麼量**。

全部逐篇讀過內容：FaceLock 讀 arXiv 全文並逐行對照官方 `methods.py`／
`evaluation/*.py`；Anti-DreamBooth 讀 ar5iv 全文的評測節；EditShield、
FaceShield 讀 arXiv HTML 全文的評測與穩健性節。

---

## 1. FaceLock（CVPR 2025）——與本專案重合度最高的一篇

Wang, Hu, Ye, Wang, Xie, Liu, Wu, Tang（TACO group）。
[arXiv:2411.16832](https://arxiv.org/abs/2411.16832) ／
[官方程式碼](https://github.com/taco-group/FaceLock)（已逐行讀過）。

### 為什麼它與本專案重合

| 項 | FaceLock | 本專案 |
|---|---|---|
| 攻擊模型 | InstructPix2Pix | InstructPix2Pix |
| 推論參數 | `num_inference_steps=100`、`image_guidance_scale=1.5`、`guidance_scale=7.5` | `steps=100`、`s_I=1.5`、`s_T=7.5` |
| 保護目標 | 編輯輸出裡的生物特徵 | 編輯輸出裡還認不認得出那個人 |
| 主讀數 | 原圖與編輯輸出的人臉餘弦 | 原圖與編輯輸出的人臉餘弦 |
| 指令 | 25 條，三類（五官／配件／背景） | 三類（clothing／accessory／background） |

**三個推論參數逐項相同。** `../EVALUATION.md` 現在寫它們「論文未載，是本專案
指定」——那句話對 IP2P 原論文成立，但 FaceLock 的官方 README 明給同樣三個值，
故本專案與它的編輯側設定天然對齊，頭對頭時不需要重跑對方的編輯。

**它明確主張既有指標會誤導**，理由與本專案踩過的坑同型（`../DEFECTS.md`、
`runs/ip2p_content_constraint/README.md` 的「三個讀數上的坑」）：

- CLIP 類分數「overemphasize the presence of elements from the editing
  instructions, often prioritizing over-editing」——編輯過頭的反而得高分。
- SSIM／PSNR「over-rely on differences between the edited image and the
  undefended source, potentially leading to a false sense of successful
  defense」——只是把像素推遠就拿得到分。
- 兩類都「can be easily manipulated to achieve deceptive results」。

### 它的六個指標（逐項對照官方 `evaluation/`）

全部比的是**編輯（防禦圖）對編輯（原圖）**，不是防禦圖對原圖。

| 指標 | 精確定義 | 由誰算 | 方向 |
|---|---|---|---|
| **FR** | `cos(E(原圖), E(編輯輸出))` | 對齊器 `minchul/cvlface_DFA_mobilenet`；辨識器 `minchul/cvlface_adaface_vit_base_kprpe_webface4m`（AdaFace ViT-Base ＋ KPRPE，WebFace4M） | ↓ 低者防禦強 |
| **CLIP-I** | `cos(E_img(編輯), E_img(原圖))` | `openai/clip-vit-base-patch32` | ↓ |
| **CLIP-S** | `cos(E_img(編輯) − E_img(原圖), E_txt(指令))` | 同上 | ↓ |
| **LPIPS** | `lpips.LPIPS(net='vgg')` | 官方 lpips 套件 | ↑ |
| **PSNR** | `torchmetrics` `data_range=1.0` | torchmetrics | ↓ |
| **SSIM** | torchmetrics | torchmetrics | ↓ |

**FR 的參照是原圖本身**（`eval_facial.py` 的 `src_dir` 是未編輯的來源），
與本專案 `src/metrics/identity.py` 的 `id_def` 結構完全相同，差別只在辨識器：
本專案用 `facenet-pytorch` 的 InceptionResnetV1（VGGFace2）＋ MTCNN，
它用 AdaFace ViT ＋ DFA 對齊器。

**它沒有 `id_orig` 那一項。** 本專案量 `id_orig`（攻擊自己就已經動了多少身分）
是為了扣掉「這個編輯本來就會把人改掉」，FaceLock 未做這件事。

**LPIPS backbone 相同**（VGG），故本專案的 LPIPS 與它的可以並列；
`../reference/BASELINE_ALIGNMENT.md` §1.2 的 backbone 警告在這一篇上不適用。

### 它的方法：損失不在編輯輸出上，在 VAE 往返上

逐行讀 `methods.py::facelock`：

```
latent = vae.encode(X_adv).latent_dist.mean
image  = vae.decode(latent).sample.clip(-1, 1)

loss = − FR(image, X) · [i ≥ 0.35N]        # 壓低重建圖的身分
     + 0.2 · MSE(latent, clean_latent)     # EditShield 的 untargeted latent 項
     + LPIPS(image, X) · [i > 0.25N]       # 壓低重建圖的感知相似度
X_adv ← X_adv + α·sign(∇loss)              # 之後投影回 L∞ 球並 clamp
```

三件值得記的事：

1. **完全不碰 UNet。** 只走 VAE 編碼與解碼，故單張成本極低（100 步）。
   本專案的補丁族單張 6000 步、約 50 分鐘；這一篇是分鐘級。
2. **兩個項有啟動排程**（FR 在 35% 之後、LPIPS 在 25% 之後才計入），
   論文正文未載，只在程式碼裡。移植時屬於 `SOURCE_AUDIT` 意義下的
   「原始碼有、論文沒有」。
3. **步長逐步衰減**：`α_i = α − (α − α/100)/N · i`。

### 預算：論文與程式碼不一致

| | 論文正文 | 官方程式碼預設 |
|---|---|---|
| `eps` | **0.02**（"we set the perturbation budget to 0.02 ... for all methods"） | **0.03** |
| `step_size` | 0.003 | 0.01 |
| `iters` | 100 | 100 |

移植時兩組都要能跑，CSV 要有欄位記下用的是哪一組，並標 `modified_from_paper`。

### 它的抗淨化設定

| 算子 | 精確設定 |
|---|---|
| Blur | 高斯模糊 `k=5`、`σ=1.5` |
| Rotate | 隨機旋轉 `(−10°, +10°)` |
| JPEG | 品質 90 / 75 / 60 |

報的數字是各算子之後的 LPIPS 與 FR。**沒有空白地板**（未扣掉「算子自己就會
把編輯推開」那一項），也**沒有等失真的隨機對照**，故它報的抗淨化數字不能與
本專案的淨增益並列——要並列必須用本專案的協定重跑它。

### 它比較的 baseline

PhotoGuard（encoder attack）、EditShield、untargeted encoder attack、
CW-L2 attack、VAE attack。後三個與 FaceLock 同在 `methods.py`，移植成本近乎零。

### 資料

CelebA-HQ 篩出的 2000 張人像；25 條指令（五官 10、配件 8、背景 7）。

---

## 2. Anti-DreamBooth（ICCV 2023）——身分保護指標的來源

Van Le, Phung, Nguyen, Dao, Tran, Tran。
[arXiv:2303.15433](https://arxiv.org/abs/2303.15433)。
威脅模型是 **DreamBooth 客製化**，不是指令式編輯，故不是 baseline；
但現行文獻的身分指標全部由它定型。

| 指標 | 精確定義 | 由誰算 | 方向 |
|---|---|---|---|
| **FDFR**（Face Detection Failure Rate） | 生成圖裡偵測不到臉的比率 | **RetinaFace** | ↑ 高者防禦強 |
| **ISM**（Identity Score Matching） | 生成圖的人臉嵌入對「使用者全部乾淨影像的平均嵌入」的餘弦 | **ArcFace** | ↓ |
| **SER-FIQ** | 人臉影像品質評估 | SER-FIQ | ↓ |
| **BRISQUE** | 通用無參照影像品質 | BRISQUE | ↑ |

擾動可見度側用 **PSNR ↑ 與 LPIPS ↓**（防禦圖對原圖）。

兩個與本專案直接相關的差別：

- **FDFR 用 RetinaFace，本專案用 MTCNN。** 本專案已記過 MTCNN 在人眼看得見的
  臉上誤報（`runs/ip2p_content_constraint/README.md`）。換成文獻的偵測器是
  一個可以直接做的對齊。
- **ISM 的參照是「乾淨影像集的平均嵌入」**，本專案的 `id_def` 參照的是單張
  原圖。多人合照時本專案的讀數不可靠（十張裡四張），ISM 的平均嵌入設計正是
  為了避開這件事，但它需要同一身分的多張影像，本專案的資料沒有。

---

## 3. FaceShield（ICCV 2025）——身分 ＋ 抗壓縮的設計

Jeong 等。[arXiv:2412.09921](https://arxiv.org/abs/2412.09921)。
威脅模型是 deepfake（換臉、屬性編輯），與本專案不同，但它的兩件事直接相關。

### 指標

防禦效果側（比的是「防禦圖的 deepfake 輸出」對「乾淨圖的 deepfake 輸出」）：

| 指標 | 定義 | 方向 |
|---|---|---|
| **L₂** | 兩個輸出的 L₂ 距離 | ↑ |
| **ISM** | 來源臉與輸出臉的相似度 | ↓ |
| **PSNR** | 兩個輸出之間 | ↓ |
| **HE** | 人眼 Likert 1–7，量「噪聲可見度」與「身分偏離」 | ↑ |

可見度側（防禦圖對原圖）：**LPIPS ↓、PSNR ↑、SSIM ↑**，另加它自訂的
**Frequency Rate（FR）↑**——擾動能量落在低頻的比例。它報自己 18.47、
baseline 約 1.6–2.1。

### 兩個可以借的機制

1. **選擇性模糊**：Sobel 找邊界（3×3 核、9×9 膨脹），只對邊界區的擾動做高斯
   模糊，`δ_blur = G(δ)⊙M_sob + δ⊙(1−M_sob)`。目的是壓可見度。
2. **低通投影**：對擾動做 8×8 DCT，只保留亮度量化表係數 < 40 的低頻分量，
   再逆變換回 RGB。**在 PGD 迴圈內每步做**，在裁到預算之前。目的是抗 JPEG。

第二項與本專案 `tint`（低頻懲罰）、`lowproj`（低頻替換）是同一個想法的兩種
實作——本專案是懲罰或替換補丁內容，它是投影擾動。**它把低頻化的理由寫成
抗 JPEG，本專案至今把它讀成「比較好看」。** 兩者是否同時成立，抗淨化那一批
會直接回答。

### 抗淨化設定

JPEG 品質 90 / 75 / 50；位元縮減 8-bit 與 3-bit；縮放 75% 與 50% 再還原
（BILINEAR 與 INTER_AREA，OpenCV 與 Pillow 都測）；高斯模糊。

### 預算與最佳化

`L∞ ≤ 12/255`，PGD 步長 `1/255`、30 步。

---

## 4. EditShield（ECCV 2024）——攻擊模型相同的另一篇

Chen 等。[arXiv:2311.12066](https://arxiv.org/abs/2311.12066)。
`../BASELINES.md` 已列為「裁切欄唯一的直接對照，尚未實作」。本次補到的細節：

| 項 | 值 |
|---|---|
| 預算 | `4/255` |
| PGD | 最多 30 步（`S=30`），`β=0.2` |
| 攻擊模型 | InstructPix2Pix（SD 1.5）與 IP2P-MagicBrush |
| 資料 | IPr2Pr 篩出 2000 張、MagicBrush test split 1000 張 |
| EOT 三個變換 | 高斯平滑 `k=5, σ=1.5`；旋轉 `5°`；中心裁切 |
| 受測的反制手段 | Spatial Smoothing（window=2）、JPEG（quality=80） |
| baseline | PhotoGuard |

指標：CLIP 影像相似度 ↓、CLIP 文圖方向相似度 ↓（描述由 **LLaVA-1.5 13B** 生成）、
PSNR ↓、SSIM ↓，另加 **GPT-4V 評分**（指令服從性與內容保真，0–1）與
**50 人的人眼評估**（20 組）。

**它的 EOT 旋轉是 5°、FaceLock 的淨化旋轉是 ±10°。** 本專案的九個算子裡
沒有旋轉，這一欄目前無法與任一篇對照。

---

## 5. 三篇的抗淨化算子與本專案既有算子的對照

| 文獻算子 | 精確設定 | 本專案 `src/purify/ops.py` |
|---|---|---|
| FaceLock Blur | 高斯 `k=5, σ=1.5` | 有 `blur`，但既有點是 σ=1.0 與 2.0，**σ=1.5 未跑過** |
| FaceLock JPEG | Q 90 / 75 / 60 | 三點全部已有 |
| FaceLock Rotate | 隨機 ±10° | **無** |
| EditShield SS | window=2 的空間平滑 | 近似 `blur`，但不是同一個算子 |
| EditShield JPEG | Q=80 | 未跑過（既有格點 90/75/60/50/40/30/20） |
| FaceShield 位元縮減 | 8-bit / 3-bit | `quantize`（既有 256/64/32/16/8 階）＝ 8-bit 與 3-bit |
| FaceShield 縮放 | 75% / 50% 降採樣再還原 | `resize_only`、`resample_roundtrip` |
| 真實世界串接 | JPEG75 → 0.5× Lanczos | `jpeg_then_resize` |

**唯一的缺口是旋轉。** 其餘全部可由既有算子覆蓋，只需補 `blur σ=1.5` 與
`jpeg 80` 兩個格點。

---

## 6. 這一族沒有解決的兩件事，本專案有

1. **沒有空白地板。** FaceLock、EditShield、FaceShield 報的都是絕對值，
   沒有扣掉「淨化算子自己就會把編輯推開」那一項。本專案的
   `scripts/phase_retention.py` 的 `--floor` 是為此設計的。
2. **沒有等失真的隨機對照。** 本專案已量到淨化之後最佳化相對同幾何隨機雜訊的
   優勢幾乎消失（裁切 1.13、jpeg→resize 1.04，
   `runs/ip2p_patch_breadth_purify/README.md`）。沒有隨機對照就分不出
   「防禦守住了」與「衣服上有一塊很吵的東西」。

反過來，本專案缺而它們有的是**辨識器的多樣性**：三篇分別用 AdaFace ViT、
ArcFace、以及各自的臉部特徵抽取器，本專案只有一個 facenet 的
InceptionResnetV1。單一辨識器的結論可能是那個網路的特性。

---

## 7. 可見的防禦：文獻上沒有對應的失真軸

本專案已裁定「可見的防禦是允許的，條件是主體不動」。本次查證的結果是
**這一族全部假設不可見**：

- FaceLock `L∞ ≤ 0.02`、EditShield `4/255`、FaceShield `12/255`、
  Anti-DreamBooth 與其餘皆為 `L∞` 球內的加性擾動。
- 可見度指標一律是 PSNR ↑ / SSIM ↑ / LPIPS ↓ / FID ↓（防禦圖對原圖），
  全部建立在「擾動應該小」這個前提上。
- 最接近的立場支撐是 Zhu et al.（CVPR 2024）對 chaotic texture 的批評
  （見 `SURVEY_WATERMARK_ATTENTION.md`），但那一篇的擾動本身仍然不可見，
  可見的只有**輸出**上的浮水印。

**後果**：本專案的失真軸與這一族**不在同一個座標系**，
`BASELINE_ALIGNMENT.md` §2.1 的「統一像素預算」對本方法無定義
（補丁內的 `L∞` 是 1.0）。可用的是 §2.2（對齊防禦效果再比失真）與
§2.3（掃強度畫取捨曲線）：把 baseline 的 `ε` 掃成一條曲線，本方法落在那條
曲線的哪一側是可以講的；單點的 PSNR 對 PSNR 不可以講。
