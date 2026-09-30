# 三篇防護論文在各自原生設定下的重現

`scripts/apa_baseline.py --data data/lo_aligned`，受害模型 SD 1.4、SDEdit
strength 0.8、`EDIT_STEPS 30`、`EDIT_SEED 20260812`。

## 這一批要回答什麼

外部比較的前提是**我們跑得出每一篇該有的實力**。在那之前，任何「我們贏了誰」
都讀不出來。所以這一批不比較，只做一件事：把 PhotoGuard、Mist、DIA 放在各自
論文／官方程式指定的預算上跑，看失真落在哪裡，再對回可引用的參考數字。

## 影像

`data/lo_aligned` 24 張裡選 10 張，人像 6、動物 4。人像偏多是因為 identity
讀數只有在人像上成立。

| shard | 影像 | 類別 | SDEdit prompt（`prompts[0]`） |
|---|---|---|---|
| shard1 | `man_00` `man_01` | man | an old woman |
| shard2 | `man_02` `woman_00` | man / woman | an old woman ／ a man |
| shard3 | `woman_01` `woman_02` | woman | a man |
| shard4 | `cat_00` `dog_00` | cat / dog | a dog ／ a cat |
| shard5 | `horse_00` `bird_00` | horse / bird | a zebra ／ a butterfly |

`prompt` 逐類別而非逐張，所以同類的圖共用同一句。

## 四個條件

預算逐項的出處見 `docs/reference/BASELINE_PROVENANCE.md`。

| 條件 | 約束（換算到 `[0,1]`） | 步數 | `grad_reps` | 這個設定來自 |
|---|---|---|---|---|
| `photoguard_c` | L2 = 8（整張圖範數） | 200 | 10 | 官方 notebook cell 10，唯一啟用的呼叫 |
| `photoguard_linf` | `L∞` 16/255 | 200 | 10 | 論文 Appendix A.2 / Table 9（p.15） |
| `mist` | `L∞` 16/255 | 100 | 1 | `mist_utils.py` CLI 預設，與論文正文一致 |
| `dia_r` | `L∞` 0.025 | 20 | 1 | 論文 §4.1 ＋ `attack_setting.json` |

PhotoGuard 之所以要兩個臂，是因為**論文與官方程式互相矛盾**，而 repo 裡不存在
同時滿足 Table 9 四欄的程式碼。兩個臂只差約束本身（`grad_reps` 都是 10），
其餘一致，這樣才讀得出約束造成了什麼。

`photoguard_linf` 的輸出在 `runs/pg_linf/{a..e}/` 與 `runs/pg_linf_probe/`。

## 量到什麼（`photoguard_c`／`mist`／`dia_r`，10 張的中位數）

| 條件 | 秒/圖 | PSNR | rms | LPIPS(VGG) | DISTS | SSIM | edit_lpips |
|---|---|---|---|---|---|---|---|
| `photoguard_c` | 6526 | 40.90 | 0.0090 | 0.3625 | 0.0333 | 0.9809 | 0.5299 |
| `mist` | 88 | 26.56 | 0.0470 | 0.6054 | 0.1608 | 0.7768 | 0.5651 |
| `dia_r` | 154 | 39.67 | 0.0104 | 0.2621 | 0.0393 | 0.9805 | 0.3148 |

`rms` 由 PSNR 反推。LPIPS 的 backbone 是 VGG16（`piq.LPIPS`，與官方
`lpips(net='vgg')` 逐位相同）。四類讀數（含 ArcFace 與 NR-IQA）在 `panel.csv`，
由 `scripts/readout_panel.py` 讀已存的 PNG 補齊，不重跑攻擊。

### 讀數一：Mist 落在參考值上

PSNR 26.56 對 DCT-Shield Table 1 的 26.62（差 0.06 dB），rms 0.0470 對 0.0467。
逐張落在 25.95–26.82，隨影像內容變動——`L∞` 約束下本來就該如此。Mist 的原生
預算本來就是 `L∞ 16/255`，與 DCT-Shield 統一給所有 baseline 的預算相同，
所以對得上。**這同時證明本專案的量測管線與那張表是可比的。**

### 讀數二：`photoguard_c` 的 PSNR 是算出來的常數

十張圖的 `fid_psnr` 是 40.894、40.894、40.894、40.895、40.895、40.896、40.896、
40.897、40.899、40.900——**全距 0.006 dB**，與影像內容無關。同一批的 `mist`
在 `L∞` 下全距是 0.87 dB，對照之下這個常數性不可能是巧合。官方的投影是
`torch.renorm(d_x, p=2, dim=0, maxnorm=16)`——`(1,3,512,512)` 沿 `dim=0` 只有
一個切片，約束的是整張影像在 `[-1,1]` 下的 L2 範數。飽和時

    rms_[0,1] = 16 / (2·√(3·512·512)) = 0.0090211
    PSNR      = −20·log₁₀(0.0090211) = 40.8948 dB

實測逐位落在這個算術值上。**在這個預算下 PSNR 被鎖住，加步數或加大
`grad_reps` 都突破不了**，所以「跑不到論文的失真量級」不是收斂問題。

### 讀數三：先前拿來對照的數字出處是錯的

`BASELINE_ALIGNMENT.md` §1.1 原本把 PhotoGuard 那一列的 28.32／0.284 標成
「論文」值。它們出自 **DCT-Shield Table 1**，協定是 `L∞ 16/255` ＋ 150 張
OmniEdit ＋ InstructPix2Pix。**PhotoGuard 自己的論文不含這組數字**；該文
Table 6 的 diffusion attack PSNR 13.58±2.23 量的是編輯結果之間的差異，
不是防禦圖對原圖。欄名已更正，並補上 §1.1.1 列出三個來源的三個不同數字。

## 成本：推估值全部偏低

先前由呼叫次數推的成本與實測的比較：

| 條件 | 推估 | 實測 | 倍數 |
|---|---|---|---|
| `photoguard_c` | 1130 s | 6562 s | 5.8× |
| `mist` | 12 s | 88 s | 7.3× |
| `dia_r` | 57 s | 154 s | 2.7× |

低估的原因是把「反傳」當成前向的兩倍計價。PhotoGuard 的 complex attack 是對
**整條展開的去噪鏈加上 VAE decode** 反傳（損失是 `‖D(z̄₀) − target‖²`），
保留的活化與 decode 的代價遠超過 2×。排後續批次一律用實測值。

## 這一批沒有的東西

- **沒有 identity 欄。** `apa_baseline.py` 的輸出早於本專案的身分讀數，
  FaceNet 與 ArcFace 都不在裡面。外部比較要用的話得在轉接層補。
- **沒有 NR-IQA 盤**（NIQE／BRISQUE／CLIP-IQA／MUSIQ／TOPIQ）。現有的自然度
  欄位只有 `nima`／`cnniqa`。
- **沒有抗淨化。** 這一批只跑 `identity` 一道。
- **inpainting 的三篇（AdvPaint／PromptFlare／DiffusionGuard）跑不了**：
  它們需要逐影像的遮罩，而 `data/lo_aligned/masks/` 不存在。依 DEC-010
  那些遮罩是**人工畫的、不由模型產生**，所以這一格要等人補圖。
