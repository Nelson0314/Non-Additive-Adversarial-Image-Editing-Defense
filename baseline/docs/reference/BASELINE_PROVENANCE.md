# Baseline 出處總表

一個 baseline 的數字要能被引用，必須同時說得出四件事：**哪一篇、那篇寫什麼、
官方程式做什麼、我們跑什麼**。四者經常不一致，而不一致本身就是要報的東西。

本檔只放這四欄與它們的出處。逐行對照的證據在 `AUDIT_*.md`，工作點如何對齊在
`BASELINE_ALIGNMENT.md`。

## 規則

1. **引用一個數字就要連它的協定一起引用。** 「PhotoGuard 的 PSNR 是 28.32」
   是一句不完整的話，因為那個數字是 DCT-Shield 在 `L∞ 16/255` ＋ IP2P 下重跑
   出來的，不是 PhotoGuard 論文的數字。
2. **欄名要寫出處，不可只寫「論文」。** 一張表裡的「論文」可能指四篇不同的
   論文，讀表的人分不出來。
3. **論文與官方程式不一致時兩者都列**，並在 `BaselineSpec.discrepancy_note`
   寫明。不得挑一個當作「那篇的設定」。
4. **反推的數不是量到的數。** 由 PSNR 反推 rms 要標星號，並記住算術平均的
   PSNR 反推出的是幾何平均的 rms。
5. **查不到就寫「未找到」並說明查了哪裡，不填一個看起來合理的值。**
   停住的成本是等一次人看；填一個猜測值的成本是整批作廢，而且**不會有人
   發現**——結果看起來完整、格式也對，錯的只有那一個沒有根據的數。
   這條不只適用於查文獻：**任何由資料算出來的參數算不出來時同樣適用。**
   `main_table` 的 eps 對齊掃描就是一例，內插用的目標值若落在掃描到的區間
   之外（只能外插），求解腳本寫一個 `*_NEEDS_DECISION` 旗標並停住，不拿外插值
   去跑——那一批是 8 張 × 6,400 秒／張。

## 預算總表

下表的七個方法共用 `src/baselines/pgd.py` 的 PGD 骨幹，`eps01` 是各自的
約束換算到 `[0,1]` 像素域的值。**換算逐篇不同，`eps01` 相同不代表束縛相同**：
束縛種類（`L∞`／`L2`）不同的兩列，該欄不是同一個量。主表十二列的逐列單位
見下一節。

| 條件 | 論文 | 約束（`eps01`） | 步長 | 步數 | 更新 | `grad_reps` | 這個設定的出處 |
|---|---|---|---|---|---|---|---|
| `photoguard_c` | [2302.06588](https://arxiv.org/abs/2302.06588) | **L2 = 8**（整張圖範數） | 1.0 | 200 | normalized grad | 10 | 官方 notebook cell 10，唯一啟用的呼叫 |
| `photoguard_linf` | 同上 | `L∞` 16/255 = 0.0627 | 2/255 | 200 | sign | 10 | 論文 Appendix A.2 / Table 9（p.15） |
| `mist` | [2305.12683](https://arxiv.org/abs/2305.12683) | `L∞` 16/255 = 0.0627 | 1/255 | 100 | sign | 1 | `mist_utils.py` CLI 預設，與論文正文一致 |
| `dia_r` / `dia_pt` | [2510.00778](https://arxiv.org/abs/2510.00778) | `L∞` 0.025 = 6.4/255 | 1/255 | 20 | sign | 1 | 論文 §4.1 ＋ `attack_setting.json` |
| `advpaint` | [repo](https://github.com/JoonsungJeon/AdvPaint) | `L∞` 0.03 = 7.65/255 | 0.03 | 100 | sign | 1 | 函式簽章預設（CLI 預設 0.1 與之矛盾，見 §落差） |
| `promptflare` | [repo](https://github.com/NAHOHYUN-SKKU/PromptFlare) | `L∞` 0.0235 = 6/255 | 1/255 | 400 | sign | 1 | `protect.py` argparse 預設 |
| `diffusionguard` | [2410.05694](https://arxiv.org/abs/2410.05694) | `L∞` 16/255 = 0.0627 | 1/255 | 800 | sign | 1 | repo `config/diffusionguard.yaml` |

**原生預算差到 2.6 倍**（6/255 到 16/255），而 `photoguard_c` 的 L2 路徑換算成
等效 rms 只有 0.0090，比 `L∞ 16/255` 的上界小一個數量級。**任何不對齊預算的
頭對頭比較都不成立**，對齊的做法見 `BASELINE_ALIGNMENT.md` §3（DEC-029：掃強度
畫取捨曲線，在曲線上標等失真與等效果兩個錨點）。

## `eps_pixel01` 欄的單位，逐列

`main_table/results/defence_*.csv` 的 `eps_pixel01` 要**連同一列的 `norm` 欄一起讀**。
`norm` 記的是束縛種類，主表十二列分成五種：

| `norm` | 條件 | 該列 `eps_pixel01` 量的是什麼 |
|---|---|---|
| `linf` | `mist`、`photoguard_linf`（0.0627）、`dayn`、`sifm`、`danp`（0.03）、`dia_pt`、`dia_r`（0.025） | 逐像素的 `L∞` 上界，`[0,1]` 尺度 |
| `l2` | `photoguard_c`（8.0） | **整張影像的 `L2` 範數**，不是逐像素值 |
| `dct_coeff_linf` | `dct_shield`、`dct_shield_y` | 空。束縛下在 DCT 係數上，`eps` 欄的 1.0 是係數域的量 |
| `delta_e00_cap` | `color_curve` | 空。束縛是 ΔE00 上限，`eps` 欄的 16.0 是 ΔE00 |
| `none` | `diffvax` | 空。前饋路徑，沒有迭代預算 |

七個 `linf` 列之間，`eps_pixel01` 是同一個量。`photoguard_c` 的 8.0 與那七個數
**不同單位**，兩種束縛的幾何不同，不存在換算式；它飽和時的等效 rms 是
`16 / (2·√(3·512·512)) = 0.0090211`，那是**反推的數**（規則 4），本專案十張圖
實測 PSNR 全距 0.002 dB。其餘四列該欄為空，不是零。

值域換算的逐篇出處在 `SOURCE_AUDIT.md` §10；該節同時記載 Mist 的 `[-1,1]`
換算有乘 2、PromptFlare 沒乘 2，兩者的程式碼寫法看起來一樣但結果差兩倍。

## 已實作、已審計，但不在主表的方法

`advpaint`、`promptflare`、`diffusionguard` 三個有模組與審計文件，其中前兩個
在上面的預算總表也有一列，但主表十二列不含它們。可查證的狀態：

| 方法 | 模組 | 測試 | 在 `REGISTRY` | 防禦圖產物 |
|---|---|---|---|---|
| `advpaint` | `src/baselines/advpaint.py` | `tests/test_baselines.py` 的 `AUDIT` 有列 | 是 | 無 |
| `promptflare` | `src/baselines/promptflare.py` | 同上 | 是 | 無 |
| `diffusionguard` | `src/baselines/diffusionguard.py` | `tests/test_diffusionguard.py` | 否 | 無 |

「防禦圖產物」查的是 `main_table/images/defence_portraits/` 與主線目錄的
`runs/` 底下有無以這三個名字命名的目錄，三者皆無——**沒有產物就不可能有主表的
64 格**。至於當初為何未跑，`HANDOFF.md` §「不在主表裡的方法，以及為什麼」只列
`uap_semantic`、`tdae`、AdvCF、`colour_field` 四項，這三個不在其中；查過
`HANDOFF.md`、`BASELINE_ALIGNMENT.md`、本檔與 `AUDIT_*.md`，**未找到書面理由**，
依規則 5 不補一個看起來合理的說法。

`tdae` 的模組也在，且有審計文件；它不進主表的理由 `HANDOFF.md` 有寫
（威脅模型使損失恆為零、梯度精確為零）。

## 可引用的參考數字，以及它們的確切出處

| 數字 | 量的是什麼 | 出處 | 協定 |
|---|---|---|---|
| PhotoGuard PSNR 28.323、LPIPS 0.284 | 防禦圖 vs 原圖 | **DCT-Shield** Table 1 Noise Perception（[ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Bala_DCT-Shield_A_Robust_Frequency_Domain_Defense_against_Malicious_Image_Editing_ICCV_2025_paper.pdf) p.18881） | `L∞` 16/255、150 張 OmniEdit、InstructPix2Pix |
| PhotoGuard PSNR 33.79 | 防禦圖 vs 原圖 | **DIA** Figure 5（p.8） | PGD ε=0.05、60 iters、SD v1.4 |
| PhotoGuard PSNR 13.58±2.23 | **編輯結果** vs 未防禦的編輯結果 | PhotoGuard 論文 Table 6（p.7） | inpainting；與上兩列不是同一個量 |
| MIST PSNR 26.62、LPIPS 0.362 | 防禦圖 vs 原圖 | DCT-Shield Table 1 | 同第一列 |
| DCT-Shield PSNR 27.61、LPIPS 0.267 | 防禦圖 vs 原圖 | DCT-Shield Table 1 | 原生 ε=1 |

**PhotoGuard 論文本身不報防禦圖對原圖的 PSNR／LPIPS。** 全文 §4.2、
Appendix C.1、Tables 6／10 查無 0.284 與 28.32。

## 論文與官方程式的落差

| 方法 | 落差 | 後果 |
|---|---|---|
| **PhotoGuard** | 論文 Table 9 寫 `L∞` 16/255 ＋ step 2/255；官方啟用路徑是 L2 renorm `maxnorm=16` ＋ normalized grad ＋ step 1。**repo 中不存在同時滿足 Table 9 四欄的程式碼。** 官方 `super_linf` 的定義有啟用，但其呼叫在 cell 11 整段被註解，且該呼叫的 `eps=0.1`／`step=0.006` 也不等於 Table 9。 | 必須跑兩個臂：`photoguard_c`（官方路徑）與 `photoguard_linf`（Table 9 重建）。後者是**依表重建**，不是官方路徑的重現。 |
| **PhotoGuard** | L2 路徑的失真與影像內容無關。飽和時 `rms = 16/(2·√(3·512·512)) = 0.0090211`、`PSNR = 40.8948 dB`；本專案十張圖實測全距 0.002 dB。 | 在 `photoguard_c` 下 PSNR 是常數，**不是收斂不足**，加步數與 `grad_reps` 都無效。 |
| **Mist** | 程式名目上 `ε=16/255`；docstring 宣稱取整與裁切後實際上界是 `(16+1)/255`，但**程式中沒有任何一行實作 +1/255**。 | 要嚴格重現論文的 17/255 必須自行把 `-e` 設為 17，並標 `modified_from_paper`。 |
| **Mist** | `rand_init` 預設 `True` 且呼叫端未關閉，δ 有隨機初始化；論文正文未提。 | 同設定重跑不會得到同一張防禦圖，比較要帶批內對照。 |
| **AdvPaint** | `eps` 兩組預設互相矛盾：函式簽章 0.06、CLI 0.1。步數簽章 100、論文正文 250。 | 本專案取 0.06／0.03／100，理由見 `AUDIT_INPAINTING_METHODS.md` §1.4。 |
| **DIA** | 論文正文與補充資料均未給步長 α。 | 由 `attack_setting.json` 取 `lr=0.003921568…`（即 1/255）。 |
| **PromptFlare** | repo 全無 `manual_seed`／`Generator`。 | 不可重現到位元；跨批對照作廢。 |
| **DCT-Shield** | [官方 repo 是空的](https://github.com/SamsungLabs/dct-shield)，project page 標 code coming soon。Table 1 的 baseline 分支（encoder 還是 complex）與 LPIPS backbone 未載；補充資料 C.1.5 只說用 `pyiqa`。 | 它報的 baseline 數字只能當 as reported，不可用來判定我們的實作對錯。 |

## 本專案的量測管線

- LPIPS 走 `piq.LPIPS()`，**與官方 `lpips(net='vgg')` 逐位相同**，不是 AlexNet。
  VGG/Alex 的比值逐條件由 1.32 變動到 17.98，與擾動結構高度相關，
  **不能用單一係數換算兩篇的數字**（`BASELINE_ALIGNMENT.md` §1.2）。
- rms 的定義是 `(d ** 2).mean().sqrt()`，對 RGB 全部元素平均（`src/metrics/suite.py`）。
  引用他人由 PSNR 反推的 rms 時，兩者的平均方式不一定相同。
