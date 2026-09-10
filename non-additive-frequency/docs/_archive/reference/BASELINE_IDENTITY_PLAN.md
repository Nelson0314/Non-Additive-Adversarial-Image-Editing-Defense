# 身分軸上的 baseline：FaceLock、PhotoGuard、DCT-Shield

使用者裁定的三個對照組。本檔是**實驗設計**，不是結果——測得的事實寫進
`runs/ip2p_baseline_identity/README.md`。逐篇的指標查證見
[`SURVEY_IDENTITY_EDITING.md`](SURVEY_IDENTITY_EDITING.md)。

## 為什麼是這三個

| 方法 | 攻擊模型 | 保護對象 | 本專案的實作狀態 |
|---|---|---|---|
| **FaceLock**（CVPR 2025） | **InstructPix2Pix** | **臉的身分** | 未實作，有官方程式碼可逐行對照 |
| **PhotoGuard**（ICML 2023） | img2img ＋ inpainting | 通用 | 已實作，但**只在 SDEdit 線上** |
| **DCT-Shield**（ICCV 2025） | InstructPix2Pix | 通用 | **IP2P 線上已可跑**（`--conditions dct_shield`） |

FaceLock 的攻擊模型與保護對象與本專案逐項相同，三個推論參數也相同
（`steps=100`、`s_I=1.5`、`s_T=7.5`），是這張表的主要對照。

---

## 1. 三個方法各要做什麼

### 1.1 DCT-Shield —— 已可跑，不需新程式

`scripts/ip2p_run.py --conditions dct_shield` 與 `dct_shield_y` 兩個變體都已
接在 IP2P 線上（`DCT_CONDS`）。定案超參數 `Q_alg=0.95`、`ε=1`、`γ=0.1`、
`N=1000`，損失是它自己的 `‖E(x')‖₂`。

**Y-only 變體不可省。** `docs/BASELINES.md` 記過：它在 JPEG 與 GrIDPure 上
遠強於 base 變體，頭對頭表上只放 base 是不誠實的。

### 1.2 PhotoGuard —— 要換變體，理由不是偷懶

本專案已實作的是 **complex attack（`photoguard_c`）**，它整條綁在
`SDWrapper.sdedit` 的取樣式上（`src/baselines/photoguard.py` 的
`attack_forward` 由我方改寫成 img2img）。要接到 IP2P 線得再改寫一次取樣式
——那是第二次「摘要重建」，可信度更低。

**改用 encoder attack。** 三個理由：

1. **FaceLock 比較的就是它**：論文寫 "PhotoGuard (encoder attack targeting
   VAE latent space)"，官方 `methods.py::encoder_attack` 即該支。
2. **它與攻擊方的取樣式無關**，只打 VAE 編碼器，換攻擊模型時逐行不動。
3. **它是 PhotoGuard 論文本身的兩個變體之一**，不是我方發明的。

移植逐行對照 FaceLock 的 `encoder_attack(targeted=False)`：

```
X_adv ← clamp(X + U(−eps, eps))
每一步 i：
    α_i  ← α − (α − α/100)/N · i          # 步長線性衰減到 α/100
    loss ← MSE(E(X_adv), E(X))            # untargeted：推離乾淨 latent
    X_adv ← X_adv + α_i · sign(∇loss)
    X_adv ← clamp(投影回 L∞ 球, −1, 1)
```

**`eps=0.03`、`step_size=0.01`、`iters=100`** 是 FaceLock README 的預設值，
非 PhotoGuard 論文的原生預算——這一點必須進 CSV 欄位並標
`modified_from_paper`。

### 1.3 FaceLock —— 新寫，逐行對照官方 `methods.py::facelock`

```
X_adv ← clamp(X + U(−eps, eps))
clean_latent ← vae.encode(X).latent_dist.mean
每一步 i：
    α_i    ← α − (α − α/100)/N · i
    latent ← vae.encode(X_adv).latent_dist.mean
    image  ← vae.decode(latent).sample.clip(−1, 1)
    loss   ← − FR(image, X) · [i ≥ 0.35N]      # 壓低重建圖的身分
             + 0.2 · MSE(latent, clean_latent)  # EditShield 的 latent 項
             + LPIPS(image, X) · [i > 0.25N]    # 壓低重建圖的感知相似度
    X_adv  ← X_adv + α_i · sign(∇loss)
    X_adv  ← clamp(投影回 L∞ 球, −1, 1)
```

**完全不碰 UNet**，只走 VAE 編碼與解碼，故單張成本是分鐘級而不是本方法的
50 分鐘。

四件必須寫進 CSV 欄位的事：

1. **兩個項有啟動排程**（FR 在 35% 之後、LPIPS 在 25% 之後才計入）。
   **論文正文沒有這一段，只在程式碼裡**，屬 `SOURCE_AUDIT` 意義下的
   「原始碼有、論文沒有」。
2. **預算論文與程式碼不一致**：論文正文 `eps=0.02`／`α=0.003`，官方程式碼
   預設 `eps=0.03`／`α=0.01`，兩者皆 100 步。**兩組都要跑**，逐列記下用的是
   哪一組。
3. **`FR` 用的是 CVLFace**：辨識器 `minchul/cvlface_adaface_vit_base_kprpe_webface4m`
   （AdaFace ViT-Base ＋ KPRPE，WebFace4M）、對齊器 `minchul/cvlface_DFA_mobilenet`。
   兩者在 HuggingFace 上公開、`trust_remote_code=True`，已確認遠端抓得到。
   **不可換成本專案既有的 facenet InceptionResnetV1**——那會變成一個消融，
   不是那一篇。
4. **`LPIPS` 用的是 `lpips.LPIPS(net='vgg')`**，與本專案的 `piq.LPIPS` 逐位
   相同（`BASELINE_ALIGNMENT.md` §1.2 已驗證），故這一項不需要另裝套件。

---

## 1.4 共用骨幹接得上，兩件事已查證

**`src/baselines/pgd.py` 的 `run_pgd` 可以直接餵 `IP2PWrapper`。** 它的第一個
參數雖然叫 `sd`，但骨幹**完全不碰它**——只原封不動傳給 `spec.prepare` 與
`spec.loss_fn`（全檔僅三處提及）。故只要新 spec 的那兩個函式只用
`encode_image`／`decode_latent`／`vae` 這種兩個 wrapper 都有的介面，
同一份程式在 SDEdit 線與 IP2P 線上都跑得動，不必分叉。

**步長排程已經有了。** FaceLock 與 PhotoGuard encoder attack 用的
`α_i = α − (α − α/100)/N · i` 就是骨幹既有的 `linear_decay_1pct`
（`step_size_at`，原本為 AdvPaint 而寫）。不需要新增排程。

**要新增的只有一項**：FaceLock 的兩個損失項有啟動排程（FR 在 35% 之後、
LPIPS 在 25% 之後才計入），而 `loss_fn(sd, x_adv, ctx)` 的簽名裡沒有步數。
兩個作法：把當前步數放進 `ctx`（骨幹每步更新一個欄位），或讓 `loss_fn`
接受可選的 `iteration`。前者不動簽名、不影響既有五篇，較安全。

## 2. 失真軸不對齊，怎麼比

三個 baseline 都是**全圖 L∞ 球內的加性擾動**，本方法是**衣物上的可見補丁**
（補丁內 `L∞ = 1.0`）。`BASELINE_ALIGNMENT.md` §2.1 的「統一像素預算」對本
方法**無定義**。

採 `BASELINE_ALIGNMENT.md` §2.3 的掃描曲線協定（DEC-029 已裁定的作法）：

| 條件 | 掃描參數 | 網格 |
|---|---|---|
| FaceLock | `eps` | 0.02（論文）、0.03（程式碼）、0.06、0.10 |
| PhotoGuard encoder | `eps` | 同上四點 |
| DCT-Shield base | `ε` | 0.4 / 0.6 / 0.8 / 1.0（1.0 是論文定案值） |
| DCT-Shield Y-only | `ε` | 同上 |
| 本方法（`free`／`tint`） | — | 單點，標在曲線上 |

- **橫軸（失真）**：DISTS 與 LPIPS **兩條都畫**。實測兩者對同一組影像的判定
  經常相反，只在單一軸上成立的結論不算數。
- **縱軸（效果）**：主讀數是身分（`id_def`，扣掉 `id_orig`），
  另報 FaceLock 的六個指標（`edit_lpips`／`edit_psnr`／`edit_ssim`／
  `edit_clip_sim`／CLIP-S／FR）。
- **兩個錨點**：等失真（比效果）與等效果（比失真），皆用
  `scripts/tradeoff_curve.py` 線性內插，**落在掃描範圍外一律拒絕外插**。

**很可能兩條曲線根本不重疊**——本方法的補丁 DISTS 落在 0.140–0.187，
而 `eps=0.02` 的全圖擾動遠低於此。若如此，`tradeoff_curve.py` 會回
`out_of_range`，而**那個結果本身就是要報的事實**：不是「本方法贏」也不是
「輸」，是「兩者不在同一個失真區間，單點比較無意義」。要把 baseline 推到
可比的失真需要把 `eps` 加到 0.06–0.10，那已遠離它們論文的設定，
每一點都要標 `modified_from_paper`。

---

## 3. 資料與協定

沿用內容軸那批，逐項不動：

- **影像**：`runs/ip2p_face_defence/clothing_plain_task_*` 的十張（由目錄反推，
  不用目錄檔——那裡登記了 20 張，只跑了 10 張）。
- **指令**：`data/attack_prompts.yaml` 的三類（clothing／accessory／background）。
- **編輯**：`steps=100`、`s_T=7.5`、`s_I=1.5`、`seed=20260812`。與 FaceLock
  的三個推論參數相同。
- **一張防禦圖接三次攻擊**：baseline 的損失都不吃攻擊指令，故三類解的是同一個
  最佳化問題，訓練一次、另外兩類 `--steps 0 --resume-weights` 重播。

## 4. 抗淨化

三個 baseline 都要跑與本方法**同一組十二個算子**
（`runs/ip2p_purify_identity/README.md`），否則抗淨化那張表只有本方法有數字。
驅動同一支 `scripts/purify_identity.py`，`--cells` 指到 baseline 的輸出目錄。

**不可引用它們論文報的抗淨化數字。** 三篇都沒有空白地板、沒有等失真的隨機
對照（`SURVEY_IDENTITY_EDITING.md` §6），與本專案的淨增益不可並列。

## 5. 成本

| 段 | 格數 | 估計 |
|---|---|---|
| FaceLock 4 個 eps × 10 張 | 40 | 防禦約 2 min/格（100 步、只走 VAE） |
| PhotoGuard encoder 4 × 10 | 40 | 同上 |
| DCT-Shield base ＋ Y-only 各 4 × 10 | 80 | 實測 150 s/格 |
| 三類編輯（訓練那類 ＋ 兩類重播） | 160 格 × 3 | 約 25 s/次編輯 |
| 抗淨化（12 算子 × 挑出的工作點） | 依錨點決定 | 每 job 約 59 min |

主表約 **9 GPU-hr**，五卡約一小時。抗淨化另計。

## 6. 這一段開工前要先確定的一件事

**PhotoGuard 換成 encoder attack 是變更既有的 baseline 定義**：
`runs/external_baselines_24img`、`runs/sdedit_mainline` 等既有批次裡的
`photoguard_c` 是 complex attack。兩者**不可混在同一欄**，報表上要寫成
`photoguard_e`（encoder）與 `photoguard_c`（complex）兩個不同的條件名。
