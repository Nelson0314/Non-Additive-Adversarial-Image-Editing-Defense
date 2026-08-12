# 裁決

每一筆自足：決定了什麼、依據、影響哪些東西。

---

## DEC-021 · 原生 APA 重現：換 reward 之後把注意力項正規化回 cross-entropy 的量級

- **背景**：使用者 2026-08-12 要求「連階段二也完全照 APA 原生做法跑一次」，
  變因為階段一（原生 LoRA vs 本專案 z*/decoder）× 架構（DDIM vs BDIA），
  階段二一律走 dual-path + L∞ 球投影，**唯一替換是 reward**
  （分類器 cross-entropy → Lo et al. 式(5) 的注意力抑制損失）
- **問題**：官方 APA-GC 的 reward 是 `CE − 10·MSE(z_0, z̄_0)`，兩項同量級，
  那個 MSE 是有作用的保真煞車。注意力抑制損失是 16 層注意力圖在遮罩區內的
  L1 **總和**，量級差三個數量級——實測 `r_attn` 455–1588 對 `fid_pen`
  0.15–0.71，**比值 1000–2000 倍**，該煞車在換 reward 之後等於被關掉。
  症狀只在數值上：訓練跑得完、L∞ 球照樣綁住幅度，只是失真遠大於原文
- **決定**（使用者裁決）：把 `R_attn` 除以它自己在第 0 次迭代的絕對值，
  使該項由 −1 起步；`fidelity_lambda` **維持官方的 10.0 不動**。
  旗標 `NativeStage2Config.normalize_attn_reward`，預設開啟
- **為什麼不是直接放大 λ**：那樣要挑一個 6.6×10⁴ 量級的數字，沒有出處也
  無法對照回原文；正規化 reward 則讓官方的 λ=10 逐字保留，偏離只有一處
- **為什麼正規化常數取第 0 次迭代而非固定值**：`r_attn` 的絕對值隨影像的
  注意力質量而變（遮罩大小、主體佔比），固定常數會讓不同影像拿到不同的
  有效權重，而那個差異不會有症狀。常數整輪共用，不逐迭代重算——逐迭代
  重算會讓該項恆為 −1、梯度尺度被抹平
- **地位**：這是相對 APA 原文的**第三個有記錄偏離**（另兩個是 reward 本身
  的替換、以及不做 diffusion augmentation），論文須載明
- **同批一併修正的兩個實作偏離**（不是決策，是缺陷）：階段二的 CFG 由
  誤用的 7.5 改回官方的 1.0；反演由「完整反演到 z_T 再完整去噪」改為
  官方 APA-GC 的淺噪聲帶（50 格排程只執行前 11 格、T_a=10）。
  三批實測的 `fid_lpips`：0.51–0.82（原始）→ 0.50–0.71（修 CFG）→
  0.42–0.48（再修噪聲帶），APA 原文 Table 3 為 0.23–0.25

## DEC-022 · 編輯強度由 0.4 改為 0.55：未防禦的編輯必須真的成功

- **決定**（使用者 2026-08-12 裁決）：APA 三張圖的抗編輯評測，SDEdit
  `strength` 由 0.4 改為 **0.55**，prompt 不動
- **問題**：0.4 下 butterfly 的**未防禦**編輯與原圖幾乎無差別——攻擊沒有發生。
  防禦強弱是拿「防禦後的編輯」對「未防禦的編輯」比的，分母不成立時整組
  抗編輯數字都沒有意義。與 `data/lo_aligned/prompts.yaml` 記過的失效模式相同
- **依據**：編輯強度 × prompt 的掃描（腳本已隨 DEC-023 精簡移除，
  取回 `git checkout a4f93451f -- scripts/apa_native_edit_sweep.py`）
  （`runs/apa_edit_sweep/`，判準為人眼）。0.55 下 butterfly 出現明顯的
  帝王蝶＋紅玫瑰、coot 變成白天鵝，且**構圖仍認得出是同一張照片**；
  0.7 則構圖整個被換掉，「編輯成功與否」不再能歸因到防禦
- **可查證的連帶**：未防禦編輯的 CLIP 對齊由 0.268→0.310（butterfly）、
  0.301→0.323（coot）
- **範圍**：只作用於 `data/apa_native` 這批的評測，不改動主線批次

## DEC-023 · 新主線與內部弱 baseline 的定義

- **決定**（使用者 2026-08-12 裁決）：以「完全原生 APA、只把 reward 換成
  targeted output」作為**內部弱 baseline**，並以此為新主線的起點。其餘方法
  待尋找
- **弱 baseline 的完整設定**（`scripts/apa_baseline.py` 的 `apa_weak`）：

  | 位置 | 設定 | 出處 |
  |---|---|---|
  | 階段一 | APA 官方 LoRA，Eq.6 denoising MSE、AdamW 1e-4、200 步固定、rank=8、noise_offset=0.1 | 官方 `visual_alignment.py` |
  | 階段二 | dual-path attack guidance（trajectory + step-level） | 官方 `pipe_ours.py` |
  | 約束 | latent L∞ 球 ε_a=0.4 | 官方 Eq.7 |
  | 更新 | L1 正規化動量 + sign + µ=0.04、N=10 | 官方 Eq.7 |
  | 反演 | DDIM，50 格排程只執行前 11 格、T_a=10、CFG=1 | 官方 APA-GC |
  | **reward** | **`−‖D(z̄_0) − y_target‖²`（唯一的替換）** | PhotoGuard-c／Mist 形式 |

- **為什麼是這一組**：原文的分類器 CE 在抗編輯場景不可執行（沒有分類器），
  且已實測與抗文字編輯無交集（FND-030）。targeted 是本專案量到位移量最高的
  非加性形式（FND-029），而其餘四個位置全部維持原生，使它是一個**定義清楚、
  可被超越**的參考點
- **為什麼叫「弱」**：它的語意抵抗與其他所有條件一樣接近零（FND-029）。
  它是位置基準，不是有效的防禦
- **連帶的程式精簡**：已測過並否決的變體（注意力抑制／分類器 CE／latent／
  CLIP 四種 reward、DISTS 進 loss 的軟約束、Adam 更新規則）自
  `apa_native_stage2.py` 移除，結論留在 FND-027…030。
  取回：`git checkout a4f93451f -- src/defense/apa_native_stage2.py`。
  本 session 的四支一次性腳本（階段一比較、DDIM 對照、編輯強度掃描、
  apa_pj 評測橋接）同批移除，取回用同一個 commit
- **舊主線降級**：與新主線無關的 FND／DEC／EXP 移入 `docs/archive/LEGACY_*.md`，
  屬於次要紀錄、不是判準來源。**工程與量測的教訓不降級**——硬體與程式犯過的
  錯留在 `DEFECTS.md`、`RUNBOOK.md` 與主線 `FINDINGS.md`（FND-015 分片重複、
  FND-026 DISTS 降採樣）
