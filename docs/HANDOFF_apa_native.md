# 交接單：APA 原生重現（2026-08-11/12）

自足，不必讀對話。判準與結論以 `FINDINGS.md`／`DECISIONS.md` 為準，
本頁只負責「這一輪做了什麼、結論是什麼、下一輪可以從哪裡接」。

## 這一輪回答的問題

指導者的三個提問：為何階段一沒用 APA 原生方法、LoRA 去哪了、為何階段二的
reward 換成自己的 loss。做法是把兩個階段都照原文重現一次再比較。

## 四個結論（各自的證據見編碼）

| 結論 | 編碼 |
|---|---|
| 原生階段一（LoRA）在 BDIA 管線上**不只無效而是有害**；DDIM 對照證明重現無誤 | `FND-027` |
| APA 的 latent L∞ 球**從未生效**（`ε_a` 恰等於 `µ×N`），且同一半徑在不同影像上是不同的可見失真，三個指標的排序都與人眼相反 | `FND-028` |
| 階段二四個軸：**reward 決定位移量、更新規則決定保真度**，但沒有任何一個軸讓語意抵抗成立（22 個非加性條件的 CLIP 掉幅都在 ±0.03 內） | `FND-029` |
| 原文自己的分類器 reward 與抗文字編輯**沒有交集** | `FND-030` |

## 重現原文時修掉的三處實作偏離

三者都不是設計選擇而是缺陷，記在 `DEC-021` 與 `apa_native_stage2.py` 的
docstring（含 before/after 與實測數字）：

1. 階段二 CFG 誤用評測期的 7.5，官方是 **1.0**
2. 反演做成「完整反演到 z_T 再完整去噪」，官方 APA-GC 是**淺噪聲帶**
   （50 格排程只執行前 11 格、T_a=10）——這才是它保住保真度的機制
3. 換 reward 後注意力項與官方 λ=10 的保真項差 1000–2000 倍，等於關掉了
   原文內建的保真煞車；已正規化回同量級

`fid_lpips` 三批演進：0.51–0.82 → 0.42–0.48 → **0.23–0.34**
（APA 原文 Table 3 為 0.23–0.25）。

## 程式（新增的最小集合）

| 檔案 | 用途 |
|---|---|
| `src/defense/apa_native_stage2.py` | APA-GC 階段二：dual-path guidance、latent 球、sign/Adam、三種 reward |
| `src/defense/optimize.py::align_apa_native` | 官方階段一目標（Eq.6 denoising MSE） |
| `src/metrics/aesthetic.py` | NIMA／CNNIQA／CLIP 影像-影像（對齊 APA Table 3 的欄位） |
| `scripts/apa_native_full_pipeline.py` | 主驅動，`--images`／`--conditions`／`--lam` 可分片 |
| `scripts/apa_native_lora_probe.py`／`_ddim_control.py` | 階段一比較與其 DDIM 對照 |
| `scripts/apa_native_edit_sweep.py` | 編輯強度掃描（DEC-022 的依據） |
| `scripts/apa_pj_evaluate.py` | 把 `run_stage.py` 訓出的 apa_pj 接進同一個 `evaluate()` |

資料：`data/apa_native/`（APA 官方三張圖 + `provenance.json` + `prompts.yaml`）。

## 已知的限制，下一輪要注意

- **n=2 或 3 影像、單一種子**，量級不宜外推；質性結論（主體被消滅 vs 紋理變粗）
  不依賴樣本數
- **階段一的比較是協定對協定**：我方是「達標即停」（A2 在第 0–60 步就停），
  原生 LoRA 是官方的固定 200 步且末端仍在震盪。不是同等訓練預算下的比較
- `get_ori_latents` 丟掉了 `finetune_decoder` 的回傳值，故 `recon_a2_reached`／
  `stop_step` **沒有進 CSV**（本輪是從 log 撈的）。要引用該數字的話先補落盤
- 加性 baseline 不經 VAE 來回，其 39–41 dB 有相當部分來自結構差異，
  **不宜與走生成路徑的格直接比保真度絕對值**

## 下一輪可以從哪裡接

依 `FND-029`，「把編輯推遠」做得到（targeted 形式最強），「讓編輯不服從
prompt」四個軸都試過仍做不到。與 `FND-024`（注意力抑制與編輯失敗無因果關係）
合起來看，還沒被否證的方向是**換一個與評測對齊的目標函數**，而不是繼續調
約束、更新規則或階段一——後三者本輪已各自試過。
