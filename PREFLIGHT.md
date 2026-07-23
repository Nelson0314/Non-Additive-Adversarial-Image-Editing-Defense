# PREFLIGHT.md — 部署前檢查報告

（`scripts/preflight_report.py` 產出；終端機輸出之存檔版本）

```text
==============================================================================
部署前檢查報告（preflight）
==============================================================================
生成時間: 2026-07-24 02:48   git commit: f02f45b
Python: 3.11.15   平台: win32/CPU（本地）→ 目標: TWCC V100 32GB

==============================================================================
[1] 假設裁定（四項，均已落實）
==============================================================================
  [PASS] 假設1 L_ref=encoder：stage0 兩基準皆算並列出（diffusion 取 5 張，
         --pg-diff-images），差距>10% 自動警告；TWCC_CHECKLIST 已加驗證項
  [PASS] 假設2 掃描鈕：SG/GC 獨立範圍（configs stage0_scan；GC 因部分 inversion
         用較小值域）；hybrid 掃 eps_a（其相似性約束＝APA 骨架 ℓ∞ ball，
         s/a 注入僅在 ball 內作用，故 eps_a 為實際約束鈕）；
         選中值落區間端點時自動警告「須擴展重掃」（已實測觸發）
  [PASS] 假設3 遮罩：規格入 config（edit.inpaint_mask: center_square, fraction 0.5），
         manifest 記錄、summary 標記「placeholder 遮罩不可與真實資料比較」，
         TWCC_CHECKLIST 加「真實資料到手後 inpaint 全部重跑」
  [PASS] 假設4 prompt_idx：維持（STRUCTURE §4.3 由指導者補記）

==============================================================================
[2] 靜態稽核
==============================================================================
--- 2.1 v4 修正落地 --------------------------------------------------------
  [PASS] advdiff 後置加法 z=z+s·g（非改 epsilon）
  [PASS] advdiff 無 σ²/√(1−ᾱ) 係數
  [PASS] advdiff z_T 更新 skip-gradient（全程 no_grad）
  [PASS] advdiff eps_latent 相對 L2 投影
  [PASS] apa-sg 軌跡梯度 skip ×scale_constant
  [PASS] apa-gc 部分 inversion（格點前 inversion_steps 步）
  [PASS] apa-gc reward 含 −mse_reg_weight·MSE
  [PASS] apa 式(12) 基底=最終生成 latent、x̂₀ detached
  [PASS] apa LoRA=peft、delete_adapters 還原
  [PASS] apa noise_offset 實作
  註: gc.sampling_steps 為文件性參數（程式以 inversion_steps 個格點區間
      上下對稱走訪，官方計數含邊界，語意相同，config 已註記）
--- 2.2 可移植性 ----------------------------------------------------------
  [PASS] 無 .cuda() 呼叫
  [PASS] 無 xformers 依賴
  [PASS] device 一律經 device.py
  [PASS] 程式碼無硬編碼真實模型名（僅 config）
  [PASS] 無硬編碼 dtype
  註: tiny→真實 SD 僅換 config 模型名；解析度/latent 尺寸均由模型 config 推得
--- 2.3 config 完整性 ------------------------------------------------------
  [PASS] configs 可載入
  [PASS] photoguard 參數齊備（SPEC §3.1–3.3）
  [PASS] advdiff 參數齊備（SPEC §4.3）
  [PASS] apa 參數齊備（SPEC §4.4，含 sg/gc/aug）
  [PASS] stage0 掃描範圍：SG/GC 獨立（假設 2）
  [PASS] placeholder 遮罩規格入 config（假設 3）
  [PASS] AdverseCleaner 64×BF+4×GF（論文核驗修正）
  [PASS] GrIDPure 論文預設 10×10 為主設定＋兩端點（v5）
  [PASS] 批次參數：protect_batch_size=1；edit_batch_size=1（v5 位元級判準實測未通過，啟用待裁定）
  [PASS] [待確認] 項目以 null 明確標記
  SPEC 未提而 config 有: eps_latent（SPEC 偽碼用、範本漏列，已註記）、
  grid_steps（官方 50 步格點）、stage0_scan/inpaint_mask/clip_model（preflight 新增）、
  seed_clip_threshold（seed 判準操作化）——均有來源註解
  命名差異: EOT 次數＝grad_reps（STRUCTURE 範本作 diffusion_eot，實作讀
  grad_reps；preflight 曾因此發現 apply_smoke 用錯鍵，已修正）
--- 2.4 測試覆蓋 ----------------------------------------------------------
  [PASS] 測試套件
         - 44 passed, 1 skipped, 7 warnings in 30.65s
  覆蓋矩陣（45 項；44 pass + 1 skip）:
    photoguard(6): PGD 兩範數/投影/衰減/EOT/尺度   sd_wrapper(4): 封裝/差分 img2img
    smoke(4): 全流程/seed 重現/inpaint             nonadditive(10): reward 梯度與方向/
      inversion 決定性/advdiff/apa sg+gc 參數化+peft 還原/hybrid/
      guidance 區間/投影生效/ℓ∞ clamp
    purify(13): 輕量三法/AdvClean 兩變體/grid 機制/分派
    metric_directions(2): 極端案例方向（piq 五項+FID）
    edit_batching(6, v5): 噪聲協定位元級一致/批次決定性/sdedit+inpaint 容差等價/
      分塊等價/位元級啟用閘門（batching 關閉時 skip）
  未覆蓋（記入風險清單）: inversion 重建品質（tiny 無意義→TWCC）、CLIP 方向（需
    外部模型→TWCC 抽查）、批次化真實模型等價（→TWCC）、stage 腳本本身
    （以煙霧串跑驗證，非單元測試）
--- 2.5 穩健性 ------------------------------------------------------------
  [PASS] stage1/2 逐筆寫入（IncrementalCsv）
  [PASS] stage1/2 斷點續跑（--resume）
  [PASS] drop 除零/負基準防護
  [WARN] config 回寫＝單行 regex＋overlay（不重寫整份 yaml）
         - 決策：不引入 ruamel.yaml——similarity_budget 為單行且有錨點、regex 已於副本驗證（註解保留、可解析）；其餘校準值寫獨立 overlay 檔，完全避開 yaml 重寫。若日後回寫欄位增多，屆時再改 ruamel.yaml
  註: FID 需完整樣本——續跑跳過任何列時自動略過 FID 並警告（結構限制，
      如需 FID 對該組完整重跑；已於 stage1 註明）

==============================================================================
[2b] 論文核驗（詳見 PAPER_VERIFICATION.md）
==============================================================================
  高等級問題: 0
  中等級: 2（均已處置）
    - AdverseCleaner 官方為 64×BF+4×GF，SPEC 記 3×BF+1×GF → 已修正實作+config
    - DAYN Table 1 僅為 img2img 情境（inpaint 為質性）→ 校準限 sdedit 列，已改 summary
  其他發現: DAYN 測試集為 SD 生成影像（索取未果可自生成，已入 TWCC_CHECKLIST）；
    DAYN Alg.1 為 sign+clip（ℓ∞）→ norm=linf 先驗提高；「LDM 淨化無效」出處
    實為 Pixel is a Barrier §6.3（SPEC 誤置於 GrIDPure，待 SPEC 下版修正）；
    GrIDPure 論文預設 10×10（README 10×20、腳本 100×1，三來源已註記）；
    AdvDiff s=10/a=10 之失敗案例＝調參上界護欄；
    逐字驗證一致: DAYN Table 1 全 30 值、κ/N/T、式(2)-(5)；PG 式(4)(5)/Table 8,9/
    A.1 seed 協定；AdvDiff 式(9)(11)/附錄 E (0,0.2]/附錄 H untargeted/75.2%；
    APA 式(4),(6)-(12)/reward hacking 動機/SG=ρ·∇ 近似；GrIDPure γ、grid、2min

==============================================================================
[3] 真實模型風險（TWCC 首次執行必檢，詳見 TWCC_CHECKLIST.md）
==============================================================================
  1. cross-attention 擷取     方法: 兩模型跑 capture_cross_attention，驗層數>0、
     解析度合理  預期: v1.4/v2.0 皆可用  不符: 依 diffusers 版本更新 processor 類名
  2. DDIM inversion 重建品質  方法: 5 張 10/50 步 inversion→去噪，量 PSNR/LPIPS
     預期: PSNR>25、LPIPS<0.1  不符: 「錨定原圖」前提受損→部分 inversion 或加步數
  3. scheduler 混用（SPEC §8 第7項）方法: 同 5 張 DDIM(匹配) vs PNDM 重建比較
     預期: 匹配 DDIM 較佳或相近  不符: 維持匹配並記錄；官方混用列消融
  4. GPU 記憶體峰值           方法: 各法單張 protect 記 peak_memory_mb
     預期: pg_diff 最重（CPU 相對量 711/361 MB）  不符(OOM): 見 [6] 降級路徑
  5. fp16                     方法: 編輯 fp16 vs fp32 指標差 <1% 即採用
     預期: 編輯可用（無梯度）  不符: 編輯退回 fp32（成本×2，見 [4]）
  6. 編輯 pipeline eta=1      方法: 確認 img2img scheduler；PNDM 忽略 eta
     預期: T2 校準吸收此差異  不符: 改 DDIMScheduler(eta=1) 為第一排查項
  7. 假設1 驗證（enc vs diff LPIPS 差 ≤10%）  stage0 內建
  8. L_ref 用 placeholder 資料無效——所有校準以真實/自生成資料為準

==============================================================================
[4] 成本估算（單位假設見下；TWCC 首日以實測單位時間重算本表）
==============================================================================
  單位假設(V100): 編輯 22s fp32 / 11s fp16（100步+CFG）; 保護/張: pg_enc 1m,
  pg_diff 50m(±60%, EOT10×T10), advdiff 0.3m, apa/hybrid 3m; GrIDPure 2m/張/設定
  方案                  s0(h) protect(h)  s1 edits   s1(h)  s2 edits   s2(h)    總計(h)
  ----------------------------------------------------------------------------
  A 完整(fp32)             17        143   144,000    1063 2,040,000   13158    14238  <-- 超過 100h
  A'完整(fp16)             17        143   144,000     623 2,040,000    6925     7565  <-- 超過 100h
  G 閘門式(建議)              11         48    21,000     118    25,000     100      229  <-- 超過 100h
  C 首日最小                 11         19       600      21     2,500      12       43
  關鍵事實:
  - 乘數效應主宰成本: 完整方案 s2 編輯 2,040,000 次——瓶頸不是 GrIDPure（125h）
    也不是保護生成（143h），而是「淨化設定×方法×seed×模型×編輯」的編輯次數
  - 完整方案 A 約 14,200h（fp32）/ 7,600h（fp16）＝單卡不可行，須大幅縮減
  - pg_diff 保護為固定重成本（每張 50m，與 seed 無關）: 150 張≈125h、50 張≈42h、
    20 張≈17h——n_img 為最有效的單一槓桿（同時壓 protect 與所有編輯）
  - 儲存估算: stage1 edited_orig（G 方案 ~6,000 png）+ protected + purified ~數 GB
  加速手段評估（4.4）:
  1. 編輯批次化【原為最大槓桿，v5 實測後封鎖】: 已實作（每樣本各自 generator+
     seed，噪聲協定與 batch=1 位元級一致，seed 協定 SPEC §2.3 未破壞）。但 v5
     啟用判準「batch=1 vs batch=4 逐位元相同」實測**未通過**——批次矩陣運算之
     歸約順序隨 batch 尺寸而異，float32 差 ~5e-6（8-bit 存檔後 ~0.02% 像素 ±1LSB），
     位元級等價原理上不可達。依判準 edit_batch_size 維持 1。**待指導者裁定**：
     若放寬為「噪聲協定位元級一致＋輸出容差 ≤1e-4」則可啟用，編輯時數 ÷4–6
  2. fp16 編輯【可行，主要槓桿，已入 G/C】: 約 ÷2；風險[3]-5 驗證後啟用；保護維持 fp32
  3. 多卡平行【最實際之加速】: 保護與編輯對影像/seed 皆 embarrassingly parallel，
     k 卡約 ÷k（G 方案 229h → 4 卡 ~57h、8 卡 ~29h）
  4. CPU 淨化與 GPU 平行【可行，收益小】: jpeg/blur/crop/advclean 為 CPU 運算
     （advclean 64×BF 約 3–5s/張），與 GPU 編輯管線重疊——節省 <2h，優先度低
  5. GrIDPure 平行化【部分可行】: 10 個 grid 可 batch 成一次前傳（256×256×10
     於 32GB 可容納）→ 約 ÷3–5；或多卡分圖平行（原文亦建議）

==============================================================================
[5] 規模方案對照（v5 閘門式循序）
==============================================================================
  閘門式原則: 保真度花在有錨點的條件（V1.4×img2img，可對 DAYN Table 1 校準）；
  縮減集中在 stage2（其結論為「drop 之相對差異」，不需 stage1 的絕對精度）。
  各階段設閘門，前一階段不通過即停，避免把預算投入注定失敗的下游。

  方案 A 完整（依 SPEC；記帳基準）
    時數: ~14,200h(fp32) / ~7,600h(fp16) — 單卡不可行；即使 8 卡亦 ~40 天
    僅作為完整規格之成本上界，不執行
  方案 G 閘門式（建議主實驗）
    stage1: 錨點條件 V1.4×img2img 跑滿（50 張×2 prompt×20 seed×5 方法）；
      補充條件 V2.0×img2img、V1.4/V2.0×inpaint 各降至 5 seed
    stage2: 積極縮減——僅 V1.4×img2img、5 seed、10 淨化設定（gridpure 2）
    時數: ~229h(fp16 單卡) ≈ 10 天；4 卡 ~57h、8 卡 ~29h（編輯/保護可平行）
      拆解: 保護 ~48h（pg_diff 42h 主宰）、stage1 編輯 ~70h、stage2 編輯 ~83h、
      GrIDPure ~17h、stage0 ~11h
    能答: 核心假設（加性 vs 非加性 drop 差異+強度曲線，於主條件）、T2 校準
      （sdedit 有錨點）、非加性在有錨點主條件之乾淨比較（跑滿 seed，統計強）、
      跨模型/inpaint 之乾淨情況（stage1 補充條件保留，5 seed）
    不能答: 淨化後之跨模型/inpaint 行為（stage2 限單條件）；補充條件之 seed 統計
      較弱（5 vs 20）；stage2 影像仍 50 張
  方案 C 首日最小（20 張/1 prompt/5 seed/V1.4/img2img/5 淨化設定/fp16）
    時數: ~43h(fp16) ≈ 2 天（其中保護 ~19h、pg_diff 17h 主宰）
    能答: 方向性初判（drop 加性 vs 非加性）、pipeline 於真實 SD 跑通、
      單位耗時實測（校正本表所有時數）
    不能答: 校準有效性（樣本過少）、曲線形狀、任何可寫入論文之定量結論
  建議路徑: C（首日，兼量測單位耗時並過 T2 閘門）→ 據實測修訂 G 之規模與時數
    → G 為主實驗；A 僅在多卡多日資源到位、或以 G 結果決定局部加密時考慮
  具體切法要點:
    - n_img=50: pg_diff 保護由 125h 降至 42h，同時壓所有編輯；50 張×2 prompt
      ×20 seed=2000 樣本/方法，對主條件統計仍充分
    - 若指導者要求錨點條件用滿 150 張: 額外 +~95h（多為 pg_diff 保護），列為旋鈕
    - stage2 ops 由 17 降至 10（保留 jpeg 3/blur 2/crop 2/advclean_bfgf/gridpure 2）
      仍覆蓋強度曲線與 GrIDPure 迭代 vs 單次深淨化之對照

==============================================================================
[6] 決策樹
==============================================================================
  T2 校準對不上 DAYN Table 1（v5 分階段搜尋，norm 固定 linf）:
    1) 檢查編輯 scheduler/eta（風險[3]-6，改 DDIMScheduler 重跑一組）
    2) sdedit_strength 未知為最大混淆——依 {0.8,0.7,0.5,0.3} 分階段掃（先 0.8）
    3) 兩 epsilon_scale × 全部 strength 皆不符 → 試 norm=l2、再擴 target_latent=gray
    4) 仍不符 → 停下回報；選項: 向作者確認 / 改以「本專案自身 PhotoGuard 復現值」
       為錨（於論文中揭露不與 DAYN 直接比較）
  stage1 非加性全面劣於 PhotoGuard（stage1→stage2 閘門）:
    回報並重新評估是否值得跑 stage2——核心假設是「淨化後的相對優勢」（交叉現象），
    乾淨情況劣勢不否證之（STRUCTURE §4.5 已載）；若決定仍跑，降為方案 C 規模先探；
    若 stage2 亦全面劣勢 → 如實報告負結果，重心轉向分析（reward 定義/相似性預算消融）
  記憶體不足（OOM）:
    編輯: fp16 → attention slicing → 步數降 50（記錄偏離）
    pg_diff: grad_reps 10→5（記錄）→ diffusion_T 10→5（最後手段，偏離 DAYN）
    apa-gc: checkpoint 已用 → sampling 格點 50→25
    advdiff/apa-sg: skip-gradient 本就最輕，預期無 OOM
  GrIDPure 時數失控: 先跑 paper 10×10 單設定完成核心比較，掃描點與 single_deep 後補
  DAYN 資料索取未果: 依論文 §4.3 以 SD V1.4 自生成 150 張（3 類×2 prompt），
    結果標記「自建資料集」並於論文揭露（僅能偵測粗差，非同條件比較）

==============================================================================
[7] 待辦（依優先序；* = 阻塞後續項）
==============================================================================
  1.* 寄信 DAYN 作者: 測試集+sdedit strength+κ 尺度/範數/target（SPEC §8 1-5）
      （不阻塞部署，但阻塞 T2 定案；等待期間以 strength 掃描+自生成資料推進）
  2.* TWCC 環境建置+下載清單（SD×2、GrIDPure ckpt 2GB）＝ TWCC_CHECKLIST
  3.* 首日: 方案 C 執行＋[3] 風險清單全項檢查＋單位耗時實測（校正 [4] 表）
  4.  T2 校準（v5 分階段: norm=linf 固定，strength {0.8,0.7,0.5,0.3} 依序，共 7 次）
  5.  stage0 正式校準（真實資料；端點警告則擴範圍重掃）
  6.  方案 G 閘門式: stage1（錨點跑滿 / 補充降 seed）→ 閘門 → stage2（積極縮減）
  7.  （資料到手後）inpaint 以真實遮罩重跑；（資源允許）局部加密至 20 seed
  8.  （待裁定）批次化: 若放寬位元級判準則啟用，編輯時數 ÷4-6

==============================================================================
總結: 【可部署——v5 決策已落實，一項技術裁定待回覆】
==============================================================================
  靜態稽核與論文核驗無 FAIL 未決項；測試全綠（含批次化等價性 6 項）；三腳本
  煙霧串跑+續跑驗證通過。v5 三項決策均已落地:
  - 規模: 閘門式循序（首日 C → 主實驗 G，見 [5]）；完整 A 單卡不可行（~7,600h fp16）
  - 資料: 自生成 150 張備援已入 TWCC_CHECKLIST，結果標記「非同條件比較」
  - strength: v5 分階段搜尋已入 TWCC_CHECKLIST 與 [6]
  待指導者裁定（不阻塞首日 C，阻塞是否啟用批次化）:
  - 批次化位元級判準實測未通過（float ~5e-6；噪聲協定位元級一致、seed 協定未破壞）。
    依 v5 判準維持 edit_batch_size=1。若放寬為「噪聲協定位元級一致＋輸出容差
    ≤1e-4」則可啟用並將編輯時數 ÷4-6（G 方案 229h → ~110h）
  （本報告由 scripts/preflight_report.py 生成；單位耗時於 TWCC 首日實測後重跑本
   腳本更新 [4][5] 數字）
```
