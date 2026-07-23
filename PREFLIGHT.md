# PREFLIGHT.md — 部署前檢查報告

（`scripts/preflight_report.py` 產出；終端機輸出之存檔版本）

```text
==============================================================================
部署前檢查報告（preflight）
==============================================================================
生成時間: 2026-07-23 19:04   git commit: fb99842
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
  [PASS] [待確認] 項目以 null 明確標記
  SPEC 未提而 config 有: eps_latent（SPEC 偽碼用、範本漏列，已註記）、
  grid_steps（官方 50 步格點）、stage0_scan/inpaint_mask/clip_model（preflight 新增）、
  seed_clip_threshold（seed 判準操作化）——均有來源註解
  命名差異: EOT 次數＝grad_reps（STRUCTURE 範本作 diffusion_eot，實作讀
  grad_reps；preflight 曾因此發現 apply_smoke 用錯鍵，已修正）
--- 2.4 測試覆蓋 ----------------------------------------------------------
  [PASS] 測試套件
         - 39 passed, 6 warnings in 21.90s
  覆蓋矩陣（39 項）:
    photoguard(6): PGD 兩範數/投影/衰減/EOT/尺度   sd_wrapper(4): 封裝/差分 img2img
    smoke(4): 全流程/seed 重現/inpaint             nonadditive(10): reward 梯度與方向/
      inversion 決定性/advdiff/apa sg+gc 參數化+peft 還原/hybrid/
      guidance 區間(新)/投影生效(新)/ℓ∞ clamp(新)
    purify(13): 輕量三法/AdvClean 兩變體/grid 機制/分派
    metric_directions(2): 極端案例方向(新，piq 五項+FID)
  未覆蓋（記入風險清單）: inversion 重建品質（tiny 無意義→TWCC）、CLIP 方向（需
    外部模型→TWCC 抽查）、stage 腳本本身（以煙霧串跑驗證，非單元測試）
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
  A 完整（依 SPEC）           19        143   144,000    1063 2,040,000   13158    14241  <-- 超過 100h
  A'（A + fp16 編輯）        19        143   144,000     623 2,040,000    6925     7567  <-- 超過 100h
  B 縮減                   17         48    12,000      88    25,000     108      213  <-- 超過 100h
  C 最小可行                 11         19       600      21     2,500      12       43
  關鍵事實:
  - 乘數效應主宰成本: 完整方案 s2 編輯 2,040,000 次——瓶頸不是 GrIDPure（125h）
    也不是保護生成（143h），而是「淨化設定×方法×seed×模型×編輯」的編輯次數
  - 完整方案 A 約 13,000+ h（fp32）/ 6,800 h（fp16）＝單卡不可行，須大幅縮減
  - pg_diff 保護為第二重（150 張×50m≈125h），且為校準錨點不可省——可縮影像數
  - 儲存估算: stage1 edited_orig ~24,000 png（~12GB）+ protected + purified ~6GB
  加速手段評估（4.4）:
  1. 編輯批次化【最大槓桿，可行】: 編輯無梯度，diffusers 支援 batch+generator
     清單（每樣本獨立 seed，位元級重現性不變）。batch=8 於 V100 約 4–6× 吞吐
     → s1/s2 編輯時數 ÷4–6。需改 edit_image 支援批次（約半天工作量）＋
     32GB 下 batch 上限實測。註: SPEC §1.4 batch_size=1 係針對「保護生成」之
     記憶體約束，編輯批次化不牴觸，但採用前須指導者確認
  2. fp16 編輯【可行，已入方案 B/C】: 約 ÷2；風險[3]-5 驗證後啟用；保護維持 fp32
  3. CPU 淨化與 GPU 平行【可行，收益小】: jpeg/blur/crop/advclean 為 CPU 運算
     （advclean 64×BF 約 3–5s/張），可與 GPU 編輯管線重疊——節省 <2h，優先度低
  4. GrIDPure 平行化【部分可行】: 10 個 grid 可 batch 成一次前傳（256×256×10
     於 32GB 可容納）→ 約 ÷3–5；或多卡分圖平行（原文亦建議）。
     縮減方案下 GrIDPure 僅 25–75h，先靠 grid batch 即可

==============================================================================
[5] 規模方案對照
==============================================================================
  方案 A 完整（依 SPEC；150 張/2 prompt/20 seed/2 模型/2 編輯/17 淨化設定）
    時數: ~6,800h(fp16) — 單卡 V100 不可行；即使 8 卡平行亦 ~35 天
    能答: SPEC 全部研究問題（完整校準、跨模型、雙編輯、完整強度曲線）
    不能答: 無——但實務上不可執行，僅作為記帳基準
  方案 B 縮減（建議）（50 張/2 prompt/5 seed/s1 雙模型雙編輯/s2 限 V1.4+sdedit/
    10 淨化設定（gridpure 3）/fp16 編輯）
    時數: ~190h ≈ 8 天（單卡）；配合編輯批次化（見[4.4]）可壓至 ~70-90h
    能答: 核心假設（加性 vs 非加性之 drop 差異+強度曲線）、T2 校準（sdedit）、
      跨模型遷移與 inpaint 之乾淨情況（stage1 保留雙模型雙編輯）
    不能答: 淨化後的跨模型/inpaint 行為（s2 縮至單模型單編輯）；seed 統計較弱
      （5 vs 20，均值標準誤約 2 倍）；影像數 1/3（類別內樣本 ~17 張/類）
  方案 C 最小可行（20 張/1 prompt/5 seed/V1.4/sdedit/5 淨化設定（gridpure 1）/fp16）
    時數: ~30h ≈ 1.5 天
    能答: 核心假設之方向性初判（drop 加性 vs 非加性）、pipeline 在真實 SD 上
      跑通、單位耗時實測（校正本表）
    不能答: 校準有效性（樣本過少，不足以對 DAYN Table 1 下結論）、曲線形狀、
      任何可寫入論文之定量結論
  建議路徑: C（首日，兼量測單位耗時）→ 據實測修訂 B 之規模 → B 為主實驗；
    A 僅在多卡/多日資源到位時考慮，或以 B 結果決定局部加密（如僅對勝出方法
    補跑 20 seed）

==============================================================================
[6] 決策樹
==============================================================================
  T2 校準四組合（epsilon_scale×norm）皆對不上 DAYN Table 1:
    1) 檢查編輯 scheduler/eta（風險[3]-6，改 DDIMScheduler 重跑一組）
    2) sdedit_strength 未知為最大混淆——掃 {0.3,0.5,0.7,0.9} 找最近組合
    3) 仍不符 → 擴至 8 組合（+target_latent=gray）
    4) 仍不符 → 停下回報；選項: 向作者確認 / 改以「本專案自身 PhotoGuard 復現值」
       為錨（於論文中揭露不與 DAYN 直接比較）
  stage1 非加性全面劣於 PhotoGuard:
    仍跑 stage2，但降為方案 C 規模——核心假設是「淨化後的相對優勢」（交叉現象），
    乾淨情況劣勢不否證之（STRUCTURE §4.5 已載）；若 stage2 亦全面劣勢 → 如實報告
    負結果，重心轉向分析（reward 定義/相似性預算之消融）
  記憶體不足（OOM）:
    編輯: fp16 → attention slicing → 步數降 50（記錄偏離）
    pg_diff: grad_reps 10→5（記錄）→ diffusion_T 10→5（最後手段，偏離 DAYN）
    apa-gc: checkpoint 已用 → sampling 格點 50→25
    advdiff/apa-sg: skip-gradient 本就最輕，預期無 OOM
  GrIDPure 時數失控: 先跑 iterative 10×20 單設定完成核心比較，掃描點後補
  DAYN 資料索取未果: 依論文 §4.3 以 SD V1.4 自生成 150 張（3 類×2 prompt），
    結果標記「自建資料集」並於論文揭露

==============================================================================
[7] 待辦（依優先序；* = 阻塞後續項）
==============================================================================
  1.* 寄信 DAYN 作者: 測試集+sdedit strength+κ 尺度/範數/target（SPEC §8 1-5）
      （不阻塞部署，但阻塞 T2 定案；等待期間以 strength 掃描+自生成資料推進）
  2.* TWCC 環境建置+下載清單（SD×2、GrIDPure ckpt 2GB）＝ TWCC_CHECKLIST
  3.* 首日: 方案 C 執行＋[3] 風險清單全項檢查＋單位耗時實測（校正 [4] 表）
  4.  T2 校準（sdedit 列 vs DAYN Table 1；四組合×strength 掃描）
  5.  stage0 正式校準（真實資料；端點警告則擴範圍重掃）
  6.  方案 B: stage1 → stage2
  7.  （資料到手後）inpaint 以真實遮罩重跑；（資源允許）局部加密至 20 seed

==============================================================================
總結: 【可部署——附三個前置條件】
==============================================================================
  靜態稽核與論文核驗無 FAIL 未決項；測試全綠；三腳本煙霧串跑+續跑驗證通過。
  部署前必須解決/決定:
  1. 規模方案拍板（建議: 首日 C → 主實驗 B）——完整 A 單卡不可行（>6,800h）
  2. 資料策略: 等作者回覆之平行方案（自生成 150 張）需指導者同意
  3. sdedit_strength 屬 [待確認]——T2 以 strength 掃描處理之作法需指導者同意
  （本報告由 scripts/preflight_report.py 生成；單位耗時於 TWCC 首日實測後重跑本
   腳本更新 [4][5] 數字）
```
