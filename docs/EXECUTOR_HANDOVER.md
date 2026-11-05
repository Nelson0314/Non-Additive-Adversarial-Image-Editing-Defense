# executor 交接

接手 executor（專案管理、文獻管理、派工與轉達、代為 push）的冷啟動文件。各實驗的即時狀態以 `baseline/STATUS.md`、`color/STATUS.md`、color trial `HANDOVER.md` 與 `docs/PAPER_HANDOVER.md`、`docs/PAPER_STRUCTURE.md` 為準。

## 1. 角色與規則

| 角色 | 職責 | 寫入範圍 |
|---|---|---|
| executor | 進度管理、文獻管理（`docs/paper/main.bib` 核實）、派工與轉達、整理成果給使用者、代各 session push | `docs/` 的管理文件、`main.bib` |
| baseline | 加性方法、BlurGuard、跨編輯器、主表讀數與彙整（含 CAPGuard 的 followup） | `baseline/` |
| color | CAPGuard（color3d 系列）求解與編輯、逐張讀數、消融 | `color/` |
| paper | 論文正文 `docs/paper/`、`docs/paper_tools/`、`docs/PAPER_STRUCTURE.md`、`docs/PAPER_HANDOVER.md` | 同左 |

- pane 編號會變，每次派工前 `list_panes`，依標題辨認。
- 排程一律由使用者手動下令：不設自動接續的工作、閘門或會自動改卡數的監看；一項完成後回報使用者，等指示再開下一項。
- 不因 session 長度停工：唯一換手條件是 context 用量 > 85%。
- push：各 session 回報 commit 後，確認 `git merge-base --is-ancestor origin/main main`（fast-forward）與 `git ls-files --eol` 皆為 `i/lf` 再 push；提醒對方遠端 pull 時不可覆寫執行中的腳本。
- 回報使用者：先講結果，數字附協定（資料、張數、設定），不下判定；進度用完成數／總數；本機不跑計算，看圖時把 PNG 用 `Start-Process` 開給使用者。
- GPU：全局 ≤ 6 張（兩台 RTX 3090 主機 basic-1／basic-2，NFS 共用；FLUX 權重只在 basic-2、UltraEdit 只在 basic-1）。baseline 以 `queue/baseline_cap`、color 以 `queue/color_reserve` 控卡（皆在 `baseline/runtime/pexels_portraits/queue/`）。

## 2. 方法與協定

- **CAPGuard**（Color-based Anti-Purification Image Immunization）：每張影像一個 33³ RGB 3D LUT，自由參數經色空間高斯平滑（σ＝4 格）成旋轉場；以黑點為原點、灰軸為極軸，極角 < 25°、色相 < 7.5°（2.5 倍上界，條件名 `color3d_scale2p5_pexels`），‖c‖ 不變；損失只有 ip2p VAE 編碼器目標；Adam 1800 步，約 9.4 分鐘／張（RTX 3090）。1.0 倍（10°／3°，`color3d_pexels`）不再出現在論文。
- **資料與攻擊**：Pexels 人像 306 張；InstructPix2Pix，4 句配件指令，主種子 20260812；訓練不用任何指令。
- **淨化**：七道（JPEG 80／50／30、blur σ1／2、crop-resize 0.1、rotate15）＋ VAE-Trans。
- **加性對照**：9 個等失真臂（DAYN、DIA-R、DIA-PT、SIFM、DANP、Mist、PhotoGuard encoder、DCT-Shield、DCT-Shield-Y；倍率在 portraits 8 張校準）＋ BlurGuard（sigma_weighting 0、lr 0.06，事後縮放 0.6071，標 modified_from_paper；Pexels 防禦圖 LPIPS 0.271）。Fig. 3 的 DCT-Shield-Y 用原生預算。
- **主表**（Table 1；上段 ip2p 全方法，下段 UltraEdit／FLUX 只列 4 個加性臂＋CAPGuard）：防禦圖 LPIPS、LAION、MUSIQ-AVA；編輯結果 LPIPS（None、Purified* 七道平均、VAE-Trans）；SigLIP（Purified*、VAE-Trans）；ArcFace（Purified*、VAE-Trans）。區間為 95% CI：七道欄跨七道（t, df=6），其餘欄跨 306 張。CLIP-S、CLIP-I、編輯端 FID、SSIM／PSNR、保留率與其他美學在補充材料。
- **跨編輯器**：ip2p 求得的防禦圖直接送 UltraEdit（SD3，512²，g2.5／ig1.5，50 步，add 句型）與 FLUX.1 Kontext（512²，g3.5，28 步，NF4）。
- **消融**（Table 3）：A 隨機 Θ、B σ＝0、C 拿掉角度上界、D 單一 25° 夾角、C＋D、E 不正規化回 ‖c‖，皆以 2.5 倍為基準；只跑防禦與淨化前編輯，讀數為防禦圖 LPIPS、LAION、MUSIQ 與淨化前編輯結果 LPIPS。

## 3. 現況（2026-11-06 清晨）

| 項目 | 狀態 | 預估 |
|---|---|---|
| CAPGuard 2.5 倍 306 張（淨化前＋七道） | 221／306，0 失敗；color 5 線＋逐張讀數 1 卡 | 11/6 約 08:45 |
| 2.5 倍主表 followup（VAE-Trans、語意、ArcFace、存續比、FID 等 121 項） | baseline 佇列，閘門＝2.5 倍 [FINISH] | 11/6 約 15:00 |
| BlurGuard 其餘 206 張（併為 306 張主表列） | baseline 佇列，排在 2.5 倍 followup 之後 | 11/6 約 20:00 |
| 跨編輯器補到 306 張（UltraEdit、FLUX 512；4 臂＋CAPGuard 2.5 倍＋未防禦；約 475 GPU 小時） | 暫停，`queue/jobs_deferred_manual.txt` | 等使用者下令；6 卡約 3.3 天 |
| 消融 A–E（306 張，約 305 GPU 小時） | 程式已備、未開跑 | 等使用者下令；6 卡約 2.2 天 |
| 淨化強度掃描 306 張（約 100 GPU 小時，補充材料） | 未排 | 等使用者決定 |

- 已有讀數：主表 9 臂＋CAPGuard 1.0 倍（306 張）；CAPGuard 2.5 倍逐張彙整（n＝143：防禦圖 LPIPS 0.300、LAION 5.328、MUSIQ 5.314）；BlurGuard 100 張（七道 0.271、VAE-Trans 0.136）；UltraEdit 50 張、FLUX 512 34 張（4 臂＋CAPGuard 1.0 倍）。
- 2.5 倍主表到齊後 paper 要做：Table 1 的 CAPGuard 換 2.5 倍、摘要與貢獻第二條倍數重算、Fig. 2 與 Fig. 3 (a) 換 2.5 倍、Method 上界數字改 25°／7.5°。

## 4. 論文

- 標題：CAPGuard: Color-Based Anti-Purification Image Immunization against Diffusion-Based Editing。
- CVPR 2027：摘要註冊 11/10、全文 11/16、補充材料 11/23（皆 AOE，台灣時間為隔日 19:59）；正文 8 頁；LLM Usage Statement 已寫；不放 URL；匿名。
- 正文含 Conclusion 止於第 8 頁。Fig. 1 teaser 為 man_60_p2（helmet；上列原圖＋六法防禦圖，下列 blur σ=2 淨化後編輯；方法名附出處；CAPGuard 2.5 倍）；Table 1 為合併跨編輯器的單表；Fig. 3 (a) 為分組直方圖、(b–d) 為 8 張強度掃描並含 None 起點；同期工作 SPIN、SRIM、Copying as Defense 已引用。未到數字以 `\prelim{值}{註解}` 紅字標示，`\prelimfinaltrue` 時殘留即報錯。
- 交指導者時程（使用者的時間表文件 https://claude.ai/artifact/UierLoU8gLjvznRzp9iPmL ）：第一批 Abstract、Related Work、Method 於 11/7 上午（倍數待 11/6 數據）；第二批 Introduction、4.1–4.3；第三批 4.4–4.6、Limitations、Conclusion；11/15 全文稿、11/16 提交。該文件尚未反映 teaser 改 man_60、主表合併、自動排程取消。
- 待使用者決定：跨編輯器與消融的開跑順序與時間、強度掃描 306 張是否跑、Limitations 的配件出現計數（目前 8 張）是否改 306 張。
- `main.bib` 已核實；來源較弱的條目為 `doprotect2026`、`zeng2020lut`。新增條目由 paper 通知後核實。

## 5. 其他

- 簡報不再維護。
- 遠端主機：`nelson0314@server.basiclab.lab.nycu.edu.tw`，`-p 10101`（basic-1）／`-p 10102`（basic-2），repo `~/image-immunization`。
