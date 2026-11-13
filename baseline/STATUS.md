# baseline：現況與接續指引

冷啟動先讀 `docs/HANDOVER.md`（範圍與分工、工作規範、結果正本、等失真設定、論文數字出處、暫時性嘗試、佇列、工具索引、已知問題與待決），
再讀本檔。協定、檔案位置與程式清單在 `README.md`；Pexels 資料組的選取與操作在 `docs/PEXELS_PORTRAITS.md`；等失真臂在 `results/aligned/README.md`。
各項工作的逐段紀錄（Pexels 佇列分段、遷移實驗、FLUX 512 協定、color3d 變體的佇列、遠端刪除候選）保留在本檔 commit `05f785f0` 的版本。

## 資料組

| 資料 | 回答什麼 | 位置 |
|---|---|---|
| data/portraits 8 張主表 | 12 個方法在各自原生預算下，把 ip2p／inpaint 的編輯結果推開多少 | `results/*.csv` |
| data/portraits 等失真臂 | 方法縮到同一個防禦圖 LPIPS 之後還推開多少 | `results/aligned/` |
| data/portraits 跨編輯器 | 同一批防禦圖換 FLUX.1-Kontext、UltraEdit 編輯；跨模型移植 | `results/flux/`、`results/ultraedit/`、`results/cross_model/` |
| Pexels 人像 306 張 | 論文主表、ImmuLUT、BlurGuard、跨編輯器（VIBE-DistilledCFG、FLUX.2 klein）的讀數 | `results/pexels_portraits/` |

論文的圖表全部取自 Pexels 306 張；data/portraits 8 張只用於等失真倍率的校準。

## 現況

- 主表、ImmuLUT 2.5 倍、BlurGuard、VIBE-DistilledCFG、FLUX.2 klein 的 306 張讀數皆已完成並入版控（`docs/HANDOVER.md` 3.1）；
  分布指標以 GPU 特徵重算，CPU 版副本在遠端 `~/backup/baseline/fid_cpu_2026-11-07/`。
- 遠端佇列無待執行工作；deferred 清單見 `docs/HANDOVER.md` 3.5。
- ISR 評審：crosseditor100（ip2p 100 張 × 4 句）已完成，結果見根目錄 `docs/isr_pilot/`（論文 Table 2）；isr50 的 ip2p 與 VIBE 拼圖未排程，使用者指示才做。
- teaser 已定案（Fig. 1 殭屍版，由 color 產生）；`docs/HANDOVER.md` 4 的三個 teaser 入口不需執行。
- BlurGuard 依論文協定的重跑未對上論文數字（`docs/HANDOVER.md` 5）。

## 待使用者決定

1. trial `blurguard_paper_protocol`、`isr_guidance_and_budget`、`isr_mllm_judge` 的 promote 或 drop。
2. BlurGuard 在 Pexels 上是否重新校準等失真倍率（現為 0.2745，錨點 0.3344）；BlurGuard 與論文數字的差距是否追查。
3. deferred 的 UltraEdit／FLUX.1 Kontext 工作與 Pexels 淨化強度掃描是否執行。
4. 遠端中間產物的清理（清單已交 executor）。

## 已知限制

- 幾何淨化的分區欄：`results/retention.csv` 中 11 個條件的 `crop_resize0.1`、`rotate15` 分區欄已以 `purified_mask()` 重算（更正前見 commit `228c59b`）；
  `results/aligned/retention.csv` 已查證無需更正。
- SD 2.x 只能用 epsilon-prediction 權重（`sd2-community/stable-diffusion-2-1-base`）。
- `scripts/generate_distortion_displacement_plane.py` 引用的 color 條件 `color3d_adaptive_set`、`color3d_0d4` 產物已由 color 刪除，重跑前須指定新的條件清單。
- 等失真臂的錨點為舊顏色列的 0.3344（使用者指示不重新對齊）。

## 遠端

- 遠端 `~/image-immunization/baseline` 為本專案（以 `git pull` 同步，push 由 executor 執行）；影像產物在 `artifacts/`，執行狀態在 `runtime/`。
- 停工作：以 `queue/claims/<工作>/owner` 的 PID 執行 `kill -TERM -- -<PID>`，再以 `pgrep -g <PID>` 確認 0 個行程、租約已釋放。
- crontab 的修改需使用者同意（備份在兩台 `~/crontab.backup.*`）。

## 規矩

- 引用外部方法的數字依 `docs/reference/BASELINE_PROVENANCE.md` 規則 1（連同該方法的協定與來源版本）。
- 共通規則見根目錄 `CLAUDE.md`。
