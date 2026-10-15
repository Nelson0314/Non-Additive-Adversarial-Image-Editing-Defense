# color：現況與接續指引

設計與結果見 `docs/DESIGN.md`，執行方式見 `README.md`，已刪除的暫時性嘗試見 `docs/TRIALS.md`。

## 現況

- 研究主線 **color3d**：33³ RGB 3D 查找表（`Lut3d`）的全域逐像素顏色映射，目標為 ip2p VAE 編碼器潛向量距離（單次前向，訓練不用編輯指令、不做淨化抽樣），PHR augmented Lagrangian 求解、只交付量化後全部可行的檢查點；約束為防禦圖 LPIPS 區間 0.27–0.30、逐像素色度位移 ≤ 8、斜率 3、相鄰顏色增益與其尾端、色調單調、膚色色相與彩度比。在 trial `trials/lut3d_defense_and_naturalness/`（不入版控）；方法、結果、消融與已退役路線見該目錄 `README.md`，現況、待辦與規範見 `HANDOVER.md`。
- color3d 8 張（ip2p、4 句、32 筆編輯）：防禦圖 LPIPS 0.287；編輯結果 LPIPS 淨化前 0.341、七道淨化平均 0.302、逐格保留率 0.892、VAE-Trans 淨化後 0.185。同協定 baseline 等失真臂（防禦圖 LPIPS 0.308–0.343）：淨化前 0.24–0.39、七道 0.12–0.24、VAE-Trans 0.02–0.07。前一版（防禦圖 LPIPS 0.207）改名 `color3d_chroma4_slope2`，作消融。
- 淨化協定：七道（jpeg80／50／30、blur1／2、crop_resize0.1、rotate15）與 VAE-Trans（arXiv 2603.13028，另列一欄）；色彩正規化類淨化不做。讀數與 baseline 的共同協定（失真、美學、位移、語意、人臉）記於專案記憶 `metric-protocol-color-baseline`；穿透拆解已移出本專案。
- trial `trials/edit_failure_mechanisms/` 保存 `lut3d_mild`（預算 0.2）的解與讀數。
- 前身方法 `color`（CIELAB (a,b) 全域換色表）與其三個 simple 條件、隱含分類器目標（selfrec）已退役，程式與產物已刪除，數據保留在 `docs/DESIGN.md` §4–§7。`method.py` 的 `ColorMap` 與支撐函式仍由 trial 引用。
- 研究方向由使用者決定。遠端是否有工作在跑：以遠端 `~/gpu_leases/` 的租約、各 trial 的 `runtime/` 與 `logs/` 為準。
- 未防禦編輯（`artifacts/undefended_edits/ip2p_si18`）與淨化後的未防禦分母（`artifacts/purified_edits/undefended/`）為 baseline 同名產物的複本，放在本專案的 `artifacts/`，不連結到 baseline。
- `requirements.lock` 未入庫；由遠端執行環境以 `vendor/scripts/generate_requirements_lock.py` 產生。

## 執行紀錄

- 遠端：`ssh -p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡）`nelson0314@server.basiclab.lab.nycu.edu.tw`，NFS 兩台共用；本專案位於 `~/image-immunization/color`。
  兩台的卡常被實驗室其他使用者占用；取卡工具只取本機的卡，需要時在另一台另開 main。
- 實測成本：color3d 求解約 13–15 分鐘／張（PHR 1800 步）；單張的編輯 ＋ 七道淨化 ＋ 淨化後編輯約 15–18 分鐘；8 張完整協定約 17 GPU 小時（4 條件 × 8 張）÷ 線數；VAE-Trans 淨化＋編輯約 7 分鐘／條件。

## 規矩

- **卡數**：依使用者口頭指定；未說明時全局合計 6 張。`run_with_gpu_lease.sh` 的 `--limit` 比的是全局租約總數，不要給；取卡失敗（沒有空卡）時以迴圈每 5 分鐘重試。
- **取卡函式的紀錄一律寫 stderr**：`$(...)` 會收走 stdout。
- **殺行程**：`pkill -f <樣式>` 會匹配到下指令的 ssh；以 PGID 或明確 PID 終止，終止後以 `ps`／租約目錄回頭驗證。`$(...)` 收到多個 PID 時要逐一處理。
- **佇列**：color3d 走 trial 的 `run_lut_queue.sh`（`conds/<條件>.args`、`jobs/<名>.txt`）；只編輯既有防禦圖時不可對有 `.args` 的條件排工作（會重新求解並覆寫），改用複製的新條件名。
- **GPU 優先序**：使用者指令 color 優先；開佇列前請 baseline 立即讓卡（見專案記憶 `color-has-gpu-priority`）。
- **分母**：新種子或新設定的編輯不寫進 `artifacts/undefended_edits/`。
- **`optimize_carrier` 回傳的字典鍵一律 `free_` 開頭**，與 `--objective` 無關。
- 共通規則見根目錄 `CLAUDE.md`。
