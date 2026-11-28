# color：現況與接續指引

冷啟動順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔 → trial `trials/lut3d_defense_and_naturalness/HANDOVER.md`（〈現行狀態〉節為現況，其次〈零〉）
→ `docs/EXECUTOR_HANDOVER.md` → 專案記憶 `MEMORY.md`。設計沿革見 `docs/DESIGN.md`，已刪除的暫時性嘗試見 `docs/TRIALS.md`，
執行方式見 `README.md`。trial 目錄不入版控，實體在遠端 `~/image-immunization/color/trials/lut3d_defense_and_naturalness/`。

## 現況

- 論文方法 **ImmuLUT**（原名 CAPGuard；條件名 `color3d_scale2p5`，Pexels 版 `color3d_scale2p5_pexels`）：33³ RGB 查找表，
  以黑點為原點、灰軸為極軸的色度旋轉場，旋轉場以色空間高斯模糊（σ ＝ 4 格點）平滑；極角變化 < 25°、方位角（色相）變化 < 7.5°、
  ‖c‖ 不變，三者由參數化保證（tanh 壓縮），無投影、無 LPIPS 上下界；損失只有 ip2p VAE 編碼器潛向量距離；Adam 1800 步，種子 0。
  程式：trial `generate_color3d_budget_defenses.py`（登錄倍率）→ `generate_color3d_defenses.py` 的 `SmoothAngleLut`。
- 論文定調：ImmuLUT 的主張為淨化後保留防禦效果；不主張未淨化時阻擋編輯（未淨化時 ImmuLUT 的編輯通常照樣完成）。
- 主表（Pexels 306 張、ip2p、四句、主種子）、消融九列（crosseditor100）、色彩校正淨化、論文圖素材均已完成；
  數字、路徑與產生腳本見 trial `HANDOVER.md`〈現行狀態〉與〈零〉。
- 消融 306 張已完成：trial `results/ablation_pexels306/`（`summary.csv`：common 301 張與 all；`paired_differences.csv`／
  `paired_differences_all.csv`：列 − Full，paired bootstrap 10,000 次、種子 20260812；`failed.csv`）。各列設定與 200 張版相同。
- 消融降強度（教授 11/25 第 1 點延伸：防禦效果相近時比較畫質）：B2、B3 各三個防禦圖 LPIPS 目標，r ＝ 0.85、0.70、0.55 倍 Full 同一張，
  容差 ±0.015；機制同消融配對（`with_lpips_interval`），只換目標 t，每個目標各自求解。條件
  `color_baseline_<huesat|affine>_lpips<0p85|0p70|0p55>_pexels`（`_pexels` 須在結尾，`run_lut_queue.sh` 依結尾判定資料集）。
  樣本 `results/ablation_strength/images.txt`（crosseditor100 排序後 `default_rng(20260812).choice(100, 30, replace=False)`）。
  協定同消融 ISR（ip2p text 7.5、image 1.8、50 步、主種子、4 句 × none／jpeg50／blur2／vae_trans）；Undefended、Full 沿用既有編輯。
  程式：`scripts/generate_ablation_strength_sample.py`、`generate_ablation_strength_defenses.py`（generator）、
  `run_ablation_strength.sh`（遠端 basic-2 以 `all` 串接 prep → 佇列 → 每條件一卡 VAE-Trans 與 CPU 美學 → 拼圖與彙整）、
  `generate_ablation_strength_sheets.py`、`measure_ablation_strength_summary.py`。
  log：trial `logs/ablation_strength/all.log`（結束行 `[STRENGTH-ALL] … rc=`）、`logs/lane_color_abl_s_<1–6>.log`、
  `logs/ablation_strength/vae_<條件>.log`。完成後產物：`results/ablation_strength/{summary.csv,summary.txt,per_image.csv}`（遠端），
  拼圖 trial `artifacts_pexels/ablation_strength_sheets/`（120 張、8 列 R1–R8 打亂，`key.csv`、`cells.csv`）。
  已完成（rc＝0）：讀數與 `key.csv` 在 `results/ablation_strength/`；Sonnet 評審與計分在 `docs/isr_pilot/ablation_strength/`（commit `64c7db69`）；
  Full、B2_lpips0p85、B3_lpips0p55 三者都擋下的格子：`scripts/measure_ablation_strength_all_blocked.py` → `results/ablation_strength_all_blocked/`，
  拼圖 trial `artifacts_pexels/ablation_strength_all_blocked/`。
- 消融降強度擴充到 crosseditor100 全部 100 張（以 JPEG 50、blur σ2、VAE-Trans 合併的 ISR 對齊；30 張樣本上 Full 19.9％、
  B2_lpips0p85 18.3％、B3_lpips0p55 23.5％）：條件 `color_baseline_huesat_lpips0p85_pexels`、`color_baseline_affine_lpips0p55_pexels`
  （各補 70 張，30 張沿用；寫進同一條件目錄）與 `color_baseline_affine_lpips0p40_pexels`（100 張，從下方夾住 Full），共 240 次求解；
  機制與協定同上一項，none 照跑。樣本 `results/ablation_strength_crosseditor100/images.txt`（crosseditor100 排序後 100 張）。
  driver `scripts/run_ablation_strength_crosseditor100.sh`（遠端 basic-2 以 setsid 依序跑 queue → check → readout → vae → sheets → summary，
  每段寫 `[STEP] <段> rc=`，失敗即 `[FATAL]` 停止；prep 已先行完成）。log：trial `logs/ablation_strength_crosseditor100/all.log`
  （結束行 `[CROSSEDITOR100-ALL] … rc=0`）、`logs/lane_color_abl_x_<1–6>.log`、`logs/ablation_strength_crosseditor100/`（租約、CPU 美學）。
  VAE-Trans：新增影像寫入 trial `artifacts_pexels/vae_trans_crosseditor100/<條件>/`；30 張樣本沿用 `artifacts_pexels/vae_trans/<條件>/`。
  readout 段以 100 張重做佇列讀數，trial `results/<條件>/` 的 fidelity.csv、`additional/aesthetic.csv` 改為 100 張（30 張數值不變）。
  完成後產物：拼圖 trial `artifacts_pexels/ablation_strength_crosseditor100_sheets/`（400 張 × 5 列 R1–R5：Undefended、Full、
  B2_lpips0p85、B3_lpips0p55、B3_lpips0p40，`default_rng(20260812)` 逐張打亂；`key.csv`、`cells.csv`），讀數遠端
  `results/ablation_strength_crosseditor100/{per_image.csv,summary.csv,summary.txt}`。
  現況：依使用者指示停止於 vae 段；求解、讀數、none／JPEG 50／blur 2 編輯已完成，VAE-Trans 部分完成，不重啟、不接續 vae、sheets、summary。
  停止時 `vae_trans_crosseditor100/<條件>/` 的 VAE-Trans 編輯數：B3_lpips0p40 180／400、B2_lpips0p85 183／280、B3_lpips0p55 178／280；
  已產出的檔案保留。5 列拼圖、`key.csv` 與 `results/ablation_strength_crosseditor100/` 的讀數三檔未產生。
  vae 段第一次失敗的原因為 artifacts 路徑是相對路徑（`bl()` 先 cd 到 baseline），修正於 commit `0567f106`。
  executor 以 70 張新拼圖加 30 張樣本的既有判定計算 JPEG 50＋blur 2 的 ISR：遠端 basic-2 `~/executor_tmp/score100/`（計分程式與輸入）、
  `~/executor_tmp/sheets70/`（70 張的拼圖與 `key.csv`）。
- 多色調 ImmuLUT（論文 Fig. 4 方向，未納入論文；目標為防禦圖像真人會套用的相片濾鏡且未淨化即擋下）：搜尋已停止，無進行中的批次。
  結論：police suit 與血腥句（`give this person rotting zombie skin`）在所有參數化下沒有任何一格於未淨化時擋下，最佳為原衣物保留但多肩章、
  臂章；woman_111、woman_152、man_32、woman_59 在外觀自然的設定下結論相同。helmet 的未淨化擋下來自旋轉場對暗色（頭髮）的大位移，
  與外觀自然直接衝突：去黑位抬升、σ ＝ 8、暗部不旋轉後頭髮恢復正常，helmet 全部出現頭盔。霧感來自濾鏡基底的黑位抬升，
  髒感來自旋轉場對暗色的大位移，LUT 格點數不是主因。已試方向：角度預算（受步長限制）、學習率與色調牽引、使用者研究濾鏡起點、
  參數式風格族（28 種起點）、ip2p UNet 端到端項、低頻空間光影、粗糙／平滑 LUT、參考圖濾鏡擬合與位移、旋轉場元件變體（濾鏡中性軸、
  依色相分配預算、極角／色相上限、clean、暗部門檻與黑位掃描）。各方向結論、最接近的候選、事件、拼圖、verdict 與程式清單見 trial
  `HANDOVER.md`〈現行狀態〉C、D。
- 遠端清理：ImmuLUT 1.0 倍（`color3d_pexels`）、濾鏡試驗、已否決的 teaser 篩選批、多色調中間解與隨機對照的影像產物已刪除
  （清單 trial `runtime/cleanup_delete_list.txt`，173 個目錄、約 17 GB）；各條件的讀數 CSV 保留在 trial `results/`。
- 升格待辦：ImmuLUT 求解鏈與論文讀取的結果 CSV 升格進 `src/`、`scripts/`、`results/`（只收 CSV，`.pt` 載體記遠端路徑），
  ImmuLUT 不再引用 `method.ColorMap` 後刪除之；trial 中已否決的部分寫入 `docs/TRIALS.md` 後刪除（細目見 trial `HANDOVER.md`〈現行狀態〉E）。
  多色調搜尋的程式與影像產物（trial `artifacts/parametric_style/`、`artifacts/user_filter/`）未升格，處置方式同上，刪除前須經使用者確認。
- 分工：color 負責 ImmuLUT 的求解、編輯、讀數與 color 相關圖素材（只寫 `color/`）；baseline 負責對照方法；paper 負責
  `docs/paper/` 與 `docs/paper_tools/`；executor 負責進度、派工、文獻與 push。

## 規矩

- 共通規則見根目錄 `CLAUDE.md`。
- 遠端：`ssh -p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡）`nelson0314@server.basiclab.lab.nycu.edu.tw`，NFS 共用；
  本機只編輯、ssh、看圖，CPU 分析與畫圖也在遠端跑；遠端 Python 為 `~/venvs/wacv/bin/python`。
- GPU：全局合計 ≤ 6 張（所有主機與 session），只取完全空的卡；color 優先。取卡用 trial `tools/run_with_lease_retry.sh`
  （`LEASE_LIMIT=6 HOSTS=<本機>`），必須在有卡的主機本機開（basic-1 連不到 basic-2）。
- 殺行程以明確 PGID 終止，終止後以 `ps` 與 `~/gpu_leases/` 回頭驗證；不用 `pgrep -f`／`pkill -f`（會匹配到 ssh 本身）。
- 本機背景等候在本機記憶體不足時會被系統停止；長佇列在遠端以前一批事件觸發下一批，不依賴本機監看。
- 遠端腳本不可就地覆寫：先寫 `.tmp` 再 `mv`；`.sh` 為 LF。
- 不改 `docs/paper/`、`docs/paper_tools/`（含 `generate_pexels_teaser_figure.py`）與 baseline 的程式；只呼叫。
- 看圖規範與剔除條件見 trial `HANDOVER.md`〈零〉0.2。
