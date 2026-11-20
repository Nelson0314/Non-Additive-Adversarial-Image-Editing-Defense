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
- 多色調 ImmuLUT（論文 Fig. 4 方向，未納入論文）：色調前置、色調初值、學習率與色調牽引的結論，以及 50 張淨化後篩選的通過清單，
  見 trial `HANDOVER.md`〈現行狀態〉C。現行方向為使用者研究的自然濾鏡（warm、cool、faded、vivid、muted、vintage）為起點、
  以未淨化即擋下為判準：程式 trial `generate_color3d_user_filter_defenses.py`，第一批 `runtime/user_filter_batch.sh`
  （man_00、woman_136、woman_164、woman_152 × 6 濾鏡 × 只套濾鏡／lr1／lr10），拼圖 `artifacts/user_filter/contact/`，事件 `[UF-BATCH1]`
  （trial `logs/night_events.log`）。
- 遠端清理：ImmuLUT 1.0 倍（`color3d_pexels`）、濾鏡試驗、已否決的 teaser 篩選批、多色調中間解與隨機對照的影像產物已刪除
  （清單 trial `runtime/cleanup_delete_list.txt`，173 個目錄、約 17 GB）；各條件的讀數 CSV 保留在 trial `results/`。
- 升格待辦：ImmuLUT 求解鏈與論文讀取的結果 CSV 升格進 `src/`、`scripts/`、`results/`（只收 CSV，`.pt` 載體記遠端路徑），
  ImmuLUT 不再引用 `method.ColorMap` 後刪除之；trial 中已否決的部分寫入 `docs/TRIALS.md` 後刪除（細目見 trial `HANDOVER.md`〈現行狀態〉E）。
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
