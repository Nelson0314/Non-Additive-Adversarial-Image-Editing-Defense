# color：現況與接續指引

冷啟動順序：根目錄 `CLAUDE.md`、`AGENTS.md` → 本檔 → trial `trials/lut3d_defense_and_naturalness/HANDOVER.md`（〈零〉節為現況）
→ `docs/EXECUTOR_HANDOVER.md` → 專案記憶 `MEMORY.md`。設計沿革見 `docs/DESIGN.md`，已刪除的暫時性嘗試見 `docs/TRIALS.md`，
執行方式見 `README.md`。trial 目錄不入版控，實體在遠端 `~/image-immunization/color/trials/lut3d_defense_and_naturalness/`。

## 現況

- 論文方法 **ImmuLUT**（原名 CAPGuard；條件名 `color3d_scale2p5`，Pexels 版 `color3d_scale2p5_pexels`）：33³ RGB 查找表，
  以黑點為原點、灰軸為極軸的色度旋轉場，旋轉場以色空間高斯模糊（σ ＝ 4 格點）平滑；極角變化 < 25°、方位角（色相）變化 < 7.5°、
  ‖c‖ 不變，三者由參數化保證（tanh 壓縮），無投影、無 LPIPS 上下界；損失只有 ip2p VAE 編碼器潛向量距離；Adam 1800 步，種子 0。
  程式：trial `generate_color3d_budget_defenses.py`（登錄倍率）→ `generate_color3d_defenses.py` 的 `SmoothAngleLut`。
- 論文定調：ImmuLUT 的主張為淨化後保留防禦效果；不主張未淨化時阻擋編輯（未淨化時 ImmuLUT 的編輯通常照樣完成）。
- 主表（Pexels 306 張、ip2p、四句、主種子）、消融九列（crosseditor100）與 200 張八列的成對差、色彩校正淨化、論文圖素材均已完成；
  數字、路徑與產生腳本見 trial `HANDOVER.md`〈零〉。
- 消融擴充至 306 張執行中：新增 106 張（`pexels_subsets/ablation_remaining106.txt`，由 `tools/generate_ablation_remaining_lists.py`
  產生）× 八列 A1–A4、B1–B4，協定與 200 張相同；`ABL_TAG=pexels306 tools/run_ablation_extension.sh lanes|rows|follow`，
  事件 `logs/ablation_pexels306_extension.log`，新增影像的 VAE-Trans 在 `results/<列>/vae_trans_remaining106/`。
  完成後的彙整需合併三段 VAE-Trans 目錄（`vae_trans/`、`vae_trans_extension/`、`vae_trans_remaining106/`）。
- 論文 Fig. 4 多色調 ImmuLUT（woman_111，police suit）：色調 LUT 前置（L ＝ R ∘ P）與色調初值（Θ 先擬合色調 LUT，預算同 ImmuLUT）
  兩種做法、7 個色調、強度 0.25–1.0，判準為不淨化與 blur2 皆保留原衣物；成立者只有與現行解外觀相近的 magenta 初值，
  未納入論文。程式 trial `generate_color3d_tone_defenses.py`、`generate_color3d_tone_init_defenses.py`，拼圖 `artifacts/fig4_tone/sheets*/`。
- 遠端清理：ImmuLUT 1.0 倍（`color3d_pexels`）、濾鏡試驗、已否決的 teaser 篩選批、多色調中間解與隨機對照的影像產物已刪除
  （清單 trial `runtime/cleanup_delete_list.txt`，173 個目錄、約 17 GB）；各條件的讀數 CSV 保留在 trial `results/`。
- 待辦（306 消融完成後）：把 ImmuLUT 求解鏈與論文讀取的結果 CSV 升格進 `src/`、`scripts/`、`results/`（只收 CSV，`.pt` 載體記遠端路徑），
  ImmuLUT 不再引用 `method.ColorMap` 後刪除之；trial 中已否決的部分寫入 `docs/TRIALS.md` 後刪除。
- 分工：color 負責 ImmuLUT 的求解、編輯、讀數與 color 相關圖素材（只寫 `color/`）；baseline 負責對照方法；paper 負責
  `docs/paper/` 與 `docs/paper_tools/`；executor 負責進度、派工、文獻與 push。

## 規矩

- 共通規則見根目錄 `CLAUDE.md`。
- 遠端：`ssh -p 10101`（basic-1，8 卡）／`-p 10102`（basic-2，7 卡）`nelson0314@server.basiclab.lab.nycu.edu.tw`，NFS 共用；
  本機只編輯、ssh、看圖，CPU 分析與畫圖也在遠端跑；遠端 Python 為 `~/venvs/wacv/bin/python`。
- GPU：全局合計 ≤ 6 張（所有主機與 session），只取完全空的卡；color 優先。取卡用 trial `tools/run_with_lease_retry.sh`
  （`LEASE_LIMIT=6 HOSTS=<本機>`），必須在有卡的主機本機開（basic-1 連不到 basic-2）。
- 殺行程以明確 PID（含子行程）終止，終止後以 `ps` 與 `~/gpu_leases/` 回頭驗證；不用 `pkill -f`（會匹配到 ssh 本身）。
- 遠端腳本不可就地覆寫：先寫 `.tmp` 再 `mv`；`.sh` 為 LF。
- 不改 `docs/paper/`、`docs/paper_tools/`（含 `generate_pexels_teaser_figure.py`）與 baseline 的程式；只呼叫。
- 看圖規範與剔除條件見 trial `HANDOVER.md`〈零〉0.2。
