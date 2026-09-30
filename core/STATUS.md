# 第 3 項：基礎模組階段

來源基準：`57a28a9324362cc9fa68a21cc1970632fefbc1a3`。

已建立可安裝的 `immunization_core` 套件、三份 pipelines 行為差異表、受害模型 adapters、活動閉包內六個 metrics 模組、device、I/O 與影像存檔介面。原 `anti-purification/src`、main_table、lab 呼叫端與所有資料保持原狀。baseline 攻擊實作未納入 core。

第 3 項尚餘：

1. 移植淨化 registry 與算子，明確處理目前 `src.purify.freq_grid` 的歷史選配來源；不得隱性匯入封存目錄或以近似算子取代真算子。
2. 依 `docs/PIPELINE_BEHAVIOR.md` 合併編輯、淨化、位移與 retention 為 `immunization_core.pipelines`，保留 `purified_mask()` 與條件過濾。
3. 建立顯式 `artifacts/layout`，取代後續活動 CLI 的兄弟目錄探索；現有 CLI 在第 4、5 項才切換。
4. 處理 color／style 閉包中的共用最佳化、色彩與 objective helpers：`color_amplitude`、`delta_e_torch`、`immunise`、`instruction_free`、`lowfreq_color`、`ncf_param`、`uniformity`。將共用部分依責任放置，改為 `optimization`、`optimize_carrier`、`randomize_carrier`、`quantize` 等美式名稱，避免納入 baseline 攻擊求解器。
5. 複製五支 GPU 租約工具至 `core/scripts/`，將 `run_on_card.sh` 改名為 `run_with_gpu_lease.sh`，使 queue 呼叫與驗證依賴可明確注入；保留全局授權卡數與租約原子性。
6. 上述模組相關測試隨行，再以完整 core 的獨立副本驗證全部公開模組；延伸處理新模組的公開契約與來源說明。

本階段的 CPU、隔離匯入及 AST 比對結果由 `.tmp/codex_audit/ITEM3_REPORT.md` 記錄；本文件只記範圍與待辦，不宣稱第 3 項全部完成。
