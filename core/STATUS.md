# 第 3 項：基礎模組與淨化階段

基礎模組來源基準：`57a28a9324362cc9fa68a21cc1970632fefbc1a3`；淨化階段基準：`51d58bf2757a691ec4d8c54a983efb74ac41bac8`。

已建立可安裝的 `immunization_core` 套件、三份 pipelines 行為差異表、受害模型 adapters、活動閉包內六個 metrics 模組、device、I/O 與影像存檔介面。原 `anti-purification/src`、main_table、lab 呼叫端與所有資料保持原狀。baseline 攻擊實作未納入 core。

第 3 項尚餘：

1. 已移植四個淨化模組；歷史 `gridpure`／`fdpure` 明確標示不可用，無封存區 import。外部權重算子的真實數值驗證尚未執行。
2. 固定淨化流程與 `purified_mask()` 已合併。尚待編輯、displacement 與 retention 主流程，保留 lab 子集順序與條件過濾。
3. 已提供顯式 `artifacts/layout`；後續 pipelines 與第 4、5 項 CLI 須接入，不新增兄弟目錄探索。
4. 處理 color／style 閉包中的共用最佳化、色彩與 objective helpers：`color_amplitude`、`delta_e_torch`、`immunise`、`instruction_free`、`lowfreq_color`、`ncf_param`、`uniformity`。將共用部分依責任放置，改為 `optimization`、`optimize_carrier`、`randomize_carrier`、`quantize` 等美式名稱，避免納入 baseline 攻擊求解器。
5. 複製五支 GPU 租約工具至 `core/scripts/`，將 `run_on_card.sh` 改名為 `run_with_gpu_lease.sh`，使 queue 呼叫與驗證依賴可明確注入；保留全局授權卡數與租約原子性。
6. 上述模組相關測試隨行，再以完整 core 的獨立副本驗證全部公開模組；延伸處理新模組的公開契約與來源說明。

基礎與淨化階段的驗證分別記錄於 `.tmp/codex_audit/ITEM3_REPORT.md`、`.tmp/codex_audit/ITEM3B_REPORT.md`；本文件不宣稱第 3 項全部完成。
