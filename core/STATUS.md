# 第 3 項：基礎模組與淨化階段

基礎模組來源基準：`57a28a9324362cc9fa68a21cc1970632fefbc1a3`；淨化階段基準：`51d58bf2757a691ec4d8c54a983efb74ac41bac8`。

已建立可安裝的 `immunization_core` 套件、三份 pipelines 行為差異表、受害模型 adapters、活動閉包內六個 metrics 模組、device、I/O 與影像存檔介面。原 `anti-purification/src`、main_table、lab 呼叫端與所有資料保持原狀。baseline 攻擊實作未納入 core。

第 3 項各子項狀態：

1. 已移植四個淨化模組；歷史 `gridpure`／`fdpure` 明確標示不可用，無封存區 import。外部權重算子的真實數值驗證尚未執行（需權重與 GPU，由協調端處理）。
2. 固定淨化流程、`purified_mask()` 與編輯、displacement、retention 主流程已合併（`pipelines.editing`、`pipelines.displacement`、`pipelines.retention`），保留 lab 子集順序與條件過濾。
3. 已提供顯式 `artifacts/layout`；第 4、5 項的專案 CLI 須接入，不新增兄弟目錄探索。
4. color／style 閉包中的共用 helpers 已依責任放入 `color.space`、`color.difference`、`color.uniformity`、`optimization.carrier`、`optimization.instruction_free`，改為 `quantize`、`optimize_carrier`、`randomize_carrier`。NCF 載體（`NCFColorParam`）、`lowfreq_color` 與 `color_amplitude` 的幅度求解器不在活動閉包內，未納入。
5. 五支 GPU 租約工具已複製至 `core/scripts/`，`run_on_card.sh` 改名為 `run_with_gpu_lease.sh`；queue 的工作執行、驗收與相依改為 `--runner`、`--validator`、`--depends` 注入。`LAB_CAP`、`LAB_MYCAP` 改為 `GPU_CAP`、`QUEUE_CAP`；租約目錄預設值維持 `$HOME/lab_leases`，改名須所有取卡入口同時切換（第 10 項）。
6. 上述模組的測試隨行；完整 core 的獨立副本通過全部 CPU 測試、各 pipeline `--help` 與全部公開模組匯入。

來源雜湊：基礎模組見 `docs/source_manifest.json`，淨化見 `docs/purifier_source_manifest.json`，其餘見 `docs/pipeline_source_manifest.json`。baseline、color、style 三個專案經各自的 `vendor/` 快照使用 core（第 4、5 項）；`anti-purification` 的舊呼叫端未切換，第 6 項整體封存。
