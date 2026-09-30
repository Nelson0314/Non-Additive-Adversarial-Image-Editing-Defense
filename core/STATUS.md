# core：現況與接續指引

共用正本。模組契約與工具用法見 `README.md`；來源與命名對照見 `docs/PROVENANCE.md`，三份流程差異見 `docs/PIPELINE_BEHAVIOR.md`，淨化見 `docs/PURIFICATION.md`。

## 範圍

- `src/immunization_core/`：受害模型 adapter（IP2P、SD、SD-inpainting、SDXL）、指標、淨化算子與固定淨化協定（`purifiers/protocol.json`）、
  編輯／位移／保留量流程、幾何遮罩、色彩與載體最佳化 helpers、I/O 與產物版面。baseline 的攻擊實作不在 core。
- `scripts/`：GPU 租約工具（`gpu_policy.sh`、`gpu_lease.sh`、`measure_free_gpus.sh`、`run_with_gpu_lease.sh`、`run_queue_worker.sh`）、
  `run_trial_lifecycle.sh`、`generate_vendor_snapshot.py`、`generate_requirements_lock.py`。
- 使用端：`baseline`、`color`、`style` 經各自的 `vendor/` 快照使用，不直接 import `core/`。

## 現況

- CPU 測試：`python -m pytest -q -p no:cacheprovider`（預設排除 `weights` 標記，需權重的案例以 `-m weights` 明確選取）。
- 各專案的 `vendor.lock.json` 記錄匯出時的 core commit；`generate_vendor_snapshot.py --check` 在三個專案皆通過。

## 已知限制

- 外部權重淨化算子（DiffPure、IMPRESS、Adverse Cleaner）只有 CPU 契約測試，未在 GPU 與真實權重上驗證數值。
  主表淨化協定的七道算子（`crop_resize0.1`、`jpeg30`、`jpeg50`、`jpeg80`、`blur1`、`blur2`、`rotate15`）不依賴這些權重。
- 歷史算子 `gridpure`、`fdpure` 沒有實作，`available=False`，使用時明確拒絕。
- NCF 載體（`NCFColorParam`）、`lowfreq_color` 與 `color_amplitude` 的幅度求解器不在任何現行專案的 import 閉包內，未納入 core；原始碼在 `archive/anti-purification/src/defense/`。
- `requirements.lock`：core 本身的鎖定檔由 `python scripts/generate_requirements_lock.py` 在執行測試的環境產生；未入庫。

## 規則

共通規則見根目錄 `CLAUDE.md`。core 的修改必須附測試；修改後依序提交 core、重新匯出三個專案的 `vendor/`、在各專案執行測試，再提交匯出結果。
