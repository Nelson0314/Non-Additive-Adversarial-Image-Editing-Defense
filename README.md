# Image Immunization

影像免疫：在照片發布前對其做一次處理，使攻擊方以公開權重的擴散編輯模型依指令改圖時，編輯結果偏離未防禦時的結果。
威脅模型為白盒（防禦方已知攻擊方的模型）、外掛模組（不改動攻擊方的權重），不同方法的比較先對齊防禦圖失真。

## 目錄

| 目錄 | 內容 | 入口 |
|---|---|---|
| [`core/`](core/) | 共用套件 `immunization_core`（受害模型 adapter、指標、淨化、編輯與讀數流程、色彩與載體最佳化）與 GPU 租約、vendor 匯出、trials、環境鎖定工具 | [`core/README.md`](core/README.md) |
| [`baseline/`](baseline/) | 主表：11 個外部防禦方法與 `color` 共 12 列，在各自原生預算下的防禦、編輯、淨化與讀數；等失真臂、加性穿透拆解、FLUX 與 UltraEdit 全表 | [`baseline/STATUS.md`](baseline/STATUS.md) |
| [`color/`](color/) | 以 CIELAB 全域色彩映射為載體的防禦 | [`color/STATUS.md`](color/STATUS.md) |
| [`style/`](style/) | 以風格編輯為載體的防禦（Neurocomputing 134591 的方法移植） | [`style/STATUS.md`](style/STATUS.md) |
| `docs/` | 論文撰寫用；版控內沒有檔案時目錄不存在 | — |
| [`archive/`](archive/) | 已結束的研究線（`anti-purification/`、`frequency-phase/`）、舊交接文件與重整紀錄（`migration/`）；只保存，不維護執行 | [`archive/migration/RESTRUCTURE_LOG.md`](archive/migration/RESTRUCTURE_LOG.md) |

共通規則見 [`CLAUDE.md`](CLAUDE.md)。

## 專案自足

`baseline`、`color`、`style` 各自可單獨複製使用，互不 import：

- `immunization_core` 以 `vendor/` 快照提供，`vendor.lock.json` 記錄來源 commit 與逐檔 SHA-256。`vendor/` 不就地修改；
  修正在 `core/` 提交後，於 repo 根執行 `python core/scripts/generate_vendor_snapshot.py <專案>` 重新匯出，`--check` 驗證快照與 lock 一致。
- 各專案的 `scripts/env.sh` 設定 `PYTHONPATH=src:vendor`；預設路徑由套件內的 `layout.py` 自專案根推得，不搜尋兄弟目錄或家目錄。
- 依賴範圍在各專案 `pyproject.toml`；實際執行環境的版本以 `vendor/scripts/generate_requirements_lock.py` 寫成 `requirements.lock`，`--check` 比對環境與鎖定檔。

## 資料保存

| 位置 | 內容 | 版控 |
|---|---|---|
| `<專案>/results/` | 數值紀錄（CSV、JSON）；量測結果無法完全重現，一律入庫 | 入庫 |
| `<專案>/data/` | 輸入影像、遮罩、prompts | 入庫 |
| `<專案>/artifacts/` | 防禦圖、編輯圖、淨化圖等影像產物，由已記錄的設定重跑可得 | 不入庫 |
| `<專案>/runtime/` | 佇列狀態、排程 log | 不入庫 |
| `<專案>/trials/` | 暫時性嘗試；成功者搬入 `src/`、`configs/`、`results/` 後刪除，失敗者刪除並在 `docs/TRIALS.md` 留一列 | 不入庫 |

所有文字檔在各平台簽出皆為 LF（根目錄 `.gitattributes`）。

## 執行環境

GPU 工作在遠端執行（`~/image-immunization` 為本 repo 的 clone，以 `git pull` 同步）；取卡一律經各專案 `vendor/scripts/` 的租約工具，
租約目錄預設 `~/gpu_leases`，由所有主機、session 與排程共用。測試只需 CPU：

```bash
cd <專案> && python -m pytest -q -p no:cacheprovider
```
