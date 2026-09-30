# 第 3 項第一段交接：core 基礎模組

基準 HEAD：`57a28a9324362cc9fa68a21cc1970632fefbc1a3`。本輪依分段授權完成可獨立驗收的基礎模組，**第 3 項尚未全部完成**。工作目錄新增 `core/` 的 40 個檔案；原有受版控檔案無修改，index 為空。未建立 commit、未 push、未連線遠端、未使用 GPU。

## 完成範圍

- `core/docs/PIPELINE_BEHAVIOR.md`：先比較主線、main_table、lab 的編輯／淨化／displacement／retention 行為。記錄 lab 的子集篩選順序、條件過濾，以及 main_table／lab 的 `purified_mask()` 幾何版本與快取鍵；列出後續合併不得改變的科學協定。
- `core/pyproject.toml`、`src/immunization_core/`、`tests/`：可安裝的 src-layout 套件。匯入不搜尋兄弟專案或使用者家目錄。
- `editors`：複製 IP2P；將原 SD 模組按責任拆為 Stable Diffusion、inpainting、SDXL 與 conditioning。公開 helper 改為 `concatenate_conditioning()`、`expand_conditioning()`。
- `metrics`：只移植活動閉包中的 `acutance`、`arcface`、`identity`、`regional`、`standard`、`suite`。既有後端、計算、空值與 CSV 欄名保留；baseline 攻擊實作未納入。
- `runtime/device`、`io`、`artifacts/images`：保留 device／precision、尺寸修正、CSV 聯集與 PNG 量化。TF32 變數改為 `IMMUNIZATION_ALLOW_TF32`，仍僅接受字串 `1` 啟用，預設關閉；此介面改名獨立成 patch。
- 複製 11 個既有測試檔，新增 4 個測試檔。既有權重案例以 `weights` 明確選取；不以廣泛例外或缺套件轉為 skip。

來源與分段範圍分別記錄於 `core/docs/source_manifest.json`、`core/docs/PROVENANCE.md`、`core/STATUS.md`。靜態閉包盤點保存在 `item3_patches/import_inventory.json`。

## 提交分組

| 順序 | 英文 commit message | 範圍 |
|---|---|---|
| 1 | `Document pipeline behavior and core migration boundaries` | core 文件、行為差異表、來源對照與待辦。 |
| 2 | `Copy shared editors metrics and IO into an installable core package` | pyproject 與複製／拆分的共用模組；此步先保留舊 TF32 變數供下一份 patch 單獨改名。 |
| 3 | `Rename the core TF32 authorization setting` | 新 core 的環境變數改名與對應 precision／TF32 測試；不修改舊來源。 |
| 4 | `Port CPU contract tests and verify standalone core imports` | 其餘移植／新增的契約與獨立副本測試。 |

上述為待提交順序，尚無本輪 commit ID。patch 3 依賴 patch 2，需依 `series.json` 順序執行。

## 驗證

| 驗證 | 結果 |
|---|---|
| `python -m pytest core/tests -q -p no:cacheprovider --basetemp=.tmp/codex_audit/pytest_item3_foundation`，`CUDA_VISIBLE_DEVICES=''`、`PYTHONIOENCODING=utf-8` | **49 passed，17 deselected**。排除的 17 個案例需要真實指標權重；未執行模型下載或 GPU 數值實驗。 |
| 無兄弟專案的複製目錄，以獨立 `PYTHONPATH` 匯入全部公開模組 | **19 個模組通過**；網路、權重下載與 CUDA 初始化呼叫設為失敗，且匯入來源全在副本內。 |
| 獨立副本 `pip install -e` | `--no-index --no-deps --no-build-isolation --target <隔離 site>` 成功；全新 `python -I` 程序使用隔離 site，19 個模組全部由副本載入，版本 `0.1.0`。未改動本機原有安裝。 |
| AST 比對 | **42 個移植函式／類別一致**。僅排除 docstrings，正規化已列出的 namespace、helper 與環境變數改名；全檔複製模組的常數亦一致。拆分模組所有被引用的 global name 可解析。 |
| 四份 patch 順序套用 | 隔離目錄中 `git apply --check --whitespace=error` 與實際套用全部通過，最終內容與工作目錄一致。 |
| PowerShell 提交腳本 | 語法解析及 pytest 暫存路徑排除案例通過；未執行提交腳本。 |
| 原始檔案 SHA-256 | **1,725 個受版控檔案全部相同**，包含 697 份 CSV、原 `anti-purification/src`、main_table 與 lab 呼叫端。另核對 3 個非排除的既有未追蹤檔案相同，包含使用者 PDF。 |

`git diff --check` 通過。本段未新增或修改 `.sh`，未對尚未移植的 GPU 租約工具宣稱 `bash -n` 驗收。

證據檔位於 `item3_patches/`：`retained_equivalence.json`、`install_verification.json`、`verification.json`。目前本機未安裝 piq／diffusers，因此真實後端建構及權重測試不在本段驗收結果內；安裝／公開匯入與無權重契約已驗證。

## pytest 暫存目錄排除

baseline 建立時先依路徑排除，再讀取雜湊；提交腳本亦在 hash 檢查之前排除：

- `anti-purification/runs/ncf_cpu_test_tmp/**`
- 任意路徑組件 `pytest-of-*`
- 任意路徑組件 `.pytest_tmp_*`

上述路徑未列入 `worktree_sha256` 或 `untracked_sha256`，不讀取受限 ACL 的舊測試資料。它們仍保留於原處。

## 協調端執行與後續派工

```powershell
& .\.tmp\codex_audit\item3_patches\commit_item3.ps1
```

腳本先核對 HEAD、空 index、既有受版控檔案無修改、已審 core 檔案／patch／證據檔 SHA-256，再依序套入 index，核對每個 commit 的 staged 檔案集合並提交。結果寫入 `item3_patches/commits.json`。本包 `phase=foundation`、`item_complete=false`。

第 3 項尚餘六類子項，詳見 `core/STATUS.md`：

1. 淨化 registry／算子與歷史選配 `freq_grid` 來源處理。
2. `immunization_core.pipelines` 合併，保留幾何遮罩、子集順序、條件過濾及科學協定。
3. 顯式產物 `artifacts/layout`。
4. 活動閉包中的共用最佳化、色彩與 objective helpers，以及相應美式名稱。
5. `core/scripts/` 五支 GPU 租約工具與 `run_with_gpu_lease.sh` 改名、queue 的可注入依賴。
6. 上述模組的相關測試與完整 core 的獨立副本驗證。

本輪停在此分段，等待協調端派下一段。無額外需裁定事項。
