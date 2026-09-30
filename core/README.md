# immunization_core

影像免疫研究的共用模型、量測與 I/O 套件。來源以複製方式移植，原有專案與呼叫端保持原狀。包含基礎模組、淨化、固定淨化流程、幾何遮罩、顯式產物根目錄、編輯與讀數流程、色彩與載體最佳化 helpers，以及 `scripts/` 下的 GPU 租約工具。

```powershell
python -m pip install -e ./core --no-deps --no-build-isolation
```

已有基礎相依的環境亦可將 `core/src` 加入 `PYTHONPATH`。套件可單獨複製至另一目錄使用，不搜尋 `anti-purification`、`lab` 或使用者家目錄。安裝相依的版本範圍不是實驗 lock；正式實驗仍須保存實際環境版本與模型 revision，第 9 項另建立專案鎖定檔。

| 公開模組 | 契約 |
|---|---|
| `editors.instruct_pix2pix` | IP2P 影像條件 scaling、雙引導尺度、逐圖 generator 與 batch 介面。 |
| `editors.stable_diffusion`、`editors.inpainting`、`editors.stable_diffusion_xl` | 共用擴散迴圈，依 adapter 保留 SD、九通道 inpainting 與 SDXL 編碼／條件差異。 |
| `editors.conditioning` | `SDXLPrompt`、`concatenate_conditioning()`、`expand_conditioning()`；序列與 pooled 嵌入共同配對。 |
| `metrics` | `suite`、`regional`、`standard`、`acutance`、`identity`、`arcface`，保留既有後端與 CSV 欄位語意。 |
| `runtime.device` | 裝置與 precision；匯入時依 `IMMUNIZATION_ALLOW_TF32` 設定 TF32，僅字串 `1` 啟用，未設定時關閉。 |
| `io`、`artifacts.images` | CSV 欄位聯集（`write_csv`）與排序欄位（`write_sorted_csv`）、資料集影像列舉、RGB 載入與 resize、PNG 量化及存檔。寫入函式會覆寫明確指定的目的檔案。 |
| `purifiers.operators`、`purifiers.protocol` | 真實淨化與訓練代理分離；固定 identity 加七道淨化的順序及強度。 |
| `pipelines.purification`、`pipelines.masks` | 明確指定資料／輸出根的淨化流程；保留主體極性及 `purified_mask()` 幾何變換。 |
| `pipelines.editing`、`pipelines.displacement`、`pipelines.retention` | 編輯與位移讀數流程；資料根、輸入根與輸出路徑皆為必填參數，配對缺側立即失敗。 |
| `color.space`、`color.difference`、`color.uniformity`、`color.shift` | sRGB／CIELab 轉換；skimage 量測與可微求解兩條 CIEDE2000 路徑（平均與 CVaR）；位移場 TV、U16 與端點讀數；逐通道 Lab 位移的分位數與最大值。 |
| `optimization.carrier`、`optimization.instruction_free` | 以 augmented Lagrangian 在色差上限內最佳化載體（`optimize_carrier`、`randomize_carrier`、`quantize`、`Cap`）；不含指令的 IP2P 目標 `FreeObjective`；`optimization.attention` 為攻擊端 UNet 的自注意力偏離與類別詞交叉注意力質量目標。 |
| `artifacts.layout` | 顯式 `ArtifactLayout` 與唯一防禦 PNG 查找，不探索舊專案、不在建構時建立目錄。 |

固定淨化入口為 `python -m immunization_core.pipelines.purification --data-root <資料集> --output-dir <輸出>`，或以 `--defenses-dir <防禦圖目錄>` 代替 `--data-root`。IMPRESS 的 lpips 後端及 Adverse Cleaner 的 OpenCV-contrib 可由 `purifiers` extra 安裝；DiffPure 另需明確提供 guided-diffusion 與檢查點。歷史 `gridpure`／`fdpure` 未包含實作，`available=False` 並於使用時明確拒絕，無近似替代。詳見 [PURIFICATION.md](docs/PURIFICATION.md)。

匯入全部公開模組不載入權重、不連網、不建立 CUDA context。模型 adapter 建構及指標物件建構／計算依其契約載入權重；所需後端列於 `editors`、`metrics`、`identity` extras。未安裝所需後端時正常報錯，不替換為近似指標。

無權重 CPU 驗證：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest core/tests -q -p no:cacheprovider --basetemp=.tmp/codex_audit/pytest_core
```

測試預設排除 `weights` 標記。需要真實指標權重的測試以 `-m weights` 明確選取；缺依賴或執行錯誤會失敗，不以廣泛例外轉為 skip。本階段不執行此組。獨立副本測試會禁止連網、CUDA 初始化，並匯入副本中的全部公開模組。

## GPU 租約工具

`scripts/` 的五支 Bash 工具共用一個租約目錄（`LEASE`，預設 `$HOME/lab_leases`）。取卡、容量檢查、擁有者驗證與釋放都在同一個 `mkdir` 鎖內；全局卡數由使用者逐次授權，以 `--cap` 或 `GPU_CAP` 寫入租約目錄，未指定時預設全局合計 6 張，計入所有主機、session 與排程。

| 工具 | 用途 |
|---|---|
| `gpu_policy.sh`、`gpu_lease.sh` | 容量政策與租約函式，供其他工具 `source`。 |
| `free_cards.sh` | 列出空閒卡；`--assert` 檢查指定卡。只產生候選清單，不保留卡。 |
| `run_with_gpu_lease.sh --work-dir <目錄> [--env-file <檔案>] <名稱> <指令...>` | 取一張卡的租約後執行單一指令，結束時釋放。 |
| `queue_worker.sh --work-dir --state-dir --log-dir --runner --validator [--depends] <佇列> <工作>...` | 佇列排程；工作執行、輸出驗收與相依由專案以指令注入，驗收通過才記為完成。 |

`trial.sh new|promote|drop <名稱>` 管理各專案不入版控的 `trials/<名稱>/`：`promote` 要求升格內容已提交，`drop` 要求 `docs/TRIALS.md` 已有該名稱的一列，並以 `TRIAL_REMOTE`、`TRIAL_REMOTE_ROOT` 同時刪除遠端副本（只刪本機時明確給 `--local-only`）。

工具以自身所在目錄互相定位，不依賴 CWD；`PY` 未設定時使用 `python`，`PYTHONPATH` 等環境由 `--env-file` 或呼叫端提供。

三份流程差異見 [PIPELINE_BEHAVIOR.md](docs/PIPELINE_BEHAVIOR.md)；來源、命名與後續範圍見 [PROVENANCE.md](docs/PROVENANCE.md) 及 [STATUS.md](STATUS.md)。
