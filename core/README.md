# immunization_core

影像免疫研究的共用模型、量測與 I/O 套件。來源以複製方式移植，原有專案與呼叫端保持原狀。此版本為第 3 項的基礎模組階段；尚未提供完整 pipelines、淨化、產物 layout、最佳化與 GPU 租約工具。

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
| `io`、`artifacts.images` | CSV 欄位聯集、RGB 載入與 resize、PNG 量化及存檔。寫入函式會覆寫明確指定的目的檔案。 |

匯入全部公開模組不載入權重、不連網、不建立 CUDA context。模型 adapter 建構及指標物件建構／計算依其契約載入權重；所需後端列於 `editors`、`metrics`、`identity` extras。未安裝所需後端時正常報錯，不替換為近似指標。

無權重 CPU 驗證：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest core/tests -q -p no:cacheprovider --basetemp=.tmp/codex_audit/pytest_core
```

測試預設排除 `weights` 標記。需要真實指標權重的測試以 `-m weights` 明確選取；缺依賴或執行錯誤會失敗，不以廣泛例外轉為 skip。本階段不執行此組。獨立副本測試會禁止連網、CUDA 初始化，並匯入副本中的全部公開模組。

三份流程差異見 [PIPELINE_BEHAVIOR.md](docs/PIPELINE_BEHAVIOR.md)；來源、命名與後續範圍見 [PROVENANCE.md](docs/PROVENANCE.md) 及 [STATUS.md](STATUS.md)。
