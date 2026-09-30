# 來源、閉包與移植邊界

基準為 `57a28a9324362cc9fa68a21cc1970632fefbc1a3`。來源 SHA-256、目的地與擷取定義見 [source_manifest.json](source_manifest.json)。以 main_table 與 lab 的活動 Python 檔為入口，解析含函式內 import 的靜態閉包；不因某個模組位於歷史 `src/` 就整個匯入 core。

| 原模組 | 本階段位置與處置 |
|---|---|
| `src.models.ip2p` | `editors.instruct_pix2pix`；保留 adapter、模型 ID 與參數。 |
| `src.models.sd` | 拆為 `editors.stable_diffusion`、`inpainting`、`stable_diffusion_xl`、`conditioning`；各類別的方法本體保留。 |
| `src.metrics.{acutance,arcface,identity,regional,standard,suite}` | `immunization_core.metrics` 同名模組；後端、數學式、缺臉／空區域／樣本數契約及資料欄名不變。 |
| `src.utils.device` | `runtime.device`；環境變數改為 `IMMUNIZATION_ALLOW_TF32`，預設仍為關閉。舊環境變數僅由未修改的舊套件解讀，新套件不建立隱式別名。 |
| `src.utils.io` | `immunization_core.io`；保留第 1 項已修正的寬高尺寸檢查、插值與 CSV 欄位聯集。 |
| `src.utils.artifacts` | 僅移植活動入口使用的 `save_image()` 及其 `_to_uint8()` 依賴至 `artifacts.images`；未引入歷史殘差／頻譜／trace 圖表工具。 |

`cat_cond`／`expand_cond` 的新公開名稱為 `concatenate_conditioning`／`expand_conditioning`。舊專案尚未切換，原 API 仍在原來源可用；後續呼叫端依表改名，數學行為不變。

靜態閉包包含上述六個 metrics 模組，不包含 `metrics.aesthetic`、`metrics.layout`、`metrics.naturalness`。此處的 `metrics.layout` 是人體分割布局指標，與待建的產物路徑 `artifacts/layout` 不同。

閉包中的 `src.baselines` 攻擊模組不納入 core，保留第 4 項處理。`src.defense` 的七個活動 helper 模組與四個淨化模組待下一段依責任移植；`src.purify.freq_grid` 是 registry 中的延遲 import，現有 `anti-purification/src` 沒有該檔。後續必須明確處理選配來源，不複製缺失 import 或搜尋兄弟目錄來掩蓋它。

## 保存的量測與模型契約

- LPIPS／DISTS 使用 piq；`RegionalLPIPS` 共用同一個 LPIPS 實例，遮罩以面積權重縮放。CSV 的 `fid_` 表示 fidelity，`frechet` 表示 FID，既有鍵名不因 API 改名變動。
- FaceNet 的缺臉為未定義值；ArcFace 使用獨立臉部偵測與其既有空欄 schema，不混用兩種身分嵌入。
- IP2P 影像條件不乘 VAE scaling factor；一般 latent 編碼保留 scaling；批次每張圖對應其 seed 與 generator。adapter 預設步數 100、image guidance 1.5，與 pipeline 的 50／1.8 明確區分。
- SD／inpainting／SDXL 的取樣、插值、九通道條件、文字／pooled 嵌入與 precision 規則不變。TF32 僅改環境變數名稱，未改啟用值。

原標準指標文件的歷史日期與決策來源為基準版本的 `src/metrics/standard.py`（2026-08-19 清單、既有 fidelity 前綴與 SigLIP 門檻校準說明）。新模組首段改述輸入、輸出及失敗條件，未修改常數或公式。其他數學及來源說明保留於相應函式／類別。

## 測試移植

複製 11 個相關測試檔至 `core/tests/`，調整 import 與檔案定位，原測試檔不動。IP2P batch 替身會暫時覆寫類別 property，新副本加入還原 fixture 避免污染後續測試。semantic 委派測試改以 `__new__` 建立不需權重的空實例；其被測路徑不讀 LPIPS／DISTS。真實 metric 權重案例以 `weights` 標記明確選取，移除廣泛捕捉例外或缺套件即 skip 的處理。
