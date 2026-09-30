# 編輯、淨化與讀數的行為差異

比對基準為 `57a28a9324362cc9fa68a21cc1970632fefbc1a3`。本表先於 `immunization_core.pipelines` 的合併實作建立。主線指 `anti-purification/scripts/`，主表指 `anti-purification/main_table/code/`，lab 指 `lab/code/`。原檔保留，後續活動入口才切換至套件。

## 三份副本的差異

| 流程／行為 | 主線 | 主表 | lab | 共用流程須保留的契約 |
|---|---|---|---|---|
| `edit_preflight.py`：來源定位 | 將 `anti-purification` 加入 `sys.path`；資料與輸出預設相對 CWD | `paths.add_source_to_syspath()`；預設 `paths.PORTRAITS`、`paths.IMAGES / "edit_preflight"` | 與主表相同的定位介面，根位置由 lab 的 `paths.py` 決定 | 新 API 顯式接收資料根與輸出目錄；不探索兄弟目錄。既有入口的路徑預設由後續專案 CLI 保有。 |
| 編輯子集與防禦圖檢查順序 | 先對全部資料套用 `apply_defended()`，再篩選 `--images` | 同主線 | 先篩選影像，再查防禦圖 | 活動共用版本採 lab 順序，保留子集目錄可執行的行為；主表切換時應記錄此輸入驗證差異。 |
| 編輯模型、參數、CSV | 除上列路徑與篩選順序外，運算流程相同 | 同主線 | 同主線 | IP2P／inpainting 的模型 ID、步數、引導尺度、seed、精度、指令序號、既有 arm 合併與 `--metrics-only` 路徑不變。 |
| `purify_run.py` | 明確要求 `--defended` 或 `--data`，以及 `--out` | 運算與參數相同；只有 import 定位與說明路徑不同 | 運算與參數相同；防禦程式名稱的文字不同 | 可合併為同一實作，輸入目錄由呼叫端提供。 |
| `edit_displacement.py` | `--data` 預設 `data/portraits` | `--data` 預設 `paths.IMAGES`，僅使用其 `masks/` | 與主表檔案內容相同 | 顯式提供 mask 根或資料根；LPIPS、主體遮罩極性、配對鍵與 `--conditions` 原樣保留。 |
| `edit_retention.py`：幾何遮罩 | 所有淨化算子使用原位置的主體遮罩；快取鍵為影像名 | `purified_mask()` 對幾何算子同步移動遮罩；快取鍵為 `(image, purifier)` | 同主表 | 合併時保留主表／lab 的幾何版本，不以主線舊副本覆蓋。 |
| retention 的幾何標籤 | 固定字串 `crop_resize0.1`、`rotate15` | 由淨化 registry 的 kind 與 strength 產生 | 同主表 | 影像與遮罩使用同一組算子設定，避免兩側強度不一致。 |
| retention 的條件過濾 | 掃描全部非底線開頭且非 `undefended` 的條件目錄 | 同主線 | 額外支援 `--conditions` | 共用 API 支援顯式條件集合；未指定時維持完整掃描。 |
| retention 的資料根 | 相對 CWD 的 `data/portraits` | `paths.IMAGES` | `paths.IMAGES` | 呼叫端明確指定位置，不由 core 尋找舊專案。 |

主線舊幾何遮罩版本留於原處供封存。主表既有數值欄的幾何更正屬第 12 項；本項不改寫任何 CSV。

## 不可在合併時改變的科學協定

| 項目 | 既有定義 |
|---|---|
| 輸入影像 | `load_image_tensor()` 轉 RGB，輸出 `(1, 3, H, W)`、`[0, 1]`；明給尺寸時以 torch bicubic、`antialias=True` 縮放並 clamp。四個流程均使用 512。 |
| 影像存檔 | detach、轉 fp32、clamp、乘 255、round、uint8；不更換量化方法。 |
| 編輯設定 | `EDIT_STEPS=50`、`EDIT_SEED=IP2P_SEED=20260812`；IP2P 編輯 image guidance 為 1.8；文字 guidance 預設 7.5，與 IP2P 指定 seed 的覆寫規則依原入口保留，inpainting guidance 為 7.5。不可混用 adapter 的 `IP2P_STEPS=100` 與 `IP2P_IMAGE_GUIDANCE=1.5` 預設。 |
| 模型精度 | 活動編輯流程顯式使用 float32；通用 adapter 的 fp16 骨幹搭配 fp32 VAE、bf16 搭配 bf16 VAE 規則保留。 |
| 淨化 registry | 依序為 identity 0、crop_resize 0.1、jpeg 30／50／80、blur 1／2、rotate 15。包含 identity 對照與七道淨化，不增加或刪減算子。 |
| 主體遮罩 | `subject_mask(repaint) = 1 - (repaint >= 0.5).float()`；輸入白色表示重繪，輸出 1 表示主體。 |
| 幾何遮罩 | 先轉主體極性，再套 `Purifier.evaluate()`；crop 使用既有 bicubic，rotate 使用既有雙線性與補零，最後以 `>= 0.5` 二值化。非幾何算子直接保留遮罩。旋轉黑角歸入背景。 |
| LPIPS 後端 | 全圖及分區共用 `MetricSuite.lpips_module` 的 `piq.LPIPS` 權重；分區以 `adaptive_avg_pool2d` 降採樣遮罩，空區域報錯。不可換為另一個 LPIPS 套件。 |
| displacement 配對 | 未防禦 arm 與防禦 arm 使用相同 scenario、image、prompt index；缺任何一側影像立即失敗。預設未防禦 arm 為 `ip2p_si18` 與 `inpaint_undefended`。 |
| retention 配對與輸出 | 兩側都經同一淨化；`net_gain = plain - purified`，`retained = purified / plain`；plain 為 0 時 retained 留空。既有欄名、rounding 與 SigLIP 門檻 0.837 不變。 |

## 合併前的驗收案例

後續 pipelines 實作須驗證：只有子集防禦圖時可執行；防禦圖歧義與缺檔拒絕；條件過濾只處理指定條件；crop／rotate 後遮罩位置及黑角極性正確；未指定條件時的掃描集合不變；配對缺側拒絕；空區域與零分母維持原行為；既有輸出鍵、順序及精度保持一致。

淨化階段已建立 `pipelines.purification`，沿用相同 registry 與運算；主體極性與 `purified_mask()` 原樣抽至 `pipelines.masks`。已提供顯式產物根目錄。

編輯、displacement 與 retention 主流程依 lab 版本移入 `pipelines.editing`、`pipelines.displacement`、`pipelines.retention`，來源雜湊見 [pipeline_source_manifest.json](pipeline_source_manifest.json)。三者的 `--data` 改為必填，不再由 `paths.py` 推定資料根；`--defended` 的缺圖與歧義由 `artifacts.layout.defended_image()` 以 `ValueError` 拒絕。上列驗收案例由 `tests/test_edit_pipelines.py` 以替身指標在 CPU 上檢驗；原呼叫端未切換。
