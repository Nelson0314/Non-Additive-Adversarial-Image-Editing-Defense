# 淨化模組與固定流程

來源基準 `51d58bf2757a691ec4d8c54a983efb74ac41bac8`。四個模組由 `anti-purification/src/purify/` 複製，`ops.py` 改名 `operators.py`，其餘為 `adverse_cleaner.py`、`diffpure.py`、`impress.py`。來源雜湊見 `purifier_source_manifest.json`。原來源與活動呼叫端不修改。

`Purifier.forward()` 保留訓練代理，`evaluate()` 保留真實算子；運算使用 fp32 再轉回輸入 dtype。JPEG 編解碼、中心 crop、Gaussian blur、rotate 的插值／補零／seed、IMPRESS 後端選擇、DiffPure 取樣式均未更換。IMPRESS 預設的 `lpips` 與顯式 `piq` 選項仍為不同實作，禁止缺依賴時自動替換。

固定淨化協定的唯一正本為 `purifiers/protocol.json`（identity 對照與七道淨化，原由 `lab/code/purify_run.py` 抽取）；`purifiers.protocol` 讀取為 `PURIFIERS`、`label()`、`purifier_labels()`，shell 以 `python -m immunization_core.purifiers.protocol --exclude-identity` 取得標籤。`pipelines.purification` 合併三份相同運算，保留明確的 `--data`／`--defended` 與 `--out`，CSV 欄位、影像命名、量化與 512 解析度不變。沒有自動探索來源路徑或初始化模型。

`pipelines.masks.subject_mask()` 由 `lab/code/edit_displacement.py` 抽取，`purified_mask()` 由 `lab/code/edit_retention.py` 抽取。先將重繪遮罩轉為主體遮罩，再套相同幾何算子與 `>=0.5` 二值化；旋轉黑角歸背景。非幾何算子直接回傳原遮罩。幾何種類仍為 crop_resize／rotate，不擴充既有協定。

`ArtifactLayout` 明確接收 dataset、artifacts、results 三個根目錄，僅固定路徑而不建立目錄。`defended_image()` 接受既有的 `<image>__def.png` 與 `<image>__<condition>__def.png`；匹配不等於一個時以 `ValueError` 拒絕。此為新 callable API；既有 CLI 的 `SystemExit` 不變，後續入口自行將驗證例外轉為退出訊息。

## 獨立的依賴邊界修正

1. 原 registry 對 `gridpure`／`fdpure` 延遲 import 缺失的 `src.purify.freq_grid`。core 保留標籤辨識，但 `available=False`，執行時明確 `NotImplementedError`；不讀 frequency-phase，不假裝已有實作。這兩者不在活動七道淨化中。
2. DiffPure 先檢查明確權重路徑，再 import guided-diffusion；檔案缺席時回報檢查點與 `DIFFPURE_CKPT`，不引用 core 不存在的下載腳本。
3. Adverse Cleaner 先執行專用 guided-filter 檢查，再 import cv2；整個 OpenCV 缺席時也有同一個明確依賴錯誤。數值本體與既有參數不變。

上述三項各自成 patch。真實外部後端／權重測試以 `weights` 標記選取；預設 CPU 測試不下載模型、不使用 GPU。移植測試內的缺依賴案例由 monkeypatch 明確構造，不依賴主機剛好未安裝套件。

本段只完成淨化與遮罩共用流程；編輯與兩種讀數主流程、lab 條件過濾接入、共用最佳化／色彩 helpers、GPU 租約工具仍待後續派工。
