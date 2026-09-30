# 文件

| 要查什麼 | 文件 |
|---|---|
| 逐篇逐行的原始碼查證 | `reference/SOURCE_AUDIT.md` |
| 各 baseline 的出處與預算換算 | `reference/BASELINE_PROVENANCE.md` |
| 等失真對齊 | `reference/BASELINE_ALIGNMENT.md` |
| 七道淨化算子的查證 | `reference/AUDIT_PURIFIERS.md` |
| 單一方法的查證 | `reference/AUDIT_<方法>.md` |
| 指標定義與比較方式 | `EVALUATION.md` |
| 外部文獻 | `reference/BIBLIOGRAPHY.md` |

以上文件複製自 `archive/anti-purification/docs/`，內容未改；其中對 `src/`、`scripts/`、`runs/` 等舊位置的引用
指向封存的原專案 `archive/anti-purification/`。`third_party/ultraedit/pipeline.py` 移植自 `HaozheZhao/UltraEdit` 的
`diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_instructpix2pix.py`，
來源版本為 commit `70e8ce5bc3bc8a6a02e1b9e0b6a1eb0058d98bc2`（該檔唯一的變更 commit；移植時 `main` 為
`1af5f0478d56d62cf6f35781a2ab63e03d656b2d`，兩者之間該檔未變動）。逐行比對結果：除 import 改為
`diffusers.` 絕對路徑與移植註解外，與來源一致。
