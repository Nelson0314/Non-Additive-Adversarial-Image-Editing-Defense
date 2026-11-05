# 論文預覽工具

論文原始檔只在 `docs/paper/` 修改；桌面 `C:\Users\nelso\OneDrive\桌面\paper.pdf` 為最新一次成功編譯的 PDF 副本。

## 安裝

Tectonic 0.17.0（`x86_64-pc-windows-msvc`，github.com/tectonic-typesetting/tectonic 的 release）的 `tectonic.exe` 放在 `C:\Users\nelso\bin\`（已在 PATH）。
套件快取於 `%LOCALAPPDATA%\TectonicProject\Tectonic`；首次編譯需網路，約 4 分鐘，其後每次 5–20 秒。參考文獻由 Tectonic 自動呼叫 BibTeX 處理。

## 使用

```powershell
python docs/paper_tools/run_paper_preview.py            # 編譯一次，失敗時結束碼為 1
python docs/paper_tools/run_paper_preview.py --watch    # 原始檔變更時重新編譯
```

- 輸出 `docs/paper/build/main.pdf`（不入版控），成功後複製到桌面 `paper.pdf`；失敗時印出 TeX 錯誤行，兩處保留前一份 PDF。
- 成功時列出未定義的引用與標籤。
- `main.tex` 內 `\ifdefined\XeTeXrevision` 區塊只在 Tectonic（XeTeX）下以 TeX Gyre 字型取代 `times`；pdfLaTeX 不執行該區塊。
- 上傳 Overleaf 時上傳整個 `docs/paper/`（`build/` 不需要）。

## 產圖腳本（遠端 `~/paper_figures/` 執行，輸出複製到 `docs/paper/fig/`）

| 腳本 | 產物 |
|---|---|
| `generate_method_overview_panels.py` | 方法總覽圖（Fig. 2）的資料圖塊，複製到 `method_overview/` 後以 Tectonic 編譯 `method_overview/method_overview.tex` |
| `import_pexels_figure_inputs.py` | 把 baseline 的 Pexels 讀數轉成 `generate_paper_figures.py` 的輸入 |
| `generate_paper_figures.py` | 失真—位移圖（`distortion_displacement.pdf`）、淨化分析圖（`purification.pdf`） |
| `generate_pexels_teaser_figure.py` | Fig. 1 teaser（`teaser.pdf`）與候選排序（`--rank`）；CAPGuard 以 `--scale2p5` 取 2.5 倍防禦圖 |
| `generate_teaser_overview.py` | 把多組 teaser 候選 PNG 疊成一張總覽圖（選圖用，不入論文） |

本機產生表格（讀遠端抓回的 CSV，寫入 `docs/paper/sec/`）：

| 腳本 | 產物 |
|---|---|
| `generate_main_table.py` | Table 1（`3_method.tex` 的 `tab:main`，含 UltraEdit、FLUX 區塊） |
| `generate_per_purification_table.py` | 補充材料逐道淨化表（`sec/X_suppl_purification.tex`） |
