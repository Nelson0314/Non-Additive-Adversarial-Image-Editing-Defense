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
| `generate_method_overview_panels.py` | Fig. 1 的資料圖塊，複製到 `method_overview/` 後以 Tectonic 編譯 `method_overview/method_overview.tex` |
| `import_pexels_figure_inputs.py` | 把 baseline 的 Pexels 讀數轉成 `generate_paper_figures.py` 的輸入 |
| `generate_paper_figures.py` | Fig. 2（`distortion_displacement.pdf`）、Fig. 3（`purification.pdf`） |
| `generate_teaser_figure.py` | teaser（暫時移出論文）與同版面的例子 |

`experiment_overview/` 為內部用實驗總覽，於該目錄以 Tectonic 編譯後複製到桌面。
