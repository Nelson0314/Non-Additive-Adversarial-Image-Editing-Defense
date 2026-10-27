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
- 上傳 Overleaf 時壓縮 `docs/paper/`（排除 `build/`）。
