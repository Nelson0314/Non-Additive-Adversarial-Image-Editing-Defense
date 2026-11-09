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
| `generate_main_table.py` | Table 1（`3_method.tex` 的 `tab:main`，含 VIBE-DistilledCFG、FLUX.2 [klein] 區塊） |
| `generate_per_purification_table.py` | 補充材料逐道淨化表（`sec/X_suppl_purification.tex`） |
| `generate_overleaf_ops.py` | Overleaf 同步：由「Overleaf 現行文字」與「repo 文字」產生詞級修改清單（JSON）與前後雜湊值 |

## Overleaf 同步

Overleaf 專案（`6ad193f8288983a2f6af5d50`）與 `docs/paper/` 須保持逐字一致。線上修改一律在 Overleaf 編輯器內進行，以保留編輯歷史；不上傳、不以檔案覆蓋。

1. 取得 Overleaf 現行文字：使用者下載專案 zip（解壓到 scratchpad 的新目錄），或以三方合併（`git merge-file`，基底為 Overleaf 上傳時的 commit）處理雙方修改。
2. `python docs/paper_tools/generate_overleaf_ops.py <Overleaf 檔> <repo 檔> ops.json`，記下前後雜湊值。
3. Claude in Chrome（使用者先以 `/chrome` 連線）開啟專案、點選該檔，於頁面執行下列程式碼，先比對 `__hash()` 等於「前」雜湊，再套用 `ops.json`，結果應等於「後」雜湊：

```js
window.__view=()=>[...document.querySelectorAll('.cm-content')].find(e=>e.offsetParent&&e.cmView).cmView.view;
window.__hash=()=>{const s=window.__view().state.doc.toString();let h=0x811c9dc5;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,0x01000193)>>>0;}return [h,s.length];};
window.__applyOps=(data)=>{const v=window.__view(),d=v.state.doc;if(d.length!==data.len)return{err:'len'};
 for(const [o,del] of data.ops){if(d.sliceString(o,o+del.length)!==del)return{err:'mismatch at '+o};}
 v.dispatch({changes:data.ops.map(([o,del,ins])=>({from:o,to:o+del.length,insert:ins})),userEvent:'input'});return window.__hash();};
```

4. 等數秒後重新載入頁面，再比對一次雜湊，確認已存到伺服器。
5. 擴充功能會擋下看似網址參數的頁面輸出（`[BLOCKED: Cookie/query string data]`），此時改以截圖閱讀，不改寫輸出繞過。
