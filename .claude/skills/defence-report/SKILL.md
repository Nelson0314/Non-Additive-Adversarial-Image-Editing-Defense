---
name: defence-report
description: 把防禦批次的產物與讀數做成逐張看圖的報告頁（Artifact）。當使用者要「報告」「看圖」「把結果整理出來」，或一批 immunise/evaluate 跑完要交付時使用。涵蓋顏色載體與位移場載體兩條線。
---

# 防禦批次的報告頁

**自然度只能由使用者的眼睛判定**——NIQE、平均色差、CVaR 尾端三道自動門檻都被
最佳化鑽過（見 `non-additive-frequency/docs/EVALUATION.md`）。所以這份報告的
工作是把**圖**和**數字**擺在同一個地方，讓判斷做得下去；它不替使用者下
「成立／不成立」的結論。

## 一頁要有的六塊

依序，缺一塊就不完整：

1. **標題與一句話結論。** 標題是名字不是摘要（例：「網格間距與那張臉」）。
   一句話說清楚量到了什麼、還卡在哪。
2. **頭條數字。** 設定數、防禦圖數、真編輯的評估列數、最強的一檔、
   最強**且仍自然**的一檔、對照線。
3. **互動檢視。** 上方一排設定（依過半穿透由強到弱），右側選照片，
   主畫面三種檢視：防禦圖／原圖／差異 ×5.5，**按住圖片暫時切回原圖**。
   右欄分兩層讀數（見下）。方向鍵切換照片與設定。
4. **攻擊之後。** 同一格的未防禦編輯與防禦後編輯並排，可切指令類。
   **這一塊不可省**——CSV 上的一個數字說不出輸出是崩壞、是換了一個人、
   還是指令根本沒做到。
5. **機制與前緣。** 每一個發現配它的表；每一步要說清楚是被**前一步的圖**
   逼出來的，不是先想好的清單。
6. **誠實區。** 重跑變異、未評估的臂、對照來自別批的地方、有損編碼的地方。

## 兩層讀數要分開

| 層 | 來源 | 回答什麼 |
|---|---|---|
| 防禦圖本身 | `scripts/field_readout.py` | 這張照片還是不是這個人、還像不像照片 |
| 編輯輸出 | `scripts/evaluate_defence.py` → `summarise_screen.py` | 攻擊者拿不拿得到可用的結果 |

一個載體可能在第二層很強，只是因為第一層已經把照片毀了。兩層分開報就看得出來。

## 建置

```
python scripts/field_readout.py --root <batch>/by_variant --out <batch>/defended_readout.csv
python scripts/summarise_screen.py --root <eval_root> --threshold 0.5 --out <eval_root>/summary.csv
python scripts/build_field_report.py \
  --batch "<批次名>=<batch>/by_variant" ...（每批一次）\
  --defended <batch>/defended_readout.csv ...（每批一次）\
  --screen <eval_root>/summary.csv ...（每批一次）\
  --edits <eval_root> --edit-variants a,b --edit-classes hat,background,glasses \
  --out <scratchpad>/report
```

然後把 `img/*.png` 轉成**無損 WebP**（防禦圖要逐像素判自然度），
`data.js` 裡的 `.png"` 換成 `.webp"`，把 `assets/template.html` 複製成
`index.html` 並改掉敘事段落。編輯圖由建置腳本收成**有損** WebP（預設 q90），
理由寫在頁面上，不藏。

## 發佈的硬限制

- **總檔數 ≤ 256**（含 `index.html` 與 `data.js`），不是每次發佈的上限而是
  整個 artifact 的上限。超過會收到 `deploy 422: manifest exceeds 256 entries`。
- **每個版本 ≤ 64 MB。** 194 張 512×512 的無損 WebP 約 55 MB，所以編輯圖要挑
  （`--edit-variants`／`--edit-classes`）。
- `files` 參數用 **map 形式**（`{"published/path": "source/path"}`）。
  list 形式要的是物件不是字串。
- 更新時**只傳新增或改動的檔案**，沒傳的會保留。
- `favicon` 只在第一次發佈給，之後省略。

## 數字的規矩

- **並列時看過半那一欄**（`majority`）。`runs/purify_heldout/` 的顏色線
  18/18/15/16 用的就是這個規則，拿它跟 `penetrated` 比會錯。
- **跨批次的對照不成立。** 同設定三次獨立跑量到 13、13、1 與 5、1、4，
  來源是 GPU kernel 的非決定性。**每一批都要自帶對照**，報告裡要講。
- **門檻掃描照放。** 格數對切點敏感，結論不可以只掛在一個門檻上。
- 未評估、失敗、被守門擋下的臂**照實寫進報告**，不要靜默省略。

## 設計

`assets/template.html` 是現行版本，改敘事、不要重寫版面。

- 色票：`--paper #f2f4f7`／`--ink #0f151c`／`--signal #10559b`（藍圖藍），
  語意色 `--pass #1c6a42`／`--edge #8a5806`／`--fail #9f2620`。
  深色由 `@media (prefers-color-scheme:dark)` 與 `:root[data-theme="dark"]`
  各改一次 token，元件一律只用 token。
- 字：標題 Archivo、內文 Source Serif 4、數字 IBM Plex Mono（`tabular-nums`）。
- 表頭用網格底紋呼應控制網格；卡片不濫用，邊框與陰影按角色給。
