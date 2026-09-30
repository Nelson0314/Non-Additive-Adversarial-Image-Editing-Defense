# 出處文件在主線目錄

這裡原本放了一份與 `../../docs/reference/` 重複的副本。
兩份已合併，**正本在主線目錄**，這裡不再保留第二份。

| 要查什麼 | 讀哪份 |
|---|---|
| 逐篇逐行的原始碼查證 | `../../docs/reference/SOURCE_AUDIT.md` |
| 各 baseline 的出處與預算換算 | `../../docs/reference/BASELINE_PROVENANCE.md` |
| 七道淨化算子的查證 | `../../docs/reference/AUDIT_PURIFIERS.md` |
| 單一方法的查證 | `../../docs/reference/AUDIT_<方法>.md` |
| 指標定義與比較方式 | `../../docs/EVALUATION.md` |
| 外部文獻 | `../../docs/reference/BIBLIOGRAPHY.md` |

合併時取的是**較新、已修正的那一份**：`AUDIT_PURIFIERS.md`（兩處引用改成
`SOURCE_AUDIT.md`，不是已作廢的帶日期檔名）、`BASELINE_PROVENANCE.md`（多出
`eps_pixel01` 的逐列單位一節，並更正「只有這一欄跨方法可比」的說法）、
`EVALUATION.md`（區分模組預設值與主表實跑的七道）、以及原本只在這裡的
`AUDIT_DCT_SHIELD.md`。
