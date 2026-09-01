# Image Immunization

在照片發布之前對它做一次處理，交出一張看起來一樣的圖，使得攻擊方用公開權重的
擴散編輯模型照指令改圖時會失敗。威脅模型是**白盒**（防禦方已知攻擊方的模型）、
**外掛模組**（不改動攻擊方的權重）、**匹配失真**（任何強度比較先把失真對齊）。

## 目錄

| 目錄 | 內容 |
|---|---|
| [`anti-purify/`](anti-purify/) | **現行階段。** 入口是 [`start.md`](anti-purify/start.md)，文獻庫在 [`reference/`](anti-purify/reference/) |
| [`non-additive-frequency/`](non-additive-frequency/) | **已告一段落。** 頻域／相位重參數化的完整程式、文件與數值記錄 |

新的工作從 [`anti-purify/start.md`](anti-purify/start.md) 開始讀——它統整了做過
什麼、什麼成立、量測協定、專案規定與機器使用，並逐處標出證據在封存目錄的哪裡。

## 前一階段留下什麼

**成立的**：紋理重相位（頻譜重參數化）在**等失真**的頭對頭上，未淨化輸給
DCT-Shield、但 **JPEG 淨化之後大幅領先**——協定完全對稱的那一組 JPEG 75 是
3.4 倍、JPEG 30 是 4.3 倍。

**不成立的**：模糊。高斯模糊在頻域乘的是實正數，**構造上不可能改相位**，
它抽掉的是載體；σ=1 之後擾動的能量存活率只剩 0.169，σ=2 剩 0.061。
十二條路只有一條動得了它，而且只有兩成。

**被誤判為失敗、其實不是的**：裁切。換過參照之後（兩側吃同一個算子、
空白地板由構造為 0）它是六個算子裡保留最好的一欄。擾動原封不動地通過了裁切，
只是位置被搬走而沒有人替它對回去——那是**同步失效不是破壞**。

## 資料保全

`runs/` 只保存**數值記錄**（CSV / JSON / log / txt / md），一律入版控——量測
結果不可重現。**影像與 HTML 報告不入版控**，它們能由已記錄的參數與種子重跑。

## 環境

本機 Python：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（不是 base）。
GPU 工作在 NYCU BASIC lab 跑，**卡是多人共用的**，派工前一律
`bash scripts/free_cards.sh --assert "<卡號>"`。細節見
[`anti-purify/start.md`](anti-purify/start.md) 第九節。
