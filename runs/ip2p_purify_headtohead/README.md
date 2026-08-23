# 抗淨化的頭對頭：**未完成的批次**

**這些數字尚未進入 `docs/RESULTS.md`**，而且**這一批沒有跑完**——NFS 在
批次中途掛掉。以下逐格說明缺哪裡，讀之前先看清楚。

## 完成度

| 條件 | 防禦圖來源 | 影像 | 狀態 |
|---|---|---|---|
| `ours_nonadd` | `runs/ip2p_overlap_sweep/h08_r25` | 13/13 | 完成 |
| `ours_add` | `runs/ip2p_axis_necessity/b_pg_r20` | 13/13 | 完成 |
| `dct_e14` | `runs/ip2p_band_calibration/dct_e14` | 13/13 | 完成 |
| `dct_e18` | `runs/ip2p_dct_band_extend/dct_e18` | 13/13 | 完成 |
| `dct_y_e14` | `runs/ip2p_dct_band_extend/dct_y_e14` | 12/13 | 差 1 張 |
| **空白地板** | 原圖本身 | **6/13** | **這是瓶頸** |

**淨增益只算得出有地板的那 6 張。** 地板逐圖差好幾倍（同一個模糊在不同影像
上推開的量不同），所以相減必須逐圖做，缺地板的格子由
`scripts/retention_table.py` 排除並回報，不補零。

`floor_all.csv` 是最早那個**沒有分片**的 process 留下的（一個 process 要跑
13 張，而其餘每片只跑 3–5 張，整批的完成時間會被它決定，實測差四小時）。
它被中止並改成三片重跑，`floor_color/scene/object` 就是那三片。兩者有重疊的
影像，`retention_table.py` 的 dedupe 只留一份。

## 兩個必須知道的缺口

1. **GrIDPure 整格缺席。** `DIFFPURE_CKPT` 只寫在 `~/env.sh`，而驅動一律不
   source 它（那支最後一行會把工作目錄換到舊樹），於是 `Purifier.available`
   判它「相依不齊」而**靜默跳過**，只留一行日誌。旗標已補進驅動，補跑腳本是
   `scripts/purify_gridpure.sh`，但那一批也被 NFS 中斷。
2. **`dct_y_e14` 用的是 `--q-alg 0.95`，不是論文的抗 JPEG 設定。** DCT-Shield
   的抗壓縮保證是單向的（補充材料 D.4）：`Q_alg = q` 只在攻擊方壓縮品質
   `q' >= q` 時有效，而 §6.3 圖 6 的 Y-only 變體用的是 **0.85**。當時選 0.95 的
   理由是「讓兩支之間唯一的差別是通道集合」，那對畫失真曲線是對的，對抗淨化
   是錯的。**在 `scripts/dct_antijpeg_configs.sh` 跑完之前，本方法在 JPEG 上的
   優勢不可引用**——它是打在一個論文自己說擋不住 JPEG-75 的設定上。
   `docs/RESULTS.md` 早已記過同一件事（SDEdit 線上 `dct_shield_y` 在 jpeg75
   拿 +0.5185、本方法 +0.1349）並明寫「頭對頭表不可只放 base 變體」。

   補這一格要**兩步**：`scripts/dct_antijpeg_configs.sh` 產生 q_alg 0.85／
   0.75 的防禦圖，再由 `scripts/purify_antijpeg.sh "<卡號>" "<tag> ..."` 把
   那些工作點當條件加進抗淨化。第一步只產生防禦圖，不做第二步的話 JPEG 那
   一欄仍然是空的。第二步的輸出落在 `antijpeg/` 底下，地板沿用上一層的
   `floor_*.csv` 不重跑；`retention_table.py --src runs/ip2p_purify_headtohead`
   會遞迴走進 `antijpeg/` 與 `gridpure/`。

## 目前算得出來的（6 張，扣地板的淨增益）

`net_gain_partial.csv`。`ours_add` 與 `dct_e18` 的未淨化位移幾乎相等
（0.683 對 0.696），故這一格接近等效果比較。

| 條件 | adverse_cleaner | blur σ=1 | crop_resize | 未淨化 | jpeg30 | jpeg75 | C&R 串接 |
|---|---|---|---|---|---|---|---|
| `dct_e14` | +0.2992 | +0.1397 | +0.0942 | +0.6840 | +0.0198 | +0.0936 | +0.0268 |
| `dct_e18` | +0.3592 | **+0.1623** | **+0.1052** | +0.6962 | +0.0333 | +0.1822 | +0.0489 |
| `dct_y_e14` | +0.2622 | +0.0248 | +0.0558 | +0.5381 | +0.0184 | +0.0736 | +0.0163 |
| **`ours_add`** | **+0.3870** | +0.1295 | +0.0847 | +0.6828 | **+0.1510** | **+0.3543** | **+0.1310** |
| `ours_nonadd` | +0.3179 | +0.1353 | +0.0486 | +0.5837 | +0.1411 | +0.2564 | +0.1268 |

方向與既有紀錄一致並延伸到三個從未對現行條件測過的算子（jpeg30、C&R 串接、
adverse_cleaner）：JPEG 家族是強項、模糊與裁切縮放是弱項。**但只有 6 張，且
DCT-Shield 的抗 JPEG 設定沒進來，所以這張表只能當方向不能當結論。**

`crop_resize` 那一欄還要再加一層保留：該算子的空白地板是 0.546，而兩邊的淨
增益都只有 0.08–0.11，**地板是訊號的五倍**。為什麼裁切這麼難擋，見
`runs/ip2p_residual_signature/README.md` 的最後一節。
