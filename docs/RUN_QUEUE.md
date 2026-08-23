# 待跑清單

**全部在 NYCU BASIC lab 跑。** 卡是多人共用的，**每卡最多 2 個 process**
（`OPERATIONS.md`）；所有掃描腳本都有 `require_slots`，工作點數超過
「卡數 × 2」會直接拒絕啟動並告訴你需要幾張卡。**卡號一律由參數給。**

順序是按「能改變結論的程度」排的，不是按成本。

---

## 第一優先 · 完成 JPEG 的頭對頭

**它決定本方法有沒有任何一個軸贏得了 DCT-Shield**，理由與背景見
`PENDING.md` 第三節。防禦圖已經跑好了（`runs/ip2p_dct_antijpeg`，七點全部
13/13），**只剩抗淨化那一半**。

失真帶內唯一可用的對手是 **`y_q85_e10`**（DISTS 0.1043 / PSNR 24.27 /
11-of-13）；其餘六點的 PSNR 全部出界，不要拿來比。

```
bash scripts/purify_antijpeg.sh "<卡號>" "y_q85_e10"
```

沒有那支腳本的話，等價於：

```
scripts/phase_retention.py --run runs/ip2p_dct_antijpeg/y_q85_e10 \
    --data data/omniedit150 --attacker ip2p --seeds 3 \
    --purifiers identity blur1 jpeg75 jpeg30 crop_resize0.1 jpeg_then_resize75 adverse_cleaner \
    --images <該分片的影像> --out runs/ip2p_purify_headtohead/dct_y_q85_<分片>.csv
```

**必須分片**（`color` 5 張 / `scene` 3 張 / `object` 5 張），分片名只能是
這三個或 `all`——`retention_table.py` 的 `tag_of()` 由檔名還原條件標籤。

**空白地板要齊。** 目前只有 6/13（`runs/ip2p_purify_headtohead/README.md`），
淨增益是逐圖相減，缺地板的格子算不出來。地板與條件無關，跑一次共用，
**也要分片**。

---

## 第二優先 · 補完 `floor_only`（我的腳本 bug 弄掉了四點）

只有加性下限、相位與幅度都不動。**這是加性裁決底下唯一沒跑過的對照**
（`DECISIONS.md` 撤回「不做加性項」時所依據的證據，是一次性探針裡
`radius 0.1` 的近似，程式已刪除）。

已完成：`f0060`（DISTS 0.0612 / PSNR 28.51 / 位移 0.6048 / 7-of-13）、
`f0090`（0.0900 / 25.74 / 0.6538 / 10-of-13）。
缺 **0.02 / 0.04 / 0.14 / 0.20** 四點——四點配兩張卡剛好。

```
bash scripts/floor_only_sweep.sh "<兩個卡號>"
```

跑完之後與 `runs/ip2p_axis_necessity/b_ph_*`（相位＋下限）做等失真內插：

```
scripts/matched_distortion_table.py --run runs/ip2p_floor_only/*/ \
    --strength spectral_floor --anchor 0.09 0.128 --out <輸出>
```

**注意 `--strength spectral_floor`**：這一批的 radius 整批相同，用預設的
`radius` 分桶會把整條曲線摺成一個點。

---

## 第三優先 · 幾何 EOT 的抗淨化

`runs/ip2p_eot_geometry` 三點已完成，但**未淨化的代價已經量出來很高**：
等失真下位移掉 18–20%、擋下率掉超過一半。它要值得，`crop_resize0.1` 的淨
增益必須從 +0.0360 升到足以補回那 18%。

```
scripts/phase_retention.py --run runs/ip2p_eot_geometry/g_rpi ...（同上分片）
```

**只跑 `g_rpi` 一個點就夠判斷**——若它在 crop 上沒有明顯改善，另外兩個更弱
的點不必跑。

---

## 第四優先 · AdvDrop（從未跑過抗淨化）

它是「統計驅動 vs 實現驅動」那個假說的判定實驗，因為它站在對角線上：與
DCT-Shield 共用 8×8 格點、與本方法共用非加性。

| | 加性？ | 格點 | 裁切保留 |
|---|---|---|---|
| DCT-Shield | 加性 | 8×8 | 98.2% |
| **AdvDrop** | **非加性（量化丟資訊）** | **8×8** | **要量** |
| 本方法 | 乘法＋加性下限 | 32×32／hop 8 | 13% |

**若它抗裁切，決定強健度的是統計結構而非加性與否；若它不抗裁切，該假說垮。**
兩種結果都有用。

先跑防禦（`--conditions advdrop --advdrop-eps`，掃 60 / 100 / 150 / 220 四點
把失真帶夾在中間），挑帶內的 1–2 點再跑抗淨化。

---

## 常規調參（有機時再跑，不要排在前四項之前）

| 腳本 | 點數 | 問什麼 |
|---|---|---|
| `band_lower_bound_sweep.sh` | 5 | `r_min` 離開 0.12（FND-042 說該離開） |
| `pricing_power_sweep.sh` | 4 | `--freq-weight-power` 0.25 → 0.35 |
| `theta_budget_sweep.sh` | 5 | 幅度相依的相位上限（arXiv:2602.06577） |
| `quantile_sweep.sh` | 6 | 空間選擇性單獨的效果 |

`gain_reach_extension.sh` 與 `floor_gate_sweep.sh … complement_rank` **已完成**
（斷線前跑完，資料已入庫）。

---

## 讀數字之前必須知道的三件事

1. **主讀數在 0.70 附近飽和**（`PENDING.md` 第〇節）。52 個工作點的位移最大
   值是 0.7078，前十名擠在 1.2% 內。**該區間比位移或比擋下數都不可解讀**，
   唯一還有鑑別力的軸是失真。
2. **比較一律等失真或等效果**，用 `matched_distortion_table.py` 內插。固定
   強度下改旗標只是沿強度軸移動——先前有三個結論這樣出錯過。
3. **CLIP 擋下數有 ±2/13 的重跑雜訊**，差異小於它時要看位移。跨方法的擋下率
   一律人眼定案，代理只用於條件內排序。

## 結果往哪裡放

`runs/` 底下，加一份 README 說明完成度。**不要動 `RESULTS.md`、
`DECISIONS.md`、`METHOD.md`**——新數字進 `PENDING.md`，等使用者裁定。
