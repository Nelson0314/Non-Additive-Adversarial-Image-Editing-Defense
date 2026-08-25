# 待跑清單

**全部在 NYCU BASIC lab 跑。** 卡是多人共用的，**每卡最多 2 個 process**
（`OPERATIONS.md`）；所有派工腳本都內建
`bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3`，指定的卡上只要有
別人的 process 就拒絕啟動。**卡號一律由參數給。**

順序是按「能改變結論的程度」排的，不是按成本。

**目前沒有任何工作在跑。** 十張主線頭對頭已收束，全部數值在
`runs/ip2p_mainline/`（含 README 與三張表），待裁定項在 `PENDING.md`。

單價（實測）：一次編輯 **46.8 秒**，故
「一個條件 × 一張影像 × 一個算子」＝ 3 種子 × 46.8 s ≈ **140 秒**。
可用槽位是 4 卡 × 2 = **8**。防禦圖每個工作點十張約 **7 分鐘**。

---

## 第一優先 · 補完等失真對照的 4 張

**它決定「低品質壓縮的優勢與預算無關」這句話能講到哪一格。**

`dct_aj85_eps2.2`（DISTS 0.1962，對本方法最強點的 0.1947 差 0.8%）是目前唯一
的等失真對手，抗淨化跑到 6/10 就隨整批撤回。就那六張重算，**jpeg50 那一格在
等失真下反轉**（eps2.2 0.4835 對本方法 0.3259，6/6 張全輸），jpeg30 仍是本方法
勝（0.2413 對 0.1818）。詳見 `runs/ip2p_mainline/README.md` 第五節。

十張補齊才能定案。`eps1.5` 同樣缺 4 張，一起補。

```bash
bash scripts/mainline_purify.sh "<卡號>" "dct_aj85_eps1.5 dct_aj85_eps2.2"
```

**成本**：2 條件 × 4 張 × 9 算子 × 140 s ≈ 2.8 h 計算，8 槽約 **40 分鐘**。
（腳本會重跑十張；只要 4 張的話用 `--images` 指定。）

---

## 第二優先 · 把品質旋鈕的最佳包絡補起來

**不補這一格，「對手把旋鈕設錯」是一句打不掉的質疑。**

現有的 DCT-Shield 全部是 `Q_alg = 0.85`。它的保證是單向的（補充材料 D.4），
預期被壓到 30 的防禦者照論文用法應設 `Q_alg = 0.30`。現有最低是 `dct_aj75`，
jpeg30 只有 0.0524（相對 aj85 的 0.0449 幾乎沒進步），是好跡象但不夠。

對稱地，本方法也有同一個旋鈕。`runs/ip2p_deliver_jpeg/` 的 `qd75` 族**只跑過
防禦、從未跑過抗淨化**，而 CVIU 2025（`reference/SURVEY_FREQUENCY.md` §1.13）
的作法是訓練品質 q = 35 遠低於攻擊品質 Q = 75，與我方目前的 0.85 方向相反。

要跑的四個防禦點：

```bash
# 對手側：Q_alg 拉到攻擊品質
bash scripts/mainline_matched.sh "<卡號>" ...   # 需先在 POINTS 加 q_alg 0.5/0.3
# 我方側：交付格點放粗
bash scripts/deliver_jpeg_sweep.sh "<卡號>" defense   # QD 0.6/0.45/0.35
```

**判準寫在前面**：看的是**曲線的形狀**不是單一格。低 QD 應該在 jpeg30 那端抬
起來、jpeg90 那端塌下去；六條曲線若只是整體平移沒有換形狀，這條路是死的。
失真必須拉回等失真再比（`scripts/matched_distortion_table.py` 內插）。

**成本**：防禦 4 點一批約 15 分鐘；抗淨化 4 條件 ＋ 地板，只留 JPEG 欄的話
5 算子 × 10 張 × 5 條件 × 140 s ≈ 9.7 h 計算，8 槽約 **1 小時 15 分**。

---

## 第三優先 · 把攻擊方的品質格點補密

交叉點落在 jpeg75 與 jpeg50 之間一大段空白裡（本方法在 75 輸、50 贏）。
`jpeg60`／`jpeg40`／`jpeg20` 已加進 `scripts/phase_retention.py` 的算子清單，
`scripts/mainline_purify.sh` 的 `PUR` 可由環境變數覆寫：

```bash
PUR="identity jpeg60 jpeg40 jpeg20" \
  bash scripts/mainline_purify.sh "<卡號>" "<tag> ..."
PUR="identity jpeg60 jpeg40 jpeg20" \
  bash scripts/mainline_purify.sh "<卡號>" "floor"     # 地板不可省
```

**兩個容易漏掉的成本**：空白地板要跟著跑（條件數 +1）；`identity` 每次都被迫
重算（`phase_retention.py` 要它當 retention 的分母），跑 3 個新品質實際是 4 個
算子。這是「一次把品質補齊」比「分批補」划算的原因。

**成本**：4 算子 × 10 張 × (6 條件 + 地板) × 140 s ≈ 10.9 h 計算，
8 槽約 **1 小時 22 分**。加到 8 條件是 **1 小時 45 分**。

---

## 不建議再花機時的方向

| 方向 | 為什麼停 | 證據 |
|---|---|---|
| `--coarsen`（θ 空間平滑） | 上機前的 CPU 探針否證了機制：JPEG 保留率在三種設定下全部微降 | `ip2p_coarsen/probe*.csv` |
| `dct_aj85_eps3.2` / `eps4.5` | 相對 eps2.2 只多 +0.0055 位移（±0.0156，10 張裡 6 張為正），多付 0.045／0.081 DISTS | `ip2p_mainline/tables/` |
| blur／crop 的專門機制 | 三個方法都沒有防禦，算子自己造成的位移是任何淨增益的 1.5–8 倍 | `ip2p_mainline/README.md` 第六節 |
| AdvDrop | 最佳化與自身機制互斥，>50% 的 DCT 格梯度方向相反 | `ip2p_advdrop_ceiling` |
| 光流／warp 位移場 | 使用者裁定放棄，不入文件 | — |

## 尚未動工但已查證的兩個旋鈕

`reference/SURVEY_FREQUENCY.md` §1.14–1.15。兩者都動方法本體，依規矩**先提
計劃討論再寫程式**。

1. **多品質集成損失**（Shin & Song 2017）。集成只能放在**損失**上，交付仍是
   單一格點——放到交付上就是已否決的 `--purify-aware`。成本：每步 K 倍前向，
   K = 5 時防禦單點由 7 分鐘變約 35 分鐘。
2. **θ 半解析度參數化**已實作為 `--coarsen`，但機制已被否證，見上表。
