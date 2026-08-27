# 分階段訓練：階段二 ＋ 信賴域

**篩選批，三張影像，探索性質。** 本目錄是數值記錄，影像不入版控。
程式在 `src/defense/param_pgd.py`（`run_stage2_pgd`）與
`src/defense/purify_aware.py`（`stage2_schedule`／`make_sequenced_ops_transform`），
派工在 `scripts/stage2_screen.sh`，判準逐字寫在該腳本的檔頭。

## 一、這一批把哪一條已否決的路重開了，以及改動在哪

`docs/RESULTS.md` 的「針對淨化最佳化沒有改善抗淨化」已否決三個變體（固定
JPEG75／課程排程／多算子 EOT）：四個算子都沒超過基準，而**未淨化強度掉
10–25%**。幾何 EOT 另跑過一次（`ip2p_purify_headtohead` 的 `g_rpi`）：crop
淨增益 0.1203 是全表最高、首次超過 DCT-Shield 的 0.1098，但未淨化位移由
0.6710 掉到 0.5741，**失去 0.0969 換回 0.0298**。

兩批的死法相同：**沒有任何東西擋住未淨化那一側往下掉**。本批只改兩處：

| | 已否決的那三個變體 | 本批 |
|---|---|---|
| 起點 | 從零開始，全程在淨化底下練 | **階段一先照現行主線練完**，階段二以它為起點 |
| 未淨化那一側 | 無約束，實測掉 10–25% | **信賴域**：每 20 步檢查，低於階段一增益的指定倍數就退回快照並把步長減半 |

信賴域量的是**增益**而不是損失的絕對值：`gain = L(原圖) − L(防禦圖)`。
`latent_norm` 的值域與影像有關且可正可負，比值沒有意義；增益對任何參數化
都取得到（`θ = 0` 時 `render` 就是原圖）。交出去的一律是**最後一個通過檢查
的快照**，所以結果必定滿足約束。

**使用者已就本探索批次明確授權。結果不論好壞都不會自動用來推翻或恢復
`RESULTS.md` 上那條裁定**——那條是對「從零開始的擴增訓練」下的。

## 二、算子為什麼依序輪替

`--stage2-order shuffle`（預設）每一輪把算子清單洗牌再依序走完。

- 不用 `random`（每步獨立抽）：只跑幾百步時短期覆蓋不均，某個算子連著被抽
  好幾次而另一個一次都沒抽到，方向會被偏掉。
- 不用 `cycle`（固定輪替）當預設：覆蓋均勻了，但每一輪的最後一個算子恆是
  「最後說話的那個」，配上 sign 更新（只看梯度方向不看大小）容易走成週期性
  的來回。

`--stage2-ramp`（由弱到強，前半段只用強度階 ≤ 1 的算子）已實作但**本批關閉**，
以免與順序這個變因混在一起。

## 三、三個工作點

| tag | 階段二步數 | 步長比例 | 信賴域 | 算子池 | 回答什麼 |
|---|---|---|---|---|---|
| `s2_tight` | 400 | 0.2 | 0.95 | 六個 | 守得很緊時還買不買得到東西 |
| `s2_hard` | 800 | 0.5 | 0.80 | 六個 | 放寬到允許賠兩成時買得到多少 |
| `s2_null` | 800 | 0.5 | 0.80 | **只有 identity** | **歸因對照**：多跑 800 步本身值多少 |

`s2_null` 與 `s2_hard` 逐字相同，只有算子池不同。**沒有這一格，不管結果是好
是壞都歸因不清**——第二段多跑的步數本身就可能讓數字變好。

兩個有算子的點合起來給的是一條**取捨曲線**（未淨化賠了多少 vs 淨化後賺了
多少），不是一個點；幾何 EOT 那次只有一個點，回答不了「這筆帳有沒有可能在
別的鬆緊度上打平」。

## 四、階段一的參照

階段一逐字等於 `scripts/mainline_defense.sh` 的 `ours_pg_m`
（`--conditions phase_gain --gain-ratio 1.0 --radius 2.0`、含加性下限）。
十張的既有讀數：DISTS 0.14528、位移 0.70715、blur1 淨增益 0.12099、
crop_resize0.1 0.10423（`runs/ip2p_mainline/tables/`）。

**S2／S3 的比較對象不是那個十張平均**，而是 `ours_pg_m` 限制到本批這三張的
子集，由 `runs/ip2p_mainline_purify/ours_pg_m_{color,object}.csv` 的逐圖列
重算。不同 n 的列不可並列（`docs/PENDING.md` 已記）。

## 五、三張影像怎麼選

`images3.txt`。規則是**每個任務型態取字母序第一張**，取色／景／物三型：
`task_attr_mod_color_11699`、`task_env_weather_112463`、
`task_obj_swap_joint_mask_276754`。`task_obj_remove_*` 不入選是因為移除型的
指令服從率最低（`DECISIONS.md`：移除物件 1/5），三張的批次不該把一格花在
最不穩的型態上。

**這是篩選批，只回答「有沒有抬升的跡象」。** 有跡象才擴到十張。

## 六、逐圖記下的欄位

`results.csv` 除了設定欄（`stage2_steps`／`stage2_ops`／`stage2_order`／
`stage2_step_scale`／`stage2_trust`／`stage2_check_every`／`stage2_ramp`）
還有結果欄：

| 欄 | 是什麼 |
|---|---|
| `stage2_reverts` / `stage2_checks` | 退了幾次 / 檢查了幾次 |
| `stage2_alpha_init` / `stage2_alpha_final` | 步長從哪裡到哪裡 |
| `stage2_gain_stage1` / `stage2_gain_final` | 階段一買到多少 / 最後守住多少 |
| `stage2_gain_ratio` | 守住的比例，由構造 ≥ `stage2_trust` |
| `stage2_stopped_early` | 步長掉到初始值的 1/32 以下而提前停 |

**這些不記下來的話「沒有改善」與「安全繩一路咬著不放」在報表上長得一模一樣。**

## 七、重跑

```bash
bash scripts/stage2_screen.sh "<卡號>"                 # 防禦（三點）
bash scripts/stage2_screen_purify.sh "<卡號>"          # 抗淨化（模糊與裁切）
```

空白地板不重跑：地板那一格的「防禦圖」就是原圖，只與（影像, 算子, 種子）
有關，本批三張是十張的子集，`runs/ip2p_mainline_purify/floor_*.csv` 已經有。
出表時 `retention_table.py --src` 收多個路徑，把兩個目錄一起餵。
