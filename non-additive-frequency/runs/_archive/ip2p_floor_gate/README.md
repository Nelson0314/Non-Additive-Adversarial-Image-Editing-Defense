# 加法下限的價目分配：均勻 對 內容相依

**這些數字尚未進入 `docs/RESULTS.md`**，等使用者裁定。

含加性下限那個主線設定的加法項，價目只看頻格、跨區塊是常數——**那正是
DCT-Shield 的形狀**（逐係數 `ε·Q(ω)`，`Q` 只看頻率）。這一批問的是：把同樣的
預算改成內容相依的分配，防禦強度會不會掉。

三個分配（`--floor-gate`，`src/residual/texture_rephase.py` 的 `FLOOR_GATES`）：

| 值 | 是什麼 |
|---|---|
| `uniform` | 現行。逐位元等於加這個旗標之前，由測試釘住。本目錄沒有它的資料，對照組是 `runs/ip2p_axis_necessity/b_pg_r*`（同旗標、同半徑） |
| `complement` | `1 − reach_b / max reach_b`，`reach_b` 是乘法那一半在該區塊碰得到的幅度總量 |
| `complement_rank` | 同上但用名次，低可達量的一半固定拿 75%（`complement` 只到 51–62%，因為可達量是重尾分布） |
| `watson` | 亮度遮蔽 × 對比遮蔽（Watson 1993；Podilchuk & Zeng 1998 的用法） |

**三者的總預算相同**：價目表被正規化到與 `uniform` 同一個平均值，故差異是
「預算花在哪裡」而不是「花多少」。半徑仍要掃，等失真的比較由
`scripts/matched_distortion_table.py` 內插。

其餘旗標與含加性下限的主線相同：`--conditions phase_gain --loss latent_norm
--steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25
--hop 8 --gain-ratio 1.0 --spectral-floor 0.04`。13 張人眼確認服從的影像。
驅動是 `scripts/floor_gate_sweep.sh`。
