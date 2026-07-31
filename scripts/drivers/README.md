# 歷史實驗驅動腳本

這八個 shell 腳本是 E7–E12 實際執行時用的驅動，原本只存在於 TWCC 的
`/work/nelson0314/`，從未入庫。2026-07-31 收回，用途是**保留參數的出處**：
各實驗報告裡的 `--lr`、`--steps`、`--limit` 等設定，在此可以查到當初實際下了什麼。

| 檔案 | 對應實驗 |
|---|---|
| `stage1.sh` | E7 步數掃描（site P，20 與 100 步） |
| `stage2.sh` | E7 步數掃描（site LA） |
| `expA.sh` | E8 全秩 vs 低秩，三個 LPIPS 預算 |
| `expB.sh` | E9 對齊容量探測 |
| `expC.sh` | E10 EOT 梯度平均 |
| `stage0w.sh` | E11 權重空間 LoRA 學習率 |
| `stage1L.sh` | E12 site L 對齊重跑 |
| `stage2W.sh` | E12 site W 對齊 |

腳本內的路徑是容器上的絕對路徑（`cd /work/nelson0314/WACV`、`source env.sh`），
**未改寫**——改了就不是當初實際執行的那份了。要重跑請對照
`scripts/twcc_env.sh`（即當時的 `env.sh`）自行調整路徑。

本次 session 的驅動（E18、E19）已直接以可執行形式放在 `scripts/`：
`e18_sweep.sh`、`e19_lam.sh`。
