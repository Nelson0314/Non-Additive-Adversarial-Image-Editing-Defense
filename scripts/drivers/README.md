# 實驗驅動腳本

分兩類：**待執行的**（下一節）與**歷史留存的**（其後）。兩者的寫法不同——
歷史腳本保留當初的絕對路徑不改寫，待執行的腳本以相對定位並允許覆寫直譯器。

## 待執行

執行順序固定。E29 沒有通過綁定者判準（每一格都是 LPIPS hinge）就不要開 E30：
被別道約束綁住的格子，τ 對它不起作用，「匹配失真的比較」不成立。

| 檔案 | 用途 | 成本 |
|---|---|---|
| `smoke_local.sh` | 本機 tiny-SD 走完「跑一格 → 綁定者診斷」，驗證參數組合 | 約 30 秒 |
| `remote_setup.sh` | 新雲端機器的環境準備、GPU 檢查、預抓 SD v1.4 權重 | 約 5 分鐘 |
| `e29_calibration.sh` | 加入色度約束後的學習率重新校準，8 格、60 步、`--no_eval` | 約 20 分鐘 |
| `e30_grid.sh` | 主網格 2 site × 3 τ × 6 圖 = 36 格 | 2–4.2 小時 |

`e30_grid.sh` 的學習率沒有預設值，必須以 `LR_C=` 與 `LR_P=` 傳入 E29 定出的值。
E27 定出的 0.3 / 0.03 是在還沒有色度約束的程式上量的，直接沿用等於假設
色度約束不影響解。

直譯器一律用絕對路徑（`PY=` 可覆寫）：Lightning AI 的背景腳本不是 login
shell，`python3` 會取到系統直譯器而缺 numpy。

## 歷史留存

這九個 shell 腳本是 E7–E12 與 E27 實際執行時用的驅動，原本只存在於遠端，
從未入庫。2026-07-31 與 2026-08-01 陸續收回，用途是**保留參數的出處**：
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
| `e27_calibration.sh` | E27 校準四輪（Lightning AI H100） |

腳本內的路徑是機器上的絕對路徑（`cd /work/nelson0314/WACV`、`source env.sh`，
或 `cd /teamspace/studios/this_studio/WACV`），**未改寫**——改了就不是當初實際
執行的那份了。要重跑請對照 `scripts/twcc_env.sh`（即當時的 `env.sh`）自行調整路徑。

E18、E19 的驅動以可執行形式放在上一層：`scripts/e18_sweep.sh`、`scripts/e19_lam.sh`。
