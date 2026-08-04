# 實驗驅動腳本

分兩類：**工具性的**（下一節）與**歷史留存的**（其後）。兩者的寫法不同——
歷史腳本保留當初的絕對路徑不改寫，工具腳本以相對定位並允許覆寫直譯器。

## 現況（2026-08-04）

**現行的雲端腳本只有兩支：`lo_l1.sh` 與 `ours_l2.sh`**，設計見
`docs/specs/2026-08-03-lo-aligned-protocol.md` §5。

| 檔案 | 對應 | 狀態 |
|---|---|---|
| `lo_l1.sh` | L1：基準論文的三個基準方法在 κ = 0.06 上 | **已執行完成**，72/72 格 → `runs/lo_baseline/` |
| `ours_l2.sh` | L4：本專案的條件走同一條評測路徑 | 執行 7 格後機器時間用盡；設定已於 2026-08-04 修正，待重跑 |

`ours_l2.sh` **刻意不傳 `--lr`、`--tau_acut`、`--tau_chroma`**：三者都由
`run_ours_lo_eval.py` 依 site 與 LPIPS 預算自己決定。手動傳單一 `--lr` 正是
`runs/ours_lo/` 那批失效的原因——site S 因此跑在其校準值的 1/12.5，三道
約束一次都沒啟動（LEDGER 6.16、3.23）。

### 已中止（保留為紀錄，不要拿來跑）

| 檔案 | 為何中止 |
|---|---|
| `e29_calibration.sh` | 已執行。判準未通過：site C 六個學習率全部由色度 hinge 綁住（E29 §3） |
| `e30_grid.sh` | 未執行。E29 的否定結果使該網格只會得到「兩個都無效」的 36 格版本（E29 §8） |
| `e31_calibration.sh`、`e31_grid.sh` | 未執行。2026-08-03 由 L1 取代——基準論文 Table 1 提供了現成的座標，不需要盲搜運作點 |

四支的頂端都已加註。它們產出的 `runs/p14_budget_thresholds/thresholds.csv`
（逐預算門檻）**仍然是現行的必要輸入**，`run_ours_lo_eval.py` 直接讀它。

**本機的序列是 `local_night.sh`**：把三支 GPU 工作串起來跑。本機是 RTX 2050
4 GB，兩個 GPU 工作並行必然互搶；CPU 密集工作與 GPU 工作並行時，CPU 那側會把
GPU 工作的 Python 執行緒餓住（實測單張耗時由 222 s 拉長到 30 分鐘以上）。

`colab_setup.sh`、`smoke_local.sh` 與 `scripts/colab_probe.py` 不受影響，
仍是換機器時的標準流程。**不要用 `remote_setup.sh`**——它會 `pip install
torch` 而有換版風險。

## 腳本

原本的執行順序是：E29 沒有通過有效約束判準（每一格都是 LPIPS hinge）就不要開
E30——被別道約束綁住的格子，τ 對它不起作用，「匹配失真的比較」不成立。
E29 實際的結果是 site C 六個學習率全部由色度 hinge 綁住，判準未通過。

| 檔案 | 用途 | 成本 |
|---|---|---|
| `smoke_local.sh` | 本機 tiny-SD 走完「跑一格 → 有效約束診斷」，驗證參數組合 | 約 30 秒 |
| `remote_setup.sh` | 一般雲端機器的環境準備、GPU 檢查、預抓 SD v1.4 權重 | 約 5 分鐘 |
| `colab_setup.sh` | 同上，但**不安裝 torch／torchvision**（見下） | 約 5–10 分鐘 |
| `e29_calibration.sh` | 加入色度約束後的學習率重新校準，8 格、60 步、`--no_eval` | 約 20 分鐘 |
| `e30_grid.sh` | 主網格 2 site × 3 τ × 6 圖 = 36 格 | 2–4.2 小時 |

上表的成本是 E27 在 H100 上的實測（每步 2.47 s、每格評測 41.4 s、峰值顯示
記憶體 10.3 GB）外推的。**換機器就不能沿用**，先跑 `scripts/colab_probe.py`
實測——它跑兩個步數不同的極短 run，相減消掉每格固定成本後得到斜率，再推算
E29／E30 的實際時間與記憶體、連線上限的判定。

`e30_grid.sh` 的學習率沒有預設值，必須以 `LR_C=` 與 `LR_P=` 傳入 E29 定出的值。
E27 定出的 0.3 / 0.03 是在還沒有色度約束的程式上量的，直接沿用等於假設
色度約束不影響解。

直譯器一律用絕對路徑（`PY=` 可覆寫）：Lightning AI 的背景腳本不是 login
shell，`python3` 會取到系統直譯器而缺 numpy。

### Colab 與其他機器的差別

`colab_setup.sh` 與 `remote_setup.sh` 只差一項，但那一項會讓整個 runtime 報廢：
Colab 的映像檔已裝好與其驅動相符的 torch，而 diffusers、peft、piq 都把 torch
列為相依，pip 解相依時可能自行升級它，結果是 `torch.cuda.is_available()` 變成
False。`colab_setup.sh` 把現有版本寫成 pip constraint 擋住升級，裝完再核對一次
版本與 CUDA，不符就中止。

Colab 的完整流程（複製 repo、探測、校準、判定、網格、逐段推上 origin）寫成
notebook：`notebooks/colab_e29_e30.ipynb`。

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
