# WACV — 白盒頻域／相位抗文字編輯防禦

在白盒條件下，以外掛模組的形態，用頻域相位重參數化防止影像被文字引導的擴散
編輯竄改，並評測其抗淨化能力。

## 從哪裡開始

**`docs/INDEX.md`** 是入口，列出每一份文件回答什麼、以及查閱路徑。

| 要知道什麼 | 讀哪份 |
|---|---|
| 目標與判準 | `docs/GOAL.md` |
| 本方法的構造與可調旋鈕 | `docs/METHOD.md` |
| 對照組是什麼、重現到什麼程度 | `docs/BASELINES.md` |
| 指標、資料集、攻擊模型、對齊協定 | `docs/EVALUATION.md` |
| 測得的事實與證據路徑 | `docs/RESULTS.md` |
| 已定案的事項 | `docs/DECISIONS.md` |
| 會靜默失效的坑 | `docs/DEFECTS.md` |
| 環境、遠端機器、執行方式 | `docs/OPERATIONS.md` |
| 外部文獻的查證 | `docs/reference/INDEX.md` |

## 目錄結構

```
src/
  residual/texture_rephase.py   本方法的算子
  defense/param_pgd.py          參數化與共用的最佳化迴圈
  baselines/                    對照組（頻域與像素加性）
  purify/ops.py                 淨化算子
  metrics/suite.py              指標
  models/ip2p.py, sd.py         攻擊模型
scripts/                        驅動與分析
tests/                          456 passed / 1 skipped / 1 xfailed
runs/                           數值記錄（影像不入版控）
docs/                           見上表
data/                           資料集與 provenance
```

## 跑起來

```
python -m pytest -q                        # 先確認基準
```

GPU 工作在遠端跑，方式見 `docs/OPERATIONS.md`。
