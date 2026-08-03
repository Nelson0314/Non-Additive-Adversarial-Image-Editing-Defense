# 下一階段：對齊 Lo et al. (CVPR 2024) 的協定

<!-- STATUS-BLOCK -->
| | |
|---|---|
| **狀態** | 現行。2026-08-03 改寫，取代同日稍早的「E31 正對照搜尋」版 |
| **設計依據** | `docs/specs/2026-08-03-lo-aligned-protocol.md` |
| **基準論文** | Lo, Yeo, Shuai, Cheng, *Distraction is All You Need*, CVPR 2024 |
| **前一輪** | `docs/RESULTS_E31_local.md`（本機階段，結果仍有效） |

> 三份索引：主張查 [`docs/LEDGER.md`](LEDGER.md)、檔案查 [`docs/INDEX.md`](INDEX.md)、
> 比對頁查 [`docs/gallery.html`](gallery.html)。本檔只講**現在要做什麼**。

---

## 1. 一句話交代現況

指導者 Ling Lo 是基準論文的第一作者，其約束（L∞ ≤ 0.06）、判準（Table 1
五指標）與 baseline（PhotoGuard 兩變體）已定為本專案的必要對齊項。用他的
判準重算既有全部資料後發現：**在他的預算上，我們的加性實作落後 PhotoGuard**
（LPIPS 0.3466 對 0.4056、PSNR 24.96 對 18.26）。因此當務之急由「找防禦有沒有
效」改為「先把加性基準重現對」。

---

## 2. 論證的兩層結構

| 層 | 要回答的 | 現況 |
|---|---|---|
| **第一層：重現** | 在 L∞ ≤ 0.06、N = 100、Table 1 判準下，我們的加性實作是否達到論文水準？ | **未達到** |
| **第二層：貢獻** | 非加性在匹配人眼可辨失真下能否勝過該基準？ | 第一層通過後才有意義 |

順序不可顛倒。第一層落後基準時，任何對其判準或設定的質疑都會被讀成
「沒調好就怪尺」。

---

## 3. 進度

### 已完成（2026-08-03，全部在本機，零雲端成本）

| 項 | 產出 |
|---|---|
| 論文本文與補充材料的協定逐項抄錄 | 規格 §2 |
| Table 1 五指標補齊 | `src/metrics/suite.py` 加入 `vif_p`、`fsim` |
| 式 (3)(4)(5) 實作 | `src/models/attention.py` 三個新函式 |
| Algorithm 1 實作 | `src/defense/linf_attack.py` |
| 三根柱子的損失 | 同上：`semantic`／`pg_encoder`／`pg_diffusion` |
| 驅動腳本 | `scripts/run_lo_baseline.py`（含 20 種子平均） |
| 對照報表 | `scripts/report_table1.py` → `docs/RESULTS_TABLE1.md` |
| 資料集規格與正規化工具 | `data/lo_aligned/prompts.yaml`、`scripts/prepare_dataset.py` |
| 既有資料在該判準下的重判 | 規格 §6 |
| 測試 | `tests/test_lo_protocol.py`，27 項 |

### 待辦

| 編號 | 內容 | 在哪跑 | 前置 |
|---|---|---|---|
| **L0** | 備齊資料集影像（人物與動物六類，建議每類 4 張），過 `--check` | 本機 | **使用者提供影像** |
| **L1** | 三個攻擊在 κ = 0.06 上跑完，20 種子評測 | 雲端 | L0 |
| **L2** | 對照 Table 1，判定第一層是否通過 | 本機 | L1 |
| **L3** | 同一批 x_adv 加測語意軸與劣化軸 | 本機 | L1 |
| **L4** | 非加性臂在匹配 LPIPS 下與 L1 的加性解比較 | 雲端 | L2 通過 |

指令：

```bash
# L0（本機）
python scripts/prepare_dataset.py --src <來源根目錄> --dst data/lo_aligned
python scripts/prepare_dataset.py --check

# L1（雲端）
python scripts/run_lo_baseline.py --data data/lo_aligned \
    --out runs/lo_baseline --attacks pg_encoder,pg_diffusion,semantic \
    --eval_seeds 20

# L2（本機）
python scripts/report_table1.py --out docs/RESULTS_TABLE1.md
```

**L4 之前不做任何其他實驗。** 明確排除的清單見規格 §7。

---

## 4. 第一層若沒通過，往哪裡查

落差來源依可能性排序，前兩項會被 L1 直接消除：

1. **單種子 vs 20 種子。** 既有 run 全部 n = 1 或 n = 6 但單種子。
2. **優化器。** 我們是 Adam + 平台停止，論文是 PGD sign × 固定 100 步。
3. **`n_edit = 10`** 對論文未公布的 SDEdit 步數。
4. **真實照片 vs 擴散生成影像。**
5. **strength 與 guidance。** 論文與補充材料都沒寫。本專案依 E26 一律用
   guidance_scale = 7.5。

---

## 5. 兩套約束不可混用

| | L∞ 球（κ = 0.06） | LPIPS 綁定 + 鈍化 + 色度 |
|---|---|---|
| 角色 | **必要的對照條件** | **我們的貢獻條件** |
| 適用 | 加性方法之間的比較 | 含非加性方法的比較 |
| 為什麼不能只用前者 | `e15_S_tau0.05`（空間變形）的 L∞ 是 0.5654，即 κ 的 9.4 倍，而實際位移不到一個像素——L∞ 量不出非加性方法的可辨失真 | — |

`report_table1.py` 強制併列 `L∞` 與 `×κ` 兩欄，就是要讓「哪幾列可以互相
比較」在表面上看得出來。

---

## 6. 本機與雲端的分工

**線上 GPU 時間只用於必須用它的部分。** 本機是 i5-12500H + RTX 2050 4 GB。

| 工作 | 在哪跑 | 實測成本 |
|---|---|---|
| 含梯度的 512² 訓練（含 PGD） | **只能雲端** | 本機 256² 已是 178 s/step，比 H100 的 512² 慢 75 倍 |
| 無梯度的 512² SDEdit（即 L1 的評測段） | 本機可 | 峰值 4873 MB、單次 222.5 s |
| 指標、判定、報表、比對頁 | 本機 | 秒級到數分鐘 |

**不要並行跑兩個 GPU 工作，也不要讓 CPU 密集工作與 GPU 工作並行**（實測單張
SDEdit 由 222 s 被拉長到 30 分鐘以上）。`scripts/drivers/local_night.sh` 因此
把本機工作串起來跑。

---

## 7. 雲端環境

- Lightning AI Studio，H100 80GB，torch 2.8.0+cu128，conda env `cloudspace`，
  repo 在 `/teamspace/studios/this_studio/WACV`。
- 背景腳本**不是 login shell**，`PY` 用絕對路徑
  `/home/zeus/miniconda3/envs/cloudspace/bin/python3`。
- 環境準備用 `scripts/drivers/colab_setup.sh`，**不是** `remote_setup.sh`
  （後者會 `pip install torch` 而有換版風險）。
- 三個已修掉的坑：numpy 2 與預裝 pandas／matplotlib／scikit-learn 的 ABI
  衝突、`pyiqa` 要 `--no-deps`、缺 `sentencepiece` 會讓 SigLIP 起不來。
- 換機器先跑 `scripts/colab_probe.py`。TF32 開／關的成本差三倍且會改變數值。
- 遠端 `git pull` 常因 `runs/` 未追蹤檔衝突而 abort，先 `mv` 到備份目錄。
- 連線資訊與 token 由使用者提供，**不得寫入任何入庫檔案**。

本機環境於 2026-08-03 補裝：`clip-anytorch`、`ftfy`、`wcwidth`、`facexlib`
（`pyiqa` 的 `clipiqa` 與 `topiq_nr` 需要），`setuptools` 由 83 降到 80。
`piq` 0.8.0 已內含 `vif_p` 與 `fsim`，不需新增相依。

---

## 8. 分支

`claude/e20-fidelity-constraint`，**未併入 main**。
未經明確授權不得併入（`CLAUDE.md`）。
