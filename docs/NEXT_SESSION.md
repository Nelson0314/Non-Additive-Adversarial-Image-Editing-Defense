# 下一階段：方向需要重新決定

寫於 2026-08-02（E29 之後），取代 2026-08-01 的版本（該版寫的是「協議已就緒、
主網格待跑」，那個計畫已因 E29 的否定結果中止）。

完整過程見 `docs/RESULTS_E29_negative.md`，其前身為 `RESULTS_E25-E26.md`、
`RESULTS_E27_calibration.md`、`RESULTS_E28_chroma.md`。

---

## 1. 一句話交代現況

E25–E28 把量測與校準修好之後，E29 做了第一次實測：**在試過的每一個運作點上，
防禦都沒有阻止文字編輯達成 prompt**，加性與非加性皆然。主網格 E30 因此沒有跑。

否定的是「現行目標函數在現行失真預算下」，不是「非加性抗編輯這個構想」。
這兩者的差別就是下一階段要處理的事。

---

## 2. 證據強度

否定結果不是靠單一指標得出的：

| 證據 | 內容 |
|---|---|
| 影像 | `runs/e29_edit_page/`、`runs/e29c_edit_page/` 的比對頁，每一張防禦後的編輯都長出了 prompt 要求的內容 |
| 語意軸（正式判準） | SigLIP：site C 0.1068、site P 0.1277，對照未防禦的 0.1202。site P 反而**上升** |
| 雜訊條件 | 用的是防禦訓練時見過的那個 ε，即對防禦**最有利**的情況 |
| 運作點涵蓋 | τ=0.05 固定 60 步，以及 τ=0.10、上限 150 步、開平台停止（網格會用到的最寬鬆點） |
| 學習率涵蓋 | site C 四個學習率跨 3 倍範圍，結論一致 |

---

## 3. 兩個結構性問題（下一階段的核心）

兩項都寫在 `RESULTS_E25-E26.md` §6 的待決清單上，而 E27／E28／E29 處理的都是
清單上的**其他**項目。詳細推導見 `RESULTS_E29_negative.md` §5。

### 3.1 目標函數最大化的量，已被判定不對應防禦成功

`objective.py:348-352`：

```
L_def = mean_i max(0, margin − LPIPS(y_def_i, y_orig_i))
```

這是 `edit_shift`。E25 §1.2 已證實它與編輯是否失敗不對應（726 格語意失敗
0 格）。E25 之後判準改成語意軸，**目標函數沒有跟著改**。兩張都服從 prompt 的
擴散輸出，彼此的 LPIPS 距離本來就可以很大；把它推到 0.42–0.47 不需要讓任何
一張停止服從 prompt。

### 3.2 失真預算比文獻低 5–8 倍

本專案運作點 `defimg_lpips` = 0.036–0.086；DCT-Shield（ICCV 2025）自報 0.267，
PhotoGuard／MIST／AdvDM／SDS／DiffusionGuard 為 0.284–0.362。τ 的三個值
0.02／0.05／0.10 全部落在文獻的十分之一到三分之一。

**現有資料無法分辨**「這個方法無效」與「任何方法在這個預算下都無效」，
因為從未在文獻的預算區間量過。

---

## 4. 已經實作但從未在真實 SD 上跑過的兩個目標

`runs/` 的 59 個有記錄的 `env.json` **全部**是 `defense_mode=untargeted`。
另外兩個模式都有測試覆蓋、都在 tiny-SD 上跑通過，資料是零：

| 模式 | 位置 | 形式 | 備註 |
|---|---|---|---|
| `targeted` | `objective.py:341-347` | 最小化與指定目標影像的距離 | `objective.py` 自己的註解寫了「無目標最大化在文獻上一貫比有目標脆弱」 |
| `crossattn`（`attn_mode="suppress"`） | `attention.py`、`optimize.py` 的 `optimize_crossattn` | 最小化內容 token 分到的注意力質量 | 三者中唯一直接對著「讓編輯不服從 prompt」 |

`suppress` 是 E25 §4.1 新增的，因為原本的 `divergence` 在 φ=0 梯度精確為零。
預設已改為 `suppress`，但預設值只在走 `crossattn` 這條路時才生效。

---

## 5. 存活的結論（不受 E29 影響）

全部與量測方法有關，不經過防禦目標：

| 結論 | 出處 |
|---|---|
| 用單一純量定義「匹配失真」，在失真種類不同的兩族之間不成立 | E20 §0 |
| 等 LPIPS 多臂探針：判別一個指標實際在收什麼費 | E20 §5.3、E28 §1 |
| `local_acutance_dev` 對位移不敏感且不可抵銷 | E20 §6 |
| `local_chroma_bias` 分得出連貫色偏與隨機色度雜訊；τ=0.8 的人眼定錨準確 | E28 §1、E29 §4.1 |
| ΔE 那一族量的是色度誤差量值，不是人眼在意的軸 | E28 §1.1 |
| `edit_shift`／`net_lpips` 不是防禦成功的判準 | E25 §1.2、E29 §4.3 |
| 攻擊端必須有 CFG；w=1 下 SD v1.4 幾乎不服從 prompt | E26 §3 |
| A 族（site L/E/W）的 VAE 重建地板高於加性運作點 | E17–E19 |
| 三道約束彼此不循環，各擋一種失效 | E28 §3 |
| site C 的綁定者恆為色度 hinge，與學習率無關（跨 3 倍範圍） | E29 §3 |
| TF32 使同一份程式在不同機器上精度與速度都不同；E27 是在 TF32 開啟下跑的 | E28 §4、E29 §2 |

---

## 6. 不要重走的死路

- `net_lpips`／`edit_shift` 當防禦成功的判準（E25、E29）
- 對抗性強健的感知度量（E-LPIPS / R-LPIPS / LipSim）——解的是相反的失效
- NLPD、VIF、GMSD、HaarPSI 當保真約束——量的是位移不是鈍化（E20）
- ΔE76 / ΔE00 / `dchroma` 當色度約束——量的是量值不是空間連貫性（E28 §1）
- low rank（使用者 2026-07-30 排除）
- site L / E / W（A 族）——VAE 重建地板高於加性運作點
- site S（使用者 2026-08-01 決定不放進重跑）
- 固定步數的網格
- **再調 site C 的學習率**——六個值已涵蓋 3 倍範圍，綁定者不變（E29 §3）
- **在 τ ≤ 0.10 下用 `untargeted` 目標再跑一次網格**——E29 已在最寬鬆的那個
  點上量過，36 格版本只會是同一個結論的複本

---

## 7. 環境與成本（已實測，可直接引用）

- 遠端 Lightning AI Studio，H100 80GB，torch 2.8.0+cu128，conda env
  `cloudspace`。repo 在 `/teamspace/studios/this_studio/WACV`。
  背景腳本**不是 login shell**，`PY` 用絕對路徑
  `/home/zeus/miniconda3/envs/cloudspace/bin/python3`。
- **環境準備用 `scripts/drivers/colab_setup.sh` 而非 `remote_setup.sh`**：
  Lightning 的映像檔已有相符的 torch，`remote_setup.sh` 會 `pip install torch`
  而有換版風險。前者以 pip constraint 釘住並在裝完核對。
- 該環境的三個坑，都已在 E29 修掉並記錄：
  1. `pip install` 會把 numpy 拉到 2.x，而預裝的 `pandas` / `matplotlib` /
     `scikit-learn` 是對 numpy 1 編的 → 升級這三個。
  2. `pyiqa` 未列在 `environment.yml`（已補），且必須 `--no-deps`。
  3. `sentencepiece` 未列，缺它 SigLIP 的 tokenizer 起不來，而 SigLIP 是判準。
- 成本（H100，TF32 開）：每步 2.36 s、每格評測 33.7 s、峰值 10.3 GB。
  TF32 關則為 7.19 s、147.6 s、25.1 GB。換機器先跑 `scripts/colab_probe.py`。
- 本機 `C:/Users/nelso/miniconda3/envs/wacv/python.exe`，RTX 2050 4 GB，
  **跑得動分析、跑不動 512² 訓練**。測試基準 253 passed / 1 skipped
  （遠端 torch 2.8 下 `test_極端值不產生NaN[de2000_map]` 失敗，
  該函式在 `src/` 與 `scripts/` 無任何使用者，屬 E28 判定的死路）。
- Colab 的完整流程在 `notebooks/colab_e29_e30.ipynb`。其第 5–7 節（E29 校準、
  判定、E30 網格）對應的計畫已中止，環境與推送的部分仍可用。

---

## 8. 分支

`claude/e20-fidelity-constraint`，**未併入 main**，已推上 origin。
