# baseline：十二個免疫方法在同一條編輯管線上的比較

本專案包含主表的讀數、產生它們的程式、baseline 攻擊實作與每個數字的出處。現況、各資料組的關係與接續指引在 `STATUS.md`。

## 目錄

| 路徑 | 內容 |
|---|---|
| `src/immunization_baseline/` | `cli/`（命令列入口）、`attacks/`（baseline 攻擊實作）、`editors.py`、`layout.py`（預設目錄）、`resume_state.py`、`third_party/ultraedit/` |
| `scripts/` | shell 入口：`evaluate_color_condition.sh`、`run_flux_conditions.sh`，以及共用環境 `env.sh` |
| `configs/prompts/` | UltraEdit 指令句型 |
| `data/` | `portraits/`（原圖、`masks/` 重繪遮罩、`prompts.yaml`）、`targets/`（Mist 目標影像） |
| `results/` | 全部 CSV，入版控 |
| `docs/` | `EVALUATION.md` 與 `reference/`（方法出處、原始碼查證、淨化算子查證） |
| `tests/` | 條件規格、續跑與完成判定、專案自足性（`pytest`，不需 GPU） |
| `vendor/` | `immunization_core` 套件與 GPU 租約工具的固定版本快照；`vendor.lock.json` 記錄來源 commit 與逐檔雜湊 |
| `docs/TRIALS.md` | 已刪除的暫時性嘗試紀錄；`trial.sh new|promote|drop` 的用法見檔首 |
| `artifacts/`、`runtime/`、`trials/` | 影像產物、執行狀態、暫時性嘗試；不入版控 |

## 讀數

主讀數（12 條件、ip2p／inpaint 兩場景，原生預算）：

| 檔 | 列數 | 內容 |
|---|---|---|
| `results/displacement.csv` | 768 | 編輯結果 LPIPS：`LPIPS(編輯(原圖), 編輯(防禦圖))`，含主體內／外分區 |
| `results/retention.csv` | 5,376 | 淨化後的編輯結果 LPIPS、淨增益、保留率，七道算子 |
| `results/additional_metrics/fidelity.csv` | 96 | 防禦圖對原圖的 FSIM／ΔE00／MSE |
| `results/additional_metrics/displacement.csv` | 768 | 編輯結果的 FSIM／MSE |
| `results/additional_metrics/retention.csv` | 5,376 | 淨化後編輯結果的 FSIM |
| `results/additional_metrics/aesthetic.csv` | 104 | 七項無參考美學指標，含八張原圖的參照列 |
| `results/additional_metrics/vmaf.csv` | 6,240 | VMAF（`libvmaf`，單幀；`pairing` 欄區分 fidelity／displacement／retention） |
| `results/defense_<條件>.csv` | 8／檔 | 防禦圖對原圖的失真與該方法的求解設定（各篇原生預算） |
| `results/aligned/defense_<條件>.csv` | 8／檔 | 十個方法縮到同一防禦圖 LPIPS 錨點（0.3344）的求解結果，見 `results/aligned/README.md` |

跨編輯器（協定與主讀數不同，見 `STATUS.md`「跨編輯器」）：

| 檔 | 列數 | 內容 |
|---|---|---|
| `results/flux/edits_<arm>.csv` | 32／檔，13 檔（分母＋12 條件） | FLUX.1-Kontext 編輯輸出的 id_orig／arcface_orig，guidance 3.5、1024×1024 |
| `results/flux/displacement.csv` | 384 | FLUX 全表的編輯結果 LPIPS，欄位與 `displacement.csv` 同組 |
| `results/ultraedit/edits/<arm>.csv` | 256／檔，13 檔 | UltraEdit（SD3）全表的逐格編輯，未淨化＋7 道算子 |
| `results/ultraedit/displacement.csv`、`results/ultraedit/retention.csv` | 384、2,688 | UltraEdit 全表的編輯結果 LPIPS 與保留率 |
| `results/additive_transfer.csv` | 704 | 加性穿透拆解（ip2p，原生 12 條件＋等失真 10 條件），見 `results/ADDITIVE_TRANSFER.md` |
| `results/sweeps/` | 數格至兩百格 | 未防禦影像上的編輯器參數掃描：`flux/`、`sdedit/`、`sdxl_ip2p/`、`ultraedit/`，及 `ip2p/` 對照；`*_off_target.csv` 為指令以外改動的讀數（背景／主體 ΔE00、LPIPS） |

十二個條件：`dct_shield_y`、`mist`、`dct_shield`、`photoguard_linf`、`danp`、`sifm`、`dayn`、`dia_pt`、`dia_r`、`photoguard_c`、`color`、`diffvax`。顏色那一列原為 `color_curve`（數值留在 commit `0dd243b`），現為 color 專案的方法 `color`；等失真臂仍以舊顏色列的 0.3344 為錨點（見 `STATUS.md`）。每個條件 8 影像 × 4 指令 × 2 場景 ＝ 64 格。

## 協定

| | ip2p | inpainting |
|---|---|---|
| 受害模型 | `timbrooks/instruct-pix2pix` | `runwayml/stable-diffusion-inpainting` |
| guidance | `s_t` 7.5、`s_i` 1.8 | 7.5 |
| 未防禦對照 arm | `ip2p_si18` | `inpaint_undefended` |

兩個場景共用種子 20260812、50 步、512²。LPIPS 為 `piq.LPIPS()`（VGG）。

束縛種類不同的列不可直接比大小，`eps` 與 `eps_pixel01` 兩欄都不可跨列比：十二列分成五種束縛（`linf`、`l2`、`dct_coeff_linf`、`delta_e00_cap`、`none`），逐列單位見 `docs/reference/BASELINE_PROVENANCE.md`「`eps_pixel01` 欄的單位，逐列」，值域換算的逐篇出處見 `docs/reference/SOURCE_AUDIT.md` §10。引用一個數字時連同其協定引用（同檔規則 1）。`rotate15` 的角度與幾何類分區讀數的已知限制見 `docs/EVALUATION.md`。

## 執行

在 baseline 專案根目錄執行；Python 套件以 `src/` 與 `vendor/` 解析：

```bash
source scripts/env.sh          # 設定 PYTHONPATH 與 PY；ENV_FILE 可指定機器相關設定
"$PY" -m immunization_baseline.cli.<入口> --help
python -m pytest tests         # pyproject.toml 已設定 pythonpath
```

所有預設目錄由 `layout.py` 依本專案根推定，每個入口皆可以參數覆寫；不搜尋其他專案。共用的編輯、淨化與讀數流程來自 `vendor/immunization_core`，更新方式為在 repo 根執行 `python core/scripts/export_vendor.py baseline`。

主流程順序：`run_edits`（未防禦）→ `generate_defenses`（外部十一條件）與 `import_defense_artifacts`（`color`）→ `run_edits --defended` → `measure_edit_displacement` → `apply_purifiers` → `run_edits`（淨化後）→ `measure_purified_displacement` → `measure_additional_metrics`（五個 stage，含 VMAF）。`check_edit_completion` 驗收編輯格與產物。

| 入口 | 用途 |
|---|---|
| `generate_defenses` | 各 baseline 在原生設定上求解防禦圖 |
| `import_defense_artifacts` | 將顏色方法的防禦圖整理成主表版面並重算保真欄 |
| `run_edits`、`apply_purifiers`、`measure_edit_displacement`、`measure_purified_displacement` | 對應 `immunization_core.pipelines` 的 editing、purification、displacement、retention，預設資料根為 `data/portraits` |
| `measure_additional_metrics` | FSIM、ΔE00、MSE、無參考美學指標與 VMAF |
| `measure_additive_transfer` | 加性穿透拆解 |
| `run_flux_edits`、`measure_flux_displacement` | FLUX.1-Kontext 全表與其編輯結果 LPIPS |
| `run_ultraedit_edits` | UltraEdit 全表（13 arm × 未淨化與 7 道算子），輸出沿用主表版面 |
| `sweep_sdedit_parameters`、`sweep_editor_parameters` | SDEdit 與 SD 系列編輯器的參數／句型掃描 |
| `measure_off_target_changes` | 編輯結果在指令以外區域的 ΔE00 與 LPIPS |
| `scripts/evaluate_color_condition.sh` | `color` 條件的整條評測鏈（`main`：匯入、編輯、淨化、淨化後重編、UltraEdit；`flux`：FLUX） |
| `scripts/run_flux_conditions.sh` | 依序對多個條件執行 FLUX 全表 |

## 影像產物

逐格影像為 512×512 RGB PNG（FLUX 為 1024×1024），位於 `artifacts/`：

| 用途 | 格數 | 路徑式樣 |
|---|---|---|
| 未防禦編輯（分母） | 64 | `artifacts/undefended_edits/{ip2p_si18｜inpaint_undefended}/<圖>__p<N>.png` |
| 防禦後編輯 | 768 | `artifacts/defended_edits/<條件>/<場景>_<條件>/<圖>__p<N>.png` |
| 淨化後·未防禦分母 | 448 | `artifacts/purified_edits/undefended/<算子>/<場景>_undefended_<算子>/<圖>__p<N>.png` |
| 淨化後·防禦 | 5,376 | `artifacts/purified_edits/<條件>/<算子>/<場景>_<條件>_<算子>/<圖>__p<N>.png` |
| 防禦圖 | 96 | `artifacts/defenses/<條件>/<圖>__<條件>__def.png` |
| 等失真防禦圖 | 80 | `artifacts/defenses_aligned/<條件>/<圖>__<條件>__def.png` |
| 淨化後防禦圖 | — | `artifacts/purified/<條件>/<算子>/<圖>__def.png` |
| FLUX、UltraEdit 編輯 | — | `artifacts/flux_edits/<arm>/`、`artifacts/ultraedit_edits/` |

原圖與重繪遮罩在 `data/portraits/`（`man/`、`woman/`、`masks/`）。
