# baseline：十二個免疫方法在同一條編輯管線上的比較

本專案包含主表的讀數、產生它們的程式、baseline 攻擊實作與每個數字的出處。現況、各資料組的關係與接續指引在 `STATUS.md`。

## 目錄

| 路徑 | 內容 |
|---|---|
| `src/immunization_baseline/` | `cli/`（命令列入口）、`attacks/`（baseline 攻擊實作）、`editors.py`、`layout.py`（預設目錄）、`resume_state.py`、`third_party/ultraedit/` |
| `scripts/` | shell 入口：`evaluate_color_condition.sh`、`evaluate_pexels_portraits.sh`、`run_flux_conditions.sh`，`run_evaluation_chain.sh`（可直接執行或被 source），以及共用環境 `env.sh` |
| `configs/conditions.yaml` | 防禦條件的唯一正本：求解族、spec、求解端文字條件、是否屬主表 |
| `configs/prompts/` | UltraEdit 指令句型 |
| `data/` | `portraits/`（原圖、`masks/` 重繪遮罩、`prompts.yaml`）、`pexels_portraits/`（306 張 Pexels 人像與影像組定義，見 `docs/PEXELS_PORTRAITS.md`）、`targets/`（Mist 目標影像） |
| `results/` | 全部 CSV，入版控 |
| `docs/` | `EVALUATION.md` 與 `reference/`（方法出處、原始碼查證、淨化算子查證） |
| `tests/` | 條件規格、續跑與完成判定、專案自足性（`pytest`，不需 GPU） |
| `vendor/` | `immunization_core` 套件與 GPU 租約工具的固定版本快照；`vendor.lock.json` 記錄來源 commit 與逐檔雜湊 |
| `docs/TRIALS.md` | 已刪除的暫時性嘗試紀錄；`run_trial_lifecycle.sh new|promote|drop` 的用法見檔首 |
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
| `results/semantic_identity.csv`、`results/aligned/semantic_identity.csv` | 6,656、5,632 | 逐張編輯結果（未淨化與七道淨化後，含 `undefended`）的指令方向 CLIP-S（`clip_dir`、`siglip_dir`）、對原圖的 FaceNet／ArcFace 身分餘弦與偵測、對未防禦編輯的 `clip_pair`／`siglip_pair`／`blocked`；由 `measure_semantic_identity` 以 CUDA 產生 |
| `results/additional_metrics/edit_aesthetic.csv` | 832 | 未防禦編輯（`condition=undefended`）與防禦後編輯的七項無參考美學指標 |
| `results/sweeps/ip2p/purifier_strength/retention.csv` | 2,080 | 淨化強度掃描：JPEG 品質 95／90／70／50／30／10、blur σ 0.5／1／2／3、縮放 0.75／0.5／0.25（`resample_roundtrip` 384／256／128）後的編輯結果 LPIPS（`disp_purified`）；`color3d` 與等失真 `mist`、`photoguard_encoder`、`dct_shield`、`dia_pt`，8 張 × 4 句、ip2p（`apply_purifier_sweep`、`pipelines.retention`） |
| `results/color_conditions/<color3d／color3d_adaptive_set>/` | 32、256、576／條件 | color 兩個條件的 inpaint 全協定：`displacement.csv`、`retention.csv`（七道＋VAE-Trans）、`semantic_identity.csv` 與 `additional_metrics/`（FSIM／MSE、edit_aesthetic、VMAF）。防禦圖與 VAE-Trans 淨化輸入取自 color 專案；未防禦分母沿用主表 |
| `results/aligned/perturbation_survival.csv` | 832 | 擾動存續比 LPIPS(P(x̃), P(x)) ÷ LPIPS(x̃, x)：等失真 11 臂與 color 的 `color3d`、`color3d_adaptive_set`，七道淨化＋VAE-Trans，8 張；PNG、CPU（`measure_perturbation_survival`）。VAE-Trans 的淨化輸出與 color 兩條件的防禦圖取自 color 專案 |
| `results/<aligned／ultraedit／flux／pexels_portraits>/additional_metrics/` | — | 同名欄位的補充讀數：`edit_aesthetic.csv`（四者）、`displacement.csv`（編輯結果 FSIM／MSE；`ultraedit`、`flux`、`pexels_portraits`、`aligned`）、`fidelity.csv` 與 `aesthetic.csv`（`aligned` 防禦圖的逐張失真與美學）；`aligned` 另有 `vmaf.csv` 與 `retention.csv`（淨化後編輯結果 FSIM），由 color 專案以 commit 3fd14f8 的 `measure_additional_metrics` 在 CPU 上產生（ffmpeg 7.0.2） |
| `results/additional_metrics/vmaf.csv` | 6,240 | VMAF（`libvmaf`，單幀；`pairing` 欄區分 fidelity／displacement／retention） |
| `results/defense_<條件>.csv` | 8／檔 | 防禦圖對原圖的失真與該方法的求解設定（各篇原生預算） |
| `results/aligned/defense_<條件>.csv` | 8／檔 | 十個方法縮到同一防禦圖 LPIPS 錨點（0.3344）的求解結果，見 `results/aligned/README.md` |

跨編輯器（協定與主讀數不同，見 `STATUS.md`「跨編輯器」）：

| 檔 | 列數 | 內容 |
|---|---|---|
| `results/flux/edits_<arm>.csv` | 32／檔，13 檔（分母＋12 條件） | FLUX.1-Kontext 編輯輸出的 id_orig／arcface_orig，guidance 3.5、1024×1024 |
| `results/flux/displacement.csv` | 384 | FLUX 全表的編輯結果 LPIPS，欄位與 `displacement.csv` 同組 |
| `results/ultraedit/edits_<arm>.csv` | 256／檔，13 檔 | UltraEdit（SD3）全表的逐格編輯，未淨化＋7 道算子 |
| `results/ultraedit/displacement.csv`、`results/ultraedit/retention.csv` | 384、2,688 | UltraEdit 全表的編輯結果 LPIPS 與保留率 |
| `results/ultraedit/semantic_identity.csv`、`results/flux/semantic_identity.csv` | 3,328、416 | 欄位同 `results/semantic_identity.csv`；FLUX 只有未淨化 |
| `results/sweeps/` | 數格至兩百格 | 未防禦影像上的編輯器參數掃描，依 `<編輯器>/<變因>/` 分層：`flux/`、`sdedit/`、`sdxl_ip2p/`、`ultraedit/`，及 `ip2p/reference/` 對照；`*_off_target.csv` 為指令以外改動的讀數（背景／主體 ΔE00、LPIPS）；對應影像在 `artifacts/sweeps/` 的同名位置 |

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

所有預設目錄由 `layout.py` 依本專案根推定，每個入口皆可以參數覆寫；不搜尋其他專案。共用的編輯、淨化與讀數流程來自 `vendor/immunization_core`，更新方式為在 repo 根執行 `python core/scripts/generate_vendor_snapshot.py baseline`。

主流程順序：`run_edits`（未防禦）→ `generate_defenses`（外部十一條件）與 `import_defense_artifacts`（`color`）→ `run_edits --defenses-dir` → `measure_edit_displacement` → `apply_purifiers` → `run_edits`（淨化後）→ `measure_purified_displacement` → `measure_additional_metrics`（五個 stage，含 VMAF）。`evaluate_edit_completion` 驗收編輯格與產物。

| 入口 | 用途 |
|---|---|
| `generate_defenses` | 各 baseline 在原生設定上求解防禦圖 |
| `import_defense_artifacts` | 將顏色方法的防禦圖整理成主表版面並重算保真欄 |
| `run_edits`、`apply_purifiers`、`measure_edit_displacement`、`measure_purified_displacement` | 對應 `immunization_core.pipelines` 的 editing、purification、displacement、retention，預設資料根為 `data/portraits` |
| `measure_additional_metrics` | FSIM、ΔE00、MSE、無參考美學指標（防禦圖：`aesthetic`；編輯結果：`edit_aesthetic`）與 VMAF |
| `measure_semantic_identity` | 逐張編輯結果的指令方向 CLIP-S、FaceNet／ArcFace 身分與影像語意相似度，涵蓋未淨化與七道淨化 |
| `run_flux_edits`、`measure_flux_displacement` | FLUX.1-Kontext 全表與其編輯結果 LPIPS；`--purifier` 以主表 ip2p 的淨化後影像為輸入 |
| `run_ultraedit_edits` | UltraEdit 全表（13 arm × 未淨化與 7 道算子），輸出沿用主表版面 |
| `sweep_sdedit_parameters`、`sweep_editor_parameters` | SDEdit 與 SD 系列編輯器的參數／句型掃描 |
| `measure_off_target_changes` | 編輯結果在指令以外區域的 ΔE00 與 LPIPS |
| `generate_subject_masks` | 以 CLIPSeg 產生資料集的 `masks/`（白 = 重繪）與 `overview/` |
| `scripts/evaluate_pexels_portraits.sh` | Pexels 人像資料組的 ip2p 評測：在一張卡上依序完成指定的影像組（可續跑） |
| `scripts/evaluate_pexels_aligned.sh` | Pexels 人像資料組的等失真臂（八條件沿用主表等失真臂的 eps 倍率），未防禦分母沿用原生評測 |
| `scripts/run_worker_when_card_free.sh` | 等到本機有空卡再啟動一個 Pexels worker |
| `scripts/run_pexels_aligned_worker.sh` | Pexels 等失真臂的取卡迴圈（呼叫 `evaluate_pexels_aligned.sh`） |
| `scripts/run_pause_for_color_queue.sh` | 為 color 佇列讓卡：暫停本機的 Pexels worker，color 的 log 出現結束標記後恢復 |
| `scripts/evaluate_portrait_condition.sh`、`scripts/measure_portrait_condition_additional.sh` | data/portraits 8 張上單一條件的完整評測（兩場景、七道淨化、位移、保留率、semantic_identity）與其後的 CPU 讀數；等失真臂重跑與 BlurGuard 原生列使用 |
| `scripts/generate_pexels_result_tables.py` | 把已完成影像組的讀數合併為 `results/pexels_portraits/` 的逐條件表 |
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
| 等失真防禦圖 | 80 | `artifacts/aligned/defenses/<條件>/<圖>__<條件>__def.png` |
| 淨化後防禦圖 | — | `artifacts/purified/<條件>/<算子>/<圖>__def.png` |
| 等失真臂的編輯 | — | `artifacts/aligned/defended_edits/`、`artifacts/aligned/purified_edits/` |
| FLUX、UltraEdit 編輯 | — | `artifacts/flux/edits/<arm>/`、`artifacts/ultraedit/edits/` |

原圖與重繪遮罩在 `data/portraits/`（`man/`、`woman/`、`masks/`）。
