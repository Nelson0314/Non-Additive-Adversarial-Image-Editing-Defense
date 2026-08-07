# 檔案索引

> `CLAUDE.md` 指定本檔為 `docs/` 的入口：哪份文件現行、哪份已封存、
> 每個 run 目錄屬於哪個實驗。
>
> **接手實驗請先讀 `HANDOVER_2026-08-08.md`**——它自足，讀完就能接手。
>
> 本檔於 2026-08-05 重建。前一版在 `8e0ffbc`（Reset the project to the minimal
> reproducible set）連同 3775 個檔案一併刪除，但 `CLAUDE.md` 的指引未同步，
> 造成入口指向不存在的檔案。舊版可用
> `git show 8e0ffbc^:docs/INDEX.md` 取回，其內容描述的是 reset 之前的目錄結構。

---

## 1. 現行設計文件（2026-08-05 一批，順序即依賴順序）

| 檔案 | 內容 | 讀它的時機 |
|---|---|---|
| `SURVEY_2026-08-05.md` | 文獻與方法選擇的依據 | 想知道為什麼選這些 baseline／這個威脅模型 |
| `DESIGN_2026-08-05.md` | 實驗設計：條件、軸、判準、主張階層 | 想知道要證什麼、怎麼算成立 |
| `ARCH_2026-08-05.md` | 架構：五段流程、三層分工、續跑語意 | 想知道程式怎麼組起來 |
| `CODE_2026-08-05.md` | 介面契約：每個模組的輸入輸出 | 動手寫程式之前 |
| `LOGIC_CHECK_2026-08-05.md` | 對上述四份的邏輯檢查與既知缺陷 A1–A13 | 想知道哪裡曾經錯過、為什麼現在的寫法是那樣 |
| `RUNBOOK_2026-08-05.md` | **上機執行手冊**：研究背景、環境、上機流程、五段內容、監察、失敗處理、機時估計、判讀指南。**自足，讀完即可執行** | **接手執行實驗時只需讀這份** |
| `RESULTS_2026-08-06.md` | **段 1 首次在 GPU 上執行的實測紀錄**：九項缺陷、其根因與修法、生成路徑重建下限的完整量測、SD v1.4/512² 的對照量測、四項待裁決事項、逐格機時 | 要引用 08-06 的數字時讀這份；`RUNBOOK` 的旗標為何是那樣也在這裡。**其 §10.3 對位移場銳利度方向的結論已被 `RESULTS_2026-08-07` §5.1 推翻** |
| `RESULTS_2026-08-07.md` | **第一階段的完整實測**：段 0 三項檢查的數據、本日四項缺陷、**訓練實際被 `tau_acut` 而非 `tau_lpips` 綁住的量測**（§4）、τ=0.20 上兩道 hinge 的逐條件實測與門檻建議（§5）、對齊指標的限制與四個替代方案（§6） | **要引用 08-07 的數字時讀這份**。**其 §6 對「LPIPS 對非加性不利」的方向已由 `RESULTS_2026-08-08` §1.3 更正**，見該檔開頭的 §6bis |
| `RESULTS_2026-08-08.md` | **共同約束的裁決與階段一重做**：ST-LPIPS 不可採用的實測與判準、**對 `RESULTS_2026-08-07` §6 方向的更正**（LPIPS 對非加性其實是寬鬆而非嚴苛）、v14r 的變因與為何重跑段 0、共用工作區的風險 | 要引用共同約束的裁決、或接手 v14r 時讀這份 |
| `HANDOVER_2026-08-08.md` | **現行交接**：img2img 那一輪的結論（主張不成立）與根因、下一批 inpainting 怎麼起、九個 commit 的清單、三個不要重踩的設計點、尚未做的三項 | **接手時先讀這份** |
| `HANDOVER_2026-08-07.md` | **已過時**：描述段 0 正在跑當下的狀態，其「兩組實驗並行、跑完即有結論」的前提已由該日的結果推翻 | 只在追溯當日決策時讀 |

---

## 2. 原始碼查證（baseline 與淨化算子）

這些檔案的存在理由是：`src/baselines/` 重現的是**別人的方法**，任何一個值被
「合理化」成看起來整齊的數字，整批比較就失效，且事後補不回來。故每一項參數
都必須有原始碼逐字佐證。

| 檔案 | 涵蓋 |
|---|---|
| `SOURCE_AUDIT_2026-08-05.md` | 全部七篇的裁決與值域對照表（§10）、排除名單（§9、§11） |
| `_audit_promptflare_photoguard.md` | PromptFlare、PhotoGuard 的逐字原文 |
| `_audit_dia_apa.md` | DIA、APA 的逐字原文 |
| `_audit_mist_diffvax.md` | Mist、DiffVax 的逐字原文 |
| `_audit_advpaint_dia_promptflare.md` | **2026-08-05 新增。** AdvPaint 全篇、DIA 的 `custom_diffusion`／`backward_ddim`／值域雙路徑、PromptFlare 的兩列嵌入——補上程式審查（見 §3）點名的七項無佐證宣稱 |
| `_audit_purify.md` | 淨化算子的原始碼查證 |

---

## 3. 2026-08-05 程式審查的處置結果

三份逐檔審查報告（`_review_defense.md`、`_review_purify_sdxl.md`、
`_review_baselines.md`）在全部項目處置完畢後**已刪除**——它們是工作文件，
留著會與程式現況分岔。原文用 `git show 4d2332c:docs/_review_<名>.md`。
結論與處置摘錄於此，因為「為什麼現在的寫法是那樣」必須留存：

| 審查對象 | 發現 | 處置 |
|---|---|---|
| `src/defense/`、`src/residual/` | **致命 1**：`build_apa` 的 `init_std=0.0` 使 N3 階段二梯度恆為零，最佳化靜默不更新 | 已修，`site_apa.py` 取 `latent_init_std=0.02` 並在 ≤0 時拋出 |
| 同上 | 預設保真係數使 L∞ 取代 LPIPS 成為綁定約束 | `LossConfig.beta_linf = 0.0` |
| 同上 | 停止門檻沿用舊模型的校準值且無 context 檢查 | `MONITOR_TOL` → `LEGACY_MONITOR_TOL`（僅作量級紀錄），`resolve_stop_tol` 取消全部回退 |
| 同上 | `config_hash` 的必填鍵未涵蓋參數化容量 | 增設 `module_params`／`optim_params`（控制點 32 與 128 曾算出同一個雜湊） |
| 同上 | 射線縮放無最終容差檢查 | `solve_k` 二分用盡即拋出 |
| 同上 | N1 的注意力目標取自無條件分支 | SDXL 的無條件分支為零張量，目標改取自條件分支 |
| `src/purify/`、SDXL 適配 | 主驅動未依 `available` 篩選，淨化評測會中途整段中止 | `unavailable_purifiers`／`annotate_unavailable`，在跑之前標成 skipped |
| 同上 | SDXL 的無條件嵌入取錯（無症狀） | `SDXLWrapper.uncond_prompt()` 依 `force_zeros_for_empty_prompt` 回傳零張量 |
| `src/baselines/` | `seed` 傳不到 `prepare`，三篇換 seed 得到逐位元相同的結果 | `pgd.py` 顯式傳遞，測試釘住 |
| 同上 | `advpaint` 的 `x_adv.sum()*0.0` 錨點使 autograd 的斷裂偵測永不觸發 | 改由第一個 norm 項起算，新增三項測試 |
| 同上 | AdvPaint 雜訊抽樣、DIA `prev_timestep`、PromptFlare 兩列嵌入三項無佐證 | 取原檔逐行核對，三項**皆判定我方實作正確**；逐字佐證見 `_audit_advpaint_dia_promptflare.md` |
| 同上 | PhotoGuard 取 `.mean` 而原作取 `.sample()` | 列入 `modification_note` 第 (7) 項；不改 `encode_image`（`vae_ckpt` 重算會抽到另一組雜訊，屬介面改動） |
| 同上 | `step_size`／`objective` 不在任何對照表內 | 新增 `AUDIT_STEP`（5 欄 × 6 篇）與三項測試 |
| 同上 | `prepare`／`loss_fn`／`run_pgd` 從未被執行過 | `tests/test_sdxl.py` §7 以極小 SDXL 結構實跑；當場攔下「五篇全部照 SD v1.x 的 API 寫」這個缺陷 |

---

## 4. 已刪除的檔案與取回方式

一律刪除而不封存（使用者 2026-08-07 指示）：留著過時檔案會與現況分岔，
而 git 本來就保存得住。取回用 `git show <sha>:<路徑>`。

| 檔案 | 原因 | 取回 |
|---|---|---|
| `ADVISOR.md` | 先驗總結與指導者協定。威脅模型（SD v1.4、strength 0.3、10 步）與現行全面不符。兩項仍被引用的結論已寫進引用處本文 | `git show 5b775cc:ADVISOR.md` |
| `HANDOFF.md` | 2026-08-04 的總覽。§0 目標、§1 環境、§2 程式清單、§5 機時、§6 慣例、§8 清洗紀錄全部過時；仍是唯一來源的 §3／§4／§7 已抽出為 `PRIOR_FINDINGS.md`（**節號原樣保留**，故既有引用仍有效） | `git show 30a6737:HANDOFF.md` |
| `docs/HANDOVER_2026-08-06.md` | 描述段 1 上機前的狀態，其三項前提當天即被推翻（校準表的反演設定錯誤、`lr.N2` 由雜訊選出、訓練點 τ=0.35 不可用） | `git show 30a6737:docs/HANDOVER_2026-08-06.md` |
| `docs/archive/2026-07-26-v2-evidence/` | 先驗證據 21 檔 5.8 MB。其結論數值（同 LPIPS 下 PSNR 差 12.70 dB、L∞ 差 28.2 倍）已寫進 `DESIGN` §3.2 與 `PRIOR_FINDINGS` | `git show 30a6737:docs/archive/2026-07-26-v2-evidence/README.md` |

`8e0ffbc` 之前的文件（`EXPLAINER.md`、`LEDGER.md`、`CONVERGENCE.md`、
`NEXT_SESSION.md`、`RESULTS_E13-E23.md`、`RESULTS_E25-E31.md` 等）於該次
reset 一併刪除，內容已由 `PRIOR_FINDINGS.md` 承接。要查原文用
`git show 8e0ffbc^:docs/<檔名>`。

---

## 5. `runs/` 目錄屬於哪個實驗

**下表 20 個目錄都是先驗實驗**（2026-08-04 定案：截至該日的所有實驗一律視為
先驗實驗，程式與評測流程整批重造）。

### 5.1 本輪的批次（2026-08-06 起）

產物寫在**版控範圍之外**（機器上的 `~/wacv_runs`，見 `shard.sh` 的 `RUNS`），
回收時再打包進 `runs/`。命名與內容：

| 批次 | 設定 | 內容與狀態 |
|---|---|---|
| `b1` | SDXL 1.0 base、1024²、bf16 | 段 0 校準表；**該表在錯誤的反演設定下產生**，N3 的兩個 lr 校準的是另一個問題，lr.N2 由雜訊選出（`RESULTS_2026-08-06` §2、§4） |
| `b1_<影像>` | 同上 | 段 1，bird_03 與 cat_02。14 格中 6 格因三個程式缺陷失敗，已修（commit `1024633`） |
| `b2_<影像>` | 同上 + `--exact-inversion --t-max 200 --purify-mode all` | 只跑 N3，驗證反演與階段一門檻兩項修正。**三張圖皆完成**，結果見 `HANDOVER_2026-08-07` §6 |
| `lrp0.001`、`lrp0.02`、`lrp0.1` | 同 b1 + `--purify-mode all` | N2 的學習率掃描，60 步。**段 0 判準失效的證據**（`RESULTS_2026-08-06` §2.4） |
| `memtest` | 同 b1 + `--purify-mode all` | 確認 `all` 在 24 GB 下對 N2／N3 可行（6 步） |
| `diag`、`diag_recon` | SDXL、1024²、bf16 | `diag_vae_floor.py` 的純 VAE 來回與 `(k_inv, t_max, exact_inversion)` 掃描 |
| `diag_v14` | SD v1.4、512²、fp32 | 同上，第二組實驗的依據（`RESULTS_2026-08-06` §8） |
| `b3`、`b3_<影像>`、`b3_merged` | SDXL、1024²、bf16 | **SDXL 正式那一組**。段 0–4 全部完成。攻擊在 strength 0.6 上本身很弱（區間 0.038 對 v14 的 0.091），結果比 v14 更沒有資訊量（`RESULTS_2026-08-07` §6c.3） |
| `v14`、`v14_<影像>`、`v14_merged` | SD v1.4、512²、fp32 | **第二組實驗，本輪的主要證據**。三張圖 2940 列零失敗。20 個（影像 × 條件）組合沒有一個通過 `mean ≥ 3σ`，主張無法判定（`RESULTS_2026-08-07` §6a） |
| `ip*` | SD inpainting、512²、fp32 | **第二階段**，由另一個 session 負責 |
| `v14r`、`v14r_<影像>`、`v14r_merged` | SD v1.4、512²、fp32、**strength 0.4**、門檻 0.16/3.2 | **階段一重做**（2026-08-08 起）。變因與判讀見 `RESULTS_2026-08-08` §2 |


| 目錄 | 實驗 | 內容 |
|---|---|---|
| `e15_P_tau0.02`、`e15_P_tau0.05`、`e15_P_tau0.10` | E15 | site P（加性低秩），三個 τ |
| `e15_S_tau0.02`、`e15_S_tau0.05`、`e15_S_tau0.10` | E15 | site S（空間變形），三個 τ |
| `e21_P_tau{0.02,0.05,0.10}` | E21 | site P，改後的目標函數 |
| `e21_Sbic_tau{0.02,0.05,0.10}` | E21 | site S（bicubic resample），三個 τ |
| `e23_P_s100_tau0.05` | E23 | site P，100 步 |
| `e23_Sbic_s100_tau0.05` | E23 | site S（bicubic），100 步 |
| `gate_S`、`gate_S_g128b`、`gate_suppress`、`gate_compare.csv` | 閘門檢查 | 控制點數 32／128 的對照，以及抑制項的開關對照 |
| `p6_purify_retention` | P6 | 抗淨化保留率（`net_lpips` 計算下曾看似成立，換 Δsiglip 後七組配對全部不成立，見 `PRIOR_FINDINGS.md` §3.2） |
| `logs` | — | 各次執行的原始 log |

`runs/` 是唯一的證據來源，容器會被刪除、實驗無法重跑。所有 CSV／JSON／log／
PNG／HTML 一律入版控，改動 `.gitignore` 的 `runs/` 區塊時必須用
`git status --porcelain --ignored` 確認沒有結果檔被排除
（曾有一條 `runs/*/**` 讓 git 停止遞迴而靜默漏掉 273 個檔案，見 `1942e38`）。

---

## 6. 根目錄的兩份

| 檔案 | 內容 |
|---|---|
| `PRIOR_FINDINGS.md` | 2026-08-04 的總覽，**大半已過時**。仍是唯一來源的只有 §3（先驗抗淨化）、§4（量測陷阱 A1–A13）、§7（文獻清單）；逐節現況見其開頭的狀態表 |
| `README.md` | 專案入口，指向本檔 |
