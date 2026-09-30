重整已達成主要結構與 CSV 保存目標，但完成判定、trials 保護及環境鎖定仍有缺口，尚不宜判定全面驗收通過。

以下以 `f89cac2` 為準；未修改檔案、未執行 Git 寫入、未連遠端、未使用 GPU。未發現 P0；P1 為可能誤判完成、遺失證據或影響結果追溯的問題。

| 嚴重度 | 類別 | 路徑:行號 | 證據 | 建議修法 |
|---|---|---|---|---|
| P1 | 完成判定 | `style/scripts/run_style_prompt_jobs.sh:40`、`:62`、`:71`、`:78` | 以 `results.csv`／`preflight.csv` 存在判定完成；未驗證預期格數、設定或影像。讀數命令失敗後仍繼續，最後可印出 `_DONE` 並成功退出。已派出的最佳化若失敗且沒有結果，`OPT` 又會阻止重派，可能永久等待。 | 記錄各階段退出碼；按工作清單驗證鍵集合、產物與設定摘要。失敗須明確重試或終止，讀數通過後才完成。 |
| P1 | 完成判定 | `color/scripts/evaluate_condition.sh:37`；`color/src/immunization_color/cli/validate_queue_job.py:72` | 階段只要有 `.done` 就略過。佇列驗收雖檢查編輯鍵與檔案存在，卻不核對 seed、guidance、輸入雜湊或協定摘要；同名條件改設定後，舊結果仍可通過。 | 將階段標記綁定解析後設定、輸入與產物雜湊；略過前重新驗證，不能只在鏈末端驗收。 |
| P1 | 數值保存 | `core/src/immunization_core/io.py:30`、`:76`；`core/src/immunization_core/pipelines/editing.py:310` | 共用 CSV writer 直接以 `w` 截斷原檔；editing 逐格重寫完整 CSV。中止或寫入失敗可破壞既有列，也未落實報告 5.1 的原子替換要求。三份 vendor 同樣如此。 | 在同目錄完整寫出暫存檔、驗證後原子替換；續跑採單一 writer，保存原有證據。 |
| P1 | trials 資料遺失 | `core/scripts/trial.sh:35` | `promote` 只確認專案沒有未提交變更，未確認任何 trial 內容已搬入或提交。剛建立 trial、尚未採用任何內容時，乾淨工作樹也會通過並刪除 trial。 | 要求升格 manifest，記錄目的檔、雜湊與 commit，確認已提交內容後才刪除。 |
| P1 | trials 刪除範圍 | `core/scripts/trial.sh:18`、`:51`、`:55` | 遠端根直接拼入 shell 命令，沒有絕對路徑、專案身份或解析後邊界檢查；中間的 `trials` 若為符號連結，也沒有阻擋。遠端填錯但不存在的路徑，`rm -rf` 可回傳成功，隨後仍刪本機。 | 本機與遠端都核對專案身份、解析後路徑與預期 trial；安全傳遞參數，確認刪除的是指定副本後才完成。 |
| P1 | 重跑相依 | `style/configs/jobs/r11.spec:4`；`style/configs/jobs/r13.spec:4`；`style/src/immunization_style/cli/measure_style_prompt_edits.py:44` | 兩份清單都沒有 `ref` 工作，但讀數必須有 `ref_<style>`。既有參照 CSV 又分別引用退役的 `pilot`、`r12_p_snow` 防禦圖；log 第 7 項已有記錄，STATUS 未揭露。文件的「重跑一輪」命令不足以建立完整讀數輸入。 | 補齊參照工作及其設定，或保存可驗證的參照輸入快照；無法恢復時，在 STATUS 明列哪些讀數不能重算。 |
| P1 | 方法出處錯置 | `baseline/src/immunization_baseline/cli/import_defense_artifacts.py:49`、`:153`；`baseline/results/defense_color.csv:2` | `--variant color` 匯入現行 color，仍寫成封存 `paper_baseline.py` 的 `color` 臂，並套用舊方法「三個項都不經過 text encoder」的描述。這不是單純保留歷史檔名，而是對現行結果寫入錯誤出處。 | 讓匯入來源明確提供方法身份與求解協定；修正八列 metadata，保留數值及更正紀錄。 |
| P2 | 環境／未完成項 | `core/STATUS.md:24`；`baseline/STATUS.md:73`；`color/STATUS.md:11`；`style/STATUS.md:11` | 四份 `requirements.lock` 均不存在。log 第 13 項末尾仍列租約目錄切換與產生 lock 為未完成，因此「只剩第 14 項」與 HEAD 紀錄不符。 | 在實際環境產生、核對並提交 lock；補記遠端切換的完成證據。本次未連遠端，不能代為確認。 |
| P2 | trials 紀錄完整性 | `core/scripts/trial.sh:44` | `drop` 只以正規表示式確認名稱出現在表格第一欄；其餘欄全空、引用即將刪除的 trial，甚至紀錄未提交，都能通過。刪除後可能沒有可追溯結論。 | 驗證必填欄、持久化來源與 ledger 提交狀態；明確區分「未量測」與漏填。 |
| P2 | trials 遠端殘留 | `core/scripts/trial.sh:40` | `promote` 只刪本機；遠端同步清理由 `drop` 才執行。成功試驗的遠端 trial 會留下，與完整生命週期管理的目標不一致。 | 升格與放棄共用副本清理流程，保留明確的本機限定選項。 |
| P2 | 命名 | `style/README.md:11`；`style/configs/jobs/r11.spec:1`；`style/configs/jobs/r13.spec:1` | 活動設定、結果目錄與 CSV 識別值仍使用 `r11`、`r13`、`cls_p_*`。log 援引原診斷 4.7 保留歷史身份，但當時「缺完整規格」的理由已因工作清單補入而改變，且根規則明禁流水號。 | 依已知設定建立描述性名稱與一對一對照，同步路徑、CSV 和測試；若要例外保留，明列其適用範圍。 |
| P2 | 操作文件／租約 | `color/README.md:37` | 範例直接執行 `evaluate_condition.sh <GPU>`、讀數腳本；兩者只設定 GPU，未自行取得租約。照 README 操作會繞過共用容量與取卡保護。 | 範例改經 `run_queue.sh` 或租約 wrapper；明示這兩支是已持有租約時使用的內部入口。 |
| P2 | CLI 文件 | `color/README.md:42`；`color/STATUS.md:24` | README 仍用 `--workdir`，工具只接受 `--work-dir`；STATUS 仍指示 `--out`，相關 CLI 已改成 `--output-csv`。不保留別名的裁定使舊命令直接失敗。 | 更新命令，加入文件範例的解析檢查。 |
| P2 | 共通規則衝突 | `CLAUDE.md:46`；`baseline/STATUS.md:82`；`core/scripts/free_cards.sh:5` | 根規則禁止使用有他人 compute app 的卡；baseline 明列使用者允許少量 context，程式也依 512 MiB 門檻判定。共通規則與既有裁定不一致。 | 將已裁定的精確條件寫入唯一規則正本，專案 STATUS 僅引用。 |
| P2 | 文件路徑 | `baseline/docs/reference/BIBLIOGRAPHY.md:76`；`baseline/src/immunization_baseline/attacks/mist.py:4` | `BASELINE_CANDIDATES.md` 連結不存在，且仍稱「現有六個條件」。Mist、PhotoGuard、PromptFlare 等 docstring 仍引用不存在的 `docs/_audit_*.md`。 | 改指現有 audit；歷史來源須附版本並標明不在專案內，更新現行條件敘述。 |
| P2 | 文件自足 | `style/docs/DESIGN.md:6` | style 的攻擊協定只寫「同 color 專案 DESIGN §1」；單獨交付 style 時，這個方法定義入口不存在。程式 import 自足不等於文件自足。 | 在 style 保存其實際協定摘要，指向本專案設定與資料；兄弟專案連結僅供比較。 |
| P2 | 用語／證據範圍 | `style/docs/DESIGN.md:41`、`:43`；`color/docs/DESIGN.md:8` | 「沒有關聯」超出所列三張人像與有限設定的證據；「Codex 診斷」沒有可查證分析來源；「撐得過淨化」仍屬口語。部分 CLI 首段仍寫「試跑」「第三支路線」，未完全改成責任描述。 | 限定樣本與設定，將機制推論標為假說並附證據；模組首段只描述用途與契約。 |
| P2 | 文件分工 | `baseline/STATUS.md:22`；`color/STATUS.md:20`；`style/STATUS.md:8` | STATUS 保留未標查核時間的「沒有工作在跑」；color 又重複整段根規則，已造成不同文件間的政策漂移。 | 遠端狀態附查核時間與依據；共通規則只保留一份，STATUS 留專案差異、限制與待辦。 |

**確認無誤：**版控頂層只有四個活動專案、`archive/` 與根文件；空 `docs/` 不存在符合 README 說明，各專案主要層級正確。實際工作目錄另有既存 `.tmp/`、`.pytest_cache/`、`.claude/`、`pytest-ig-fixed-regression/` 與論文 PDF，不能宣稱實體根目錄已完全清空，但它們不屬入庫結構，本次未動。活動程式未發現兄弟專案 import、`sys.path` 拼接或載入 archive 的執行期相依；三份 vendor 各 50 檔與 lock、目前 core 完全一致。272 份活動 Python（含 vendor）通過 AST 解析，活動 shell 無 CRLF。228 份 CSV、38,938 列已逐值比對；除裁定的改名、路徑轉換及幾何更正外，原有欄位一致。欄名改寫確為 89 表僅換表頭、2 表改值；幾何更正確為 1,408 列、2,814 值、最大差異 0.10288。readout 退出碼、baseline 編輯驗收、FLUX／UltraEdit 續跑保護、style 選點可行性、尺寸判定、租約原子操作及 queue 重派修正均仍在新位置。文獻原名、歷史來源路徑、`__immunised.png` 格式相容及舊環境變數反向測試的保留理由成立；上表的現行 color 出處不在此例外內。

**測試限制：**本次選取可唯讀執行的 CPU 子集，合計 **116 passed、18 failed**；失敗涉及缺少 `diffusers`／`skimage`，以及 PyTorch 嘗試建立唯讀環境禁止的暫存快取。style 全套收集另缺 `piq`。core、baseline、color 分別收集到 184、78、41 個預設案例，與 log 數量相符；這不等於重新證實全部通過。完整 `--help`、隔離副本與含暫存檔的回歸測試未重新完成，也未安裝依賴。

**需要使用者裁定：**

- `baseline/results/aligned/retention.csv` 的 **1,280 列幾何分區欄**是否另行重算；此限制已正確記錄，不能視為本次已更正。
- style 已退役參照輸入若確實無法恢復，要重建並重新量測，或將相關結果明確列為不可重算的歷史證據。
- 若仍要保留 `r11`／`r13` 等活動名稱，需要明確例外；其餘工程缺口可依既有計畫修正，不需重新裁定。

