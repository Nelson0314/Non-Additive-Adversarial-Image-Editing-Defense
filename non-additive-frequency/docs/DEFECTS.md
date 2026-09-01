# 已知缺陷

只收**會靜默失效**的缺陷——補錯了不會拋錯、輸出看起來正常、但量到的是別的
東西。修好之後仍留在此處，因為同型的錯誤會再犯。

已刪除的方向（注意力抑制、APA 兩階段、位移場）留下的缺陷紀錄一併刪除；
下列各條的教訓仍適用於現行程式。

---

## DEF · 半徑被夾到 π，使不同的強度跑出同一組設定

- **症狀**：相位半徑掃描到 θ≥3 之後，失真不再上升。當時判讀為「相位是週期量
  所以有天花板」。
- **根因**：`PhaseParam.set_radius` 寫成 `self.radius = min(r, math.pi)`，
  於是 `--radius 3.5` 與 `--radius 4.5` 建構出**逐位元相同**的模組。天花板
  確實存在，但觀測到的那一段是夾取造成的。
- **修正**：`radius` 本身不封頂，只在傳給 `theta_max` 時封頂。這也讓可學幅度
  增益的上界能跟著 `radius` 走——增益沒有週期性，不該被 π 夾。
- **測試**：`tests/test_param_pgd.py::test_set_radius_clamps_phase_to_pi`
  改為驗證「`radius` 不變、`theta_max` 封頂、`project()` 後 θ 不超過 π」。

## DEF · 由防禦圖反推 DCT 係數來驗證頻帶約束

- **症狀**：驗證「δ 只落在允許的頻帶」時，色度通道判定失敗。
- **根因**：解碼含夾取與 4:2:0 重取樣，AC 的擾動會滲進反推出的 DC 與相鄰
  頻帶。**頻帶約束在像素端根本驗不出來。**
- **修正**：`run_dct_shield` 與 `run_djsma` 直接回傳 δ 本體，測試檢查 δ 而非
  反推值。
- **同型**：任何「先解碼再編碼」的往返都不保值，不可用於驗證編碼域的約束。

## DEF · `_merged` 目錄被自己的 glob 匹配到

- **症狀**：合併分片時輸出目錄被下一輪的 glob 吃回去，列數靜默翻倍。
- **修正**：合併腳本在開頭檢查輸出目錄是否落在來源的 glob 底下，是就拒絕。

## DEF · `env.sh` 把工作目錄換回舊的 repo

- **症狀**：遠端腳本報「找不到資料檔」，而該檔明明存在。
- **根因**：`env.sh` 最後一行是 `cd $HOME/WACV`（舊 repo）。腳本若先 `cd` 到
  新 repo 再 `source env.sh`，工作目錄會被換回去，所有相對路徑指向錯的樹。
- **修正**：**先 `source`，再 `cd`。** 見 [OPERATIONS.md](OPERATIONS.md)。
- **為什麼危險**：兩棵樹的 CLI 若相容，就會安靜地用另一棵樹的程式碼跑完，
  連 `src` 都跟著換。

## DEF · 監看腳本把 ssh 的 stderr 併入判斷用的取值

- **症狀**：批次還在跑就被判成收工。
- **根因**：判斷值走了合併 stderr 的路徑，再用 `tr -dc '0-9'` 抽數字；ssh
  警告訊息裡的埠號被抽出來當成計數。
- **修正**：判斷用的取值一律丟掉 stderr，並只接受明確的形狀。

## DEF · `pkill -f` 的樣式匹配到 ssh 連線本身

- **症狀**：想殺遠端的工作，結果連自己的 ssh session 一起殺掉，輸出被截斷。
- **修正**：樣式用中括號寫法（`[i]p2p_run`），使樣式字串本身不匹配。

## DEF · 政策謂詞散在多處，只改其中一部分不會報錯

- **根因**：同一個判斷條件被就地展開在三個地方。
- **後果**：漏改其一的結果是行為不一致，而輸出看起來完全正常。
- **修正**：集中為單一函式，並由測試計算該字串在原始碼中出現的次數。
- **同型**：逐影像種子也曾寫兩份，兩者必須同分布，否則比較到的差異裡混著
  起點的差異。

## DEF · 參數組由名稱推導，新增位置時靜默落回預設

- **根因**：以 `spec.site == "..."` 比對名稱來挑參數組，與本專案「依能力分派、
  不比對名稱」的慣例相反。
- **修正**：改由設定物件明寫。`src/residual/base.py` 以「能力」而非型別對外
  表達（像素側實作 `pixel_residual`，去噪側實作 `eps_hook`），**新增位置時
  不要依注入位置的名稱寫分支**。

## DEF · 隨機起點只加在其中一條建構路徑

- **症狀**：探測階段以「參數放大到上限仍達不到目標」中止。
- **根因**：需要非零起點的參數化有多條建構路徑（訓練、探測、量測腳本），
  起點只加在其中一條。
- **修正**：移到共同入口 `build_module`，並在投影器內加零方向的檢查。
- **通則**：同一個缺陷會在每一條漏掉的路徑上各出現一次，且錯誤訊息描述的是
  症狀而非原因。

## DEF · 停止準則在硬約束模式下永不觸發

- **症狀**：整輪最貴的格全部跑滿步數上限，依協議不可用於跨條件比較。
- **根因**：停止準則要求「某道約束的懲罰為正」才允許停止。投影把約束變成硬
  約束後，參數恆坐在球面上、懲罰恆為零——判準分不出「還沒被綁住」與「一直
  被綁得剛剛好」。
- **修正**：投影式條件關閉該要求，只看監看量走平。

## DEF · 跨批複製校準表使「前置階段是否完成」的檢查恆為真

- **根因**：以「該檔存在嗎」判斷前置階段有沒有跑完，而該檔是從別批複製來的。
- **規則**：跨批複製校準表**只在條件集合相同時合法**。條件不同會塞進一個錯的
  表而檢查看不出來。

## DEF · `.gitignore` 的 glob 讓 git 停止遞迴

- **症狀**：273 個結果檔被靜默排除在版控外。
- **根因**：`runs/` 區塊裡的一條 `runs/*/**` 讓 git 停止遞迴。
- **規則**：改動 `.gitignore` 的 `runs/` 區塊時，必須用
  `git status --porcelain --ignored` 確認沒有結果檔被排除。

## DEF · Windows 產生的 shell 腳本帶 CRLF

- **症狀**：遠端 bash 報 `set: - : invalid option`、`ambiguous redirect`、
  `syntax error near unexpected token`。
- **根因**：在 Windows 上用 `write_text` 寫檔會把 `\n` 轉成 `\r\n`。
- **修正**：`.gitattributes` 已把 `*.sh` 釘成 `text eol=lf`；寫檔後仍要
  明確 `replace(b"\r\n", b"\n")` 再上傳。

## DEF · `echo ... | while read` 讓迴圈跑在 subshell 裡

- **症狀**：腳本的 `wait` 等不到任何背景工作，主控立刻結束。
- **修正**：改用 here-string（`done <<< "$VAR"`）。

## DEF · 主線驅動整支 import 不進來，而 456 條測試全過

- **症狀**：`python -m pytest -q` 全數通過，但 `scripts/ip2p_run.py` 執行時
  立刻 `ImportError`。全部 GPU 工作都被擋住。
- **根因**：該檔引用 `WatermarkSpec` 與 `run_dct_watermark`，兩者早已改名為
  `DJSMASpec` 與 `run_djsma`。**沒有任何測試 import 這支驅動**，於是重新命名
  沒有在任何地方留下症狀。`dct_wm` 那一支還傳了 `topk`／`block_frac`／`eps`／
  `steps` 四個 `DJSMASpec` 根本沒有的欄位。
- **修正**：`dct_wm` 接到現行的 DJSMA 實作（`saliency="grad"`，因為本威脅模型
  沒有分類器，故 `modified_from_paper` 為真）；parser 抽成 `build_parser()`
  使 CLI 可在不載權重的情況下受測；新增一條測試 import `scripts/` 底下每一
  支驅動。
- **後果**：`runs/ip2p_fair_comparison/b1_wm` 的四個工作點由**已被取代的實作**
  產生，現行程式重跑不出來。那批數字只能標為 superseded，不可進比較表。
- **同型**：只有被 import 的程式碼會被測試保護。驅動腳本若只在遠端執行，
  重新命名與簽名改動一律不會有症狀。

## DEF · 只活在 CLI 預設值裡的設定，掃描之後在報表上分不出來

- **症狀**：合併分片後，兩組不同設定的列長得一模一樣。
- **根因**：`quantile`（紋理閘分位數）、`hop`、防禦端的 `steps` 三者只存在於
  `argparse` 的預設值，沒有寫進 `results.csv`。這與 `r_min` 當初漏記是同一型。
- **修正**：四個欄位（`quantile`／`hop`／`gate_edge_power`／`defense_steps`）
  逐列寫出，並由 `tests/test_ip2p_run_columns.py` 以字面字串釘住。
- **順帶查到**：本方法跑 100 步 PGD，DCT-Shield 跑 1000 步（該篇 §5.4）。
  **頭對頭表把兩者並排而預算差十倍**，此前不在任何欄位裡。

## DEF · `runs/ip2p_fair_comparison` 底下共存六個 CSV schema

- **症狀**：跨批分析時，缺欄位的舊列與新列被併成同一個工作點。
- **根因**：欄位一路加上來（26／27／30／32／33／37 欄），嚴格巢狀但沒有版本
  標記。`.get(k, "")` 這種寫法會把兩組不同的設定靜默併掉。
- **修正**：`scripts/distortion_axis_analysis.py` 帶一張明確的遷移表，寫出每個
  欄位**還不存在時**批次必然用的值（欄位不存在 ⇒ 該功能未實作 ⇒ 只可能是關閉
  的那個值），並對**未登記的欄位直接拋錯**而不是猜。
- **規則**：新增旋鈕時同時加進 `SETTING_DEFAULTS`，否則跨批分析會拒絕執行。

## DEF · 影像清單帶 CRLF，`--images` 一張都對不上

- **症狀**：`data/omniedit150 底下沒有符合 --images 的影像`，而清單裡的名字
  逐字看起來完全正確。
- **根因**：清單在 Windows 上產生再上傳，帶 CRLF。驅動腳本用
  `tr '
' ' '` 把它攤成一行，只換掉 LF，於是每個名字尾端掛著一個 CR。
- **修正**：驅動腳本在讀清單前檢查並就地去掉 CR。
- **同型**：`.gitattributes` 已把 `*.sh` 釘成 `text eol=lf`，但**資料檔不在
  那條規則裡**。凡是從 Windows 上傳、又要餵進 shell 的純文字清單都有這個坑。


## DEF · `PYTHONIOENCODING=utf-8` 會讓抓子行程輸出的測試變成 `TypeError`

- **症狀**：`tests/test_matched_distortion_table.py::test_兩個錨點都沒給時拋錯`
  失敗於 `TypeError: can only concatenate str (not "NoneType") to str`，
  而同一份程式碼在沒設那個環境變數時 1141 條全過。訊息完全沒提到編碼。
- **根因**：該測試用 `subprocess.run(..., capture_output=True, text=True)` 抓
  驅動腳本的中文錯誤訊息。`PYTHONIOENCODING` 會被子行程繼承，於是子行程把
  中文寫成 UTF-8；但 `subprocess` 的 `text=True` **不看 `PYTHONIOENCODING`**，
  它用 `locale.getencoding()`（本機是 cp950）解碼，UTF-8 的中文位元組解不開。
  讀取執行緒因此拋 `UnicodeDecodeError` 而死，`out.stderr` 變成 `None`，
  下一行的字串相加才報錯。真正的例外只出現在 pytest 的 warning 區。
- **修正**：跑測試時不要加 `PYTHONIOENCODING`。要根治的話是在那個
  `subprocess.run` 明確給 `encoding="utf-8"`——目前**沒有改**，因為改了會讓
  這條測試不再涵蓋「訊息以本地編碼寫出」的情形。
- **同型**：任何用 `capture_output=True, text=True` 抓中文輸出的測試都有這個
  坑。判斷方法是看 pytest 的 `PytestUnhandledThreadExceptionWarning`，
  失敗訊息本身不會說是編碼問題。
