# lab/ 交接

**這個目錄是獨立的實驗室。所有程式、資料、產出都在 `lab/` 裡面。**
需要的檔案從 `../anti-purification/` 複製，或以符號連結指過去；
**對主線目錄一律唯讀，連 `docs/` 也不寫。**

| 要什麼 | 讀哪份 |
|---|---|
| 現況、還在跑什麼、怎麼查 | 本檔 |
| 每個臂為什麼長這樣、量到什麼 | `docs/DESIGN.md` |
| 任務界定與威脅模型 | `docs/BRIEF.md` |
| 文獻與設計審查 | `docs/LITERATURE.md`、`docs/DESIGN_REVIEW.md` |

---

## 現況

**十一個臂跑完整條鏈，跨臂讀數已出；第十二個臂 `style_opt` 在跑。**

| 檔 | 列數 | 內容 |
|---|---|---|
| `results/displacement.csv` | 704 | 位移，11 臂 × 8 影像 × 4 指令 × 2 場景 |
| `results/retention.csv` | 4,928 | 保留率，11 臂 × 7 算子 × 64 |
| `results/fidelity.csv` | 88 | 四個失真指標逐張（LPIPS／ΔE00／PSNR／L∞／rms） |

報告頁（v2，十一臂）<https://claude.ai/artifact/VdWw6PoWtrQd2xTLc5Qyvt>

### 正在跑的東西

遠端一支 `scripts/finish_batch.sh`（`nohup setsid`，log 在
`runs/logs/finish_batch.log`）佔 basic-2 一張卡，依序做三件事：

1. ~~十一臂讀數~~ **已完成**
2. `style_opt` 的完整鏈（試跑已過 rc=0；防禦圖約 9 分鐘一張）
3. **十二臂讀數**——只有 `style_opt` 的 sentinel 滿 18 才跑

第 3 步跑完之後**要重建報告頁**（見下「怎麼重建報告頁」）。
注意退役之後臂數是 **10**，不是 12。

---

## 十個現行的臂

| 臂 | 線 | 載體 | 最佳化 | 預算 |
|---|---|---|---|---|
| `inpaint_outside_face` | 不碰受保護區 | 只保留臉，其餘全部重繪 | 無 | **無** |
| `inpaint_bg` | 不碰受保護區 | 只重繪主體之外 | 無 | **無** |
| `style_affine` | 生成濾鏡 | 三通道局部仿射色彩轉移 | 無 | ΔE00 16／臉 8 |
| `style_filter` | 生成濾鏡 | 低頻色彩場，高斯低通 | 無 | ΔE00 16／臉 8 |
| `style_filter_guided` | 生成濾鏡 | 同上，單通道 guided filter | 無 | ΔE00 16／臉 8 |
| `style_opt` | 生成濾鏡 | 風格由最佳化在字典凸包上選 | 60 步 | ΔE00 16／臉 8 |
| `curve_dual_spatial` | 分區預算 | 兩條曲線 ＋ 平滑混合場 | 900 步 | ＋帶 TV ≤ 2 |
| `curve_dual_spatial_anchored` | 分區預算 | 同上，起點改成該張自己的膚色色調 | 900 步 | ＋彩度、同色 |
| `curve_dual_chroma` | 分區預算 | 單一全域曲線，預算分在色度 | 900 步 | ΔE00 16／臉 8 |
| `ab_warp` | 分區預算 | CIELAB `(a,b)` 平面的 RBF 位移場 | 900 步 | ＋彩度、同色 |

**臂的參數全部定義在 `scripts/defence_cmd.sh` 的 `case`，只有那一個地方。**

**已退役**：`style_random`、`style_low`（直接交付 SDEdit 輸出）。
逐格圖已刪，防禦圖與配方留著。理由見 `docs/DESIGN.md`「已退役」一節。

---

## 怎麼跑

```
bash scripts/arm_chain.sh <GPU> <臂名>     # 一張卡跑完一個臂
bash scripts/dispatch.sh <臂名>...          # 等空卡、自動派、失敗換卡重試
bash scripts/readout.sh <GPU>               # 跨臂讀數（displacement + retention）
python code/defence_fidelity.py --out results/fidelity.csv   # 四個失真指標
```

鏈的四個階段：`defence` → `edit_ip2p`／`edit_inpaint` → `purify` →
`pedit_<算子>_<場景>` × 14，共 **18 個 sentinel**，寫在 `runs/state/`，
**只在該階段 rc=0 時才寫**。重跑會跳過已完成的階段。

### 實測成本

| 類型 | 防禦圖 | 其餘鏈 | 合計 |
|---|---|---|---|
| 不做最佳化的臂 | 0.02 h | 1.50 h | **1.5 GPU-h** |
| 900 步求解的臂 | 8.1 h | 1.48 h | **9.6 GPU-h** |
| `style_opt`（60 步） | 約 1.2 h | 1.5 h | 約 2.7 GPU-h |

鏈的 1.5 GPU-h 與臂無關（64 格編輯 ＋ 7 道淨化 ＋ 448 格淨化後編輯）。

---

## GPU 規則（踩過才寫的）

- **五張卡的上限是兩個 session 加起來**，不是每人五張。
- 租約目錄 `~/lab_leases/`，一張卡一個檔，格式 `<主機> <pid> <名稱>`，
  **跨機計數**。`reap()` 依 pid 存活清理。另一個 session 也用同一份。
- `scripts/free_cards.sh` **會把自己正在用的卡報成「空」**（它只扣別人的
  佔用）。判「這張能不能拿」一定要同時看租約目錄。
- **鏈失敗會換一張卡重試**（最多八次）。`--assert` 通過之後、權重載完之前
  被別人佔走是常態，**單一個 `rc≠0` 是正常形狀**，不是故障訊號。
  真正該回報的是 `[GIVE-UP]` 與「全停了」。
- **改遠端的腳本一定要 staging ＋ `mv`，不可以就地覆寫。** `tar x` 是截斷
  重寫同一個 inode，而 bash 邊讀邊執行，正在跑的排程器會死於
  `Stale file handle`，log 上只有那一行。

```
tar xzf ../lab.tgz -C .stage
cd .stage && find lab -type f | while read -r f; do
  mkdir -p "../$(dirname "$f")"; mv -f "$f" "../$f"
done
```

- **取卡的函式，紀錄一律 `>&2`。** `GPU=$(claim)` 會把 stdout 的紀錄一起收走，
  `CUDA_VISIBLE_DEVICES` 變成一串文字，**torch 靜默退回 CPU**——所有結果都
  算得出來、格式也對，只是慢兩個數量級（一次 20 分鐘的讀數跑了七小時）。
  `finish_batch.sh` 拿到卡之後有兩道守門：卡號必須是純數字，
  `torch.cuda.is_available()` 必須為真，否則寫 `[FATAL]` 停住。
  那一次的紀錄留在 `runs/logs/finish_batch_cpu_bug.log` 與
  `runs/_stale_STYLE_OPT_NEEDS_DECISION.cpu_bug`。

---

## 分母不重跑

`runs/edit_preflight` 與 `runs/edit_purified/undefended` 是**符號連結**，
指向主表已產好的 `../runs/edit_preflight`（`ip2p_si18`、`inpaint_undefended`）
與 `../runs/edit_purified/undefended`。**重跑會換掉分母，條件之間就不再可比。**

---

## 怎麼重建報告頁

```
# 遠端：併拼圖（檔案數上限 256，所以逐格圖要併成 sprite sheet）
python lab/scripts/build_report_assets.py --out lab/report/img --quality 70
# 本機：組 data.js
python lab/scripts/build_report_data.py --out lab/report/data.js
# 發佈：同一個 file_path 即更新同一個網址
```

- **單一版本上限 64 MB。** q80 的編輯圖會到 62 MB，太貼近，現行是 q70（54.4 MB）。
- 防禦圖、混合場、色度平面、pipeline 原輸出是**無損**（自然度要逐像素判）；
  編輯圖是有損，這件事寫在頁面上不藏。
- `build_report_data.py` 的 `PENDING` 清單讓還沒跑完的臂**照樣出現在每一張表**，
  格子寫「還沒跑完」——表的形狀在資料進來之前就固定，補資料不必改版面。

---

## 三個還沒做的

1. **逐格看圖。** 位移與保留率都是數字，而本專案的規矩是成立與否看圖。
   報告頁的兩個並列區就是為這件事做的，但 5,632 格沒有人逐格看過。
2. **等失真對齊。** 十個臂掛在三種預算上，四個臂**沒有可縮的旋鈕**
   （兩個 `inpaint_*` 的重繪區是生成的、`style_affine` 的旋鈕是濾波器半徑）。
   主表那條線已經把十個 baseline 對齊到 LPIPS 0.3344，它的編輯端還在排。
3. **`metrics_union.py` 在 lab 上跑不起來**（路徑寫死指向 `main_table/`），
   FSIM 與美學那幾張補充表沒有。

---

## 與另一個 session 的分工

`image-immunization-5e` 在做**主表十二條件的等失真對齊**，用同一批卡、同一個
租約目錄。它那邊的產出在 `../main_table/results/aligned/`（commit `2daeff9`），
**是另一批的數字，引用要連協定一起引用**。兩邊來回辯出來的結論寫在
`docs/DESIGN.md`「等失真比較」一節。它接下來要跑編輯端，需要兩到三張卡。

---

## 論文

使用者指定的那一篇：Yibo Wang, Yong Zhou, Bing Liu, Rui Yao,
*Style-controllable adversarial example generation via image editing and prompt
embedding optimization*, Neurocomputing 2026, DOI `10.1016/j.neucom.2026.134591`。

**全文沒有取得**：ScienceDirect 對非機構 IP 回 403，簽章 URL 已逾期，
非開放取用。已取得的 metadata 在 `docs/paper/neucom_134591.json`。
本目錄的設計依據是標題給出的兩個元件與本專案自己的量測，**不是論文內文**。
