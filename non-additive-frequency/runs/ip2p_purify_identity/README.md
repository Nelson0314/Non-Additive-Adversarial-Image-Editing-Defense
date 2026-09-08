# 抗淨化：攻擊方先淨化再編輯

現行威脅模型（受保護的臉逐位元不動、擾動全部放在衣物上、主讀數是編輯後還
認不認得出那個人）底下的抗淨化。**只讀已存的防禦圖，不重訓。**

驅動：`scripts/purify_identity.py`（單一 job）、
`scripts/purify_identity_round.sh`（派工）。身分讀數由
`scripts/identity_probe.py` 事後在 CPU 上補算。

## 四個臂

| 臂 | 防禦圖來源 | 未淨化時的成功數 | 在這張表裡回答什麼 |
|---|---|---|---|
| `free` | `runs/ip2p_face_defence/*_plain_*` | 14/25 | 內容軸最強的一格，淨化後剩多少 |
| `tint` | `runs/ip2p_content_constraint/tint_*` | 13/25 | 低頻懲罰。它是唯一兩軸同時改善的一格 |
| `rand` | `runs/ip2p_content_constraint/rand_*` | 3/25 | **同幾何、隨機內容、零最佳化的地板** |
| `floor` | 原圖本身 | — | 淨化算子自己造成的位移（空白地板） |

**`rand` 不可省。** 本專案已量到淨化之後最佳化相對同幾何隨機雜訊的優勢幾乎
消失（裁切 1.13、jpeg→resize 1.04，`runs/ip2p_patch_breadth_purify/README.md`）。
沒有這一臂就分不出「防禦守住了」與「衣服上有一塊很吵的東西」。
FaceLock、EditShield、FaceShield 三篇報的抗淨化都沒有這個對照
（`docs/reference/SURVEY_IDENTITY_EDITING.md` §6）。

## 十二個淨化算子與它們的出處

強度**不自己另立新數字**，沿用論文明載的值。逐項對照見
`docs/reference/SURVEY_IDENTITY_EDITING.md` §5。

| 標籤 | 設定 | 出處 | 幾何類 |
|---|---|---|---|
| `identity` | — | 保留率的分母，不可排除 | |
| `jpeg90` | q=90 | FaceLock | |
| `jpeg80` | q=80 | EditShield 的 JPEG Compression countermeasure | |
| `jpeg75` | q=75 | FaceLock；亦即真實世界串接協定的 C | |
| `jpeg60` | q=60 | FaceLock | |
| `blur1.5` | 高斯 σ=1.5 | FaceLock 的「k=5, σ=1.5」 | |
| `rotate10` | 隨機 ±10° | FaceLock「random rotation between (-10, 10)」 | ✓ |
| `crop_resize0.1` | 每邊裁 10% | DIA | ✓ |
| `jpeg_then_resize75` | JPEG75 → 0.5× Lanczos | arXiv:2604.23688 的 C&R 串接 | ✓ |
| `resize_only` | 降採樣再還原 | FaceShield 的 75%／50% 那一族 | ✓ |
| `quantize8` | 8 階 | FaceShield 的 3-bit | |
| `adverse_cleaner` | 導向濾波 | 本專案既有 | |

**`rotate` 是這一批新增的算子**（`src/purify/ops.py::rotate_random`）。
FaceLock 與 EditShield 的抗淨化表都有旋轉欄，本專案先前一個都沒有，
那一欄無法與任何外部方法對照。插值核（雙線性）與邊界填補（補零）
**論文未載，是本專案指定**，報表上標 `modified_from_paper`。

## 參照依算子分兩種

    幾何類    edit_orig := 編輯(p(原圖))
    其餘      edit_orig := 編輯(原圖)      ← 直接沿用來源目錄已存的那一張

`reference` 欄逐列記下該列踩的是哪一種（`purified_orig` 或 `orig`）。
判定用 `Purifier.kind` 不是標籤字串——`jpeg_then_resize75` 的字首是 `jpeg`。

幾何類換參照的理由是取景與像素格點：`crop_resize` 是繞中心的 1.2488× 放大、
`rotate` 讓四角離開畫面，就算完全沒有防禦，`編輯(p(原圖))` 與 `編輯(原圖)`
之間也會差很多（舊參照下實測空白地板 0.5506，而 LPIPS 在兩張不相干的自然
影像之間只飽和到 0.772）。其餘算子不換——它們的地板反映的是算子對**影像
內容**的破壞，那正是扣地板要扣掉的東西。

**非幾何類的 `edit_orig` 不重跑，直接沿用來源目錄的 PNG。** 燒測驗證過這是
對的：`tint`／`clothing`／`123744` 在 `identity` 算子下量到 `edit_lpips`
0.4238，來源 `results.csv` 是 0.4239，差的只有 PNG 的 uint8 量化。

**`floor` 的幾何類恰為 0**：兩側同算子、同輸入、同種子，逐位元相同。
量到非 0 表示編輯不是確定性的或種子沒對齊，`purify_identity.py` 直接拋錯
——那是量測本身壞了，不是一個小數字。

## 成本（實測）

| | 每格 |
|---|---|
| 非幾何算子（一次編輯） | 22 s |
| 幾何算子（兩次編輯） | 44 s |

每個 job（10 格 × 12 算子）約 59 分鐘，四臂三類共 12 個 job 約 **11.7 GPU-hr**。

## 讀這張表要注意的三件事

1. **結論要拉圖。** 這個專案已多次出現代理讀數與影像判讀相反：
   `face_found == False` 會在人眼看得見的臉上誤報、NIMA／CNNIQA 分不出
   「雜訊」與「印花」、注意力探測曾經量到的其實是區域面積。
2. **相差 1–2 格的排序讀不出來。** 同一個最佳化跑兩次不會得到同一張防禦圖
   （同卡差 2.65 灰階、換卡 8.27、`best_eval` 最多差 137%）。
3. **多人合照的身分讀數不可靠**（十張裡有四張：軍人、武士群像、五人合照、
   情侶）。MTCNN 只取一張臉，兩張圖取到不同人時餘弦沒有意義。
