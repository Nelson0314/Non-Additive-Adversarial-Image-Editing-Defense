# lab/ 交接

這個目錄是獨立的實驗室：所有程式、資料、產出都在這裡，需要的檔案從
`../anti-purification/` 複製或以符號連結指過去，**不寫回主線目錄**。

設計與理由在 `docs/DESIGN.md`。任務界定在 `docs/BRIEF.md`。
文獻與設計審查在 `docs/LITERATURE.md`、`docs/DESIGN_REVIEW.md`。

## 五個臂

| 臂 | 線 | 載體 | 最佳化 |
|---|---|---|---|
| `style_random` | 生成 | SDEdit 直接交付，`strength 0.35` | 無 |
| `style_low` | 生成 | SDEdit 直接交付，`strength 0.15` | 無 |
| `style_filter` | 生成＋分區 | SDEdit 的低頻色彩位移場套回原圖 | 無（二分搜尋定振幅） |
| `curve_dual_spatial` | 分區 | 兩條色調曲線 ＋ 平滑混合場 | 900 步增廣 Lagrange |
| `curve_dual_chroma` | 分區 | 單一全域色調曲線，兩道上限 | 900 步增廣 Lagrange |

臂的參數全部定義在 `scripts/defence_cmd.sh` 的 `case`，**只有那一個地方**。

## 現況

遠端 `/nfs/home/nelson0314/image-immunization/lab/`（home 跨機同步，兩台都看得到）。

三個臂已派工：`style_random`（basic-2 卡 0）、`curve_dual_spatial`
（basic-2 卡 1）、`curve_dual_chroma`（basic-1 卡 6）。
`style_filter` 與 `style_low` 排在兩台的 dispatcher 佇列裡，等卡。

- `style_random` 的防禦圖、兩個場景的編輯、淨化都已完成，正在跑淨化後編輯。
- 兩個 curve 臂在跑 900 步的求解。單張約 45–70 分鐘，八張約 6–9 小時。

## 怎麼看現在跑到哪

```
ssh -p 10102 nelson0314@server.basiclab.lab.nycu.edu.tw
cd /nfs/home/nelson0314/image-immunization
ls ~/lab_leases/                       # 目前佔著哪幾張卡（跨機，上限五張）
tail -3 lab/runs/logs/<臂名>.log        # 單一臂的進度
tail -5 lab/runs/logs/dispatch_b1.log  # 排程器（b1／b2 各一份）
ls lab/runs/state/                     # 已完成的階段，一個 sentinel 一個階段
```

`lab/runs/state/<臂>.<階段>.done` 只在該階段 rc=0 時才寫。重跑 `arm_chain.sh`
會跳過已完成的階段，中途被砍掉的階段會重跑。

## 鏈的順序

`scripts/arm_chain.sh <GPU> <臂名>`，一張卡跑完一個臂：

1. `defence` → `lab/runs/defence/<臂>/`
2. `edit_ip2p`、`edit_inpaint` → `lab/runs/edit_defended/<臂>/`
3. `purify` → `lab/runs/purified/<臂>/<算子>/`
4. `pedit_<算子>_<場景>` × 14 → `lab/runs/edit_purified/<臂>/<算子>/`

所有臂跑完之後再跑一次跨臂讀數：

```
bash lab/scripts/readout.sh <GPU>
```

產出 `lab/results/displacement.csv` 與 `lab/results/retention.csv`。

## 分母不重跑

`lab/runs/edit_preflight` 與 `lab/runs/edit_purified/undefended` 是**符號連結**，
指向主表已產好的 `runs/edit_preflight`（`ip2p_si18`、`inpaint_undefended`）與
`runs/edit_purified/undefended`。重跑會換掉分母，條件之間就不再可比。

## 排程器

`scripts/dispatch.sh <臂名>...` 在遠端輪詢（180 秒），有空卡就派一個臂。
全域五張的上限用 NFS 上的租約目錄 `~/lab_leases/` 實作，一張卡一個檔，
**跨機計數**。持有者不在了的租約由 `reap` 清掉。

兩台各跑一份 dispatcher，佇列不重疊：

| 主機 | 佇列 | log |
|---|---|---|
| basic-2 | `curve_dual_spatial`、`style_filter` | `lab/runs/logs/dispatch_b2.log` |
| basic-1 | `curve_dual_chroma`、`style_low` | `lab/runs/logs/dispatch_b1.log` |
| basic-2 | `style_random`（已派出） | `lab/runs/logs/dispatch_style.log` |

## 已經量到的

- **`style_random`（`strength 0.35`）的防禦圖把人換掉了。** 八張的 FaceNet
  餘弦 0.26–0.76，三張低於 0.55。失真反而比 `colour_curve_ours` 小
  （LPIPS 0.19–0.30 對 0.3367、PSNR 22.6–25.4 對 16.67）。
  逐張數在 `runs/defence/style_random/results.csv`。
- **低頻轉移把換人的部分擋掉了。** `style_filter` 的兩張試跑裡，SDEdit 原始
  輸出的身分餘弦是 0.14／0.31，經低頻轉移交付的是 0.62／0.87。
- **兩道上限在 `curve_dual_spatial` 上確實同時綁得住。** 20 步的試跑裡臉那道
  一開始違反 +0.848，第 10 步起可行，終點 `cap_violations 0`。

## 下一步

1. 等五個臂的鏈跑完（`ls lab/runs/state/ | wc -l`，每個臂 18 個 sentinel）。
2. 跑 `scripts/readout.sh`。
3. **逐格看圖**。本專案的規矩是成立與否看圖，不是看位移的數字。
   要看的四樣：防禦圖本身（有沒有換人、有沒有接縫）、過渡帶的特寫
   （`curve_dual_spatial`、`style_filter` 有 `__field.png` 可對照）、
   防禦後的編輯、淨化後的編輯。
4. 還沒做：**`style_opt`**（對風格 embedding 做最佳化）。設計審查建議把
   embedding 限制在風格字典的凸包上（`e(a) = Σ a_j·CLIP(content+style_j)`，
   `a_j ≥ 0`、`Σ a_j = 1`）而不是自由的 77×768 張量，代理改用受害 UNet 的
   條件 ε-prediction 誤差（ip2p 走 8 通道、inpaint 走 9 通道的真正條件入口）。
   成本估計：全鏈 checkpoint、N=20、100 次更新約 25–70 分鐘／張。
   理由與四個問題的完整回答在 `docs/DESIGN_REVIEW.md`。
5. 還沒做：`metrics_union.py` 在 lab 上跑不起來（它的路徑寫死指向
   `main_table/`），失真與美學那幾張補充表暫時沒有。

## 論文

使用者指定的那一篇是

> Yibo Wang, Yong Zhou, Bing Liu, Rui Yao,
> *Style-controllable adversarial example generation via image editing and
> prompt embedding optimization*, Neurocomputing, 2026.
> DOI `10.1016/j.neucom.2026.134591`（PII `S0925231226019892`）。

**全文沒有取得**：ScienceDirect 對非機構 IP 回 403，使用者給的簽章 URL 已逾期
（`X-Amz-Expires=300`），Elsevier API 無金鑰只回 coredata，OpenAlex 與
Semantic Scholar 都沒有摘要，非開放取用。已取得的 metadata 在
`docs/paper/neucom_134591.json`。要全文請在有機構授權的網路上自行下載。
