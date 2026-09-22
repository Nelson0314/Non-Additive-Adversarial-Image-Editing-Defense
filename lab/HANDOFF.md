# lab/ 交接

這個目錄是獨立的實驗室：所有程式、資料、產出都在這裡，需要的檔案從
`../anti-purification/` 複製或以符號連結指過去，**不寫回主線目錄**。

設計與理由在 `docs/DESIGN.md`。任務界定在 `docs/BRIEF.md`。
文獻與設計審查在 `docs/LITERATURE.md`、`docs/DESIGN_REVIEW.md`。

## 六個臂

| 臂 | 線 | 載體 | 最佳化 |
|---|---|---|---|
| `style_random` | 生成 | SDEdit 直接交付，`strength 0.35` | 無 |
| `style_low` | 生成 | SDEdit 直接交付，`strength 0.15` | 無 |
| `style_filter` | 生成＋分區 | SDEdit 的低頻色彩位移場套回原圖（高斯低通） | 無（二分搜尋定振幅） |
| `style_filter_guided` | 生成＋分區 | 同上，但平滑走 guided filter（導引＝原圖亮度） | 無（二分搜尋定振幅） |
| `curve_dual_spatial` | 分區 | 兩條色調曲線 ＋ 平滑混合場 | 900 步增廣 Lagrange |
| `curve_dual_chroma` | 分區 | 單一全域色調曲線，兩道上限 | 900 步增廣 Lagrange |

臂的參數全部定義在 `scripts/defence_cmd.sh` 的 `case`，**只有那一個地方**。

## 現況

遠端 `/nfs/home/nelson0314/image-immunization/lab/`（home 跨機同步，兩台都看得到）。

進度以 sentinel 為準，一個臂 18 個：

```
for a in style_random style_low style_filter style_filter_guided          curve_dual_spatial curve_dual_chroma; do
  printf "%-22s %s/18
" "$a" "$(ls -1 lab/runs/state/${a}.*.done 2>/dev/null | wc -l)"
done
```

- `style_random` 已經跑完整條鏈（18/18）。
- 兩個 curve 臂在跑 900 步的求解。**實測單張約 105 分鐘**（比原估的 45–70
  分鐘慢，`resample` 的固定驗證抽樣與 `probe_every` 都要額外前向），
  八張約 14 小時。
- `style_low` 與 `style_filter_guided` 在等卡。

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
| basic-1 | `style_low`（重派，原本那次 OOM） | `lab/runs/logs/dispatch_b1_retry.log` |
| basic-2 | `style_filter_guided` | `lab/runs/logs/dispatch_b2_guided.log` |
| basic-2 | `style_random`（已完成） | `lab/runs/logs/dispatch_style.log` |

**鏈失敗會換一張卡重試**（最多八次）。已經踩過一次：`--assert` 通過之後、
權重還沒載完的二十秒內，另一位使用者把同一張卡佔走 19.65 GiB，工作 OOM
而死。那不是程式的錯，但排程器不重試的話那個臂會永遠停在那裡。

## 已經量到的

- **`style_random`（`strength 0.35`）的防禦圖把人換掉了。** 八張的 FaceNet
  餘弦 0.26–0.76，三張低於 0.55。失真反而比 `colour_curve_ours` 小
  （LPIPS 0.19–0.30 對 0.3367、PSNR 22.6–25.4 對 16.67）。
  逐張數在 `runs/defence/style_random/results.csv`。
- **低頻轉移把換人的部分擋掉了。** `style_filter` 的兩張試跑裡，SDEdit 原始
  輸出的身分餘弦是 0.14／0.31，經低頻轉移交付的是 0.62／0.87。
- **兩道上限在 `curve_dual_spatial` 上確實同時綁得住。** 20 步的試跑裡臉那道
  一開始違反 +0.848，第 10 步起可行，終點 `cap_violations 0`。
- **`style_filter` 的八張都保住了身分**：FaceNet 餘弦 0.66–0.97，而同一批
  SDEdit 原始輸出是 0.007–0.32。臉那道上限八張全部貼著 8.0，整圖是 8.8–13.1
  ——整圖那道沒有綁住，綁住的是 `--max-scale 3.0`，所以這一臂的失真**低於**
  `colour_curve_ours` 的 ΔE00 16，不是等失真比較。
- **`style_filter` 的高斯低通在主體輪廓外留下一圈光暈。** `strength 0.6` 下
  SDEdit 已經把髮型與輪廓換掉，`Δ` 含大量結構差異，高斯低通把它抹過輪廓。
  α 場本身沒有問題（`__field.png` 是平滑橢圓，沒有硬邊），問題在 `Δ_lf`。
  `style_filter_guided` 是對這件事的處理：平滑改走 guided filter，只在原圖的
  同質區內平均，色彩的轉折因此落在主體自己的輪廓上。兩臂其餘逐項相同。

## 下一步

1. 等六個臂的鏈跑完（`ls lab/runs/state/ | wc -l`，滿是 108）。
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
