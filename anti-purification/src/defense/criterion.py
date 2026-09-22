"""使用者判準的目標函數：三個條件取軟極小。

判準
────────────────────────────────────────────────────────────────────
攻擊者要拿到的是一張**可用的**、**認得出是同一個人的**、**指令已完成**的
照片。三個條件是連言，**任一條垮掉就算防禦成功**，所以防禦方要最小化的是三項
的極小值，不是三項的和，也不是三項的極大值。

先前 `scripts/color_ceiling.py` 的 `search_objective` 用的是等權重的和
（`subject_id_def + clip_s_def`）。兩個問題：兩項的動態範圍差 4–5 倍
（實測 5–95% 是 [0.303, 0.978] 對 [−0.015, 0.136]），於是身分項主導；而且
「保住身分但指令對齊少一點」與「身分整個毀掉」可以拿到同分，那不是連言。

正規化
────────────────────────────────────────────────────────────────────
每一項都除以**同一格未防禦攻擊**的對應值，所以 1.0 代表「這個條件跟沒有防禦
時一樣完好」，0.0 代表「這個條件垮了」。逐格正規化而不是逐批，因為三個量的
絕對值隨影像與指令而變。

指令那一項量的是**對齊增益**：`align(編輯輸出) − align(該條線自己的輸入)`，
兩側各自扣掉自己的輸入，防禦本身把圖推近或推遠指令句的那一段就被扣掉了。
只用終點對齊不行：CLIP 的絕對相似度有很大的常數項，量到的是 0.289 對 0.298，
動態範圍被壓到 3%，那一項在軟極小裡永遠不會是最小的那個。也不是方向
（`E_img(編輯) − E_img(輸入)` 對指令句的投影）。方向分數在這裡不可用：防禦本身
就把顏色推掉一大段，那個位移進到差向量裡，分數被沖到零，而圖上指令完成得好好
的。`runs/carrier_search_probe/` 第一批 18 列有 8 列是這樣拿到 0 分的，逐張看圖
確認過。方向分數仍照報，只是不進目標函數。

未防禦的那一張本身就不成立時（臉找不到、指令沒往前走），該項記 1.0——防禦
拿不到分。那是**攻擊失敗**，不是防禦成功，要用旗標分開記。

主體不見了
────────────────────────────────────────────────────────────────────
「編輯輸出上偵測不到臉」不可以當成身分項的滿分。先前的程式把它記成 `0.0`，
等於明著獎勵搜尋去把顏色推到偵測器失手，而人眼看過去那張臉還在。
`runs/carrier_search_probe/` 第一批 18 列裡有 10 列是這樣拿到 0 分的。

現行的作法是在呼叫端就不讓這個狀態出現：`metrics.identity.anchored_identity`
在**原圖主體框的固定座標**上裁編輯輸出取嵌入，偵測器不參與，`subject_iou`
傳 1.0。偵測式的讀數另外記在 `*_detected` 與 `subject_box_iou_*`。
`subject_lost` 因此只在真的取不到嵌入時才會是 1。
"""
from __future__ import annotations

import math
from typing import Dict, Optional

TERMS = ('id_norm', 'use_norm')
REPORTED = ('id_norm', 'dir_norm', 'use_norm')


def _value(v) -> Optional[float]:
    if v is None or v in ('', 'nan'):
        return None
    v = float(v)
    return None if v != v else v


def _ratio(defended, control, eps: float) -> float:
    return min(1.0, max(0.0, defended / control)) if control > eps else 1.0


def normalise_terms(*, id_def, id_control, clip_def, clip_control,
                    quality_def, quality_control, subject_iou,
                    eps: float = 1e-3) -> Dict[str, float]:
    """三項各自正規化到 [0,1]，1.0 表示該條件與未防禦時一樣完好。

    `quality_*` 是無參考品質分數，**越小越自然**（NIQE 那一族），所以比值取
    倒過來。`subject_iou` 是主體框與編輯輸出中最接近的臉框的 IoU。
    """
    row: Dict[str, float] = {}
    idc, idd = _value(id_control), _value(id_def)
    iou = _value(subject_iou) or 0.0
    row['control_subject_lost'] = int(idc is None)
    row['subject_lost'] = int(iou <= 0 or idd is None)
    if idc is None:
        row['id_norm'] = 1.0
    elif row['subject_lost']:
        row['id_norm'] = 0.0
    else:
        row['id_norm'] = _ratio(idd, idc, eps)

    cc, cd = _value(clip_control), _value(clip_def)
    row['control_direction_flat'] = int(cc is None or cc <= eps)
    row['dir_norm'] = 1.0 if (cc is None or cd is None) else _ratio(cd, cc, eps)

    qc, qd = _value(quality_control), _value(quality_def)
    if qc is None or qd is None or qd <= eps:
        row['use_norm'] = 1.0
    else:
        row['use_norm'] = _ratio(qc, qd, eps)
    return row


def softmin(values, tau: float = 0.05) -> float:
    """以 `softmax(-v/tau)` 為權重的加權平均。

    `tau → 0` 時趨近 `min`，`tau → ∞` 時趨近算術平均，值域因此不會跌出各項的
    範圍。用它而不是硬 `min`：硬極小在平原上只對當下最弱的那一項有選擇壓力，
    子代之間常常同分；加權平均讓另外兩項仍有微弱的貢獻，排序不會整片打平。

    **不要用 log-sum-exp 的那個軟極小**：它是 `min` 的平滑下界，會跌到所有項
    之下，分數就不再與三項同尺度。
    """
    vs = [float(v) for v in values]
    if not vs:
        raise ValueError('沒有可取極小的項')
    if tau <= 0:
        return min(vs)
    m = min(vs)
    ws = [math.exp(-(v - m) / tau) for v in vs]
    total = sum(ws)
    return sum(w * v for w, v in zip(ws, vs)) / total


def criterion_score(terms: Dict[str, float], tau: float = 0.05,
                    keys=TERMS) -> float:
    """要最小化的純量。缺項直接拋錯，不補預設值。

    **`dir_norm` 不在 `TERMS` 裡。** 指令那一項照算照報，但不進目標函數：三種
    寫法（方向分數、終點對齊、對齊增益）都在圖上被推翻過——搜到的點被判成
    「指令沒完成」，而防禦後的編輯圖上衣服確實換成指令要的顏色、人也認得出來。
    CLIP 在這個粒度上的增益只有 1% 上下，與防禦自己造成的位移同量級，量不出
    那個條件。量不準的項不可以主導連言，所以現在最小化的是身分與可用性兩項，
    指令那一項留在 CSV 裡等一個量得準的做法（VQA 或人眼）。
    """
    missing = [k for k in keys if k not in terms]
    if missing:
        raise KeyError(f'缺少 {missing}；列出的項都要有才談得上連言')
    return softmin([terms[k] for k in keys], tau)
