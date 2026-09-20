"""補丁的**載體**：用人體解析指出「衣服在哪裡」，讓花紋落在自然的地方。

為什麼需要它
────────────────────────────────────────────────────────────────────
`PatchParam` 原本的支撐是主體遮罩的補集（或補集裡的一塊方塊）。補集包含牆面
與地板，於是學出來的花紋貼在牆上，讀起來是一塊壞掉的區域。本模組把支撐再
交集一層**語意區域**：衣物。花紋止於衣物輪廓，讀起來是布料上的印花。

「補丁要落在語意上合理的位置」在文獻裡是標準作法——PhysPatch
（[arXiv:2508.05167](https://arxiv.org/abs/2508.05167)）稱之為 semantic-based
mask initialization；Adversarial T-shirt（ECCV 2020）則是把圖案當成印在布上的
東西來建模。本模組只做「指出載體」這一半，**位置不最佳化**。

模型與類別
────────────────────────────────────────────────────────────────────
`mattmdjaga/segformer_b2_clothes`：SegFormer-B2 在 ATR 上微調，18 類。
本模組用到的三個載體都是 ATR 類別編號的集合，**編號寫錯不會拋錯、只會指到
別的東西**，故由測試釘住。

三個本專案指定的參數（無論文出處，故是 CSV 欄位不是註解）
────────────────────────────────────────────────────────────────────
- `kind`：載體的類別集合（`clothes`／`upper`／`pants`）。
- `min_area` / `max_area`：面積的可用區間，**出界拋錯**。太小表示這張影像上
  沒有衣物（支撐會退化），太大表示解析器把整張圖都算成衣服。
- `carrier_source`：模型 repo 字串，逐列寫進 CSV 供重現。

**不做膨脹與羽化。** `subject_mask` 往外羽化是為了「寧可多保留一點主體」；
載體相反——往外只會讓花紋溢出衣物輪廓，而那正是要避免的不自然。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

CARRIER_REPO = "mattmdjaga/segformer_b2_clothes"

# ATR 的 18 類。留全表是為了讓「編號指到什麼」在原始碼裡看得見。
ATR_LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
    5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe",
    10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg",
    14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf",
}

# 四個載體。`clothes` 是接下來兩個的聯集；`background` 不是衣物，見下。
CARRIER_CLASSES: Dict[str, Tuple[int, ...]] = {
    "clothes": (4, 5, 6, 7),
    "upper": (4, 7),
    "pants": (5, 6),
    "background": (0,),
    # **空的類別集合＝不由語意限制，整張畫面都可以放。** 存在理由是「同面積
    # 的花紋放在別的地方效果如何」這個對照：用 ATR 的背景類別再侵蝕到同面積
    # 會長出一個貼著人形輪廓的怪形狀，那時變的就不只是位置，還有形狀。
    # `frame` 把形狀交給後面的旋鈕決定——`match_area(mode="scale")` 給均勻
    # 鋪滿整張畫面，`scatter_support` 給幾塊分散的固定形狀——兩者都與語意無關。
    # 受保護的臉仍然由支撐的構造（`w = 1[m ≤ 0] · c`）與挖臉框兩道排除。
    "frame": (),
}

FACE_CLASS = 11

# 本專案指定的預設值。改動要進 CSV。
MIN_AREA = 0.01
MAX_AREA = 0.60

MIN_FACE = 0.01

_MODEL = None
_PROC = None


def _load(device):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import (AutoModelForSemanticSegmentation,
                                  SegformerImageProcessor)

        _PROC = SegformerImageProcessor.from_pretrained(CARRIER_REPO)
        _MODEL = AutoModelForSemanticSegmentation.from_pretrained(
            CARRIER_REPO).to(device).eval()
    return _MODEL, _PROC


def carrier_from_seg(seg: torch.Tensor, kind: str,
                     min_area: float = MIN_AREA,
                     max_area: float = MAX_AREA,
                     min_face: float = MIN_FACE) -> torch.Tensor:
    """(H,W) 的類別圖 ＋ 載體名稱 → (1,1,H,W) 的 0/1 載體遮罩。

    與 `carrier_mask` 拆開是為了讓「哪些類別算是衣服」與「面積守門」在不載
    權重的情況下受測。
    """
    if kind not in CARRIER_CLASSES:
        raise ValueError(
            f"kind 必須是 {'／'.join(CARRIER_CLASSES)} 之一，收到 {kind!r}")
    if seg.dim() != 2:
        raise ValueError(f"seg 必須是 (H,W) 的類別圖，收到 {tuple(seg.shape)}")
    face = float((seg == FACE_CLASS).to(torch.float32).mean())
    if face < min_face:
        raise ValueError(
            f"臉的面積只有 {face:.4f}，低於 {min_face}：這張影像上很可能"
            "**沒有人**。ATR 沒有「沒有人」這個輸出，毛皮、菌傘、皮膚都會被"
            "標成 Upper-clothes（實測貓 0.5311、蘑菇 0.3490、手掌 0.2929），"
            "把那些當成衣物載體會讓花紋貼在動物身上而讀數看起來完全正常。")
    ids = CARRIER_CLASSES[kind]
    if not ids:
        # 空集合＝整張畫面。**仍然走上面那道有人守門**：這一族的對照全部是
        # 人像，而「這張圖裡有沒有人」與「花紋放哪裡」是兩件事。
        m = torch.ones_like(seg, dtype=torch.float32)
    else:
        m = torch.zeros_like(seg, dtype=torch.float32)
        for i in ids:
            m = torch.maximum(m, (seg == i).float())
    area = float(m.mean())
    if not (min_area <= area <= max_area):
        raise ValueError(
            f"載體 {kind!r} 的面積 {area:.4f} 落在 [{min_area}, {max_area}] 之外"
            f"（類別 {ids} = {[ATR_LABELS[i] for i in ids]}）。"
            "太小表示這張影像上沒有可用的衣物，太大表示解析器把整張圖都算成"
            "衣服。**不靜默接受**：空載體會讓支撐退化成補集，而報表上的"
            "patch_carrier 欄仍然寫著這個名稱。")
    return m[None, None]


@torch.no_grad()
def parse_atr(x01: torch.Tensor, device=None) -> torch.Tensor:
    """(1,3,H,W) → (H,W) 的 ATR 類別圖，已放大回輸入尺寸。"""
    from PIL import Image

    dev = device or x01.device
    model, proc = _load(dev)
    h, w = x01.shape[-2:]
    arr = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    pil = Image.fromarray(arr.astype("uint8"))
    inputs = proc(images=pil, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    logits = model(**inputs).logits
    up = F.interpolate(logits.float(), size=(h, w), mode="bilinear",
                       align_corners=False)
    return up.argmax(dim=1)[0]


def carrier_mask(x01: torch.Tensor, kind: str,
                 min_area: float = MIN_AREA,
                 max_area: float = MAX_AREA,
                 min_face: float = MIN_FACE,
                 device=None) -> torch.Tensor:
    """(1,3,H,W) ＋ 載體名稱 → (1,1,H,W)，1 = 可以放花紋的地方。

    回傳的遮罩**還沒有**與主體補集取交集——那一步在 `PatchParam.reset` 裡做，
    因為主體遮罩是逐圖的、且「主體逐位元不動」必須由支撐的構造保證。
    """
    seg = parse_atr(x01, device=device)
    m = carrier_from_seg(seg, kind, min_area=min_area, max_area=max_area,
                         min_face=min_face)
    return m.to(device=x01.device, dtype=x01.dtype)


# ── 載體的貼合與融入 ────────────────────────────────────────────────
#
# 產物的不自然有很大一部分來自**邊界本身**，不是花紋內容：ATR 的分割是粗的，
# 硬 0/1 邊界會讓花紋溢出到皮膚、或在衣物中途硬切。下面四個函式是四個正交的
# 旋鈕，全部**永不增加**任何一個像素的權重——主體那一側恆為 0 的保證因此不
# 受影響。


def erode_mask(m: torch.Tensor, radius: int) -> torch.Tensor:
    """圓形結構元素的侵蝕（邊界往**內**縮）。`radius = 0` 時是恆等。

    用最小池化實作，是 `subject_mask._dilate` 的對偶。花紋因此永遠停在衣物
    輪廓之內，不會壓到皮膚或背景。
    """
    if radius <= 0:
        return m
    k = 2 * radius + 1
    return -F.max_pool2d(-m, kernel_size=k, stride=1, padding=radius)


def feather_inward(m: torch.Tensor, width: int) -> torch.Tensor:
    """往**內**羽化：邊界附近的權重由 1 線性降到 0，內部仍是 1。"""
    if width <= 0:
        return m
    acc = m.clone()
    cur = m
    for _ in range(width):
        cur = erode_mask(cur, 1)
        acc = acc + cur
    return (acc / (width + 1)).clamp(0.0, 1.0)


def _box(t: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    return F.avg_pool2d(F.pad(t, (radius,) * 4, mode="reflect"),
                        kernel_size=k, stride=1)


def guided_refine(m: torch.Tensor, guide: torch.Tensor, radius: int,
                  eps: float) -> torch.Tensor:
    """導引濾波：把粗的分割邊界**吸附到影像本身的邊緣**。

    He et al. 的 guided filter，導引影像取灰階的原圖。ATR 的輸出是 352² 放大
    回來的，邊界在細節上（袖口、領口、衣襬）與真實輪廓差好幾個像素；用影像
    自己當導引可以把邊界拉回去，而**不需要任何新模型**。

    `radius = 0` 時是恆等。輸出夾在 [0,1]。
    """
    if radius <= 0:
        return m
    g = (0.299 * guide[:, 0:1] + 0.587 * guide[:, 1:2] + 0.114 * guide[:, 2:3])
    mean_g, mean_m = _box(g, radius), _box(m, radius)
    cov = _box(g * m, radius) - mean_g * mean_m
    var = _box(g * g, radius) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_m - a * mean_g
    return (_box(a, radius) * g + _box(b, radius)).clamp(0.0, 1.0)


def ring_support(face_mask: torch.Tensor, inner: int, outer: int
                 ) -> torch.Tensor:
    """緊貼受保護主體外緣的**環帶**：往外 `inner` 到 `outer` 像素之間。

    點陣賭的是「碰到所有 token」，環帶賭的是「碰到**對的** token」。兩者
    互補：若環帶用 3–5% 的面積打得贏衣服上的 23%，這個方法的定位就從
    「大面積擾動」變成「精準的小標記」。

    `inner` 是與主體之間留的間隙。**不可以是 0**：主體遮罩已經往外羽化過
    （`face_subject_mask` 預設膨脹 16、羽化 24），貼著零邊界會讓支撐與羽化帶
    重疊，而「主體那一側恆為 0」的保證是靠 `w = 1[m ≤ 0] · c` 給的——重疊處
    的權重會被那個硬條件砍掉，於是實際拿到的環帶比要求的窄，且報表上看不出來。
    """
    if not 0 < inner < outer:
        raise ValueError(
            f"需要 0 < inner < outer，收到 inner={inner}、outer={outer}。"
            "inner 為 0 時支撐會與主體的羽化帶重疊，實際面積比要求的窄而"
            "報表上看不出來。")
    hard = (face_mask > 0.5).to(face_mask.dtype)
    from src.defense.subject_mask import _dilate

    return (_dilate(hard, outer) - _dilate(hard, inner)).clamp(0.0, 1.0)


def lattice_support(carrier: torch.Tensor, pitch: int, radius: float
                    ) -> torch.Tensor:
    """規則格點上的小圓斑：間距 `pitch` 像素、半徑 `radius` 像素。

    SD 的 VAE 降採樣 8 倍，故 `pitch = 8` 時每一個 latent 格恰好落進一個圓斑，
    **token 覆蓋率 100% 而像素面積只有 `π·radius²/pitch²`**（radius 2 時 19.6%）。
    這是「攤平整張畫面」以外，另一條達到滿覆蓋率的路，而且產物是規則點陣
    ——一眼看得出是刻意放上去的標記，不是壞掉的檔案。

    與 `scatter_support` 的分界：那一支是**少數幾個大斑**（位置由最遠點取樣
    決定、半徑由目標面積反推），這一支是**很多個小點**（位置由格點決定、
    面積由 pitch 與 radius 決定）。兩者的支撐構造不同，故是兩個函式而不是
    一個帶旗標的。

    `carrier` 之外的地方一律為 0，故臉的排除與「主體那一側恆為 0」的保證
    與其他支撐完全相同。面積不由呼叫端指定——它是 pitch 與 radius 的函數，
    **回傳之後由呼叫端讀 `mean()` 記進 CSV**，不在這裡反推，那會讓
    「要求的」與「拿到的」兩個數混在一起。
    """
    if pitch < 2:
        raise ValueError(f"pitch 必須 >= 2，收到 {pitch}")
    if not 0.0 < radius:
        raise ValueError(f"radius 必須為正，收到 {radius}")
    if 2.0 * radius >= pitch:
        raise ValueError(
            f"radius {radius} 對 pitch {pitch} 太大：圓斑會互相接觸，"
            "支撐退化成連通的一片，那就不是點陣了。要滿版請用 --carrier-match scale。")
    h, w = carrier.shape[-2:]
    dev, dt = carrier.device, carrier.dtype
    # 格點取在每一格的中心（`pitch // 2` 的偏移），故 pitch = 8 時中心落在
    # 4, 12, 20…，每個 8×8 的 latent 格恰好被涵蓋一次。
    yy = torch.arange(h, device=dev, dtype=torch.float32)
    xx = torch.arange(w, device=dev, dtype=torch.float32)
    off = pitch // 2
    dy = ((yy - off) % pitch) - (pitch - 1) / 2.0
    dx = ((xx - off) % pitch) - (pitch - 1) / 2.0
    # 上面的取模讓每個像素算出它到**最近**格點的位移，故不必逐格畫圓。
    d2 = dy[:, None] ** 2 + dx[None, :] ** 2
    dots = (d2 <= radius * radius).to(dt)[None, None]
    return dots * carrier.to(dt)


def scatter_support(carrier: torch.Tensor, count: int, area: float,
                    seed: int) -> torch.Tensor:
    """在載體內取 `count` 個**分散**的圓斑，總面積約為 `area`（佔全圖比例）。

    為什麼要它：鋪滿整件衣服的花紋面積大、也最顯眼。把同樣的預算拆成幾塊
    分散的斑點，讀起來更接近「衣服上本來就有的圖案」。
    """
    if count < 1:
        raise ValueError(f"count 必須 >= 1，收到 {count}")
    h, w = carrier.shape[-2:]
    inside = (carrier > 0.5)[0, 0]
    avail = float(inside.float().mean())
    if avail < area:
        raise ValueError(
            f"載體面積 {avail:.4f} 放不下要求的 {area:.4f}。"
            "**不自動縮小**：縮小之後面積欄與實際跑的對不上。")
    # **裝置要跟著輸入走。** 本機測試只跑 CPU，故「在 CPU 上建張量」這類錯
    # 誤要到 GPU 上才會炸；`torch.Generator()` 則刻意留在 CPU——跨裝置的
    # 隨機數序列不同，同一個種子會抽出不同的中心點。
    dev = carrier.device
    ys, xs = torch.nonzero(inside, as_tuple=True)
    g = torch.Generator().manual_seed(int(seed))
    first = int(torch.randint(len(ys), (1,), generator=g))
    picked = [(int(ys[first]), int(xs[first]))]
    d2 = (ys - picked[0][0]).float() ** 2 + (xs - picked[0][1]).float() ** 2
    for _ in range(count - 1):
        i = int(torch.argmax(d2))
        picked.append((int(ys[i]), int(xs[i])))
        nd = (ys - picked[-1][0]).float() ** 2 + (xs - picked[-1][1]).float() ** 2
        d2 = torch.minimum(d2, nd)

    yy, xx = torch.meshgrid(torch.arange(h, device=dev).float(),
                            torch.arange(w, device=dev).float(), indexing="ij")
    keep = (carrier > 0.5).to(carrier.dtype)

    def build(r):
        out = torch.zeros(1, 1, h, w, dtype=carrier.dtype, device=dev)
        for cy, cx in picked:
            out[0, 0] = torch.maximum(
                out[0, 0],
                (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).to(out.dtype))
        return out * keep

    lo, hi = 0.5, float(max(h, w))
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if float(build(mid).mean()) < area:
            lo = mid
        else:
            hi = mid
    out = build(hi)
    got = float(out.mean())
    if got < area * 0.9:
        raise ValueError(
            f"分散 {count} 塊在這個載體上最多只鋪得出 {got:.4f}，"
            f"達不到要求的 {area:.4f}。**不靜默接受**——面積欄與實際跑的"
            "對不上，而報表上看不出來。塊數調少或面積調小。")
    return out


def legal_area(carrier: torch.Tensor, mask: torch.Tensor) -> float:
    """載體與主體補集相交之後的面積。**這就是 `patch_area` 記的那個量。**

    `PatchParam.reset` 的 `complement` 分支寫的是
    `support = (mask <= 0).float() * carrier`，本函式取它的平均。兩處各寫一份
    的話，「對齊到的面積」與「報表上的面積」會是兩個東西而且看不出來。
    """
    legal = (mask.to(carrier.device) <= 0.0).to(carrier.dtype) * carrier
    return float(legal.mean())


MATCH_MODES = ("erode", "scale")


def match_area(carrier: torch.Tensor, mask: torch.Tensor, target: float,
               mode: str = "erode") -> torch.Tensor:
    """把載體縮到**合法面積**恰好等於 `target`。兩種縮法，形狀完全不同。

    `mode="erode"`（預設）把邊界往內縮，保留原本的輪廓；
    `mode="scale"` 把整片權重乘上一個常數，**形狀不變、強度變淡**——
    交出去的圖是 `(1−α)·x + α·φ`，也就是花紋均勻鋪滿整個合法區域。

    兩者在 `mean(w)` 這個讀數上完全等價（都恰好等於 `target`），但視覺與機制
    不同：侵蝕是「一小塊全改」，縮放是「一大片各改一點」。**要問「同樣的預算
    攤開還是集中比較好」就必須把這兩個分開**，合成一個旋鈕會讓那個問題問不出來。
    """
    if mode not in MATCH_MODES:
        raise ValueError(f"mode 必須是 {MATCH_MODES} 之一，收到 {mode!r}")
    if mode == "scale":
        if not 0.0 < target <= 1.0:
            raise ValueError(f"target 必須落在 (0,1]，收到 {target}")
        a0 = legal_area(carrier, mask)
        if a0 < target:
            raise ValueError(
                f"載體的合法面積只有 {a0:.5f}，達不到要求的 {target:.5f}。"
                "縮放模式只能把權重調淡，調不出比載體本身更大的面積。")
        # 面積對常數是線性的，故等式精確成立。乘上小於 1 的常數只會讓權重
        # 下降，「主體那一側恆為 0」的保證不受影響。
        return carrier * (target / a0)
    return _match_area_erode(carrier, mask, target)


def _match_area_erode(carrier: torch.Tensor, mask: torch.Tensor, target: float
                      ) -> torch.Tensor:
    """把載體往內縮，使**合法面積**恰好等於 `target`。

    構造是**逐級侵蝕加一層線性內插**：一級一級往內縮，停在第一個
    `area(r+1) < target <= area(r)` 的地方，再回傳 `(1−λ)·E_r + λ·E_{r+1}`，
    其中 `λ = (area(r) − target) / (area(r) − area(r+1))`。面積對 λ 是線性的，
    故等式**精確成立**（浮點誤差內）。

    只用整數半徑會落在離散的階梯上——背景區域的周長很長，一級侵蝕就可能讓
    面積掉好幾個百分點，於是「對齊」實際上會差很多而報表上看不出來。

    兩層都只會讓權重下降，故「主體那一側恆為 0」的保證不受影響。
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target 必須落在 (0,1]，收到 {target}")
    prev = carrier
    a_prev = legal_area(prev, mask)
    if a_prev < target:
        raise ValueError(
            f"載體的合法面積只有 {a_prev:.5f}，達不到要求的 {target:.5f}。"
            "**不自動縮小目標**：縮小之後兩組的面積不再相同，而「同面積」"
            "正是這個對照要控制的變因。")
    for _ in range(max(carrier.shape[-2:])):
        cur = erode_mask(prev, 1)
        a_cur = legal_area(cur, mask)
        if a_cur < target:
            if a_prev <= a_cur:      # 侵蝕不再讓面積下降，無法再逼近
                return prev
            lam = (a_prev - target) / (a_prev - a_cur)
            return (1.0 - lam) * prev + lam * cur
        prev, a_prev = cur, a_cur
    return prev


# ATR 裡屬於「臉與頭部」的類別。新的威脅模型把它當**受保護主體**：
# 發布出去的照片裡臉逐位元不動，而攻擊者編輯後的輸出裡認不出那是誰。
FACE_SUBJECT_CLASSES = (11, 2)          # Face, Hair


def face_subject_mask(x01: torch.Tensor, dilate: int = 16, feather: int = 24,
                      device=None) -> torch.Tensor:
    """(1,3,H,W) → (1,1,H,W)，1 = 受保護的臉與頭部。"""
    from src.defense.subject_mask import _dilate, _feather

    seg = parse_atr(x01, device=device)
    hard = torch.zeros_like(seg, dtype=x01.dtype)
    for i in FACE_SUBJECT_CLASSES:
        hard = torch.maximum(hard, (seg == i).to(x01.dtype))
    return _feather(_dilate(hard[None, None], dilate), feather)


def exclude_boxes(m: torch.Tensor, boxes, margin: int = 8) -> torch.Tensor:
    """把矩形框（含 `margin` 像素的外擴）從遮罩裡挖掉。

    用途是把**偵測到的人臉框**從衣物載體裡排除。ATR 沒有「這一塊是皮膚不是
    衣服」的把關，特寫人像上它會把臉頰與額頭整片標成 Upper-clothes——那時
    載體就是人臉，花紋會畫在臉上。MTCNN 的框是獨立於 ATR 的訊號。
    """
    if not boxes:
        return m
    out = m.clone()
    h, w = m.shape[-2:]
    for x0, y0, x1, y1 in boxes:
        a = max(0, int(y0) - margin); b = min(h, int(y1) + margin + 1)
        c = max(0, int(x0) - margin); d = min(w, int(x1) + margin + 1)
        out[..., a:b, c:d] = 0.0
    return out


def box_conflict(m: torch.Tensor, boxes) -> float:
    """臉框裡有多少比例被判成載體。**這是失效的直接量度。**

    高值代表 ATR 把臉判成衣服。回傳所有框的最大值；沒有框時回 0.0
    （沒有框就沒有這個衝突，是否可用由別的關卡決定）。
    """
    if not boxes:
        return 0.0
    h, w = m.shape[-2:]
    worst = 0.0
    for x0, y0, x1, y1 in boxes:
        a = max(0, int(y0)); b = min(h, int(y1) + 1)
        c = max(0, int(x0)); d = min(w, int(x1) + 1)
        if b <= a or d <= c:
            continue
        worst = max(worst, float(m[..., a:b, c:d].mean()))
    return worst


def carrier_stats(c: torch.Tensor) -> dict:
    """供 CSV 逐列記下的兩個欄位。

    `carrier_area` 是**交集之前**的載體面積；交集之後的實際支撐面積由
    `PatchParam.geometry()` 的 `patch_area` 給，兩者不同而且都要記——差值就是
    「有多少衣服落在受保護主體裡因而不可用」。
    """
    return {"carrier_area": round(float(c.mean()), 5),
            "carrier_source": CARRIER_REPO}
