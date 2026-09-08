"""可見的對抗補丁：主體之外一塊高幅度、**內容可學**的區域。

這一族在問什麼
────────────────────────────────────────────────────────────────────
`runs/patch_probe/README.md` 已經量過**未經最佳化**的補丁：把主體以外的
每一個像素換成白噪聲（面積 37–39%），主體內位移只有 0.111–0.149；角落一塊
最大的合法方塊（面積 5.5–10.5%）只有 0.010–0.036。相位族在更低的失真下是
0.41–0.67。

那一批留下的唯一缺口，是它**沒有最佳化**：

- 內容是固定的（噪聲／中灰／另一張照片），不是學出來的。
- 損失從來沒有寫成「只對**主體**那些 latent token 負責」。

`runs/ig_probe/by_region_*.csv` 量到殘差的 64% 落在主體上，而補丁碰不到主體，
只能靠 UNet 的長程作用間接影響它——白噪聲把背景項打到 0.087（已飽和），
主體項卻只到 0.626。本參數化要問的是：**梯度能不能在那條長程通道上，
找到白噪聲找不到的東西。**

構造
────────────────────────────────────────────────────────────────────
    x_def = where(support, clamp(c, 0, 1), x)

- `support` 是一塊矩形，由主體遮罩的**補集**中選出，且與遮罩**零重疊**——
  羽化帶也算，所以主體區域逐位元不動。
- `c` 初始化為原圖，故**訓練起點即恆等**（與相位族 θ=0、色彩族零半徑同性質）。
- `radius` 是**面積比例**，不是幅度。補丁內的幅度不設上界（只夾在 [0,1]），
  這正是「可見的防禦」與 L∞ 球那一族的分界：預算花在**面積**上而不是幅度上。

**支撐之外的參數不會動**：`render` 用 `where` 挑，支撐外的梯度恆為零。
存成整張影像大小是為了讓 `project` 與存檔的形狀與影像對齊，不是因為那些值
有意義。

一個必須說清楚的限制
────────────────────────────────────────────────────────────────────
主體遮罩太大或太分散時，某個面積可能**沒有**與遮罩零重疊的位置。實測兩張
目標影像的最大合法邊長只有 120 與 166（佔全圖 5.5% 與 10.5%）。
那時 `reset` 直接拋錯而不是縮小補丁——安靜地縮小會讓 `radius` 欄與實際跑的
面積對不上，而報表上看不出來。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import math

import torch
import torch.nn.functional as F


def box_blur(t: torch.Tensor, k: int) -> torch.Tensor:
    """邊長 `k` 的盒式平均，反射填充，輸出與輸入同形狀。`k <= 1` 時是恆等。

    **只有這一份實作。** `--patch-tint` 的低頻懲罰與 `--patch-lowfreq` 的低頻
    替換必須用同一個模糊，否則同一個 σ 在兩個旗標下指的是不同的尺度，而
    報表上兩者並排看起來像是可比的。

    `avg_pool2d(k, stride=1)` 會把邊長減少 `k−1`，故總填充量必須是 `k−1`；
    `k` 為偶數時左右不等，寫成 `pad = k // 2` 會讓輸出比輸入大一格。
    """
    if k <= 1:
        return t
    lo, hi = (k - 1) // 2, (k - 1) - (k - 1) // 2
    return F.avg_pool2d(F.pad(t, (lo, hi, lo, hi), mode="reflect"),
                        kernel_size=k, stride=1)


def placements(mask: torch.Tensor, side: int, stride: int = 16):
    """所有**與遮罩零重疊**的左上角座標，以及遮罩重心。

    合法的定義取 `mask[rect].max() == 0`：羽化帶的值大於零，故補丁連羽化帶
    都不碰，主體區域逐位元不動。

    積分圖使每個候選位置是 O(1)；候選數是 O(HW/stride²)，逐格取 max 會慢。
    """
    m = mask[0, 0]
    h, w = m.shape
    if side > h or side > w:
        return [], (0.0, 0.0)
    cs = torch.cumsum(torch.cumsum(m.float(), dim=0), dim=1)
    pad = torch.zeros(h + 1, w + 1, dtype=cs.dtype, device=cs.device)
    pad[1:, 1:] = cs

    tot = float(m.sum())
    if tot > 0:
        ys = torch.arange(h, device=m.device, dtype=m.dtype)
        xs = torch.arange(w, device=m.device, dtype=m.dtype)
        cy = float((m.sum(dim=1) * ys).sum() / tot)
        cx = float((m.sum(dim=0) * xs).sum() / tot)
    else:
        cy, cx = h / 2.0, w / 2.0

    ok = []
    for top in range(0, h - side + 1, stride):
        for left in range(0, w - side + 1, stride):
            s = (pad[top + side, left + side] - pad[top, left + side]
                 - pad[top + side, left] + pad[top, left])
            if float(s) == 0.0:
                ok.append((top, left))
    return ok, (cy, cx)


def choose(cands: Sequence[Tuple[int, int]], centre, mode: str):
    """`far` 取離主體重心最遠、`near` 取最近。同距時取座標最小者以求確定性。"""
    if not cands:
        return None
    cy, cx = centre

    def key(p):
        top, left = p
        d = ((top - cy) ** 2 + (left - cx) ** 2) ** 0.5
        return (-d if mode == "far" else d, top, left)

    return min(cands, key=key)


def largest_legal_side(mask: torch.Tensor, stride: int = 16,
                       min_side: int = 32) -> int:
    """最大的偶數邊長，使該邊長至少有一個與遮罩零重疊的位置。沒有則回 0。

    二分搜尋是合法的：邊長 `s` 有合法位置時，同一個左上角的 `s-2` 方塊是它的
    子集、也必為零，且落在同一個步長網格上，故「有合法位置」對邊長單調。
    """
    h, w = mask.shape[-2:]
    lo, hi = min_side, min(h, w)
    best = 0
    while lo <= hi:
        mid = int(round((lo + hi) / 2))
        mid -= mid % 2
        if mid < min_side:
            break
        if placements(mask, mid, stride)[0]:
            best = mid
            lo = mid + 2
        else:
            hi = mid - 2
    return best


def crop_safe_box(h: int, w: int, keep: float):
    """裁切後仍然留存的中央方框，回傳 `(top, left, bottom, right)`。

    `keep` 是每一邊留下的比例：`crop_resize0.1` 每邊裁 10%，留存的是中央
    80%，故 `keep = 0.8`。**這個值必須與要防的裁切算子對齊**，寫死一個
    看起來合理的數字會讓「不易裁切」變成一句沒有根據的話，故它是 CSV 欄位。
    """
    if not 0.0 < keep <= 1.0:
        raise ValueError(f"crop_keep 必須落在 (0,1]，收到 {keep}")
    mh = int(round(h * (1.0 - keep) / 2.0))
    mw = int(round(w * (1.0 - keep) / 2.0))
    return (mh, mw, h - mh, w - mw)


def place_many(mask: torch.Tensor, side: int, count: int, stride: int,
               mode: str, crop_keep: float = 1.0):
    """挑 `count` 個互不重疊、且與遮罩零重疊的方塊，回傳左上角座標串列。

    `crop_safe` 模式再加一層限制：整塊必須落在 `crop_safe_box` 之內，並取
    **離畫面中心最近**者——裁切是繞中心的，離中心越近存活率越高。

    貪心挑選（挑一塊、把與它重疊的候選剔掉、再挑下一塊）。方塊同尺寸且
    候選在同一個步長網格上，故「互不重疊」只要比較左上角距離即可。
    **放不下第二塊時拋錯**：安靜地少放一塊會讓 `patch_count` 欄與實際面積
    對不上，而報表上看不出來。
    """
    h, w = mask.shape[-2:]
    cands, centre = placements(mask, side, stride)
    if mode == "crop_safe":
        t0, l0, t1, l1 = crop_safe_box(h, w, crop_keep)
        cands = [(t, l) for t, l in cands
                 if t >= t0 and l >= l0 and t + side <= t1 and l + side <= l1]
        centre = (h / 2.0, w / 2.0)
        mode = "near"
    chosen = []
    for _ in range(count):
        pos = choose(cands, centre, mode)
        if pos is None:
            raise ValueError(
                f"擺不下第 {len(chosen) + 1} 塊（要求 count={count}，邊長 "
                f"{side}）：沒有與主體遮罩零重疊、與已選方塊不重疊"
                + (f"、且整塊落在中央 {crop_keep:.0%} 內" if crop_keep < 1.0
                   else "")
                + "的位置。**不自動減少塊數**——少放一塊會讓 patch_count 欄"
                  "與實際面積對不上。")
        chosen.append(pos)
        top, left = pos
        cands = [(t, l) for t, l in cands
                 if abs(t - top) >= side or abs(l - left) >= side]
    return chosen


def rect_support(shape, top: int, left: int, side: int) -> torch.Tensor:
    """方形支撐，(1,1,H,W) 的 bool。"""
    s = torch.zeros(1, 1, shape[-2], shape[-1], dtype=torch.bool)
    s[..., top:top + side, left:left + side] = True
    return s


class PatchParam:
    """φ = 補丁內容，`x_def = where(support, clamp(c,0,1), x)`。

    `radius` 是**面積比例**（0.05 = 全圖的 5%）。`mask` 是主體遮罩（1 = 主體），
    補丁只會落在與它零重疊的地方。`placement` 決定同尺寸的多個合法位置裡挑
    哪一個：`far` 離主體最遠、`near` 最近。
    """

    name = "patch"

    def __init__(self, radius: float = 0.05, mask: Optional[torch.Tensor] = None,
                 placement: str = "far", stride: int = 16,
                 min_side: int = 32, init: str = "identity",
                 tile: int = 0, res: int = 1, alpha: float = 1.0,
                 count: int = 1, crop_keep: float = 0.8,
                 lowfreq: int = 0, chroma: bool = False):
        if placement not in ("far", "near", "complement", "crop_safe"):
            raise ValueError(
                "placement 必須是 far／near／complement／crop_safe，"
                f"收到 {placement!r}")
        if count < 1:
            raise ValueError(f"count 必須 >= 1，收到 {count}")
        if init not in ("identity", "random"):
            raise ValueError(f"init 必須是 identity 或 random，收到 {init!r}")
        if tile < 0:
            raise ValueError(f"tile 必須 >= 0（0 = 不平舖），收到 {tile}")
        if res < 1:
            raise ValueError(f"res 必須 >= 1（1 = 全解析度），收到 {res}")
        if tile and res > tile:
            raise ValueError(
                f"res={res} 大於 tile={tile}：一塊磚降取樣之後不足一個像素")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha 必須落在 (0,1]，收到 {alpha}")
        if lowfreq < 0:
            raise ValueError(f"lowfreq 必須 >= 0（0 = 關閉），收到 {lowfreq}")
        self.radius = radius
        self.init = init
        # 三個「美化」旋鈕。都不改變支撐，只改變支撐內長什麼樣子。
        self.tile = tile
        # `res` 是**帶限**旋鈕，與 `tile` 是兩件事。平舖把內容變成週期的
        # （梳狀譜，諧波一路到 Nyquist，模糊照樣抹得掉磚內的細節），
        # `res` 把內容限制在 `(H/res)·(W/res)` 個雙線性基底張成的子空間裡，
        # 自由度因此降 `res²` 倍。
        #
        # **不要寫成「依構造沒有細於 res 像素的分量」——那不成立。** 雙線性
        # 升取樣的核是三角波，頻譜是 sinc²，主瓣之外有旁瓣。實測（隨機內容、
        # 扣掉直流、128² 上量）：
        #
        #     res    E(f > 1/res)   E(f > 1/8)   E(f > 1/2)
        #       1        0.218         0.989        0.806
        #       4        0.162         0.659        0.007
        #      16        0.184         0.023        0.004
        #
        # 也就是：**恰好高於截止的那一段一直有 12–22% 的旁瓣洩漏**，而遠高於
        # 截止的那一段掉得非常快（`f > 1/2` 由 0.806 掉到 0.007，115 倍）。
        # 對「撐過重取樣與模糊」而言重要的是後者，故這個旋鈕仍然對得上目標，
        # 但宣稱時要用量到的數字，不要用「依構造」。
        #
        # 機制取自 IAM（arXiv:2402.16586）的 interpolation smoothing，但那篇
        # 動的是**更新步驟**（在半解析度上走一步再升回去），此處動的是
        # **參數化本身**（可學張量就只有那麼大），故約束在每一步都成立而不是
        # 每一步之後被抹一次。
        self.res = res
        self.alpha = alpha
        # 另外兩個美化旋鈕，性質與上面兩個不同：它們是**約束**不是懲罰，
        # 於是不必調權重、也不與主損失搶預算。見 `_constrain`。
        self.lowfreq = lowfreq
        self.chroma = chroma
        self.mask = mask
        self.placement = placement
        self.stride = stride
        self.min_side = min_side
        self.c: Optional[torch.Tensor] = None
        self.support: Optional[torch.Tensor] = None
        # 載體（`src/defense/carrier_mask.py`）。`None` 時呼叫路徑逐位元不變。
        self.carrier: Optional[torch.Tensor] = None
        self.carrier_kind = "none"
        # 逐列要寫進 CSV 的三個實際幾何。`radius` 是要求的面積，這三個是
        # **拿到的**面積與位置，兩者不一定相同（邊長取整到偶數）。
        self.side = 0
        self.top = 0
        self.left = 0
        # 固定形狀浮水印：`count` 塊同尺寸方塊，`crop_keep` 是裁切後仍留存的
        # 中央比例（只有 `crop_safe` 用得到，但兩者都逐列寫進 CSV）。
        self.count = count
        self.crop_keep = crop_keep
        self.rects: List[Tuple[int, int]] = []

    def set_carrier(self, carrier: Optional[torch.Tensor], kind: str) -> None:
        """掛上載體遮罩（1 = 可以放花紋的地方）。

        張量與名稱**必須同時給或同時不給**：只給其中一個的症狀是 CSV 的
        `patch_carrier` 欄與實際跑的支撐對不上，而報表上看不出來。
        """
        if (carrier is None) != (kind == "none"):
            raise ValueError(
                f"carrier 與 carrier_kind 必須一致：收到 carrier="
                f"{'None' if carrier is None else '張量'}、carrier_kind={kind!r}。"
                "只給其中一個會讓報表上的載體名稱與實際支撐對不上。")
        self.carrier = carrier
        self.carrier_kind = kind

    def side_for(self, h: int, w: int) -> int:
        """面積比例 → 單塊邊長，取整到偶數。

        `radius` 是 `count` 塊**加起來**的面積，故單塊面積是 `radius / count`。
        這樣「一塊 4%」與「兩塊各 2%」在同一個總預算下可以直接對照，問的是
        「把同樣的面積拆開有沒有比較好」。`count = 1` 時與原式逐位元相同。
        """
        s = int(round((self.radius * h * w / self.count) ** 0.5))
        return max(0, s - (s % 2))

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        if self.mask is None:
            raise ValueError(
                "PatchParam 需要主體遮罩才知道補丁可以放哪裡。"
                "**不預設放中央**——那會蓋到主體。")
        h, w = x01.shape[-2:]
        if self.carrier is not None and self.placement != "complement":
            raise ValueError(
                f"載體 {self.carrier_kind!r} 只能配 placement=complement，"
                f"收到 {self.placement!r}。矩形擺放模式會把載體靜默忽略，"
                "症狀是報表上寫著載體名稱而實際跑的是一塊方塊。")
        if self.count > 1 and self.placement == "complement":
            raise ValueError(
                f"count={self.count} 不能配 placement=complement："
                "補集是單一個不規則區域，「幾塊」在那個構造下沒有意義。"
                "多塊要配 crop_safe／far／near。")
        if self.count > 1 and self.carrier is not None:
            raise ValueError(
                f"count={self.count} 不能配載體 {self.carrier_kind!r}："
                "載體是不規則區域、方塊擺放是固定形狀，兩種支撐構造互斥。")
        if self.placement == "complement":
            # 支撐 = **整個遮罩補集**（逐像素等於零才算，羽化帶不算）。
            # 這是「主體逐位元不動」這個約束底下可用面積的最大值，radius
            # 在這個模式下不使用——面積由遮罩決定，不由我們挑。
            support = (self.mask.to(x01.device) <= 0.0)
            if self.carrier is not None:
                # 再交集一層語意載體。交集的順序不影響結果，但**主體那一側
                # 必須是硬條件**：支撐與 `mask > 0` 不相交是由這一行的構造
                # 保證的，不是由事後檢查保證的。
                #
                # 載體可以是**軟權重**（往內羽化之後是 [0,1]），此時支撐也是
                # 軟的：`w = (mask <= 0) · carrier`。`w = 0` 的地方
                # `(1−w)·x + w·c` 逐位元等於 `x`，故主體不動的保證不變。
                cw = self.carrier.to(device=x01.device, dtype=x01.dtype)
                support = support.to(x01.dtype) * cw
                if not bool((support > 0).any()):
                    raise ValueError(
                        f"載體 {self.carrier_kind!r} 與主體補集的交集是空的："
                        "這張影像上沒有落在受保護主體之外的衣物。"
                        "**不自動退回補集**——安靜地換掉支撐會讓 patch_carrier "
                        "欄與實際跑的東西對不上，而報表上看不出來。")
            self.support = support
            self.side, self.top, self.left = 0, 0, 0
            self.c = self._init_content(x01, seed)
            return
        side = self.side_for(h, w)
        if side < self.min_side:
            raise ValueError(
                f"radius={self.radius} 在 {h}×{w} 上換算成邊長 {side}，"
                f"小於 min_side={self.min_side}。補丁太小，跑了也讀不出東西。")
        m = self.mask.to(x01.device)
        if not placements(m, side, self.stride)[0]:
            best = largest_legal_side(m, self.stride, self.min_side)
            raise ValueError(
                f"面積 {self.radius}（邊長 {side}）在這張影像上沒有與主體遮罩"
                f"零重疊的位置。最大的合法邊長是 {best}"
                f"（面積 {best * best / (h * w):.4f}）。"
                "**不自動縮小**：縮小之後 radius 欄與實際面積對不上，"
                "而報表上看不出來。")
        self.rects = place_many(m, side, self.count, self.stride,
                                self.placement, self.crop_keep)
        self.top, self.left = self.rects[0]
        self.side = side
        sup = torch.zeros(1, 1, h, w, dtype=torch.bool)
        for top, left in self.rects:
            sup |= rect_support(x01.shape, top, left, side)
        self.support = sup.to(x01.device)
        self.c = self._init_content(x01, seed)

    def _content_shape(self, x01: torch.Tensor):
        """可學張量的形狀。平舖時只學一塊磚；`res > 1` 時只學降取樣後的網格。

        兩者相乘：`tile=64, res=4` 學的是 16×16，參數量由 3·512² 掉到 3·16²
        （**三個數量級**）。`res` 用 `ceil` 取整，故升取樣時的目標尺寸恆能
        涵蓋磚長／畫面，不必補邊。
        """
        h, w = (self.tile, self.tile) if self.tile else x01.shape[-2:]
        if self.res > 1:
            h = (h + self.res - 1) // self.res
            w = (w + self.res - 1) // self.res
        if self.tile:
            return (1, 3, h, w)
        return (x01.shape[0], x01.shape[1], h, w)

    def _field(self, x01: torch.Tensor) -> torch.Tensor:
        """把可學張量攤成整張畫面大小的內容場。

        兩步，順序寫死：**先升取樣、再平舖**。

        1. `res > 1` 時把可學張量雙線性升回全解析度。輸出落在 `(H/res)²` 個
           雙線性基底張成的子空間裡，遠高於 `1/res` 的能量被壓掉兩個數量級
           （旁瓣的實測見 `__init__`）。
        2. 平舖時用 `repeat` 把磚鋪滿再裁到畫面大小——**規則的重複本身就是
           「這是刻意放上去的標記」的視覺訊號**，而且參數量由 3·H·W 掉到
           3·tile²（512² → 64² 是 64 倍）。

        順序不可對調：先平舖再升取樣的話，升取樣會跨過磚與磚的接縫做內插，
        週期性被抹掉一條，而且那條接縫的寬度隨 `res` 變——兩個旋鈕就不再獨立。
        """
        c = self.c
        if self.res > 1:
            h, w = (self.tile, self.tile) if self.tile else x01.shape[-2:]
            c = F.interpolate(c, size=(int(h), int(w)), mode="bilinear",
                              align_corners=False)
        if not self.tile:
            return c
        h, w = x01.shape[-2:]
        reps = (h + self.tile - 1) // self.tile, (w + self.tile - 1) // self.tile
        return c.repeat(1, 1, reps[0], reps[1])[..., :h, :w]

    def _init_content(self, x01: torch.Tensor, seed: int) -> torch.Tensor:
        """補丁內容的起點。

        `identity`（預設）從原圖開始，故第 0 步的輸出與原圖逐位元相同——
        與相位族 θ=0、色彩族零半徑同性質，收斂曲線的第一點是恆等。

        `random` 在支撐內抽均勻噪聲。**某些損失非用它不可**：
        `edit_divergence` 的形式是 `−‖f(x_def) − f(x)‖²`，在 `x_def = x` 處
        值與**梯度都恰為零**，恆等起點是一個駐點，PGD 一步也走不動而
        `trace.csv` 看起來只是「損失很小」。這正是 FND-053 那一族的零梯度
        陷阱。`scripts/ip2p_run.py` 會在這個組合上直接拋錯，不讓它靜默空轉。
        """
        shape = self._content_shape(x01)
        if self.init == "identity":
            if self.tile:
                # 平舖時沒有「與原圖相同」這回事（一塊磚蓋不出原圖），
                # 故恆等起點取畫面均值——它是最接近「不改變觀感」的常數場。
                return x01.mean().expand(shape).clone().requires_grad_(True)
            if self.res > 1:
                # `res > 1` 時子空間裡沒有原圖，但**有一個離原圖最近的點**：
                # 原圖的降取樣。升回去就是原圖的低通版本，故第 0 步的輸出
                # **不是逐位元恆等**，而是支撐內被低通了一次。這與平舖的常數
                # 場不同（那個離原圖很遠），也與 `res = 1` 的逐位元恆等不同。
                # 實際差多少由 `content_stats` 量出，不用推論的。
                return F.interpolate(x01.detach(), size=shape[-2:],
                                     mode="bilinear", align_corners=False,
                                     antialias=True).clone().requires_grad_(True)
            return x01.detach().clone().requires_grad_(True)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.rand(shape, generator=g).to(
            device=x01.device, dtype=x01.dtype)
        if self.tile or self.res > 1:
            # 兩者的可學張量都小於畫面，與 `x01` 形狀不同，混不了色；
            # 支撐之外的內容在 `render` 的 `where` 那一步本來就用不到。
            return noise.detach().requires_grad_(True)
        # 軟支撐時 `where` 會拋「condition 不是布林」——那條路徑當初漏改了。
        # 這裡與 `render` 用同一個混色式子，`w = 0` 處逐位元等於原圖。
        if self.support.dtype == torch.bool:
            return torch.where(self.support, noise, x01
                               ).detach().requires_grad_(True)
        w = self.support.to(noise.dtype)
        return ((1.0 - w) * x01 + w * noise).detach().requires_grad_(True)

    def _constrain(self, field: torch.Tensor, x01: torch.Tensor) -> torch.Tensor:
        """把內容場投影到「低頻／亮度跟著衣服走」的集合上。**夾取之前呼叫。**

        兩個旋鈕都是**約束**，這是它們與 `--patch-tint`（低頻懲罰）的分界：

        - 懲罰要調 λ，而且與主損失搶同一份預算：低頻誤差只是被壓小。
        - 約束不必調權重，低頻誤差在夾取之前**恆為零**，最佳化只能在剩下的
          自由度裡走。本專案的半徑一向用投影而非懲罰實作，這一條同性質。

        `lowfreq`（頻帶替換）

            field ← field − B_σ(field) + B_σ(x)

          σ 就是 `--patch-tint` 的 σ，兩者共用 `box_blur`，故同一個數字在兩個
          旗標下指同一個尺度。夾取之前 `B_σ` 這一帶逐位元等於原圖那一帶，
          衣服的顏色與明暗完全保留，可學的只有比 σ 細的紋理。
          **注意 `B_σ` 不是冪等的**，故「低頻完全等於原圖」只在這一次替換的
          意義下成立；夾取又會再動它一點。實際殘留由 `content_stats` 逐格量出
          並寫進 CSV，不用推論的。

        `chroma`（凍結亮度）

            field ← field + ( Y(x) − Y(field) )        逐通道加同一個純量

          BT.601 的三個權重和為 1，故這個平移讓 `Y(field') = Y(x)` **精確成立**
          （夾取之前），而 `B − Y`、`R − Y` 兩個色度分量逐位元不變。於是褶皺、
          陰影、垂墜全部是原本那件衣服的，可學的只有顏色。
          用直接加減而不是 RGB↔YCbCr 來回轉，是因為來回轉會引入兩次矩陣乘的
          浮點誤差，而那正好落在要宣稱「精確」的那個量上。

        兩個同時開時**先低頻、後亮度**：亮度那一步是逐點的，擺在後面才能讓
        `Y(field) = Y(x)` 這個較強的等式成立。順序寫死在這裡而不是由旗標決定
        ——順序若是隱含的，兩種順序的結果在報表上分不出來。
        """
        if self.lowfreq:
            field = field - box_blur(field, self.lowfreq) + box_blur(x01, self.lowfreq)
        if self.chroma:
            from src.defense.color_param import luma

            field = field + (luma(x01) - luma(field))
        return field

    @torch.no_grad()
    def content_stats(self, x01: torch.Tensor) -> dict:
        """逐格量出兩個約束在**交出去的那張圖上**還差多少。

        兩個都在夾取之前成立，夾取之後不一定。**這兩欄是量出來的不是推論的**
        ——「約束成立」若只寫在 docstring 裡，夾取把它破壞掉時不會有任何症狀。

        分母取「不加約束時的同一個量」，故 1.0 代表約束完全沒有作用、
        0.0 代表完全成立。支撐之外逐位元等於原圖，兩個量都只在支撐內取。
        """
        if self.c is None or self.support is None:
            return {"patch_lowfreq_err": "", "patch_luma_err": ""}
        w = self.support.to(x01.dtype)
        denom = w.sum().clamp_min(1.0)
        free = self._field(x01).clamp(0.0, 1.0)
        got = self.render(x01)

        def _err(a, b, sigma):
            d = (box_blur(a, sigma) - box_blur(b, sigma)) * w
            return float(d.pow(2).sum() / (denom * a.shape[1]))

        out = {}
        if self.lowfreq:
            base = _err(free, x01, self.lowfreq)
            out["patch_lowfreq_err"] = round(
                (_err(got, x01, self.lowfreq) / base) if base > 0 else 0.0, 6)
        else:
            out["patch_lowfreq_err"] = ""
        if self.chroma:
            from src.defense.color_param import luma

            base = float((((luma(free) - luma(x01)) * w) ** 2).sum() / denom)
            cur = float((((luma(got) - luma(x01)) * w) ** 2).sum() / denom)
            out["patch_luma_err"] = round((cur / base) if base > 0 else 0.0, 6)
        else:
            out["patch_luma_err"] = ""
        return out

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        field = self._constrain(self._field(x01), x01).clamp(0.0, 1.0)
        # `alpha < 1` 時原圖從標記底下透出來，讀成「疊上去的浮水印」而不是
        # 「這塊區域壞掉了」。支撐之外一律逐位元保留原圖。
        blended = ((1.0 - self.alpha) * x01 + self.alpha * field).clamp(0.0, 1.0)
        if self.support.dtype == torch.bool:
            return torch.where(self.support, blended, x01)
        # 軟支撐：`w = 0` 時 `1·x + 0·blended` 逐位元等於 `x`，故主體不動的
        # 保證與布林支撐相同。`w ∈ {0,1}` 時兩條路徑的輸出也逐位元相同。
        w = self.support.to(blended.dtype)
        return (1.0 - w) * x01 + w * blended

    def params(self) -> List[torch.Tensor]:
        return [self.c]

    @torch.no_grad()
    def project(self) -> None:
        self.c.clamp_(0.0, 1.0)

    def set_radius(self, r: float) -> None:
        self.radius = r

    def geometry(self) -> dict:
        """逐列寫進 CSV 的實際幾何。要求的面積與拿到的面積不一定相同。"""
        # 軟支撐時面積取權重的平均——那才是「實際被改動了多少」，
        # 而 `w > 0` 的像素數會把羽化帶整條算成滿的。
        area = (float(self.support.to(torch.float32).mean())
                if self.support is not None else 0.0)
        return {"patch_side": self.side, "patch_top": self.top,
                "patch_left": self.left, "patch_placement": self.placement,
                "patch_area": round(area, 5), "patch_tile": self.tile,
                # `res` 與 `tile` 是兩個獨立的軸，兩欄都要有：同一個參數量
                # （例如 tile=64,res=1 與 tile=0,res=8）在頻譜上是兩件不同的
                # 事，只記參數量分不出來。
                "patch_res": self.res,
                "patch_alpha": self.alpha,
                # 固定形狀浮水印的三個幾何。`patch_rects` 是每一塊的左上角，
                # `patch_top`／`patch_left` 只是第一塊——兩塊時它們不足以還原
                # 支撐，故整份座標另記一欄。
                "patch_count": self.count,
                "patch_crop_keep": self.crop_keep,
                # 兩個內容約束由**實際跑的物件**給，與 `patch_carrier` 同一條
                # 理由：旗標與物件分岔時，只有這一邊會說實話。
                "patch_lowfreq": self.lowfreq,
                "patch_chroma": int(self.chroma),
                "patch_rects": ";".join(f"{t},{l}" for t, l in self.rects),
                # 載體名稱由**實際跑的物件**給，不由 argparse 給：兩者分岔時
                # 這一欄才會說實話。
                "patch_carrier": self.carrier_kind}


class PatchPaletteParam(PatchParam):
    """補丁內容限制在 **K 個可學顏色** 上：`field = softmax(c/τ) · palette`。

    存在理由
    ────────────────────────────────────────────────────────────────
    `docs/DIRECTION.md` §5.4 與證據清單第 5 項：現行四個參數化（自由、
    低頻懲罰、凍結亮度、色彩映射）**沒有一個產出「像標記」的東西**，
    產物都是彩色雜訊。而 §3.4b 的階梯又說效果來自**逐像素的高頻自由度**，
    所以「變好看」的既有作法（低頻懲罰、凍結亮度）都是靠砍掉那個自由度換來的，
    效果同步掉。

    調色盤走另一條路：**保留逐像素的空間自由度，只砍掉顏色的連續性。**
    每個像素仍然可以獨立選它要哪一個顏色（那是逐像素、可以很高頻的），
    但只能從 K 個顏色裡選。產物因此是「幾個色塊構成的圖樣」——網版印刷、
    貼紙、布料印花長的樣子——而不是連續的彩色雜訊。

    這與 `lowfreq`／`chroma` 是**正交**的一刀：那兩個砍的是頻帶與通道，
    這個砍的是**色彩的基數**。三者可以各自獨立掃。

    實測的自然度讀數
    ────────────────────────────────────────────────────────────────
    `src/metrics/naturalness.py`，同一張影像、同一個支撐、隨機起點、
    未最佳化（量的是**參數化本身**長什麼樣，不是學出來的東西）。
    支撐內原圖有 468 個顏色，`colours_ratio` 以它為 1.0：

    | 參數化 | 色數 | 色數比 | 熵 drop | 譜斜率 drop |
    |---|---|---|---|---|
    | `free`（逐像素自由） | 3596 | **7.7** | −1.80 | 3.19 |
    | `chroma`（凍結亮度） | 3175 | **6.8** | −0.21 | 0.36 |
    | `res=8`（帶限） | 1754 | **3.7** | −0.14 | **−4.85** |
    | `palette K=8` | 180 | **0.4** | −1.91 | 3.21 |
    | `palette K=4` | 96 | **0.2** | −1.57 | 3.24 |
    | `palette K=6, tile=64` | 75 | **0.2** | −1.83 | 3.21 |

    **這是本專案第一個把色數壓到低於原圖的參數化。** 既有的每一個都是往上走
    （`free` 7.7 倍、`chroma` 6.8 倍），也就是它們都在支撐內**增加**顏色的
    多樣性——那正是「看起來像壞掉」的直接來源。

    兩件必須一起看的事：

    - **調色盤不改變空間頻率。** `palette` 的譜斜率 drop 是 3.21，與 `free`
      的 3.19 幾乎相同；把能量往低頻搬的是 `res`（−4.85）。這是刻意的分工，
      也是為什麼兩個旋鈕要分開：**色彩的基數與空間的頻帶是兩件事**，
      合成一個旋鈕就分不出哪一個買到了什麼。
    - **熵 drop 沒有跟著改善**（−1.91 對 `free` 的 −1.80）。區塊熵量的是
      局部的變異程度，而「少數幾個顏色排成高頻圖樣」的區塊熵並不低。
      **所以熵不是「像不像標記」的代理讀數**，色數才是。
      這一項與本專案已記過的四次「代理讀數與判讀相反」同型，
      結論仍然要拉圖確認。

    構造
    ────────────────────────────────────────────────────────────────
        logits  c        (1, K, h, w)，值域 [0,1]
        palette p        (1, K, 3)，值域 [0,1]
        field            = Σ_k softmax(c/τ)_k · p_k

    `h, w` 跟著 `tile` 與 `res` 走（與 `PatchParam` 同一條路徑），所以三個
    旋鈕可以疊。

    **`logits` 的值域取 [0,1] 而不是無界**，兩個理由：`project()` 因此與
    `PatchParam` 逐字相同（同一個盒子），而 `run_param_pgd` 的單一步長同時
    適用於 logits 與 palette——兩個張量若盒寬差幾個數量級，共用一個 `α` 的
    那一個就會走不動或一步撞到底。溫度 `τ` 負責把 [0,1] 的差距放大成夠硬的
    指派：`τ = 0.05` 時最大與最小 logit 的機率比是 `e^20`，實務上就是硬指派。

    起點
    ────────────────────────────────────────────────────────────────
    `identity` 起點取**原圖在這個子空間裡最近的點**：調色盤由支撐內像素的
    亮度分位數取 K 個顏色，logits 指向每個像素最近的那一個。於是第 0 步的
    輸出就是「原圖的 K 色海報化」——與 `res` 的低通起點同性質，
    **不是逐位元恆等**（那個子空間裡沒有原圖）。實際偏差由 `content_stats`
    量出。
    """

    name = "patch_palette"

    def __init__(self, *args, palette: int = 6, temp: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        if palette < 2:
            raise ValueError(f"palette 必須至少為 2，收到 {palette}")
        if temp <= 0.0:
            raise ValueError(f"temp 必須為正，收到 {temp}")
        self.palette = palette
        self.temp = temp
        self.p: Optional[torch.Tensor] = None

    def _content_shape(self, x01: torch.Tensor):
        """logits 的形狀。通道數是 `K` 不是 3——**這是它與父類唯一的形狀差別**。"""
        base = super()._content_shape(x01)
        return (base[0], self.palette, base[2], base[3])

    def _field(self, x01: torch.Tensor) -> torch.Tensor:
        # 升取樣與平舖沿用父類（順序也一樣：先升取樣、再平舖），故 `res` 與
        # `tile` 在這個參數化上的意義與 `PatchParam` 完全相同。
        logits = super()._field(x01)
        w = torch.softmax(logits / self.temp, dim=1)          # (N,K,H,W)
        return torch.einsum("nkhw,nkc->nchw", w, self.p)

    def _init_content(self, x01: torch.Tensor, seed: int) -> torch.Tensor:
        shape = self._content_shape(x01)
        # **調色盤先建起來**：logits 的恆等起點要指向「最近的那個顏色」，
        # 沒有調色盤就沒有「最近」可言。
        self.p = self._init_palette(x01, seed).requires_grad_(True)
        if self.init == "identity":
            small = F.interpolate(x01.detach(), size=shape[-2:], mode="bilinear",
                                  align_corners=False, antialias=True)
            # 每個像素指向最近的調色盤顏色：那一格 logit 為 1，其餘為 0。
            d = (small[:, None] - self.p.detach()[..., None, None]).pow(2).sum(2)
            return F.one_hot(d.argmin(1), self.palette).permute(
                0, 3, 1, 2).to(x01.dtype).clone().requires_grad_(True)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.rand(shape, generator=g).to(
            device=x01.device, dtype=x01.dtype).requires_grad_(True)

    @torch.no_grad()
    def _init_palette(self, x01: torch.Tensor, seed: int) -> torch.Tensor:
        """K 個起始顏色，取自**支撐內**像素的亮度分位數。

        用分位數而不是隨機抽或 k-means：分位數是決定性的（同一張圖同一組
        顏色，不必記種子），而且依構造涵蓋整個明暗範圍——隨機抽在大片同色的
        衣服上很可能抽到 K 個幾乎一樣的顏色，那時 `softmax` 選誰都一樣，
        梯度也就沒有方向。
        """
        from src.defense.color_param import luma

        w = (self.support.to(torch.bool) if self.support is not None
             else torch.ones_like(x01[:, :1], dtype=torch.bool))
        px = x01.permute(0, 2, 3, 1).reshape(-1, 3)
        sel = w.reshape(-1)
        if int(sel.sum()) >= self.palette:
            px = px[sel]
        y = luma(px.t()[None, :, :, None])[0, 0, :, 0]
        order = torch.argsort(y)
        idx = torch.linspace(0, len(order) - 1, self.palette).round().long()
        return px[order[idx]][None].clone()

    def params(self) -> List[torch.Tensor]:
        return [self.c, self.p]

    @torch.no_grad()
    def project(self) -> None:
        self.c.clamp_(0.0, 1.0)
        self.p.clamp_(0.0, 1.0)

    def geometry(self) -> dict:
        g = super().geometry()
        # 逐列寫出**實際用到幾個顏色**，不只是要求了幾個：`softmax` 可以把
        # 某些顏色的權重壓到零，那時「K=8」與「K=3」在產物上是同一件事，
        # 而只記 K 分不出來。
        with torch.no_grad():
            used = int(torch.softmax(self.c / self.temp, dim=1
                                     ).amax(dim=(0, 2, 3)).gt(0.5).sum())
        g.update({"patch_palette": self.palette, "patch_palette_temp": self.temp,
                  "patch_palette_used": used})
        return g


class PatchVoronoiParam(PatchPaletteParam):
    """內容由 **K 個可學種子點的 Voronoi 圖** ＋ K 個可學顏色決定。

    存在理由
    ────────────────────────────────────────────────────────────────
    移植自 *Structured Adversarial Camouflage via Voronoi Diagrams*
    （arXiv:2606.17711）。該篇最佳化**種子點位置**與**每格顏色**，胞的結構
    由 Voronoi 圖決定、調色盤固定且可印，理由是「離散受限的色彩空間降低對
    數位渲染的過擬合」，並報跨 YOLOv9–12 的黑箱轉移。

    與 `PatchPaletteParam` 的差別是**幾何**：調色盤把「每個像素挑哪個顏色」
    留成一個逐像素的自由場（K·H·W 個參數），Voronoi 把它換成
    「離哪個種子最近」（K·2 個參數）。於是

        自由補丁   3·H·W          ≈ 786 000
        調色盤 K=8 K·H·W/S² + 3K  ≈  24 600（S=8）
        Voronoi K  5·K            ≈     320（K=64）

    **少三個數量級，而且產物依構造是多邊形色塊**——那正是
    `docs/DIRECTION.md` 證據清單第 5 項要的「像標記不像壞掉」。

    構造
    ────────────────────────────────────────────────────────────────
        seeds  s   (1, K, 2)   歸一化座標，值域 [0,1]
        palette p  (1, K, 3)   值域 [0,1]
        w = softmax( −‖xy − s‖² / τ_v )        逐像素對 K 個種子
        field = Σ_k w_k · p_k

    `τ_v` 越小胞邊界越硬。**用 softmax 而不是硬 argmin**，否則種子點沒有梯度
    ——argmin 對座標的導數處處為零，那是本專案已記過的零梯度陷阱。

    **不與 `tile`／`res` 疊。** 那兩個旋鈕作用在「可學張量的空間解析度」上，
    而這裡的可學張量是**座標**不是場，疊起來沒有定義。給了就拋錯，不靜默忽略。
    """

    name = "patch_voronoi"

    def __init__(self, *args, seeds: int = 48, seed_temp: float = 4e-4,
                 **kwargs):
        # **自己的旗標先驗證。** 放到 `super().__init__` 之後的話，`seeds=1`
        # 會先撞上父類的「palette 必須至少為 2」，錯誤訊息指到一個使用者沒有
        # 給的旗標上。
        if seeds < 2:
            raise ValueError(f"seeds 必須至少為 2，收到 {seeds}")
        if seed_temp <= 0.0:
            raise ValueError(f"seed_temp 必須為正，收到 {seed_temp}")
        kwargs.setdefault("palette", seeds)
        super().__init__(*args, **kwargs)
        if self.palette != seeds:
            raise ValueError(
                f"seeds={seeds} 與 palette={self.palette} 必須相同："
                "每一個 Voronoi 胞恰好配一個顏色")
        if self.tile or self.res > 1:
            raise ValueError(
                "patch_voronoi 不可與 tile／res 併用：那兩個旋鈕作用在可學"
                "張量的空間解析度上，而這裡的可學張量是座標不是場")
        self.seeds = seeds
        self.seed_temp = seed_temp

    def _content_shape(self, x01: torch.Tensor):
        """可學張量是**座標**：(1, K, 2)。"""
        return (1, self.seeds, 2)

    def _init_content(self, x01: torch.Tensor, seed: int) -> torch.Tensor:
        # 調色盤與父類同一條路徑（亮度分位數），此處只給種子點。
        self.p = self._init_palette(x01, seed).requires_grad_(True)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        n = int(math.ceil(math.sqrt(self.seeds)))
        # **抖動網格而不是純隨機**：純隨機會抽出成團的種子，那時好幾個胞是
        # 空的、對應的顏色拿不到梯度。網格保證覆蓋，抖動保證不對稱。
        ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        base = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1)[:self.seeds]
        pts = (base.to(torch.float32) + 0.5) / n
        jitter = (torch.rand(pts.shape, generator=g) - 0.5) / n
        return (pts + jitter).clamp(0, 1)[None].to(
            device=x01.device, dtype=x01.dtype).requires_grad_(True)

    def _field(self, x01: torch.Tensor) -> torch.Tensor:
        h, w = x01.shape[-2:]
        dev, dt = x01.device, x01.dtype
        yy = (torch.arange(h, device=dev, dtype=dt) + 0.5) / h
        xx = (torch.arange(w, device=dev, dtype=dt) + 0.5) / w
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coord = torch.stack([gx, gy], -1).reshape(1, -1, 2)      # (1,HW,2)
        d2 = (coord[:, :, None, :] - self.c[:, None, :, :]).pow(2).sum(-1)
        wgt = torch.softmax(-d2 / self.seed_temp, dim=-1)        # (1,HW,K)
        out = torch.einsum("nqk,nkc->nqc", wgt, self.p)          # (1,HW,3)
        return out.reshape(1, h, w, 3).permute(0, 3, 1, 2)

    @torch.no_grad()
    def project(self) -> None:
        self.c.clamp_(0.0, 1.0)
        self.p.clamp_(0.0, 1.0)

    def geometry(self) -> dict:
        # **跳過 `PatchPaletteParam.geometry`，直接呼叫 `PatchParam` 的。**
        # 前者算 `patch_palette_used` 時假設 `self.c` 是 (1,K,H,W)，而這裡的
        # `self.c` 是 (1,K,2) 的座標，形狀對不上會拋 IndexError。
        g = PatchParam.geometry(self)
        g.update({"patch_palette": self.palette,
                  "patch_palette_temp": self.temp,
                  "patch_seeds": self.seeds,
                  "patch_seed_temp": self.seed_temp})
        # 實際渲染出來用到幾個胞——種子點可以被最佳化推到重疊，那時胞數少於 K，
        # 而只記 K 分不出來。
        with torch.no_grad():
            h = w = 64
            yy = (torch.arange(h, device=self.c.device) + 0.5) / h
            gy, gx = torch.meshgrid(yy, yy, indexing="ij")
            coord = torch.stack([gx, gy], -1).reshape(1, -1, 2).to(self.c.dtype)
            d2 = (coord[:, :, None, :] - self.c[:, None, :, :]).pow(2).sum(-1)
            g["patch_palette_used"] = int(d2.argmin(-1).unique().numel())
        return g


class PatchVoronoiRandomParam(PatchVoronoiParam):
    """同 K、同色票的隨機 Voronoi，**不最佳化**，只在 reset 時抽一次種子。

    色票不重抽（與最佳化臂相同），隨機的只有種子點位置——差別因此純粹在
    「胞長在哪裡」，與 `PatchPaletteRandomParam` 的分工一致。
    """

    name = "patch_voronoi_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        g = torch.Generator(device="cpu").manual_seed(int(seed) + 9973)
        self.c = torch.rand((1, self.seeds, 2), generator=g).to(
            device=x01.device, dtype=x01.dtype).requires_grad_(False)
        self.p = self.p.detach().requires_grad_(False)

    def params(self) -> List[torch.Tensor]:
        return []

    @torch.no_grad()
    def project(self) -> None:
        return


class PatchPaletteRandomParam(PatchPaletteParam):
    """同幾何、同 K 的隨機調色盤補丁，**不最佳化**，只在 reset 時抽一次。

    存在理由與 `PatchRandomParam` 相同：讓「最佳化」與「隨機」在完全相同的
    幾何**與相同的參數化**下對照，差別只剩梯度。

    **這個對照非有不可。** `docs/DIRECTION.md` §3.5 記過一個直接相關的事故：
    色彩網格的隨機解與最佳化解**看起來一樣**，也就是產物的外觀由參數化決定
    而不由最佳化決定。調色盤同樣是一個會大幅改變外觀的參數化，沒有這個臂就
    分不出「像印花」是學出來的還是參數化本來就長那樣。

    調色盤本身**不重抽**：它由 `_init_palette` 的亮度分位數給，與最佳化臂
    完全相同。隨機的只有 logits，也就是「每個像素挑哪一個顏色」。這樣兩臂
    的色票相同，差別純粹在圖樣。
    """

    name = "patch_palette_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.rand(self._content_shape(x01), generator=g).to(
            device=x01.device, dtype=x01.dtype)
        self.c = noise.requires_grad_(False)
        self.p = self.p.detach().requires_grad_(False)

    def params(self) -> List[torch.Tensor]:
        return []

    @torch.no_grad()
    def project(self) -> None:
        return


class PatchRandomParam(PatchParam):
    """同面積的隨機補丁，**不最佳化**，只在 reset 時抽一次。

    存在理由：`runs/patch_probe/` 已經量過固定內容（噪聲／中灰／照片），
    但那一批的位置與面積是另一套挑法。這個類別讓「最佳化」與「隨機」在
    **完全相同的幾何**下對照，差別只剩梯度。
    """

    name = "patch_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        # **形狀必須走 `_content_shape`，不可以用 `x01.shape`。** 後者是全解析
        # 度，於是 `res` 與 `tile` 兩個旗標對隨機臂靜默失效：`_field` 的
        # `interpolate` 變成同尺寸的恆等、`repeat` 之後的裁切又切回原張量，
        # 兩步都不拋錯。症狀是隨機臂與 `free` 的隨機臂**逐位元相同**，而 CSV
        # 的 `patch_res` 欄仍然寫著 16——地板量錯了而表面上一切正常。
        # `res=1, tile=0` 時本式與 `x01.shape` 給出同形狀、同抽樣順序，
        # 故既有的 `patch_rand` 批次逐位元不變。
        noise = torch.rand(self._content_shape(x01), generator=g).to(
            device=x01.device, dtype=x01.dtype)
        self.c = noise.requires_grad_(False)

    def params(self) -> List[torch.Tensor]:
        return []

    @torch.no_grad()
    def project(self) -> None:
        return


class PatchPolarParam(PatchParam):
    """內容場只依**一個極座標變數**：`c = f(r)` 或 `c = g(φ)`。

    存在理由
    ────────────────────────────────────────────────────────────────
    `runs/registration_ceiling/` 量到：內容場的**粗細**這個軸解決低通、
    解決不了未對位那一種讀法，而且已經走到頭——上限由支撐邊界與 `δ = c − x`
    裡那份 −x 給，兩者都與 `c` 的參數化無關。要動幾何那一欄必須換軸，
    讓擾動**對變換本身不變**而不是讓它更平滑。

    極座標下把場寫成只依一個變數就得到它：

        c = f(r)   只依半徑   繞中心旋轉任意角度 → 逐點不變
        c = g(φ)   只依角度   繞中心縮放任意倍率 → 逐點不變

    **不變性對整個群成立，不是對某一個參數值。** 這是與已否決的 log-periodic
    候選（`runs/log_periodic_probe/`）的分界：那個構造要求 `log r` 上以 `ln s`
    為週期，只在 `s = 1.2488` 有值、相鄰的 1.15 與 1.30 歸零，等於對評測算子的
    一個特定參數 co-adapt。`g(φ)` 對裁切的餘弦在比例 0.02–0.25 的八個點上
    全部是 1.000（`runs/polar_carrier/` 第二節）。

    本專案的兩個幾何算子恰好各對上一個：`crop_resize` 是 `CROP_MODE = "center"`
    的中心裁切再放大（純中心縮放），`rotate_random` 繞影像中心。

    **原點是影像中心，不是支撐的質心。** 算子繞的是影像中心；取錯時不變性
    完全不成立，而且沒有任何症狀。守門在 `tests/test_polar_carrier.py`。

    構造
    ────────────────────────────────────────────────────────────────
        可學張量  c  (1, 3, bins)          一條一維剖面
        場        field[.., y, x] = interp(c, u(y, x))

    `u` 是 `r` 或 `φ` 映到 `[0, bins)` 的座標，逐像素預先算好。內插是線性的，
    故梯度通到剖面上相鄰的兩格。**角度用環繞內插**（`φ = ±π` 是同一點）：
    不環繞會在畫面上留一條半徑方向的硬邊，而那條邊是高頻的，正是要避開的。

    參數量是 `3 · bins`（預設 768），比 `c = φ` 的 786 432 少三個數量級。

    **已量到的限制**：貼到真實載體上之後，全圖的 `align` 只由 +0.51 升到
    +0.59（裁切）、+0.72 升到 +0.81（旋轉）——與「把 S 由 32 加粗到 512」
    同一個量級。掉的一大截在**支撐邊界**上（扣掉邊界 32 px 之後裁切是 +0.86）。
    上限沒有被打破，因為上限本來就不是內容場給的。效果讀數尚未量。
    """

    name = "patch"

    #: 剖面的格數，**同時是不變性的頻寬上限**。
    #:
    #: 不變性不是無條件成立的：格數太多時剖面本身是高頻的，場在取樣上就是
    #: 混疊的，而算子的雙線性重取樣會把混疊的部分毀掉。實測（128 px、隨機
    #: 剖面、角度場對裁切 0.02–0.25 的最小值／半徑場對旋轉 5°）：
    #:
    #:     bins      8     16     32     64    128    256
    #:     角度   0.9998 0.9992 0.9969 0.9856 0.9462 0.7879
    #:     半徑   1.0000 0.9999 0.9992 0.9908 0.9118 0.6424
    #:
    #: **32 是取捨點**：不變性還在 0.997 以上，而參數量只有 3·32 = 96
    #: （`c = φ` 是 786 432，少四個數量級）。要再細必須同時接受不變性下降，
    #: 而那正是這個構造存在的唯一理由。守門在 `tests/test_polar_carrier.py`。
    POLAR_BINS = 32

    def __init__(self, *args, polar: str = "radial", bins: int = POLAR_BINS,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if polar not in ("radial", "angular"):
            raise ValueError(f"polar 必須是 radial 或 angular，收到 {polar!r}")
        if bins < 8:
            raise ValueError(f"bins 必須至少為 8，收到 {bins}")
        # 三個旗標作用在「二維可學張量的空間解析度」上，而這裡的可學張量是
        # **一條一維剖面**。給了會被 `_field` 靜默忽略，而 CSV 的欄位照樣
        # 寫著設定值——與 `PatchVoronoiParam` 的守門同一條理由。
        if self.tile:
            raise ValueError("--patch-polar 不與 --patch-tile 併用："
                             "可學張量是一維剖面，磚長在這個構造下沒有意義")
        if self.res > 1:
            raise ValueError("--patch-polar 不與 --patch-res 併用："
                             "可學張量是一維剖面，二維降取樣在這個構造下沒有意義")
        self.polar = polar
        self.bins = int(bins)
        self._lo = None
        self._hi = None
        self._w = None

    def _content_shape(self, x01: torch.Tensor):
        return (1, 3, self.bins)

    @torch.no_grad()
    def _build_index(self, x01: torch.Tensor) -> None:
        """逐像素的剖面座標。**以影像中心為原點**，見類別 docstring。"""
        h, w = x01.shape[-2:]
        ys = torch.linspace(-1.0, 1.0, h, device=x01.device,
                            dtype=x01.dtype)[:, None]
        xs = torch.linspace(-1.0, 1.0, w, device=x01.device,
                            dtype=x01.dtype)[None, :]
        if self.polar == "radial":
            u = (ys ** 2 + xs ** 2).sqrt()
            u = u / u.max().clamp_min(1e-8) * (self.bins - 1)
            lo = u.floor().long().clamp(0, self.bins - 1)
            hi = (lo + 1).clamp(max=self.bins - 1)
        else:
            phi = torch.atan2(ys.expand(h, w), xs.expand(h, w))
            u = (phi + math.pi) / (2.0 * math.pi) * self.bins
            u = u % self.bins
            lo = u.floor().long() % self.bins
            hi = (lo + 1) % self.bins          # 環繞，`φ = ±π` 沒有接縫
        self._lo, self._hi = lo, hi
        self._w = (u - u.floor())[None, None]

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        # 索引要在 `_init_content` 之前備好——恆等起點用得到它。
        self._build_index(x01)
        super().reset(x01, seed)

    def _field(self, x01: torch.Tensor) -> torch.Tensor:
        if self._lo is None:
            raise RuntimeError("PatchPolarParam 尚未 reset，索引還沒建立")
        lo = self.c[:, :, self._lo]            # (1,3,H,W)，對 c 可微
        hi = self.c[:, :, self._hi]
        return lo * (1.0 - self._w) + hi * self._w

    def _init_content(self, x01: torch.Tensor, seed: int) -> torch.Tensor:
        """恆等起點取**支撐內每一格的平均色**——子空間裡離原圖最近的點。

        與 `res > 1` 同性質：子空間裡沒有原圖，但有一個最近點，故第 0 步的
        輸出不是逐位元恆等，而是支撐內被沿著另一個變數平均了一次。
        空的格（該半徑／角度上沒有支撐像素）落回支撐內的整體平均，不落回 0
        ——0 是黑色，會在剖面上造出一條假的硬邊。
        """
        if self.init != "identity":
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            return torch.rand(self._content_shape(x01), generator=g).to(
                device=x01.device, dtype=x01.dtype).requires_grad_(True)
        with torch.no_grad():
            wgt = (self.support.to(x01.dtype) if self.support is not None
                   else torch.ones_like(x01[:, :1]))
            idx = self._lo.reshape(-1)
            num = torch.zeros(3, self.bins, device=x01.device, dtype=x01.dtype)
            den = torch.zeros(self.bins, device=x01.device, dtype=x01.dtype)
            wv = wgt.reshape(-1)
            num.index_add_(1, idx, (x01[0].reshape(3, -1) * wv))
            den.index_add_(0, idx, wv)
            fallback = ((x01[0].reshape(3, -1) * wv).sum(1)
                        / wv.sum().clamp_min(1e-8))
            prof = torch.where(den[None] > 1e-6, num / den.clamp_min(1e-8),
                               fallback[:, None])
        return prof[None].clone().clamp(0.0, 1.0).requires_grad_(True)

    def geometry(self) -> dict:
        g = super().geometry()
        # 兩欄都由**實際跑的物件**給，不由 argparse 給：分岔時只有這一邊說實話。
        g["patch_polar"] = self.polar
        g["patch_polar_bins"] = self.bins
        return g


class PatchPolarRandomParam(PatchPolarParam):
    """同幾何、同 bins 的隨機剖面，**不最佳化**。

    存在理由與 `PatchRandomParam` 相同：讓「最佳化」與「隨機」在完全相同的
    幾何與參數化下對照，差別只剩梯度。`runs/polar_carrier/` 的三關全部是在
    隨機剖面上量的，故這個臂與那份表直接可比。
    """

    name = "patch_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        # **形狀走 `_content_shape`**：見 `PatchRandomParam.reset` 記過的坑。
        self.c = torch.rand(self._content_shape(x01), generator=g).to(
            device=x01.device, dtype=x01.dtype).requires_grad_(False)

    def params(self) -> List[torch.Tensor]:
        return []

    @torch.no_grad()
    def project(self) -> None:
        return
