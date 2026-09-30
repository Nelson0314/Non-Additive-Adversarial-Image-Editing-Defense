"""主體遮罩極性與幾何淨化；先翻極性再變換，旋轉黑角歸背景。"""
import torch
from immunization_core.purifiers.operators import Purifier
from immunization_core.purifiers.protocol import PURIFIERS, label

GEOMETRIC_KINDS = ("crop_resize", "rotate")
PURIFIER_SPEC = {label(kind, strength): (kind, strength) for kind, strength in PURIFIERS}
GEOMETRIC = tuple(tag for tag, (kind, _) in PURIFIER_SPEC.items() if kind in GEOMETRIC_KINDS)

def subject_mask(repaint: torch.Tensor) -> torch.Tensor:
    """由 inpainting 遮罩換算主體遮罩。

    **極性要翻**：`data/*/masks` 是白＝重繪，而重繪區正是主體以外的地方；
    `split_displacement` 要的是主體遮罩（1 = 主體）。不翻的話主體內與主體外
    兩欄會整組對調，而且不會報錯。
    """
    return 1.0 - (repaint >= 0.5).float()

def purified_mask(mask: torch.Tensor, purifier: str) -> torch.Tensor:
    """把主體遮罩送過與影像同一道淨化算子。

    **只有幾何類需要這一步。** `crop_resize0.1` 與 `rotate15` 改掉的是取景：
    淨化後的兩張圖裡，主體已經不在原來的像素座標上，拿未變換的遮罩去切，
    `disp_purified_subject` 與 `disp_purified_background` 兩欄切到的就不是
    主體與背景。非幾何類（jpeg、blur）不動座標，遮罩原樣即可。

    變換順序是**先翻極性再變換**：`subject_mask` 先把重繪遮罩換成主體遮罩
    （1 = 主體），再套算子。`rotate` 邊界補零，於是旋轉後離開畫面的區域取值 0，
    落在 `split_displacement` 的補集側（背景）——那塊在淨化後的圖上是黑角，
    不是主體。反過來先變換再翻極性會把黑角算成主體。

    插值後重新二值化（門檻 0.5）：`crop_resize` 走 bicubic、`rotate` 走雙線性，
    邊緣會出現中間值，而 `split_displacement` 要的是 0／1 的權重。
    """
    if purifier not in GEOMETRIC:
        return mask
    kind, strength = PURIFIER_SPEC[purifier]
    with torch.no_grad():
        moved = Purifier(kind, strength).evaluate(mask)
    return (moved >= 0.5).float()
