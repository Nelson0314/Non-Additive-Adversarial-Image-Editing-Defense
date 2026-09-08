"""編輯後的輸出裡，還認不認得出那是誰。

為什麼需要它
────────────────────────────────────────────────────────────────────
現行的主讀數是主體區位移（遮罩加權的 LPIPS），它量的是**像素變了多少**，
量不到「還認不認得出來」。兩者會分歧，而且已經分歧過：
`runs/ip2p_patch_carrier/` 的六格位移 0.14–0.27、對隨機 3.7 倍，看起來像
有效的防禦，但拉圖看到受保護的工具箱照樣變成紅色。

在「不管怎麼換，都看不出那個人是誰」這個目標下，該量的是身分本身。

三個數，缺一不可
────────────────────────────────────────────────────────────────────
    id_orig    cos( E(原圖), E(編輯(原圖)) )      攻擊自己就已經動了多少身分
    id_def     cos( E(原圖), E(編輯(防禦圖)) )    防禦要把它壓低
    face_found 編輯輸出裡還偵測得到臉嗎

**`id_orig` 不可省。** 實測它落在 0.77–0.96 而不是 1.0——編輯本身就會改變
身分，不扣掉它就分不出「防禦壓低了身分」與「這個編輯本來就會把人改掉」。
`face_found = False` 是最強的形式：輸出裡根本沒有人臉可比。

本專案指定的元件（無論文出處，故逐列寫進 CSV）
────────────────────────────────────────────────────────────────────
- 偵測：MTCNN（`facenet-pytorch` 2.6.0 的實作）。
- 嵌入：InceptionResnetV1，`pretrained="vggface2"`。
- `margin=20`、`image_size=160` 是該套件範例的值，本模組原樣沿用。

**判定門檻不在本模組裡。** VGGFace2 上「同一人」的餘弦門檻常見取 0.5–0.6，
那只是參照；有沒有效果由使用者看數字與影像判斷（`CLAUDE.md` 的「不設判準」）。

**這不是唯一的辨識器。** 單一網路的結論可能是那個網路的特性，故
`scripts/literature_metrics.py` 另外報兩個文獻用的組合：AdaFace ViT+KPRPE ＋
DFA 對齊器（FaceLock）與 ArcFace ＋ RetinaFace（Anti-DreamBooth／FaceShield）。
**三者的絕對值不可互比**——實測同一組防禦圖在三個辨識器上的降幅差五倍。
對照表見 `docs/EVALUATION.md` 的「身分軸的指標」。
"""

from __future__ import annotations

from typing import Optional

import torch

DETECTOR_IMAGE_SIZE = 160
DETECTOR_MARGIN = 20
EMBED_WEIGHTS = "vggface2"
# 只作參照，不進任何算式，也不用來判定成敗。
SAME_PERSON_REFERENCE = 0.55

_MTCNN = None
_NET = None


def _load(device):
    global _MTCNN, _NET
    if _MTCNN is None:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        _MTCNN = MTCNN(image_size=DETECTOR_IMAGE_SIZE, margin=DETECTOR_MARGIN,
                       post_process=True, device=device)
        _NET = InceptionResnetV1(pretrained=EMBED_WEIGHTS).eval().to(device)
    return _MTCNN, _NET


@torch.no_grad()
def face_boxes(x01: torch.Tensor, device=None):
    """(1,3,H,W) → 偵測到的人臉框串列 `[(x0, y0, x1, y1), ...]`，沒有則空串列。

    存在理由不在身分讀數，而在**載體的守門**：ATR 會把臉的皮膚判成
    Upper-clothes（實測一張特寫上臉頰與額頭整片 0.4802，而 Face 類只有
    0.0696），於是「衣物載體」變成人臉。MTCNN 的框是一個獨立於 ATR 的訊號，
    兩者不一致正是那個失效的簽名。

    這裡用的偵測器與 `embed` 是同一個實例，不另外載一份。
    """
    from PIL import Image

    dev = device or x01.device
    mt, _ = _load(dev)
    arr = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    boxes, _ = mt.detect(Image.fromarray(arr.astype("uint8")))
    if boxes is None:
        return []
    return [tuple(float(v) for v in b) for b in boxes]


@torch.no_grad()
def embed(x01: torch.Tensor, device=None) -> Optional[torch.Tensor]:
    """(1,3,H,W) → (512,) 的身分向量；偵測不到臉時回 `None`。

    **回 `None` 不是錯誤，是一個讀數。** 編輯輸出裡沒有臉可比，正是「看不出
    那個人是誰」最強的形式，故呼叫端要把它與「相似度很低」分開記。
    """
    from PIL import Image

    dev = device or x01.device
    mt, net = _load(dev)
    arr = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    face = mt(Image.fromarray(arr.astype("uint8")))
    if face is None:
        return None
    return net(face[None].to(dev))[0]


def similarity(a: Optional[torch.Tensor],
               b: Optional[torch.Tensor]) -> Optional[float]:
    """兩個身分向量的餘弦相似度。任一為 `None` 時回 `None`。"""
    if a is None or b is None:
        return None
    return float(torch.nn.functional.cosine_similarity(a[None], b[None]))


def identity_row(x_orig: torch.Tensor, edit_orig: torch.Tensor,
                 edit_def: torch.Tensor, device=None) -> dict:
    """三張影像 → 逐列寫進 CSV 的六個欄位。

    `id_drop` 是 `id_orig − id_def`，**只在兩者都有臉時才有值**；其中一邊沒有
    臉時留空，因為那時相減沒有意義而填 0 會被讀成「沒有效果」。
    """
    e0 = embed(x_orig, device)
    eo = embed(edit_orig, device)
    ed = embed(edit_def, device)
    id_orig = similarity(e0, eo)
    id_def = similarity(e0, ed)
    drop = (id_orig - id_def) if (id_orig is not None and id_def is not None) else None
    return {
        "face_found_orig": e0 is not None,
        "face_found_edit_orig": eo is not None,
        "face_found_edit_def": ed is not None,
        "id_orig": "" if id_orig is None else round(id_orig, 5),
        "id_def": "" if id_def is None else round(id_def, 5),
        "id_drop": "" if drop is None else round(drop, 5),
        "id_embed_weights": EMBED_WEIGHTS,
    }
