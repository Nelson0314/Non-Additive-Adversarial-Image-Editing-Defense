"""編輯後的輸出裡，還認不認得出那是誰。"""

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


def _iou(a, b) -> float:
    """兩個 `(x0, y0, x1, y1)` 框的 IoU。"""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1-x0) * max(0.0, y1-y0)
    area = ((a[2]-a[0])*(a[3]-a[1])) + ((b[2]-b[0])*(b[3]-b[1])) - inter
    return 0.0 if area <= 0 else inter/area


@torch.no_grad()
def embed_box(x01: torch.Tensor, box, device=None) -> torch.Tensor:
    """(1,3,H,W) ＋ 一個框 → 該框的身分向量。

    對齊與標準化逐行沿用 `MTCNN.extract`：同一個 `image_size`、`margin`、
    `post_process`，差別只在框由呼叫端指定而不是取第一個。
    """
    from facenet_pytorch.models.mtcnn import fixed_image_standardization
    from facenet_pytorch.models.utils.detect_face import extract_face
    from PIL import Image

    dev = device or x01.device
    _, net = _load(dev)
    arr = (x01[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    face = extract_face(Image.fromarray(arr.astype("uint8")), list(box),
                        DETECTOR_IMAGE_SIZE, DETECTOR_MARGIN, None)
    return net(fixed_image_standardization(face)[None].to(dev))[0]


@torch.no_grad()
def subject_identity_row(x_orig: torch.Tensor, edit_orig: torch.Tensor,
                         edit_def: torch.Tensor, device=None) -> dict:
    """把身分讀數錨定在**原圖那一張臉**上，並一併報三張圖各有幾張臉。

    與 `identity_row` 的差別只有一處：兩張編輯輸出的嵌入取「與原圖主體框
    重疊最多的那一個框」，而不是面積最大的框。重疊為零時記 `None`——那是
    「主體的位置上沒有臉了」，是一個讀數，不是錯誤，也不可用鄰近的臉頂替。
    """
    subject = face_boxes(x_orig, device)
    if not subject:
        raise ValueError("原圖偵測不到臉，無法錨定主體；這一格不可用本讀數")
    anchor = max(subject, key=lambda q: (q[2]-q[0])*(q[3]-q[1]))
    e0 = embed_box(x_orig, anchor, device)

    row = {"n_faces_orig": len(subject)}
    sims = {}
    for name, y in (("edit_orig", edit_orig), ("edit_def", edit_def)):
        boxes = face_boxes(y, device)
        row[f"n_faces_{name}"] = len(boxes)
        best = max(boxes, key=lambda q: _iou(anchor, q)) if boxes else None
        iou = _iou(anchor, best) if best is not None else 0.0
        row[f"subject_box_iou_{name}"] = round(iou, 5)
        sims[name] = (None if iou <= 0
                      else similarity(e0, embed_box(y, best, device)))

    drop = (sims["edit_orig"] - sims["edit_def"]
            if None not in sims.values() else None)
    row.update({
        "subject_id_orig": "" if sims["edit_orig"] is None else round(sims["edit_orig"], 5),
        "subject_id_def": "" if sims["edit_def"] is None else round(sims["edit_def"], 5),
        "subject_id_drop": "" if drop is None else round(drop, 5),
        "id_embed_weights": EMBED_WEIGHTS,
    })
    return row


def anchored_identity(x_orig: torch.Tensor, edit_orig: torch.Tensor,
                      edit_def: torch.Tensor, device=None, out_box=None) -> dict:
    """身分讀數改用**固定框**，不再依賴在編輯輸出上重新偵測到臉。

    `subject_identity_row` 取「與主體框重疊最多的偵測框」，偵測不到就記空值。
    那個約定在當判準時會被鑽：把顏色推到偵測器失手，身分項就拿到最好的分數，
    而人眼看過去那張臉還在、還認得出是同一個人。原 `runs/carrier_search_probe/`（不在 repo 內）
    的第一批裡 18 列有 10 列是這樣拿到 0 分的。

    這裡改成直接在**原圖主體框的座標**上裁編輯輸出並取嵌入，偵測器不參與，
    所以「偵測不到」不再是一個可以被最佳化的狀態。

    `out_box` 給輸出側用另一個座標。編輯前做過裁切或縮放時，輸出的像素座標與
    原圖不再一致，沿用原框會把**座標錯位**算進身分下降，而且配對的分母消不掉它
    （兩側的錯位不同）。參考嵌入永遠取自原圖的 `anchor`。偵測式的讀數仍一併回報
    （`*_detected`、`subject_box_iou_*`），兩者分開看得見。
    """
    subject = face_boxes(x_orig, device)
    if not subject:
        raise ValueError('原圖偵測不到臉，無法錨定主體；這一格不可用本讀數')
    anchor = max(subject, key=lambda q: (q[2]-q[0])*(q[3]-q[1]))
    e0 = embed_box(x_orig, anchor, device)
    target = anchor if out_box is None else out_box

    row = {'subject_anchor': '|'.join(str(round(float(v), 1)) for v in anchor),
           'subject_out_box': '|'.join(str(round(float(v), 1)) for v in target)}
    for name, y in (('edit_orig', edit_orig), ('edit_def', edit_def)):
        row[f'subject_id_{name}'] = round(
            float(similarity(e0, embed_box(y, target, device))), 5)
        boxes = face_boxes(y, device)
        best = max(boxes, key=lambda q: _iou(target, q)) if boxes else None
        iou = _iou(target, best) if best is not None else 0.0
        row[f'subject_box_iou_{name}'] = round(iou, 5)
        sim = None if iou <= 0 else similarity(e0, embed_box(y, best, device))
        row[f'subject_id_{name}_detected'] = '' if sim is None else round(sim, 5)
    row['subject_id_drop'] = round(
        row['subject_id_edit_orig'] - row['subject_id_edit_def'], 5)
    row['id_embed_weights'] = EMBED_WEIGHTS
    return row


def embed_box_differentiable(x01: torch.Tensor, box, device=None) -> torch.Tensor:
    """與 `embed_box` 同一個框、同一個網路，但整條路徑保持可微。

    `embed_box` 透過 PIL 的 `extract_face` 裁切與縮放，圖在那裡離開了計算圖，
    所以它只能當讀數，不能當內層的損失。這裡用張量運算重做同一件事：
    照 `facenet_pytorch.models.utils.detect_face.extract_face` 的算式擴框，
    夾在影像邊界內，再用雙線性插值縮到 `DETECTOR_IMAGE_SIZE`，
    最後套同一個 `(x*255 − 127.5) / 128` 標準化。

    縮放用 `antialias=True`，與 PIL 的 `BILINEAR` 在縮小時的抗鋸齒行為對齊；
    兩條路徑仍非逐值相同（PIL 先量化成 uint8、座標取整的細節也不同），所以
    **回報一律走 `embed_box`**，這一條只給內層用。
    """
    dev = device or x01.device
    _, net = _load(dev)
    h, w = x01.shape[-2:]
    x0, y0, x1, y1 = (float(v) for v in box)
    mx = DETECTOR_MARGIN * (x1 - x0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    my = DETECTOR_MARGIN * (y1 - y0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    a = max(0, int(x0 - mx / 2))
    b = max(0, int(y0 - my / 2))
    c = min(w, int(x1 + mx / 2))
    d = min(h, int(y1 + my / 2))
    if c - a < 2 or d - b < 2:
        raise ValueError(f'擴框之後的裁切區域退化：{(a, b, c, d)}')
    crop = x01[..., b:d, a:c].to(dev)
    face = torch.nn.functional.interpolate(
        crop, size=(DETECTOR_IMAGE_SIZE, DETECTOR_IMAGE_SIZE),
        mode='bilinear', align_corners=False, antialias=True)
    return net((face * 255.0 - 127.5) / 128.0)[0]
