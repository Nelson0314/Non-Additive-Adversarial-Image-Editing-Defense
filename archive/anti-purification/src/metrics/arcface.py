"""ArcFace 身分餘弦與 RetinaFace 偵測失敗率。

為什麼與 `identity.py` 並存而不是取代它
────────────────────────────────────────────────────────────────────
`identity.py` 用的是 FaceNet（`InceptionResnetV1`／VGGFace2），本專案所有既有
的身分讀數都是它量的。外部比較的那一族（FaceLock、DeContext、NullEdit）報的
ISM 慣例上是 **ArcFace**。兩個都報，跨批可以接回自己的歷史，對外可以接上文獻。

**兩個讀數不同調時不可以挑一個報。** 它們的訓練資料與對齊方式都不同，同一對
影像給出不同的餘弦是預期內的事，那本身是要呈現的資訊。

`retina_fail`：偵測失敗本身是一個讀數
────────────────────────────────────────────────────────────────────
身分餘弦在偵測不到臉時是空值，而「編輯之後臉沒了」與「臉還在但換了人」是兩種
不同的失效，餘弦欄位分不出來——空值會被誤讀成缺資料。NullEdit 用偵測失敗率補
這個洞，這裡照做：`det_n` 記偵測到幾張臉，`det_score` 記最高的偵測分數。

對齊方式與 `anchored_identity` 不同，這是刻意的
────────────────────────────────────────────────────────────────────
`identity.anchored_identity` 以**原圖的臉框**去裁編輯圖，問的是「原本那個位置
上的臉還是不是同一個人」。本模組讓 insightface 在每張圖上各自偵測再取嵌入，
問的是「這張圖裡的臉還是不是同一個人」。前者在臉被移動或消失時仍給得出數字，
後者在那種情形下回報偵測失敗。文獻的 ISM 是後者，故此處照文獻。

模型
────────────────────────────────────────────────────────────────────
`insightface` 的 `buffalo_l` 包：偵測 `det_10g.onnx`、辨識 `w600k_r50.onnx`
（ArcFace，512 維、已 L2 正規化）。權重由 insightface 自行快取，不在本 repo。
"""
from __future__ import annotations

from typing import Optional

_APP = None


def _app(device: str = "cpu"):
    """`FaceAnalysis` 單例。`device='cuda'` 時仍可能因 onnxruntime 未裝 GPU
    provider 而回落到 CPU——那不影響讀數，只影響速度，故不視為錯誤。"""
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        app = FaceAnalysis(name="buffalo_l", providers=providers,
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
        _APP = app
    return _APP


def _to_bgr(x01):
    """(1,3,H,W) `[0,1]` torch 張量 → insightface 要的 BGR uint8 陣列。"""
    import numpy as np

    a = (x01.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255)
    return a.round().astype(np.uint8)[:, :, ::-1]


def faces(x01, device: str = "cpu"):
    """偵測到的臉，依偵測分數由高到低排序。沒有則空串列。"""
    fs = _app(device).get(_to_bgr(x01))
    return sorted(fs, key=lambda f: float(f.det_score), reverse=True)


def embed(x01, device: str = "cpu"):
    """最高分那張臉的 ArcFace 嵌入（512 維、已正規化）。偵測不到回 `None`。"""
    fs = faces(x01, device)
    return None if not fs else fs[0].normed_embedding


def similarity(a, b) -> Optional[float]:
    """兩個嵌入的餘弦。任一為 `None` 回 `None`——**不要補 0**，
    偵測不到與相似度為零是兩件事。"""
    if a is None or b is None:
        return None
    import numpy as np

    return float(np.dot(a, b))


def arcface_row(x_orig, edit_orig, edit_def, device: str = "cpu") -> dict:
    """三張圖 → ArcFace 與偵測的讀數。

    欄位與 `identity.identity_row` 平行，前綴 `arc_`：

        arc_id_orig   原圖 對 未防禦編輯      攻擊本身改了多少身分
        arc_id_def    原圖 對 防禦後編輯      防禦之後還剩多少身分
        arc_id_drop   前者減後者              防禦造成的身分變化
        arc_det_*     各圖偵測到幾張臉與最高偵測分數

    **`arc_id_orig` 不可省。** 與 `identity.py` 的 `id_orig` 同一條理由：
    未防禦的編輯自己就換了人時，`arc_id_def` 低不代表防禦有效。
    """
    def det(x):
        fs = faces(x, device)
        return (len(fs), None if not fs else round(float(fs[0].det_score), 5),
                None if not fs else fs[0].normed_embedding)

    n0, s0, e0 = det(x_orig)
    n1, s1, e1 = det(edit_orig)
    n2, s2, e2 = det(edit_def)
    id_orig = similarity(e0, e1)
    id_def = similarity(e0, e2)
    drop = None if (id_orig is None or id_def is None) else id_orig - id_def
    return {
        "arc_id_orig": "" if id_orig is None else round(id_orig, 5),
        "arc_id_def": "" if id_def is None else round(id_def, 5),
        "arc_id_drop": "" if drop is None else round(drop, 5),
        "arc_det_n_orig": n0, "arc_det_score_orig": "" if s0 is None else s0,
        "arc_det_n_edit_orig": n1, "arc_det_score_edit_orig": "" if s1 is None else s1,
        "arc_det_n_edit_def": n2, "arc_det_score_edit_def": "" if s2 is None else s2,
    }
