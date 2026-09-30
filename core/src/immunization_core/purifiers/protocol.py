"""活動主表與 lab 共用的 identity 對照及七道淨化；順序與強度固定。"""

#: 逐項 (kind, strength)。標籤與 `phase_retention.label()` 同式，
#: 故兩邊的 CSV 可以直接對起來。
PURIFIERS = (
    ("identity", 0.0),
    ("crop_resize", 0.1),
    ("jpeg", 30),
    ("jpeg", 50),
    ("jpeg", 80),
    ("blur", 1.0),
    ("blur", 2.0),
    ("rotate", 15.0),
)


def label(kind: str, strength: float) -> str:
    return kind if not strength else f"{kind}{strength:g}"
