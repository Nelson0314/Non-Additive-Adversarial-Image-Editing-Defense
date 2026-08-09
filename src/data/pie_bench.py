"""PIE-Bench 載入與抽樣 — `docs/reference/CODE_CONTRACTS.md` §1.5。

出處：PnP Inversion（ICLR 2024），[cure-lab/PnPInversion](https://github.com/cure-lab/PnPInversion)。
DIA（ICCV 2025）用的是同一個 benchmark，故本專案與 DIA 的數字可直接對照。

## 資料結構

```
data/
├── annotation_images/
│   ├── 0_random_140/  1_change_object_80/  2_add_object_80/
│   ├── 3_delete_object_80/  4_change_attribute_content_40/
│   ├── 5_change_attribute_pose_40/  6_change_attribute_color_40/
│   ├── 7_change_attribute_material_40/  8_change_background_80/
│   └── 9_change_style_80/
└── mapping_file.json
```

`mapping_file.json` 的每一筆：

```json
"000000000000": {
    "image_path": "0_random_140/000000000000.jpg",
    "original_prompt": "a slanted mountain bicycle on the road in front of a building",
    "editing_prompt": "a slanted [rusty] mountain bicycle on the road in front of a building",
    "editing_instruction": "Make the frame of the bike rusty",
    "editing_type_id": "0",
    "blended_word": "bicycle bicycle",
    "mask": [...]
}
```

## 兩個容易踩到的地方

**方括號必須剝掉。** `editing_prompt` 用 `[...]` 標記被編輯的片段，那是給人看的標記，
不是 prompt 的一部分。直接餵給 CLIP／SigLIP 或 SD 會把 `[` 與 `]` 當成 token，
使語意分數與編輯結果都偏掉，而且不會有任何症狀——輸出仍是一張合理的圖。

**類型編號有 10 個（0–9），不是 9 個。** DIA 論文寫的「9 sub-tasks」指的是
編輯子任務 1–9；編號 0 是 `random`（140 張，志願者或文獻來源的 prompt），
不屬於任何單一編輯類型。分層抽樣時兩種算法都合理，故由 `include_random` 明示，
不預設——這正是「一個值為誰校準只寫在註解裡」那類缺陷的來源。

## 樣本數的唯一入口

`n` 是全流程唯一的樣本數參數。**任何地方不得出現字面值樣本數**，
使 N 由 3 擴到 150 只需改一個設定值（`tests/test_scale_n.py` 以 grep 斷言此事）。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# 編號 → 目錄前綴。用於核對 image_path 與 editing_type_id 是否一致——
# 兩者不一致代表資料集被改動過，必須立刻知道而不是靜默採用其中一個。
EDITING_TYPES: Dict[str, str] = {
    "0": "random",
    "1": "change_object",
    "2": "add_object",
    "3": "delete_object",
    "4": "change_attribute_content",
    "5": "change_attribute_pose",
    "6": "change_attribute_color",
    "7": "change_attribute_material",
    "8": "change_background",
    "9": "change_style",
}

# 編號 0 是 random，不屬於單一編輯類型
EDIT_SUBTASKS = tuple(k for k in EDITING_TYPES if k != "0")

_BRACKET = re.compile(r"[\[\]]")


def strip_brackets(prompt: str) -> str:
    """剝掉 `[...]` 標記，保留其中的文字。

    `"a slanted [rusty] mountain bicycle"` → `"a slanted rusty mountain bicycle"`。

    只移除括號本身而非括號內的內容：括號標的是「被編輯的片段」，
    那段文字正是目標語意所在，刪掉會讓 target prompt 退化成 source prompt。
    """
    return _BRACKET.sub("", prompt)


@dataclass(frozen=True)
class Sample:
    """一張影像與其標註。`image_path` 為絕對路徑，其餘照原始欄位。"""

    key: str
    image_path: Path
    source_prompt: str
    target_prompt: str          # 已剝除方括號
    target_prompt_raw: str      # 原始帶括號的字串，供追溯
    instruction: str
    subtask: str                # editing_type_id
    blended_word: str
    mask: Optional[list]        # RLE；SDEdit 威脅模型用不到，保留供 inpainting 方向

    @property
    def subtask_name(self) -> str:
        return EDITING_TYPES[self.subtask]


class DatasetError(RuntimeError):
    """資料集結構不符預期。一律拋出——靜默略過壞掉的筆會讓 N 對不上。"""


def load(root, require_images: bool = True) -> List[Sample]:
    """讀取 `mapping_file.json`，回傳依 key 排序的 Sample 清單。

    `require_images=False` 供單元測試使用，此時不檢查影像檔是否存在。
    正式流程必須為 True：缺圖若到訓練時才發現，該格已經佔掉了排程。
    """
    root = Path(root)
    mapping = root / "mapping_file.json"
    if not mapping.exists():
        raise DatasetError(
            f"找不到 {mapping}。PIE-Bench 取自 https://github.com/cure-lab/PnPInversion，"
            "需要 annotation_images/ 與 mapping_file.json 兩者。"
        )
    raw = json.loads(mapping.read_text(encoding="utf-8"))

    out: List[Sample] = []
    for key in sorted(raw):
        e = raw[key]
        subtask = str(e["editing_type_id"])
        if subtask not in EDITING_TYPES:
            raise DatasetError(f"{key} 的 editing_type_id={subtask!r} 不在 0–9 內")

        rel = e["image_path"]
        # 目錄前綴帶編號（`0_random_140/…`），與 editing_type_id 必須一致。
        # 不一致代表資料集被改動過或抽樣時混入了別的類型，那會讓分層失去意義。
        prefix = rel.split("/")[0].split("_")[0]
        if prefix != subtask:
            raise DatasetError(
                f"{key} 的 image_path 前綴 {prefix!r} 與 editing_type_id {subtask!r} 不符"
            )

        path = root / "annotation_images" / rel
        if require_images and not path.exists():
            raise DatasetError(f"{key} 的影像不存在：{path}")

        target_raw = e["editing_prompt"]
        out.append(Sample(
            key=key,
            image_path=path,
            source_prompt=e["original_prompt"],
            target_prompt=strip_brackets(target_raw),
            target_prompt_raw=target_raw,
            instruction=e.get("editing_instruction", ""),
            subtask=subtask,
            blended_word=e.get("blended_word", ""),
            mask=e.get("mask"),
        ))
    return out


def stratified_pick(samples: Sequence[Sample], n: int, seed: int,
                    include_random: bool = False) -> List[Sample]:
    """由不同子任務各取，盡量讓 n 張分散在最多的類型上。

    `include_random=False` 排除編號 0（random）——它不屬於任何單一編輯類型，
    混進來會讓「三張圖分屬三種編輯型態」這個聲明不成立。預設排除，
    但由參數明示而非寫死，因為擴大 N 時可能會需要它。

    n 大於可用類型數時，多出來的名額依序回到已取過的類型，
    仍以每類型至多相差一張的方式分配。
    """
    if n <= 0:
        raise ValueError(f"n 必須為正，收到 {n}")

    pool = [s for s in samples
            if include_random or s.subtask in EDIT_SUBTASKS]
    if len(pool) < n:
        raise DatasetError(
            f"可用樣本 {len(pool)} 張少於要求的 n={n}"
            f"（include_random={include_random}）"
        )

    import random
    rng = random.Random(seed)

    by_type: Dict[str, List[Sample]] = {}
    for s in pool:
        by_type.setdefault(s.subtask, []).append(s)
    for v in by_type.values():
        rng.shuffle(v)

    # 依類型編號輪流取，使前 k 張必落在 k 個不同類型上
    order = sorted(by_type)
    picked: List[Sample] = []
    round_idx = 0
    while len(picked) < n:
        progressed = False
        for t in order:
            if len(picked) >= n:
                break
            if round_idx < len(by_type[t]):
                picked.append(by_type[t][round_idx])
                progressed = True
        if not progressed:
            raise DatasetError("樣本耗盡；此處應已被上方的長度檢查擋下")
        round_idx += 1
    return picked


def filter_editable(samples: Sequence[Sample], edit_effect_fn,
                    threshold: float) -> List[Sample]:
    """只留「未防禦編輯確實成功」者。

    `edit_effect_fn(sample) -> float` 由呼叫端提供，回傳
    `SigLIP(編輯輸出, target) − SigLIP(原圖, target)`。抽成參數是為了讓本模組
    不依賴 SD 與評分模型，單元測試才不需要 GPU。

    此過濾是三個目標共用的前提，有量測依據：先驗實驗的 24 張裡有 6 張的
    未防禦編輯不構成有效編輯，其中一張的 edit_effect 為 −0.0007
    （編輯後反而更遠離 target）。在這類影像上量免疫效果不具意義。

    **CLIP 不可用於此判定**：實測 +0.0101 ± 0.0169，標準差大於均值；
    SigLIP 通過同一對照（+0.0276 ± 0.0237）。
    """
    return [s for s in samples if edit_effect_fn(s) > threshold]
