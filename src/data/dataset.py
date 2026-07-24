"""測試資料載入 — SPEC.md §2.4。

- DAYN 測試集（150 張、3 類物件、每類 2 個編輯 prompt）向作者索取；
  取得前以 placeholder（決定性合成影像 + 合理 prompt）開發，
  config["data"]["is_placeholder"] 明確標記。
- 真實資料：data root 下每類一個子資料夾放影像，prompts.yaml 定義各類之
  編輯 prompt（{class: [prompt1, prompt2]}）。

每筆樣本：{"image_id", "concept", "image" (1,3,H,W)[0,1], "edit_prompts" [str, str]}
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.utils.device import get_device

# placeholder 之三類物件與每類兩個編輯 prompt（對應 SPEC §2.4 兩種惡意情境：
# 改變特定內容、保留特定內容而改動其他區域）。DAYN 實際內容待確認（§8 第 2 項）。
PLACEHOLDER_PROMPTS = {
    "dog": ["a cat sitting on the grass", "a dog under heavy rain at night"],
    "car": ["a wrecked car after an accident", "a car driving in a flooded street"],
    "person": ["an elderly person with gray hair", "a person standing in a burning room"],
}


def load_dataset(config: dict, max_images: int = None) -> list[dict]:
    data_cfg = config["data"]
    if data_cfg.get("is_placeholder", True):
        return _placeholder_dataset(config, max_images)
    return _folder_dataset(config, max_images)


def _placeholder_dataset(config: dict, max_images: int = None) -> list[dict]:
    """決定性合成影像（低頻雜訊上採樣），數量與類別結構依 config。"""
    h = config["generation"]["height"]
    w = config["generation"]["width"]
    n = config["data"]["n_images"]
    if max_images is not None:
        n = min(n, max_images)
    classes = list(PLACEHOLDER_PROMPTS)

    g = torch.Generator().manual_seed(config["runtime"]["seed"])
    samples = []
    for i in range(n):
        concept = classes[i % len(classes)]
        low = torch.rand(1, 3, 8, 8, generator=g)
        image = F.interpolate(low, size=(h, w), mode="bicubic", align_corners=False)
        samples.append(
            {
                "image_id": f"placeholder_{i:03d}",
                "concept": concept,
                "image": image.clamp(0, 1).to(get_device()),
                "edit_prompts": PLACEHOLDER_PROMPTS[concept],
            }
        )
    return samples


def _folder_dataset(config: dict, max_images: int = None) -> list[dict]:
    """root/<class>/*.png|jpg + root/prompts.yaml。影像 resize 至 config 之解析度。"""
    import torchvision.transforms as T
    from PIL import Image

    root = Path(config["data"]["root"])
    prompts = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    h = config["generation"]["height"]
    w = config["generation"]["width"]
    tf = T.Compose([T.Resize((h, w)), T.ToTensor()])

    samples = []
    for cls, cls_prompts in prompts.items():
        for p in sorted((root / cls).glob("*")):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            image = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(get_device())
            samples.append(
                {
                    "image_id": f"{cls}/{p.stem}",
                    "concept": cls,
                    "image": image,
                    "edit_prompts": cls_prompts,
                }
            )
            if max_images is not None and len(samples) >= max_images:
                return samples
    return samples
