"""三篇顏色論文自己的受害任務：ImageNet 分類器加 C&W margin 損失。

為什麼一定要有這一層
────────────────────────────────────────────────────────────────────
要說「這個方法移植到我的場景失效」，必須先證明**移植是對的**。沒有這一層，
「在擴散編輯上是零」與「我實作錯了」在證據上分不開，而後者是報告最容易被
問死的地方。

三篇的受害任務都是 ImageNet 分類器：

| 論文 | 載體 | 原文的受害模型 |
|---|---|---|
| ReColorAdv (NeurIPS 2019) | CIELUV 3D LUT | ResNet-50 等 |
| NCF (NeurIPS 2022) | 逐語意類別的 Lab MK 矩陣 | ResNet-50、Inception-v3 等 |
| AdvCF (TIFS 2023) | 分段線性 tone curve | ResNet-50 等 |

損失也是它們自己的：**C&W 的 logit margin**，不是本專案的 `enc`／`cond`／`id`。
`non-additive-frequency/CLAUDE.md` 寫明「跑 baseline 時用它自己的損失」。

`modified_from_paper`
────────────────────────────────────────────────────────────────────
- 三篇各自的受害模型清單不完全相同，這裡統一用 torchvision 的預訓練權重，
  預設 ResNet-50（`IMAGENET1K_V1`，三篇都涵蓋的一個）。
- 論文報的是整個 ImageNet 驗證集上的成功率；這裡是子集，樣本數逐批記在 CSV。
- 未指定目標類別時走 untargeted，判定是 top-1 是否翻掉，與三篇一致。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

# torchvision 的 ImageNet 前處理常數。載體作用在 [0,1] 的 sRGB 上，
# 正規化留在這裡做，載體因此不必知道受害模型的前處理。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_MODEL = 'resnet50'


class VictimClassifier:
    """torchvision 預訓練分類器，附 C&W margin 損失。

    `score(y01)` 回傳**要最小化**的純量，與本 repo 其他目標函數的號誌一致：
    值越小代表攻擊越成功。
    """

    def __init__(self, name: str = DEFAULT_MODEL, device=None,
                 kappa: float = 0.0):
        import torchvision.models as models

        self.name = name
        self.kappa = float(kappa)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        builder = getattr(models, name, None)
        if builder is None:
            raise ValueError(f'torchvision.models 沒有 {name!r}')
        self.model = builder(weights='IMAGENET1K_V1').to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device)[None, :, None, None]
        self.std = torch.tensor(IMAGENET_STD, device=self.device)[None, :, None, None]
        self.label: Optional[int] = None

    def logits(self, x01: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        x = x01.to(self.device).clamp(0, 1)
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear',
                              align_corners=False, antialias=True)
        return self.model((x - self.mean) / self.std)

    @torch.no_grad()
    def anchor(self, x01: torch.Tensor) -> Dict[str, float]:
        """把乾淨影像的 top-1 記成攻擊目標。攻擊成功＝把這一類打掉。"""
        lg = self.logits(x01)
        self.label = int(lg.argmax(1)[0])
        prob = lg.softmax(1)[0, self.label]
        return {'victim': self.name, 'clean_label': self.label,
                'clean_prob': float(prob)}

    def margin(self, x01: torch.Tensor) -> torch.Tensor:
        """C&W 的 untargeted margin：`z_true − max_{i≠true} z_i`。

        攻擊成功時為負。回傳 `clamp_min(−kappa)`，`kappa = 0` 時就是
        「翻過決策邊界即止」，與三篇的設定相同。
        """
        if self.label is None:
            raise RuntimeError('先呼叫 anchor() 記下乾淨影像的類別')
        lg = self.logits(x01)[0]
        true = lg[self.label]
        other = torch.cat((lg[:self.label], lg[self.label + 1:])).max()
        return (true - other).clamp_min(-self.kappa)

    def score(self, y01: torch.Tensor) -> torch.Tensor:
        return self.margin(y01)

    def terms(self, y01: torch.Tensor) -> Dict[str, torch.Tensor]:
        """與其他目標函數同介面，讓 `optimise_carrier` 的 CSV 欄位對得起來。"""
        return {'margin': self.margin(y01)}

    @torch.no_grad()
    def report(self, y01: torch.Tensor) -> Dict[str, float]:
        lg = self.logits(y01)[0]
        top = int(lg.argmax())
        return {'pred_label': top, 'pred_prob': float(lg.softmax(0)[top]),
                'true_prob': float(lg.softmax(0)[self.label]),
                'margin': float(self.margin(y01)),
                'flipped': int(top != self.label)}


class RegularisedVictim:
    """受害模型加上載體自己的正則項，兩者都進同一個純量。

    ReColorAdv 的平滑正則是**方法的一部分**，不是外加的約束——拿掉它產出的
    不是該載體的極限而是壞掉的圖（`docs/reference/BIBLIOGRAPHY.md`）。所以它
    與 margin 一起進損失，不走 `Cap` 那條路徑。
    """

    def __init__(self, victim: VictimClassifier, carrier, weight: float):
        self.victim = victim
        self.carrier = carrier
        self.weight = float(weight)

    def score(self, y01: torch.Tensor) -> torch.Tensor:
        s = self.victim.margin(y01)
        if self.weight and hasattr(self.carrier, 'smoothness'):
            s = s + self.weight * self.carrier.smoothness()
        return s

    def terms(self, y01: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {'margin': self.victim.margin(y01)}
        if hasattr(self.carrier, 'smoothness'):
            out['smoothness'] = self.carrier.smoothness()
        return out
