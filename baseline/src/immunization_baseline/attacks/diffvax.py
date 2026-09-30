"""DiffVax —— Ozden, Kara, Akcin, Zaman, Srivastava, Chinchali, Rehg，ICLR 2026。

*DiffVax: Optimization-Free Image Immunization Against Diffusion-Based
Editing*（arXiv:2411.17957）。官方程式 `ozdentarikcan/DiffVax`，本檔對照的
commit 為 `77fe66a8e6d50223a929081ec0851d9a49153903`。逐條佐證見
`docs/reference/AUDIT_MIST_DIFFVAX.md` 的「對象二：DiffVax」。

機制
──────────────────────────────────────────────────────────────────────

DiffVax 是**前饋式 immunizer**：一個 UNet++（`NestedUNet`，9,170,721 參數）
對輸入影像做一次前向，直接輸出要加上去的擾動，不做逐影像的迭代最佳化。
訓練目標（`src/diffvax/immunization/diffvax_immunization.py:140-159`）：

    L = alpha * L_noise + L_edit
    L_edit  = ||SD_inpaint(I_im, ~M, P) * ~M||_1 / sum(~M)
    L_noise = ||(I_im - I) * M||_1 / sum(M)

`L_edit` 把 inpainting 的輸出壓向全零（`target_image = torch.zeros_like(...)`，
在 `[-1,1]` 值域即中灰）；`L_noise` 壓住擾動的幅度。**alpha = 4**
（`configs/train.yml`；函式簽名的預設 `alpha=1` 不是實跑值）。

本檔只實作推論路徑（`immunize_img`）與這兩個損失項的算術
（`diffvax_terms`，供測試釘住方向），不實作訓練迴圈——官方權重已隨 repo 提供。

遮罩的命名與極性（會靜默寫反的地方）
──────────────────────────────────────────────────────────────────────

**程式與論文的 mask 命名互補**：

| 符號 | 含意 | 本檔的變數 |
|---|---|---|
| 論文 `~M`、程式 `mask_batch` | **編輯區域**（inpainting 要重繪的地方） | `mask_edit`，1 = 重繪 |
| 論文 `M`、程式 `1 - mask_batch` | **免疫區域**（擾動加在這裡） | `1 - mask_edit` |

本 repo 的遮罩由 `scripts/make_masks.py` 產生於
`data/<資料集>/masks/<影像>.png`，其極性寫在該檔的 docstring：
**白（255）＝ 要重繪的區域，黑（0）＝ 保留的區域**。

因此 **本 repo 的白 = DiffVax 的 `mask_batch` = 1 = 論文的 `~M`**，
兩邊極性一致，載入時不需要反相。官方的門檻同樣取 0.5
（`src/diffvax/utils.py:38-39` 把 `<0.5` 歸 0、`>=0.5` 歸 1），本檔照做。

反過來說，**擾動落在白色區域之外**（黑色的保留區）。這是 DiffVax 的設計：
重繪區的像素反正會被模型換掉，在那裡下擾動沒有作用，所以免疫區是保留區。
`tests/test_diffvax.py` 以「白區內擾動為 0」釘住這個對應。

與論文的落差（照原始碼，不照論文正文）
──────────────────────────────────────────────────────────────────────

1. **immunizer 吃的是 masked image，不是完整原圖。** 論文寫
   「The immunizer model f(.;theta) takes an input image I」，但
   `scripts/train.py:72-76` 傳入 `prepare_mask_and_masked_image` 回傳的第二個
   值（`masked_image = image * (mask < 0.5)`，即**編輯區已歸零**的影像），
   完整原圖 `non_masked_image_torch` 載入後**從未被使用**；
   `scripts/demo.py:288-294` 與 `app.py:92-98` 的推論路徑同樣傳 masked image。
   **這代表 DiffVax 在免疫階段就必須事先知道編輯遮罩**——它不是一個
   「不知道攻擊者要改哪裡」也能用的防禦，這是它與逐影像 PGD 基線在
   威脅模型上的結構性差異，不是實作細節。本檔照原始碼傳 masked image。
2. **沒有硬性 L-infinity 上界。** `self.eps = 32/255`、`self.step_size = 1`
   （`diffvax_immunization.py:48-49`）與 `immunize_img(epsilon=32)` 的參數
   在整個 DiffVax 路徑中**從未被引用**；唯一的幅度約束是訓練時的
   `alpha * L_noise` 這個軟性懲罰，推論時只有 `clamp(-1, 1)`。
   本檔的 `immunize(..., linf_eps=None)` **預設關閉**約束，與原始碼一致；
   要做等失真比較時才明給一個值，那是本專案的加工，不是原論文設定。
3. **只支援 inpainting。** `src/diffvax/attack.py` 只實例化
   `StableDiffusionInpaintPipeline`，可微前傳硬編碼 9 通道輸入；全 repo 對
   `img2img` / `SDEdit` / `InstructPix2Pix` 的搜尋為 0 筆。綁定的模型是
   `runwayml/stable-diffusion-inpainting`（SD 1.5 inpainting）。
   本專案的評測骨幹走 img2img/SDEdit，兩者的編輯管線不同，見下節。
4. **輸出層無 activation**（`model.py:103` 是一個 1x1 Conv），擾動的幅度
   完全由訓練時的 `alpha * L_noise` 決定，沒有結構上的上界。

權重
──────────────────────────────────────────────────────────────────────

`checkpoints/diffvax_trained.pth` **直接入庫在官方 repo**（不是 LFS、不是
release asset），`git clone` 即取得。實測：36,782,602 bytes，
sha256 `b52500bf07a23aaf3d499adfaebe3e894c5d4fe2b4607fd6aa7fa64665bc43c0`，
`collections.OrderedDict`（純 state_dict，非完整模型），212 個 tensor、
9,170,721 個參數、fp32（`num_batches_tracked` 為 int64），鍵名與本檔的
`NestedUNet` 逐一對應。**對應的訓練步數無從得知**：`iter_num: 1000000` 是
epoch 數、沒有 early stopping、`torch.save` 只在跑完或 loss 變 NaN 時觸發。

尺寸
──────────────────────────────────────────────────────────────────────

官方一律 512x512（`app.py:87-89` 明寫 "model requirement"，
`utils.load_image_from_path` 的 `size=(512, 512)`）。`NestedUNet` 本身有
四層 `MaxPool2d(2,2)`，故邊長必須是 16 的倍數；不是的話 `torch.cat` 會因為
上採樣後的尺寸對不上而報錯。本檔**不靜默縮放**，邊長不整除 16 時直接拋錯，
由呼叫端決定要縮放到什麼尺寸——靜默縮放會讓「這張防禦圖是在什麼解析度上
產生的」從產物看不出來。512 以外的邊長在原作的訓練分布外，這件事本身
沒有被官方量過。

怎麼接進本專案的評測
──────────────────────────────────────────────────────────────────────

**本檔刻意不提供 `BaselineSpec`、也不進 `REGISTRY`。** `REGISTRY` 的契約是
給逐影像 PGD 的（`eps` / `steps` / `step_size` / `update_rule` / `loss_fn`），
DiffVax 一項都沒有；且 `tests/test_baselines.py` 以 `AUDIT == REGISTRY`
稽核，塞進去會讓那份稽核表失去意義。接法是在批次腳本裡直接呼叫：

    from immunization_baseline.attacks.diffvax import immunize, load_immunizer

    model = load_immunizer(ckpt_path, device=device)          # 一次
    for item in items:                                        # 每張
        x01 = load_image_tensor(item["path"], device, size=512)
        m01 = load_image_tensor(Path(data) / "masks" / f"{item['name']}.png",
                                device, size=512)             # 白＝重繪
        x_def01 = immunize(model, x01, m01)["x_def01"]

`x_def01` 的形狀與值域（`(1,3,H,W)`、`[0,1]`）和 `run_pgd(...).x_adv01`
相同，故後續的 `evaluate(...)` 不必改。**但要在報表註明兩件事**：
(a) 這一條需要遮罩，其餘基線不需要；(b) DiffVax 的擾動只存在於遮罩之外，
而本專案的評測走 img2img/SDEdit（全圖重建），與 DiffVax 訓練時的
inpainting 管線不同——防禦圖仍然可以評，但那不是原作者量過的設定。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import nn

# ---------------------------------------------------------------------------
# 官方常數。改動這裡等於宣稱 AUDIT_MIST_DIFFVAX.md 的查證結果有誤。
# ---------------------------------------------------------------------------

#: `configs/train.yml` 的 `alpha`。函式簽名的預設 1 不是實跑值。
DIFFVAX_ALPHA = 4.0

#: `diffvax_immunization.py:50-51` 的 `clamp_min` / `clamp_max`。
CLAMP_MIN = -1.0
CLAMP_MAX = 1.0

#: `utils.prepare_mask_and_masked_image` 的二值化門檻。
MASK_THRESHOLD = 0.5

#: 官方唯一量過的邊長（`app.py:87-89`）。
PAPER_RESOLUTION = 512

#: `NestedUNet` 的四層 MaxPool2d(2,2) 要求邊長整除 16。
SIZE_DIVISOR = 16

#: `model.py:70` 的 `nb_filter`。
NB_FILTER = (32, 64, 128, 256, 512)

#: `checkpoints/diffvax_trained.pth` 的實測值（見模組 docstring）。
CHECKPOINT_BYTES = 36_782_602
CHECKPOINT_SHA256 = (
    "b52500bf07a23aaf3d499adfaebe3e894c5d4fe2b4607fd6aa7fa64665bc43c0"
)
CHECKPOINT_NUM_TENSORS = 212
CHECKPOINT_NUM_PARAMS = 9_170_721

#: 訓練時綁定的編輯模型（`configs/train.yml` 的 `attack_model_link`）。
ATTACK_MODEL_ID = "runwayml/stable-diffusion-inpainting"

#: `configs/train.yml` 與 `diffvax_immunization.py` 的訓練設定，供報表引用。
TRAIN_SETTINGS = {
    "dataset": "ozdentarikcan/DiffVaxDataset (train 800 / validation 200)",
    "iter_num": 1_000_000,      # 實為 epoch 數，無 early stopping
    "batch_size": 5,
    "optimizer": "Adam",
    "learning_rate": 1e-5,
    "alpha": DIFFVAX_ALPHA,
    "seed": 5,
    "amp": "torch.cuda.amp.GradScaler，模型與影像走 fp16",
    "train_num_inference_steps": 4,
}


# ---------------------------------------------------------------------------
# 網路。鍵名必須與官方 checkpoint 逐一對應，故屬性名照抄不改。
# 來源：`src/diffvax/model.py`，其檔頭註明改自
# https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
# ---------------------------------------------------------------------------


class VGGBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU（`model.py:7-25`）。"""

    def __init__(self, in_channels: int, middle_channels: int,
                 out_channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class NestedUNet(nn.Module):
    """UNet++（`model.py:66-134`）。`deep_supervision=False`，輸出 3 通道。

    **輸出層是一個 1x1 Conv，沒有 activation 也沒有範圍限制**
    （`model.py:103`），故前向的輸出不是擾動的上界——上界只有後面的
    `clamp(-1, 1)`。
    """

    def __init__(self, num_classes: int = 3, input_channels: int = 3,
                 deep_supervision: bool = False):
        super().__init__()
        nb = list(NB_FILTER)
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb[0], nb[0])
        self.conv1_0 = VGGBlock(nb[0], nb[1], nb[1])
        self.conv2_0 = VGGBlock(nb[1], nb[2], nb[2])
        self.conv3_0 = VGGBlock(nb[2], nb[3], nb[3])
        self.conv4_0 = VGGBlock(nb[3], nb[4], nb[4])

        self.conv0_1 = VGGBlock(nb[0] + nb[1], nb[0], nb[0])
        self.conv1_1 = VGGBlock(nb[1] + nb[2], nb[1], nb[1])
        self.conv2_1 = VGGBlock(nb[2] + nb[3], nb[2], nb[2])
        self.conv3_1 = VGGBlock(nb[3] + nb[4], nb[3], nb[3])

        self.conv0_2 = VGGBlock(nb[0] * 2 + nb[1], nb[0], nb[0])
        self.conv1_2 = VGGBlock(nb[1] * 2 + nb[2], nb[1], nb[1])
        self.conv2_2 = VGGBlock(nb[2] * 2 + nb[3], nb[2], nb[2])

        self.conv0_3 = VGGBlock(nb[0] * 3 + nb[1], nb[0], nb[0])
        self.conv1_3 = VGGBlock(nb[1] * 3 + nb[2], nb[1], nb[1])

        self.conv0_4 = VGGBlock(nb[0] * 4 + nb[1], nb[0], nb[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2),
                    self.final3(x0_3), self.final4(x0_4)]
        return self.final(x0_4)


# ---------------------------------------------------------------------------
# 載入
# ---------------------------------------------------------------------------


def load_immunizer(checkpoint: Union[str, Path],
                   device: Optional[torch.device] = None,
                   dtype: torch.dtype = torch.float32,
                   strict: bool = True) -> NestedUNet:
    """載入官方權重成一個可推論的 `NestedUNet`。

    輸入輸出契約
        checkpoint  官方 `checkpoints/diffvax_trained.pth`。內容是一個
                    純 `state_dict`（`OrderedDict`），**不是**完整模型物件，
                    也不含 optimizer 狀態或 epoch 數。
        回傳        `NestedUNet(num_classes=3)`，`eval()` 模式、
                    `requires_grad_(False)`。輸入 `(N,3,H,W)`、輸出
                    `(N,3,H,W)`，H、W 必須整除 16。

    `strict=True`（預設）：鍵名對不上就拋錯。官方權重共 212 個 tensor、
    9,170,721 個參數；鍵名若對不上代表拿到的不是這份權重，靜默放行會讓
    後面每一張防禦圖都是隨機初始化的輸出，而那看起來仍然像一張合理的圖。
    """
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 DiffVax 權重 {path}。官方權重直接入庫在 "
            f"ozdentarikcan/DiffVax 的 checkpoints/diffvax_trained.pth，"
            f"git clone 即取得（{CHECKPOINT_BYTES} bytes，"
            f"sha256 {CHECKPOINT_SHA256}）。")

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(
            f"{path} 的內容是 {type(state).__name__}，不是 state_dict。")

    model = NestedUNet(num_classes=3)
    model.load_state_dict(state, strict=strict)
    model.eval().requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model.to(dtype)


# ---------------------------------------------------------------------------
# 遮罩
# ---------------------------------------------------------------------------


def binarise_mask(mask: torch.Tensor,
                  threshold: float = MASK_THRESHOLD) -> torch.Tensor:
    """任意形狀的重繪遮罩 -> `(N,1,H,W)` 的二值 `mask_edit`（1 = 重繪）。

    接受 `(N,1,H,W)` 或 `(N,3,H,W)`（`make_masks.py` 經
    `torchvision.utils.save_image` 寫出的是三通道的灰階 PNG，讀回來是
    `(1,3,H,W)`）。三通道時取第 0 通道——**不取平均**，因為平均會把
    「三個通道不一致」這種壞遮罩靜默變成中間值，再被門檻切掉。

    門檻 0.5 與 `utils.prepare_mask_and_masked_image`（`utils.py:38-39`）
    一致。輸入的值域是 `[0,1]`：白（1）＝ 重繪 ＝ 論文的 `~M`
    ＝ 官方程式的 `mask_batch`。
    """
    if mask.dim() != 4:
        raise ValueError(f"遮罩要是 (N,C,H,W)，收到 {tuple(mask.shape)}")
    if mask.shape[1] == 3:
        chan = mask[:, :1]
        spread = (mask - chan).abs().max()
        if spread > 1e-3:
            raise ValueError(
                f"三通道遮罩的通道之間不一致（最大差 {float(spread):.4f}）。"
                "重繪遮罩必須是灰階，通道不一致代表拿到的不是遮罩。")
        mask = chan
    elif mask.shape[1] != 1:
        raise ValueError(f"遮罩的通道數要是 1 或 3，收到 {mask.shape[1]}")
    return (mask >= threshold).to(mask.dtype)


def _check_size(x: torch.Tensor) -> None:
    h, w = x.shape[-2:]
    if h % SIZE_DIVISOR or w % SIZE_DIVISOR:
        raise ValueError(
            f"DiffVax 的 UNet++ 有四層 MaxPool2d(2,2)，邊長必須整除 "
            f"{SIZE_DIVISOR}，收到 {h}x{w}。官方唯一量過的是 "
            f"{PAPER_RESOLUTION}x{PAPER_RESOLUTION}；要換尺寸請由呼叫端"
            "明確縮放，本檔不靜默縮放。")


# ---------------------------------------------------------------------------
# 推論
# ---------------------------------------------------------------------------


def immunize(model: NestedUNet,
             x01: torch.Tensor,
             mask01: torch.Tensor,
             *,
             threshold: float = MASK_THRESHOLD,
             linf_eps: Optional[float] = None,
             restore_edit_region: bool = True) -> Dict[str, torch.Tensor]:
    """對一張（或一批）影像做前饋式免疫。對應 `immunize_img` 加上 demo 的合成。

    輸入輸出契約
        x01     `(N,3,H,W)`、`[0,1]`（本 repo 的通用影像約定）。
        mask01  `(N,1,H,W)` 或 `(N,3,H,W)`、`[0,1]`，**白＝重繪**。
        回傳    dict，全部 `(N,·,H,W)`：
                `x_def01`   防禦圖，`[0,1]`，形狀同 `x01`。
                `delta`     `[-1,1]` 值域上的擾動，遮罩內恆為 0。
                `mask_edit` 二值化後的 `mask_batch`，1 = 重繪。
                `x_masked`  進入 UNet++ 的 masked image（`[-1,1]`）。
                `x_adv`     `[-1,1]` 上 clamp 後的 masked image 加擾動，
                            即官方 `immunize_img` 的回傳值（**重繪區是 0**）。

    步驟（逐行對應官方程式）
        1. `x = 2*x01 - 1`                      `utils.py:33`
        2. `mask_edit = (mask01 >= 0.5)`        `utils.py:38-39`
        3. `x_masked = x * (1 - mask_edit)`     `utils.py:43`
           —— **進網路的是編輯區已歸零的影像**，見模組 docstring 落差 1。
        4. `delta = f(x_masked) * (1 - mask_edit)`
                                                `diffvax_immunization.py:74-76`
        5. `x_adv = clamp(x_masked + delta, -1, 1)`
                                                `diffvax_immunization.py:78`
        6. 合成回原圖：重繪區放回原像素，其餘放 `x_adv`。
           對應 `demo.py:297` 的 `recover_image(..., background=True)`
           （`utils.py:23` 的 `mask * init_image + (1 - mask) * image`）。
           `restore_edit_region=False` 可關掉，那會直接輸出重繪區為中灰的
           `x_adv`——那不是官方交出去的圖，只在檢查用。

    `linf_eps`
        **預設 `None`，即不施加任何硬性約束**，與官方一致：官方的
        `self.eps = 32/255` 與 `immunize_img(epsilon=32)` 從未被引用，
        唯一的幅度控制是訓練時的 `alpha * L_noise`。明給一個值會在步驟 5
        之前把 `delta` 夾到 `[-linf_eps, linf_eps]`（`[-1,1]` 值域上量），
        **那是本專案為了等失真比較而加的，不是原論文設定，報表必須註明。**
    """
    if x01.dim() != 4 or x01.shape[1] != 3:
        raise ValueError(f"影像要是 (N,3,H,W)，收到 {tuple(x01.shape)}")
    if x01.shape[-2:] != mask01.shape[-2:]:
        raise ValueError(
            f"影像 {tuple(x01.shape[-2:])} 與遮罩 {tuple(mask01.shape[-2:])} "
            "的尺寸不同。")
    _check_size(x01)

    mask_edit = binarise_mask(mask01.to(x01.dtype), threshold)
    keep = 1.0 - mask_edit                      # 免疫區 = 論文的 M

    x = x01 * 2.0 - 1.0
    x_masked = x * keep

    out = model(x_masked)
    delta = out * keep
    if linf_eps is not None:
        if linf_eps <= 0:
            raise ValueError(f"linf_eps 要是正數，收到 {linf_eps}")
        delta = delta.clamp(-linf_eps, linf_eps) * keep

    x_adv = torch.clamp(x_masked + delta, CLAMP_MIN, CLAMP_MAX)

    if restore_edit_region:
        composed = mask_edit * x + keep * x_adv
    else:
        composed = x_adv

    return {
        "x_def01": ((composed + 1.0) / 2.0).clamp(0.0, 1.0),
        "delta": delta,
        "mask_edit": mask_edit,
        "x_masked": x_masked,
        "x_adv": x_adv,
    }


# ---------------------------------------------------------------------------
# 訓練損失（只有算術，不含訓練迴圈）
# ---------------------------------------------------------------------------


def diffvax_terms(edited: torch.Tensor,
                  x_adv: torch.Tensor,
                  x_masked: torch.Tensor,
                  mask_edit: torch.Tensor,
                  alpha: float = DIFFVAX_ALPHA) -> Dict[str, torch.Tensor]:
    """`diffvax_immunization.py:157-159` 的兩項損失，逐字對應。

        loss1 = ||(edited - 0) * mask_edit||_1 / sum(mask_edit)      = L_edit
        loss2 = ||alpha * (x_adv - x_masked) * (1 - mask_edit)||_1
                / sum(1 - mask_edit)                            = alpha*L_noise
        total = loss1 + loss2

    `edited` 是 inpainting pipeline 的可微輸出（`[-1,1]`）；官方的
    `target_image = torch.zeros_like(img_out)`，故第一項化簡為輸出本身的
    加權 L1——**把重繪區壓成中灰**，這是「編輯失敗」的定義。

    第二項的減數是 `img_batch`，而 `img_batch` 在訓練迴圈裡就是 masked
    image（不是原圖），故 `x_adv - x_masked` 恰等於 clamp 前後的 `delta`。
    本函式照此簽名，不要傳原圖進來。

    官方兩項都額外除以 512。L1 範數是一階齊次，分子分母的 512 互相消去，
    數學上等價，只是 fp16 的數值縮放；本函式不寫那個 512。

    `alpha` 在官方寫在 L1 範數**內部**，因 `alpha > 0` 與寫在外部等價。

    **分子跨 3 個通道、分母只有 1 個通道。** 官方的 `mask_batch` 形狀是
    `(N,1,H,W)`（`utils.py:37` 的 `mask[None, None]`），而 `img_out` 與
    `img_adv` 是 `(N,3,H,W)`；範數靠 broadcast 加總三個通道，`.sum()` 卻只
    數一個通道。兩項都差同一個常數 3，對最佳化沒有影響，但**回報的數值
    不是「每像素每通道的平均絕對值」，而是它的 3 倍**。本函式照官方保留
    這個常數（`tests/test_diffvax.py` 釘住），不要「修正」——修掉之後
    `alpha = 4` 這個比例仍然成立，但數值與官方的訓練日誌就對不起來了。
    """
    keep = 1.0 - mask_edit
    l_edit = (edited * mask_edit).norm(p=1) / mask_edit.sum()
    l_noise = ((x_adv - x_masked) * keep).norm(p=1) / keep.sum()
    alpha_l_noise = alpha * l_noise
    return {
        "l_edit": l_edit,
        "l_noise": l_noise,
        "alpha_l_noise": alpha_l_noise,
        "total": l_edit + alpha_l_noise,
    }


__all__ = [
    "ATTACK_MODEL_ID",
    "CHECKPOINT_BYTES",
    "CHECKPOINT_NUM_PARAMS",
    "CHECKPOINT_NUM_TENSORS",
    "CHECKPOINT_SHA256",
    "CLAMP_MAX",
    "CLAMP_MIN",
    "DIFFVAX_ALPHA",
    "MASK_THRESHOLD",
    "NB_FILTER",
    "NestedUNet",
    "PAPER_RESOLUTION",
    "SIZE_DIVISOR",
    "TRAIN_SETTINGS",
    "VGGBlock",
    "binarise_mask",
    "diffvax_terms",
    "immunize",
    "load_immunizer",
]
