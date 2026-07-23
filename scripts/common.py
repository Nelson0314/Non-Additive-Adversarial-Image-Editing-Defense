"""stage 腳本共用輔助：config 載入、smoke 縮減、遮罩、路徑處理。

執行方式：於 repo 根目錄 `python scripts/stageX_*.py [args]`。
--smoke 為本地 tiny 模型之流程驗證模式（縮減超參數，非正式設定）。
"""

import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_configs(config_dir: str = "configs") -> dict:
    """回傳 {"base", "additive", "nonadditive", "purify"}；
    若存在 stage0 產出的 nonadditive_calibrated.yaml 則覆蓋合併（淺層逐區塊）。"""
    d = REPO_ROOT / config_dir
    cfgs = {
        "base": load_yaml(d / "base.yaml"),
        "additive": load_yaml(d / "additive.yaml"),
        "nonadditive": load_yaml(d / "nonadditive.yaml"),
        "purify": load_yaml(d / "purify.yaml"),
    }
    calibrated = d / "nonadditive_calibrated.yaml"
    if calibrated.exists():
        for section, overrides in (load_yaml(calibrated) or {}).items():
            if isinstance(overrides, dict):
                cfgs["nonadditive"].setdefault(section, {}).update(overrides)
    return cfgs


def apply_smoke(cfgs: dict, sd) -> None:
    """本地 tiny 模型之縮減設定（in-place）。解析度依模型原生尺寸推得。"""
    scale = 2 ** (len(sd.vae.config.block_out_channels) - 1)
    n = sd.unet.config.sample_size * scale
    cfgs["base"]["generation"].update({"height": n, "width": n, "num_inference_steps": 4})
    cfgs["base"]["edit"].update({"n_seeds": 2, "sdedit_strength": 0.5})
    cfgs["base"]["data"]["n_images"] = 2
    cfgs["additive"]["photoguard"].update({"n_iter": 3, "diffusion_T": 3, "grad_reps": 2})
    cfgs["nonadditive"]["advdiff"].update({"T": 5, "N": 2, "guidance_range": [0.0, 0.5]})
    cfgs["nonadditive"]["apa"].update(
        {"grid_steps": 10, "T_a": 3, "N": 2, "lora_steps": 3, "lora_rank": 2, "lora_alpha": 2}
    )
    cfgs["nonadditive"]["apa"]["gc"].update({"inversion_steps": 3, "sampling_steps": 4})
    # gridpure 之 grid 尺寸依影像縮放（正式 256/128 針對 512×512）
    cfgs["purify"]["gridpure"].update({"grid_size": n // 2, "stride": n // 4})
    for s in cfgs["purify"]["gridpure"]["settings"]:
        s.update({"pure_steps": 2, "iterations": 2} if s["name"] == "iterative"
                 else {"pure_steps": 4, "iterations": 1})
    cfgs["purify"]["gridpure"]["pure_steps_scan"] = [1, 2]
    cfgs["purify"]["jpeg"]["quality"] = [80, 50]
    cfgs["purify"]["blur"]["sigma"] = [1.0]
    cfgs["purify"]["crop_resize"]["ratio"] = [0.2]


def center_mask(h: int, w: int, fraction: float = 0.5) -> torch.Tensor:
    """placeholder 遮罩：中央方形（1=重繪），邊長 = fraction × 影像邊長。
    規格由 config edit.inpaint_mask 提供並記錄於輸出；遮罩幾何實質影響保護
    難度，placeholder 之結果與真實資料集結果不可直接比較（preflight 假設 3）。"""
    mask = torch.zeros(1, 1, h, w)
    mh, mw = round(h * fraction), round(w * fraction)
    top, left = (h - mh) // 2, (w - mw) // 2
    mask[..., top : top + mh, left : left + mw] = 1.0
    return mask


def sample_mask(sample: dict, config: dict) -> torch.Tensor:
    """真實遮罩（sample["mask"]，DAYN 資料到手後由 dataset 提供）優先；
    否則依 config edit.inpaint_mask 產生 placeholder。"""
    if "mask" in sample:
        return sample["mask"]
    spec = config["edit"].get("inpaint_mask") or {"type": "center_square", "fraction": 0.5}
    if spec["type"] != "center_square":
        raise ValueError(f"未知 placeholder 遮罩類型: {spec['type']}")
    h, w = config["generation"]["height"], config["generation"]["width"]
    return center_mask(h, w, spec.get("fraction", 0.5))


def used_placeholder_mask(samples: list[dict], edit_methods: list[str]) -> bool:
    return "inpaint" in edit_methods and any("mask" not in s for s in samples)


def model_tag(name: str) -> str:
    """模型名稱 → 檔名安全標籤。"""
    return name.replace("/", "__")


class IncrementalCsv:
    """逐筆寫入之 results.csv（斷點續跑用）。

    每列 append 後立即 flush；欄位以首列之鍵為準（後續列缺鍵留空、多鍵報錯）。
    既有檔案（--resume）讀回已完成列，done() 供跳過判斷。
    """

    def __init__(self, path, key_fields: list[str]):
        import csv

        self._csv = csv
        self.path = Path(path)
        self.key_fields = key_fields
        self.fieldnames = None
        self._done = set()
        self.rows = []
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.fieldnames = reader.fieldnames
                for row in reader:
                    self.rows.append(row)
                    self._done.add(self._key(row))

    def _key(self, row: dict) -> tuple:
        return tuple(str(row[k]) for k in self.key_fields)

    def done(self, row_key: dict) -> bool:
        return self._key(row_key) in self._done

    def append(self, row: dict) -> None:
        new_file = self.fieldnames is None
        if new_file:
            self.fieldnames = list(row)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = self._csv.DictWriter(f, fieldnames=self.fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        self.rows.append(row)
        self._done.add(self._key(row))
