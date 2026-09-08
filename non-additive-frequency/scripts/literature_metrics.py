"""用文獻這一族的指標重算已存的格。**只讀已存的影像，不佔卡。**

為什麼要有這一支
────────────────────────────────────────────────────────────────────
與本專案威脅模型同型的三篇（FaceLock CVPR 2025、Anti-DreamBooth ICCV 2023、
FaceShield ICCV 2025）用的指標與本專案既有的欄位**大部分重疊但不完全一樣**。
逐項對照見 `docs/reference/SURVEY_IDENTITY_EDITING.md`：

| 文獻指標 | 定義 | 方向 | 本專案 |
|---|---|---|---|
| **FR**（FaceLock） | `cos(E(原圖), E(編輯輸出))` | ↓ | `id_def`（辨識器不同） |
| **CLIP-I** | `cos(E_img(編輯), E_img(原圖))` | ↓ | `edit_clip_sim` |
| **CLIP-S** | `cos(E_img(編輯)−E_img(原圖), E_txt(指令))` | ↓ | **本檔新增** |
| **LPIPS** | 兩張編輯輸出之間 | ↑ | `edit_lpips` |
| **PSNR**／**SSIM** | 同上 | ↓ | `edit_psnr`／`edit_ssim` |
| **FDFR**（Anti-DreamBooth） | 偵測不到臉的比率 | ↑ | `face_found`（偵測器不同） |

六個裡有四個本專案每一格早就算好存在 `results.csv` 裡，缺的是 CLIP-S 與
**辨識器的多樣性**。本檔補上這兩件事，讓同一批影像可以用文獻的指標讀一次。

三個辨識器，為什麼要多於一個
────────────────────────────────────────────────────────────────────
本專案現行只有 facenet 的 InceptionResnetV1（VGGFace2）＋ MTCNN。
單一辨識器的結論可能是那個網路的特性，而不是防禦的性質——尤其本專案已記過
**MTCNN 在人眼看得見的臉上會誤報**（`runs/ip2p_content_constraint/README.md`）。

    facenet   InceptionResnetV1(vggface2) + MTCNN     本專案既有，主欄
    cvlface   AdaFace ViT-Base+KPRPE + DFA 對齊器     FaceLock 用的那一組
    arcface   ArcFace + RetinaFace                    Anti-DreamBooth／FaceShield

**缺相依時該欄留空並在 CSV 的 `<辨識器>_status` 欄寫明原因，不靜默跳過**
——少一個辨識器時表上只會看到那一欄是空的，而不是以為它算過了。

用法：
    python scripts/literature_metrics.py --out runs/x/literature.csv \\
        --entry tint_clothing=runs/ip2p_content_constraint/tint_clothing_task_123
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# **`onnxruntime` 必須在 `torch` 之前載入。** 兩者都連結 CUDA runtime，
# 而它們載入的版本不同；先讓 torch（連同 CVLFace 的 KPRPE CUDA 擴充）把
# 符號綁定完，再 import `onnxruntime` 的 C 擴充會 **segfault**——實測位置在
# `onnxruntime/capi/_pybind_state.py:32`，且在處理任何一格之前就發生，
# 不是資料的問題。順序反過來就不會。
#
# 缺 `insightface` 是正常情形（該辨識器是選配），故這裡吞掉 ImportError；
# **但不吞其他例外**——真正壞掉的 onnxruntime 應該在 `_ArcFace` 那裡被記進
# `arcface_status` 欄，而不是在這裡消失。
try:
    import onnxruntime as _ort  # noqa: F401  只為載入順序，不使用
except ImportError:
    pass

import torch  # noqa: E402  必須在 onnxruntime 之後

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.identity import identity_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

# 三個辨識器的來源。**逐項寫明，不由程式猜**。
RECOGNISERS = {
    "facenet": "facenet-pytorch InceptionResnetV1(vggface2) + MTCNN（本專案既有）",
    "cvlface": "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
               " + minchul/cvlface_DFA_mobilenet（FaceLock）",
    "arcface": "insightface ArcFace + RetinaFace（Anti-DreamBooth／FaceShield）",
}


def _load(p: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def _cell(run: Path):
    """(原圖, 編輯(原圖), 編輯(防禦圖), 條件, 指令)。缺一就拋錯，不猜。"""
    res = run / "results.csv"
    if not res.exists():
        raise SystemExit(f"{run} 沒有 results.csv")
    rows = list(csv.DictReader(res.open(encoding="utf-8")))
    if len(rows) != 1:
        raise SystemExit(f"{run}/results.csv 有 {len(rows)} 列，預期恰好 1 列")
    row = rows[0]
    image, cond = row["image"], row["condition"]
    paths = {k: run / f"{image}__{v}.png" for k, v in (
        ("orig", "orig"), ("edit_orig", f"{cond}__edit_orig"),
        ("edit_def", f"{cond}__edit_def"))}
    for k, p in paths.items():
        if not p.exists():
            raise SystemExit(f"{run} 缺 {p.name}")
    return ({k: _load(p) for k, p in paths.items()},
            image, cond, row["instruction"], row.get("attack_category", ""))


class _CVLFace:
    """FaceLock 用的辨識器。`compute_score` 逐行對照官方 `utils.py`。

    模型走 `trust_remote_code=True`，兩個 repo 都是公開的。載入失敗時
    `available` 為 False，呼叫端把該欄留空並記下原因——**不退回別的辨識器**，
    那會讓 CSV 上同一欄混著兩個模型的數字。
    """

    def __init__(self, device):
        self.device = device
        self.status = ""
        self.aligner = None
        self.model = None
        try:
            self._load()
        except Exception as e:  # noqa: BLE001 — 缺權重／缺網路都要記下原因
            self.status = f"{type(e).__name__}: {e}"[:200]

    def _load(self):
        """檔案抓取逐行對照 FaceLock 的 `utils.py::download`；建模型那一步**繞過
        `transformers` 的 `PreTrainedModel` 殼**，直接呼叫 repo 內的工廠函式。

        為什麼不能用 `AutoModel.from_pretrained(...)`
        ────────────────────────────────────────────────────────────
        兩個 repo 的 `wrapper.py` 有三個相對於 repo 根目錄的依賴：
        `from aligners import get_aligner`／`from models import get_model`
        （兩者都是 **repo 內的目錄**，不是 PyPI 套件）、
        `open('pretrained_model/model.yaml')` 與
        `load_state_dict_from_path('pretrained_model/model.pt')`。
        故必須先把 `files.txt` 列的檔案抓齊，再 `chdir` 進去、把該目錄放進
        `sys.path`，整個建構過程都要在裡面完成。

        繞過 `PreTrainedModel` 的理由：本環境的 `transformers` 是 5.x，
        它要求模型有 `all_tied_weights_keys`，而這兩個 repo 的自訂子類寫於
        4.x 時代、沒有那個屬性，`from_pretrained` 直接 `AttributeError`。
        **降級 transformers 不可行**——CLIP、SigLIP 與 IP2P 全部靠它。
        而那一層殼是可以安全繞過的：兩個 `wrapper.py` 的 `forward` 逐字都是
        `return self.model(*args, **kwargs)`，**恆等轉發**，繞過它不改變任何
        計算，只是少了一個 config 容器。
        """
        import os
        import importlib
        import yaml
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf

        root = Path(os.environ.get("HF_HOME",
                                   str(Path.home() / ".cache/huggingface")))
        # (屬性, repo, repo 內的工廠模組, 工廠函式)
        for attr, repo, mod_name, fn_name in (
                ("aligner", "minchul/cvlface_DFA_mobilenet",
                 "aligners", "get_aligner"),
                ("model", "minchul/cvlface_adaface_vit_base_kprpe_webface4m",
                 "models", "get_model")):
            path = root / repo
            path.mkdir(parents=True, exist_ok=True)
            if not (path / "files.txt").exists():
                hf_hub_download(repo, "files.txt", local_dir=str(path))
            names = (path / "files.txt").read_text().splitlines()
            for f in [n for n in names if n] + ["config.json", "wrapper.py"]:
                if not (path / f).exists():
                    hf_hub_download(repo, f, local_dir=str(path))

            cwd = os.getcwd()
            os.chdir(path)
            sys.path.insert(0, str(path))
            # 兩個 repo 都有叫 `models`／`aligners` 的頂層目錄，名稱會撞。
            # 載完第二個之前必須把第一個從 `sys.modules` 移除，否則
            # `import models` 會拿到前一個 repo 的版本——**不會拋錯**，
            # 只會載到錯的網路。
            stale = [k for k in sys.modules
                     if k == mod_name or k.startswith(mod_name + ".")]
            for k in stale:
                del sys.modules[k]
            try:
                factory = getattr(importlib.import_module(mod_name), fn_name)
                conf = OmegaConf.create(
                    yaml.safe_load(open("pretrained_model/model.yaml")))
                m = factory(conf)
                m.load_state_dict_from_path("pretrained_model/model.pt")
            finally:
                # **一定要還原。** `chdir` 是行程全域狀態，留在 snapshot 目錄
                # 裡的話後續所有相對路徑（`runs/`、`data/`）都會指錯地方，
                # 而且不會拋錯——只會寫到別的地方去。
                os.chdir(cwd)
                sys.path.pop(0)
                for k in [k for k in sys.modules
                          if k == mod_name or k.startswith(mod_name + ".")]:
                    del sys.modules[k]
            setattr(self, attr, m.to(self.device).eval())

    @property
    def available(self) -> bool:
        return self.model is not None and self.aligner is not None

    @torch.no_grad()
    def score(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """兩張圖的身分餘弦。輸入 `[0,1]`，內部轉成官方 `pil_to_input` 的
        `[-1,1]`（`Normalize(mean=0.5, std=0.5)`）。"""
        import inspect
        feats = []
        for x in (a, b):
            inp = (x.to(self.device).float().clamp(0, 1) * 2.0 - 1.0)
            aligned, _, ldmks, _, _, _ = self.aligner(inp)
            # 繞過了 `PreTrainedModel` 那層殼，故 FaceLock 的
            # `fr_model.model.net` 在這裡是 `self.model.net`。
            sig = inspect.signature(self.model.net.forward)
            f = (self.model(aligned, ldmks)
                 if sig.parameters.get("keypoints") is not None
                 else self.model(aligned))
            feats.append(torch.nn.functional.normalize(f, dim=-1))
        return float((feats[0] * feats[1]).sum(-1).mean())


class _ArcFace:
    """Anti-DreamBooth 與 FaceShield 用的那一組：ArcFace 嵌入 ＋ RetinaFace 偵測。

    `insightface` 的 `buffalo_l` 套件同時提供兩者。缺套件時 `available`
    為 False。**RetinaFace 的偵測結果另外報**——Anti-DreamBooth 的 FDFR
    就是它，而本專案既有的 `face_found` 用的是 MTCNN，兩者已知會分歧。
    """

    def __init__(self, device):
        self.status = ""
        self.app = None
        try:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0 if device.type == "cuda" else -1,
                             det_size=(640, 640))
        except Exception as e:  # noqa: BLE001
            self.status = f"{type(e).__name__}: {e}"[:200]

    @property
    def available(self) -> bool:
        return self.app is not None

    def _embed(self, x: torch.Tensor):
        arr = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        faces = self.app.get(arr[:, :, ::-1])  # insightface 吃 BGR
        if not faces:
            return None
        f = max(faces, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
        e = torch.from_numpy(f.normed_embedding.astype(np.float32))
        return e

    def score(self, a: torch.Tensor, b: torch.Tensor):
        """回傳 `(餘弦或 None, a 偵測到臉嗎, b 偵測到臉嗎)`。

        任一側偵測不到時餘弦為 None——**那不是 0**。偵測失敗本身是
        Anti-DreamBooth 的 FDFR 要報的東西，混進餘弦欄會讓兩件事都失真。
        """
        ea, eb = self._embed(a), self._embed(b)
        found = (ea is not None, eb is not None)
        if ea is None or eb is None:
            return None, *found
        return float((ea * eb).sum()), *found


# argv 太長時 `import onnxruntime` 會 **segfault**。實測（同一台機器、同一個
# venv）：把 `--entry` 重複 270 次仍可，400 次必掛，而載入的擴充模組只有
# numpy 與 PIL——torch 都還沒進來，故與載入順序無關。原因是 onnxruntime 的
# `onnxruntime_pybind11_state.so` 在 `ld.so` 載入時對 stack 空間敏感，
# 而 argv 與 environ 就放在 stack 頂端。
#
# **不要靠縮短路徑或調 `ulimit -s` 繞過**——那只是把門檻往後推，下次擴大批次
# 又會踩到，而且症狀是 segfault 不是錯誤訊息。清單改由檔案讀進來，argv 就
# 恆為常數長度。
MAX_ARGV_ENTRIES = 150


def _read_entries(args):
    """由 `--entry`（少量）或 `--entries-file`（大量）取得 (label, 目錄)。"""
    specs = list(args.entry)
    if args.entries_file is not None:
        for line in args.entries_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                specs.append(line)
    if not specs:
        raise SystemExit("要給 --entry 或 --entries-file")
    if len(args.entry) > MAX_ARGV_ENTRIES:
        raise SystemExit(
            f"--entry 給了 {len(args.entry)} 個，超過 {MAX_ARGV_ENTRIES}："
            "argv 過長會讓 import onnxruntime segfault（見本檔說明）。"
            "改用 --entries-file。**這裡明確拒絕而不是讓它去 segfault**")
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"每一筆要寫成 label=目錄，收到 {spec!r}")
        label, path = spec.split("=", 1)
        out.append((label, Path(path)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", default=[],
                    help="label=目錄。可重複。格數多時改用 --entries-file")
    ap.add_argument("--entries-file", type=Path, default=None,
                    help="每行一個 `label=目錄`，`#` 開頭與空行略過。"
                         "**格數超過約兩百時必須用這個而不是 --entry**——"
                         "見下方 `_read_entries` 的說明")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight", action="store_true",
                    help="只檢查目錄與檔案，不載任何權重")
    args = ap.parse_args()

    entries = _read_entries(args)

    if args.preflight:
        for label, run in entries:
            _, image, cond, instruction, cat = _cell(run)
            print(f"  {label:28s} {image:34s} {cond:12s} {cat:11s} {instruction}")
        print(f"[preflight] {len(entries)} 格通過，未載入權重")
        return

    device = torch.device(args.device)
    suite = MetricSuite(device=device)
    cvl, arc = _CVLFace(device), _ArcFace(device)
    for name, obj in (("cvlface", cvl), ("arcface", arc)):
        print(f"[{name}] {'可用' if obj.available else '不可用：' + obj.status}",
              flush=True)

    rows = []
    for label, run in entries:
        imgs, image, cond, instruction, cat = _cell(run)
        x, eo, ed = imgs["orig"], imgs["edit_orig"], imgs["edit_def"]
        r = {"label": label, "image": image, "condition": cond,
             "category": cat, "instruction": instruction}

        # facenet：本專案既有的三個數（id_orig／id_def／face_found）。
        r.update(identity_row(x, eo, ed, device=args.device))

        # cvlface：FaceLock 的 FR。**參照是原圖**，與它的 `eval_facial.py` 相同。
        if cvl.available:
            r["fr_cvlface_orig"] = round(cvl.score(x, eo), 5)
            r["fr_cvlface_def"] = round(cvl.score(x, ed), 5)
            r["cvlface_status"] = "ok"
        else:
            r["fr_cvlface_orig"] = r["fr_cvlface_def"] = ""
            r["cvlface_status"] = cvl.status

        # arcface：Anti-DreamBooth 的 ISM 與 FDFR。
        if arc.available:
            s_o, _, found_eo = arc.score(x, eo)
            s_d, _, found_ed = arc.score(x, ed)
            r["ism_arcface_orig"] = "" if s_o is None else round(s_o, 5)
            r["ism_arcface_def"] = "" if s_d is None else round(s_d, 5)
            r["retinaface_found_edit_orig"] = found_eo
            r["retinaface_found_edit_def"] = found_ed
            r["arcface_status"] = "ok"
        else:
            for k in ("ism_arcface_orig", "ism_arcface_def",
                      "retinaface_found_edit_orig", "retinaface_found_edit_def"):
                r[k] = ""
            r["arcface_status"] = arc.status

        # CLIP-I 與 CLIP-S。前者本專案既有（`edit_clip_sim`），一併重算是為了
        # 讓這張表自足；後者是新增的。
        r["clip_i"] = round(suite.image_similarity(eo, ed)["clip"], 5)
        d_orig = suite.direction_similarity(x, eo, instruction)
        d_def = suite.direction_similarity(x, ed, instruction)
        r["clip_s_orig"] = round(d_orig["clip"], 5)
        r["clip_s_def"] = round(d_def["clip"], 5)
        r["siglip_s_orig"] = round(d_orig["siglip"], 5)
        r["siglip_s_def"] = round(d_def["siglip"], 5)

        # LPIPS／PSNR／SSIM：兩張編輯輸出之間，與 FaceLock 的三個一致。
        r.update({f"edit_{k}": round(v, 5)
                  for k, v in suite.pairwise(eo, ed).items()})
        rows.append(r)
        print(f"  {label:28s} {image:30s} "
              f"id_def={r['id_def']} clip_s_def={r['clip_s_def']}", flush=True)
        write_csv(args.out, rows)

    print(f"[literature_metrics] {len(rows)} 列寫入 {args.out}")
    print("辨識器來源：")
    for k, v in RECOGNISERS.items():
        print(f"  {k:9s} {v}")


if __name__ == "__main__":
    main()
