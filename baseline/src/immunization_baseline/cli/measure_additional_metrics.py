"""把外部文獻用過、但本專案尚未記錄的指標補齊。

用途
────────────────────────────────────────────────────────────────────
主表的欄位是本專案自己定的標準清單（`immunization_core/metrics/standard.py`）：
LPIPS／SSIM／PSNR／VIFp／DISTS。但十一個外部方法的原論文各自報的指標不同，
其中 **FSIM** 出現在五篇（DAYN §4.3、DANP §V-B、SIFM §VII-A、TDAE、
DiffVax 的 `evaluate.py`），本專案的 `MetricSuite` 支援它、卻沒有寫進任何
一張結果表。**CIEDE2000** 是本專案自己的預算單位，同樣只有顏色那一列有值。
**VMAF** 是影像／視訊壓縮文獻常用的融合指標（Netflix 開發，`libvmaf`）。

這一支只讀既有的 PNG，不重跑攻擊、不重跑編輯、不碰 GPU（FSIM、ΔE00、VMAF
都是 CPU 即可算的傳統或古典融合指標）。輸出獨立成檔，不覆寫任何既有 CSV。

四個輸出
────────────────────────────────────────────────────────────────────
| 檔 | 對象 | 補的欄位 |
|---|---|---|
| `additional_metrics/fidelity.csv`   | 防禦圖 vs 原圖 | `fid_fsim`、`fid_delta_e00`、`fid_mse` |
| `additional_metrics/displacement.csv` | 編輯(原圖) vs 編輯(防禦圖) | `disp_fsim`、`disp_mse` |
| `additional_metrics/retention.csv`  | 淨化後的同一對 | `disp_purified_fsim` |
| `additional_metrics/aesthetic.csv`  | 防禦圖本身（無參考） | 美學與自然度四項 |
| `additional_metrics/vmaf.csv`       | 上面前三種配對，各自的 VMAF | `vmaf`，`pairing` 欄標配對種類 |

必要與選配指標
────────────────────────────────────────────────────────────────────
| stage | 必要（缺依賴或計算失敗即中止） | 選配（後端缺席時留空並記原因） |
|---|---|---|
| fidelity | `fid_fsim`、`fid_delta_e00`、`fid_mse` | — |
| displacement | `disp_fsim`、`disp_mse` | — |
| retention | `disp_purified_fsim` | — |
| aesthetic | — | `aes_*` 七項（pyiqa 模型與權重） |
| vmaf | — | `vmaf`（需含 libvmaf 的 ffmpeg） |

選配指標只在建立後端時判定可用性；後端缺席時該欄留空，並在 `unavailable_metrics` 欄以
`<欄名>: <原因>` 記錄（多項以 `; ` 分隔，全部可用時為空字串）。建立成功後的計算錯誤一律中止。
每列另記 `device`：本工具固定以 CPU 計算。

**美學那一組是無參考指標**：它們只看一張圖，不跟原圖比，所以原圖自己也要量一份
當參照列——`niqe 4.12` 這個數單獨擺出來沒有意義，要對著同一張原圖的值看。

`mse` 由 rms 反推不可靠（rms 的平均方式見 `immunization_core/metrics/suite.py`），故直接算。

VMAF 的餵法：單張圖各自視為一支 1 幀的「影片」直接餵給 `libvmaf`
（reference=原圖或未防禦編輯，distorted=防禦圖或防禦後編輯）。VMAF 原生設計
給有真實運動的視訊用，其中一個時序特徵（`integer_motion`）在有連續幀時量
畫面運動。曾在 5 對影像上比較過「單幀」與「把同一張圖複製成 5 幀的靜止序列」
兩種餵法：兩者 VMAF 分數逐位元相同，`integer_motion`／`integer_motion2`
在兩種餵法、每一幀都是 0.0（libvmaf 對沒有前一幀的第一幀本就給 0，而複製出
的後續幀彼此相同，運動量自然也是 0）。因此對「靜態影像比一張靜態影像」這種
配對，兩種餵法沒有差異，這裡採單幀，不用另外拼接影片。**VMAF 的訓練失真是
壓縮與縮放，這裡比的是免疫擾動與生成式編輯，不在它的訓練分佈內**——這點只
記錄，其餘不下判定。

輸入與輸出位置
────────────────────────────────────────────────────────────────────
| 參數 | 預設 | 使用的 stage |
|---|---|---|
| `--defenses-root` | `artifacts/defenses` | fidelity、aesthetic、vmaf |
| `--displacement-csv` | `results/displacement.csv` | displacement、vmaf |
| `--retention-csv` | `results/retention.csv` | retention、vmaf |
| `--purified-edits-root` | `artifacts/purified_edits` | retention、vmaf |
| `--path-root` | baseline 專案根 | displacement、vmaf（`displacement.csv` 相對影像路徑的基準） |
| `--output-dir` | `results/additional_metrics` | 全部 |

預設值即 `immunization_baseline.layout` 的對應常數。其他專案量測自己的產物時，
以上述參數指向自己的目錄，並以 `--output-dir` 指定獨立的輸出目錄，不寫入 baseline 的表。

用法
    python -m immunization_baseline.cli.measure_additional_metrics --stage fidelity
    python -m immunization_baseline.cli.measure_additional_metrics --stage displacement
    python -m immunization_baseline.cli.measure_additional_metrics --stage retention
    python -m immunization_baseline.cli.measure_additional_metrics --stage vmaf
    python -m immunization_baseline.cli.measure_additional_metrics --stage fidelity \\
        --defenses-root ../color/artifacts/defenses \\
        --output-dir ../color/results/additional_metrics
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from immunization_baseline import layout  # noqa: E402
from immunization_core.io import write_rows_atomic  # noqa: E402

#: `numpy`／`piq`／`torch` 只有 fidelity／displacement／retention／aesthetic 這四個
#: stage 要用，`vmaf` 不碰張量、只呼叫 `ffmpeg`。四個重依賴延到 `main()` 裡依
#: stage 決定要不要載入，讓 `--stage vmaf` 能在沒裝 torch 的機器（例如本機）上跑。
RESOLUTION = 512

#: 未防禦對照的條件名；淨化目錄版面與 `pipelines.retention.cell()` 相同。
UNDEFENDED = "undefended"


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_png(raw: str, path_root: Path) -> Path:
    """CSV 記的影像路徑；相對路徑以 `path_root`（預設 baseline 專案根）為基準，絕對路徑原樣使用。"""
    p = Path(raw)
    if p.is_absolute():
        return p
    return (path_root / p).resolve()


def write_csv(path: Path, rows: list) -> None:
    """欄位取全部列的聯集、依首次出現排序——`vmaf` stage 的三種配對欄位不同
    （fidelity 沒有 scenario／prompt_index，retention 多一個 purifier），
    只取 `rows[0]` 的鍵在其他 stage 上不出錯，是因為那些 stage 本來就每列
    同構，換到 `vmaf` 才會露出來。"""
    fields: list = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_rows_atomic(path, fields, rows, restval="")


def load(path, device):
    return load_image_tensor(str(path), device, size=RESOLUTION)


def fsim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(piq.fsim(a, b, data_range=1.0))


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean())


def delta_e00(a: torch.Tensor, b: torch.Tensor) -> float:
    from skimage.color import deltaE_ciede2000, rgb2lab
    x = a[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    y = b[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    return float(deltaE_ciede2000(rgb2lab(x), rgb2lab(y)).mean())


def optional_backends(factories: dict) -> tuple:
    """建立選配指標的後端；回傳 (可用的後端, 欄名 → 缺席原因)。"""
    available, missing = {}, {}
    for column, factory in factories.items():
        try:
            available[column] = factory()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            missing[column] = f"{type(error).__name__}: {error}".replace("\n", " ")
    return available, missing


def unavailable_text(missing: dict) -> str:
    return "; ".join(f"{column}: {reason}" for column, reason in missing.items())


def stage_fidelity(device, defenses: Path, out_dir: Path) -> None:
    """防禦圖對原圖。檔名式樣 `<name>__orig.png` 與 `<name>__<cond>__def.png`。"""
    root = defenses
    rows = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        condition = directory.name
        for defended in sorted(directory.glob(f"*__{condition}__def.png")):
            name = defended.name.split("__")[0]
            original = directory / f"{name}__orig.png"
            if not original.is_file():
                raise SystemExit(f"缺原圖：{original}。保真那一欄的分母缺一張就不可比")
            a, b = load(original, device), load(defended, device)
            with torch.no_grad():
                rows.append({
                    "condition": condition, "image": name,
                    "fid_fsim": round(fsim(a, b), 5),
                    "fid_delta_e00": round(delta_e00(a, b), 4),
                    "fid_mse": round(mse(a, b), 8),
                    "original_png": original.as_posix(),
                    "defended_png": defended.as_posix(),
                    "device": str(device),
                })
        print(f"[DONE] {condition:20s} 累計 {len(rows)} 列", flush=True)
    write_csv(out_dir / "fidelity.csv", rows)
    print(f"[ALLDONE] {out_dir / 'fidelity.csv'}（{len(rows)} 列）", flush=True)


def stage_displacement(device, displacement_csv: Path, path_root: Path, out_dir: Path) -> None:
    """編輯(原圖) vs 編輯(防禦圖)。路徑直接取 `displacement.csv` 自己記的兩欄。"""
    source = read_csv(displacement_csv)
    rows = []
    for i, row in enumerate(source, 1):
        a = load(resolve_png(row["undefended_png"], path_root), device)
        b = load(resolve_png(row["defended_png"], path_root), device)
        with torch.no_grad():
            rows.append({
                "condition": row["condition"], "scenario": row["scenario"],
                "image": row["image"], "prompt_index": row["prompt_index"],
                "disp_fsim": round(fsim(a, b), 5),
                "disp_mse": round(mse(a, b), 8),
                "device": str(device),
            })
        if i % 64 == 0:
            print(f"[{i}/{len(source)}]", flush=True)
    write_csv(out_dir / "displacement.csv", rows)
    print(f"[ALLDONE] {out_dir / 'displacement.csv'}（{len(rows)} 列）", flush=True)


def cell(root: Path, condition: str, purifier: str, scenario: str,
         name: str, index: str) -> Path:
    arm = f"{scenario}_{condition}_{purifier}"
    return root / condition / purifier / arm / f"{name}__p{index}.png"


def stage_retention(device, retention_csv: Path, purified_edits: Path, out_dir: Path) -> None:
    """淨化後的同一對。條件與算子的組合直接照 `retention.csv` 的列。"""
    root = purified_edits
    source = read_csv(retention_csv)
    rows = []
    for i, row in enumerate(source, 1):
        a = cell(root, UNDEFENDED, row["purifier"], row["scenario"],
                 row["image"], row["prompt_index"])
        b = cell(root, row["condition"], row["purifier"], row["scenario"],
                 row["image"], row["prompt_index"])
        if not a.is_file() or not b.is_file():
            raise SystemExit(f"缺檔：{a if not a.is_file() else b}")
        with torch.no_grad():
            rows.append({
                "condition": row["condition"], "purifier": row["purifier"],
                "scenario": row["scenario"], "image": row["image"],
                "prompt_index": row["prompt_index"],
                "disp_purified_fsim": round(fsim(load(a, device), load(b, device)), 5),
                "device": str(device),
            })
        if i % 256 == 0:
            print(f"[{i}/{len(source)}]", flush=True)
    write_csv(out_dir / "retention.csv", rows)
    print(f"[ALLDONE] {out_dir / 'retention.csv'}（{len(rows)} 列）", flush=True)


#: (欄名, pyiqa 模型, 越小越好)。七項都是**無參考**：只看一張圖，不跟原圖比。
#: 前四項訓練自人類的美感評分，後三項量的是「偏離自然影像統計多遠」。
AESTHETIC = (
    ("aes_laion", "laion_aes", False),
    ("aes_nima", "nima", False),
    ("aes_musiq_ava", "musiq-ava", False),
    ("aes_topiq_iaa", "topiq_iaa", False),
    ("aes_clipiqa", "clipiqa", False),
    ("aes_niqe", "niqe", True),
    ("aes_brisque", "brisque", True),
)


def stage_aesthetic(device, defenses: Path, out_dir: Path) -> None:
    """防禦圖與原圖各量一份無參考的美學／自然度讀數。"""
    def factory(name):
        def create():
            import pyiqa
            return pyiqa.create_metric(name, device=device)
        return create

    models, missing = optional_backends({column: factory(name) for column, name, _ in AESTHETIC})
    for column, reason in missing.items():
        print(f"[UNAVAILABLE] {column}：{reason}", flush=True)

    root = defenses
    targets = []
    seen = set()
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        condition = directory.name
        for defended in sorted(directory.glob(f"*__{condition}__def.png")):
            name = defended.name.split("__")[0]
            targets.append((condition, name, defended))
            original = directory / f"{name}__orig.png"
            if name not in seen and original.is_file():
                seen.add(name)
                targets.append(("original", name, original))

    rows = []
    for i, (condition, name, path) in enumerate(targets, 1):
        x = load(path, device).clamp(0, 1)
        row = {"condition": condition, "image": name, "png": path.as_posix()}
        with torch.no_grad():
            for column, _, _ in AESTHETIC:
                row[column] = round(float(models[column](x)), 5) if column in models else ""
        row["device"] = str(device)
        row["unavailable_metrics"] = unavailable_text(missing)
        rows.append(row)
        if i % 16 == 0:
            print(f"[{i}/{len(targets)}]", flush=True)
    write_csv(out_dir / "aesthetic.csv", rows)
    print(f"[ALLDONE] {out_dir / 'aesthetic.csv'}（{len(rows)} 列）",
          flush=True)


def vmaf_score(distorted: Path, reference: Path, tmp_dir: Path, tag: str) -> float:
    """對一對 512x512 PNG 算 VMAF，各自視為一支單幀影片。

    `log_path` 用相對於 `cwd` 的檔名，不用帶槽的絕對路徑——`libvmaf` 的選項字串
    以冒號分隔，Windows 路徑的磁碟機冒號會被誤判成選項分隔符。
    """
    import json
    import subprocess

    log_name = f"vmaf_{tag}.json"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(distorted), "-i", str(reference),
         "-lavfi", f"libvmaf=log_fmt=json:log_path={log_name}",
         "-f", "null", "-"],
        cwd=tmp_dir, check=True,
    )
    log_path = tmp_dir / log_name
    with log_path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    log_path.unlink()
    return float(data["frames"][0]["metrics"]["vmaf"])


def libvmaf_available() -> bool:
    """ffmpeg 存在且含 libvmaf 濾鏡；否則拋錯，由 `optional_backends` 記為缺席原因。"""
    import subprocess

    listing = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
                             text=True, check=True).stdout
    if " libvmaf " not in listing:
        raise RuntimeError("ffmpeg 未包含 libvmaf 濾鏡")
    return True


def stage_vmaf(device, defenses: Path, displacement_csv: Path, retention_csv: Path,  # noqa: ARG001
               purified_edits: Path, path_root: Path, out_dir: Path) -> None:
    """對保真、位移、淨化後三種既有配對各補一欄 VMAF，寫成單一檔，`pairing` 欄區分。

    `device` 與其他 stage 的簽名一致；VMAF 不用 GPU 與 torch。"""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pairs = []  # (pairing, meta_dict, reference_path, distorted_path)

    root = defenses
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        condition = directory.name
        for defended in sorted(directory.glob(f"*__{condition}__def.png")):
            name = defended.name.split("__")[0]
            original = directory / f"{name}__orig.png"
            if not original.is_file():
                raise SystemExit(f"缺原圖：{original}")
            pairs.append(("fidelity",
                          {"condition": condition, "image": name},
                          original, defended))

    disp_source = read_csv(displacement_csv)
    for row in disp_source:
        pairs.append(("displacement",
                      {"condition": row["condition"], "scenario": row["scenario"],
                       "image": row["image"], "prompt_index": row["prompt_index"]},
                      resolve_png(row["undefended_png"], path_root),
                      resolve_png(row["defended_png"], path_root)))

    ret_root = purified_edits
    ret_source = read_csv(retention_csv)
    for row in ret_source:
        a = cell(ret_root, UNDEFENDED, row["purifier"], row["scenario"],
                 row["image"], row["prompt_index"])
        b = cell(ret_root, row["condition"], row["purifier"], row["scenario"],
                 row["image"], row["prompt_index"])
        if not a.is_file() or not b.is_file():
            raise SystemExit(f"缺檔：{a if not a.is_file() else b}")
        pairs.append(("retention",
                      {"condition": row["condition"], "purifier": row["purifier"],
                       "scenario": row["scenario"], "image": row["image"],
                       "prompt_index": row["prompt_index"]},
                      a, b))

    print(f"[PAIRS] fidelity={sum(1 for p in pairs if p[0] == 'fidelity')} "
          f"displacement={sum(1 for p in pairs if p[0] == 'displacement')} "
          f"retention={sum(1 for p in pairs if p[0] == 'retention')} "
          f"total={len(pairs)}", flush=True)

    _, missing = optional_backends({"vmaf": libvmaf_available})
    if missing:
        print(f"[UNAVAILABLE] vmaf：{missing['vmaf']}", flush=True)
        rows = [{"pairing": pairing, **meta, "vmaf": "", "device": "cpu",
                 "unavailable_metrics": unavailable_text(missing)}
                for pairing, meta, _, _ in pairs]
        write_csv(out_dir / "vmaf.csv", rows)
        print(f"[ALLDONE] {out_dir / 'vmaf.csv'}（{len(rows)} 列，vmaf 不可用）", flush=True)
        return

    rows = [None] * len(pairs)
    with tempfile.TemporaryDirectory(prefix="vmaf_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        def work(i):
            pairing, meta, reference, distorted = pairs[i]
            score = vmaf_score(distorted, reference, tmp_dir, tag=str(i))
            row = {"pairing": pairing, **meta, "vmaf": round(score, 5), "device": "cpu",
                   "unavailable_metrics": ""}
            return i, row

        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(work, i) for i in range(len(pairs))]
            for future in as_completed(futures):
                i, row = future.result()
                rows[i] = row
                done += 1
                if done % 256 == 0:
                    print(f"[{done}/{len(pairs)}]", flush=True)

    write_csv(out_dir / "vmaf.csv", rows)
    print(f"[ALLDONE] {out_dir / 'vmaf.csv'}（{len(rows)} 列）", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """路徑參數的預設值即 `layout` 常數；不給參數時讀寫位置與現行 baseline 表相同。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True,
                        choices=("fidelity", "displacement", "retention", "aesthetic", "vmaf"))
    parser.add_argument("--defenses-root", dest="defenses", type=Path, default=layout.DEFENSES,
                        help="條件防禦圖的根目錄，`<條件>/<圖>__orig.png` 與 `<條件>/<圖>__<條件>__def.png`"
                             "（fidelity、aesthetic、vmaf）")
    parser.add_argument("--displacement-csv", dest="displacement", type=Path,
                        default=layout.RESULTS / "displacement.csv",
                        help="位移表，取其 `undefended_png`／`defended_png` 兩欄（displacement、vmaf）")
    parser.add_argument("--retention-csv", dest="retention", type=Path,
                        default=layout.RESULTS / "retention.csv",
                        help="淨化後保留表，取其條件、算子、場景、影像、指令列（retention、vmaf）")
    parser.add_argument("--purified-edits-root", dest="purified_edits", type=Path,
                        default=layout.PURIFIED_EDITS,
                        help="淨化後編輯的根目錄，`<條件>/<算子>/<場景>_<條件>_<算子>/`（retention、vmaf）")
    parser.add_argument("--path-root", type=Path, default=layout.PROJECT,
                        help="位移表中相對影像路徑的基準目錄（displacement、vmaf）")
    parser.add_argument("--output-dir", dest="out", type=Path,
                        default=layout.RESULTS / "additional_metrics",
                        help="輸出 CSV 的目錄；量測其他專案的產物時須指定獨立目錄")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "vmaf":
        device = None
    else:
        global np, piq, torch, load_image_tensor  # noqa: PLW0603
        import numpy as np
        import piq
        import torch
        from immunization_core.io import load_image_tensor
        device = torch.device("cpu")
    stages = {
        "fidelity": lambda: stage_fidelity(device, args.defenses, args.out),
        "displacement": lambda: stage_displacement(device, args.displacement, args.path_root,
                                                   args.out),
        "retention": lambda: stage_retention(device, args.retention, args.purified_edits,
                                             args.out),
        "aesthetic": lambda: stage_aesthetic(device, args.defenses, args.out),
        "vmaf": lambda: stage_vmaf(device, args.defenses, args.displacement, args.retention,
                                   args.purified_edits, args.path_root, args.out),
    }
    stages[args.stage]()


if __name__ == "__main__":
    main()
