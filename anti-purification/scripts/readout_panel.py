"""把已存的防禦圖與編輯圖補齊成四類讀數。**只讀檔，不重跑任何攻擊或編輯。**

為什麼要這一支
────────────────────────────────────────────────────────────────────
`scripts/baseline_run.py` 的輸出早於本專案現行的判準，它的 CSV 裡沒有身分欄，
自然度也只有 `nima`／`cnniqa` 兩欄。外部比較要看的是四類，缺三類。

重跑攻擊來補欄位是錯的：`photoguard_c` 一張圖要 6562 秒，而它的產物已經在
磁碟上。這一支走 `phase_retention.py` 的同一條路——**吃已存的 PNG，補讀數**。

四類讀數
────────────────────────────────────────────────────────────────────
    位移    LPIPS DISTS PSNR SSIM VIFp ΔE00 rms L∞
    語意    CLIP SigLIP CLIP-I
    身分    FaceNet（`identity.py`）與 ArcFace（`arcface.py`）各一套，
            外加 RetinaFace 的偵測數與偵測分數
    自然度  NIQE BRISQUE CLIP-IQA MUSIQ TOPIQ-NR

**自然度的五欄只作解釋，不當判準。** 本專案的最終自然度判定是使用者看圖。
SCOOTER（arXiv:2507.07776）在六個無約束對抗攻擊上量到客觀指標的排序與人眼
幾乎相反，而 NR-IQA 指標本身可被對抗攻擊（arXiv:2310.06958）——這是把自然度
門檻放進損失會被鑽過的機制原因。照報是為了讓表上有可對照的數，不是為了判定。

兩個半邊，方向相反
────────────────────────────────────────────────────────────────────
    fid_*    防禦圖 對 原圖          失真，越小越好
    edit_*   防禦後編輯 對 未防禦編輯  防禦效果，越大越好

自然度兩邊都量：`nat_orig_*` 是原圖自己的分數，`nat_def_*` 是防禦圖的。
**`nat_orig_*` 不可省**——單張影像的分數逐圖差很多，不扣掉基準就分不出
「防禦讓它變難看」與「這張照片本來分數就低」。

檔名慣例（`baseline_run.py` 產生）
────────────────────────────────────────────────────────────────────
    {image}__orig.png
    {image}__{condition}__def.png
    {image}__{condition}__edit_orig.png
    {image}__{condition}__edit_def.png

用法
    python scripts/readout_panel.py --runs runs/baseline_restore/shard1 \
        runs/baseline_restore/shard2 --out runs/baseline_restore/panel.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.defense.assets import load_image  # noqa: E402

NR_METRICS = ["niqe", "brisque", "clipiqa", "musiq", "topiq_nr"]


def _nr_suite(device):
    """pyiqa 的無參考指標。逐個建立，缺哪個就跳過並在標準輸出寫明——
    **不靜默略過**，缺欄位與讀數為零在 CSV 上長得一樣。"""
    import pyiqa

    out = {}
    for name in NR_METRICS:
        try:
            out[name] = pyiqa.create_metric(name, device=device)
        except Exception as e:  # noqa: BLE001
            print(f"[readout_panel] 無參考指標 {name} 建立失敗，本批不報此欄："
                  f"{type(e).__name__}: {e}")
    return out


def _nr_row(suite, x01, prefix: str) -> dict:
    row = {}
    for name, m in suite.items():
        with torch.no_grad():
            row[f"{prefix}_{name}"] = round(float(m(x01)), 5)
    return row


def _orig_index(data_root):
    """`{影像名: 路徑}`。資料集的版面是每類一子目錄，影像名即檔名去副檔名。"""
    if data_root is None:
        return {}
    return {p.stem: p for p in sorted(data_root.glob("*/*.png"))
            if p.parent.name not in ("headmasks", "masks")}


def _pairs(run_dir: Path):
    """要量的格，由 `__edit_def.png` 反推。

    支援兩種檔名。`baseline_run.py` 一格只有一組編輯：

        {image}__{condition}__edit_def.png

    `reedit_ip2p.py` 的一格是 (條件 × 指令 × 種子)：

        {image}__{condition}__{instruction}__s{seed}__edit_def.png

    兩者都由 `__` 切開後取第一段當影像名、最後接 `__edit_def` 的前綴當格名。
    防禦圖一律是 `{image}__{condition}__def.png`（同一條件的所有指令共用一張），
    所以格名要再拆出 condition。
    """
    out = []
    for p in sorted(run_dir.glob("*__edit_def.png")):
        tag = p.name[: -len("__edit_def.png")]
        image, _, rest = tag.partition("__")
        if not rest:
            continue
        condition = rest.split("__")[0]
        out.append((image, condition, tag))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None,
                    help="批次目錄裡沒有 {image}__orig.png 時（`dct_shield_run.py` "
                         "就不寫），從這個資料集根目錄的每類子目錄找同名 PNG。"
                         "找不到仍然中止，不靜默跳過。")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arcface-device", default="cpu",
                    help="insightface 走 onnxruntime，與 torch 分開指定；"
                         "CPU 已足夠快，預設不佔用顯示記憶體")
    ap.add_argument("--arcface", choices=["require", "skip"], default="require",
                    help="require（預設）：insightface 缺席就中止，因為 ArcFace "
                         "與偵測失敗率是外部比較要報的欄位，靜默少兩欄會讓表看"
                         "起來完整但不是。skip：明確放棄那幾欄，只在沒有裝 "
                         "insightface 的機器上做管線測試時用")
    args = ap.parse_args()

    from src.metrics import identity
    from src.metrics.suite import MetricSuite

    arcface = None
    if args.arcface == "require":
        from src.metrics import arcface  # noqa: F811
    else:
        print("[readout_panel] --arcface skip：本批不含 arc_* 與 RetinaFace 欄位")

    device = torch.device(args.device)
    suite = MetricSuite(device=device)
    nr = _nr_suite(device)
    print(f"[readout_panel] 無參考指標：{sorted(nr)}")

    orig_index = _orig_index(args.data)
    rows = []
    for run_dir in args.runs:
        for image, condition, tag in _pairs(run_dir):
            f = {"def": run_dir / f"{image}__{condition}__def.png",
                 "edit_orig": run_dir / f"{tag}__edit_orig.png",
                 "edit_def": run_dir / f"{tag}__edit_def.png"}
            f["orig"] = run_dir / f"{image}__orig.png"
            if not f["orig"].exists() and image in orig_index:
                f["orig"] = orig_index[image]
            missing = [k for k, p in f.items() if not p.exists()]
            if missing:
                raise SystemExit(
                    f"{tag} 缺 {missing}。批次目錄沒有 "
                    f"{{image}}__orig.png 時要給 --data 指向資料集根目錄；"
                    f"少一張圖就少一列，不可以靜默跳過。")
            x = {k: load_image(p, device) for k, p in f.items()}

            row = {"run": run_dir.name, "image": image, "condition": condition,
                   "cell": tag}
            # `pairwise` 已含 psnr/linf/ssim/lpips/dists/vif_p/fsim/rms/
            # deltaE00/frac_gt_16_255/acutance_ratio，不另行重算。
            for k, v in suite.pairwise(x["orig"], x["def"]).items():
                row[f"fid_{k}"] = v
            for k, v in suite.pairwise(x["edit_orig"], x["edit_def"]).items():
                row[f"edit_{k}"] = v
            # 影像—影像的 CLIP／SigLIP 餘弦。與 baseline_run 的 edit_clip_*
            # 不是同一個量：那邊是影像—文字的語意對齊，這邊問「還是不是同一
            # 張照片」。兩個都要，故分開命名。
            for k, v in suite.image_similarity(x["orig"], x["def"]).items():
                row[f"fid_imgsim_{k}"] = round(float(v), 5)
            for k, v in suite.image_similarity(x["edit_orig"], x["edit_def"]).items():
                row[f"edit_imgsim_{k}"] = round(float(v), 5)

            row.update(identity.identity_row(x["orig"], x["edit_orig"], x["edit_def"]))
            if arcface is not None:
                row.update(arcface.arcface_row(x["orig"], x["edit_orig"],
                                               x["edit_def"],
                                               device=args.arcface_device))
            row.update(_nr_row(nr, x["orig"], "nat_orig"))
            row.update(_nr_row(nr, x["def"], "nat_def"))
            rows.append(row)
            print(f"[readout_panel] {tag} 完成")

    if not rows:
        raise SystemExit("沒有任何可讀的 (image, condition) 組合")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[readout_panel] 寫出 {len(rows)} 列到 {args.out}")


if __name__ == "__main__":
    main()
