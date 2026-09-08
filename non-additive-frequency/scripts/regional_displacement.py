"""把已存批次的位移拆成「主體內」與「主體外」。**不重跑編輯，只讀已存的圖。**

問的是什麼
────────────────────────────────────────────────────────────────────
現行的效果讀數是**全圖** `LPIPS(編輯(原圖), 編輯(防禦圖))`。這個量對色彩族
與（將來的）補丁族會被灌水：防禦端改掉的顏色或貼上的內容原封不動穿過編輯，
被 LPIPS 算成「編輯被推開了」，而**受保護的主體可能完全沒有被保住**。

`runs/ip2p_color_masked/` 的遮罩 r=0.20 是這個疑慮的具體實例：它的
`image_guidance` 固定評估只由 0.158 降到 0.064（機制幾乎沒被碰到），
全圖位移卻有 0.415，並據此報出目前最好的等失真倍率 1.192。

分區之後可以直接問：**主體那一塊的編輯有沒有被改變。**

三個欄位分別是什麼
────────────────────────────────────────────────────────────────────
    lpips_full        全圖，與既有的 `edit_lpips` 同一個量（可交叉核對）
    lpips_subject     只在主體遮罩內
    lpips_background  只在主體遮罩外

同時把**失真**也拆一次（`def` 對 `orig`）：`dist_subject` 是防禦端有沒有真的
沒動主體的直接檢查。`--subject-mask` 的批次應該接近 0（羽化帶會有殘量），
無遮罩的批次不會。

遮罩與量測必須與訓練端同源
────────────────────────────────────────────────────────────────────
- 遮罩走 `src/defense/subject_mask.py`，文字取自 `data/decoy_catalogue.yaml`
  的 `objects`，三個形狀參數的預設值與 `scripts/ip2p_run.py` 相同並逐列寫進
  CSV。**遮罩不同就不是同一條軸**，這三個值不可只活在預設值裡。
- LPIPS 走 `MetricSuite.lpips_module` 的同一份權重，經
  `src/metrics/regional.py` 的空間加權平均。遮罩全為 1 時逐位元退回
  `piq.LPIPS()`，由 `tests/test_regional_lpips.py` 釘住。

用法：

    python scripts/regional_displacement.py --out runs/regional/results.csv
        --images task_attr_mod_color_11699 task_attr_mod_color_6205
        --entry color_masked_r020=runs/ip2p_color_masked/m_r020_plant
        --entry phase_ig=runs/ip2p_ig_loss/ig_eot
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", required=True,
                    metavar="LABEL=DIR",
                    help="一個已存批次的輸出目錄")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--catalogue", type=Path,
                    default=Path("data/decoy_catalogue.yaml"),
                    help="主體名詞的來源。威脅模型的前提是防護對象已知")
    ap.add_argument("--out", type=Path, required=True)
    # 三個形狀參數的預設值與 scripts/ip2p_run.py 相同。改了就不是同一條軸，
    # 故逐列寫進 CSV 而不是只活在這裡。
    ap.add_argument("--subject-mask-threshold", type=float, default=0.30)
    ap.add_argument("--subject-mask-dilate", type=int, default=16)
    ap.add_argument("--subject-mask-feather", type=int, default=24)
    ap.add_argument("--device", default=None)
    return ap


def check_args(args) -> None:
    if not args.catalogue.exists():
        raise SystemExit(f"找不到主體名詞目錄 {args.catalogue}")
    for spec in args.entry:
        label, sep, d = spec.partition("=")
        if not sep or not label or not d:
            raise SystemExit(f"--entry 的格式是 LABEL=DIR，收到 {spec!r}")
        if not (Path(d) / "results.csv").exists():
            raise SystemExit(f"{d} 底下沒有 results.csv，無法取得條件名")


def subject_texts(catalogue: Path, name: str):
    """主體名詞。**查不到就拋錯**——猜一個會量到別的區域而且看不出來。"""
    import yaml

    spec = yaml.safe_load(catalogue.read_text(encoding="utf-8"))
    objs = (spec or {}).get("objects") or {}
    if name not in objs:
        raise SystemExit(
            f"{catalogue} 的 objects 裡沒有 {name} 的主體名詞。"
            "分區讀數需要知道要保護的是誰，不可猜。")
    texts = objs[name]
    return [texts] if isinstance(texts, str) else list(texts)


def triple(d: Path, name: str, cond: str):
    """該條件的 (防禦圖, 編輯原圖, 編輯防禦圖) 三張路徑，缺一即回 None。"""
    paths = {sub: d / f"{name}__{cond}__{sub}.png"
             for sub in ("def", "edit_orig", "edit_def")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print(f"[skip] 缺 {missing[0]}", flush=True)
        return None
    return paths


def main() -> None:
    args = build_parser().parse_args()
    check_args(args)

    from src.defense.subject_mask import mask_stats, subject_mask
    from src.metrics.regional import RegionalLPIPS, split_displacement
    from src.metrics.suite import MetricSuite

    dev = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu"))
    suite = MetricSuite(device=dev)
    regional = RegionalLPIPS(suite.lpips_module)

    rows = []
    for name in args.images:
        x = load_image_tensor(args.data / name / f"{name}.png", dev,
                              size=RESOLUTION)
        texts = subject_texts(args.catalogue, name)
        m = subject_mask(x, texts,
                         threshold=args.subject_mask_threshold,
                         dilate=args.subject_mask_dilate,
                         feather=args.subject_mask_feather)
        stats = mask_stats(m)
        print(f"{name[15:]:26s}遮罩 {texts}  {stats}", flush=True)

        for spec in args.entry:
            label, _, d = spec.partition("=")
            d = Path(d)
            with (d / "results.csv").open(encoding="utf-8-sig",
                                          newline="") as fh:
                recs = [r for r in csv.DictReader(fh) if r["image"] == name]
            for r in recs:
                cond = r["condition"]
                paths = triple(d, name, cond)
                if paths is None:
                    continue
                xd = load_image_tensor(paths["def"], dev, size=RESOLUTION)
                eo = load_image_tensor(paths["edit_orig"], dev, size=RESOLUTION)
                ed = load_image_tensor(paths["edit_def"], dev, size=RESOLUTION)

                disp = split_displacement(regional, eo, ed, m)
                dist = split_displacement(regional, x, xd, m)
                rows.append({
                    "image": name, "label": label, "condition": cond,
                    "subject_text": " | ".join(texts),
                    **stats,
                    "subject_mask_threshold": args.subject_mask_threshold,
                    "subject_mask_dilate": args.subject_mask_dilate,
                    "subject_mask_feather": args.subject_mask_feather,
                    "disp_full": round(disp["lpips_full"], 5),
                    "disp_subject": round(disp["lpips_subject"], 5),
                    "disp_background": round(disp["lpips_background"], 5),
                    "dist_full": round(dist["lpips_full"], 5),
                    "dist_subject": round(dist["lpips_subject"], 5),
                    "dist_background": round(dist["lpips_background"], 5),
                    # 交叉核對用：`disp_full` 與這一欄應相符。不符代表讀到的
                    # 圖與那一列不是同一次跑出來的。
                    "edit_lpips_recorded": r.get("edit_lpips", ""),
                    "fid_dists_recorded": r.get("fid_dists", ""),
                    "loss_trained": r.get("loss", ""),
                })
                print(f"{name[15:]:26s}{label + '/' + cond:26s}"
                      f"位移 全{disp['lpips_full']:.4f} "
                      f"主體{disp['lpips_subject']:.4f} "
                      f"背景{disp['lpips_background']:.4f}  "
                      f"失真 主體{dist['lpips_subject']:.4f}  "
                      f"（記錄的全圖位移 {r.get('edit_lpips', '')}）", flush=True)
                write_csv(args.out, rows)

    write_csv(args.out, rows)
    print(f"\n寫出 {args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
