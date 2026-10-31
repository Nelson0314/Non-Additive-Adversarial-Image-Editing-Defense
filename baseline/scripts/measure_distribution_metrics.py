"""防禦圖與編輯結果的 CLIP-IQA（逐張）與 FID／KID（集合對集合），失真端與位移端（CPU）。

端別與集合（A 為受測集合，B 為參照集合；逐條件、逐淨化各一組）：
    distortion             A＝防禦圖，B＝原圖
    displacement / none    A＝防禦後編輯（ip2p，四句），B＝未防禦編輯（同影像、同句）
    displacement / <算子>  A＝淨化後防禦圖的編輯，B＝同一淨化的未防禦編輯（七道）
    displacement / vae_trans  A、B 為 VAE-Trans 淨化後的編輯（color trial 的 VAE-Trans 產物）
資料組（--dataset）：
    portraits   data/portraits 8 張；等失真 11 臂（artifacts/aligned/）與 color3d（color trial artifacts/，只讀）
    pexels      Pexels 人像中已完成的影像（見 pexels_groups）；n_a 為格數，影像數另記於 n_images
特徵與指標：
    CLIP-IQA   pyiqa `clipiqa`，逐張（與 measure_additional_metrics 的 aes_clipiqa 相同）
    FID、KID   piq.FID、piq.KID（三次多項式核，n_subsets 100，subset_size＝較小集合的張數），特徵為 piq 的 InceptionV3
               （pytorch-fid 權重 pt_inception-2015-12-05，pool3 2048 維，輸入縮放到 299）；兩集合各 n_a、n_b 張
小樣本下 FID 的共變異數估計秩不足（2048 維、≤ 32 張），數值隨樣本數變動大，n 逐列記錄。

輸出：<輸出目錄>/<dataset>_clipiqa.csv（逐張）、<dataset>_distribution.csv（逐組 FID、KID、CLIP-IQA 平均）。
用法（遠端 baseline 專案根）：python scripts/measure_distribution_metrics.py <輸出目錄> --dataset portraits
大量影像時先以 --shard i --shards n 平行抽取（各片存 .pt），再不帶 --shard 執行一次合併計算各組。
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import piq
import pyiqa
import torch
from PIL import Image
from piq.feature_extractors import InceptionV3

HOME = Path.home()
CT = HOME / "image-immunization" / "color" / "trials" / "lut3d_defense_and_naturalness" / "artifacts"
IMAGES = ("man_00", "man_01", "man_02", "man_03", "woman_00", "woman_01", "woman_02", "woman_03")
ARMS = ("danp", "dayn", "dct_shield", "dct_shield_y", "dia_pt", "dia_r", "mist", "photoguard_c",
        "photoguard_encoder", "photoguard_linf", "sifm")
PURIFIERS = ("blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15")
PROMPTS = range(4)


def portraits_groups():
    undefended = Path("artifacts/undefended_edits/ip2p_si18")
    pur_undef = Path("artifacts/aligned/purified_edits/undefended")
    vae = CT / "purification" / "vae_trans" / "edits"
    cells = lambda d: [d / f"{im}__p{p}.png" for im in IMAGES for p in PROMPTS]  # noqa: E731
    for c in ARMS + ("color3d",):
        if c == "color3d":
            dfn, edits = CT / "defenses" / c, CT / "defended_edits" / c / f"ip2p_{c}"
            pur = lambda op: CT / "purified_edits" / c / op / f"ip2p_{c}_{op}"  # noqa: E731
        else:
            dfn, edits = Path("artifacts/aligned/defenses") / c, Path("artifacts/aligned/defended_edits") / c / f"ip2p_{c}_aligned"
            pur = lambda op: Path("artifacts/aligned/purified_edits") / c / op / f"ip2p_{c}_{op}"  # noqa: E731
        yield c, "distortion", "", [dfn / f"{im}__{c}__def.png" for im in IMAGES], [dfn / f"{im}__orig.png" for im in IMAGES]
        yield c, "displacement", "none", cells(edits), cells(undefended)
        for op in PURIFIERS:
            yield c, "displacement", op, cells(pur(op)), cells(pur_undef / op / f"ip2p_undefended_{op}")
        yield c, "displacement", "vae_trans", cells(vae / f"ip2p_{c}_vae_trans"), cells(vae / "ip2p_undefended_vae_trans")


def pexels_groups():
    """Pexels 人像：等失真臂（artifacts/pexels_portraits/aligned/，第一批與 sets/<影像組>/）與 color3d_pexels（color trial
    artifacts_pexels/），只取兩側檔案都存在的格；VAE-Trans 取 artifacts/pexels_portraits/vae_trans/<根>/（scripts/evaluate_pexels_followup.sh）。"""
    base = Path("artifacts/pexels_portraits")
    roots = [(base / "aligned", base, base / "vae_trans")] + [
        (base / "aligned" / "sets" / s.name, base / "sets" / s.name, base / "vae_trans" / "sets" / s.name)
        for s in sorted((base / "aligned" / "sets").iterdir())]
    vae_root = {im: base / "vae_trans" / s["root"] for s in json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
                for im in s["images"]}

    def vae_denominator(vroot, im, p):
        return vroot / "native/purified_edits/undefended/vae_trans/ip2p_undefended_vae_trans" / f"{im}__p{p}.png"

    groups = {}

    def add(c, end, purifier, a, b):
        if a.is_file() and b.is_file():
            groups.setdefault((c, end, purifier), ([], []))
            groups[(c, end, purifier)][0].append(a)
            groups[(c, end, purifier)][1].append(b)

    for aroot, uroot, vroot in roots:
        if not (aroot / "defenses").is_dir():
            continue
        for cdir in sorted((aroot / "defenses").iterdir()):
            c = cdir.name
            for d in sorted(cdir.glob(f"*__{c}__def.png")):
                im = d.name.split("__")[0]
                add(c, "distortion", "", d, cdir / f"{im}__orig.png")
                for p in PROMPTS:
                    add(c, "displacement", "none", aroot / "defended_edits" / c / f"ip2p_{c}" / f"{im}__p{p}.png",
                        uroot / "undefended_edits" / "ip2p_si18" / f"{im}__p{p}.png")
                    for op in PURIFIERS:
                        add(c, "displacement", op,
                            aroot / "purified_edits" / c / op / f"ip2p_{c}_{op}" / f"{im}__p{p}.png",
                            uroot / "purified_edits" / "undefended" / op / f"ip2p_undefended_{op}" / f"{im}__p{p}.png")
                    add(c, "displacement", "vae_trans",
                        vroot / "aligned/purified_edits" / c / "vae_trans" / f"ip2p_{c}_vae_trans" / f"{im}__p{p}.png",
                        vae_denominator(vroot, im, p))
    cp, n = CT.parent / "artifacts_pexels", "color3d_pexels"
    for shard in sorted((cp / "defense_shards" / n).iterdir()):
        im = shard.name
        add("color3d", "distortion", "", shard / f"{im}__{n}__def.png", shard / f"{im}__orig.png")
        e = cp / "edits_by_image" / n / im
        for p in PROMPTS:
            add("color3d", "displacement", "none", e / "defended" / f"ip2p_{n}" / f"{im}__p{p}.png",
                cp / "undefended_edits" / "ip2p_si18" / f"{im}__p{p}.png")
            for op in PURIFIERS:
                add("color3d", "displacement", op, e / "purified" / op / f"ip2p_{n}_{op}" / f"{im}__p{p}.png",
                    cp / "purified_edits" / "undefended" / op / f"ip2p_undefended_{op}" / f"{im}__p{p}.png")
            add("color3d", "displacement", "vae_trans",
                vae_root[im] / "color3d/purified_edits" / n / "vae_trans" / f"ip2p_{n}_vae_trans" / f"{im}__p{p}.png",
                vae_denominator(vae_root[im], im, p))
    for (c, end, purifier), (a, b) in sorted(groups.items()):
        yield c, end, purifier, a, b


def load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dataset", choices=("portraits", "pexels"), required=True)
    ap.add_argument("--shard", type=int, default=None,
                    help="只抽取第 i 片（共 --shards 片）不重複影像的 CLIP-IQA 與特徵，存成 <輸出目錄>/<dataset>_features_<i>.pt")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--require-images", type=int, default=None, help="每組的影像數須為此值，否則失敗（合併時）")
    ap.add_argument("--tag", default="", help="分片檔名的附加段（<dataset>_features_<tag>_<i>.pt）；已存在的分片檔中有的影像不重算")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    groups = list(portraits_groups() if args.dataset == "portraits" else pexels_groups())
    missing = [str(p) for g in groups for p in g[3] + g[4] if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"缺 {len(missing)} 個檔案，例如 {missing[:3]}")

    clipiqa = pyiqa.create_metric("clipiqa", device="cpu", as_loss=False)
    inception = InceptionV3().eval()
    q_cache, f_cache = {}, {}
    for part in sorted(args.out.glob(f"{args.dataset}_features_*.pt")):
        saved = torch.load(part)
        q_cache.update(saved["quality"])
        f_cache.update(saved["feature"])
    if args.shard is not None:
        paths = sorted({str(p) for g in groups for p in g[3] + g[4]} - set(f_cache))[args.shard::args.shards]
        q_cache, f_cache = {}, {}
        for k, p in enumerate(paths):
            with torch.no_grad():
                x = load(p)
                q_cache[p], f_cache[p] = float(clipiqa(x)), inception(x)[0].flatten(1)[0]
            if k % 500 == 0:
                print(f"[FEATURES] {args.shard}/{args.shards} {k}/{len(paths)}", flush=True)
        name = f"{args.dataset}_features_{args.tag}_{args.shard}.pt" if args.tag else f"{args.dataset}_features_{args.shard}.pt"
        torch.save({"quality": q_cache, "feature": f_cache}, args.out / name)
        print(f"[ALLDONE] shard {args.shard}", flush=True)
        return

    def quality(p):
        if p not in q_cache:
            with torch.no_grad():
                q_cache[p] = float(clipiqa(load(p)))
        return q_cache[p]

    def feature(p):
        if p not in f_cache:
            with torch.no_grad():
                f_cache[p] = inception(load(p))[0].flatten(1)[0]
        return f_cache[p]

    if args.require_images is not None:
        wrong = [(g[0], g[1], g[2], n) for g in groups
                 if (n := len({Path(p).name.split("__")[0] for p in g[3]})) != args.require_images]
        if wrong:
            raise SystemExit(f"{len(wrong)} 組的影像數不是 {args.require_images}，例如 {wrong[:3]}")
    rows, per_image = [], []
    for c, end, purifier, a, b in groups:
        qa, qb = [quality(str(p)) for p in a], [quality(str(p)) for p in b]
        fa, fb = torch.stack([feature(str(p)) for p in a]), torch.stack([feature(str(p)) for p in b])
        n = min(len(a), len(b))
        with torch.no_grad():
            fid = float(piq.FID()(fa, fb))
            kid = float(piq.KID(average=True, n_subsets=100, subset_size=n)(fa, fb))
        rows.append({"dataset": args.dataset, "condition": c, "end": end, "purifier": purifier,
                     "n_images": len({Path(p).name.split("__")[0] for p in a}), "n_a": len(a), "n_b": len(b), "fid": round(fid, 3), "kid": round(kid, 6), "kid_negative": kid < 0,
                     "clipiqa_a": round(float(np.mean(qa)), 5), "clipiqa_b": round(float(np.mean(qb)), 5),
                     "feature_extractor": "piq InceptionV3 pool3 (pt_inception-2015-12-05)", "device": "cpu"})
        per_image += [{"dataset": args.dataset, "condition": c, "end": end, "purifier": purifier, "png": str(p),
                       "clipiqa": round(q, 5)} for p, q in zip(a, qa)]
        print(f"[DONE] {c} {end} {purifier} fid={fid:.2f} kid={kid:.4f}", flush=True)
    for name, data in ((f"{args.dataset}_distribution.csv", rows), (f"{args.dataset}_clipiqa.csv", per_image)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    print(f"[ALLDONE] {args.out}", flush=True)


if __name__ == "__main__":
    main()
