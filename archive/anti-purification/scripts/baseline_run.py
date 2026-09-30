"""四個防護 baseline 在各自原生設定下的求解與評測。

每一篇用**它自己的損失與預算**（`src/baselines/pgd.py::run_pgd`），逐欄的出處
在 `docs/reference/BASELINE_PROVENANCE.md`：

    photoguard_c      官方 notebook cell 10 唯一啟用的呼叫：L2 renorm maxnorm=16
    photoguard_linf   論文 Appendix A.2 / Table 9：ℓ∞ 16/255、step 2/255
    mist              mist_utils.py CLI 預設：ℓ∞ 16/255、α 1/255、100 步
    dia_r             論文 §4.1 ＋ attack_setting.json：ℓ∞ 0.05、20 步

PhotoGuard 有兩個臂，是因為**論文與官方程式互相矛盾**，而 repo 裡不存在同時
滿足 Table 9 四欄的程式碼。兩個臂只差約束本身，其餘一致。

評測：保真度（`x_def` 對原圖）＋ 抗編輯（SDEdit 後對「未防禦的編輯」比較）。
**未防禦的編輯必須真的成功**，否則抗編輯那一欄的分母不成立。

身分讀數、ArcFace 與 NR-IQA 不在這裡算——它們由 `scripts/readout_panel.py`
讀已存的 PNG 補上，不必重跑攻擊。

用法：
    python scripts/baseline_run.py --out runs/<批次> --data data/lo_aligned         --images horse_00 man_00 bird_03
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.baselines import dia, mist, photoguard  # noqa: E402
from src.baselines.pgd import run_pgd  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402
from src.metrics.aesthetic import AestheticSuite  # noqa: E402
from src.metrics.standard import standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
RESOLUTION = 512

# 評測設定。strength 0.55 由 DEC-022 定出——0.4 下有影像的未防禦編輯
# 根本沒發生，分母不成立。換資料集時**必須重新確認未防禦的編輯有成功**。
# 2026-08-17：0.55 下九張影像沒有一張服從 prompt（`runs/editsweep`），
# 使用者的驗收條件是「一切實驗都必須在編輯成功的前提下」，故升到 0.8。
# 0.8 會把臉整個換掉。原本的對策是加頭部遮罩，2026-08-17 使用者改為在 prompt
# 裡寫出本人姓名讓模型自己生回同一個人，遮罩與 `data/set0817/headmasks/` 一併
# 撤回；`load_dataset` 找不到遮罩就不做 latent 混合。
EDIT_STRENGTH = 0.8
EDIT_GUIDANCE = 7.5
EDIT_STEPS = 30
EDIT_SEED = 20260812

CONDITIONS = ["photoguard_c", "photoguard_linf", "mist", "dia_r"]


def load_dataset(root: Path, prompt_index: int = 0) -> list:
    """讀 `root` 的每類一子目錄版面（`prompts.yaml` 的鍵即子目錄名）。

    `prompt_index` 選 `prompts.yaml` 的第幾個編輯 prompt。0 是「改掉指定內容」
    （A dog -> A cat），1 是「保留該內容、改動其他區域」（A dog in the park）。
    兩者是不同的惡意情境，防禦的難度不同——第二個 prompt 不要求模型改掉主體，
    故未防禦的編輯改動較小，抗編輯那一欄的分母也較小。
    **換 prompt 必須重新確認未防禦的編輯有成功。**
    """
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for c in sorted(spec):
        for img in sorted((root / c).glob("*.png")):
            hm = root / "headmasks" / f"{img.stem}.png"
            out.append({"name": img.stem, "class": spec[c]["content"],
                        "path": img, "content": spec[c]["content"],
                        "prompt": spec[c]["prompts"][prompt_index],
                        "prompt_index": prompt_index,
                        "head_mask": hm if hm.exists() else None})
    return out


def run_additive(sd, item, name: str, seed: int,
                 strength: float = EDIT_STRENGTH) -> dict:
    """`strength` 只有 photoguard_c 會用到——它的攻擊在指定的 img2img 強度上
    最佳化（Salman et al. 的 diffusion attack）。其餘兩個加性方法與強度無關，
    故同一組防禦圖可以在多個強度上重複評測。"""
    spec = {"photoguard_c": photoguard.SPEC,
            "photoguard_linf": photoguard.SPEC_PAPER_LINF,
            "mist": mist.SPEC,
            "dia_r": dia.SPEC_R}[name]
    kw = {}
    if name in ("photoguard_c", "photoguard_linf"):
        kw = {"mask": None, "strength": strength}
    elif name == "mist":
        # fused mode 把兩次 VAE 編碼與一次完整 UNet 前向放在同一張圖上。
        kw = {"use_ckpt": True, "vae_ckpt": True,
              "target01": load_image_tensor(
                  Path("data/targets/MIST.png"), sd.device, size=RESOLUTION)}
    elif name == "dia_r":
        # DIA-R 把整條反演加整條重建留在同一張圖上，兩個開關都必須開，
        # 否則 OOM（`executors.baseline_kwargs` 記過同一條規則）。
        kw = {"use_ckpt": True, "vae_ckpt": True}
    return {"x_def": run_pgd(sd, item["path01"], spec, seed=seed, **kw).x_adv01.detach()}


def head_keep(item, x01):
    """讀出該影像的頭部保留遮罩 (1,1,H,W)，沒有就回 None。

    只有人物有遮罩，動物沒有——動物在 0.8 下換場景／戴帽都成功且主體還在，
    不需要保留區。遮罩存的是灰階 PNG，羽化過的邊要原樣用，二值化會在
    混合處留下接縫。
    """
    p = item.get("head_mask")
    if p is None:
        return None
    from src.utils.io import load_image_tensor
    m = load_image_tensor(p, x01.device, size=x01.shape[-1])
    return m[:, :1]


def evaluate(sd, suite, aes, item, x_def, strength: float = EDIT_STRENGTH):
    x01 = item["path01"]
    m = suite.pairwise(x01, x_def)
    a = aes.measure(x_def)
    fid = {"lpips": m["lpips"], "ssim": m["ssim"], "dists": m["dists"],
           "psnr": m["psnr"], "clip_img": aes.clip_image_similarity(x01, x_def),
           "nima": a["nima"], "cnniqa": a["cnniqa"]}

    noise = sd.sample_edit_noise(sd.encode_image(x01), seed=EDIT_SEED)
    emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
    keep = head_keep(item, x01)
    with torch.no_grad():
        edit_orig = sd.sdedit(x01, emb, noise, EDIT_STEPS, strength=strength,
                              guidance_scale=EDIT_GUIDANCE, emb_uncond=emb_u,
                              keep01=keep)
        edit_def = sd.sdedit(x_def.clamp(0, 1), emb, noise, EDIT_STEPS,
                             strength=strength, guidance_scale=EDIT_GUIDANCE,
                             emb_uncond=emb_u, keep01=keep)
    so = suite.semantic(edit_orig, item["prompt"])
    sd_ = suite.semantic(edit_def, item["prompt"])
    # DEC-028 的統一指標清單：兩個半邊都要五項成對指標。此前失真半邊缺
    # VIFp、防禦半邊只有 LPIPS，於是本專案的表與 DCT-Shield（arXiv:2504.17894）
    # Table 1 無法逐欄對照。`standard_row` 缺欄位會拋錯，不會靜默少報。
    prot = suite.pairwise(edit_orig, edit_def)
    return ({**{f"fid_{k}": round(v, 4) for k, v in fid.items()},
             **standard_row("fid_", m), **standard_row("edit_", prot),
             "edit_strength": strength,
             "edit_lpips": round(float(prot["lpips"]), 4),
             "edit_clip_orig": round(so["clip"], 4),
             "edit_clip_def": round(sd_["clip"], 4),
             "edit_clip_drop": round(so["clip"] - sd_["clip"], 4),
             "edit_siglip_orig": round(so["siglip"], 4),
             "edit_siglip_def": round(sd_["siglip"], 4),
             "edit_siglip_drop": round(so["siglip"] - sd_["siglip"], 4)},
            edit_orig, edit_def)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="用 prompts.yaml 的第幾個編輯 prompt（0 改內容、1 改場景）")
    ap.add_argument("--edit-strength", type=float, default=EDIT_STRENGTH,
                    help="SDEdit 的 strength。只有 photoguard_c 的攻擊會用到它；"
                         "其餘條件的防禦圖與強度無關，故同一批可在多個強度上"
                         "重複評測（reeval_edits.py 也有這個旗標）")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    aes = AestheticSuite(device=sd.device)
    dataset = load_dataset(args.data, prompt_index=args.prompt_index)
    if args.images:
        dataset = [d for d in dataset if d["name"] in args.images]
    conds = args.conditions or CONDITIONS

    rows = []
    for item in dataset:
        item["path01"] = load_image_tensor(item["path"], sd.device,
                                                     size=RESOLUTION)
        save_image(item["path01"], args.out / f"{item['name']}__orig.png")
        print(f"\n########## {item['name']} ({item['class']}) ##########", flush=True)
        for cond in conds:
            print(f"=== {item['name']} / {cond} ===", flush=True)
            t0 = time.time()
            res = run_additive(sd, item, cond, args.seed,
                               strength=args.edit_strength)
            metrics, eo, ed = evaluate(sd, suite, aes, item, res["x_def"],
                                       strength=args.edit_strength)
            for tag, img in (("def", res["x_def"]), ("edit_orig", eo), ("edit_def", ed)):
                save_image(img, args.out / f"{item['name']}__{cond}__{tag}.png")
            row = {"image": item["name"], "condition": cond,
                   "total_seconds": round(time.time() - t0, 1), **metrics}
            rows.append(row)
            print(row, flush=True)
            write_csv(args.out / "results.csv", rows)
    print(f"\n表：{args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
