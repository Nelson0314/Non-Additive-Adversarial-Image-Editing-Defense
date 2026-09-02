"""語意誘餌：防禦端先對原圖下一個**良性**編輯，交出那張圖。

這一族的產物裡**沒有噪聲**
────────────────────────────────────────────────────────────────────
所有既有的防禦（加性 δ、相位重參數化、DCT-Shield、明暗場、色彩重映射）交出的
都是「原圖 ＋ 某種擾動」，於是淨化算子有東西可以洗。本族交出的是**另一張真的
照片**：背景換了、多了一個物件。淨化在定義上是恆等——JPEG、模糊、裁切、灰階
對它做的事，與對任何一張自然照片做的事完全一樣，沒有「擾動的存活率」這個量。

問的因此是一個乾淨的問題：**一個自然的內容改動，能不能讓攻擊方的指令編輯
失敗？** 假說是語意碰撞與注意力誤導——指令說「把盆栽改成粉紅色」，而畫面上
已經有一大塊粉紅色的東西。

`data/decoy_catalogue.yaml` 是誘餌指令的目錄，挑選原則寫在該檔的檔頭。
帶 `{object}` 的模板由同一份檔案的 `objects` 對照表逐影像代入——**受保護主體
的名稱是防禦方本來就知道的**（威脅模型的前提），攻擊指令不是；沒有登記主體的
影像會跳過該模板並印出理由。

三件必須照實記的事
────────────────────────────────────────────────────────────────────
1. **防禦方與攻擊方用同一個模型、不同的種子。** 白盒假設下防禦方本來就拿得到
   權重。種子分開是因為同一個種子會讓兩次取樣走同一條軌跡，那是人為的優勢。
   兩個種子都逐列寫進 CSV。
2. **失真會很大，而且那不是缺陷。** 使用者已裁定可見的防禦是允許的，條件是
   產物自然、且不動到受保護的主體。`fid_*` 各欄照報，但**不可拿來與加性族
   等失真對齊**——那個軸在這裡沒有意義。
3. **現行的位移讀數在這一族上不可解讀。** `LPIPS(編輯(原圖), 編輯(誘餌圖))`
   會被內容差異灌水，而空白地板扣不掉那一份。共防禦參照
   （`src/defense/codefense.py`）對這一族是 `not_applicable`——誘餌不是逐點
   映射，沒有辦法把同一個 `D` 套到 `編輯(原圖)` 上。**判定要靠影像**，
   本腳本因此把原圖、誘餌圖、兩邊的編輯結果全部留下。

輸出的版面與 `ip2p_run.py` 相同（`<影像>__<條件>__def.png` ＋ `results.csv`），
故 `scripts/phase_retention.py` 可以直接吃它跑抗淨化。

用法：

    python scripts/semantic_decoy.py --out runs/ip2p_decoy \\
        --images task_attr_mod_color_11699 ... --groups collision
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torchvision.utils as vutils  # noqa: E402
import yaml  # noqa: E402

from apa_baseline import load_dataset  # noqa: E402
from src.metrics.standard import standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.ip2p import (  # noqa: E402
    IP2P_IMAGE_GUIDANCE, IP2P_SEED, IP2P_STEPS, IP2P_TEXT_GUIDANCE, IP2PWrapper,
)
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
# 防禦端的取樣種子。**與攻擊端的 `IP2P_SEED` 不同**：同一個種子會讓兩次取樣
# 走同一條軌跡，那是人為的優勢，不是方法的性質。這個值是本專案指定的。
DECOY_SEED = 20260902


# 目錄裡不是誘餌組的鍵。`objects` 是「受保護主體叫什麼」的對照表，供帶
# `{object}` 的模板逐影像代入。
RESERVED_KEYS = ("objects",)


def catalogue(path: Path, groups) -> tuple:
    """→ (誘餌清單, 受保護主體對照表)。

    帶 `{object}` 的模板**不在這裡代入**——同一個模板在不同影像上是不同的
    指令，代入要等到知道是哪張影像。
    """
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    available = [k for k in spec if k not in RESERVED_KEYS]
    unknown = set(groups) - set(available)
    if unknown:
        raise SystemExit(
            f"目錄裡沒有這些組：{sorted(unknown)}；有的是 {sorted(available)}")
    out = []
    for g in groups:
        for i, text in enumerate(spec[g]):
            out.append({"group": g, "index": i, "instruction": text,
                        "condition": f"decoy_{g}_{i}"})
    return out, dict(spec.get("objects") or {})


def fill(template: str, name: str, objects: dict) -> Optional[str]:
    """把 `{object}` 換成該影像的受保護主體。沒登記就回傳 None。

    **主體的名稱來自 `objects` 對照表，不是從攻擊指令解析出來的。**
    威脅模型的前提是防護對象已知；攻擊指令不是。
    """
    if "{object}" not in template:
        return template
    obj = objects.get(name)
    return None if not obj else template.replace("{object}", obj)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--catalogue", type=Path,
                    default=Path("data/decoy_catalogue.yaml"))
    ap.add_argument("--groups", nargs="+", default=["background", "collision"])
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decoy-seed", type=int, default=DECOY_SEED,
                    help="防禦端的取樣種子。與攻擊端的種子分開，理由見模組 "
                         "docstring；逐列寫進 CSV")
    ap.add_argument("--decoy-steps", type=int, default=IP2P_STEPS)
    ap.add_argument("--decoy-text-guidance", type=float,
                    default=IP2P_TEXT_GUIDANCE)
    ap.add_argument("--decoy-image-guidance", type=float,
                    default=IP2P_IMAGE_GUIDANCE,
                    help="誘餌要**真的改到畫面**，所以影像引導可以比攻擊端低；"
                         "預設沿用攻擊端的值，改動要進報表")
    args = ap.parse_args()

    decoys, objects = catalogue(args.catalogue, args.groups)
    dataset = {d["name"]: d for d in load_dataset(args.data)}
    missing = [n for n in args.images if n not in dataset]
    if missing:
        raise SystemExit(f"資料集裡沒有這些影像：{missing}")

    sd = IP2PWrapper(dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.images:
        item = dataset[name]
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        vutils.save_image(x01.clamp(0, 1), args.out / f"{name}__orig.png")
        # 攻擊方對**原圖**的編輯：判定誘餌有沒有用的對照組。
        e_orig = sd.edit(x01.clamp(0, 1), item["prompt"], seed=IP2P_SEED,
                         steps=IP2P_STEPS, s_t=IP2P_TEXT_GUIDANCE,
                         s_i=IP2P_IMAGE_GUIDANCE)
        for d in decoys:
            t0 = time.time()
            cond = d["condition"]
            decoy_text = fill(d["instruction"], name, objects)
            if decoy_text is None:
                print(f"[skip] {name}／{cond}：模板要 {{object}} 而 objects "
                      f"對照表裡沒有這張影像的主體", flush=True)
                continue
            # 防禦端：良性編輯，自己的種子。
            x_def = sd.edit(x01.clamp(0, 1), decoy_text,
                            seed=args.decoy_seed, steps=args.decoy_steps,
                            s_t=args.decoy_text_guidance,
                            s_i=args.decoy_image_guidance).clamp(0, 1)
            # 攻擊端：對誘餌圖下原本的惡意指令。
            e_def = sd.edit(x_def, item["prompt"], seed=IP2P_SEED,
                            steps=IP2P_STEPS, s_t=IP2P_TEXT_GUIDANCE,
                            s_i=IP2P_IMAGE_GUIDANCE)
            for sub, img in (("def", x_def), ("edit_orig", e_orig),
                             ("edit_def", e_def)):
                vutils.save_image(img.clamp(0, 1),
                                  args.out / f"{name}__{cond}__{sub}.png")
            fid = suite.pairwise(x01, x_def)
            eff = suite.pairwise(e_orig, e_def)
            sim = suite.image_similarity(e_orig, e_def)
            rows.append({
                "image": name, "condition": cond,
                "attacker": "instruct-pix2pix",
                "instruction": item["prompt"], "task": item.get("class", ""),
                "decoy_group": d["group"], "decoy_index": d["index"],
                "decoy_instruction": decoy_text,
                "decoy_template": d["instruction"],
                # 防禦端的四個設定。**全部是本專案指定、論文無出處**，
                # 故是欄位不是註解。
                "decoy_seed": args.decoy_seed,
                "decoy_steps": args.decoy_steps,
                "decoy_s_t": args.decoy_text_guidance,
                "decoy_s_i": args.decoy_image_guidance,
                # 攻擊端的設定，與其餘各批相同。
                "edit_steps": IP2P_STEPS, "s_t": IP2P_TEXT_GUIDANCE,
                "s_i": IP2P_IMAGE_GUIDANCE, "edit_seed": IP2P_SEED,
                # `radius` 這一欄在本族沒有意義，留空而不是填 0——填 0 會被
                # 等失真內插當成一個真的工作點。
                "radius": "",
                **standard_row("fid_", fid),
                **standard_row("edit_", eff),
                "fid_deltaE00": round(fid["deltaE00"], 4),
                "edit_clip_sim": round(sim["clip"], 5),
                "edit_siglip_sim": round(sim["siglip"], 5),
                "total_seconds": round(time.time() - t0, 1),
            })
            write_csv(args.out / "results.csv", rows)
            print(f"{name:38s} {cond:24s} "
                  f"fid_dists={fid['dists']:.4f} edit_lpips={eff['lpips']:.4f} "
                  f"siglip={sim['siglip']:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)

    print(f"\n表：{args.out / 'results.csv'}（{len(rows)} 列）")
    print("**判定要看圖**：本族的位移讀數被內容差異灌水，共防禦參照對它是 "
          "not_applicable。")


if __name__ == "__main__":
    main()
