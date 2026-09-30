"""用 **InstructPix2Pix** 對已存的防禦圖跑真編輯。**不重跑任何攻擊。**

為什麼要這一支
────────────────────────────────────────────────────────────────────
各 baseline 的防禦效果參考值（DCT-Shield Table 1 的 Edit Protection 半邊）是在
**IP2P 上量的**，不是 SDEdit。`baseline_run.py` 的評測走 SD 1.4 SDEdit，那一側
只能對失真的半邊；要確認**防禦效果**也重現，編輯器必須換成 IP2P。

換掉 SDEdit 還解決另一件事：SDEdit strength 0.8 下未防禦的編輯自己就把人換掉
（`id_orig` 0.115，同一人的參照是 0.55），身分那一欄的分母是零。降到 0.4 時
`id_orig` 回到 0.560 但編輯不再發生。**SDEdit 兩件事不能兼顧，IP2P 可以**——
它保結構，未防禦的編輯會照指令做事而且人還是同一個。

語意類：指令有沒有被做到
────────────────────────────────────────────────────────────────────
CLIP 與 SigLIP 各報五欄（`_input`／`_orig`／`_def`／`_gain_orig`／`_gain_def`）。
**`_input` 不可省**：不扣掉原圖自己對該指令的對齊，就分不出「指令做到了」與
「原圖本來就像那句話」。`gain_orig` 太小代表**未防禦的編輯沒成功**，那一格的
分母不成立、整格不可解讀，與 `id_orig` 太低是同一種失效。

指令與種子
────────────────────────────────────────────────────────────────────
指令式句型（不是描述式 caption），涵蓋配件、背景、非臉部區域換色。**防禦圖是
唯讀的**，求解時看不到任何指令，所以這裡的每一句都是 held-out。逐指令逐種子
各一列，不取平均——同一格換一顆種子就可能翻盤。

種子是固定的，而且**實測過**：同一顆種子連跑三次，IP2P 的輸出逐位元相同
（max|diff| = 0）；換一顆種子則 max|diff| 0.70。編輯端因此是可重現的分母，
與求解端的 GPU 非決定性（見 `docs/EVALUATION.md`）不同。

`red_shirt` 對**整體調色類的防禦**有已知混淆：那種方法會改寫整張影像的顏色，
「衣服是什麼顏色」因此分不出是防禦造成還是編輯造成。當 baseline 的指令沒問題，
拿去比整體調色的方法時這一欄要另外處理。

產物的檔名與 `baseline_run.py` 相同，故 `readout_panel.py` 可以直接吃這個目錄：

    {image}__orig.png
    {image}__{condition}__{instruction}__s{seed}__edit_orig.png
    {image}__{condition}__{instruction}__s{seed}__edit_def.png
    {image}__{condition}__def.png            （由來源批次複製，保留真圖）

**`edit_orig` 逐指令逐種子只算一次**：它只與原圖、指令、種子有關，與防禦條件
無關，再依各條件的檔名分別存出。

用法
    python scripts/reedit_ip2p.py --runs runs/baseline_restore/shard1 ... \\
        --data data/lo_aligned --out runs/ip2p_edits --shard 1/5
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# `configs/evaluate_instruction.json` 的 IP2P 設定，逐欄沿用。
IP2P_STEPS = 50
S_T = 7.5
S_I = 1.5
SEEDS = (17001, 27001, 47001)

# 四句指令，全部經 `scripts/instruction_preflight.py` 的未防禦預檢逐張看圖確認：
# 指令確實被執行、而且**主體沒有被換掉**。人與動物共用同一句，故用 "it" 而不是
# "the person"——十張圖裡有四張是動物，逐類換句會讓「指令難度」與「影像類別」
# 綁在一起。
#
# 預檢剔除掉的三句，理由各不相同，都要寫下來免得再被撿回來：
#   `make it look older`      man_00 上身分只剩 0.30（換人），woman_00 上是
#                             0.99（什麼都沒發生）。兩種失效都讓那一格的分母
#                             不成立。
#   `put it in a snowy street` 只疊了雪、背景沒換；兩張圖的 CLIP 增益差 150 倍
#                             （0.0003 對 0.0451），不一致。
#   `make it smile`           **圖上明顯成功**（人在笑、LPIPS 0.098 的局部改動、
#                             身分 0.876），但 CLIP 增益是 −0.0126。CLIP 看不出
#                             細微的表情編輯，所以它不能單獨當指令完成度的判準。
#                             這一句不用，但這個發現要留著。
INSTRUCTIONS = {
    "hat": "put a hat on it",
    "glasses": "put a pair of glasses on it",
    "beach_bg": "change the background to a beach",
    "red_shirt": "make the shirt red",
}


def _orig_index(data_root: Path) -> dict:
    return {p.stem: p for p in sorted(data_root.glob("*/*.png"))
            if p.parent.name not in ("headmasks", "masks")}


def _pairs(run_dir: Path):
    out = []
    for p in sorted(run_dir.glob("*__def.png")):
        stem = p.name[: -len("__def.png")]
        image, _, condition = stem.partition("__")
        if condition:
            out.append((image, condition, p))
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()

    from src.defense.assets import load_image, save_png
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper

    ip2p = IP2PWrapper(dtype=torch.bfloat16)
    device = ip2p.device
    suite = MetricSuite(device=device)

    origs = _orig_index(args.data)
    cells = []
    for run_dir in args.runs:
        for image, condition, def_png in _pairs(run_dir):
            if image not in origs:
                raise SystemExit(f"資料集裡沒有 {image}；--data 指錯了目錄")
            cells.append({"image": image, "condition": condition, "def": def_png})

    i, n = (int(v) for v in args.shard.split("/"))
    # 依影像分片，讓同一張圖的 `edit_orig` 只算一次。
    names = sorted({c["image"] for c in cells})
    mine = {nm for k, nm in enumerate(names) if k % n == i - 1}
    cells = [c for c in cells if c["image"] in mine]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"分片 {i}/{n}：{len(mine)} 張影像、{len(cells)} 個 (影像, 條件)、"
          f"{len(INSTRUCTIONS)} 指令 × {len(args.seeds)} 種子", flush=True)

    rows = []
    csv_path = args.out / f"results_shard{i}of{n}.csv"
    for name in sorted(mine):
        x01 = load_image(origs[name], device)
        save_png(x01, args.out / f"{name}__orig.png")
        conds = [c for c in cells if c["image"] == name]
        for c in conds:                      # 保留真圖：防禦圖原樣複製過來
            shutil.copy2(c["def"], args.out / f"{name}__{c['condition']}__def.png")

        for iname, instruction in INSTRUCTIONS.items():
            for seed in args.seeds:
                t0 = time.time()
                with torch.no_grad():
                    edit_orig = ip2p.edit(x01, instruction, seed=seed,
                                          steps=IP2P_STEPS, s_t=S_T, s_i=S_I)
                for c in conds:
                    tag = f"{name}__{c['condition']}__{iname}__s{seed}"
                    save_png(edit_orig, args.out / f"{tag}__edit_orig.png")
                    x_def = load_image(c["def"], device)
                    with torch.no_grad():
                        edit_def = ip2p.edit(x_def.clamp(0, 1), instruction,
                                             seed=seed, steps=IP2P_STEPS,
                                             s_t=S_T, s_i=S_I)
                    save_png(edit_def, args.out / f"{tag}__edit_def.png")
                    prot = suite.pairwise(edit_orig, edit_def)
                    fid = suite.pairwise(x01, x_def)
                    # 語意類：指令有沒有被做到。三個量問不同的事，都要報。
                    #   `*_input`   原圖對指令的對齊（基準，不可省——不扣掉它
                    #               就分不出「指令做到了」與「原圖本來就像」）
                    #   `*_orig`    未防禦編輯對指令的對齊
                    #   `*_def`     防禦後編輯對指令的對齊
                    #   `*_gain_*`  對齊增益 = 編輯後 − 原圖
                    # 防禦有效時 `gain_def` 應低於 `gain_orig`；`gain_orig` 自己
                    # 太小就代表**未防禦的編輯沒成功**，那一格不可解讀。
                    s_in = suite.semantic(x01, instruction)
                    s_or = suite.semantic(edit_orig, instruction)
                    s_de = suite.semantic(edit_def, instruction)
                    sem = {}
                    for key in ("clip", "siglip"):
                        sem[f"{key}_input"] = round(s_in[key], 4)
                        sem[f"{key}_orig"] = round(s_or[key], 4)
                        sem[f"{key}_def"] = round(s_de[key], 4)
                        sem[f"{key}_gain_orig"] = round(s_or[key] - s_in[key], 4)
                        sem[f"{key}_gain_def"] = round(s_de[key] - s_in[key], 4)
                        sem[f"{key}_gain_drop"] = round(
                            (s_or[key] - s_in[key]) - (s_de[key] - s_in[key]), 4)
                    row = {"image": name, "condition": c["condition"],
                           "instruction": iname, "instruction_text": instruction,
                           "seed": seed, "seconds": round(time.time() - t0, 1),
                           **{f"fid_{k}": v for k, v in fid.items()},
                           **{f"edit_{k}": v for k, v in prot.items()},
                           **sem}
                    rows.append(row)
                    print(f"[reedit] {name} {c['condition']} {iname} s{seed} "
                          f"edit_lpips={prot['lpips']:.4f}", flush=True)
                    with csv_path.open("w", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)
    print(f"\n表：{csv_path}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
