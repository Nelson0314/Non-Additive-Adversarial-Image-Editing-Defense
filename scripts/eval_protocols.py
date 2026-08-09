#!/usr/bin/env python
"""依文獻的判準分層評測，一次報完三層而不挑一層。

    python scripts/eval_protocols.py --batch runs/v14r_merged \
        --margin runs/v14r_margin.csv --out runs/v14r_protocols

## 為什麼要分層

`reference/SURVEY` §2.2 已經指認六篇 baseline 的判準**沒有交集**。實測之後
問題更具體：用文獻的判準，本專案的每一個條件都「有效」；用「編輯有沒有失敗」
判，全部無效（`RESULTS_2026-08-08` §7.5、§9.6）。兩者不是矛盾，是兩個不同的
問題被同一個詞（「防禦成功」）指稱。

故本腳本把判準分成三層，每一層標明**它對齊哪幾篇**，一次全部輸出。挑一層
報告等於挑一個對自己有利的定義。

| 層 | 問的問題 | 量 | 對齊 |
|---|---|---|---|
| 1 位移 | 防禦把編輯輸出推開多遠 | PSNR↓ SSIM↓ VIF_p↓ FSIM↓ LPIPS↑ MSE↑ | DAYN Table 1（前五項）、DIA（加 MSE）、PhotoGuard、AdvPaint、PromptFlare |
| 2 語意 | 編輯有沒有被導離 prompt | CLIP／SigLIP 對齊度變化；**類別 margin 的判定** | 連續量對齊 PhotoGuard／DIA／PromptFlare／DiffVax 的 CLIP 欄；判定對齊 SIFM 的 ISR 語意分支 |
| 3 感知劣化 | 輸出是不是單純變糟了 | ΔNIQE、銳利度比、RMS | SIFM 的 ISR 劣化分支；PromptFlare 的美學欄（本專案無權重，見下） |

第 1 層是與所有 baseline 逐欄對照的唯一途徑；第 2 層是本專案真正要問的；
**第 3 層的存在是為了讓「靠劣化撐起來的免疫」暴露出來而不是被計入成果**。

## 判準事前宣告（跑之前寫定，不得事後調整）

第 2 層的判定沿用 `class_margin.py`：

    margin(y) = SigLIP(y, 目標類) − SigLIP(y, 原類)
    語意失敗 := margin(y_def) ≤ 0        （防禦後的編輯**不**被判成目標類）

該判定已在三張原圖上校準（margin −0.0713／−0.0575／−0.0613，全部正確判成
原類），且與人眼判定在 dog_03 的四個 seed 上逐格一致。

**ISR 的劣化分支無法照原文重現。** SIFM 的判定由 MLLM 做，本專案沒有那個
judge。故不以單一門檻宣告一個「ISR」數字，改為**掃描門檻並把整條曲線報出來**：

    劣化 := NIQE(y_def) − NIQE(y_ref) ≥ θ      （NIQE 越低越好）
    ISR(θ) := 語意失敗 ∪ 劣化 的比例
    θ ∈ {0.5, 1.0, 2.0, 3.0, ∞}                （∞ 即只看語意分支）

掃描而非取一點，是因為取一點就等於在事後選一個對自己有利的門檻，而 OR 判準
對該門檻極為敏感——這正是本腳本要讓讀者看到的東西。

## 三個算不出來的量，以及為什麼

- **FID、Precision／Recall**（PhotoGuard、Mist、AdvPaint 的主指標）：分布層級
  指標，需要數百張才穩定。本批 3 張影像 × 5 seed = 15 個樣本，算出來的數字
  沒有意義。`SURVEY` §4.2 已記過這一點。
- **Aesthetic Score、PickScore**（PromptFlare）：需要額外的模型權重，未下載。
- **DIA 的背景保留指標**：依賴 PIE-Bench 的編輯 mask 隔離背景。img2img 沒有
  mask，該隔離程序無對應物（`SURVEY` §4.2）。

不算不等於不報——上列三項在輸出的 `unavailable.md` 裡逐項寫明理由，
否則讀者會以為那些欄位是被忽略的。
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import grid                             # noqa: E402

# 第 1 層。值為「免疫成功的方向」，與 `suite.HIGHER_IS_BETTER`（描述影像相似度）
# 相反：此處描述的是免疫效果。
LAYER1 = [
    ("edit_psnr", "PSNR", "lower", "DAYN·DIA"),
    ("edit_ssim", "SSIM", "lower", "DAYN·DIA"),
    ("edit_vif_p", "VIF_p", "lower", "DAYN·PhotoGuard"),
    ("edit_fsim", "FSIM", "lower", "DAYN·PhotoGuard·DiffVax"),
    ("edit_lpips", "LPIPS", "higher", "DAYN·DIA·AdvPaint·PromptFlare"),
    ("edit_mse", "MSE", "higher", "DIA"),
]
LAYER2_CONT = [
    ("effect_clip", "ΔCLIP", "higher", "PhotoGuard·DIA·PromptFlare·DiffVax"),
    ("effect_siglip", "ΔSigLIP", "higher", "本專案（既有主判定）"),
]
LAYER3 = [
    ("d_niqe", "ΔNIQE", "higher", "SIFM 劣化分支"),
    ("edit_acutance_ratio", "銳利度比", "—", "本專案"),
    ("edit_rms", "RMS", "—", "本專案"),
]
THETAS: Sequence[Optional[float]] = (0.5, 1.0, 2.0, 3.0, None)
# 由格點的登記表導出，不再各自寫一份。
#
# 2026-08-09 修正。before：這裡（與 `class_margin.py`／`edit_success_page.py`
# 的 argparse 預設值）各自寫死八個名字，於是新增條件之後判定層會**靜默漏掉
# 它們**——表格看起來完整，只是少了幾列，而那幾列正是本輪要判定的對象。
# 這與 `condition_spec` 的 docstring 講的是同一件事：條件表在兩處各寫一份時，
# 兩份的鍵集合必須由程式保證相同。
CONDS = ["control", *grid.CONDITIONS]


def num(r: Dict[str, str], k: str) -> Optional[float]:
    v = r.get(k)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_grid(p: Path) -> List[Dict[str, str]]:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt(v: Optional[float], nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def mean(vals: List[Optional[float]]) -> Optional[float]:
    got = [v for v in vals if v is not None]
    return st.fmean(got) if got else None


def table(title: str, note: str, spec, rows_by_cond, out: List[str]) -> None:
    out.append(f"\n### {title}\n")
    out.append(note + "\n")
    head = "| 條件 | " + " | ".join(f"{lab} ({d})" for _, lab, d, _ in spec) + " |"
    out.append(head)
    out.append("|" + "---|" * (len(spec) + 1))
    for c in CONDS:
        cells = []
        for key, _, _, _ in spec:
            cells.append(fmt(mean([num(r, key) for r in rows_by_cond.get(c, [])])))
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("對齊的文獻：" + "；".join(
        f"{lab} → {src}" for _, lab, _, src in spec))
    out.append("")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--margin", type=Path, required=True)
    ap.add_argument("--purify", default="identity",
                    help="第 1／3 層取哪個淨化算子；identity 即未淨化")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=None,
                    help="不給時取 grid.csv 裡實際出現的全部 τ")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    grid = load_grid(args.batch / "grid.csv")
    with args.margin.open(newline="", encoding="utf-8") as fh:
        mrows = list(csv.DictReader(fh))

    # ΔNIQE 由兩側的絕對值導出。NIQE 越低越好，故「防禦側比對照側差」是正值。
    for r in grid:
        a, b = num(r, "edit_niqe_a"), num(r, "edit_niqe_b")
        r["d_niqe"] = "" if (a is None or b is None) else f"{b - a}"

    out: List[str] = [
        "# 分層評測：一次報完三層判準",
        "",
        f"批次 `{args.batch.name}`，淨化算子 `{args.purify}`。",
        "",
        "本頁的存在理由：文獻的判準沒有交集，而用不同判準會得到相反的結論。",
        "挑一層報告等於挑一個對自己有利的定義，故三層一次全部列出。",
        "判準的事前宣告見 `scripts/eval_protocols.py` 的 docstring。",
    ]

    # ---- 第 1 層與第 3 層：逐 τ，數值在 grid.csv 每一格都齊全 ----
    #
    # τ 由 `grid.csv` 實際出現的值決定，不寫死。before 的預設是
    # `[0.05, 0.1, 0.2, 0.35]`——第一階段的 τ 軸——於是 s3a（τ_train=0.50）
    # 跑出來的報表**完全沒有主表那一點**，而它照樣印出四張看起來正常的表。
    taus = args.taus
    if taus is None:
        taus = sorted({num(r, "tau") for r in grid
                       if r.get("stage") == "eval"
                       and num(r, "tau") is not None})
    for tau in taus:
        sel = [r for r in grid
               if r.get("stage") == "eval" and r.get("status") == "done"
               and r.get("purify_kind") == args.purify
               and num(r, "tau") is not None
               and abs(num(r, "tau") - tau) < 1e-9]
        if not sel:
            continue
        by = defaultdict(list)
        for r in sel:
            by[r["condition"]].append(r)
        # 對照側（φ=0）在 eval 裡不是一個條件，其位移恆為零，故不列。
        out.append(f"\n---\n\n## τ = {tau:g}（{len(sel)} 格）")
        table("第 1 層 · 位移",
              "比的是**未防禦的編輯**對**防禦後的編輯**。"
              "這是與全部 baseline 逐欄對照的唯一途徑。",
              LAYER1, by, out)
        table("第 2 層 · 語意（連續量）",
              "正值代表防禦把編輯導離 prompt。"
              "**這一層的連續量在本批的變異與訊號同量級**，"
              "見 `RESULTS_2026-08-08` §8。",
              LAYER2_CONT, by, out)
        table("第 3 層 · 感知劣化",
              "ΔNIQE 為正代表防禦後的輸出比未防禦的**更糟**。"
              "此層是代價，不是成果——若某一側的免疫靠劣化撐著，這裡會顯示出來。",
              LAYER3, by, out)

    # ---- 第 2 層的判定與 ISR：**逐 τ 分開**，不可混算 ----
    #
    # 2026-08-09 改。before：本段對整份 margin 檔一次算完、不分 τ。那在舊批次
    # 是對的——四個 τ 的 PNG 互相覆寫，每格本來就只剩一個
    # （`RESULTS_2026-08-08` §10）。`e9a35a5c6` 讓檔名帶上 τ 之後前提消失，
    # 混算等於把不同失真預算的條件放進同一個分數，而 τ 正是本實驗設計的
    # 共同貨幣（`DESIGN` §3.2）。s3a 實測混出來的後果：各條件落在各自不同的
    # τ 上，而表格看起來完全正常。
    mtaus = sorted({m["tau"] for m in mrows if m["tau"]}, key=float)
    out.append("\n---\n\n## 第 2 層 · 語意判定與 ISR（θ 掃描）\n")
    out.append(
        "判定 `語意失敗 := margin(y_def) ≤ 0`。"
        f"margin 檔涵蓋 τ = {', '.join(mtaus)}，**逐 τ 分開列**——"
        "跨 τ 混算等於拿不同失真預算的條件互比。"
        "對照側（φ=0）沒有 τ 這個軸，故它在每一個 τ 的表裡都出現且逐 τ 相同。\n")
    out.append(
        "**ISR 不是原文的重現**：SIFM 的判定由 MLLM 做，本專案沒有那個 judge，"
        "故劣化分支改用 ΔNIQE 並**掃描門檻**。取單一門檻等於事後選一個有利的值，"
        "而 OR 判準對該門檻極為敏感——那正是要讓讀者看到的事。\n")

    # margin 與 grid 的 ΔNIQE 依 (條件, 影像, seed) 對接。
    dn = {}
    for r in grid:
        if (r.get("purify_kind") == args.purify and r.get("stage") == "eval"
                and r.get("status") == "done"):
            key = (r["condition"], r["image_id"], r["seed"], r["tau"])
            dn[key] = num(r, "d_niqe")

    labels = ["語意失敗率"] + [
        ("ISR 僅語意" if th is None else f"ISR θ={th:g}") for th in THETAS]
    for tau in mtaus:
        out.append(f"\n### τ = {tau}\n")
        out.append("| 條件 | " + " | ".join(labels) + " |")
        out.append("|" + "---|" * (len(labels) + 1))
        for c in CONDS:
            sem, isr = [], {th: [] for th in THETAS}
            for m in mrows:
                if m["condition"] != c or m["purify_dir"] != f"{args.purify}_0":
                    continue
                # 對照側沒有 τ，在每一個 τ 的表裡都算進去（它是共用分母）
                if m["tau"] and m["tau"] != tau:
                    continue
                marg = float(m["margin"])
                fail = marg <= 0
                sem.append(fail)
                d = dn.get((c, m["image_id"], m["seed"], m["tau"]))
                for th in THETAS:
                    if th is None:
                        isr[th].append(fail)
                    else:
                        isr[th].append(fail or (d is not None and d >= th))
            if not sem:
                continue
            cells = [f"{sum(sem)}/{len(sem)}"]
            for th in THETAS:
                v = isr[th]
                cells.append(f"{sum(v)}/{len(v)}")
            out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")
    out.append(
        "`control` 是未防禦的對照，其「語意失敗」即**攻擊本身失敗**的比例，"
        "是全部條件的分母。任何條件要算防禦有效，該欄必須**高於** control。\n\n"
        "`control` 逐 θ 不變是正確的：對照側就是 `y_ref` 本身，"
        "ΔNIQE 依定義為零，故 OR 的劣化分支對它永遠不成立。")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocols.md").write_text("\n".join(out), encoding="utf-8")

    (args.out / "unavailable.md").write_text("""# 文獻指標中本批算不出來的，以及為什麼

不算不等於忽略。逐項寫明理由，否則讀者會以為這些欄位是被跳過的。

| 指標 | 出自 | 為什麼算不出來 |
|---|---|---|
| FID、Precision／Recall | PhotoGuard-c、Mist、AdvPaint 的主指標 | 分布層級指標，需要數百張才穩定。本批 3 張影像 × 5 seed = 15 個樣本，算出來沒有意義（`reference/SURVEY` §4.2） |
| Aesthetic Score、PickScore | PromptFlare | 需要額外模型權重，未下載 |
| 背景保留（mask 隔離的 PSNR／LPIPS／MSE／SSIM） | DIA | 依賴 PIE-Bench 的編輯 mask。img2img 沒有 mask，該隔離程序無對應物（`SURVEY` §4.2） |
| ISR（原文形式） | SIFM | 判定由 MLLM 做，本專案沒有該 judge。已改為 ΔNIQE 門檻掃描的代理，**不是重現** |
| 人類排名 | DiffVax（67 名受試者） | 需要受試者招募 |

另有兩項是**可算但本批不具鑑別力**的：

- `cnn_denoise_substitute` 淨化算子缺權重，五個算子中有一個從頭到尾沒有資料
  （`RESULTS_2026-08-08` §7.1）。
- `cat_02` 在全部條件下的攻擊都成功（5/5），該影像不提供訊息
  （`RESULTS_2026-08-08` §9.6）。
""", encoding="utf-8")

    print(f"寫入 {args.out / 'protocols.md'}")
    print(f"寫入 {args.out / 'unavailable.md'}")


if __name__ == "__main__":
    main()
