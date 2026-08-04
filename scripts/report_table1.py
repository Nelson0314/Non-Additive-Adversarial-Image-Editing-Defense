"""把任一批 run 整理成 Lo et al. (CVPR 2024) Table 1 的格式。

    python scripts/report_table1.py --out docs/RESULTS_TABLE1.md

判準是該表的五個指標，全部量「免疫後的編輯輸出」與「原始編輯結果」之間的
不相似度。**方向與影像品質相反**：PSNR／SSIM／VIFp／FSIM 越低代表免疫越
成功，LPIPS 越高代表越成功。

為什麼一定要併列 L∞
─────────────────────────────────────────────────────────────────────

Lo 的協定把全部方法固定在 L∞ ≤ 0.06 上比較，那是「匹配失真」的執行方式。
本專案既有的 run 用的是 LPIPS 綁定約束，實際 L∞ 由 0.06 到 0.85 都有——
`e15_S_tau0.05`（空間變形）是 0.5654，即 κ 的 9.4 倍。不併列 L∞ 就會拿
不同預算的數字互相比較，那是無效的比較。

**L∞ 對非加性方法本身也不是好的失真尺**（把一條邊緣移動不到一個像素，
L∞ 可接近 1 而人眼幾乎看不出來），這正是本專案改用 LPIPS 的理由。此處
併列兩者，讓「哪一組數字可以互相比較」在表面上就看得出來，不必回去翻設定。
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from pathlib import Path

# 已知的兩批負對照。它們的設定錯在哪、為什麼不可用於比較，逐批寫死在這裡
# 而不是靠讀者記得——`ours_lo_earlystop` 的每一個數字都比 `ours_lo` 好看
# （PF 的 LPIPS 0.4372 對 0.2593），因為它停在 3.3 倍預算的失真上。
NEGATIVE_CONTROLS = {
    "ours_lo_linfbound": "負對照：綁定約束錯成 L∞（beta_linf 未覆寫），"
                         "見 LEDGER 6.12",
    "ours_lo_earlystop": "負對照：停止準則在約束仍被違反時觸發，"
                         "且 site PF 用了會震盪的 lr 0.03，見 LEDGER 6.13／6.14",
}

# 論文 Table 1 的原始數字。SD v1.4 那三欄。
PUBLISHED = [
    ("[Lo] PhotoGuard encoder attack", 0.06,
     dict(psnr=18.8437, ssim=0.6318, vif_p=0.2118, fsim=0.7757, lpips=0.4131)),
    ("[Lo] PhotoGuard diffusion attack", 0.06,
     dict(psnr=18.2617, ssim=0.6504, vif_p=0.2656, fsim=0.7693, lpips=0.4056)),
    ("[Lo] semantic attack", 0.06,
     dict(psnr=15.1487, ssim=0.4470, vif_p=0.1462, fsim=0.6584, lpips=0.5901)),
]
COLS = ["psnr", "ssim", "vif_p", "fsim", "lpips"]
ARROW = {"psnr": "↓", "ssim": "↓", "vif_p": "↓", "fsim": "↓", "lpips": "↑"}


def run_validity(runs_root: Path, run: str, site: str = "") -> str:
    """這一批 run 的數字能不能拿來比較？回傳一句話，空字串表示可用。

    三種失效，全部由該 run 自己記錄的設定判定，不靠人記得：

    1. **攻擊方沒有 classifier-free guidance**（w < 2）。E26 實測 w = 1 時
       SD v1.4 幾乎不服從 prompt，那批等於在防禦一個不存在的攻擊，
       每一個 `edit_*` 都失效（LEDGER 3.2、7.5）。這一項影響 E2–E23 全部。
    2. **已知的負對照**，見 NEGATIVE_CONTROLS。
    3. **有格子跑滿步數上限**。跑滿代表量到的是「這個步長走到哪裡」而不是
       「該方法在此預算下的能力」，不可用於跨 site 比較（LEDGER 6.4，
       理由見 docs/RESULTS_E13-E23.md §5.4）。

    只讀該 run 目錄裡的檔案。判不出來時回傳空字串而不是猜——把「沒有記錄」
    寫成「可用」與寫成「不可用」都是編造。
    """
    d = runs_root / run
    notes = []
    if run in NEGATIVE_CONTROLS:
        notes.append(NEGATIVE_CONTROLS[run])

    cfg = {}
    has_meta = False
    for fn in ("env.json", "protocol.json"):
        p = d / fn
        if p.exists():
            has_meta = True
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
    w = cfg.get("guidance_scale", cfg.get("guidance"))
    if w is None:
        # results.csv 逐列也記 guidance_scale（lo_* 那幾批只有這一個來源，
        # 它們沒有 env.json）。取第一列即可，同一批 run 的設定不會逐列不同。
        f = d / "results.csv"
        if f.exists():
            with open(f, newline="", encoding="utf-8") as fh:
                first = next(csv.DictReader(fh), {})
            v = first.get("guidance_scale")
            if v not in (None, ""):
                w = float(v)
    if w is not None and float(w) < 2.0:
        notes.append(f"作廢：攻擊方 guidance = {float(w):g}，見 LEDGER 3.2／7.5")
    elif w is None:
        # `guidance_scale` 這個欄位是 E26 修好攻擊端之後才寫進 env.json 與
        # results.csv 的。完全查不到這個值，代表該批跑在修好之前，即隱含的
        # w = 1。這正是 LEDGER 3.17 掃全庫 env.json 時用的判準
        # （`guidance_scale ≥ 2` 的只有三個 run），此處把那次人工掃描變成
        # 表格自己會做的事。
        #
        # 沒有 env.json 的那幾批（e2_phi0、e16_S_disp6.0、e23_*_tau0.10 等，
        # 都是中途中止的實驗）同樣落在這裡，措辭改為「查不到」而非斷言 w=1
        # ——把「沒有記錄」寫成「已知是 1」仍然是編造，即使方向正確。
        src = "env.json 沒有 guidance_scale 欄" if has_meta else "沒有任何設定紀錄"
        notes.append(
            f"查不到攻擊方 guidance（{src}），即跑在 E26 修好攻擊端之前，"
            "隱含 w = 1，見 LEDGER 3.2／3.17／7.5")

    # 步數上限逐 (影像, site) 判定，不整批連坐：`ours_lo / PF` 的四格裡
    # man_00 在 48 步收斂、man_03 在 122 步收斂，另外兩格才跑滿。把整批標成
    # 「跑滿」會抹掉那個差別，而那個差別正是「哪一格能用」的答案。
    s = d / "summary.csv"
    if s.exists() and cfg.get("steps"):
        cap = int(cfg["steps"])
        capped, total = [], 0
        with open(s, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if site and (r.get("attack") or r.get("site") or "") != site:
                    continue
                total += 1
                done = r.get("steps_done")
                if done and int(done) >= cap:
                    capped.append(r.get("image", "?"))
        if capped:
            notes.append(
                f"{len(capped)}/{total} 格跑滿 {cap} 步上限"
                f"（{', '.join(capped)}），該格量到的是「這個步長走到哪裡」"
                "而非該位置的能力，不可用於跨 site 比較（LEDGER 6.4）"
            )
    return "；".join(notes)


def collect(runs_root: Path, pool_prompts: bool = True):
    """讀 runs/*/results.csv 的乾淨攻擊列，回傳逐 (run, site) 的平均。

    乾淨攻擊 = `purify == 'identity'` 且淨化強度為 0。這一列才是「沒有任何
    後處理時的免疫效果」，也就是 Table 1 量的東西；其餘各列是抗淨化，
    Lo 的表沒有那一項。

    `pool_prompts` 會把 `<run>_p1` 併入 `<run>`。論文補充材料 §A 的 Table 1
    是「每個物件兩個編輯 prompt」一起平均，兩個 prompt 分開的數字都不是
    該表的對照對象。兩半的影像數與種子數相同，故直接對所有列取平均，
    與先分半再平均等值。
    """
    groups = {}
    for f in sorted(glob.glob(str(runs_root / "*" / "results.csv"))):
        run = os.path.basename(os.path.dirname(f))
        if pool_prompts:
            run = re.sub(r"_p\d+$", "", run)
        for r in csv.DictReader(open(f, encoding="utf-8")):
            # lo_baseline 的 results.csv 沒有 purify 欄（該協定不含淨化），
            # 其餘 run 有。缺欄視為乾淨攻擊，有欄則必須是 identity + 0。
            if "purify" in r:
                if r["purify"] != "identity" or (r.get("strength") or "0") != "0.0":
                    continue
            key = (run, r.get("attack") or r.get("site") or "")
            vals = {}
            for c in COLS:
                v = r.get(f"edit_{c}")
                vals[c] = float(v) if v not in (None, "") else None
            li = r.get("pert_linf") or r.get("defimg_linf")
            lp = r.get("pert_lpips") or r.get("defimg_lpips")
            vals["_linf"] = float(li) if li else None
            vals["_lpips"] = float(lp) if lp else None
            groups.setdefault(key, []).append(vals)
    out = []
    for (run, site), vs in groups.items():
        row = {"label": f"{run} / {site}" if site else run, "n": len(vs),
               "run": run, "note": run_validity(runs_root, run, site)}
        for c in COLS + ["_linf", "_lpips"]:
            got = [v[c] for v in vs if v[c] is not None]
            row[c] = st.mean(got) if got else None
        out.append(row)
    return out


def c_a_in_prompt(content: str, prompt: str) -> bool:
    """c_a 是否以**完整詞**出現在編輯 prompt 裡。

    必須看詞邊界，不能用子字串：本專案的類別裡 `man` 是 `woman` 的子字串，
    子字串比對會把 man 類的「a woman」判成「c_a 有出現」，方向剛好相反。
    """
    return re.search(rf"\b{re.escape(content.strip())}\b", prompt,
                     flags=re.IGNORECASE) is not None


def by_prompt(runs_root: Path, base: str):
    """把 `base` 與 `base_p*` 的列拆成逐 (攻擊, 編輯 prompt) 的平均。

    存在的理由：semantic attack 的輸入 c_a 由防禦方選，可能出現也可能不出現
    在攻擊方寫的編輯 prompt 裡。論文 §4.3 把「不出現時仍有效」當成優點提出，
    等於承認兩種情況的難度不同。合併後的那張表看不出這件事，但它決定
    semantic 這根基準方法能不能與論文對照，必須單獨列出。兩個 PhotoGuard 變體
    不使用 c_a，其逐 prompt 差異只反映 prompt 本身的編輯幅度。
    """
    groups = {}
    for f in sorted(glob.glob(str(runs_root / "*" / "results.csv"))):
        run = os.path.basename(os.path.dirname(f))
        if re.sub(r"_p\d+$", "", run) != base:
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            key = (r["attack"], c_a_in_prompt(r["content"], r["prompt"]))
            groups.setdefault(key, []).append(
                {c: float(r[f"edit_{c}"]) for c in COLS})
    out = []
    for (atk, in_prompt), vs in sorted(groups.items()):
        row = {"attack": atk, "c_a_in_prompt": in_prompt, "n": len(vs)}
        for c in COLS:
            row[c] = st.mean(v[c] for v in vs)
        out.append(row)
    return out


def render_by_prompt(rows):
    head = ("| 攻擊 | c_a 出現在編輯 prompt 裡 | 情境 | n | "
            + " | ".join(f"{c.upper()} {ARROW[c]}" for c in COLS) + " |")
    lines = [head, "|" + "---|" * (4 + len(COLS))]
    for r in sorted(rows, key=lambda r: (r["attack"], r["c_a_in_prompt"])):
        scen = "改動其他區域（編輯 prompt 2）" if r["c_a_in_prompt"] \
            else "改掉該內容（編輯 prompt 1）"
        lines.append(
            f"| `{r['attack']}` | {'是' if r['c_a_in_prompt'] else '否'} | "
            f"{scen} | {r['n']} | "
            + " | ".join(fmt(r[c], 4 if c != "psnr" else 2) for c in COLS) + " |"
        )
    return "\n".join(lines)


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def render_cells(runs_root: Path, run: str) -> str:
    """逐格列出某一批 run 的 summary.csv，含 `steps_done`。

    上面的表是逐 (run, site) 的平均，而平均會把收斂的格與跑滿上限的格混在
    一起：`ours_lo / PF` 的四格裡 man_00 在 48 步收斂、man_03 在 122 步收斂，
    man_01 與 man_02 跑滿 150。那個平均值（LPIPS 0.2593）因此不對應任何一個
    可用的量測。逐格列出來，讀者才看得到哪一格能用。
    """
    f = runs_root / run / "summary.csv"
    if not f.exists():
        return ""
    cfg = {}
    p = runs_root / run / "protocol.json"
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
    cap = int(cfg.get("steps", 0)) or None

    lines = [
        f"\n## `{run}` 逐格（含步數）\n",
        "`steps_done` 到達上限的格子**不可用於跨 site 比較**：量到的是"
        "「這個步長走到哪裡」而非該位置在此預算下的能力"
        "（LEDGER 6.4，理由見 `docs/RESULTS_E13-E23.md` §5.4）。\n",
        "| 影像 | site | 步數 | 收斂 | 擾動 LPIPS | 擾動 L∞ | "
        "PSNR ↓ | SSIM ↓ | VIF_P ↓ | FSIM ↓ | LPIPS ↑ |",
        "|" + "---|" * 11,
    ]
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            done = int(r["steps_done"])
            ok = "—" if cap is None else ("**否**" if done >= cap else "是")
            lines.append(
                f"| {r['image']} | {r['attack']} | {done} | {ok} | "
                f"{float(r['pert_lpips']):.4f} | {float(r['pert_linf']):.4f} | "
                + " | ".join(
                    f"{float(r['edit_' + c]):.{4 if c != 'psnr' else 2}f}"
                    for c in COLS) + " |"
            )
    return "\n".join(lines) + "\n"


def _body(rows, kappa, with_note):
    ncol = 5 + len(COLS) + (1 if with_note else 0)
    head = "| 方法 | n | L∞ | ×κ | LPIPS(擾動) | " + " | ".join(
        f"{c.upper()} {ARROW[c]}" for c in COLS) + " |"
    if with_note:
        head += " 為什麼不可用 |"
    lines = [head, "|" + "---|" * ncol]
    for r in sorted(rows, key=lambda r: -(r["lpips"] or 0)):
        ratio = f"{r['_linf']/kappa:.1f}×" if r["_linf"] else "—"
        line = (f"| `{r['label']}` | {r['n']} | {fmt(r['_linf'])} | {ratio} | "
                f"{fmt(r['_lpips'])} | "
                + " | ".join(fmt(r[c], 4 if c != "psnr" else 2) for c in COLS)
                + " |")
        if with_note:
            line += f" {r['note']} |"
        lines.append(line)
    return lines, ncol


def render(rows, kappa):
    """分成兩張表：可用的、與不可用的。

    先前這裡只有一張依 LPIPS 排序的表，兩批負對照與整批 w = 1 的舊資料就
    混在同一個排名裡。實際後果看得到：`ours_lo_earlystop / PF` 的 LPIPS
    0.4372 排在 `ours_lo / PF` 的 0.2593 之上，而前者是停在 3.3 倍失真預算
    上的錯誤設定。排名本身就是一種主張，把不可比的列放進同一個排名等於
    在主張它們可比。
    """
    ok = [r for r in rows if not r["note"]]
    bad = [r for r in rows if r["note"]]

    lines, ncol = _body(ok, kappa, with_note=False)
    lines.append("|" + " |" * ncol)
    for label, k, v in PUBLISHED:
        lines.append(
            f"| **{label}** | 150×20 | ≤{k} | 1.0× | — | "
            + " | ".join(fmt(v[c], 4 if c != "psnr" else 2) for c in COLS)
            + " |"
        )
    out = "\n".join(lines)

    if bad:
        blines, _ = _body(bad, kappa, with_note=True)
        out += (
            "\n\n### 不可用於比較的列\n\n"
            "**這些數字是真的，判定不是。** 保留是因為它們是負對照——"
            "知道「設定錯成什麼樣會得到什麼數字」與知道正確設定的數字一樣"
            "重要，而且 `runs/` 是唯一的證據來源、實驗無法重跑。\n\n"
            + "\n".join(blines)
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/RESULTS_TABLE1.md")
    ap.add_argument("--kappa", type=float, default=0.06)
    ap.add_argument("--min_n", type=int, default=1)
    ap.add_argument("--split_base", default="lo_baseline",
                    help="要額外做逐編輯 prompt 分解的 run 名稱（不含 _p1 尾綴）")
    ap.add_argument("--cells", default="ours_lo",
                    help="要逐格列出（含 steps_done）的 run 名稱，逗號分隔")
    args = ap.parse_args()

    rows = [r for r in collect(Path(args.runs)) if r["n"] >= args.min_n]
    if not rows:
        raise SystemExit(f"{args.runs} 底下沒有可判讀的乾淨攻擊列")

    body = render(rows, args.kappa)
    split = by_prompt(Path(args.runs), args.split_base)
    split_md = (
        f"\n## 逐編輯 prompt 的分解（`{args.split_base}`）\n\n"
        "論文補充材料 §A 的每個物件有兩個編輯 prompt，上表是兩者一起平均。\n"
        "此處拆開，因為 semantic attack 的輸入 c_a 由防禦方選定，可能出現、\n"
        "也可能不出現在攻擊方寫的編輯 prompt 裡；論文 §4.3 把「不出現時仍\n"
        "有效」列為額外的優點，等於承認兩種情況難度不同。兩個 PhotoGuard 變體\n"
        "不使用 c_a，其兩列的差異只反映 prompt 本身的編輯幅度，可作為對照。\n\n"
        + render_by_prompt(split) + "\n"
    ) if split else ""
    cells_md = "".join(
        render_cells(Path(args.runs), c.strip())
        for c in args.cells.split(",") if c.strip()
    )
    text = f"""# Table 1 對照：本專案 vs Lo et al. (CVPR 2024)

<!-- 由 scripts/report_table1.py 產生，不要手改 -->

判準為 Lo et al., *Distraction is All You Need*, CVPR 2024, Table 1 的五個
指標，全部量「免疫後的編輯輸出」與「原始編輯結果」之間的不相似度。
箭頭是**免疫成功**的方向，與影像品質的方向相反。

`×κ` 是該列實際的 L∞ 相對論文預算 κ = {args.kappa} 的倍數。**只有 ×κ 接近
1.0 的列可以與論文的數字直接比較**；倍數明顯大於 1 的列是在更大的失真
預算上取得的成績，不構成對照。缺 VIFp／FSIM 的列（顯示為 —）是 2026-08-03
之前跑的，當時 `suite.pairwise` 還沒有這兩項。

{body}
{split_md}{cells_md}
## 讀這張表的三個前提

1. **資料集不同。** 論文是用擴散模型生成的 150 張圖（dog／horse／man，
   每類 2 個編輯 prompt），每格平均 20 個隨機種子；本專案用的是自己的
   影像。跨資料集的絕對值不構成嚴格比較，只有同一列內的相對關係是乾淨的。
2. **論文未公布 strength 與 guidance_scale。** 本專案依 E26 的結論一律用
   guidance_scale = 7.5（w = 1 時 SD v1.4 幾乎不服從 prompt，該設定下
   量不出編輯效果）。
3. **這張表只有第 1 類判準。** 語意軸與感知劣化軸見 `docs/LEDGER.md`；
   三類之間的等級相關為 0.140 / −0.207 / 0.014（n = 217），即這張表的
   高分不預測另外兩類的高分。
"""
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()
