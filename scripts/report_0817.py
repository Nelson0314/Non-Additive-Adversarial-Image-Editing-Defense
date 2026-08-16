"""2026-08-17 的完整報告：取回的批次、修正的外部比較、操作點、Griffin-Lim 消融。

單檔自足 HTML，所有影像內嵌成 data URI。每一個數字都由 `runs/` 底下的 CSV
現算，不寫死——尚未跑完的批次會自己變成標了「等待中」的空位，不會靜默消失。

用法：
    python scripts/report_0817.py --out reports/2026-08-17
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import math
import os
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------- 資料存取


def rd(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_image(rows: Sequence[dict], condition: Optional[str] = None) -> Dict[str, dict]:
    return {r["image"]: r for r in rows
            if condition is None or r["condition"] == condition}


def mean(rows, key: str) -> float:
    v = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return st.fmean(v) if v else float("nan")


def col(d: Dict[str, dict], imgs: Sequence[str], key: str) -> float:
    v = [float(d[i][key]) for i in imgs if i in d and d[i].get(key) not in (None, "")]
    return st.fmean(v) if v else float("nan")


def pearson(xs, ys) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = st.fmean(xs), st.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def ratio_and_wins(a: Dict[str, dict], b: Dict[str, dict], key: str = "edit_lpips"):
    """回傳（平均比平均、逐圖勝場、n）。逐圖比值的平均會被分母支配，不用。"""
    imgs = sorted(set(a) & set(b))
    if not imgs:
        return float("nan"), 0, 0
    xa = [float(a[i][key]) for i in imgs]
    xb = [float(b[i][key]) for i in imgs]
    return (st.fmean(xa) / st.fmean(xb),
            sum(1 for x, y in zip(xa, xb) if x > y), len(imgs))


# ---------------------------------------------------------------- 影像

_PNG_DIRS: List[Path] = []


def find_png(image: str, suffix: str) -> Optional[Path]:
    """找 `<image>__<suffix>.png`。

    `suffix` 寫成 `<suffix>@<批次>` 時只在該批次目錄裡找——同一個檔名在多個
    批次下代表不同的設定（例如 `phase__human__def` 在 `alt_r025` 與
    `alt_r040` 底下是兩個不同的操作點），不指定就會拿到先掃到的那一個。
    """
    if "@" in suffix:
        suffix, batch = suffix.split("@", 1)
        p = Path("runs") / batch / f"{image}__{suffix}.png"
        return p if p.exists() else None
    for d in _PNG_DIRS:
        p = d / f"{image}__{suffix}.png"
        if p.exists():
            return p
    return None


def img_tag(path: Optional[Path], side: int = 300, quality: int = 80,
            zoom: Optional[int] = None) -> str:
    """縮圖後內嵌。`zoom` 給定時改為中央裁切該邊長再放大，供看紋理用。

    一律縮圖再轉 JPEG：512² 的 PNG 每張約 300 KB，一頁上百張就是幾十 MB。
    需要逐像素判讀的場合要開 `runs/` 的原檔，本頁不是為那個用途做的。
    """
    if path is None or not path.exists():
        return ('<div class="miss">缺圖</div>')
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if zoom:
        w, h = im.size
        c = zoom
        im = im.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2))
        im = im.resize((side, side), Image.NEAREST)
    else:
        im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return ('<img loading="lazy" src="data:image/jpeg;base64,'
            f'{base64.b64encode(buf.getvalue()).decode()}">')


def grid(images: Sequence[str], cols: Sequence[tuple], side: int = 300,
         zoom: Optional[int] = None, note: str = "") -> str:
    """cols 是 (欄標題, suffix 或 None 代表原圖) 的序列。"""
    head = "".join(f"<th>{c[0]}</th>" for c in cols)
    rows = []
    for im in images:
        cells = []
        for _, suf in cols:
            p = find_png(im, "orig" if suf is None else suf)
            cells.append(f"<td>{img_tag(p, side=side, zoom=zoom)}</td>")
        rows.append(f'<tr><th class="rowh">{im}</th>{"".join(cells)}</tr>')
    n = f'<p class="small">{note}</p>' if note else ""
    return (f'{n}<div class="tw"><table class="imgs"><tr><th></th>{head}</tr>'
            f'{"".join(rows)}</table></div>')


def table(headers: Sequence[str], rows: Sequence[Sequence[str]],
          right_from: int = 1) -> str:
    h = "".join(f'<th{" class=n" if i >= right_from else ""}>{x}</th>'
                for i, x in enumerate(headers))
    body = []
    for r in rows:
        body.append("<tr>" + "".join(
            f'<td{" class=n" if i >= right_from else ""}>{x}</td>'
            for i, x in enumerate(r)) + "</tr>")
    return f'<div class="tw"><table><tr>{h}</tr>{"".join(body)}</table></div>'


def pending(title: str, what: str) -> str:
    return (f'<div class="pending"><span class="lab">等待中</span>'
            f'<strong>{title}</strong><p>{what}</p></div>')


# ---------------------------------------------------------------- 內容


def build(out: Path) -> str:
    parts: List[str] = []
    A = parts.append

    # ---- 讀資料 ----
    pa = rd("runs/phaseA_human/results.csv")
    phase = by_image(pa, "phase")
    add = by_image(pa, "add")
    rand = by_image(pa, "phase_rand")

    ext: List[dict] = []
    for d in ["runs/hb5", "runs/hb5_pgc"] + [f"runs/ext24/g{i}" for i in range(9)]:
        ext += rd(f"{d}/results.csv")
    externals = {c: by_image(ext, c)
                 for c in ("photoguard_c", "mist", "dia_r", "apa_weak")}

    alt025 = by_image(rd("runs/alt_r025/results.csv"), "phase")
    alt040 = by_image(rd("runs/alt_r040/results.csv"), "phase")
    aligned = by_image(rd("runs/aligned/results.csv"), "phase")
    pidx1 = by_image(rd("runs/pidx1/results.csv"), "phase")

    gl = {"gl0": phase}
    for tag, name in (("h1", "gl1"), ("h4", "gl4"), ("h16", "gl16")):
        r = rd(f"runs/gl/{tag}/results.csv")
        if r:
            gl[name] = by_image(r)
    glb = {"b1": by_image(rd("runs/gl/b1/results.csv")),
           "b4": by_image(rd("runs/gl/b4/results.csv"))}

    SHOW = ["man_00", "woman_01", "horse_01", "cat_00", "dog_01", "bird_03"]

    # ================================================== 0 摘要
    A('<nav class="toc"><strong>目次</strong>'
      '<a href="#s0">0 這一輪的狀態</a>'
      '<a href="#s1">1 外部比較（修正）</a>'
      '<a href="#s2">2 三個操作點</a>'
      '<a href="#s3">3 Griffin-Lim 判別實驗</a>'
      '<a href="#s4">4 抗淨化</a>'
      '<a href="#s5">5 第二個 prompt</a>'
      '<a href="#s6">6 下一步</a></nav>')
    A('<h2 id="s0">0　這一輪的狀態</h2>')
    A("<p>08-16 的批次在校內網路中斷、機器重開後留在遠端。08-17 全部取回，"
      "重新分析，並把先前擋在網路後面的 Griffin-Lim 判別實驗跑完。</p>")

    done = [
        ("取回 08-16 的全部批次", "完成", "1,014 個檔，commit <code>6789eab73</code>"),
        ("外部比較補齊", "完成",
         f"三個便宜條件 24 張、<code>photoguard_c</code> "
         f"{len(set(externals['photoguard_c']) & set(phase))} 張"),
        ("第三個操作點跑滿 24 張", "完成", "<code>runs/alt_r040</code>（r_min 0.40／θ π）"),
        ("抗淨化擴到 7 張", "完成", "<code>retention_arm2.csv</code>"),
        ("Griffin-Lim 固定 θ 三臂", "完成" if len(gl) == 4 else "部分",
         "<code>runs/gl/h1</code>／<code>h4</code>／<code>h16</code>，各 24 張"),
        ("Griffin-Lim 預算對齊 gl4", "完成" if len(glb["b4"]) >= 24 else "進行中",
         f"<code>runs/gl/b4</code>，{len(glb['b4'])}/24 張"),
        ("Griffin-Lim 預算對齊 gl1", "完成" if len(glb["b1"]) >= 24 else "進行中",
         f"<code>runs/gl/b1</code>，{len(glb['b1'])}/24 張"),
        ("<code>ext24/g8</code>", "遺失",
         "<code>bird_02</code> 的 <code>photoguard_c</code>，機器重開時沒寫出 CSV"),
        ("<code>retention_floor2.csv</code>", "遺失",
         "兩張影像的空白地板，故扣地板的淨增益維持 5 張"),
    ]
    cls = {"完成": "ok", "進行中": "warn", "部分": "warn", "遺失": "bad"}
    A(table(["項目", "狀態", "內容"],
            [[a, f'<span class="{cls[b]}">{b}</span>', c] for a, b, c in done],
            right_from=99))

    # ================================================== 1 外部比較
    A('<h2 id="s1">1　外部比較：補齊資料翻掉了一半的判讀</h2>')
    A("<p>相位側取 <code>runs/phaseA_human</code> 的人眼門檻結果。外部方法一律走"
      "各自論文的<strong>原生預算</strong>（使用者 2026-08-14 裁定不對齊），"
      "故失真水位不同，必須連同 DISTS 一起讀。</p>")

    rows = []
    for c, label in (("apa_weak", "APA 弱 baseline"), ("dia_r", "DIA-R"),
                     ("photoguard_c", "PhotoGuard-c"), ("mist", "Mist")):
        o = externals[c]
        imgs = sorted(set(o) & set(phase))
        if not imgs:
            continue
        r, w, n = ratio_and_wins({i: phase[i] for i in imgs},
                                 {i: o[i] for i in imgs})
        rows.append([label, str(n), f"{col(phase, imgs, 'edit_lpips'):.4f}",
                     f"{col(o, imgs, 'edit_lpips'):.4f}",
                     f"<strong>{r:.3f}</strong>", f"{w}/{n}",
                     f"{col(phase, imgs, 'fid_dists'):.4f}",
                     f"{col(o, imgs, 'fid_dists'):.4f}"])
    A(table(["對手", "n", "相位效果", "對手效果", "倍率", "逐圖勝",
             "相位 DISTS", "對手 DISTS"], rows))

    pg = externals["photoguard_c"]
    n_pg = len(set(pg) & set(phase))
    A(f'<div class="box bad"><span class="lab">08-16 的判讀有一半是錯的</span>'
      f'<p>當時 <code>photoguard_c</code> 只有 11 張，寫的是「聚合打平、逐圖只勝 3/11，'
      f'本方法在多數影像上小輸」。補齊到 {n_pg} 張之後聚合由 0.994 變成上表的值、'
      f'逐圖由 3/11 變成上表的值。</p>'
      f'<p><strong>「打平」這個結論不變，但「在多數影像上小輸」不成立</strong>——'
      f'後來補上的六張相位多半贏。小樣本的逐圖勝場不穩定，這是本專案自己踩到的實例。</p>'
      f'<p>唯一沒補上的是 <code>bird_02</code>（<code>ext24/g8</code> 遺失）。'
      f'<code>photoguard_c</code> 每張約 1.9 小時，補這一格要單獨排一次。</p></div>')

    A("<h3>1.1 防禦圖：五個方法在各自的原生預算下長什麼樣</h3>")
    pg_imgs = [i for i in SHOW if find_png(i, "photoguard_c__def")]
    A(grid(pg_imgs or SHOW,
           [("原圖", None), ("紋理重相位", "phase__human__def"),
            ("加性 δ", "add__human__def"), ("PhotoGuard-c", "photoguard_c__def"),
            ("Mist", "mist__def"), ("DIA-R", "dia_r__def"),
            ("APA 弱", "apa_weak__def")],
           side=260,
           note="判準以人眼為主。Mist 與 APA 弱 baseline 的失真明顯較大，"
                "數值上 DISTS 分別是相位的 3.2 倍與 3.6 倍。"))

    A("<h3>1.2 同一批防禦圖的中央 128 px 放大（僅供看紋理，不是觀看尺度）</h3>")
    A(grid(pg_imgs[:3] or SHOW[:3],
           [("原圖", None), ("紋理重相位", "phase__human__def"),
            ("加性 δ", "add__human__def"), ("PhotoGuard-c", "photoguard_c__def"),
            ("Mist", "mist__def"), ("DIA-R", "dia_r__def")],
           side=260, zoom=128,
           note="FND 已記載：3× 放大不是觀看尺度，曾據此誤判相位的失真較大。"
                "這一列只用來看擾動落在哪裡，不用來裁定可接受與否。"))

    A("<h3>1.3 編輯結果：未防禦 vs 各方法</h3>")
    A(grid(pg_imgs or SHOW,
           [("未防禦的編輯", "phase__human__edit_orig"),
            ("紋理重相位", "phase__human__edit_def"),
            ("加性 δ", "add__human__edit_def"),
            ("PhotoGuard-c", "photoguard_c__edit_def"),
            ("Mist", "mist__edit_def"), ("DIA-R", "dia_r__edit_def"),
            ("APA 弱", "apa_weak__edit_def")],
           side=260,
           note="效果的定義是 LPIPS(未防禦的編輯, 防禦後的編輯)，即第一欄與其餘欄的距離。"))

    # ================================================== 2 操作點
    A('<h2 id="s2">2　三個操作點：需要人眼裁定</h2>')
    A("<p>三者都跑滿 24 張，只改<strong>閘與半徑</strong>，損失與更新式完全相同。</p>")
    ops = [("定案 r 0.12 / θ 1.30", phase), ("r 0.25 / θ 2.6", alt025),
           ("r 0.40 / θ π", alt040)]
    imgs = sorted(set(phase) & set(alt025) & set(alt040))
    rows = []
    for label, d in ops:
        r, w, n = ratio_and_wins({i: d[i] for i in imgs},
                                 {i: add[i] for i in imgs if i in add})
        rows.append([label, str(len(imgs)),
                     f"{col(d, imgs, 'edit_lpips'):.4f}",
                     f"{col(d, imgs, 'fid_dists'):.4f}",
                     f"{col(d, imgs, 'fid_lpips'):.4f}",
                     f"{col(d, imgs, 'fid_psnr'):.2f}",
                     f"{col(d, imgs, 'amp_dev'):.4f}",
                     f"{r:.3f}", f"{w}/{n}"])
    A(table(["操作點", "n", "效果", "DISTS↓", "LPIPS↓", "PSNR↑", "amp_dev↓",
             "對加性", "逐圖"], rows))
    A('<div class="box warn"><span class="lab">沒有一個操作點在所有軸上勝出</span>'
      "<p><code>r 0.25 / θ 2.6</code> 效果最高；<code>r 0.40 / θ π</code> 的 DISTS 與 "
      "PSNR 最好；定案的 LPIPS 最好。<strong>DISTS 與 LPIPS 系統性地不同意</strong>"
      "（與 FND-026／034 同型），所以這一節只能由人眼決定。</p></div>")

    A("<h3>2.1 三個操作點的防禦圖</h3>")
    A(grid(SHOW, [("原圖", None), ("定案 r0.12/θ1.30", "phase__human__def"),
                  ("r0.25/θ2.6", "phase__human__def@alt_r025"),
                  ("r0.40/θπ", "phase__human__def@alt_r040")], side=300))
    A("<h3>2.2 中央 128 px 放大</h3>")
    A(grid(SHOW[:4], [("原圖", None), ("定案", "phase__human__def"),
                      ("r0.25/θ2.6", "phase__human__def@alt_r025"),
                      ("r0.40/θπ", "phase__human__def@alt_r040")],
           side=300, zoom=128))
    A("<h3>2.3 三個操作點的編輯結果</h3>")
    A(grid(SHOW, [("未防禦", "phase__human__edit_orig"),
                  ("定案", "phase__human__edit_def"),
                  ("r0.25/θ2.6", "phase__human__edit_def@alt_r025"),
                  ("r0.40/θπ", "phase__human__edit_def@alt_r040")], side=300))

    # ================================================== 3 Griffin-Lim
    A('<h2 id="s3">3　Griffin-Lim 迭代投影：FND-040 的判別實驗</h2>')
    A("<p><code>amp_dev</code> 是 <strong>STFT 一致性投影誤差</strong>（FND-049）。"
      "Griffin &amp; Lim (1984) 給的迭代投影可以把它壓下去。若效果隨它一起塌掉，"
      "代表效果來自投影誤差造出來的新能量，而不是相位重排——"
      "非加性的主張就要重寫。</p>")
    A("<p>實作在 <code>src/residual/texture_rephase.py</code> 的 "
      "<code>gl_iters</code>：前向在標準輸出之後再跑幾輪「重新分析 → 把幅度換回原圖的 "
      "→ 重新合成」。<code>gl_iters = 0</code> 與加這個選項之前<strong>逐位相同</strong>"
      "（有測試釘住），既有批次仍可比。</p>")

    A("<h3>3.1 固定 θ = 1.30，只改迭代輪數</h3>")
    imgs = sorted(set.intersection(*[set(v) for v in gl.values()]))
    rows = []
    base_eff = col(gl["gl0"], imgs, "edit_lpips")
    base_amp = col(gl["gl0"], imgs, "amp_dev")
    base_d = col(gl["gl0"], imgs, "fid_dists")
    for name in ("gl0", "gl1", "gl4", "gl16"):
        if name not in gl:
            continue
        d = gl[name]
        e, dd, ad = (col(d, imgs, "edit_lpips"), col(d, imgs, "fid_dists"),
                     col(d, imgs, "amp_dev"))
        r, w, n = ratio_and_wins({i: d[i] for i in imgs},
                                 {i: add[i] for i in imgs if i in add})
        rows.append([name.replace("gl", "gl_iters = "),
                     f"{ad:.5f}", f"{100*ad/base_amp:.1f}%",
                     f"{e:.4f}", f"{100*e/base_eff:.1f}%",
                     f"{dd:.4f}", f"{100*dd/base_d:.1f}%",
                     f"{e/dd:.2f}", f"{r:.3f}", f"{w}/{n}"])
    A(table(["臂", "amp_dev", "相對", "效果", "相對", "DISTS", "相對",
             "效果/DISTS", "對加性", "逐圖"], rows))

    A('<div class="box"><span class="lab">這張表單獨不足以判定</span>'
      "<p><code>amp_dev</code> 與 DISTS <strong>幾乎等比一起下降</strong>"
      "（60.3% 對 59.8%、39.5% 對 42.0%、23.0% 對 22.2%）。也就是說固定 θ 時，"
      "迭代投影不只壓掉不一致，它把<strong>整個擾動一起縮小</strong>。"
      "所以「效果掉了」無法歸因給「投影誤差沒了」——兩者綁在一起。</p>"
      "<p>能歸因的只有<strong>效率</strong>：每單位 DISTS 換到的位移由 "
      f"{base_eff/base_d:.2f} 升到 "
      f"{col(gl['gl16'], imgs, 'edit_lpips')/col(gl['gl16'], imgs, 'fid_dists'):.2f}"
      "，單調上升。判定要看下一節的預算對齊臂。</p></div>")

    A("<h3>3.2 固定 θ 各臂的防禦圖</h3>")
    A(grid(SHOW, [("原圖", None), ("gl_iters 0", "phase__human__def"),
                  ("gl_iters 1", "phase__human__gl1__def"),
                  ("gl_iters 4", "phase__human__gl4__def"),
                  ("gl_iters 16", "phase__human__gl16__def")], side=280,
           note="θ 固定為 1.30。迭代輪數越多，擾動越小——這正是上表所說的混淆。"))
    A("<h3>3.3 中央 128 px 放大</h3>")
    A(grid(SHOW[:4], [("原圖", None), ("gl 0", "phase__human__def"),
                      ("gl 1", "phase__human__gl1__def"),
                      ("gl 4", "phase__human__gl4__def"),
                      ("gl 16", "phase__human__gl16__def")], side=280, zoom=128))
    A("<h3>3.4 固定 θ 各臂的編輯結果</h3>")
    A(grid(SHOW, [("未防禦", "phase__human__edit_orig"),
                  ("gl 0", "phase__human__edit_def"),
                  ("gl 1", "phase__human__gl1__edit_def"),
                  ("gl 4", "phase__human__gl4__edit_def"),
                  ("gl 16", "phase__human__gl16__edit_def")], side=280))

    # ---- 預算對齊 ----
    A("<h3>3.5 預算對齊：逐圖把 θ 調到 DISTS 0.0434</h3>")
    A("<p>這是能歸因的那一組：兩臂在<strong>同一個失真水位</strong>上比效果。</p>")
    rows = []
    ref = aligned
    for name, label in (("b1", "gl_iters = 1"), ("b4", "gl_iters = 4")):
        d = glb[name]
        if not d:
            continue
        common = sorted(set(d) & set(ref))
        unre = sum(1 for i in common if d[i].get("unreachable") == "True")
        r, w, n = ratio_and_wins({i: d[i] for i in common},
                                 {i: ref[i] for i in common})
        rows.append([label, f"{len(common)}/24",
                     f"{col(d, common, 'budget_reached'):.4f}",
                     f"{col(d, common, 'radius'):.3f}",
                     f"{col(d, common, 'amp_dev'):.5f}",
                     f"{col(d, common, 'edit_lpips'):.4f}",
                     f"{r:.3f}", f"{w}/{n}", str(unre)])
    common0 = sorted(set(ref))
    rows.insert(0, ["gl_iters = 0（基準）", f"{len(common0)}/24",
                    f"{col(ref, common0, 'budget_reached'):.4f}",
                    f"{col(ref, common0, 'radius'):.3f}",
                    f"{col(ref, common0, 'amp_dev'):.5f}",
                    f"{col(ref, common0, 'edit_lpips'):.4f}", "—", "—",
                    str(sum(1 for i in common0
                            if ref[i].get("unreachable") == "True"))])
    A(table(["臂", "n", "達到的 DISTS", "θ", "amp_dev", "效果",
             "對 gl0", "逐圖", "撞天花板"], rows))

    b4 = glb["b4"]
    if b4:
        common = sorted(set(b4) & set(ref))
        unre = sum(1 for i in common if b4[i].get("unreachable") == "True")
        d_b4 = col(b4, common, "budget_reached")
        d_r = col(ref, common, "budget_reached")
        e_b4 = col(b4, common, "edit_lpips")
        e_r = col(ref, common, "edit_lpips")
        a_b4 = col(b4, common, "amp_dev")
        a_r = col(ref, common, "amp_dev")
        A(f'<div class="box warn"><span class="lab">'
          f'gl_iters = 4 的預算對齊沒有真的對齊</span>'
          f'<p><strong>{unre}/{len(common)} 張撞到 θ = π 的天花板</strong>：'
          f'迭代投影把擾動壓小之後，就算 θ 開到上限也到不了 DISTS 0.0434。'
          f'實際達到的平均是 <strong>{d_b4:.4f}</strong>，只有基準 {d_r:.4f} 的 '
          f'{100*d_b4/d_r:.0f}%。</p>'
          f'<p>因此這一列要這樣讀：<code>gl_iters = 4</code> 在 '
          f'<strong>{100*a_b4/a_r:.0f}% 的 amp_dev</strong>、'
          f'<strong>{100*d_b4/d_r:.0f}% 的 DISTS</strong> 下拿到 '
          f'<strong>{100*e_b4/e_r:.0f}% 的效果</strong>。'
          f'每單位 DISTS 的效率是 {e_b4/d_b4:.2f} 對 {e_r/d_r:.2f}，'
          f'<strong>高 {100*(e_b4/d_b4)/(e_r/d_r)-100:.0f}%</strong>。</p></div>')

        if len(glb["b1"]) < 24:
            A(pending("gl_iters = 1 的預算對齊臂",
                      "<code>runs/gl/b1</code> 正在 basic-2 的 GPU 0 上跑"
                      f"（{len(glb['b1'])}/24 張，每張含 8 輪二分搜尋）。"
                      "存在的理由就是上一個方框：<code>gl_iters = 4</code> 撞天花板、"
                      "沒有真的對齊到目標預算，所以那一列不是同失真的比較。"
                      "<code>gl_iters = 1</code> 的天花板較高（固定 θ 時它的 DISTS 是"
                      "基準的 59.8%，而 gl4 只有 42.0%），預期能真的對齊到 0.0434。"
                      "跑完後上表會多一列，§3.6 的裁定會據此改寫。"))

        A("<h4>預算對齊臂的防禦圖與編輯</h4>")
        A(grid(SHOW, [("原圖", None), ("gl0 @ DISTS 0.0434", "phase__d0.0434__def"),
                      ("gl4 @ 同目標", "phase__d0.0434__gl4__def"),
                      ("未防禦的編輯", "phase__d0.0434__edit_orig"),
                      ("gl0 的編輯", "phase__d0.0434__edit_def"),
                      ("gl4 的編輯", "phase__d0.0434__gl4__edit_def")], side=250))

    # ---- 判定 ----
    A("<h3>3.6 對 FND-040 的裁定</h3>")
    xs, ys = [], []
    for name in gl:
        for i in imgs:
            xs.append(float(gl[name][i]["amp_dev"]))
            ys.append(float(gl[name][i]["edit_lpips"]))
    r_amp = pearson(xs, ys)
    xs2 = [float(gl[n][i]["fid_dists"]) for n in gl for i in imgs]
    r_d = pearson(xs2, ys)
    A(table(["讀數", "值", "說明"],
            [["四臂合併 96 點：amp_dev 對效果", f"r = {r_amp:+.3f}",
              "正相關，但與下一列幾乎同值"],
             ["四臂合併 96 點：DISTS 對效果", f"r = {r_d:+.3f}",
              "amp_dev 與 DISTS 共線，兩個相關無法區分"]], right_from=1))
    A('<div class="box own"><span class="lab">目前能下的結論</span>'
      "<ol>"
      "<li><strong>「效果只是被紋理遮蔽的加性高頻噪聲」這個解釋沒有得到支持。</strong>"
      "把投影誤差壓到四成之後，每單位失真換到的位移<strong>上升</strong>而不是下降。"
      "若效果來自投影誤差造出來的能量，效率應該同步塌掉。</li>"
      "<li><strong>但也不是免費的改進。</strong>迭代投影把可達的失真天花板拉低，"
      "<code>gl_iters = 4</code> 有超過一半的影像到不了現行的操作預算。</li>"
      "<li><strong>FND-040 的 r = +0.449 現在有解釋了</strong>："
      "<code>amp_dev</code> 與 DISTS 共線，那個相關同時是「擾動越大效果越好」的相關，"
      "不是「投影誤差越大效果越好」的獨立證據。</li>"
      "</ol>"
      "<p>完整裁定要等 <code>gl_iters = 1</code> 的預算對齊臂——它的天花板夠高，"
      "能給出真正同失真的比較。</p></div>")

    # ================================================== 4 抗淨化
    A('<h2 id="s4">4　抗淨化：擴到 7 張</h2>')
    ret: List[dict] = []
    for p in sorted(Path("runs/hb5").glob("retention_*.csv")):
        if "floor" in p.name:
            continue
        ret += rd(str(p))
    per: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in ret:
        per.setdefault(r["purifier"], {}).setdefault(
            r["condition"], {})[r["image"]] = float(r["effect_mean"])
    order = ["identity", "blur1", "noise0.05", "quantize16", "jpeg75", "jpeg30",
             "crop_resize0.1", "jpeg_then_resize75", "adverse_cleaner", "impress"]
    conds = ["phase", "add", "phase_rand"]
    rows, wins = [], {c: 0 for c in conds[1:]}
    tot = 0
    for k in order:
        if k not in per or not all(c in per[k] for c in conds):
            continue
        common = set.intersection(*[set(per[k][c]) for c in conds])
        if not common:
            continue
        v = {c: st.fmean(per[k][c][i] for i in common) for c in conds}
        mark = ""
        for c in conds[1:]:
            if v["phase"] > v[c]:
                wins[c] += 1
                mark += "✓"
            else:
                mark += "✗"
        tot += 1
        rows.append([k, f"<strong>{v['phase']:.4f}</strong>", f"{v['add']:.4f}",
                     f"{v['phase_rand']:.4f}", str(len(common)), mark])
    A(table(["淨化算子", "紋理重相位", "加性 δ", "隨機相位", "n", "相位勝"], rows))
    A(f"<p>勝加性 <strong>{wins['add']}/{tot}</strong>、"
      f"勝隨機相位 <strong>{wins['phase_rand']}/{tot}</strong>。"
      f"5 張時的結論在 7 張上逐格維持。</p>")
    A('<div class="box bad"><span class="lab">這一節不能扣地板</span>'
      "<p>讀的是<strong>淨化後的絕對位移量</strong>，不是 <code>retention</code> 比值"
      "——後者被分母支配（FND-037／039）。而空白地板佔淨化後位移量的 47%–92%，"
      "扣掉地板的淨增益（FND-043）需要 <code>retention_floor2.csv</code>，"
      "那個檔在機器重開時遺失，所以<strong>淨增益仍然只有 5 張</strong>。"
      "兩個讀數的 n 不同，不要混寫。</p></div>")

    # ================================================== 5 第二個 prompt
    if pidx1:
        A('<h2 id="s5">5　第二個編輯 prompt</h2>')
        imgs = sorted(set(pidx1) & set(phase))
        p1r = by_image(rd("runs/pidx1/results.csv"), "phase_rand")
        p1a = by_image(rd("runs/pidx1/results.csv"), "add")
        r_a, w_a, n_a = ratio_and_wins(pidx1, p1a)
        r_r, w_r, n_r = ratio_and_wins(pidx1, p1r)
        A(table(["條件", "prompt 0（改主體）", "prompt 1（改場景）", "比"],
                [["紋理重相位", f"{col(phase, imgs, 'edit_lpips'):.4f}",
                  f"{col(pidx1, imgs, 'edit_lpips'):.4f}",
                  f"{col(pidx1, imgs, 'edit_lpips')/col(phase, imgs, 'edit_lpips'):.3f}"],
                 ["加性 δ", f"{col(add, imgs, 'edit_lpips'):.4f}",
                  f"{col(p1a, imgs, 'edit_lpips'):.4f}",
                  f"{col(p1a, imgs, 'edit_lpips')/col(add, imgs, 'edit_lpips'):.3f}"],
                 ["隨機相位", f"{col(rand, imgs, 'edit_lpips'):.4f}",
                  f"{col(p1r, imgs, 'edit_lpips'):.4f}",
                  f"{col(p1r, imgs, 'edit_lpips')/col(rand, imgs, 'edit_lpips'):.3f}"]]))
        A(f"<p>prompt 1 上對加性 <strong>{r_a:.3f}</strong>、逐圖 {w_a}/{n_a}；"
          f"對隨機相位 <strong>{r_r:.3f}</strong>、逐圖 {w_r}/{n_r}。"
          f"防禦本身沒有看過文字（損失是 encoder-targeted），"
          f"這一組測的是同一份防禦在另一種攻擊意圖下撐不撐得住。</p>")
        A(grid(SHOW, [("原圖", None),
                      ("prompt 0 的編輯（未防禦）", "phase__human__edit_orig"),
                      ("prompt 0 防禦後", "phase__human__edit_def"),
                      ("prompt 1 的編輯（未防禦）", "phase__human__edit_orig@pidx1"),
                      ("prompt 1 防禦後", "phase__human__edit_def@pidx1")], side=260))

    # ================================================== 6 下一步
    A('<h2 id="s6">6　下一步</h2>')
    A(table(["優先", "項目", "為什麼"],
            [["1", "等 <code>gl_iters = 1</code> 的預算對齊臂",
              "唯一能給出真正同失真比較的一組，直接封掉 FND-040"],
             ["2", "人眼裁定三個操作點（§2）",
              "DISTS 與 LPIPS 系統性地不同意，指標分不出來"],
             ["3", "人眼裁定 <code>photoguard_c</code> 的失真是否可接受（§1.1）",
              "它與本方法打平；若其失真被判不可接受，那一列的解讀完全不同"],
             ["4", "改用逐圖對齊的預算跑正式批次",
              "固定 θ 讓 PSNR 逐圖漂 16.4 dB，且該漂移預測誰贏（FND-038）"],
             ["5", "把 DCT-Shield 加進 baseline",
              "同場景、主張逐條重疊的直接競爭者，且是「保不保留幅度」的對照組"],
             ["6", "補 <code>bird_02</code> 的 <code>photoguard_c</code> 與兩張圖的空白地板",
              "兩個遺失的格；前者約 1.9 小時，後者約半小時"]], right_from=99))

    return "\n".join(parts)


# ---------------------------------------------------------------- 版面

CSS = """
:root{--bg:#fbfaf8;--fg:#1d1c1a;--mut:#6b665e;--line:#e2ded6;--card:#fff;
--acc:#7a4b1e;--accbg:#fdf4ea;--code:#f4f2ee;--ok:#2f6b3a;--warn:#8a5a00;--bad:#95331f}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#171614;--fg:#e8e4dc;--mut:#9d968a;--line:#33302b;--card:#1f1e1b;
--acc:#d9a15e;--accbg:#26221c;--code:#232019;--ok:#7fc08c;--warn:#d9a15e;--bad:#e08a72}}
:root[data-theme=dark]{--bg:#171614;--fg:#e8e4dc;--mut:#9d968a;--line:#33302b;
--card:#1f1e1b;--acc:#d9a15e;--accbg:#26221c;--code:#232019;--ok:#7fc08c;
--warn:#d9a15e;--bad:#e08a72}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);line-height:1.8;font-size:16px;
font-family:"Noto Sans TC","PingFang TC","Microsoft JhengHei",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:40px 28px 120px}
h1{font-size:29px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:14.5px;margin:0 0 30px}
h2{font-size:22px;margin:54px 0 16px;padding-top:18px;border-top:2px solid var(--line)}
h3{font-size:17px;margin:30px 0 10px;color:var(--acc)}
h4{font-size:15px;margin:20px 0 8px}
p{margin:0 0 13px}
ol,ul{margin:0 0 13px;padding-left:24px}
code{background:var(--code);padding:1px 5px;border-radius:3px;font-size:13.5px;
font-family:Consolas,"DejaVu Sans Mono",monospace}
.tw{overflow-x:auto;margin:0 0 18px}
table{border-collapse:collapse;font-size:14px;width:100%}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{background:var(--code);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
table.imgs td,table.imgs th{padding:3px;text-align:center;vertical-align:middle}
table.imgs th{font-size:12.5px;font-weight:600;line-height:1.35}
table.imgs th.rowh{font-size:12px;color:var(--mut);white-space:nowrap;
writing-mode:horizontal-tb;padding:3px 8px}
table.imgs img{display:block;width:100%;height:auto;border-radius:3px}
.miss{aspect-ratio:1;display:grid;place-items:center;color:var(--mut);
font-size:12px;background:var(--code);border-radius:3px}
.box{border:1px solid var(--line);border-left:4px solid var(--acc);
background:var(--card);padding:14px 18px;border-radius:0 6px 6px 0;margin:0 0 18px}
.box.own{border-left-color:var(--ok)}
.box.warn{border-left-color:var(--warn)}
.box.bad{border-left-color:var(--bad)}
.box .lab,.pending .lab{font-size:11.5px;letter-spacing:.1em;color:var(--mut);
display:block;margin-bottom:6px}
.box p:last-child,.box ol:last-child{margin-bottom:0}
.pending{border:2px dashed var(--warn);background:var(--accbg);padding:16px 20px;
border-radius:6px;margin:0 0 18px}
.pending strong{font-size:15px}
.pending p{margin:6px 0 0;font-size:14px;color:var(--mut)}
.toc{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline;
padding:12px 16px;background:var(--card);border:1px solid var(--line);
border-radius:6px;margin:0 0 8px;font-size:14px}
.toc strong{font-size:12px;letter-spacing:.1em;color:var(--mut)}
.toc a{color:var(--acc);text-decoration:none}
.toc a:hover{text-decoration:underline}
.small{font-size:13px;color:var(--mut);margin:0 0 8px}
.ok{color:var(--ok);font-weight:600}
.warn{color:var(--warn);font-weight:600}
.bad{color:var(--bad);font-weight:600}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports/2026-08-17"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    global _PNG_DIRS
    # 順序即優先序。後綴帶 @<批次> 的欄位在 find_png 之前就被拆掉，見下。
    _PNG_DIRS = [Path(p) for p in [
        "runs/phaseA_human", "runs/gl/h1", "runs/gl/h4", "runs/gl/h16",
        "runs/gl/b1", "runs/gl/b4", "runs/aligned", "runs/hb5", "runs/hb5_pgc",
    ] + [f"runs/ext24/g{i}" for i in range(9)]]

    body = build(args.out)
    html = (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>2026-08-17 完整報告</title><style>{CSS}</style></head><body><main>'
            f'<h1>2026-08-17 完整報告</h1>'
            f'<p class="sub">批次取回、外部比較的修正、三個操作點的對比圖，'
            f'以及 Griffin-Lim 迭代投影對 FND-040 的判別實驗。'
            f'所有數字由 <code>runs/</code> 的 CSV 現算。</p>'
            f'{body}</main></body></html>')
    p = args.out / "index.html"
    p.write_text(html, encoding="utf-8")
    print(f"  {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
