"""比對頁與 attention 對照頁 —— `DESIGN` §1.1 的**主判準**產物。

這兩頁的失效形態是「渲染得出來但沒有內容」：`<img>` 指到不存在的檔案、
某一組格點整組漏掉、不適用的格靜默留白。三者都不會有錯誤訊息，而使用者
是靠這一頁下判斷的——先驗實驗「連原始圖片被文字編輯都沒有成功」正是看
`compare.html` 才發現的。

全部在 CPU 上跑，不需要 SD 權重：兩個產生器都只讀 `_cells/*.json` 的紀錄
與其中的相對路徑，不讀影像內容。
"""

import re


from src.experiment.attention_page import build_attention_html
from src.experiment.compare_page import COLUMNS, build_compare_html


def _train(cond="N1", img="dog_00"):
    return {
        "id": f"train/{cond}/{img}", "stage": "train", "status": "done",
        "condition": cond, "image": img, "config": {},
        "artifacts": [f"{cond}/{img}/orig.png", f"{cond}/{img}/x_def.png",
                      f"{cond}/{img}/residual.png"],
    }


def _ray(cond="N1", img="dog_00", tau=0.2):
    return {
        "id": f"rayscale/{cond}/{img}/tau{tau:g}", "stage": "rayscale",
        "status": "done", "condition": cond, "image": img,
        "config": {"tau": tau},
        "artifacts": [f"{cond}/{img}/x_def_tau{tau:g}.png",
                      f"{cond}/{img}/residual_tau{tau:g}.png"],
    }


def _control(img="dog_00", purify=("jpeg", 30.0), seed=0):
    k, s = purify
    d = f"control/{img}/purify/{k}{s:g}"
    return {
        "id": f"control/phi0/{img}/purify{k}{s:g}/seed{seed}",
        "stage": "control", "status": "done", "condition": "phi0",
        "image": img, "config": {"purify": list(purify), "seed": seed},
        "artifacts": [f"{d}/edit_seed{seed}.png",
                      f"{d}/attn/seed{seed}_agg.png",
                      f"{d}/attn/attn_stats.csv"],
    }


def _eval(cond="N1", img="dog_00", tau=0.2, purify=("jpeg", 30.0), seed=0,
          **extra):
    k, s = purify
    d = f"{cond}/{img}/purify/{k}{s:g}"
    cell = {
        "id": f"eval/{cond}/{img}/purify{k}{s:g}/seed{seed}/tau{tau:g}",
        "stage": "eval", "status": "done", "condition": cond, "image": img,
        "config": {"tau": tau, "purify": list(purify), "seed": seed},
        "artifacts": [f"{d}/x_purified.png", f"{d}/edit_seed{seed}.png",
                      f"{d}/attn/tau{tau:g}_seed{seed}_agg.png",
                      f"{d}/attn/tau{tau:g}_seed{seed}_res64_layer00.png",
                      f"{d}/attn/attn_stats.csv"],
        "effect_siglip": 0.031, "retention": 0.82, "retention_usable": True,
        "fid_lpips": 0.2, "fid_psnr": 28.1,
    }
    cell.update(extra)
    return cell


def _batch():
    return [_train(), _ray(), _control(), _eval()]


def _srcs(html):
    return re.findall(r'<img[^>]*src="([^"]+)"', html)


# ---------------------------------------------------------------------------
# compare.html
# ---------------------------------------------------------------------------

def test_一格六張圖全部有來源():
    """六張圖依因果鏈排列：原圖 → 防禦圖 → 殘差 → 淨化後 → 兩側編輯。
    少任何一張，那一格就無法判讀，而頁面照樣渲染得出來。"""
    html = build_compare_html(_batch(), batch="b1")
    assert len(_srcs(html)) == len(COLUMNS) == 6
    assert 'class="miss"' not in html


def test_兩側編輯並排且來自不同目錄():
    """整頁的重點是最後兩張並排——同一淨化、同一種子、同一 prompt，
    唯一差別是防禦有沒有開。兩者若取自同一個檔案，比較恆等於零。"""
    srcs = _srcs(build_compare_html(_batch(), batch="b1"))
    defended, control = srcs[-2], srcs[-1]
    assert defended != control
    assert defended.startswith("N1/") and control.startswith("control/")


def test_缺產物顯示為缺而不是留白():
    cells = [c for c in _batch() if c["stage"] != "control"]
    html = build_compare_html(cells, batch="b1")
    assert 'class="miss"' in html, "對照側缺檔必須看得出來"


def test_帶tau的新產物名解析得到():
    """2026-08-08 起防禦側的 `edit`／`x_purified` 檔名帶 τ。

    before：`edit_seed{k}.png`，四個 τ 寫同一個檔名而互相覆寫；
    after：`edit_tau{τ:g}_seed{k}.png`。`_artifact` 是後綴比對，故頁面那一側
    必須跟著改，否則新批次整片缺圖。
    """
    cells = _batch()
    ev = next(c for c in cells if c["stage"] == "eval")
    d = "N1/dog_00/purify/jpeg30"
    ev["artifacts"] = [f"{d}/x_purified_tau0.2_seed0.png",
                       f"{d}/edit_tau0.2_seed0.png",
                       f"{d}/attn/tau0.2_seed0_agg.png"]
    html = build_compare_html(cells, batch="b1")
    assert 'class="miss"' not in html
    srcs = _srcs(html)
    assert f"{d}/edit_tau0.2_seed0.png" in srcs
    assert f"{d}/x_purified_tau0.2_seed0.png" in srcs


def test_舊批次的無tau產物名仍然解析得到():
    """v14／b3／v14r 的逐格紀錄裡是舊名，而那三批是唯一的證據來源。

    沒有回退路徑的話，替既有批次重新產生報表會得到一整頁缺圖，而那不是
    資料的問題，是改名的副作用。
    """
    html = build_compare_html(_batch(), batch="b1")   # fixture 用的就是舊名
    assert 'class="miss"' not in html
    srcs = _srcs(html)
    assert "N1/dog_00/purify/jpeg30/edit_seed0.png" in srcs
    assert "N1/dog_00/purify/jpeg30/x_purified.png" in srcs


def test_tau的字面格式不影響配對():
    """逐格紀錄讀回的 τ 可能是字串或數。`0.20` 與 `0.2` 比不中的症狀同樣
    是缺圖，而缺圖會被當成「那一格沒跑」。"""
    cells = _batch()
    ev = next(c for c in cells if c["stage"] == "eval")
    ev["config"]["tau"] = "0.20"
    d = "N1/dog_00/purify/jpeg30"
    ev["artifacts"] = [f"{d}/edit_tau0.2_seed0.png"]
    html = build_compare_html(cells, batch="b1")
    assert f"{d}/edit_tau0.2_seed0.png" in _srcs(html)


def test_不適用的格顯示理由而不是消失():
    """`DESIGN` §1.1 要求每一格都要能看到東西，而「這格為什麼沒有圖」
    本身就是要看的資訊。整組消失會讓讀者以為那個 τ 沒有被跑。"""
    cells = _batch() + [{
        "id": "eval/N3/dog_00/purifyjpeg30/seed0/tau0.05", "stage": "eval",
        "status": "skipped", "condition": "N3", "image": "dog_00",
        "config": {"tau": 0.05, "purify": ["jpeg", 30.0], "seed": 0},
        "skipped_reason": "N3 走生成路徑，低於 VAE 重建下限",
    }]
    html = build_compare_html(cells, batch="b1")
    assert "VAE 重建下限" in html and "N3" in html


def test_改寫過的baseline在頁面上被標出():
    """報表把改寫過的 baseline 讀成原論文設定，比不標註更糟。"""
    cells = [_train("advpaint"), _ray("advpaint"), _control(),
             _eval("advpaint", modified_from_paper=True)]
    assert "改寫" in build_compare_html(cells, batch="b1")


def test_retention不可用會被標出():
    cells = [_train(), _ray(), _control(),
             _eval(retention_usable=False)]
    assert "不可用" in build_compare_html(cells, batch="b1")


def test_同一組影像tau淨化下各條件並排():
    """分組必須讓九個條件在其他變因全相同時上下對照——要判的正是
    「非加性在同失真、同淨化下是否勝過加性」。以條件分組會把它們隔開。"""
    cells = []
    for cond in ("N1", "N2", "photoguard_c"):
        cells += [_train(cond), _ray(cond), _eval(cond)]
    cells.append(_control())
    html = build_compare_html(cells, batch="b1")
    assert html.count("<section") == 1, "三個條件應落在同一組內"
    for cond in ("N1", "N2", "photoguard_c"):
        assert f">{cond}<" in html


def test_其餘種子收在details內但仍在頁面上():
    """種子之間的差異是量測噪聲，逐張看的價值低於條件之間的差異；
    但「每一格都必須有影像」不排除任何一格，故不可整批丟掉。"""
    cells = _batch() + [_control(seed=1), _eval(seed=1)]
    html = build_compare_html(cells, batch="b1")
    assert "<details>" in html
    assert "seed 1" in html


def test_全部影像路徑都是相對的():
    """頁面與產物一起搬移之後仍要能看。絕對路徑在別台機器上必定壞掉。"""
    for src in _srcs(build_compare_html(_batch(), batch="b1")):
        assert not src.startswith(("/", "file:", "http")), src
        assert ":" not in src, f"疑似 Windows 絕對路徑：{src}"


def test_惡意字串不會破壞頁面():
    """影像識別碼來自檔名，`<` 未跳脫會讓整頁的結構壞掉。"""
    cells = [_eval(img='a"><script>x</script>')]
    html = build_compare_html(cells, batch="b1")
    assert "<script>" not in html


def test_空批次仍產生可看的頁面():
    html = build_compare_html([], batch="b1")
    assert "compare b1" in html and "主判準" in html


def test_lazy載入():
    """4,050 格 × 6 張圖。沒有 lazy 的話一開啟就要求瀏覽器載入兩萬多張。"""
    html = build_compare_html(_batch(), batch="b1")
    assert html.count('loading="lazy"') == len(_srcs(html))


# ---------------------------------------------------------------------------
# attention.html
# ---------------------------------------------------------------------------

def test_attention頁兩側聚合圖並排():
    html = build_attention_html(_batch(), batch="b1")
    srcs = _srcs(html)
    assert len(srcs) == 2
    assert srcs[0].startswith("N1/") and srcs[1].startswith("control/")


def test_attention頁列出逐層圖與統計表():
    html = build_attention_html(_batch(), batch="b1")
    assert "attn_stats.csv" in html
    assert "逐層圖" in html


def test_attention頁標出哪一組是全層():
    html = build_attention_html([_train(), _ray(), _control(),
                                 _eval(attn_full=True)], batch="b1")
    assert "全層" in html


def test_attention頁寫明亮度不可跨圖比較():
    """`save_heatmap` 對每張圖各自正規化，不同圖的亮度沒有可比性。
    這件事必須寫在頁面上，否則讀者會拿兩張圖的亮度下結論。"""
    html = build_attention_html(_batch(), batch="b1")
    assert "不可直接比較" in html and "attn_stats.csv" in html


def test_attention頁跳過未完成的格():
    cells = [_train(), _ray(), _control(),
             dict(_eval(), status="failed", error="boom")]
    html = build_attention_html(cells, batch="b1")
    assert "boom" not in html


def test_錨點與導覽連結對得上():
    """錨點若被跳脫成 `&quot;`，頁面雖然安全但 `href` 與 `id` 對不上，
    導覽全部失效——而那沒有任何症狀。"""
    cells = [_eval(img='a"><script>x</script>'), _eval(img="dog_00")]
    html = build_compare_html(cells, batch="b1")
    ids = set(re.findall(r'<section id="([^"]+)"', html))
    hrefs = set(re.findall(r'<a href="#([^"]+)"', html))
    assert ids and ids == hrefs
    for a in ids:
        assert re.fullmatch(r"[A-Za-z0-9_-]+", a), a


def test_attention頁的錨點也安全():
    cells = [_control(img='b<img src=x>'), _eval(img='b<img src=x>')]
    html = build_attention_html(cells, batch="b1")
    assert "<img src=x>" not in html
    for a in re.findall(r'<section id="([^"]+)"', html):
        assert re.fullmatch(r"[A-Za-z0-9_-]+", a), a
