"""併自既有批次報告的兩個章節：淨化保留率與 c_a 抑制的實測。

由 `build_report.py` 呼叫。**寫成函式而不是 exec 的片段**——片段看起來
像模組但沒有自己的命名空間，靜態檢查會把每一個外部名稱判為未定義，
而那些告警與真正的拼字錯誤混在一起就沒有人會看了。
"""
import csv as _csv
import statistics as _st



def render(H, DATA, keys, labels, IMGS):
    """把兩個章節追加進 `H`。參數即這兩節需要的全部外部狀態。"""
    # ── 淨化保留率 ────────────────────────────────────────────────────────
    H.append('<div class="head"><p class="eyebrow">三之二</p>'
             '<h2>保留率：非加性與加性的機制差異</h2></div>')
    H.append('<p class="measure">以不淨化時的位移量為 100%，看各設定下還剩多少。'
             '外部參照的三個加性方法在此呈現相反的形態——那是參數化的差別，'
             '不是本報告三個條件之間的差別。</p>')

    _rows = []
    _all = [("attn", "A · 注意力抑制"), ("target", "B · 目標輸出"),
            ("random", "C · 隨機對照")] + list(DATA["ref_labels"].items())
    for _k, _name in _all:
        _base = DATA["purify"][_k]["identity_0"]
        _cells = "".join(
            '<td class="num">%.0f%%</td>' % (DATA["purify"][_k][key] / _base * 100)
            for key in keys if key in DATA["purify"][_k])
        _dim = ' style="opacity:.65"' if _k in DATA["ref_labels"] else ""
        _rows.append('<tr%s><td style="text-align:left">%s</td>%s</tr>'
                     % (_dim, _name, _cells))
    H.append('<div class="tw"><table><caption>保留率（以不淨化為 100%）</caption>'
             '<thead><tr><th>條件</th>'
             + "".join("<th>%s</th>" % l for l in labels)
             + '</tr></thead><tbody>' + "\n".join(_rows) + '</tbody></table></div>')
    H.append('<div class="callout"><b>兩類參數化在模糊上的形態相反。</b>'
             '三個非加性條件在模糊 0.5 到 0.75 之間維持 100% 以上，'
             '三個加性方法由 84% 掉到 43–52%。去噪器（專為去除加性雜訊設計）'
             '上差距更大：非加性 123–131%、加性 32–43%。'
             '但絕對水準仍是加性領先，故這是機制的差異，不是效果的勝出。</div>')

    # ── 注意力抑制的實測 ──────────────────────────────────────────────────
    H.append('<div class="head"><p class="eyebrow">六之二</p>'
             '<h2>注意力抑制在評測期的實測</h2></div>')
    H.append('<div class="callout warn"><b>兩個量不可互相引用。</b>'
             '前一節的注意力圖，其擷取用的是<b>攻擊方 prompt 的 token span</b>；'
             '而條件 A 的訓練作用在<b>防禦方指名的詞 c_a</b> 上。'
             'c_a 不在攻擊方的 prompt 裡，它的注意力只有在以 c_a 為條件時才有定義。'
             '下表是後者，以獨立的量測取得。</div>')


    def _probe(path):
        d = {}
        for r in _csv.DictReader(open(path, encoding="utf-8")):
            d.setdefault((r["image_id"], r["condition"]), []).append(
                (float(r["rel_to_first_pct"]), r["trained_at_t"] == "True"))
        return d


    _pr = _probe("runs/ca_probe/ca_attention.csv")
    _prb = _probe("runs/ca_probe_base/ca_attention.csv")
    _names = []
    for _d in (_pr, _prb):
        for _, _c in _d:
            if _c not in _names:
                _names.append(_c)

    _rows = []
    for _img in IMGS:
        for _c in _names:
            _v = _pr.get((_img, _c)) or _prb.get((_img, _c))
            if not _v:
                continue
            _near = [a for a, t in _v if t]
            _far = [a for a, t in _v if not t]
            _rows.append(
                '<tr><td style="text-align:left">%s</td>'
                '<td style="text-align:left">%s</td>'
                '<td class="num">%+.1f%%</td><td class="num">%s</td>'
                '<td class="num">%s</td></tr>'
                % (_img, _c, _st.fmean(a for a, _ in _v),
                   ("%+.1f%%" % _st.fmean(_near)) if _near else "—",
                   ("%+.1f%%" % _st.fmean(_far)) if _far else "—"))
    H.append('<div class="tw"><table><caption>遮罩內 c_a 注意力 L1 相對未防禦的變化'
             '（掃 12 個 timestep，同一組遮罩與同一個 c_a 嵌入）</caption>'
             '<thead><tr><th>影像</th><th>條件</th><th>全部 t</th>'
             '<th>訓練施力點附近</th><th>其餘 t</th></tr></thead><tbody>'
             + "\n".join(_rows) + '</tbody></table></div>')
    H.append('<p class="measure">兩件事同時成立。其一，抑制在<b>有施力與沒施力的 '
             'timestep 上幾乎相同</b>（差距 1–3 個百分點），故取樣點數不是瓶頸。'
             '其二，在本報告的失真預算下抑制只有個位數到十幾個百分點，'
             '而在允許約四倍失真的設定上訓練時可達 89–94%——差距來自量級，'
             '不是來自最佳化是否收斂。</p>')
    H.append('<p class="measure">外部參照的加性方法未針對 c_a 最佳化，'
             '卻在部分影像上取得同量級的抑制（horse_03 上 −11.1%），'
             '而其位移量高出約 3 倍。抑制與位移之間因此不是單調對應。</p>')
