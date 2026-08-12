"""抗淨化折線圖章節。由 `build_report.py` 呼叫。

寫成函式而不是 exec 的片段，理由見 `report_extra_sections.py`。
"""
import json as _json



def render(H, D):
    """把折線圖那一節追加進 `H`。`D` 是 scratchpad 目錄。"""
    _C = _json.load(open(str(D / "purify_charts.json"), encoding="utf-8"))

    H.append('<div class="head"><p class="eyebrow">三之三</p>'
             '<h2>抗淨化：非加性與加性的強度曲線</h2></div>')
    H.append('<p class="measure">橫軸為淨化強度，四個算子家族各一格。'
             '實線為本報告的三個非加性條件，虛線為外部參照的三個加性方法。'
             '縱軸為保留率——以該條件自己不淨化時的位移量為 100%，'
             '故它量的是「淨化把防禦洗掉多少」，與絕對水準是兩件事。</p>')
    H.append('<figure>' + _C["retention_svg"] +
             '<figcaption><span class="t">保留率對淨化強度</span>'
             '<span class="sub">四個家族的形態一致：非加性隨強度上升或持平，'
             '加性下降。量化與 JPEG 兩格的縱軸上界不同，'
             '因為非加性在那裡超過 250%</span></figcaption></figure>')
    H.append('<div class="callout"><b>形態相反，但這不是效果的勝出。</b>'
             '模糊 σ=0.75 上加性掉到 43–52%、非加性維持 110–127%；'
             '雜訊與 JPEG 同一形態。但保留率的分母是各自的起點，'
             '而加性的起點高出 2.5–4 倍——見下方的絕對值圖。</div>')

    H.append('<figure>' + _C["absolute_svg"] +
             '<figcaption><span class="t">位移量對淨化強度（絕對值）</span>'
             '<span class="sub">同一組資料不取比值。加性的曲線即使下降，'
             '在多數設定上仍高於非加性</span></figcaption></figure>')
    H.append('<p class="measure">兩張圖要並排讀。保留率說的是「防禦被洗掉多少」，'
             '絕對值說的是「洗完之後還剩多少」。非加性在前者佔優、在後者落後，'
             '兩者同時成立且不矛盾——它們是同一組數字的兩種正規化。</p>')
