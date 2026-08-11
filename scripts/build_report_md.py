"""HackMD 版：純 Markdown，架構圖用 mermaid（HackMD 原生支援）。"""
import csv
import json
import os
from pathlib import Path

D = Path(os.path.expandvars(
    r"$TEMP\claude\C--WACV-s3\f97b0be2-7c2c-4175-8705-a671a63a1017\scratchpad"))
DATA = json.load(open(D / "report_data.json", encoding="utf-8"))
m = DATA["main"]
ARM = [("attn", "A · 注意力抑制"), ("target", "B · 目標輸出"),
       ("random", "C · 隨機對照")]
PUR = [("identity_0", "不淨化"), ("blur_0.25", "模糊 0.25"),
       ("blur_0.5", "模糊 0.5"), ("blur_0.75", "模糊 0.75"),
       ("noise_0.005", "雜訊 0.005"), ("noise_0.01", "雜訊 0.01"),
       ("quantize_128", "量化 128"), ("quantize_64", "量化 64"),
       ("quantize_32", "量化 32"), ("quantize_16", "量化 16"),
       ("jpeg_75", "JPEG 75"), ("jpeg_30", "JPEG 30"),
       ("crop_resize_0.1", "裁切縮放"), ("adverse_cleaner_0", "去噪器"),
       ("diffpure_150", "DiffPure")]
L = []
A = L.append

A("# 非加性抗文字編輯防禦：注意力抑制與目標輸出兩種訓練目標的比較")
A("")
A("> Stable Diffusion v1.4 · 512² · fp32 · strength 0.4 · "
  "失真預算 Δ = 0.04（相對 DISTS） · 3 影像 × 5 種子 × 15 個淨化設定 · 2026-08-11")
A("")
A("在固定的參數化、失真預算與保真約束下，唯一的變因是防禦項 `L_def`。"
  "主指標為**位移量**——防禦後的編輯輸出離未防禦的編輯輸出多遠。")
A("")
A("[TOC]")
A("")
A("## 摘要")
A("")
A(f"- :x: **兩個訓練目標都沒有產生可用的抗編輯效果。**不淨化時位移量："
  f"注意力抑制 `{m['attn']['edit_lpips']:.4f}`、目標輸出 "
  f"`{m['target']['edit_lpips']:.4f}`、隨機對照 `{m['random']['edit_lpips']:.4f}`。"
  f"最佳化相對隨機的增益是 1.56× 與 1.20×，而外部參照的加性方法是 0.24–0.36。")
A(f"- :x: **與評測指標對齊的損失反而更差。**目標輸出直接最小化「離未防禦編輯的"
  f"距離」，即評測所量的東西，其位移量卻低於注意力抑制 14.4%"
  f"（配對 n=225，較優比例 25%）。訓練期的代理編輯鏈（10 步）與評測期的真實"
  f"攻擊鏈（50 步）之間的落差，大於損失形式的差別。")
A("- :heavy_check_mark: **投影式約束使訓練與評測綁在同一個預算上。**"
  "兩個訓練條件的射線縮放係數為 0.938–1.000，即訓練所得的 φ 本身就落在評測的"
  "預算球面上；隨機對照為 0.500–0.875。三者實測的 DISTS 為 "
  "0.0756 / 0.0752 / 0.0752。")
A("")
A("## 一、系統架構")
A("")
A("```mermaid")
A("flowchart TB")
A('  X["原圖 x<br/>512 × 512"] --> G')
A('  subgraph G["防禦圖生成 G(x; φ) · 生成路徑"]')
A('    direction TB')
A('    Z["生成起點<br/>由最佳化求得的 latent z*<br/>使 decode(z*) ≈ x"]')
A('    I["精確反演 BDIA<br/>k = 10 步，模組關閉"]')
A('    U["去噪 10 步 UNet<br/>每步一次噪聲預測"]')
A('    V["解碼器<br/>含逐影像微調的 35,715 個參數"]')
A('    P["φ ∈ ℝ^163840<br/>10 步 × 4×64×64 低秩張量<br/>rank ≤ 32"]')
A('    Z --> I --> U --> V')
A('    P -.注入.-> U')
A('  end')
A('  G --> XD["防禦圖 x_def<br/>失真預算 Δ = 0.04"]')
A('  X -.對照支線 φ = 0.-> CT')
A('  XD --> PU["淨化算子<br/>16 個設定"] --> AT')
A('  CT["未防禦的同一條鏈<br/>同 prompt / seed / 淨化"] --> AT')
A('  AT["攻擊方 SDEdit<br/>stock SD v1.4，50 步<br/>prompt 未知"] --> Y0 & Y1')
A('  Y0["未防禦的編輯 y₀"] --> J')
A('  Y1["防禦後的編輯 y"] --> J')
A('  J["位移量 = LPIPS(y₀, y)<br/>越大代表編輯被推得越遠"]')
A("```")
A("")
A("### 三個條件（`L_def` 之外逐項相同）")
A("")
A("| | `L_def` | 說明 |")
A("|---|---|---|")
A("| **A 注意力抑制** | `‖Att(x_def, c_a) ⊙ M‖₁` | "
  "壓低防禦方指名的詞 c_a 在其對應區域的注意力 |")
A("| **B 目標輸出** | `‖SDEdit(x_def; c_∅) − y_target‖²` | "
  "把代理編輯鏈的輸出推向固定目標影像 |")
A("| **C 隨機對照** | 不最佳化 | 同參數化的高斯方向，縮放至同一個 Δ |")
A("")
A("### 保真約束（三者相同）")
A("")
A("每一步梯度更新之後，把 φ 的方向參數縮放回這個球面：")
A("")
A("$$\\mathrm{metric}\\big(x,\\;G(x;\\varphi)\\big)-"
  "\\mathrm{metric}\\big(x,\\;G(x;0)\\big)=\\Delta$$")
A("")
A("用的度量與 Δ 與評測期逐字相同。`G(x;0)` 是這一格自己的原點——加性方法的"
  "原點恆為 0（未加擾動即原圖），生成路徑的原點是它的重建誤差（實測 DISTS "
  "0.031–0.041），扣掉之後兩類位置比的是同一份增量。")
A("")
A("銳利度與色偏兩項不隨縮放單調變化，無法以縮放保證，故以可行性過濾承擔："
  "只有滿足它們的步才有資格成為最佳步。")
A("")
A("## 二、位移量")
A("")
A("比的是**未防禦的編輯**對**防禦後的編輯**，同一張影像、同一個 prompt、"
  "同一個噪聲種子、同一個淨化算子。")
A("")
A("| 條件 | LPIPS ↑ | PSNR ↓ | SSIM ↓ | VIF_p ↓ | FSIM ↓ |")
A("|---|---|---|---|---|---|")
for k, name in ARM:
    a = m[k]
    A(f"| {name} | **{a['edit_lpips']:.4f}** | {a['edit_psnr']:.2f} | "
      f"{a['edit_ssim']:.4f} | {a['edit_vif_p']:.4f} | {a['edit_fsim']:.4f} |")
A("| *以下為外部參照* | | | | | |")
for k, lab in DATA["ref_labels"].items():
    a = m[k]
    A(f"| *{lab}* | *{a['edit_lpips']:.4f}* | *{a['edit_psnr']:.2f}* | "
      f"*{a['edit_ssim']:.4f}* | *{a['edit_vif_p']:.4f}* | *{a['edit_fsim']:.4f}* |")
A("")
A(":::info\n外部參照使用不同的參數化與不同型態的保真約束，"
  "其數值僅用於判斷絕對水準，不構成受控比較。\n:::")
A("")
A("### 配對比較（同影像、同淨化、同種子；以 A 為基準）")
A("")
A("| 條件 | A 的位移量 | 本條件 | 差 | 本條件較優 | n |")
A("|---|---|---|---|---|---|")
for k, name in (("target", "B · 目標輸出"), ("random", "C · 隨機對照")):
    p = DATA["paired"][k]
    A(f"| {name} | {p['attn']:.4f} | {p['other']:.4f} | "
      f"{(p['other']/p['attn']-1)*100:+.1f}% | {p['win']*100:.0f}% | {p['n']} |")
A("")
A(":::warning\n**兩個訓練目標都只小幅超過隨機方向。**注意力抑制對隨機是 1.56×、"
  "目標輸出是 1.20×。同參數化的隨機擾動在同一個失真預算下已經取得多數的位移。\n:::")
A("")
CH = json.load(open(D / "purify_charts.json", encoding="utf-8"))

A("## 三、抗淨化：非加性與加性的強度曲線")
A("")
A("橫軸為淨化強度，四個算子家族各一格。**實線為非加性、虛線為加性。**"
  "縱軸為保留率——以該條件自己不淨化時的位移量為 100%，"
  "故它量的是「淨化把防禦洗掉多少」，與絕對水準是兩件事。")
A("")
A("![保留率對淨化強度](data:image/png;base64,%s)" % CH["retention_png"])
A("")
A(":::success\n**四個家族的形態一致：非加性隨強度上升或持平，加性下降。**"
  "模糊 σ=0.75 上加性掉到 43–52%、非加性維持 110–127%；量化與 JPEG 上非加性"
  "超過 250%。但保留率的分母是各自的起點，而加性的起點高出 2.5–4 倍。\n:::")
A("")
A("### 同一組資料不取比值")
A("")
A("![位移量對淨化強度](data:image/png;base64,%s)" % CH["absolute_png"])
A("")
A("兩張圖要並排讀。保留率說的是「防禦被洗掉多少」，"
  "絕對值說的是「洗完之後還剩多少」。**非加性在前者佔優、在後者落後，"
  "兩者同時成立且不矛盾**——它們是同一組數字的兩種正規化。")
A("")
A("## 三之二、逐設定的數值")
A("")
A("| 條件 | " + " | ".join(lab for _, lab in PUR) + " |")
A("|---" * (len(PUR) + 1) + "|")
for k, name in ARM:
    A(f"| {name} | " + " | ".join(
        f"{DATA['purify'][k].get(key, float('nan')):.4f}" for key, _ in PUR) + " |")
A("")
A("### 保留率（以不淨化為 100%）")
A("")
A("| 條件 | " + " | ".join(lab for _, lab in PUR) + " |")
A("|---" * (len(PUR) + 1) + "|")
for k, name in ARM + list(DATA["ref_labels"].items()):
    base = DATA["purify"][k]["identity_0"]
    it = " | ".join(f"{DATA['purify'][k][key] / base * 100:.0f}%"
                    for key, _ in PUR if key in DATA["purify"][k])
    A(f"| {name} | {it} |")
A("")
A(":::success\n**兩類參數化在模糊上的形態相反。**三個非加性條件在模糊 0.5 到 "
  "0.75 之間維持 100% 以上，三個加性方法由 84% 掉到 43–52%。"
  "去噪器（專為去除加性雜訊設計）上差距更大：非加性 123–131%、加性 32–43%。"
  "但絕對水準仍是加性領先，故這是機制的差異，不是效果的勝出。\n:::")
A("")
A("## 四、防禦圖的失真")
A("")
A("| 條件 | LPIPS | DISTS | PSNR | 銳利度比 | \\|1−銳利度比\\| | ΔNIQE |")
A("|---|---|---|---|---|---|---|")
for k, name in ARM:
    a = m[k]
    A(f"| {name} | {a['fid_lpips']:.4f} | {a['fid_dists']:.4f} | "
      f"{a['fid_psnr']:.2f} | {a['fid_acutance_ratio']:.3f} | "
      f"{a['acut_dev']:.4f} | {a['dniqe']:+.3f} |")
A("")
A("三個條件在同一個失真預算上，故 DISTS 幾乎相同；差別在失真的**樣式**。"
  "ΔNIQE 為負代表防禦後的編輯輸出品質不比未防禦的差——位移量不是靠把輸出"
  "弄糟換來的。")
A("")
A("## 五、注意力抑制在評測期的實測")
A("")
A(":::danger\n**兩個量不可互相引用。**評測管線的注意力擷取用的是**攻擊方 prompt "
  "的 token span**，而條件 A 的訓練作用在**防禦方指名的詞 c_a** 上。"
  "c_a 不在攻擊方的 prompt 裡，它的注意力只有在以 c_a 為條件時才有定義。"
  "下表是後者，以獨立的量測取得。\n:::")
A("")


def probe(path):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d.setdefault((r["image_id"], r["condition"]), []).append(
            (float(r["rel_to_first_pct"]), r["trained_at_t"] == "True"))
    return d


import statistics as st
pr = probe("runs/ca_probe/ca_attention.csv")
prb = probe("runs/ca_probe_base/ca_attention.csv")
names = []
for d in (pr, prb):
    for _, c in d:
        if c not in names:
            names.append(c)
A("| 影像 | 條件 | 全部 t | 訓練施力點附近 | 其餘 t |")
A("|---|---|---|---|---|")
for img in ("horse_00", "horse_03", "woman_03"):
    for c in names:
        v = pr.get((img, c)) or prb.get((img, c))
        if not v:
            continue
        near = [a for a, t in v if t]
        far = [a for a, t in v if not t]
        A(f"| {img} | {c} | {st.fmean(a for a, _ in v):+.1f}% | "
          f"{('%+.1f%%' % st.fmean(near)) if near else '—'} | "
          f"{('%+.1f%%' % st.fmean(far)) if far else '—'} |")
A("")
A("兩件事同時成立。其一，抑制在**有施力與沒施力的 timestep 上幾乎相同**"
  "（差距 1–3 個百分點），故取樣點數不是瓶頸。其二，在本報告的失真預算下抑制"
  "只有個位數到十幾個百分點，而在允許約四倍失真的設定上訓練時可達 89–94%"
  "——差距來自量級，不是來自最佳化是否收斂。")
A("")
A("外部參照的加性方法未針對 c_a 最佳化，卻在部分影像上取得同量級的抑制"
  "（horse_03 上 −11.1%），而其位移量高出約 3 倍。**抑制與位移之間不是單調對應。**")
A("")
A("## 六、訓練")
A("")
A("| 條件 | 影像 | 步數 | 射線縮放係數 | 秒 | 停止原因 |")
A("|---|---|---|---|---|---|")
for k, name in ARM:
    for img in ("horse_00", "horse_03", "woman_03"):
        e = DATA["train"][k].get(img, {})
        A(f"| {name} | {img} | {e.get('steps', '—')} | "
          f"{e.get('scale_k', float('nan')):.3f} | "
          f"{(e.get('seconds') or 0):.0f} | {e.get('stop', '—')} |")
A("")
A("射線縮放係數為 1 表示訓練所得的 φ 已落在評測的預算球面上，縮放為空操作。")
A("")
A("## 七、適用範圍與限制")
A("")
A("- 樣本為 3 張影像 × 5 個噪聲種子。條件之間的比較是配對的（n = 225），"
  "但跨影像的推論受限於樣本數。")
A("- 本報告的主指標是位移量。**位移量大不等於編輯在語意上失敗**——"
  "先前的量測顯示兩者可以背離，故本報告不由位移量推論編輯是否被導離 prompt。")
A("- 15 個淨化設定之外另有 7 個更強的設定已量測但不納入曲線："
  "在那些設定下未防禦的編輯自身已被毀掉（NIQE 相對不淨化的增幅 ≥ 1.0），"
  "其位移量比的是兩張都已毀掉的圖。原始資料保留。")
A("- 條件 C 不經訓練，其射線縮放係數與另兩者不同（0.500–0.875 對 "
  "0.938–1.000）。三者最終落在同一個失真預算上，但抵達的方式不同。")
A("- **同一個 DISTS 在兩個參數化上，人眼看到的破壞程度不同。**"
  "扣掉各自原點的作法解決了「原點不同」，但沒有解決「同一增量的可見程度不同」。"
  "跨參數化的比較須以此為前提。")
A("")
A("---")
A("")
A("圖片與逐格資料見自足 HTML 版 `report_s3t20.html`。"
  "資料來源：`runs/s3t20_pj_merged`、`runs/s3t20_tj_merged`、"
  "`runs/s3t20_r_merged`、`runs/s3t20_merged`、`runs/ca_probe*`。"
  "全部指標由同一組實作計算，未經挑選。")

out = Path(r"C:\WACV-s3\report_s3t20.md")
out.write_text("\n".join(L), encoding="utf-8")
print(f"寫入 {out}（{out.stat().st_size / 1024:.0f} KB，{len(L)} 行）")
