"""抗淨化折線圖：橫軸為淨化強度，四個算子家族各一格。

輸出兩份：SVG（供 HTML 內嵌）與 PNG base64（供 Markdown 內嵌）。
"""
import base64
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = Path(os.path.expandvars(
    r"$TEMP\claude\C--WACV-s3\f97b0be2-7c2c-4175-8705-a671a63a1017\scratchpad"))
DATA = json.load(open(D / "report_data.json", encoding="utf-8"))

plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 每個家族：(標題, [(鍵, 顯示強度, 排序值)])。排序值即「越大越強」。
FAM = [
    ("高斯模糊 σ", [("blur_0.25", "0.25", 0.25), ("blur_0.5", "0.5", 0.5),
                    ("blur_0.75", "0.75", 0.75)]),
    ("高斯雜訊 σ", [("noise_0.005", "0.005", 0.005),
                    ("noise_0.01", "0.01", 0.01)]),
    ("量化階數", [("quantize_128", "128", 1), ("quantize_64", "64", 2),
                  ("quantize_32", "32", 3), ("quantize_16", "16", 4)]),
    ("JPEG 品質", [("jpeg_75", "75", 1), ("jpeg_30", "30", 2)]),
]
SERIES = [
    ("attn", "非加性 A · 注意力抑制", "#0E5A61", "-", "o"),
    ("target", "非加性 B · 目標輸出", "#2E8B57", "-", "s"),
    ("random", "非加性 C · 隨機對照", "#5B6472", "-", "^"),
    ("photoguard_c", "加性 · PhotoGuard-c", "#A8231F", "--", "o"),
    ("dia_r", "加性 · DIA-R", "#A5680A", "--", "s"),
    ("mist", "加性 · Mist", "#8A5A9E", "--", "^"),
]


def build(retention: bool):
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9),
                             gridspec_kw={"width_ratios": [3, 2, 4, 2]})
    for ax, (title, pts) in zip(axes, FAM):
        for key, color, ls, mk, label in [(s[0], s[2], s[3], s[4], s[1])
                                          for s in SERIES]:
            base = DATA["purify"][key]["identity_0"]
            xs, ys = [], []
            for pk, _, order in pts:
                if pk not in DATA["purify"][key]:
                    continue
                v = DATA["purify"][key][pk]
                xs.append(order)
                ys.append(v / base * 100 if retention else v)
            ax.plot(xs, ys, color=color, linestyle=ls, marker=mk,
                    markersize=4.5, linewidth=1.9, label=label)
        ax.set_xticks([o for _, _, o in pts])
        ax.set_xticklabels([lab for _, lab, _ in pts], fontsize=10)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        ax.grid(alpha=.25, linewidth=.7)
        ax.tick_params(labelsize=9.5)
        if retention:
            ax.axhline(100, color="#8A93A0", lw=1, ls=":")
        ax.set_xlabel("← 弱　　淨化強度　　強 →", fontsize=9.5, color="#5B6472")
    axes[0].set_ylabel("保留率（%，以不淨化為 100）" if retention
                       else "位移量 edit_lpips", fontsize=10.5)
    # 逐格自動縮放、下界固定為 0。統一上界會把量化那一格（非加性衝到 300%）
    # 切掉，而切掉的正是差異最大的地方。
    for ax in axes:
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9.8,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


out = {}
for name, ret in (("retention", True), ("absolute", False)):
    fig = build(ret)
    s = io.StringIO(); fig.savefig(s, format="svg", bbox_inches="tight")
    out[name + "_svg"] = s.getvalue()[s.getvalue().index("<svg"):]
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=155, bbox_inches="tight")
    out[name + "_png"] = base64.b64encode(b.getvalue()).decode()
    plt.close(fig)
    print(f"{name}: svg {len(out[name+'_svg'])//1024} KB, "
          f"png {len(out[name+'_png'])//1024} KB")

json.dump(out, open(D / "purify_charts.json", "w", encoding="utf-8"))
print("寫入", D / "purify_charts.json")
