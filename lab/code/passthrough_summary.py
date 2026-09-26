"""穿透分離讀數的彙整：各臂 D／P／D_T、D_seed、淨化底線、D_T 配對差、VQA。

只讀 `lab/results/passthrough/*.csv` 與 `lab/results/fidelity.csv`，CPU 即可。
配對差的區間為影像群集 bootstrap（種子 1729、10,000 次）。新臂要進配對比較時改
檔尾的 PAIRS 與 VQA_ARMS。

    python lab/code/passthrough_summary.py
"""
import csv, statistics as st, random, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
B = str(ROOT / "results/passthrough") + "/"
rd=lambda f: list(csv.DictReader(open(B+f,encoding="utf-8")))
main=rd("passthrough.csv"); seeds={s:rd(f"passthrough_seed{s}.csv") for s in (20260813,20260814,20260815,20260816)}
fid={}
for r in csv.DictReader(open(ROOT / "results/fidelity.csv", encoding="utf-8")): fid[(r["arm"],r["image"])]=float(r["lpips"])
fit={(r["arm"],r["image"]):r for r in rd("fit.csv")}
def boot(vals_by_img, n=10000, seed=1729):
    rng=random.Random(seed); imgs=list(vals_by_img); out=[]
    for _ in range(n):
        s=[x for _ in imgs for x in vals_by_img[rng.choice(imgs)]]; out.append(st.mean(s))
    out.sort(); return out[int(.025*n)], out[int(.975*n)]
arms=[a for a in dict.fromkeys(r["arm"] for r in main)]
print("== 主種子：每臂平均（32 格）  D=現行位移  P=純穿透預測  D_T=扣穿透後")
print(f"{'arm':20s} inLPIPS  D_full  P_full  DT_full [95%CI]        DT_subj  DT_bg   fitdE  extrap")
for a in arms:
    R=[r for r in main if r["arm"]==a]
    f=lambda c: st.mean(float(r[c]) for r in R)
    byimg=collections.defaultdict(list)
    for r in R: byimg[r["image"]].append(float(r["DT_lpips_full"]))
    lo,hi=boot(byimg)
    lp=[fid[(a,i)] for i in byimg if (a,i) in fid]
    print(f"{a:20s} {st.mean(lp) if lp else float('nan'):.3f}   {f('D_lpips_full'):.3f}   {f('P_lpips_full'):.3f}   {f('DT_lpips_full'):.3f} [{lo:.3f},{hi:.3f}]   {f('DT_lpips_subject'):.3f}    {f('DT_lpips_background'):.3f}   {st.mean(float(fit[(a,i)]['fit_deltaE00_mean']) for i in byimg):.2f}   {f('extrapolated_frac'):.3f}")
ss=rd("seed_spread.csv")
print("\n== 換種子的離散（未防禦，同格不同種子兩兩 LPIPS，320 對）")
for reg in ("full","subject","background"): print(f"  Dseed_{reg}: mean {st.mean(float(r['Dseed_lpips_'+reg]) for r in ss):.3f}")
print("\n== 多種子平均（5 種子，每格先平均）")
for a in ("ab_warp","ab_warp_ch","ab_warp_s16","ab_prism","ab_warp_ch_comm","ab_warp_ch_free"):
    cells=collections.defaultdict(list)
    for r in [x for x in main if x["arm"]==a]+[x for s in seeds.values() for x in s if x["arm"]==a]:
        cells[(r["image"],r["prompt_index"])].append((float(r["D_lpips_full"]),float(r["DT_lpips_full"]),float(r["DT_lpips_subject"])))
    if not cells: continue
    m=[tuple(st.mean(v[i] for v in vs) for i in range(3)) for vs in cells.values()]
    print(f"  {a:14s} D {st.mean(x[0] for x in m):.3f}  DT_full {st.mean(x[1] for x in m):.3f}  DT_subj {st.mean(x[2] for x in m):.3f}  (per-seed DT_full: {' '.join(f'{st.mean(float(x['DT_lpips_full']) for x in s if x['arm']==a):.3f}' for s in [main]+list(seeds.values()))})")
bl=rd("passthrough_baseline.csv")
print("\n== 非防禦性改動（淨化後未防禦編輯）")
for p in ("jpeg30","jpeg50","jpeg80","blur1","blur2"):
    R=[r for r in bl if r["purifier"]==p]
    print(f"  {p:7s} inLPIPS {st.mean(float(r['input_lpips']) for r in R):.3f}  D_full {st.mean(float(r['D_lpips_full']) for r in R):.3f}")
print("\n== 用 D_T 重做配對（主種子；多種子臂另列 5 種子平均）")
def pair(a,b,col="DT_lpips_full",src=main):
    A={(r["image"],r["prompt_index"]):float(r[col]) for r in src if r["arm"]==a}
    Bv={(r["image"],r["prompt_index"]):float(r[col]) for r in src if r["arm"]==b}
    ks=[k for k in A if k in Bv]; d=[A[k]-Bv[k] for k in ks]
    if not ks: return
    byimg=collections.defaultdict(list)
    for k in ks: byimg[k[0]].append(A[k]-Bv[k])
    lo,hi=boot(byimg)
    print(f"  {a} − {b} [{col[3:]}]: {st.mean(d):+.4f} [{lo:+.3f},{hi:+.3f}] ({sum(x>0 for x in d)}/{len(d)})")
PAIRS = (("ab_warp_s16","ab_warp"),("ab_warp_s12","ab_warp"),("ab_warp_ch","ab_warp"),("ab_warp","colour_curve_ours"),("ab_warp_s16","colour_curve_ours"),("ab_prism","ab_warp"),("ab_warp_ch_comm","ab_warp_ch_free"))
for a,b in PAIRS:
    pair(a,b); pair(a,b,"DT_lpips_subject")
import glob
v=[r for f in sorted(glob.glob(B+"vqa_*.csv")) for r in csv.DictReader(open(f,encoding="utf-8"))]
print("\n== VQA：物件有出現（yes）的格數")
neg=[r for r in v if r["arm"]=="original"]
print(f"  原圖負例：yes {sum(r['verdict']=='yes' for r in neg)}/{len(neg)}（應為 0）")
VQA_ARMS = ("undefended","ab_warp","ab_warp_ch","ab_warp_s16","ab_prism","ab_warp_ch_comm","ab_warp_ch_free")
for a in VQA_ARMS:
    per=[sum(r['verdict']=='yes' for r in v if r['arm']==a and str(r['seed'])==str(s)) for s in (20260812,20260813,20260814,20260815,20260816)]
    idv=[float(r["id_edit"]) for r in v if r["arm"]==a and r["id_edit"] not in ("",None)]
    if not idv: continue
    print(f"  {a:12s} yes/32 per seed {per}  total {sum(per)}/160  id_edit mean {st.mean(idv):.3f} (n={len(idv)})")
