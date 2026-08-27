"""8×8 DCT 域「保長旋轉」的失真天花板：在給定的旋轉角上界下，它到得了失真帶嗎。

要回答什麼
────────────────────────────────────────────────────────────────────
本方法現在的相位擾動走的是「全域切窗 STFT → 旋轉相位 → 反轉回像素 → 再走一次
JPEG」。提案是把同一件事直接做在 JPEG 自己的 8×8 DCT 係數上。DCT 係數是實數、
沒有現成的相位，候選的對應物是把**同一個區塊內的兩格係數當成平面向量做旋轉**
——旋轉保長，正是「保幅度、改相位」在逐區塊意義下的類比。

旋轉的失真是封閉式的。8×8 DCT 是正交歸一（`jpeg_codec.dct_matrix` 的
docstring 已證），區塊之間不重疊，所以把一對係數 `c = (c_i, c_j)` 轉 `theta`
之後，**該區塊像素值的 L2 變化恰為**

    ‖Δ‖₂ = ‖R(theta)c − c‖₂ = 2‖c‖₂·sin(theta/2)

沒有重疊相加的抵銷、也沒有 STFT 一致性投影誤差。於是「最極端的解」可以直接
構造出來：把**每一對合格的係數都轉到角度上界**。那個失真就是這個參數化在該
上界下的天花板，與跑幾步最佳化無關。

這支腳本量的就是那個天花板，模式與 `scripts/advdrop_ceiling.py` 相同
（該檔用同一個手法證明 AdvDrop 進不了失真帶，限制在機制不在步數）。

三個變體，差別在旋轉發生在量化的哪一側
────────────────────────────────────────────────────────────────────
| 變體 | 旋轉作用在 | 交付什麼 | 為什麼要量它 |
|---|---|---|---|
| `pre_float` | 未量化的浮點 DCT 係數 | 浮點影像（不壓縮） | 參數化本身的絕對天花板 |
| `pre_deliver` | 未量化的浮點 DCT 係數 | 品質 QD 的 JPEG | 量化把天花板砍掉多少 |
| `post_int` | **量化後的整數係數**，旋轉後再四捨五入 | 品質 QD 的 JPEG | DCT-Shield 的動作空間；交付即參數 |

`post_int` 是提案真正想要的那一格：交出去的東西就是我們挑的那組整數係數，
攻擊方以相同或更高品質重壓接近恆等（`runs/ip2p_deliver_jpeg/README.md` 量到
同品質重壓只改 0.02–0.17% 的係數）。但整數格點也可能讓旋轉無處可去——量化後
的 AC 係數大多是 0，而**轉動一對零向量得到的還是零向量**。這正是要先量的事。

配對規則
────────────────────────────────────────────────────────────────────
預設 `transpose`：把 `(u, v)` 與 `(v, u)` 配成一對（`u < v`，共 28 對，
對角線 8 格與 DC 不參與）。理由是**保長只有在兩軸的價錢相同時才有感知意義**：
轉置對的徑向頻率相同、JPEG 亮度量化表上的階距幾乎相同（例如 base[0][1]=11、
base[1][0]=12），旋轉交換的是「橫紋 vs 直紋」的方向，不是能量的大小或所在的
頻帶。對照組 `zigzag` 把 zigzag 序上相鄰的兩格配對——那兩格的頻率與價錢都不
同，旋轉會把能量搬過頻帶，「保長」在感知上不成立，量它是為了讓上面那句話
變成可判定的，而不是一個好聽的說法。

判讀規則（跑之前就寫下，不是看到數字才定）
────────────────────────────────────────────────────────────────────
K1  `post_int` 在 theta = pi（旋轉能造出的最極端解）下的十張平均 DISTS
    **構不到 0.1286**（本方法工作點失真帶的下界，`runs/ip2p_mainline`），
    這條路在整數格點上是死的——與 AdvDrop 同一種死法，限制在機制不在步數。
    此時只剩 `pre_float` 那條路，而它必須再走一次 JPEG 才交付得出去，
    也就是退回現行「量化交付」的處境，沒有拿到新性質。
K2  帶上紋理閘與徑向帶通之後（`--gate texture`）天花板掉出失真帶，代表閘與
    參數化互斥：合格的係數都在量化已經清空的地方。此時提案要嘛放棄閘，
    要嘛承認需要加性項——後者會讓「非加性重參數化」的主張再退一步
    （`docs/DECISIONS.md` 已為 `--spectral-floor` 付過一次這個代價）。
K3  `zero_pair_frac`（合格的對裡兩格量化值都是 0 的比例）在交付品質上
    **超過 0.90**，則容量問題與現行方法在平坦區的死法同型，不是新機制。
K4  `delta_within_1`（旋轉造成的整數位移 |δ| ≤ 1 的比例）**超過 0.95** 時，
    `post_int` 的動作空間是 DCT-Shield 的 ε=1 球的**子集**。子集本身不是
    缺點（約束正是提案的內容），但此時新穎性不能建立在「動作不同」上，
    只能建立在「約束不同」上，論文的寫法必須跟著改。

用法
────────────────────────────────────────────────────────────────────
    python scripts/dct_rotation_ceiling.py --out runs/dct_phase_design/ceiling.csv

**純 CPU、不需要擴散模型、不需要最佳化。** 只有 DCT、旋轉、量化、IDCT 與指標。
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.baselines.jpeg_codec import (  # noqa: E402
    CHANNEL_NAMES, block_dct, block_idct, dct_matrix, jpeg_decode, jpeg_encode,
    quant_table, rgb_to_ycbcr, subsample_420, upsample_420, ycbcr_to_rgb,
    normalize_quality,
)
from src.metrics.suite import MetricSuite  # noqa: E402
from src.residual.texture_rephase import pixel_texture_mask  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512
# 本方法工作點的失真帶（`runs/ip2p_mainline`、`docs/RESULTS.md`）。
BAND_LO, BAND_HI = 0.1286, 0.1447
# 七軸比較裡本方法更低的那一點 `ours_ph_q`。
OURS_PH_Q_DISTS = 0.0928
THETAS = (0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2, 2.0, math.pi)


def zigzag_order() -> List[Tuple[int, int]]:
    """JPEG 的 zigzag 掃描序（ITU-T T.81 Figure 5），(u, v) = (列, 行)。"""
    cells = sorted(((u, v) for u in range(8) for v in range(8)),
                   key=lambda c: (c[0] + c[1],
                                  c[1] if (c[0] + c[1]) % 2 == 0 else -c[1]))
    return cells


def build_pairs(rule: str, r_min: float) -> List[Tuple[Tuple[int, int],
                                                       Tuple[int, int]]]:
    """回傳 [( (u1,v1), (u2,v2) ), ...]。DC 一律排除。

    徑向座標取 `r = sqrt(u² + v²) / 8`，與 `texture_rephase.radial_gate` 的
    尺度一致（該處 Nyquist 歸一到 1）。**8×8 的格點上最小的非零半徑是
    0.125**，所以 `r_min = 0.12` 在這個格點上只排除 DC——徑向閘在 DCT 域的
    解析度只有 8 階，這是與 32×32 rfft2 格點的一個實際差別，不是實作疏漏。

    DC 排除的理由不只是半徑：DC 是區塊的平均亮度，動它等於整塊變亮，
    `src/baselines/dct_shield.py` 已記載那正是 DCT-Shield 平坦區方格的來源。
    """
    def radius(c):
        return math.sqrt(c[0] ** 2 + c[1] ** 2) / 8.0

    if rule == "transpose":
        pairs = [((u, v), (v, u)) for u in range(8) for v in range(u + 1, 8)]
    elif rule == "zigzag":
        order = [c for c in zigzag_order() if c != (0, 0)]
        pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
    else:
        raise ValueError(f"未知的配對規則 {rule!r}，可用的是 transpose／zigzag")
    out = []
    for a, b in pairs:
        if a == (0, 0) or b == (0, 0):
            continue
        if radius(a) < r_min or radius(b) < r_min:
            continue
        out.append((a, b))
    return out


def rotate_pairs(coef: torch.Tensor, pairs, theta: torch.Tensor,
                 gate: torch.Tensor) -> torch.Tensor:
    """對 (N, hb, wb, 8, 8) 的係數，把每一對做平面旋轉。

    `theta` 與 `coef` 的前三維廣播（逐區塊的角度，含正負號樣式）；`gate` 是
    (N, hb, wb) 的逐區塊閘，乘在角度上——與本方法的作法同型（閘決定擾動被
    允許出現在哪裡，不改變更新規則）。
    """
    out = coef.clone()
    ang = theta * gate
    cos, sin = torch.cos(ang), torch.sin(ang)
    for (u1, v1), (u2, v2) in pairs:
        a = coef[..., u1, v1]
        b = coef[..., u2, v2]
        out[..., u1, v1] = cos * a - sin * b
        out[..., u2, v2] = sin * a + cos * b
    return out


def sign_pattern(shape, mode: str, seed: int, device, dtype) -> torch.Tensor:
    """逐區塊的角度正負號。L2 失真與正負號無關（`2‖c‖sin(θ/2)` 只看 |θ|），
    但 DISTS 量的是紋理統計量，同調與隨機的樣式未必同價，故兩者都量。"""
    if mode == "coherent":
        return torch.ones(shape, device=device, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, shape, generator=g).to(device=device, dtype=dtype)
    return s * 2.0 - 1.0


def block_texture_gate(x01: torch.Tensor, hb: int, wb: int,
                       sigma: float, edge_power: float) -> torch.Tensor:
    """把本方法的紋理閘搬到**編解碼器自己的**區塊格點上，形狀 (N, hb, wb)。

    直接呼叫 `texture_rephase.texture_gate` 不行：它用 `block_mean`，會先
    reflect padding `block//2` 再以 `hop` 展開，512 上得到 65×65 個重疊視窗，
    與 `jpeg_encode` 的 64×64 個**不重疊**區塊對不齊，相乘之後錯位而且沒有
    症狀。故走 `pixel_texture_mask`（同一條公式，只是把區塊平均換成高斯平滑）
    再以 `avg_pool2d` 落到編解碼器的格點上。色度平面因 4:2:0 邊長減半，
    池化核跟著加倍。
    """
    m = pixel_texture_mask(x01, sigma=sigma, energy_quantile=0.0,
                           edge_power=edge_power)          # (N,1,H,W)
    k = x01.shape[-1] // wb
    return F.avg_pool2d(m, kernel_size=k, stride=k).squeeze(1)


def to_planes(x01: torch.Tensor):
    """`[0,1]` RGB → JPEG 的三個平面（已 level shift）。與 `jpeg_encode` 同路。"""
    ycc = rgb_to_ycbcr(x01 * 255.0)
    return [ycc[:, 0:1] - 128.0,
            subsample_420(ycc[:, 1:2]) - 128.0,
            subsample_420(ycc[:, 2:3]) - 128.0]


def from_planes(planes) -> torch.Tensor:
    ycc = torch.cat([planes[0] + 128.0,
                     upsample_420(planes[1] + 128.0),
                     upsample_420(planes[2] + 128.0)], dim=1)
    return (ycbcr_to_rgb(ycc) / 255.0).clamp(0.0, 1.0)


def gates_for(x01: torch.Tensor, coefs: Dict[str, torch.Tensor], mode: str,
              sigma: float, edge_power: float) -> Dict[str, torch.Tensor]:
    out = {}
    for name, c in coefs.items():
        hb, wb = c.shape[1], c.shape[2]
        if mode == "band":
            out[name] = torch.ones(c.shape[0], hb, wb, device=c.device,
                                   dtype=c.dtype)
        elif mode == "texture":
            out[name] = block_texture_gate(x01, hb, wb, sigma, edge_power)
        else:
            raise ValueError(f"未知的閘 {mode!r}，可用的是 band／texture")
    return out


def variant_pre_float(x01, pairs, theta, gates, sign, qd, d):
    planes = to_planes(x01)
    outs = []
    for i, name in enumerate(CHANNEL_NAMES):
        c = block_dct(planes[i], d)
        r = rotate_pairs(c, pairs, theta * sign[name], gates[name])
        outs.append(block_idct(r, d))
    return from_planes(outs), {}


def variant_pre_deliver(x01, pairs, theta, gates, sign, qd, d):
    q = normalize_quality(qd)
    planes = to_planes(x01)
    coef = {}
    for i, name in enumerate(CHANNEL_NAMES):
        tbl = quant_table(q, chroma=(name != "Y"), device=x01.device,
                          dtype=x01.dtype)
        c = block_dct(planes[i], d)
        r = rotate_pairs(c, pairs, theta * sign[name], gates[name])
        coef[name] = torch.round(r / tbl)
    return jpeg_decode(coef, q), {}


def variant_post_int(x01, pairs, theta, gates, sign, qd, d):
    """量化後旋轉，再四捨五入回整數。**交付的就是這組整數係數的解碼結果。**

    旋轉結果不是整數，必須再取整——於是實際交出去的動作是整數位移
    `δ = round(R(θ)α) − α`，也就是 DCT-Shield 的動作空間裡的一個受約束子集。
    `delta_within_1` 這一欄量的就是那個子集有多小。
    """
    q = normalize_quality(qd)
    alpha = jpeg_encode(x01, qd)
    coef, stats = {}, {}
    n_in_pair = n_zero = n_small = n_tot = 0
    for name in CHANNEL_NAMES:
        a = alpha[name]
        r = rotate_pairs(a, pairs, theta * sign[name], gates[name])
        rr = torch.round(r)
        coef[name] = rr
        delta = rr - a
        for (u1, v1), (u2, v2) in pairs:
            p = torch.stack([a[..., u1, v1], a[..., u2, v2]], dim=-1)
            n_zero += int((p.abs().sum(dim=-1) == 0).sum())
            n_in_pair += p[..., 0].numel()
            dd = torch.stack([delta[..., u1, v1], delta[..., u2, v2]], dim=-1)
            n_small += int((dd.abs() <= 1).sum())
            n_tot += dd.numel()
    stats["zero_pair_frac"] = n_zero / max(1, n_in_pair)
    stats["delta_within_1"] = n_small / max(1, n_tot)
    return jpeg_decode(coef, q), stats


VARIANTS = {"pre_float": variant_pre_float,
            "pre_deliver": variant_pre_deliver,
            "post_int": variant_post_int}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--images", type=Path,
                    default=Path("runs/ip2p_fair_comparison/images10.txt"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--qd", type=float, default=0.85,
                    help="交付品質，與 runs/ip2p_deliver_jpeg 的主線一致")
    ap.add_argument("--pairing", default="transpose",
                    choices=["transpose", "zigzag"])
    ap.add_argument("--r-min", type=float, default=0.12)
    ap.add_argument("--gates", nargs="+", default=["band", "texture"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--signs", nargs="+", default=["random", "coherent"])
    ap.add_argument("--thetas", type=float, nargs="+", default=list(THETAS))
    ap.add_argument("--gate-sigma", type=float, default=2.0,
                    help="紋理閘的高斯 σ（像素）。要能分辨 8×8 之內的內容")
    ap.add_argument("--gate-edge-power", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in
             args.images.read_text(encoding="utf-8").splitlines()
             if ln.strip()][:args.n]
    # **一律 CPU**：這是上機前的探針，不佔共用的卡。
    device = torch.device("cpu")
    suite = MetricSuite(device=device)
    d = dct_matrix(device, torch.float32)
    pairs = build_pairs(args.pairing, args.r_min)
    print(f"配對規則 {args.pairing}：{len(pairs)} 對／區塊"
          f"（共 64 格，DC 與徑向 < {args.r_min} 已排除）", flush=True)

    rows: List[Dict] = []
    for idx, name in enumerate(names):
        hits = sorted((args.data / name).glob("*.png")) + \
            sorted((args.data / name).glob("*.jpg"))
        if not hits:
            raise SystemExit(f"{args.data / name} 下沒有影像")
        x = load_image_tensor(hits[0], device, size=RESOLUTION).clamp(0, 1)
        alpha = jpeg_encode(x, args.qd)
        base = {k: torch.ones(v.shape[0], v.shape[1], v.shape[2])
                for k, v in alpha.items()}
        sign = {}
        for smode in args.signs:
            sign[smode] = {k: sign_pattern(tuple(base[k].shape), smode,
                                           seed=1000 + idx, device=device,
                                           dtype=torch.float32)
                           for k in alpha}
        for gmode in args.gates:
            gates = gates_for(x, alpha, gmode, args.gate_sigma,
                              args.gate_edge_power)
            gate_mean = float(statistics.fmean(
                float(g.mean()) for g in gates.values()))
            for vname in args.variants:
                fn = VARIANTS[vname]
                for smode in args.signs:
                    for th in args.thetas:
                        t = torch.tensor(th, dtype=torch.float32)
                        with torch.no_grad():
                            x_adv, stats = fn(x, pairs, t, gates,
                                              sign[smode], args.qd, d)
                            m = suite.pairwise(x, x_adv)
                        rows.append({
                            "image": name, "variant": vname, "gate": gmode,
                            "sign": smode, "theta": round(th, 4),
                            "gate_mean": round(gate_mean, 5),
                            "dists": round(m["dists"], 6),
                            "lpips": round(m["lpips"], 6),
                            "psnr": round(m["psnr"], 4),
                            "ssim": round(m["ssim"], 6),
                            "rms": round(m["rms"], 6),
                            "linf": round(m["linf"], 6),
                            **{k: round(v, 6) for k, v in stats.items()},
                        })
        write_csv(args.out, rows)
        print(f"  {idx + 1}/{len(names)} {name}", flush=True)

    print()
    print(f"十張平均（交付品質 QD = {args.qd}，配對 {args.pairing}）")
    print(f"{'變體':<12s}{'閘':<9s}{'sign':<10s}{'theta':>7s}"
          f"{'DISTS':>9s}{'PSNR':>8s}{'RMS':>9s}{'零對比例':>10s}"
          f"{'|δ|<=1':>9s}")
    for vname in args.variants:
        for gmode in args.gates:
            for smode in args.signs:
                for th in args.thetas:
                    sel = [r for r in rows if r["variant"] == vname
                           and r["gate"] == gmode and r["sign"] == smode
                           and abs(r["theta"] - th) < 1e-3]
                    if not sel:
                        continue
                    def f(k, s=sel):
                        vals = [r[k] for r in s if k in r]
                        return statistics.fmean(vals) if vals else float("nan")
                    print(f"{vname:<12s}{gmode:<9s}{smode:<10s}{th:7.3f}"
                          f"{f('dists'):9.5f}{f('psnr'):8.2f}{f('rms'):9.5f}"
                          f"{f('zero_pair_frac'):10.4f}"
                          f"{f('delta_within_1'):9.4f}")

    print()
    print(f"失真帶 DISTS {BAND_LO}–{BAND_HI}；`ours_ph_q` 是 {OURS_PH_Q_DISTS}。")
    for vname in args.variants:
        for gmode in args.gates:
            top = [r for r in rows if r["variant"] == vname
                   and r["gate"] == gmode
                   and abs(r["theta"] - max(args.thetas)) < 1e-3]
            if not top:
                continue
            ceil = max(statistics.fmean(
                [r["dists"] for r in top if r["sign"] == s] or [float("nan")])
                for s in args.signs)
            verdict = "進得了帶" if ceil >= BAND_LO else "**構不到帶**"
            print(f"  {vname:<12s} 閘={gmode:<8s} theta={max(args.thetas):.3f} "
                  f"天花板 DISTS {ceil:.5f} → {verdict}")
    print(f"表：{args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
