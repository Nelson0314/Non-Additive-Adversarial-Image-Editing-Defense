"""脆弱浮水印在一般有損處理下的存活率。**不需要 GPU，不載任何模型權重。**

問的是什麼
────────────────────────────────────────────────────────────────────
`anti-purify/start.md` §7.4 指出脆弱浮水印難的不是脆弱那一半（文獻已量到擴散
重生把三個主流浮水印的位元正確率由 99.8% 打到 7.4%／12.8%／24.5%），而是
**「被 JPEG 與模糊之後不崩潰」那一半**。兩半都掉到 0.5 附近的方案沒有用：
舉證需要「一般處理後仍讀得回來、編輯後讀不回來」這個落差，只有落差才是證據。

本探針只量前一半，因為它不需要擴散模型。每張影像：

1. 用 `src/watermark/qim.py` 嵌入一組固定種子的位元酬載；
2. 用 `src/metrics/suite.py` 的 `pairwise` 量**防禦圖對原圖的失真**；
3. 對每個 `src/purify/ops.py` 的算子跑一次，再萃取一次，記位元正確率。

擴散編輯那一欄要 GPU，**不在本檔**。算子清單是資料驅動的（`--op KIND:STRENGTH`
可重複），要加那一欄時在 `run_op` 補一個分支即可，其餘的欄位與 CSV 結構不動。

為什麼每個旋鈕都是欄位而不是註解
────────────────────────────────────────────────────────────────────
`delta`、`coeffs`、`repeat`、`payload_bits`、`seed` 只要有一個與嵌入端不同，
萃取讀數就落在 0.5 附近——而 0.5 附近**也正是浮水印被編輯打掉的長相**。兩者
無法由讀數本身分辨。本專案已被「設定只活在 argparse 預設值裡、合併批次之後
分不出來」咬過多次，故上列全部逐列寫進 CSV，`--delta`、`--coeffs`、`--repeat`
一律必填（§7.4 沒有指定這三個，依專案規定不填看起來合理的預設）。

`crop_resize` 那一欄預期就是亂猜（約 0.5）：區塊 DCT 綁在固定的 8×8 格點上，
裁切把格點整個推開。**這是本方案的已知限制，量出來記錄，不在此處理。**

用法：

    python scripts/watermark_probe.py --out runs/wm_qim/results.csv
        --delta 48 --coeffs 1,2 2,1 2,2 1,3 --repeat 64 --payload-bits 64
        --images task_attr_mod_color_11699 task_attr_mod_color_6205
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.metrics.suite import MetricSuite  # noqa: E402
from src.purify.ops import GEOMETRIC_KINDS, KINDS, Purifier  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402
from src.watermark.qim import (  # noqa: E402
    BLOCK, band_coeffs, bit_accuracy, embed, extract, random_bits, slot_count,
)
from src.watermark.sync import search_alignment  # noqa: E402

RESOLUTION = 512

# 題目指定要涵蓋的算子。**每一列都會把 kind 與 strength 寫進 CSV**，所以這個
# 常數只決定「沒給 --op 時跑哪些」，不會變成日後分不出來的隱藏設定。
PROBE_OPS: Tuple[Tuple[str, float], ...] = (
    ("identity", 0.0),
    ("jpeg", 75.0),
    ("jpeg", 30.0),
    ("blur", 1.0),
    ("blur", 2.0),
    ("noise", 0.02),
    ("quantize", 16.0),
    ("crop_resize", 0.1),
)

# `noise` 與 `quantize` 的強度在 §7.4 裡沒有指定，取自 `src/purify/ops.py`
# 的 `eval_sweep`（noise 的 0.02 與 quantize 的 16 都在該掃描的既有格點上），
# 故與本專案其他表格同軸。這一點寫在這裡是因為它是**選擇**，不是預設值：
# 兩者都可由 `--op` 覆蓋，且逐列進 CSV。


def parse_coeff_token(tok: str) -> List[Tuple[int, int]]:
    """`"1,2"` 或 `"band:3-5"`。格式錯就拋，不猜。"""
    if tok.startswith("band:"):
        rng = tok[len("band:"):]
        if rng.count("-") != 1:
            raise ValueError(f"band 的格式是 band:LO-HI，收到 {tok!r}")
        lo, hi = rng.split("-")
        return list(band_coeffs(int(lo), int(hi)))
    if tok.count(",") != 1:
        raise ValueError(f"係數的格式是 U,V 或 band:LO-HI，收到 {tok!r}")
    u, v = (int(t) for t in tok.split(","))
    # 範圍在這裡就擋掉，否則 `check_args` 放行、要到 `embed` 才由
    # `src/watermark/qim.py` 拋出——那已經在載完指標權重之後了。
    if not (0 <= u < BLOCK and 0 <= v < BLOCK):
        raise ValueError(f"係數索引必須落在 [0,{BLOCK})，收到 {tok!r}")
    return [(u, v)]


def parse_coeffs(tokens: List[str]) -> Tuple[Tuple[int, int], ...]:
    out: List[Tuple[int, int]] = []
    for tok in tokens:
        out.extend(parse_coeff_token(tok))
    return tuple(out)


# 攻擊方的擴散編輯。**不是淨化算子**，故不在 `KINDS` 裡，另外列出來。
# 這一欄問的是脆弱那一半：編輯過後位元還讀不讀得回來。
EDIT_OP = "ip2p_edit"
# 只走 VAE 的編碼—解碼，不做任何編輯。這一欄是歸因用的：擴散編輯必定經過
# 這個瓶頸，所以 `ip2p_edit` 掉下來的位元裡有多少是**任何**擴散再生都會掉的
# （與指令無關、與內容無關），看這一欄就知道。兩者相近代表偵測的是「這張圖
# 進過擴散模型」，那比「這張圖被下了某個指令」更通用。
VAE_OP = "vae_roundtrip"
MODEL_OPS = (EDIT_OP, VAE_OP)


def parse_op(tok: str) -> Tuple[str, float]:
    """`"jpeg:30"`、`"identity"`、`"ip2p_edit"`。強度省略時為 0.0。"""
    kind, sep, s = tok.partition(":")
    if kind not in MODEL_OPS and kind not in KINDS:
        raise ValueError(
            f"未知的算子 {kind!r}；可用的是 {sorted(KINDS)} 加上 "
            f"{list(MODEL_OPS)}")
    return kind, (float(s) if sep else 0.0)


def coeffs_to_field(coeffs: Tuple[Tuple[int, int], ...]) -> str:
    """CSV 欄位用的字串。分號分隔，不含空白，可直接餵回 `--coeffs`。"""
    return ";".join(f"{u},{v}" for u, v in coeffs)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    # 下面三個沒有預設值：§7.4 未指定，且填錯的症狀與「被編輯打掉」相同。
    ap.add_argument("--delta", type=float, default=None,
                    help="**必填**。QIM 的量化步長，作用在 [0,255] luma 的 "
                         "DCT 係數上。Δ/4 是本方案能吃下的最大係數擾動")
    ap.add_argument("--coeffs", nargs="+", default=None,
                    help="**必填**。載體係數，U,V 或 band:LO-HI，可混用")
    ap.add_argument("--repeat", type=int, default=None,
                    help="**必填**。每個位元重複幾個區塊，萃取端多數決")
    ap.add_argument("--payload-bits", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20250903,
                    help="酬載的種子。位元序列不入版控，只有 (bits, seed) 能還原")
    ap.add_argument("--op", action="append", default=None, metavar="KIND[:STRENGTH]",
                    help=f"可重複。未給時用 PROBE_OPS："
                         f"{' '.join(f'{k}:{s:g}' for k, s in PROBE_OPS)}")
    ap.add_argument("--op-seed", type=int, default=0,
                    help="`noise` 的種子，逐列寫進 CSV")
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="有給時把浮水印圖存成 PNG，供日後的擴散編輯欄使用")
    ap.add_argument("--device", default="cpu",
                    help="淨化那幾欄不需要 GPU；預設 cpu。要跑 "
                         f"{EDIT_OP} 那一欄時必須給 cuda")
    # 攻擊模型的三個推論參數：論文未載、是本專案指定的，改動會讓新舊批次
    # 不可比（`anti-purify/start.md` 第五節）。預設值與全專案一致。
    ap.add_argument("--edit-steps", type=int, default=100)
    ap.add_argument("--text-guidance", type=float, default=7.5)
    ap.add_argument("--image-guidance", type=float, default=1.5)
    ap.add_argument("--edit-seed", type=int, default=20260812)
    ap.add_argument("--sync-bits", type=int, default=0,
                    help="酬載的前幾個位元當成雙方講好的前導碼，用來在**幾何類**"
                         "算子後搜尋幾何。0（預設）時完全不搜，裁切那一欄就是"
                         "沒有對齊的讀數。實測 16 位元鎖不住（吻合率 0.75、"
                         "選到錯的幾何），32 位元可以（0.94–0.97）。"
                         "正確率一律只算**前導碼以外**的位元")
    ap.add_argument("--sync-scales", nargs="+", type=float,
                    default=[1.24, 1.245, 1.25, 1.255],
                    help="搜尋的尺度候選。預設涵蓋 crop_resize 0.1 的 1.2488")
    ap.add_argument("--sync-radius", type=int, default=10,
                    help="位移的搜尋半徑（像素）。要涵蓋整格以上，"
                         "因為整格錯位信心度看不出來")
    return ap


def check_args(args) -> None:
    """把「補錯了不會報錯，只會量到 0.5」的情形擋在讀圖與載指標之前。"""
    if args.delta is None:
        raise SystemExit(
            "--delta 必填：§7.4 沒有指定量化步長，而填錯的讀數（0.5 附近）"
            "與『浮水印被編輯打掉』無法分辨。中頻參考值：JPEG q=75 的 luma "
            "步長約 7–10、q=30 約 22–32，Δ/4 要蓋過其一半")
    if args.delta <= 0:
        raise SystemExit(f"--delta 必須為正，收到 {args.delta}")
    if args.coeffs is None:
        raise SystemExit(
            "--coeffs 必填：DC 會隨全域色調平移、最高頻會被 JPEG 量化成零，"
            "沒有一個可以當預設。例：--coeffs 1,2 2,1 2,2 1,3")
    try:
        coeffs = parse_coeffs(args.coeffs)
    except ValueError as e:
        raise SystemExit(str(e))
    if len(set(coeffs)) != len(coeffs):
        raise SystemExit(f"--coeffs 有重複的格：{coeffs_to_field(coeffs)}")
    if args.repeat is None:
        raise SystemExit(
            "--repeat 必填：冗餘倍數決定多數決能救回多少，沒有預設值。"
            f"512×512 在 {len(coeffs)} 個係數下共有 "
            f"{slot_count(RESOLUTION, RESOLUTION, len(coeffs))} 個槽")
    if args.repeat < 1:
        raise SystemExit(f"--repeat 必須是正整數，收到 {args.repeat}")
    if args.payload_bits < 1:
        raise SystemExit(f"--payload-bits 必須是正整數，收到 {args.payload_bits}")

    total = slot_count(RESOLUTION, RESOLUTION, len(coeffs))
    need = args.payload_bits * args.repeat
    if need > total:
        raise SystemExit(
            f"{args.payload_bits} 個位元乘 repeat={args.repeat} 需要 {need} 個槽，"
            f"但 {RESOLUTION}×{RESOLUTION} 在 {len(coeffs)} 個係數下只有 {total} 個")

    # 沒給 --op 時也要驗 `PROBE_OPS`，否則日後改動那個常數會在跑到一半才炸。
    try:
        ops = [parse_op(t) for t in args.op] if args.op else list(PROBE_OPS)
    except ValueError as e:
        raise SystemExit(str(e))
    for kind, _ in ops:
        if kind in MODEL_OPS:
            # 攻擊方的編輯不是 `Purifier`，它的相依是 IP2P 權重與一張卡。
            # 在 CPU 上跑得動但要幾十分鐘，等於整批卡死而不報錯，故當成
            # 設定錯誤擋下來。
            if torch.device(args.device).type != "cuda":
                raise SystemExit(
                    f"--op {kind} 需要 --device cuda："
                    f"在 {args.device} 上跑 IP2P 會慢到整批卡住而不報錯。")
            continue
        p = Purifier(kind)
        if not p.available:
            raise SystemExit(
                f"算子 {kind} 的相依不齊備（`Purifier.available` 為 False），"
                f"本探針不靜默略過。請移除該 --op 或補齊相依")

    if not args.data.is_dir():
        raise SystemExit(f"--data 不是目錄：{args.data}")
    for name in args.images:
        p = args.data / name / f"{name}.png"
        if not p.exists():
            raise SystemExit(f"找不到影像 {p}")


def run_op(kind: str, strength: float, seed: int, x: torch.Tensor,
           edit_fn=None) -> torch.Tensor:
    """套用一個算子。

    淨化那幾欄走 `Purifier.evaluate`（真實實作、`torch.no_grad`），不自行
    實作任何算子：本專案其他表格量的就是這一份，換一份實作就不可並列。

    `ip2p_edit` 是**攻擊方的編輯**，不是淨化。它需要逐圖的指令，故由呼叫端
    傳一個已經綁好指令的 `edit_fn` 進來；沒傳就拋錯而不是靜默跳過——靜默
    跳過的症狀是「表上少一欄而其餘看起來完全正常」。
    """
    if kind in MODEL_OPS:
        if edit_fn is None:
            raise ValueError(
                f"{kind} 需要 edit_fn（攻擊方的編輯或 VAE 往返），"
                f"呼叫端沒有提供。")
        return edit_fn(x)
    return Purifier(kind, strength, seed=seed).evaluate(x)


def main() -> None:
    args = build_parser().parse_args()
    check_args(args)

    coeffs = parse_coeffs(args.coeffs)
    ops = [parse_op(t) for t in args.op] if args.op else list(PROBE_OPS)
    dev = torch.device(args.device)
    suite = MetricSuite(device=dev)

    # 攻擊方的編輯只在有那一欄時才載模型：淨化那幾欄純 CPU，不該為了一個
    # 沒被要求的欄位吃掉一張卡。
    prompts = {}
    ip2p = None
    if any(k in MODEL_OPS for k, _ in ops):
        from apa_baseline import load_dataset

        from src.models.ip2p import IP2PWrapper
        dataset = {d["name"]: d for d in load_dataset(args.data)}
        missing = [n for n in args.images if n not in dataset]
        if missing:
            raise SystemExit(f"{args.data} 底下沒有這些影像：{missing}")
        prompts = {n: dataset[n]["prompt"] for n in args.images}
        ip2p = IP2PWrapper(dtype=torch.float32)
        print(f"模型欄：已載入 IP2P（{ip2p.device}）", flush=True)
    bits = random_bits(args.payload_bits, args.seed)
    total = slot_count(RESOLUTION, RESOLUTION, len(coeffs))

    # 每一列都自帶全部設定：合併批次之後，光看 CSV 就分得出哪一列是哪一組。
    knobs = {
        "delta": args.delta,
        "coeffs": coeffs_to_field(coeffs),
        "n_coeffs": len(coeffs),
        "repeat": args.repeat,
        "payload_bits": args.payload_bits,
        "seed": args.seed,
        "slots_total": total,
        "slots_used": args.payload_bits * args.repeat,
        "op_seed": args.op_seed,
        "resolution": RESOLUTION,
        # 攻擊模型的三個推論參數論文未載、是本專案指定的，改動會讓新舊批次
        # 不可比，故逐列寫出而不是只活在 argparse 的預設值裡。
        "edit_steps": args.edit_steps,
        "s_t": args.text_guidance,
        "s_i": args.image_guidance,
        "edit_seed": args.edit_seed,
    }

    rows = []
    for name in args.images:
        x = load_image_tensor(args.data / name / f"{name}.png", dev, size=RESOLUTION)
        # verify=True：嵌入端自我萃取對不上就直接拋。飽和影像承載不了這組
        # 參數時應該當場中止，而不是把一個讀不出來的浮水印混進 identity 欄，
        # 讓它看起來像是被某個算子打掉的。
        xw = embed(x, bits, delta=args.delta, coeffs=coeffs,
                   repeat=args.repeat, verify=True)
        if args.save_dir is not None:
            # `embed` 的輸出已經在 uint8 網格上，故這一步的 PNG 量化是恆等，
            # 存下來的圖與本檔量到的那一張逐位元相同。
            from torchvision.utils import save_image
            args.save_dir.mkdir(parents=True, exist_ok=True)
            save_image(xw, args.save_dir / f"{name}__wm.png")

        dist = suite.pairwise(x, xw)
        base = {"image": name, **knobs,
                "wm_lpips": round(dist["lpips"], 6),
                "wm_dists": round(dist["dists"], 6),
                "wm_psnr": round(dist["psnr"], 4),
                "wm_ssim": round(dist["ssim"], 6),
                "wm_linf": round(dist["linf"], 6),
                "wm_rms": round(dist["rms"], 6)}
        print(f"{name}  失真 LPIPS={dist['lpips']:.4f} DISTS={dist['dists']:.4f} "
              f"PSNR={dist['psnr']:.2f}", flush=True)

        def model_fn(img, _name=name, _kind=EDIT_OP):
            if _kind == VAE_OP:
                # 只走瓶頸：encode 之後直接 decode，不加噪也不去噪。
                # `encode_image` 有乘 scaling_factor，故解碼前要除回去。
                with torch.no_grad():
                    z = ip2p.encode_image(img.to(ip2p.device))
                    z = z / ip2p.vae.config.scaling_factor
                    out = ip2p.vae.decode(z.to(ip2p.vae.dtype)).sample
                return ((out / 2 + 0.5).clamp(0, 1)).to(img)
            return ip2p.edit(img.to(ip2p.device), prompts[_name],
                             seed=args.edit_seed, steps=args.edit_steps,
                             s_t=args.text_guidance,
                             s_i=args.image_guidance).to(img)

        for kind, strength in ops:
            y = run_op(kind, strength, args.op_seed, xw,
                       edit_fn=(lambda im, _k=kind: model_fn(im, _kind=_k))
                       if kind in MODEL_OPS else None)
            # 幾何類算子把 8x8 的格點整個推開，不對齊就讀不回來。**只有這一類
            # 需要搜尋**：其餘算子不動幾何，搜了只會多花時間並引入選錯的風險。
            info = {}
            if args.sync_bits and kind in GEOMETRIC_KINDS:
                got, info = search_alignment(
                    y, args.payload_bits, delta=args.delta, coeffs=coeffs,
                    repeat=args.repeat, scales=args.sync_scales,
                    sync=bits[:args.sync_bits], offset_radius=args.sync_radius)
            else:
                got = extract(y, args.payload_bits, delta=args.delta,
                              coeffs=coeffs, repeat=args.repeat)
            # 前導碼佔掉的位元不計入正確率——它是雙方已知的，算進去會灌水。
            k = args.sync_bits
            acc = bit_accuracy(got[k:], bits[k:]) if k else bit_accuracy(got, bits)
            if kind in MODEL_OPS and args.save_dir is not None:
                from torchvision.utils import save_image
                save_image(y.clamp(0, 1),
                           args.save_dir / f"{name}__wm_{kind}.png")
            rows.append({**base, "op": kind, "op_strength": strength,
                         "instruction": prompts.get(name, ""),
                         "sync_bits": args.sync_bits,
                         "sync_used": bool(info),
                         "sync_scale": round(info["scale"], 4) if info else "",
                         "sync_match": round(info["sync_match"], 4) if info else "",
                         "sync_valid_frac": (round(info["valid_frac"], 4)
                                             if info else ""),
                         "bit_acc": round(acc, 6)})
            print(f"    {kind}:{strength:g}  位元正確率 {acc:.4f}", flush=True)
            write_csv(args.out, rows)

    write_csv(args.out, rows)
    print(f"\n寫出 {args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()
