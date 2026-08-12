#!/usr/bin/env python
"""訓練期壓下的 c_a 抑制，在**沒有被取樣到的 timestep** 上還在不在。

    python scripts/ca_attention_probe.py --out runs/ca_probe         --cond "apa+A=runs/s3t20_r_merged/apa" ... <run_stage 的旗標>

## 這支腳本問的問題

`apa` 的訓練把 `‖Att(x_def, c_a) ⊙ M‖₁` 壓掉 77–87%，而評測期的 `edit_lpips`
只有 0.107。中間缺一個環節：那個抑制的**適用範圍**有多大。

訓練只在 `attn_timesteps = 2` 個 timestep 上施力（`_build_attn_step` 把
`[0, t_edit]` 均分取兩點），而攻擊方走 50 步。若抑制只存在於被取樣到的那兩點，
它對攻擊鏈上其餘 48 步就沒有作用——那會是「訓練目標達成、防禦無效」的直接
機制解釋，也指出 DEC-016 C2（2 → 4）該往哪個方向調。

故本腳本在**同一組條件**下（c_a 的嵌入、原圖取的遮罩 M、同一個式 (3) 聚合）
把 timestep 掃密，逐條件比較遮罩內的 L1 與它佔全圖的比例。

## 為什麼不是「在攻擊鏈上量 c_a」

那個問法不可執行，而**第一版正是這樣寫的，量到的是錯的東西**：攻擊鏈的條件
嵌入是攻擊方的 prompt（`a zebra`），而 `token_span(tokenizer, "horse")` 回傳
的是 (1, 2)——`a zebra` 的第 1 格是 `a`。兩者拼起來量到的是「a」這個 token
的注意力，與 c_a 無關，而數字看起來完全正常（各條件差 −0.7% 到 +2.2%）。
c_a 不在攻擊 prompt 裡，它的注意力只有在以 c_a 為條件時才有定義。

**與隨機對照 `Ra` 並列是必要的**：若隨機擾動也造成同樣的下降，那個下降就不是
最佳化取得的。
"""

import csv
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.run_stage import (build_parser, build_resources,  # noqa: E402
                               resolve_thresholds)
from src.experiment.attn_capture import AttnCapture  # noqa: E402
from src.experiment.executors import (eval_noise_seed,  # noqa: E402
                                      load_image_tensor)
from src.models.attention import token_span  # noqa: E402


def _mask_and_span(res, entry, n_t):
    """式 (4) 的遮罩 M 與 c_a 的 token 區間，取法與訓練期逐字相同。

    遮罩由**原圖**取（對 φ 為常數），在多個 timestep 上平均後再二值化。
    """
    from src.models.attention import (aggregate_token_attention,
                                      attention_region_mask)
    from src.models.attention import CrossAttentionRecorder

    span = token_span(res.sd.tokenizer, entry.content)
    emb = res.sd.encode_text(entry.content).detach()
    t_edit = (res.sd.num_train_timesteps - 1 if entry.mask is not None
              else min(int(res.sd.num_train_timesteps * res.cfg.strength),
                       res.sd.num_train_timesteps - 1))
    ts = torch.linspace(0, t_edit, n_t + 1)[1:].round().long()
    abar = res.sd.alphas_cumprod(res.device)
    rec = CrossAttentionRecorder(res.sd.unet)
    lat = res.sd.latent_shape(*entry.x01.shape[-2:])
    with torch.no_grad():
        z0 = res.sd.encode_image(entry.x01)
        atts = []
        for i, t in enumerate(ts):
            n = res.sd.sample_edit_noise(torch.empty(lat, device=res.device),
                                         seed=res.cfg.seed + i)
            zt = abar[t].sqrt() * z0 + (1 - abar[t]).sqrt() * n
            if res.sd.is_inpainting:
                m9, zm = res.sd.mask_latents(entry.x01, entry.mask)
                zt = torch.cat([zt, m9.to(zt.dtype), zm.to(zt.dtype)], dim=1)
            with rec:
                res.sd._eps(zt, t, emb)
            atts.append(aggregate_token_attention(rec.maps, span, reduce="sum"))
            rec.clear()
        ref = torch.stack(atts).mean(dim=0)
    return attention_region_mask(ref, res.cfg.attn_mask_tau), span, emb, ts, abar


def probe(res, entry, x, mask, span, emb, ts, abar):
    """逐 timestep 量遮罩內的注意力 L1 與它佔全圖的比例。

    條件嵌入取 **c_a 本身**，與訓練期的 `emb_attn` 逐字相同。c_a 不在攻擊
    prompt 裡，故「c_a 的注意力」只有在以 c_a 為條件時才有定義（見模組
    docstring 對第一版錯誤的說明）。
    """
    from src.models.attention import (CrossAttentionRecorder,
                                      aggregate_token_attention,
                                      masked_attention_fraction,
                                      masked_attention_l1)

    rec = CrossAttentionRecorder(res.sd.unet)
    lat = res.sd.latent_shape(*x.shape[-2:])
    side = int(mask.shape[-1])
    out = []
    with torch.no_grad():
        z0 = res.sd.encode_image(x)
        for i, t in enumerate(ts):
            n = res.sd.sample_edit_noise(torch.empty(lat, device=res.device),
                                         seed=res.cfg.seed + i)
            zt = abar[t].sqrt() * z0 + (1 - abar[t]).sqrt() * n
            if res.sd.is_inpainting:
                m9, zm = res.sd.mask_latents(x, entry.mask)
                zt = torch.cat([zt, m9.to(zt.dtype), zm.to(zt.dtype)], dim=1)
            with rec:
                res.sd._eps(zt, t, emb)
            att = aggregate_token_attention(rec.maps, span, side=side,
                                            reduce="sum")
            rec.clear()
            out.append((int(t), float(masked_attention_l1(att, mask)),
                        float(masked_attention_fraction(att, mask))))
    return out


def main(argv=None) -> int:
    ap = build_parser()
    ap.add_argument("--cond", action="append", required=True,
                    help="`名稱=條件目錄`。名稱內不可有 `=`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-timesteps", type=int, default=12,
                    help="掃描幾個 timestep（均分於 [0, t_edit]）")
    # `build_parser()` 的 `stage` 與 `--batch` 是給五段驅動用的必填項，本腳本
    # 兩者都用不到（`build_resources` 直接收 `--out` 當批次目錄）。在此補上
    # 佔位值而不是叫呼叫端硬打——那會讓命令列出現一個沒有意義卻不能省的字。
    argv = list(sys.argv[1:] if argv is None else argv)
    args, unknown = ap.parse_known_args(["eval", "--batch", "ca_probe", *argv])
    resolve_thresholds(args, verbose=False)

    cols = []
    for c in args.cond:
        name, _, root = c.partition("=")
        if not root or not Path(root).is_dir():
            raise SystemExit(f"--cond {c!r} 的目錄 {root!r} 不存在")
        cols.append((name, Path(root)))

    args.out.mkdir(parents=True, exist_ok=True)
    res = build_resources(args, args.out)
    rows = []
    for image_id in args.images:
        entry = res.image(image_id)
        mask, span, emb, ts, abar = _mask_and_span(res, entry, args.n_timesteps)
        # 訓練實際施力的兩點：`_build_attn_step` 把 [0, t_edit] 均分取
        # `attn_timesteps` 個。標出來才看得出「有施力的 t」與「沒施力的 t」
        # 之間有沒有差別——那正是本腳本要判的事。
        t_edit = int(ts[-1])
        trained = set(int(v) for v in torch.linspace(
            0, t_edit, res.cfg.attn_timesteps + 1)[1:].round().long())
        print(f"\n[{image_id}] c_a={entry.content!r}  遮罩覆蓋 "
              f"{float(mask.mean()) * 100:.1f}%  訓練施力於 t={sorted(trained)}",
              flush=True)
        base = None
        for name, root in cols:
            png = root / image_id / "x_def_tau0.04.png"
            x = (entry.x01 if not png.exists()
                 else load_image_tensor(png, res.device, args.resolution))
            got = probe(res, entry, x, mask, span, emb, ts, abar)
            if base is None:
                base = {t: (l1, fr) for t, l1, fr in got}
            near = [(t, l1, fr) for t, l1, fr in got
                    if min(abs(t - u) for u in trained) <= t_edit / (2 * args.n_timesteps)]
            far = [(t, l1, fr) for t, l1, fr in got if (t, l1, fr) not in near]
            def rel(sub):
                return (st.fmean(l1 / base[t][0] - 1 for t, l1, _ in sub) * 100
                        if sub else float("nan"))
            print(f"  {name:<20} L1 全域 {st.fmean(l1 for _, l1, _ in got):>8.2f}"
                  f"（{rel(got):+6.1f}%）  施力點附近 {rel(near):+6.1f}%  "
                  f"其餘 t {rel(far):+6.1f}%", flush=True)
            for t, l1, fr in got:
                rows.append({"image_id": image_id, "condition": name,
                             "ca": entry.content, "t": t, "masked_l1": l1,
                             "masked_fraction": fr,
                             "rel_to_first_pct": (l1 / base[t][0] - 1) * 100,
                             "trained_at_t": min(abs(t - u) for u in trained)
                             <= t_edit / (2 * args.n_timesteps),
                             "source": "orig" if not png.exists() else str(png)})

    with (args.out / "ca_attention.csv").open("w", newline="",
                                              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "ca_attention.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫入 {args.out / 'ca_attention.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
