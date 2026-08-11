#!/usr/bin/env python
"""訓練期壓下的 c_a 抑制，有沒有活過攻擊方的編輯鏈。

    python scripts/ca_attention_probe.py --out runs/ca_probe \
        --cond "apa+A=runs/s3t20_r_merged/apa" ... <run_stage 的旗標>

## 這支腳本存在的理由

`apa` 的訓練把 `‖Att(x_def, c_a) ⊙ M‖₁` 壓掉 77–87%，而評測期的 `edit_lpips`
只有 0.107。中間缺一個環節：**那個抑制在攻擊方真正的取樣鏈上還在不在。**

現有的 `attention.html` 答不了這個問題。`_sdedit` 的 `capture_span` 取的是
**攻擊方 prompt** 的內容 token（`a zebra` 的 `zebra`），而訓練期施力的是防禦方
指名的 c_a（`horse`）——兩者是不同的詞、不同的 span。拿前者去說後者，是把
兩個量當成同一個。

本腳本在同一條 50 步編輯鏈上，改以 **c_a 的 span** 擷取注意力，逐條件比較：

- 抑制活過來了 → 防禦圖的 c_a 質量顯著低於未防禦，且低於隨機對照
- 抑制沒活過來 → 三者相同，那麼「壓低 c_a 注意力」這個著力點在訓練期成功、
  在攻擊期無效，`edit_lpips` 為何只有 0.107 就有了機制上的解釋

**與隨機對照 `Ra` 並列是必要的**：若防禦圖與未防禦有差、但隨機擾動也有同樣
的差，那個差就不是抑制造成的。

## 不動評測管線

刻意寫成獨立腳本而不是改 `eval_executor`：改後者會讓既有批次的 `config_hash`
全部改變、整批判為未完成。這裡只讀已經存到磁碟的 `x_def`，重跑一次編輯鏈。
"""

import csv
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_stage import (build_parser, build_resources,  # noqa: E402
                               resolve_thresholds)
from src.experiment.attn_capture import AttnCapture  # noqa: E402
from src.experiment.executors import (eval_noise_seed,  # noqa: E402
                                      load_image_tensor)
from src.models.attention import token_span  # noqa: E402


def probe(res, entry, x, seeds):
    """在攻擊鏈上以 c_a 的 span 擷取注意力，回傳逐 seed 的統計。

    prompt、噪聲、步數、guidance 全部取評測期的同一組值——差一個就不是
    「攻擊方實際會走的那條鏈」，而這支腳本問的正是那條鏈上發生什麼。
    """
    span = token_span(res.sd.tokenizer, entry.content)
    emb = res.sd.encode_text(entry.attack_prompt()).detach()
    emb_u = res.sd.uncond_prompt().detach()
    lat = res.sd.latent_shape(x.shape[-2], x.shape[-1])
    out = []
    for s in seeds:
        noise = res.sd.sample_edit_noise(
            torch.empty(lat, device=x.device), seed=eval_noise_seed(res, s))
        cap = AttnCapture(res.sd, res.cfg.steps, span)
        with cap, torch.no_grad():
            res.sd.edit(x, emb, noise, res.cfg.steps,
                        guidance_scale=res.cfg.guidance, emb_uncond=emb_u,
                        mask=entry.mask,
                        strength=(None if entry.mask is not None
                                  else res.cfg.strength),
                        step_hook=cap.step_hook)
        rows = cap.rows
        out.append((st.fmean(r["content_mass_mean"] for r in rows),
                    st.fmean(r["entropy"] for r in rows)))
    return out


def main(argv=None) -> int:
    ap = build_parser()
    ap.add_argument("--cond", action="append", required=True,
                    help="`名稱=條件目錄`。名稱內不可有 `=`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe-seeds", type=int, default=3)
    args, unknown = ap.parse_known_args(argv)
    resolve_thresholds(args, verbose=False)

    cols = []
    for c in args.cond:
        name, _, root = c.partition("=")
        if not root or not Path(root).is_dir():
            raise SystemExit(f"--cond {c!r} 的目錄 {root!r} 不存在")
        cols.append((name, Path(root)))

    args.out.mkdir(parents=True, exist_ok=True)
    res = build_resources(args, args.out)
    seeds = list(range(args.probe_seeds))
    rows = []
    for image_id in args.images:
        entry = res.image(image_id)
        print(f"\n[{image_id}] c_a={entry.content!r}  "
              f"攻擊 prompt={entry.attack_prompt()!r}", flush=True)
        base = None
        for name, root in cols:
            png = root / image_id / "x_def_tau0.04.png"
            x = (entry.x01 if not png.exists()
                 else load_image_tensor(png, res.device, args.resolution))
            got = probe(res, entry, x, seeds)
            mass = st.fmean(m for m, _ in got)
            ent = st.fmean(e for _, e in got)
            if base is None:
                base = (mass, ent)
            rel = (mass / base[0] - 1) * 100
            print(f"  {name:<22} c_a 質量 {mass:.5f}（{rel:+.1f}%）  "
                  f"熵 {ent:.4f}  來源 {'原圖' if not png.exists() else png.name}",
                  flush=True)
            rows.append({"image_id": image_id, "condition": name,
                         "ca": entry.content, "ca_mass": mass,
                         "ca_entropy": ent, "rel_to_first_pct": rel,
                         "n_seeds": len(seeds),
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
