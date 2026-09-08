"""第二輪篩選：加上臉框守門，並用精修後的載體重算面積。

只處理第一輪已判定有臉的 55 張（`face_candidates.csv`），不重掃全部。

五道關：
  1. ATR 的 Face+Hair 面積 ≥ 0.01        （第一輪已過）
  2. MTCNN 偵測得到臉                     （身分讀數需要）
  3. **臉框內被判成衣物的比例 < 0.15**    ← 新增。ATR 會把臉的皮膚判成
     Upper-clothes，實測一張特寫上是 0.6561 而正常影像是 0.000–0.033。
  4. 精修後的合法載體 ≥ 0.05
  5. 逐張看圖（本腳本不做，輸出連絡表供人眼排除剛體誤判）

精修 = 導引濾波吸附 → 挖掉臉框 → 內縮 3 → 往內羽化 8 → 交集主體補集。
"""
import csv, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.defense.carrier_mask import (CARRIER_CLASSES, box_conflict,
                                      erode_mask, exclude_boxes,
                                      feather_inward, guided_refine, parse_atr)
from src.defense.subject_mask import _dilate, _feather
from src.metrics.identity import face_boxes

SC = ROOT / "runs/carrier_survey"
def main() -> None:
    """**這一段原本在 module 層級執行**，於是 `import` 這支腳本就會跑完整個
    實驗並覆寫結果檔。`tests/test_ip2p_run_columns.py` 的
    `test_every_driver_script_imports` 會 import 每一支 `scripts/*.py`，
    所以每跑一次測試就重寫一次結果；在缺那些資料檔的機器上則直接
    `FileNotFoundError`。工作一律放在這裡。"""
    prev = list(csv.DictReader(open(SC/"face_candidates.csv", encoding="utf-8")))
    todo = [r["image"] for r in prev if r["mtcnn_face"] == "True"]
    print(f"處理 {len(todo)} 張（第一輪 MTCNN 有偵測到臉的）", flush=True)

    rows = []
    for i, name in enumerate(todo):
        base = np.asarray(Image.open(ROOT/f"data/omniedit150/{name}/{name}.png"
                                     ).convert("RGB")).astype(np.float32)/255
        x = torch.from_numpy(base).permute(2, 0, 1)[None]
        seg = parse_atr(x).numpy()
        cl = torch.from_numpy(np.isin(seg, CARRIER_CLASSES["clothes"]
                                      ).astype(np.float32))[None, None]
        boxes = face_boxes(x)
        conflict = box_conflict(cl, boxes)
        hard = torch.from_numpy(np.isin(seg, (11, 2)).astype(np.float32))[None, None]
        m = _feather(_dilate(hard, 16), 24)
        refined = (guided_refine(cl, x, 4, 1e-3) > 0.5).float()
        refined = exclude_boxes(refined, boxes, margin=8)
        carrier = feather_inward(erode_mask(refined, 3), 8) * (m <= 0.0).float()
        rows.append({"image": name, "task": name.rsplit("_", 1)[0],
                     "n_faces": len(boxes),
                     "box_conflict": round(conflict, 5),
                     "face_core": round(float((m >= 1.0).float().mean()), 5),
                     "clothes_raw": round(float(cl.mean()), 5),
                     "carrier_refined": round(float(carrier.mean()), 5)})
        print(f"  [{i+1}/{len(todo)}] {name} conflict={conflict:.4f} "
              f"carrier={rows[-1]['carrier_refined']:.4f}", flush=True)

    out = SC/"face_candidates2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    bad = [r for r in rows if r["box_conflict"] >= 0.15]
    ok = [r for r in rows if r["box_conflict"] < 0.15 and r["carrier_refined"] >= 0.05]
    print(f"\n臉框衝突 >= 0.15 被擋下 {len(bad)} 張：")
    for r in sorted(bad, key=lambda r: -r["box_conflict"]):
        print(f"  {r['image']:<34s} conflict {r['box_conflict']:.4f}")
    print(f"\n通過且精修載體 >= 0.05 的有 {len(ok)} 張，>= 0.10 的有"
          f" {sum(1 for r in ok if r['carrier_refined']>=0.10)} 張")
    for r in sorted(ok, key=lambda r: -r["carrier_refined"])[:24]:
        print(f"  {r['image']:<34s} 臉{r['n_faces']}個 衝突{r['box_conflict']:.3f}"
              f" 臉核心{r['face_core']:.3f} 載體{r['carrier_refined']:.4f}")
    print(f"\n寫出 {out}")


if __name__ == "__main__":
    main()
