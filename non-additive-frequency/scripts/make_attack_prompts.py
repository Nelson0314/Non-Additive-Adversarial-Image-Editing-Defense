"""產生 `data/attack_prompts.yaml`：三類攻擊指令 ＋ 影像清單。

衣物名詞是**逐張看圖填的**（`fig_final_set.png`）。目標色不是目測，是量出來
的：取載體區的平均色相，選 red／blue 裡離它較遠的那一個，並把選擇寫成欄位
——**不埋在句子裡**，否則跨影像比較時「目標色不同」這個變因看不見。
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import colorsys
import numpy as np, torch, yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.defense.carrier_mask import (CARRIER_CLASSES, erode_mask, exclude_boxes,
                                      face_subject_mask, feather_inward,
                                      guided_refine, parse_atr)
from src.metrics.identity import face_boxes

GARMENT = {
    "task_obj_swap_joint_mask_533428": "the jackets",
    "task_env_weather_195273":         "the robe",
    "task_obj_remove_410264":          "the shirt",
    "task_obj_remove_284852":          "the shirt",
    "task_env_weather_114555":         "the dress",
    "task_attr_mod_color_123744":      "the kimono",
    "task_obj_swap_rand_mask_92564":   "the shirts",
    "task_env_weather_70149":          "the kimono",
    "task_obj_remove_509477":          "the outfits",
    "task_obj_swap_rand_mask_303550":  "the jacket",
    "task_obj_remove_720199":          "the uniform",
    "task_env_weather_121086":         "the jacket",
    "task_env_weather_126577":         "the robe",
    "task_obj_swap_joint_mask_476374": "the vest",
    "task_attr_mod_color_202962":      "the t-shirt",
    "task_obj_remove_257910":          "the dress",
    "task_attr_mod_color_41275":       "the jacket",
    "task_attr_mod_color_6205":        "the overalls",
    "task_obj_add_709390":             "the coat",
    "task_attr_mod_color_158070":      "the coat",
}
# 看圖排除，理由留著。
DROPPED = {
    "task_attr_mod_color_42929":
        "ATR 把盾牌與金色臂鎧判成 Upper-clothes，載體覆蓋盾牌而不是衣服。"
        "沒有現成的獨立訊號能分辨「被舉著的剛體」與「衣服」，只能看圖排除。",
    "task_env_weather_262001":
        "畫面裡是猴子。ATR 的有人守門與 MTCNN 的臉部偵測都放行了。",
}
FLAGGED = {
    "task_obj_remove_720199": "載體裡混著看不出是不是制服的暗色區塊。",
    "task_obj_remove_704852": "玩具卡車有一部分被算進載體。",
    "task_obj_remove_284852": "沙發扶手可能沾到載體邊緣。",
}
# 目標色的色相（HSV 的 H，0–1）。
TARGETS = {"red": 0.0, "blue": 0.6111}


def hue_distance(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def main() -> None:
    """**這一段原本在 module 層級執行**，於是 `import` 這支腳本就會跑完整個
    實驗並覆寫結果檔。`tests/test_ip2p_run_columns.py` 的
    `test_every_driver_script_imports` 會 import 每一支 `scripts/*.py`，
    所以每跑一次測試就重寫一次結果；在缺那些資料檔的機器上則直接
    `FileNotFoundError`。工作一律放在這裡。"""
    out = {"images": {}, "dropped": DROPPED, "flagged": FLAGGED}
    for name, garment in GARMENT.items():
        base = np.asarray(Image.open(ROOT/f"data/omniedit150/{name}/{name}.png"
                                     ).convert("RGB")).astype(np.float32)/255
        x = torch.from_numpy(base).permute(2, 0, 1)[None]
        seg = parse_atr(x).numpy()
        cl = torch.from_numpy(np.isin(seg, CARRIER_CLASSES["clothes"]
                                      ).astype(np.float32))[None, None]
        bx = face_boxes(x)
        ref = exclude_boxes((guided_refine(cl, x, 4, 1e-3) > 0.5).float(), bx, margin=8)
        m = face_subject_mask(x)
        w = feather_inward(erode_mask(ref, 3), 8) * (m <= 0.0).float()
        sel = (w[0, 0] > 0.5).numpy()
        px = base[sel] if sel.sum() > 50 else base.reshape(-1, 3)
        hs = np.array([colorsys.rgb_to_hsv(*p)[0] for p in px[::7]])
        sat = np.array([colorsys.rgb_to_hsv(*p)[1] for p in px[::7]])
        # 低飽和的像素色相不可靠，用飽和度加權
        mean_h = float((hs * sat).sum() / max(sat.sum(), 1e-6))
        colour = max(TARGETS, key=lambda c: hue_distance(mean_h, TARGETS[c]))
        out["images"][name] = {
            "garment": garment,
            "carrier_area": round(float(w.mean()), 5),
            "carrier_hue": round(mean_h, 4),
            "target_colour": colour,
            "prompts": {
                "clothing":   f"turn {garment} {colour}",
                "accessory":  "add a scarf to the person",
                "background": "change the background to a snowy street",
            },
        }
        print(f"{name:<34s} {garment:<14s} 色相 {mean_h:.3f} → {colour:<5s} "
              f"載體 {out['images'][name]['carrier_area']:.4f}", flush=True)

    txt = yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=100)
    (ROOT/"data/attack_prompts.yaml").write_text(
        "# 三類攻擊指令。**兩類完全固定，只有 clothing 的衣物名詞逐張變**——\n"
        "# 跨影像比較時不可以混進「句子寫法不同」這個變因。\n"
        "#\n"
        "# 目標色由量測決定（載體區飽和度加權的平均色相，取離 red/blue 較遠者），\n"
        "# 並寫成 `target_colour` 欄位而不是只埋在句子裡。\n"
        "#\n"
        "# 受保護主體不在這裡登記：它由 ATR 的 Face+Hair 類別直接給\n"
        "# （`src/defense/carrier_mask.py` 的 `face_subject_mask`），不需要文字。\n\n"
        + txt, encoding="utf-8")
    print(f"\n寫出 data/attack_prompts.yaml：{len(out['images'])} 張、"
          f"排除 {len(DROPPED)} 張、標記 {len(FLAGGED)} 張")


if __name__ == "__main__":
    main()
