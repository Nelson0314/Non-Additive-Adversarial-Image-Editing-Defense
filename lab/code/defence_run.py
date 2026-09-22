"""只產防禦圖：各 baseline 在**自己的原生設定**上對每張影像求解，不跑編輯評測。

為什麼要跟 `baseline_run.py` 分開
────────────────────────────────────────────────────────────────────
`baseline_run.py` 把「求解」與「SDEdit 編輯評測」綁在同一個迴圈裡，於是
編輯那一段的任何問題都會擋住求解。本專案既有的作法是把攻擊與評測分開
（（`../scripts/reedit_ip2p.py`、`../scripts/readout_panel.py`、
`../frequency-phase/scripts/phase_retention.py`
都只讀已存的防禦圖，不重跑攻擊），這一支把求解端也獨立出來：
**跑完之後，編輯與讀數可以事後便宜地重跑。**

求解端用什麼 prompt
────────────────────────────────────────────────────────────────────
**攻擊指令不會進到這裡。** 本專案的威脅模型裡，編輯指令是攻擊方寫的，
防禦方看不到（`../scripts/immunise.py::assert_no_instructions` 是顏色線那邊的
同一條規則）。十一個條件的求解端各自帶著該篇自己的文字條件，逐條出處：

| 條件 | 求解端的 prompt | 出處 |
|---|---|---|
| `photoguard_c` | `""`（空字串） | 官方 notebook cell 10 實際執行的呼叫，`src/baselines/photoguard.py:114,122-125` |
| `photoguard_linf` | `""`（空字串） | 同上，兩臂只差約束種類 |
| `mist` | `"a painting"` | `mist_v3.py` 恆定值，`src/baselines/mist.py:54-55`；另吃目標影像 `../data/targets/MIST.png` |
| `dia_r` / `dia_pt` | `""`（三個分支皆空） | `attack_setting.json`，`src/baselines/dia.py:62,228-229` |
| `dct_shield` / `dct_shield_y` | 無文字條件 | 損失是 `‖E(x')‖₂`，完全不經過 text encoder |
| `dayn` / `sifm` / `danp` | 該類別的 `content` | 資料集的 `prompts.yaml`，**不是常數**，見下 |
| `diffvax` | 無文字條件 | 推論端只有 immunizer 的一次前向；prompt 只在訓練時進 `L_edit` |

**`diffvax` 與其餘十個條件不同的兩件事**（報表要分開標）：它是前饋式免疫器，
**要遮罩**（擾動只存在於重繪區之外），而且**沒有硬性 `L∞` 預算**（輸出層是
1×1 Conv、無 activation），掛不上其餘條件的失真錨點。遮罩讀
`<data>/masks/<影像>.png`，權重由 `--diffvax-ckpt` 指定。

**TDAE 不在這個表裡。** 依論文重建的模組仍在 `src/baselines/tdae.py`，但它
沒有被接進本檔的條件集合；理由見 `docs/reference/AUDIT_TDAE.md` 的
「為什麼不進本次的外部比較」。

前七個條件的字串由各 baseline 模組內部持有，呼叫端不傳、也不能傳。CSV 的
`solver_prompt` 欄逐列把實際生效的值抄出來（空字串就寫空字串），
`solver_prompt_source` 欄寫它的出處，讓表本身說得出「防禦方看到了什麼」。

**`prompts.yaml` 只有 `content` 那一個鍵會被讀，而且只在跑 DAYN、SIFM、DANP
時讀。** 這三篇的求解端都要一個文字條件，而三篇原本各自指定的來源都是攻擊
端的東西：DAYN 的 Eq. 2 要 `c_a`（要保護的內容，本來就由防禦方選）、SIFM
§VII-A 用「該圖原本的編輯 prompt」、DANP §V-A 用資料集附帶的逐張編輯指令。
本專案的威脅模型下求解端看不到編輯指令，故三者一律代入**該影像類別的
`content`**（man 類是 `man`、woman 類是 `woman`），同一個來源、同一個值，
CSV 的 `solver_prompt_source` 欄逐列寫明這件事。SIFM 與 DANP 因此偏離各自
論文的條件來源，該偏離是**本檔的接線決定**，不在那兩個模組的
`modification_note` 裡。

攻擊端的編輯指令（`edits.ip2p` / `edits.inpaint`）在任何條件下都不會進到求解
端：那是攻擊方寫的，兩者在威脅模型裡屬於不同的人。`--conditions` 不含這三個
條件時完全不碰 `prompts.yaml`（動物那一組還沒有這份檔）。

**值域。** 既有六個 PGD 條件（photoguard 兩臂、mist、dia 兩臂、dayn）在
`[-1,1]` 上求解，`sifm` 與 `danp` 在 `[0,1]`（兩篇的 Algorithm 1 就寫
`clip_{0,1}`），故同樣是 `eps=0.03`，前者的像素幅度是 0.015、後者是 0.03。
`dayn` 的 `eps=0.06` 才與這兩篇同樣是 `eps_pixel01=0.03`。`run_pgd` 自己
用 `spec.value_range.from01` / `.to01` 進出值域，呼叫端一律傳 `[0,1]` 張量、
不做換算；CSV 的 `eps` 是該篇值域的值、`eps_pixel01` 是 `[0,1]` 等價值，
**比較預算時要看後者**。

預算
────────────────────────────────────────────────────────────────────
一律用各篇的原生設定，**不在這裡調**。`eps`（該篇值域）與 `eps_pixel01`
（`[0,1]` 等價值）、`steps`、`grad_reps` 逐列寫進 CSV，出處見
`docs/reference/BASELINE_PROVENANCE.md`。

產出
────────────────────────────────────────────────────────────────────
    {out}/{image}__orig.png            原圖（求解用的 512² 版本）
    {out}/{image}__{cond}__def.png     防禦圖
    {out}/results.csv                  **逐列寫入**，不是跑完才寫

用法
    python code/defence_run.py --out images/defence_portraits/photoguard_c \\
        --conditions photoguard_c --images man_00 man_01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402
import yaml  # noqa: E402

from src.baselines import danp, dayn, dia, diffvax, mist, photoguard, sifm  # noqa: E402
from src.baselines.pgd import run_pgd  # noqa: E402
from src.baselines.dct_shield import (  # noqa: E402
    PAPER_DEFAULT_QUALITY, PAPER_EPS, PAPER_JPEG_FIG_QUALITY, PAPER_STEPS,
    DCTShieldSpec, run_dct_shield,
)
from src.metrics.standard import standard_row  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
RESOLUTION = 512

# PhotoGuard-c 的攻擊在一個指定的 img2img 強度上最佳化（Salman et al. 的
# diffusion attack），故求解端需要這個數。0.8 沿用 `../runs/baseline_restore`
# 那一批（`baseline_run.py::EDIT_STRENGTH`），改它會改掉防禦圖本身，
# 不是評測設定。ℓ∞ 臂走同一條 `attack_forward`，故同樣吃這個值。
PG_STRENGTH = 0.8

PGD_SPECS = {
    "photoguard_c": photoguard.SPEC,
    "photoguard_linf": photoguard.SPEC_PAPER_LINF,
    "mist": mist.SPEC,
    "dia_r": dia.SPEC_R,
    "dia_pt": dia.SPEC_PT,
    "dayn": dayn.SPEC_PAPER,
    "sifm": sifm.SPEC_PAPER,
    "danp": danp.SPEC_PAPER,
}
# 三個要看資料集 `content` 的條件（DAYN 的 c_a、SIFM 的 φ_t 條件、DANP 的 DAA
# 條件），故 `--conditions` 含其中之一時才讀 prompts.yaml；動物那一組還沒有
# prompts.yaml，其餘六個條件不受影響。
CONTENT_CONDITIONS = ("dayn", "sifm", "danp")
DCT_CONDITIONS = ("dct_shield", "dct_shield_y")

# DiffVax 不是逐影像最佳化：一個訓練好的 UNet++ 前向一次就吐出擾動，所以
# `BaselineSpec` 的每一欄（`eps`／`steps`／`step_size`／`update_rule`／
# `loss_fn`）在它身上都沒有對應值，不進 `PGD_SPECS`，也不進
# `src.baselines.REGISTRY`（`tests/test_baselines.py` 以 `AUDIT == REGISTRY`
# 稽核那張表）。接法照 `src/baselines/diffvax.py` 的「怎麼接進本專案的評測」：
# `load_immunizer` 載一次，之後逐張 `immunise`。
#
# 它與其餘十個條件有兩處不同，報表要分開標：
#   1. **要遮罩。** 擾動只存在於重繪區之外（論文的免疫區 `M`），沒有遮罩
#      就沒有「哪裡該加擾動」的定義。其餘條件都不吃遮罩。
#   2. **沒有硬性 `L∞` 預算。** 輸出層是 1×1 Conv、後面沒有 activation，
#      幅度由訓練時的 `L_noise`（alpha=4）壓住，不是由約束擋住。CSV 的
#      `eps`／`eps_pixel01` 因此留空，**不是 0**：填 0 會被讀成「預算為零」。
FEEDFORWARD_CONDITIONS = ("diffvax",)
CONDITIONS = tuple(PGD_SPECS) + DCT_CONDITIONS + FEEDFORWARD_CONDITIONS

#: 官方權重的預設位置。`--diffvax-ckpt` 可覆寫。
DIFFVAX_CKPT = Path.home() / "thirdparty" / "diffvax" / "diffvax_trained.pth"

#: 遮罩目錄（`../scripts/make_masks.py` 的產出，白＝重繪）相對於 `--data` 的位置。
MASK_SUBDIR = "masks"

# 逐條件的求解端文字條件與它的出處。值本身由各 baseline 模組持有，這裡只是
# 把它抄進報表——**不是**設定的來源。抄錯會被 `_check_prompts` 擋下來。
# DAYN／SIFM／DANP 的文字條件逐類別不同，值在資料集的 `content`，不是這裡的
# 常數；`solver_prompt_of` 會把實際生效的那一個字串抄進 CSV。
CONTENT_PROMPT_SOURCE = ("prompts.yaml 的逐類別 content（文字條件由防禦方選，"
                         "不從攻擊指令推；見 dayn.py 的「c_a 怎麼定位」）")

SOLVER_PROMPT = {
    "photoguard_c": ("", "官方 notebook cell 10：prompt=\"\"（photoguard.py:114,122-125）"),
    "photoguard_linf": ("", "同 photoguard_c，兩臂只差約束種類"),
    "mist": (mist.MIST_PROMPT, "mist_v3.py 恆定 'a painting'（mist.py:54-55）"),
    "dia_r": (dia.DIA_PROMPTS["uncond"], "attack_setting.json：三分支皆空字串（dia.py:62,228-229）"),
    "dia_pt": (dia.DIA_PROMPTS["uncond"], "同 dia_r"),
    "dct_shield": ("", "無文字條件：損失是 ‖E(x')‖₂，不經過 text encoder"),
    "dct_shield_y": ("", "同 dct_shield"),
    "diffvax": ("", "無文字條件：推論端只有 immunizer 的一次前向，"
                    "prompt 只在訓練時進 L_edit（diffvax.py::immunise）"),
}


def solver_prompt_of(cond: str, item: dict) -> tuple:
    """回傳這一格實際生效的求解端文字條件與它的出處。

    四個條件是常數，DAYN／SIFM／DANP 逐類別取資料集的 `content`。
    **沒有 content 就拋錯**，不退回空字串：空的文字條件仍然算得出一個解，
    但那個解保護的不是這張圖裡的東西。
    """
    if cond not in CONTENT_CONDITIONS:
        return SOLVER_PROMPT[cond]
    content = item.get("content")
    if not content:
        raise SystemExit(
            f"{cond} 需要 {item['class']} 這一類的 content（求解端的文字條件），"
            "但資料集的 prompts.yaml 沒有給")
    return content, CONTENT_PROMPT_SOURCE


def _check_prompts() -> None:
    """抄進報表的字串必須與模組持有的值相同，不同就拋錯。

    這一欄是「防禦方看到了什麼」的唯一書面證據，靜默抄錯會讓整批的威脅模型
    說不清楚，故不容許它與程式脫節。
    """
    assert SOLVER_PROMPT["mist"][0] == mist.MIST_PROMPT
    assert SOLVER_PROMPT["dia_r"][0] == dia.DIA_PROMPTS["uncond"] == ""
    assert "dayn" not in SOLVER_PROMPT, "DAYN 的 c_a 逐類別，不可寫成常數"
    assert "sifm" not in SOLVER_PROMPT, "SIFM 的 φ_t 條件逐類別，不可寫成常數"
    assert "danp" not in SOLVER_PROMPT, "DANP 的 DAA 條件逐類別，不可寫成常數"
    assert SOLVER_PROMPT["diffvax"][0] == "", "DiffVax 的推論端不吃文字條件"
    assert "diffvax" not in PGD_SPECS, "DiffVax 不是 PGD 族，不可有 spec"
    assert dayn.SPEC_PAPER.name == "dayn"
    assert sifm.SPEC_PAPER.name == "sifm"
    assert danp.SPEC_PAPER.name == "danp"
    assert photoguard.SPEC.name == "photoguard_c"
    assert photoguard.SPEC_PAPER_LINF.name == "photoguard_linf"


def content_by_class(root: Path) -> dict:
    """逐類別的 `content`（c_a）。只讀這一個鍵，**不讀 `edits`**。

    `edits` 是攻擊方寫的編輯指令，求解端看不到它（見模組 docstring）。
    """
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    return {cls: entry["content"] for cls, entry in spec.items()
            if cls != "edits" and isinstance(entry, dict) and "content" in entry}


def load_images(root: Path, names=None) -> list:
    """每類一個子目錄的版面，只讀影像本身。

    `prompts.yaml` 不讀：它放的是攻擊端的編輯指令（見模組 docstring）。
    類別由子目錄名決定，只用來標 CSV 的 `class` 欄。
    """
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name in ("masks", "headmasks", "overview"):
            continue
        for img in sorted(d.glob("*.png")):
            out.append({"name": img.stem, "class": d.name, "path": img})
    if names:
        keep = set(names)
        out = [d for d in out if d["name"] in keep]
        missing = keep - {d["name"] for d in out}
        if missing:
            raise SystemExit(f"{root} 裡找不到這些影像：{sorted(missing)}")
    return out


def solve(sd, cond: str, x01: torch.Tensor, seed: int,
          content: str = "", immunizer=None,
          mask01: "torch.Tensor | None" = None) -> tuple:
    """回傳 (防禦圖, 這一格的設定欄位)。

    預算欄全部取自 spec，不是寫死的字面值。例外只有 `pg_strength`：那個數
    論文沒有，由本檔的具名常數指定，故單獨成欄，讓表分得出哪一格是論文的
    數、哪一格是本專案的。

    `immunizer` 與 `mask01` 只有 `diffvax` 會用到，其餘條件收到也不看。
    """
    if cond in FEEDFORWARD_CONDITIONS:
        # 前饋式免疫：沒有迴圈、沒有 SD、沒有文字條件，一次前向就出擾動。
        # 遮罩的極性在本 repo 與官方一致（白＝重繪＝官方的 `mask_batch`），
        # 載入時不反相；擾動落在白區之外。
        if immunizer is None or mask01 is None:
            raise SystemExit(
                f"{cond} 需要 immunizer 與遮罩，呼叫端沒有給。"
                "遮罩在 <data>/masks/<影像>.png（白＝重繪）。")
        out = diffvax.immunise(immunizer, x01, mask01)
        cfg = {"eps": "", "eps_pixel01": "",
               "norm": "none", "steps": 1, "grad_reps": "",
               "q_alg": "", "pg_strength": "",
               "modified_from_paper": False, "modification_note": "",
               "spec_source": ("arXiv:2411.17957 ＋ 官方權重 "
                               f"{diffvax.CHECKPOINT_SHA256[:12]}…；"
                               "前饋式，無迭代、無 L∞ 預算")}
        return out["x_def01"].detach(), cfg

    if cond in DCT_CONDITIONS:
        # DCT-Shield 的兩個臂只差作用通道與 JPEG 品質因子，其餘照論文
        # Algorithm 1（`--mode paper` 的路徑，見 ../scripts/dct_shield_run.py）。
        q = PAPER_JPEG_FIG_QUALITY if cond.endswith("_y") else PAPER_DEFAULT_QUALITY
        spec = DCTShieldSpec(
            name=cond, q_alg=q, eps=PAPER_EPS, steps=PAPER_STEPS,
            channels=("Y",) if cond.endswith("_y") else ("Y", "Cb", "Cr"),
            modified_from_paper=False, modification_note="",
            source="arXiv:2504.17894 補充材料 Algorithm 1")
        x_def = run_dct_shield(sd, x01, spec, log_every=250).x_def
        cfg = {"eps": spec.eps, "eps_pixel01": "",
               "norm": "dct_coeff_linf", "steps": spec.steps, "grad_reps": 1,
               "q_alg": q, "pg_strength": "",
               "modified_from_paper": False, "modification_note": "",
               "spec_source": spec.source}
        return x_def, cfg

    spec = PGD_SPECS[cond]
    kw = {}
    if cond.startswith("photoguard"):
        kw = {"mask": None, "strength": PG_STRENGTH}
    elif cond == "mist":
        # fused 模式：兩次 VAE 編碼與一次完整 UNet 前向在同一張圖上，
        # 不開 checkpoint 會 OOM。目標影像是該篇自己的 MIST.png。
        kw = {"use_ckpt": True, "vae_ckpt": True,
              "target01": load_image_tensor(paths.TARGETS / "MIST.png",
                                            sd.device, size=RESOLUTION)}
    elif cond.startswith("dia"):
        # DIA 把整條反演（R 再加整條重建）留在同一張圖上，兩個開關都要開。
        kw = {"use_ckpt": True, "vae_ckpt": True}
    elif cond == "dayn":
        # 100 步 × 10 個時刻 = 1000 次帶梯度的 UNet 前向，不開 checkpoint 會 OOM。
        # `content` 是 Eq. 2 的 c_a；本檔不傳 prompt，dayn.prepare 也擋掉它。
        kw = {"use_ckpt": True, "content": content}
    elif cond == "sifm":
        # `sifm.prepare` 的文字條件參數名是 `prompt`（式 (3)(4) 的 c），收到
        # None 會拋 ValueError。論文 §VII-A 用該圖原本的編輯 prompt，本專案的
        # 威脅模型下求解端看不到指令，故與 DAYN 同源、同值地用該類別的 content。
        #
        # **不開 checkpoint。** SIFM 用 forward hook 取中間特徵，而 checkpoint
        # 區塊的前向在 no_grad 下執行，hook 拿到的張量不在計算圖上，梯度會靜默
        # 變成零而輸出仍是一張合理的防禦圖（見 sifm.py「為什麼不提供 use_ckpt」）。
        # `sifm.prepare` 也沒有這個參數。
        kw = {"prompt": content}
    elif cond == "danp":
        # `danp.prepare` 的參數名同樣是 `prompt`，收到 None 會拋
        # NotImplementedError。Eq. 3 的 K_l 來自 φ(c)、Eq. 10/11 的遮罩與損失
        # 都以它為條件；論文 §V-A 用資料集附帶的逐張編輯指令，本專案改用該
        # 類別的 content，與 DAYN 同一個來源。
        # `use_ckpt` 要開：一次 loss_fn 有兩次完整 UNet 前向（其中一次在
        # no_grad 內），不開會 OOM；DANP 不用 hook，checkpoint 不影響梯度。
        kw = {"prompt": content, "use_ckpt": True}
    x_def = run_pgd(sd, x01, spec, seed=seed, **kw).x_adv01.detach()
    cfg = {"eps": spec.eps, "eps_pixel01": spec.eps_pixel01,
           "norm": spec.norm, "steps": spec.steps, "grad_reps": spec.grad_reps,
           "q_alg": "", "pg_strength": PG_STRENGTH if cond.startswith("photoguard") else "",
           "modified_from_paper": spec.modified_from_paper,
           "modification_note": spec.modification_note,
           "spec_source": spec.source}
    return x_def, cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diffvax-ckpt", type=Path, default=DIFFVAX_CKPT,
                    help="DiffVax 官方權重（只有 --conditions 含 diffvax 時會讀）")
    ap.add_argument("--tag", default="",
                    help="CSV 檔名的後綴，寫成 results_<tag>.csv。"
                         "同一個條件目錄被多個行程（不同影像子集）同時寫入時"
                         "必須給，否則兩個行程的 results.csv 會互相覆蓋——"
                         "PNG 不會，因為影像名不同。")
    args = ap.parse_args()

    _check_prompts()
    conds = args.conditions or list(CONDITIONS)
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"沒有這些條件：{unknown}，可用的是 {list(CONDITIONS)}")
    args.out.mkdir(parents=True, exist_ok=True)

    csv_path = args.out / (f"results_{args.tag}.csv" if args.tag else "results.csv")
    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    items = load_images(args.data, args.images)
    if any(cond in CONTENT_CONDITIONS for cond in conds):
        contents = content_by_class(args.data)
        for item in items:
            item["content"] = contents.get(item["class"], "")

    # 前饋式條件：權重載一次（不是逐張），遮罩逐張在下面讀。缺任何一張的
    # 遮罩就當場停住，不跳過——少一張的批次看起來仍然是完整的一批。
    immunizer = None
    if any(cond in FEEDFORWARD_CONDITIONS for cond in conds):
        immunizer = diffvax.load_immunizer(args.diffvax_ckpt, device=sd.device)
        missing = [d["name"] for d in items
                   if not (args.data / MASK_SUBDIR / f"{d['name']}.png").is_file()]
        if missing:
            raise SystemExit(
                f"{args.data / MASK_SUBDIR} 裡缺這些遮罩：{missing}。"
                "DiffVax 的擾動只存在於重繪區之外，沒有遮罩就沒有定義。")

    print(f"影像 {[d['name'] for d in items]}｜條件 {conds}", flush=True)

    rows = []
    for item in items:
        x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
        save_image(x01, args.out / f"{item['name']}__orig.png")
        mask01 = None
        if immunizer is not None:
            mask01 = load_image_tensor(
                args.data / MASK_SUBDIR / f"{item['name']}.png",
                sd.device, size=RESOLUTION)
        for cond in conds:
            prompt, prompt_src = solver_prompt_of(cond, item)
            t0 = time.time()
            x_def, cfg = solve(sd, cond, x01, args.seed,
                               content=item.get("content", ""),
                               immunizer=immunizer, mask01=mask01)
            secs = time.time() - t0
            save_image(x_def.clamp(0, 1), args.out / f"{item['name']}__{cond}__def.png")
            m = suite.pairwise(x01, x_def.clamp(0, 1))
            rows.append({
                "image": item["name"], "class": item["class"], "condition": cond,
                "solver_prompt": prompt, "solver_prompt_source": prompt_src,
                "seed": args.seed, **cfg,
                "total_seconds": round(secs, 1),
                "fid_psnr": round(float(m["psnr"]), 4),
                "fid_lpips": round(float(m["lpips"]), 4),
                "fid_rms": round(float(m["rms"]), 6),
                "fid_linf": round(float(m["linf"]), 6),
                **standard_row("fid_", m),
                "data_root": str(args.data).replace("\\", "/"),
            })
            write_csv(csv_path, rows)
            print(f"[DONE] {item['name']:12s} {cond:16s} "
                  f"psnr={m['psnr']:.3f} lpips={m['lpips']:.4f} "
                  f"rms={m['rms']:.5f} ({secs:.0f}s)", flush=True)
    print(f"\n[ALLDONE] 表：{args.out / 'results.csv'}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
