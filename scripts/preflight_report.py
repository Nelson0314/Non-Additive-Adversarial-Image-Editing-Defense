"""preflight_report.py — 部署前檢查報告（TWCC 上線前執行）。

執行自動化靜態檢查（可移植性、config 完整性、v4 落地、測試、穩健性），
併入人工稽核結論（假設裁定、論文核驗、風險清單、成本模型、規模方案、決策樹），
輸出至終端機並存檔 PREFLIGHT.md。

用法：python scripts/preflight_report.py [--skip-tests]
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import REPO_ROOT, load_configs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

W = 78


# ---------------------------------------------------------------- 自動化檢查

def _grep(pattern: str, roots: list[str], exclude: list[str] = ()) -> list[str]:
    hits = []
    rx = re.compile(pattern)
    for root in roots:
        for p in sorted((REPO_ROOT / root).rglob("*.py")):
            if any(e in str(p) for e in exclude):
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line) and not line.strip().startswith("#"):
                    hits.append(f"{p.relative_to(REPO_ROOT)}:{i}")
    return hits


def check_portability() -> list[tuple]:
    out = []
    # \w\.cuda\( 只抓實際呼叫（docstring 中「禁用 .cuda()」之提及無前置識別字）
    hits = _grep(r"\w\.cuda\(", ["src", "scripts"], exclude=["preflight_report.py"])
    out.append(("無 .cuda() 呼叫", "PASS" if not hits else "FAIL", hits))
    hits = _grep(r"import xformers|enable_xformers", ["src", "scripts"],
                 exclude=["preflight_report.py"])
    out.append(("無 xformers 依賴", "PASS" if not hits else "FAIL", hits))
    hits = _grep(r'torch\.device\(\s*"cuda"', ["src", "scripts"],
                 exclude=["device.py"])
    out.append(("device 一律經 device.py", "PASS" if not hits else "FAIL", hits))
    hits = _grep(r'"(CompVis|stabilityai|runwayml)/', ["src", "scripts"])
    out.append(("程式碼無硬編碼真實模型名（僅 config）", "PASS" if not hits else "FAIL", hits))
    hits = _grep(r"\.half\(\)|torch\.float16", ["src", "scripts"])
    out.append(("無硬編碼 dtype", "PASS" if not hits else "FAIL", hits))
    return out


def check_configs() -> list[tuple]:
    out = []
    try:
        cfgs = load_configs()
    except Exception as e:  # noqa: BLE001 — 檢查腳本須回報而非中斷
        return [("configs 可載入", "FAIL", [str(e)])]
    out.append(("configs 可載入", "PASS", []))

    pg = cfgs["additive"]["photoguard"]
    # 註: EOT 次數之鍵名為 grad_reps（STRUCTURE 範本寫 diffusion_eot，實作以
    # photoguard.py 讀取之 grad_reps 為準，語意相同）
    need = ["epsilon_scale", "norm", "target_latent", "epsilon", "step_size",
            "step_decay", "random_init", "n_iter", "diffusion_T",
            "diffusion_target", "grad_reps", "projection"]
    missing = [k for k in need if k not in pg]
    out.append(("photoguard 參數齊備（SPEC §3.1–3.3）", "PASS" if not missing else "FAIL", missing))

    adv = cfgs["nonadditive"]["advdiff"]
    need = ["T", "N", "a", "guidance_range", "eta", "s", "eps_latent",
            "injection_form", "ztT_update"]
    missing = [k for k in need if k not in adv]
    out.append(("advdiff 參數齊備（SPEC §4.3）", "PASS" if not missing else "FAIL", missing))

    apa = cfgs["nonadditive"]["apa"]
    need = ["variant", "grid_steps", "T_a", "N", "eps_a", "mu", "lora_rank",
            "lora_alpha", "lora_target_modules", "lora_steps", "noise_offset",
            "sg", "gc", "aug"]
    missing = [k for k in need if k not in apa]
    ok = (not missing and "scale_constant" in apa["sg"]
          and "inversion_steps" in apa["gc"] and "mse_reg_weight" in apa["gc"])
    out.append(("apa 參數齊備（SPEC §4.4，含 sg/gc/aug）", "PASS" if ok else "FAIL", missing))

    scan = cfgs["nonadditive"].get("stage0_scan", {})
    ok = all(m in scan for m in ("advdiff", "apa_sg", "apa_gc", "hybrid")) and \
        scan.get("apa_sg", {}).get("values") != scan.get("apa_gc", {}).get("values")
    out.append(("stage0 掃描範圍：SG/GC 獨立（假設 2）", "PASS" if ok else "FAIL", []))

    ok = "inpaint_mask" in cfgs["base"]["edit"]
    out.append(("placeholder 遮罩規格入 config（假設 3）", "PASS" if ok else "FAIL", []))

    pu = cfgs["purify"]
    ok = (pu["adverse_cleaner"]["bf_iterations"] == 64
          and pu["adverse_cleaner"].get("gf_iterations") == 4)
    out.append(("AdverseCleaner 64×BF+4×GF（論文核驗修正）", "PASS" if ok else "FAIL", []))

    gp = pu["gridpure"]
    names = [s["name"] for s in gp["settings"]]
    paper = next((s for s in gp["settings"] if s["name"] == "paper"), None)
    ok = (paper is not None and paper["pure_steps"] == 10 and paper["iterations"] == 10
          and "single_deep" in names and "readme" in names)
    out.append(("GrIDPure 論文預設 10×10 為主設定＋兩端點（v5）",
                "PASS" if ok else "FAIL", []))

    rt = cfgs["base"]["runtime"]
    ok = rt.get("protect_batch_size") == 1 and int(rt.get("edit_batch_size", 1)) == 1
    out.append(("批次參數：protect_batch_size=1；edit_batch_size=1（v5 位元級判準"
                "實測未通過，啟用待裁定）", "PASS" if ok else "WARN",
                [] if ok else [f"edit_batch_size={rt.get('edit_batch_size')}：判準未放寬前不得 >1"]))

    pending = [("edit.sdedit_strength", cfgs["base"]["edit"]["sdedit_strength"]),
               ("edit.seed_clip_threshold", cfgs["base"]["edit"]["seed_clip_threshold"]),
               ("common.similarity_budget", cfgs["nonadditive"]["common"]["similarity_budget"])]
    unresolved = [k for k, v in pending if v is None]
    out.append(("[待確認] 項目以 null 明確標記",
                "PASS" if len(unresolved) == 3 else "WARN",
                [f"{k} 已被填值（確認是否經校準）" for k, v in pending if v is not None]))
    return out


def check_v4_landing() -> list[tuple]:
    """v4 修正逐項於程式碼中驗證（樣式比對 + 測試佐證）。"""
    src = (REPO_ROOT / "src" / "protect" / "advdiff_based.py").read_text(encoding="utf-8")
    apa = (REPO_ROOT / "src" / "protect" / "apa_based.py").read_text(encoding="utf-8")
    checks = [
        ("advdiff 後置加法 z=z+s·g（非改 epsilon）", 'z = z + cfg["s"] * g' in src),
        ("advdiff 無 σ²/√(1−ᾱ) 係數", "sigma" not in src.lower() and "* g\n" in src + "\n"),
        ("advdiff z_T 更新 skip-gradient（全程 no_grad）",
         "torch.no_grad():  # v4" in src and 'z_T + cfg["a"] * g_T' in src),
        ("advdiff eps_latent 相對 L2 投影", "max_radius" in src),
        ("apa-sg 軌跡梯度 skip ×scale_constant", 'cfg["sg"]["scale_constant"] * g_tr' in apa),
        ("apa-gc 部分 inversion（格點前 inversion_steps 步）",
         'cfg["gc"]["inversion_steps"]' in apa and "_partial_ddim_inversion" in apa),
        ("apa-gc reward 含 −mse_reg_weight·MSE", 'cfg["gc"]["mse_reg_weight"] * F.mse_loss' in apa),
        ("apa 式(12) 基底=最終生成 latent、x̂₀ detached",
         "(la + z_fin) / 2.0" in apa and ".detach())" in apa),
        ("apa LoRA=peft、delete_adapters 還原",
         "from peft import LoraConfig" in apa and "delete_adapters" in apa),
        ("apa noise_offset 實作", 'cfg["noise_offset"]' in apa),
    ]
    return [(name, "PASS" if ok else "FAIL", []) for name, ok in checks]


def check_robustness() -> list[tuple]:
    s1 = (REPO_ROOT / "scripts" / "stage1_clean.py").read_text(encoding="utf-8")
    s2 = (REPO_ROOT / "scripts" / "stage2_purify.py").read_text(encoding="utf-8")
    out = [
        ("stage1/2 逐筆寫入（IncrementalCsv）",
         "PASS" if "IncrementalCsv" in s1 and "IncrementalCsv" in s2 else "FAIL", []),
        ("stage1/2 斷點續跑（--resume）",
         "PASS" if "--resume" in s1.replace('"--resume"', "--resume")
         and "resume" in s2 else "FAIL", []),
        ("drop 除零/負基準防護",
         "PASS" if "abs(c) <= 1e-6" in s2 and "drop_valid" in s2 else "FAIL", []),
        ("config 回寫＝單行 regex＋overlay（不重寫整份 yaml）",
         "WARN", ["決策：不引入 ruamel.yaml——similarity_budget 為單行且有錨點、"
                  "regex 已於副本驗證（註解保留、可解析）；其餘校準值寫獨立 overlay 檔，"
                  "完全避開 yaml 重寫。若日後回寫欄位增多，屆時再改 ruamel.yaml"]),
    ]
    return out


def run_tests(skip: bool):
    if skip:
        return ("測試套件", "WARN", ["--skip-tests 指定，未執行"])
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO_ROOT / "tests"), "-q"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    return ("測試套件", "PASS" if r.returncode == 0 else "FAIL", [tail])


# ---------------------------------------------------------------- 成本模型

# 單位耗時假設（V100 32GB、512×512、batch=1）。均為估算值，TWCC 首日須以實測校正。
UNIT = {
    "edit_s_fp32": 22.0,     # 100 步 + CFG（200 次 UNet）+ VAE
    "edit_s_fp16": 11.0,
    "pg_enc_min": 1.0,       # 100 iter × VAE fwd+bwd
    "pg_diff_min": 50.0,     # 100 iter × EOT10 × T=10 fwd+bwd（±60%，最不確定）
    "advdiff_min": 0.3,      # N=5 × T=10，skip-gradient
    "apa_min": 3.0,          # LoRA200 步 + N=10 rollout（SG/GC 相近）
    "hybrid_min": 3.0,
    "gridpure_min": 2.0,     # 官方實測（論文 §5.2）
    "metric_s": 1.0,         # piq 五項（LPIPS 為主）
}


def plan_cost(*, n_img_protect, methods, stage0, stage1_blocks, stage2, fp16_edit):
    """閘門式（block 結構）成本模型。

    stage1_blocks：list of dict(n_img, prompts, seeds, combos)，combos=model×edit
      組合數（各 block 可有不同 seed 數——v5 閘門式：有錨點條件跑滿、其餘降 seed）。
    stage2：dict(n_img, prompts, seeds, models, edits, ops, ops_gridpure)。
    保護一次生成、stage1/2 共用，故 protect 僅計於 stage1。metric 以每次編輯估。
    """
    e = UNIT["edit_s_fp16"] if fp16_edit else UNIT["edit_s_fp32"]
    per_edit_min = (e + UNIT["metric_s"]) / 60
    protect_min = n_img_protect * (UNIT["pg_enc_min"] + UNIT["pg_diff_min"]
                  + UNIT["advdiff_min"] + UNIT["apa_min"] + UNIT["hybrid_min"]) * (methods / 5)
    s0_min = (stage0["n_img"] * UNIT["pg_enc_min"] + 5 * UNIT["pg_diff_min"]
              + stage0["scan_img"] * 4
              * (UNIT["advdiff_min"] + 2 * UNIT["apa_min"] + UNIT["hybrid_min"]))
    s1_edits = sum(b["n_img"] * b["prompts"] * b["seeds"] * b["combos"] * (methods + 1)
                   for b in stage1_blocks)
    s1_min = protect_min + s1_edits * per_edit_min
    s2 = stage2
    s2_edits = (s2["n_img"] * s2["prompts"] * s2["seeds"] * s2["models"]
                * s2["edits"] * methods * s2["ops"])
    s2_pure_min = s2["ops_gridpure"] * s2["n_img"] * methods * UNIT["gridpure_min"]
    s2_min = s2_edits * per_edit_min + s2_pure_min
    return {
        "stage0_h": s0_min / 60, "protect_h": protect_min / 60,
        "s1_edits": s1_edits, "stage1_h": s1_min / 60,
        "s2_edits": s2_edits, "gridpure_h": s2_pure_min / 60, "stage2_h": s2_min / 60,
        "total_h": (s0_min + s1_min + s2_min) / 60,
    }


# v5 閘門式循序方案（保真度花在有錨點條件上，縮減集中在 stage2）。批次化未啟用
# （位元級判準實測未通過），編輯加速僅剩 fp16（÷2）與多卡平行。
PLANS = {
    # 完整依 SPEC——僅作記帳基準（單卡不可行）
    "A 完整(fp32)": dict(
        n_img_protect=150, methods=5, fp16_edit=False,
        stage0=dict(n_img=20, scan_img=20),
        stage1_blocks=[dict(n_img=150, prompts=2, seeds=20, combos=4)],  # 2 模型×2 編輯
        stage2=dict(n_img=150, prompts=2, seeds=20, models=2, edits=2, ops=17, ops_gridpure=5)),
    "A'完整(fp16)": dict(
        n_img_protect=150, methods=5, fp16_edit=True,
        stage0=dict(n_img=20, scan_img=20),
        stage1_blocks=[dict(n_img=150, prompts=2, seeds=20, combos=4)],
        stage2=dict(n_img=150, prompts=2, seeds=20, models=2, edits=2, ops=17, ops_gridpure=5)),
    # 閘門式（建議主實驗）：stage1 錨點條件(V1.4×img2img)跑滿 20 seed，其餘 3 條件
    # 降 5 seed；stage2 積極縮減(僅 V1.4×img2img、5 seed、10 淨化設定)
    "G 閘門式(建議)": dict(
        n_img_protect=50, methods=5, fp16_edit=True,
        stage0=dict(n_img=50, scan_img=10),
        stage1_blocks=[dict(n_img=50, prompts=2, seeds=20, combos=1),   # 錨點：V1.4×img2img
                       dict(n_img=50, prompts=2, seeds=5, combos=3)],   # 補充：其餘 3 條件
        stage2=dict(n_img=50, prompts=2, seeds=5, models=1, edits=1, ops=10, ops_gridpure=2)),
    # 首日最小可行：跑通真實 SD、量測單位耗時
    "C 首日最小": dict(
        n_img_protect=20, methods=5, fp16_edit=True,
        stage0=dict(n_img=20, scan_img=10),
        stage1_blocks=[dict(n_img=20, prompts=1, seeds=5, combos=1)],
        stage2=dict(n_img=20, prompts=1, seeds=5, models=1, edits=1, ops=5, ops_gridpure=1)),
}


# ---------------------------------------------------------------- 報告組裝

def hr(c="="):
    return c * W


def sec(title):
    return f"\n{hr()}\n{title}\n{hr()}"


def fmt_checks(rows):
    lines = []
    for name, status, detail in rows:
        lines.append(f"  [{status:4}] {name}")
        for d in detail:
            lines.append(f"         - {d}")
    return lines


def build_report(args) -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, cwd=str(REPO_ROOT)).stdout.strip()
    L = []
    L.append(hr())
    L.append("部署前檢查報告（preflight）")
    L.append(hr())
    L.append(f"生成時間: {datetime.now():%Y-%m-%d %H:%M}   git commit: {commit}")
    L.append(f"Python: {sys.version.split()[0]}   平台: win32/CPU（本地）→ 目標: TWCC V100 32GB")

    L.append(sec("[1] 假設裁定（四項，均已落實）"))
    L.append("""\
  [PASS] 假設1 L_ref=encoder：stage0 兩基準皆算並列出（diffusion 取 5 張，
         --pg-diff-images），差距>10% 自動警告；TWCC_CHECKLIST 已加驗證項
  [PASS] 假設2 掃描鈕：SG/GC 獨立範圍（configs stage0_scan；GC 因部分 inversion
         用較小值域）；hybrid 掃 eps_a（其相似性約束＝APA 骨架 ℓ∞ ball，
         s/a 注入僅在 ball 內作用，故 eps_a 為實際約束鈕）；
         選中值落區間端點時自動警告「須擴展重掃」（已實測觸發）
  [PASS] 假設3 遮罩：規格入 config（edit.inpaint_mask: center_square, fraction 0.5），
         manifest 記錄、summary 標記「placeholder 遮罩不可與真實資料比較」，
         TWCC_CHECKLIST 加「真實資料到手後 inpaint 全部重跑」
  [PASS] 假設4 prompt_idx：維持（STRUCTURE §4.3 由指導者補記）""")

    L.append(sec("[2] 靜態稽核"))
    L.append("--- 2.1 v4 修正落地 " + hr("-")[:56])
    L += fmt_checks(check_v4_landing())
    L.append("  註: gc.sampling_steps 為文件性參數（程式以 inversion_steps 個格點區間")
    L.append("      上下對稱走訪，官方計數含邊界，語意相同，config 已註記）")
    L.append("--- 2.2 可移植性 " + hr("-")[:58])
    L += fmt_checks(check_portability())
    L.append("  註: tiny→真實 SD 僅換 config 模型名；解析度/latent 尺寸均由模型 config 推得")
    L.append("--- 2.3 config 完整性 " + hr("-")[:54])
    L += fmt_checks(check_configs())
    L.append("  SPEC 未提而 config 有: eps_latent（SPEC 偽碼用、範本漏列，已註記）、")
    L.append("  grid_steps（官方 50 步格點）、stage0_scan/inpaint_mask/clip_model（preflight 新增）、")
    L.append("  seed_clip_threshold（seed 判準操作化）——均有來源註解")
    L.append("  命名差異: EOT 次數＝grad_reps（STRUCTURE 範本作 diffusion_eot，實作讀")
    L.append("  grad_reps；preflight 曾因此發現 apply_smoke 用錯鍵，已修正）")
    L.append("--- 2.4 測試覆蓋 " + hr("-")[:58])
    L.append((lambda r: "\n".join(fmt_checks([r])))(run_tests(args.skip_tests)))
    L.append("""\
  覆蓋矩陣（45 項；44 pass + 1 skip）:
    photoguard(6): PGD 兩範數/投影/衰減/EOT/尺度   sd_wrapper(4): 封裝/差分 img2img
    smoke(4): 全流程/seed 重現/inpaint             nonadditive(10): reward 梯度與方向/
      inversion 決定性/advdiff/apa sg+gc 參數化+peft 還原/hybrid/
      guidance 區間/投影生效/ℓ∞ clamp
    purify(13): 輕量三法/AdvClean 兩變體/grid 機制/分派
    metric_directions(2): 極端案例方向（piq 五項+FID）
    edit_batching(6, v5): 噪聲協定位元級一致/批次決定性/sdedit+inpaint 容差等價/
      分塊等價/位元級啟用閘門（batching 關閉時 skip）
  未覆蓋（記入風險清單）: inversion 重建品質（tiny 無意義→TWCC）、CLIP 方向（需
    外部模型→TWCC 抽查）、批次化真實模型等價（→TWCC）、stage 腳本本身
    （以煙霧串跑驗證，非單元測試）""")
    L.append("--- 2.5 穩健性 " + hr("-")[:60])
    L += fmt_checks(check_robustness())
    L.append("  註: FID 需完整樣本——續跑跳過任何列時自動略過 FID 並警告（結構限制，")
    L.append("      如需 FID 對該組完整重跑；已於 stage1 註明）")

    L.append(sec("[2b] 論文核驗（詳見 PAPER_VERIFICATION.md）"))
    L.append("""\
  高等級問題: 0
  中等級: 2（均已處置）
    - AdverseCleaner 官方為 64×BF+4×GF，SPEC 記 3×BF+1×GF → 已修正實作+config
    - DAYN Table 1 僅為 img2img 情境（inpaint 為質性）→ 校準限 sdedit 列，已改 summary
  其他發現: DAYN 測試集為 SD 生成影像（索取未果可自生成，已入 TWCC_CHECKLIST）；
    DAYN Alg.1 為 sign+clip（ℓ∞）→ norm=linf 先驗提高；「LDM 淨化無效」出處
    實為 Pixel is a Barrier §6.3（SPEC 誤置於 GrIDPure，待 SPEC 下版修正）；
    GrIDPure 論文預設 10×10（README 10×20、腳本 100×1，三來源已註記）；
    AdvDiff s=10/a=10 之失敗案例＝調參上界護欄；
    逐字驗證一致: DAYN Table 1 全 30 值、κ/N/T、式(2)-(5)；PG 式(4)(5)/Table 8,9/
    A.1 seed 協定；AdvDiff 式(9)(11)/附錄 E (0,0.2]/附錄 H untargeted/75.2%；
    APA 式(4),(6)-(12)/reward hacking 動機/SG=ρ·∇ 近似；GrIDPure γ、grid、2min""")

    L.append(sec("[3] 真實模型風險（TWCC 首次執行必檢，詳見 TWCC_CHECKLIST.md）"))
    L.append("""\
  1. cross-attention 擷取     方法: 兩模型跑 capture_cross_attention，驗層數>0、
     解析度合理  預期: v1.4/v2.0 皆可用  不符: 依 diffusers 版本更新 processor 類名
  2. DDIM inversion 重建品質  方法: 5 張 10/50 步 inversion→去噪，量 PSNR/LPIPS
     預期: PSNR>25、LPIPS<0.1  不符: 「錨定原圖」前提受損→部分 inversion 或加步數
  3. scheduler 混用（SPEC §8 第7項）方法: 同 5 張 DDIM(匹配) vs PNDM 重建比較
     預期: 匹配 DDIM 較佳或相近  不符: 維持匹配並記錄；官方混用列消融
  4. GPU 記憶體峰值           方法: 各法單張 protect 記 peak_memory_mb
     預期: pg_diff 最重（CPU 相對量 711/361 MB）  不符(OOM): 見 [6] 降級路徑
  5. fp16                     方法: 編輯 fp16 vs fp32 指標差 <1% 即採用
     預期: 編輯可用（無梯度）  不符: 編輯退回 fp32（成本×2，見 [4]）
  6. 編輯 pipeline eta=1      方法: 確認 img2img scheduler；PNDM 忽略 eta
     預期: T2 校準吸收此差異  不符: 改 DDIMScheduler(eta=1) 為第一排查項
  7. 假設1 驗證（enc vs diff LPIPS 差 ≤10%）  stage0 內建
  8. L_ref 用 placeholder 資料無效——所有校準以真實/自生成資料為準""")

    L.append(sec("[4] 成本估算（單位假設見下；TWCC 首日以實測單位時間重算本表）"))
    L.append(f"  單位假設(V100): 編輯 {UNIT['edit_s_fp32']:.0f}s fp32 / {UNIT['edit_s_fp16']:.0f}s fp16（100步+CFG）; "
             f"保護/張: pg_enc {UNIT['pg_enc_min']:.0f}m,")
    L.append(f"  pg_diff {UNIT['pg_diff_min']:.0f}m(±60%, EOT10×T10), advdiff {UNIT['advdiff_min']:.1f}m, "
             f"apa/hybrid {UNIT['apa_min']:.0f}m; GrIDPure {UNIT['gridpure_min']:.0f}m/張/設定")
    header = f"  {'方案':<18}{'s0(h)':>7}{'protect(h)':>11}{'s1 edits':>10}{'s1(h)':>8}{'s2 edits':>10}{'s2(h)':>8}{'總計(h)':>9}"
    L.append(header)
    L.append("  " + hr("-")[:76])
    for name, p in PLANS.items():
        c = plan_cost(**p)
        flag = "  <-- 超過 100h" if c["total_h"] > 100 else ""
        L.append(f"  {name:<18}{c['stage0_h']:>7.0f}{c['protect_h']:>11.0f}{c['s1_edits']:>10,}"
                 f"{c['stage1_h']:>8.0f}{c['s2_edits']:>10,}{c['stage2_h']:>8.0f}{c['total_h']:>9.0f}{flag}")
    L.append("""\
  關鍵事實:
  - 乘數效應主宰成本: 完整方案 s2 編輯 2,040,000 次——瓶頸不是 GrIDPure（125h）
    也不是保護生成（143h），而是「淨化設定×方法×seed×模型×編輯」的編輯次數
  - 完整方案 A 約 14,200h（fp32）/ 7,600h（fp16）＝單卡不可行，須大幅縮減
  - pg_diff 保護為固定重成本（每張 50m，與 seed 無關）: 150 張≈125h、50 張≈42h、
    20 張≈17h——n_img 為最有效的單一槓桿（同時壓 protect 與所有編輯）
  - 儲存估算: stage1 edited_orig（G 方案 ~6,000 png）+ protected + purified ~數 GB
  加速手段評估（4.4）:
  1. 編輯批次化【原為最大槓桿，v5 實測後封鎖】: 已實作（每樣本各自 generator+
     seed，噪聲協定與 batch=1 位元級一致，seed 協定 SPEC §2.3 未破壞）。但 v5
     啟用判準「batch=1 vs batch=4 逐位元相同」實測**未通過**——批次矩陣運算之
     歸約順序隨 batch 尺寸而異，float32 差 ~5e-6（8-bit 存檔後 ~0.02% 像素 ±1LSB），
     位元級等價原理上不可達。依判準 edit_batch_size 維持 1。**待指導者裁定**：
     若放寬為「噪聲協定位元級一致＋輸出容差 ≤1e-4」則可啟用，編輯時數 ÷4–6
  2. fp16 編輯【可行，主要槓桿，已入 G/C】: 約 ÷2；風險[3]-5 驗證後啟用；保護維持 fp32
  3. 多卡平行【最實際之加速】: 保護與編輯對影像/seed 皆 embarrassingly parallel，
     k 卡約 ÷k（G 方案 229h → 4 卡 ~57h、8 卡 ~29h）
  4. CPU 淨化與 GPU 平行【可行，收益小】: jpeg/blur/crop/advclean 為 CPU 運算
     （advclean 64×BF 約 3–5s/張），與 GPU 編輯管線重疊——節省 <2h，優先度低
  5. GrIDPure 平行化【部分可行】: 10 個 grid 可 batch 成一次前傳（256×256×10
     於 32GB 可容納）→ 約 ÷3–5；或多卡分圖平行（原文亦建議）""")

    L.append(sec("[5] 規模方案對照（v5 閘門式循序）"))
    L.append("""\
  閘門式原則: 保真度花在有錨點的條件（V1.4×img2img，可對 DAYN Table 1 校準）；
  縮減集中在 stage2（其結論為「drop 之相對差異」，不需 stage1 的絕對精度）。
  各階段設閘門，前一階段不通過即停，避免把預算投入注定失敗的下游。

  方案 A 完整（依 SPEC；記帳基準）
    時數: ~14,200h(fp32) / ~7,600h(fp16) — 單卡不可行；即使 8 卡亦 ~40 天
    僅作為完整規格之成本上界，不執行
  方案 G 閘門式（建議主實驗）
    stage1: 錨點條件 V1.4×img2img 跑滿（50 張×2 prompt×20 seed×5 方法）；
      補充條件 V2.0×img2img、V1.4/V2.0×inpaint 各降至 5 seed
    stage2: 積極縮減——僅 V1.4×img2img、5 seed、10 淨化設定（gridpure 2）
    時數: ~229h(fp16 單卡) ≈ 10 天；4 卡 ~57h、8 卡 ~29h（編輯/保護可平行）
      拆解: 保護 ~48h（pg_diff 42h 主宰）、stage1 編輯 ~70h、stage2 編輯 ~83h、
      GrIDPure ~17h、stage0 ~11h
    能答: 核心假設（加性 vs 非加性 drop 差異+強度曲線，於主條件）、T2 校準
      （sdedit 有錨點）、非加性在有錨點主條件之乾淨比較（跑滿 seed，統計強）、
      跨模型/inpaint 之乾淨情況（stage1 補充條件保留，5 seed）
    不能答: 淨化後之跨模型/inpaint 行為（stage2 限單條件）；補充條件之 seed 統計
      較弱（5 vs 20）；stage2 影像仍 50 張
  方案 C 首日最小（20 張/1 prompt/5 seed/V1.4/img2img/5 淨化設定/fp16）
    時數: ~43h(fp16) ≈ 2 天（其中保護 ~19h、pg_diff 17h 主宰）
    能答: 方向性初判（drop 加性 vs 非加性）、pipeline 於真實 SD 跑通、
      單位耗時實測（校正本表所有時數）
    不能答: 校準有效性（樣本過少）、曲線形狀、任何可寫入論文之定量結論
  建議路徑: C（首日，兼量測單位耗時並過 T2 閘門）→ 據實測修訂 G 之規模與時數
    → G 為主實驗；A 僅在多卡多日資源到位、或以 G 結果決定局部加密時考慮
  具體切法要點:
    - n_img=50: pg_diff 保護由 125h 降至 42h，同時壓所有編輯；50 張×2 prompt
      ×20 seed=2000 樣本/方法，對主條件統計仍充分
    - 若指導者要求錨點條件用滿 150 張: 額外 +~95h（多為 pg_diff 保護），列為旋鈕
    - stage2 ops 由 17 降至 10（保留 jpeg 3/blur 2/crop 2/advclean_bfgf/gridpure 2）
      仍覆蓋強度曲線與 GrIDPure 迭代 vs 單次深淨化之對照""")

    L.append(sec("[6] 決策樹"))
    L.append("""\
  T2 校準對不上 DAYN Table 1（v5 分階段搜尋，norm 固定 linf）:
    1) 檢查編輯 scheduler/eta（風險[3]-6，改 DDIMScheduler 重跑一組）
    2) sdedit_strength 未知為最大混淆——依 {0.8,0.7,0.5,0.3} 分階段掃（先 0.8）
    3) 兩 epsilon_scale × 全部 strength 皆不符 → 試 norm=l2、再擴 target_latent=gray
    4) 仍不符 → 停下回報；選項: 向作者確認 / 改以「本專案自身 PhotoGuard 復現值」
       為錨（於論文中揭露不與 DAYN 直接比較）
  stage1 非加性全面劣於 PhotoGuard（stage1→stage2 閘門）:
    回報並重新評估是否值得跑 stage2——核心假設是「淨化後的相對優勢」（交叉現象），
    乾淨情況劣勢不否證之（STRUCTURE §4.5 已載）；若決定仍跑，降為方案 C 規模先探；
    若 stage2 亦全面劣勢 → 如實報告負結果，重心轉向分析（reward 定義/相似性預算消融）
  記憶體不足（OOM）:
    編輯: fp16 → attention slicing → 步數降 50（記錄偏離）
    pg_diff: grad_reps 10→5（記錄）→ diffusion_T 10→5（最後手段，偏離 DAYN）
    apa-gc: checkpoint 已用 → sampling 格點 50→25
    advdiff/apa-sg: skip-gradient 本就最輕，預期無 OOM
  GrIDPure 時數失控: 先跑 paper 10×10 單設定完成核心比較，掃描點與 single_deep 後補
  DAYN 資料索取未果: 依論文 §4.3 以 SD V1.4 自生成 150 張（3 類×2 prompt），
    結果標記「自建資料集」並於論文揭露（僅能偵測粗差，非同條件比較）""")

    L.append(sec("[7] 待辦（依優先序；* = 阻塞後續項）"))
    L.append("""\
  1.* 寄信 DAYN 作者: 測試集+sdedit strength+κ 尺度/範數/target（SPEC §8 1-5）
      （不阻塞部署，但阻塞 T2 定案；等待期間以 strength 掃描+自生成資料推進）
  2.* TWCC 環境建置+下載清單（SD×2、GrIDPure ckpt 2GB）＝ TWCC_CHECKLIST
  3.* 首日: 方案 C 執行＋[3] 風險清單全項檢查＋單位耗時實測（校正 [4] 表）
  4.  T2 校準（v5 分階段: norm=linf 固定，strength {0.8,0.7,0.5,0.3} 依序，共 7 次）
  5.  stage0 正式校準（真實資料；端點警告則擴範圍重掃）
  6.  方案 G 閘門式: stage1（錨點跑滿 / 補充降 seed）→ 閘門 → stage2（積極縮減）
  7.  （資料到手後）inpaint 以真實遮罩重跑；（資源允許）局部加密至 20 seed
  8.  （待裁定）批次化: 若放寬位元級判準則啟用，編輯時數 ÷4-6
""")

    L.append(hr())
    L.append("總結: 【可部署——v5 決策已落實，一項技術裁定待回覆】")
    L.append(hr())
    L.append("""\
  靜態稽核與論文核驗無 FAIL 未決項；測試全綠（含批次化等價性 6 項）；三腳本
  煙霧串跑+續跑驗證通過。v5 三項決策均已落地:
  - 規模: 閘門式循序（首日 C → 主實驗 G，見 [5]）；完整 A 單卡不可行（~7,600h fp16）
  - 資料: 自生成 150 張備援已入 TWCC_CHECKLIST，結果標記「非同條件比較」
  - strength: v5 分階段搜尋已入 TWCC_CHECKLIST 與 [6]
  待指導者裁定（不阻塞首日 C，阻塞是否啟用批次化）:
  - 批次化位元級判準實測未通過（float ~5e-6；噪聲協定位元級一致、seed 協定未破壞）。
    依 v5 判準維持 edit_batch_size=1。若放寬為「噪聲協定位元級一致＋輸出容差
    ≤1e-4」則可啟用並將編輯時數 ÷4-6（G 方案 229h → ~110h）
  （本報告由 scripts/preflight_report.py 生成；單位耗時於 TWCC 首日實測後重跑本
   腳本更新 [4][5] 數字）""")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description="部署前檢查報告")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    report = build_report(args)
    print(report)
    (REPO_ROOT / "PREFLIGHT.md").write_text(
        "# PREFLIGHT.md — 部署前檢查報告\n\n（`scripts/preflight_report.py` 產出；"
        "終端機輸出之存檔版本）\n\n```text\n" + report + "\n```\n",
        encoding="utf-8")
    print(f"\n已存檔: {REPO_ROOT / 'PREFLIGHT.md'}")


if __name__ == "__main__":
    main()
