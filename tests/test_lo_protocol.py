"""Lo et al. (CVPR 2024) 協定的實作正確性。

這批函式現在是本專案的**文獻基準**，我們自己的方法要跟它比。基準算錯的話，
比較的結論全部無效，而且錯的方向不可預期——故式 (3)(4)(5) 與 Algorithm 1
各自都要有釘住的測試。

不需要 SD 權重：注意力聚合吃的是張量、PGD 迴圈吃的是任意可微損失。
"""

import json
import math

import pytest
import torch

from src.defense.linf_attack import LinfAttackConfig, build_attack, pgd_linf
from src.models.attention import (
    aggregate_token_attention,
    attention_region_mask,
    masked_attention_l1,
)


# ---------------------------------------------------------------------------
# 式 (3)：跨層上採樣後相加
# ---------------------------------------------------------------------------


def _maps(sides, tokens=8, batch=1):
    """造一組假的注意力圖，每層一個 (B, h·h, T)。"""
    return [torch.rand(batch, s * s, tokens) for s in sides]


def test_聚合後的形狀取最大邊長():
    out = aggregate_token_attention(_maps([8, 4, 2]), span=(1, 3))
    assert out.shape == (1, 8, 8)


def test_可指定輸出邊長():
    out = aggregate_token_attention(_maps([8, 4]), span=(1, 3), side=16)
    assert out.shape == (1, 16, 16)


def test_是相加不是平均():
    # 三層各自為常數 c，上採樣後仍是 c，相加應得 3c 而非 c。
    # 這一項分得出式 (3) 的 Σ 與「取平均」——後者會讓值域變成 [0,1]，
    # 遮罩門檻的意義隨之改變。
    maps = [torch.full((1, s * s, 4), 0.25) for s in (4, 4, 4)]
    out = aggregate_token_attention(maps, span=(0, 4))
    assert out.mean().item() == pytest.approx(3 * 0.25 * 4)


def test_只取內容token的質量():
    m = torch.zeros(1, 4, 6)
    m[..., 2] = 1.0          # 只有第 2 個 token 有質量
    single = aggregate_token_attention([m], span=(2, 3))
    outside = aggregate_token_attention([m], span=(0, 2))
    assert single.mean().item() == pytest.approx(1.0)
    assert outside.mean().item() == pytest.approx(0.0)


def test_非完全平方的query數必須報錯():
    # 靜默猜長寬比會讓注意力圖被轉置，遮罩落在錯的區域而且毫無症狀。
    with pytest.raises(ValueError, match="完全平方數"):
        aggregate_token_attention([torch.rand(1, 12, 4)], span=(0, 2))


def test_空span必須報錯():
    with pytest.raises(ValueError, match="內容 token"):
        aggregate_token_attention(_maps([4]), span=(2, 2))


def test_空的注意力清單必須報錯():
    with pytest.raises(ValueError, match="為空"):
        aggregate_token_attention([], span=(0, 1))


# ---------------------------------------------------------------------------
# 式 (4)：二值遮罩
# ---------------------------------------------------------------------------


def test_遮罩以峰值正規化後比較():
    att = torch.tensor([[[0.0, 1.0], [4.0, 8.0]]])   # 峰值 8
    # τ=0.5 → 只有 > 4 的通過，即 8 那一格
    m = attention_region_mask(att, tau=0.5)
    assert m.tolist() == [[[0.0, 0.0], [0.0, 1.0]]]


def test_遮罩門檻放寬會納入更多格點():
    att = torch.tensor([[[0.0, 1.0], [4.0, 8.0]]])
    assert attention_region_mask(att, tau=0.1).sum() == 3
    assert attention_region_mask(att, tau=0.5).sum() == 1


def test_遮罩不帶梯度():
    att = torch.rand(1, 4, 4, requires_grad=True)
    assert not attention_region_mask(att, 0.5).requires_grad


def test_遮罩永遠至少含有峰值():
    # 以峰值正規化後，峰值那格恆為 1 而 τ < 1，故遮罩不可能為空。這是本
    # 實作的結構性保證，也是「L1 恆為 0、梯度恆為零」這個失效不會發生的
    # 理由。改掉正規化方式就會破壞它。
    att = torch.zeros(1, 4, 4)
    att[0, 2, 3] = 7.0
    m = attention_region_mask(att, tau=0.999)
    assert m.sum() == 1
    assert m[0, 2, 3] == 1.0


def test_門檻超出開區間必須報錯():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="開區間"):
            attention_region_mask(torch.rand(1, 4, 4), tau=bad)


def test_全零的參照圖必須報錯():
    with pytest.raises(ValueError, match="沒有分到任何注意力質量"):
        attention_region_mask(torch.zeros(1, 4, 4), tau=0.5)


# ---------------------------------------------------------------------------
# 式 (5)：遮罩內的 L1
# ---------------------------------------------------------------------------


def test_只計遮罩內的反應():
    att = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert masked_attention_l1(att, mask).item() == pytest.approx(5.0)


def test_取和不取平均():
    att = torch.ones(1, 4, 4)
    mask = torch.ones(1, 4, 4)
    assert masked_attention_l1(att, mask).item() == pytest.approx(16.0)


def test_遮罩外的像素仍收得到梯度():
    # 本實作先算注意力再遮罩（見函式 docstring 的偏離說明）。若改成「先遮罩
    # 影像再算注意力」，遮罩外的輸入梯度會恆為零，這條測試會失敗。
    att = torch.rand(1, 4, 4, requires_grad=True)
    mask = torch.zeros(1, 4, 4)
    mask[..., 0, 0] = 1.0
    masked_attention_l1(att, mask).backward()
    assert att.grad[0, 0, 0].item() != 0.0


def test_空間尺寸不符必須報錯():
    with pytest.raises(ValueError, match="空間"):
        masked_attention_l1(torch.rand(1, 4, 4), torch.rand(1, 8, 8))


# ---------------------------------------------------------------------------
# Algorithm 1：L∞ 球上的 PGD
# ---------------------------------------------------------------------------


def _quadratic_terms(target):
    """一個最小值在 `target` 的可微損失，用來檢查迴圈往正確方向走。"""

    def terms(x):
        yield ((x - target) ** 2).mean()

    return terms


def test_擾動永遠不超過預算():
    x = torch.full((1, 3, 8, 8), 0.5)
    tgt = torch.zeros_like(x)
    cfg = LinfAttackConfig(kappa=0.02, steps=30, log_every=1000)
    r = pgd_linf(x, _quadratic_terms(tgt), cfg)
    assert float((r.x_adv - x).abs().max()) <= cfg.kappa + 1e-6
    assert all(h["eff_linf"] <= cfg.kappa + 1e-6 for h in r.history)


def test_往損失下降的方向走():
    x = torch.full((1, 3, 8, 8), 0.5)
    tgt = torch.zeros_like(x)
    cfg = LinfAttackConfig(kappa=0.05, steps=20, log_every=1000)
    r = pgd_linf(x, _quadratic_terms(tgt), cfg)
    assert r.history[-1]["loss"] < r.history[0]["loss"]
    # 目標在 0，故 x_adv 應比原圖更接近 0
    assert float(r.x_adv.mean()) < float(x.mean())


def test_輸出留在有效值域內():
    # 原圖貼著邊界時，clamp 必須生效，否則存成 PNG 的影像與最佳化的對象
    # 不是同一張。
    x = torch.full((1, 3, 8, 8), 0.01)
    tgt = torch.full_like(x, -5.0)      # 把 x 往負的方向拉
    cfg = LinfAttackConfig(kappa=0.5, steps=10, log_every=1000)
    r = pgd_linf(x, _quadratic_terms(tgt), cfg)
    assert float(r.x_adv.min()) >= 0.0
    assert float(r.x_adv.max()) <= 1.0


def test_每輪由原圖重新投影而非累積偏移():
    # 論文 Algorithm 1 第 13 行字面上是 x_adv <- x_adv - δ，會讓偏移累積到
    # Σδ_i 而不受 κ 限制。此處釘住本實作採用的標準 PGD 形式：跑很多步之後
    # 偏移量仍等於 δ 本身，而不是步數乘以 δ。
    x = torch.full((1, 3, 4, 4), 0.5)
    tgt = torch.zeros_like(x)
    cfg = LinfAttackConfig(kappa=0.03, steps=50, step_size=0.01, log_every=1000)
    r = pgd_linf(x, _quadratic_terms(tgt), cfg)
    assert float((r.x_adv - x).abs().max()) == pytest.approx(0.03, abs=1e-6)


def test_多項損失取平均梯度():
    # Algorithm 1 第 11 行是 mean，不是 sum。以兩個方向相反、大小不同的項
    # 檢查：若取 sum，步長會被項數放大而軌跡不同。
    x = torch.zeros(1, 1, 2, 2)

    def terms(v):
        yield (v ** 2).mean()
        yield (v ** 2).mean()

    cfg = LinfAttackConfig(kappa=0.1, steps=3, step_size=0.01, log_every=1000)
    r_two = pgd_linf(x, terms, cfg)

    def one(v):
        yield (v ** 2).mean()

    r_one = pgd_linf(x, one, cfg)
    assert torch.allclose(r_two.x_adv, r_one.x_adv)


def test_多項損失各自擁有計算圖():
    # 迴歸測試。`pgd_linf` 對每一項各呼叫一次 retain_graph=False 的 grad，
    # 若各項共用上游子圖，第一項反傳就會把它釋放，第二項拋出
    # 「Trying to backward through the graph a second time」。
    # semantic attack 原本把 VAE 編碼提到 timestep 迴圈外，正是這個形狀，
    # 本機實測在真實 SD 上炸掉（2026-08-03）。
    x = torch.full((1, 1, 4, 4), 0.5)

    def shared_prefix_terms(v):
        for k in (1.0, 2.0):
            # 前置計算在迴圈內重算 —— 這是本模組要求的正確寫法
            pre = v * 3.0
            yield (pre * k).sum()

    cfg = LinfAttackConfig(kappa=0.05, steps=2, step_size=0.01, log_every=1000)
    r = pgd_linf(x, shared_prefix_terms, cfg)
    assert len(r.history) == 2
    assert r.history[0]["grad_norm"] > 0


def test_沒有任何損失項必須報錯():
    def empty(x):
        return iter(())

    with pytest.raises(RuntimeError, match="沒有產生任何項"):
        pgd_linf(torch.zeros(1, 1, 4, 4), empty,
                 LinfAttackConfig(steps=1, log_every=1000))


def test_預算非正必須報錯():
    with pytest.raises(ValueError, match="必須為正"):
        pgd_linf(torch.zeros(1, 1, 4, 4), _quadratic_terms(torch.zeros(1)),
                 LinfAttackConfig(kappa=0.0))


def test_論文的預設值():
    # κ 與 N 是論文寫死的比較條件，改掉它們就不再是同一個協定。
    cfg = LinfAttackConfig()
    assert cfg.kappa == 0.06
    assert cfg.steps == 100
    assert cfg.timesteps == 10


def test_論文未公布的超參數取本專案選定的值():
    # 這四個論文與補充材料都沒寫，由本專案指定（2026-08-03）。釘住它們是
    # 因為它們是「我們的選擇」而非「論文的條件」：日後若重現不成功，這裡
    # 就是要回頭檢查的四個旋鈕，值被無聲改掉會讓那次檢查失去基準。
    cfg = LinfAttackConfig()
    assert cfg.strength == 0.3          # 對齊 PhotoGuard 的 img2img 評測
    assert cfg.guidance_scale == 7.5    # E26：w=1 下 SD v1.4 幾乎不服從 prompt
    assert cfg.mask_tau == 0.5          # 作用在峰值正規化後的 [0,1] 尺度上
    assert cfg.step_size is None        # None → κ/10，見 pgd_linf


def test_步長預設為預算的十分之一():
    # 起點取 0.5 而非 0：貼著值域下界時位移會被 clamp 掉，量到的是 0 而
    # 不是步長。這是寫測試時實際踩到的。
    x = torch.full((1, 1, 4, 4), 0.5)
    cfg = LinfAttackConfig(kappa=0.06, steps=1, log_every=1000)
    r = pgd_linf(x, _quadratic_terms(torch.zeros_like(x)), cfg)
    assert float((r.x_adv - x).abs().max()) == pytest.approx(0.006, abs=1e-6)


def test_未知的攻擊名稱必須報錯():
    with pytest.raises(ValueError, match="未知的攻擊"):
        build_attack("nope", None, LinfAttackConfig(), torch.zeros(1))


def test_有目標的攻擊缺目標圖必須報錯():
    with pytest.raises(ValueError, match="x_target"):
        build_attack("pg_encoder", None, LinfAttackConfig(), torch.zeros(1))


# ---------------------------------------------------------------------------
# 接續執行：L1 要跑約三小時，中途斷掉不能把已算好的結果一起丟掉
# ---------------------------------------------------------------------------


def test_已完成的格由summary判定(tmp_path):
    from scripts.run_lo_baseline import append_csv, completed_pairs

    p = tmp_path / "summary.csv"
    assert completed_pairs(p) == set()          # 檔案不存在時為空集合
    append_csv(p, [{"image": "dog_00", "attack": "semantic", "x": 1}])
    append_csv(p, [{"image": "dog_00", "attack": "pg_encoder", "x": 2}])
    assert completed_pairs(p) == {("dog_00", "semantic"), ("dog_00", "pg_encoder")}


def test_附加不重寫表頭(tmp_path):
    from scripts.run_lo_baseline import append_csv

    p = tmp_path / "r.csv"
    append_csv(p, [{"a": 1, "b": 2}])
    append_csv(p, [{"a": 3, "b": 4}, {"a": 5, "b": 6}])
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "a,b"
    assert len(lines) == 4                      # 表頭 + 三列


def test_表頭不符必須拒絕附加(tmp_path):
    # 欄位集合變了代表程式改過。混在同一個檔裡會讓後續判讀無聲錯位，
    # 那比中止難查得多。
    from scripts.run_lo_baseline import append_csv

    p = tmp_path / "r.csv"
    append_csv(p, [{"a": 1, "b": 2}])
    with pytest.raises(RuntimeError, match="表頭"):
        append_csv(p, [{"a": 1, "c": 3}])


def test_空列拒絕寫出(tmp_path):
    from scripts.run_lo_baseline import append_csv

    with pytest.raises(RuntimeError, match="沒有任何列"):
        append_csv(tmp_path / "r.csv", [])


# ---------------------------------------------------------------------------
# 協定守衛：--resume 只看 (影像, 攻擊)，不看參數，換了參數會無聲混雜
# ---------------------------------------------------------------------------


class _Args:
    """`check_protocol` 只讀 PROTOCOL_KEYS，其餘欄位不影響。"""

    def __init__(self, **kw):
        from scripts.run_lo_baseline import PROTOCOL_KEYS

        for k in PROTOCOL_KEYS:
            setattr(self, k, kw.pop(k, 0))
        assert not kw, f"不是協定參數：{list(kw)}"


def test_首次執行寫下協定(tmp_path):
    from scripts.run_lo_baseline import check_protocol

    check_protocol(tmp_path, _Args(prompt_index=0))
    got = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    assert got["prompt_index"] == 0


def test_同一組協定可以接續(tmp_path):
    from scripts.run_lo_baseline import check_protocol

    check_protocol(tmp_path, _Args(prompt_index=1, kappa=0.06))
    check_protocol(tmp_path, _Args(prompt_index=1, kappa=0.06))   # 不得拋出


def test_換了編輯prompt不得接續同一目錄(tmp_path):
    # 這正是 2026-08-03 發現的問題：論文補充材料 §A 每個物件有兩個編輯
    # prompt，Table 1 是兩者一起平均。兩半混進同一個目錄後 summary.csv
    # 不記 prompt，事後看不出來。
    from scripts.run_lo_baseline import check_protocol

    check_protocol(tmp_path, _Args(prompt_index=0))
    with pytest.raises(SystemExit, match="prompt_index"):
        check_protocol(tmp_path, _Args(prompt_index=1))


def test_換了預算不得接續同一目錄(tmp_path):
    from scripts.run_lo_baseline import check_protocol

    check_protocol(tmp_path, _Args(kappa=0.06))
    with pytest.raises(SystemExit, match="kappa"):
        check_protocol(tmp_path, _Args(kappa=0.03))


def test_補跑攻擊不算換協定(tmp_path):
    # `--attacks` 刻意不在 PROTOCOL_KEYS 裡：分批補跑攻擊是正當的接續。
    from scripts.run_lo_baseline import PROTOCOL_KEYS

    assert "attacks" not in PROTOCOL_KEYS
    assert "limit" not in PROTOCOL_KEYS


# ---------------------------------------------------------------------------
# 逐編輯 prompt 的分解
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content,prompt,expect", [
    ("man", "a woman", False),      # 子字串比對會判成 True，方向剛好相反
    ("man", "a man in a snowy street", True),
    ("dog", "a cat", False),
    ("dog", "a dog in the park", True),
    ("cat", "A CAT on a snowy roof", True),   # 不分大小寫
    ("horse", "a horse and a cow", True),
    ("bird", "a butterfly", False),
])
def test_c_a是否以完整詞出現在prompt裡(content, prompt, expect):
    from scripts.report_table1 import c_a_in_prompt

    assert c_a_in_prompt(content, prompt) is expect
