"""E31 Task 1：劣化階梯的規格測試。

階梯本身要跑指標與寫比對頁，需要影像；此處只測純函數 `ladder_arms`——
它決定「階梯跨越的範圍」，而範圍不夠寬時人眼永遠看不到「不能用」那一級，
門檻就定不出來。這一段不該只能靠跑完整階梯才看得出對錯。
"""

from scripts.p11_degrade_ladder import ladder_arms


def test_階梯涵蓋四個算子各四級():
    arms = ladder_arms()
    assert len(arms) == 16
    kinds = [k for k, _ in arms]
    for k in ("blur", "jpeg", "noise", "quantize"):
        assert kinds.count(k) == 4


def test_每個算子的強度是單調變差的():
    # blur 與 noise 的強度值越大越糟，jpeg 品質與量化階數越小越糟。
    # 兩種方向都要能在比對頁上由左至右呈現「越來越差」。
    arms = ladder_arms()
    for kind, worse_is_larger in (("blur", True), ("jpeg", False),
                                  ("noise", True), ("quantize", False)):
        s = [v for k, v in arms if k == kind]
        assert s == sorted(s, reverse=not worse_is_larger), (kind, s)
