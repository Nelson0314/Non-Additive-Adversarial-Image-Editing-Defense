"""格點識別與設定雜湊 — `docs/CODE_2026-08-05.md` §3。

斷點續跑的判定是「產物存在**且**該格的 `config_hash` 等於當前設定的雜湊」。
只看檔名會在改了設定卻沒改路徑時靜默沿用舊結果——那正是本專案重複十次的
缺陷型態（`PRIOR_FINDINGS` §4.5）。

## 兩個刻意的設計

**必填鍵必須「存在」，但值為 None 時不進雜湊。**
兩者看似矛盾，實際上分別擋住兩種錯誤：

- 必填檢查擋的是「忘了把某個變因放進雜湊」。訓練格沒有 τ、評測格沒有 lr，
  但兩者都必須明寫 `"tau": None`，強迫呼叫端對每個欄位表態。
- None 不進雜湊擋的是「同一格因為欄位有無而算出兩個雜湊」。`{"tau": None}`
  與省略 `tau` 指的是同一件事，不該產生兩個目錄。

**`gpu` 與 `precision` 是必填。** 這是「兩張卡不可混跑」的程式化保證：
換卡會使全部格點的雜湊改變、自動視為未完成。V100 走 fp16、RTX 5090 走 bf16，
數值路徑不同，混跑會產生無法歸因的變因（`ARCH` §7.2）。

## 浮點的表示

以 `repr()` 輸出。Python 3 的 `repr(float)` 是最短往返表示，
`repr(0.1) == "0.1"` 而非 `"0.1000000000000000055511151231257827"`，
故同一個字面值在不同機器上得到同一個字串。用 `str()` 在 Python 3 下等價，
但明寫 `repr` 是為了讓「要的是往返表示」這件事在程式上看得出來。
"""

import hashlib
import json
from typing import Any, Dict

# 每一格的設定都必須表態的欄位。缺任何一個即拋出——
# 「忘了把某個變因放進雜湊」是本專案已知的缺陷型態。
REQUIRED_KEYS = (
    "spec_version",
    "model",
    "resolution",
    "guidance",
    "steps",
    "strength",
    "gpu",
    "precision",
    "condition",
    "loss_params",
    # 2026-08-05 新增。before：只有 `loss_params`，於是「同一個 condition、
    # 不同的參數化容量」會算出**相同的雜湊**——A7 原文點名的控制點 32 與 128
    # 正是這個情形，`is_done` 會把第二組判為已完成並沿用第一組的產物。
    # 最佳化的旋鈕（步數、停止準則）同理：它們改變結果卻不改變路徑。
    # after：參數化與最佳化各自一個必填鍵，缺任一即拋出。
    "module_params",
    "optim_params",
    "lr",
    "tau",
    "purify",
    "seed",
    "image_id",
)


class ConfigIncomplete(ValueError):
    """設定缺少必填鍵。訊息必須列出缺了哪些，否則呼叫端要自己猜。"""


def _canonical(obj: Any) -> Any:
    """遞迴正規化：丟掉 None、字典鍵排序、浮點轉最短往返表示。

    浮點轉成字串而非保留數值型別，是因為 `json.dumps` 對 float 的輸出格式
    受 repr 影響但不保證跨版本一致；先自己轉成字串就沒有這個依賴。
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, bool):
        # bool 是 int 的子類別，必須在 int 之前處理，否則 True 會變成 1
        return obj
    if isinstance(obj, dict):
        out = {}
        for k in sorted(obj):
            v = _canonical(obj[k])
            if v is not None:
                out[str(k)] = v
        return out
    if isinstance(obj, (list, tuple)):
        # 順序有意義（例如淨化算子的序列），故不排序
        return [_canonical(v) for v in obj]
    return obj


def canonical_json(cfg: Dict[str, Any]) -> str:
    """正規化後的 JSON 字串。相同設定必得相同字串。"""
    return json.dumps(_canonical(cfg), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def check_complete(cfg: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigIncomplete(
            f"設定缺少必填鍵 {missing}；即使該格不適用也必須明寫為 None。"
            f"目前有的鍵：{sorted(cfg)}"
        )


def config_hash(cfg: Dict[str, Any], length: int = 12) -> str:
    """sha256(canonical_json(cfg)) 的前 length 個十六進位字元。

    截短到 12 字元（48 bit）是為了目錄名可讀。本專案單批格點數在 10³ 量級，
    生日碰撞機率約 10⁶/2⁴⁹ ≈ 2e-9，可忽略。
    """
    check_complete(cfg)
    digest = hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()
    return digest[:length]


def cell_id(stage: str, condition: str, image_id: str, **parts) -> str:
    """人可讀的格點識別碼，例如 `eval/N1/pie_0007/tau0.20/jpeg30/seed3`。

    雜湊用於判定，識別碼用於閱讀與目錄命名——兩者職責分開。
    識別碼刻意不含雜湊：同一格在改設定後應該出現在同一個位置，
    由 `meta.json` 的雜湊指出它已過期，而不是散落成兩個目錄。
    """
    segs = [stage, condition, image_id]
    for k in sorted(parts):
        v = parts[k]
        if v is None:
            continue
        segs.append(f"{k}{v}" if not isinstance(v, float) else f"{k}{v:g}")
    return "/".join(str(s) for s in segs)
