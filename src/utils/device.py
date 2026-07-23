"""裝置抽象 — STRUCTURE.md §2.6、SPEC.md §1.4。

全專案禁用 .cuda()，一律經此模組取得 device。GPU 廠牌不確定，
可能非 NVIDIA（ROCm 版 torch 亦經 torch.cuda 介面，此處天然涵蓋）。
不得依賴 xformers。

執行 `python -m src.utils.device` 會印出裝置資訊並附加至 NOTES.md。
"""

import json
import os
import platform
from datetime import datetime
from pathlib import Path

import torch

_NOTES_PATH = Path(__file__).resolve().parents[2] / "NOTES.md"


def get_device() -> torch.device:
    """偵測可用裝置。本地回傳 cpu、TWCC 回傳 cuda，同一份程式碼不需修改。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _system_memory_gb() -> float | None:
    """回傳系統 RAM（GB）。Linux 經 os.sysconf、Windows 經 ctypes，不依賴額外套件。"""
    try:
        if hasattr(os, "sysconf") and os.sysconf_names.get("SC_PHYS_PAGES"):
            return round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2
            )
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / 1024**3, 2)
    except Exception:
        return None


def get_device_info() -> dict:
    """回傳廠牌、型號、記憶體、後端（cuda/rocm/cpu）與版本資訊。"""
    info = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        is_rocm = getattr(torch.version, "hip", None) is not None
        info.update(
            {
                "backend": "rocm" if is_rocm else "cuda",
                "vendor": "AMD" if is_rocm else "NVIDIA",
                "model": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "device_count": torch.cuda.device_count(),
                "compute_capability": f"{props.major}.{props.minor}",
                "cuda_version": torch.version.cuda,
            }
        )
    else:
        info.update(
            {
                "backend": "cpu",
                "vendor": platform.machine(),
                "model": platform.processor() or "unknown",
                "total_memory_gb": _system_memory_gb(),
            }
        )
    return info


def append_info_to_notes(info: dict, notes_path: Path = _NOTES_PATH) -> None:
    """將裝置資訊以帶時間戳的區塊附加至 NOTES.md。"""
    block = (
        f"\n### 裝置資訊 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        f"（get_device_info()）\n\n```json\n"
        f"{json.dumps(info, ensure_ascii=False, indent=2)}\n```\n"
    )
    with open(notes_path, "a", encoding="utf-8") as f:
        f.write(block)


if __name__ == "__main__":
    device_info = get_device_info()
    print(f"device: {get_device()}")
    print(json.dumps(device_info, ensure_ascii=False, indent=2))
    append_info_to_notes(device_info)
    print(f"已附加至 {_NOTES_PATH}")
