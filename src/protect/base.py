"""保護方法統一介面（STRUCTURE.md §2.1、SPEC.md §1.3）。

所有方法實作此介面，使 stage 腳本可統一呼叫、確保公平比較。
加性與非加性僅在 protect() 內部不同，對外一致。
"""

from abc import ABC, abstractmethod

import torch


class ProtectionMethod(ABC):
    """所有保護方法的抽象介面。"""

    def __init__(self, sd_wrapper, config: dict):
        self.sd = sd_wrapper
        self.cfg = config

    @abstractmethod
    def protect(self, image: torch.Tensor, concept: str) -> torch.Tensor:
        """產生受保護影像。

        Args:
            image:   (1,3,H,W)，值域 [0,1]
            concept: 欲保護之語意概念，如 "dog"。
                     PhotoGuard 不使用此參數。

        Returns:
            (1,3,H,W)，值域 [0,1]
        """

    @property
    @abstractmethod
    def is_additive(self) -> bool: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def peak_memory_mb(self) -> float:
        """回傳上次 protect() 的峰值記憶體，供資源比較用。"""
        raise NotImplementedError
