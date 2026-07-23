"""隨機種子控制：統一設定 python / numpy / torch 之 seed，確保可重現。"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 含 CUDA（torch.manual_seed 會同時設定所有裝置）
