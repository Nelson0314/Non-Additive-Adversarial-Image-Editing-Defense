"""測試用的單段顏色載體，實作 `optimize_carrier` 與 `raw_excursion` 所需介面。"""
import torch


class OffsetCarrier:
    """逐像素 RGB 偏移 `amplitude · tanh(delta)`，輸出 clamp 至 [0, 1]。"""

    def __init__(self, x01, amplitude: float = 0.3, seed: int = 0):
        self.amplitude = float(amplitude)
        self.support = torch.ones_like(x01[:, :1])
        self.reset(x01, seed)

    def reset(self, x01, seed: int) -> None:
        g = torch.Generator().manual_seed(int(seed))
        self.delta = (0.5 * torch.randn(x01.shape, generator=g, dtype=x01.dtype)
                      ).requires_grad_(True)

    @property
    def stages(self):
        return [self]

    def params(self):
        return [self.delta]

    @torch.no_grad()
    def project(self) -> None:
        self.delta.clamp_(-3.0, 3.0)

    def set_amplitude(self, values) -> None:
        self.amplitude = float(values[0] if isinstance(values, (list, tuple)) else values)

    def state_dict(self):
        return {"delta": self.delta.detach().clone()}

    def load_state_dict(self, state) -> None:
        self.delta = state["delta"].clone().requires_grad_(True)

    def raw_rgb(self, x01):
        return x01 + self.amplitude * torch.tanh(self.delta)

    def render(self, x01):
        return self.raw_rgb(x01).clamp(0, 1)
