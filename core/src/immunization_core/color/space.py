"""sRGB（D65）與 CIELab 之間的可微轉換；輸入輸出皆為 (B,3,H,W)。

`rgb_to_lab` 將半精度輸入升為 float32。與 skimage `rgb2lab` 的差異在 0.005 個
Lab 單位以內；求解端的色差與載體共用本實作。
"""
import torch


def rgb_to_lab(x):
    x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    linear = torch.where(x > .04045, ((x + .055) / 1.055).clamp_min(1e-10).pow(2.4), x / 12.92)
    matrix = x.new_tensor([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear) / x.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    f = torch.where(xyz > (6/29)**3, xyz.clamp_min(1e-10).pow(1/3), xyz / (3*(6/29)**2) + 4/29)
    return torch.stack((116*f[:, 1]-16, 500*(f[:, 0]-f[:, 1]), 200*(f[:, 1]-f[:, 2])), 1)


def lab_to_rgb(lab):
    l, a, b = lab.unbind(1)
    fy = (l+16)/116
    f = torch.stack((fy+a/500, fy, fy-b/200), 1)
    xyz = torch.where(f > 6/29, f.pow(3), 3*(6/29)**2*(f-4/29))
    xyz = xyz * lab.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    matrix = lab.new_tensor([[3.2404542, -1.5371385, -.4985314], [-.9692660, 1.8760108, .0415560], [.0556434, -.2040259, 1.0572252]])
    linear = torch.einsum('ij,bjhw->bihw', matrix, xyz)
    return torch.where(linear > .0031308, 1.055*linear.clamp_min(1e-10).pow(1/2.4)-.055, 12.92*linear)
