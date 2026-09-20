"""NCF §3.3 Eqs. 4–6,9, reconstructed from the full paper (code unavailable).

modified_from_paper: L_IG, ATR hard protection, explicit covariance eigenvalue
floor in physical CIELab units. One global MK matrix; epsilon=0 retains T0.
"""
from copy import deepcopy

import numpy as np
import torch
from PIL import Image


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


def regularize_covariance(cov, floor):
    if floor <= 0 or not torch.isfinite(cov).all() or cov.shape != (3, 3):
        raise ValueError('covariance must be finite 3x3; eigenvalue floor must be positive')
    if not torch.allclose(cov, cov.T, atol=1e-7, rtol=1e-6):
        raise ValueError('non-symmetric covariance')
    values, vectors = torch.linalg.eigh(cov.double())
    if values.min() < -1e-7:
        raise ValueError(f'indefinite covariance: {values.tolist()}')
    fixed = (vectors * values.clamp_min(floor)) @ vectors.T
    return fixed, {'cov_floor': floor, 'cov_min_eigenvalue': float(values.min()),
                   'cov_regularized': int((values < floor).any()),
                   'cov_regularization_modified_from_paper': True}


def _sqrt(c):
    v, q = torch.linalg.eigh(c)
    return (q * v.clamp_min(0).sqrt()) @ q.T


def mk_matrix(source, target):
    """Eq. 6, SPD inputs: T Sigma_source T^T = Sigma_target."""
    s = _sqrt(source.double())
    inv = torch.linalg.inv(s)
    return inv @ _sqrt(s @ target.double() @ s) @ inv


class NCFColorParam:
    """One global Lab transfer matrix applied through a support mask.

    `support` is (1,1,H,W) in [0,1]: 1 = this pixel may be recoloured, 0 = it is
    copied from the original bit for bit. The two accepted regions are expressed
    the same way — a whole-image colour filter is `support = 1` everywhere, a
    clothing carrier is the validated `carrier_mask(x, "clothes")` path.

    `epsilon_lab` bounds the perturbation's **effect** rather than its raw
    entries. The raw box alone is unsafe because `T0`'s own scale varies by an
    order of magnitude with the target: a palette with little L variance yields
    `T0[0,0] = 0.065`, and the paper's `epsilon = 0.2` is then wide enough to
    flip the sign of the luminance mapping, which is what destroyed the first
    run. Bounding `sqrt(sum_j (dT_ij * sigma_j)^2)` in Lab units makes one unit
    of budget mean the same displacement whichever channel it comes through.

    `whiten` changes what is optimised, not what is representable. The learnable
    tensor becomes `U` with `dT[i,j] = U[i,j] / sigma_j`, so a step of fixed size
    in `U` moves every output channel by the same amount in Lab units, and
    `epsilon_lab` becomes a plain ball `||U_i|| <= epsilon_lab` instead of a
    rescaling after the fact. This matters because the raw `dT` columns differ by
    an order of magnitude in scale: sign-PGD gives all nine entries the same step,
    which is the right direction for an additive L-infinity ball and the wrong one
    here. The perturbation this project applies is not additive, so the optimizer
    inherited from that family is not automatically the right one.

    `radius` and `epsilon_lab` bound the learnable `delta` **and nothing else**.
    The colour displacement of this carrier comes almost entirely from `T0` (the
    palette) and from the amplitude, neither of which either number touches. On
    the whole-frame filter at amplitude 1, `delta = 0` already reaches dE00 8.2659
    and 12.7213 on two registered images, and driving `delta` to the projected box
    corner adds 0.0136 and 0.6133 — 0.2% and 4.6% of the displacement. The values
    0.2 and 5 are inherited from NCF, whose objective was classifier
    misclassification; they are not a bound on how strong this defence is.
    """

    name = 'ncf'

    def __init__(self, target_mean, target_cov, *, support, radius=.2,
                 cov_floor=1e-4, epsilon_lab=None, whiten=False):
        self.target_mean = torch.as_tensor(target_mean, dtype=torch.float64)
        self.target_cov = torch.as_tensor(target_cov, dtype=torch.float64)
        self.support = support
        self.cov_floor = cov_floor
        self.epsilon_lab = None if epsilon_lab is None else float(epsilon_lab)
        if self.epsilon_lab is not None and not (self.epsilon_lab > 0):
            raise ValueError('epsilon_lab must be positive when given')
        self.whiten = bool(whiten)
        if self.whiten and self.epsilon_lab is None:
            raise ValueError('whiten needs epsilon_lab: without it the ball has no radius '
                             'and the raw box no longer bounds anything meaningful')
        self.set_radius(radius)

    def reset(self, x01, seed=0):
        if x01.ndim != 4 or x01.shape[:2] != (1, 3):
            raise ValueError('NCF requires one RGB image')
        if self.support.shape != (1, 1, *x01.shape[-2:]):
            raise ValueError('support must be a 1x1xHxW mask')
        w = self.support.to(device=x01.device, dtype=torch.float64)[0, 0]
        if not ((w >= 0) & (w <= 1)).all():
            raise ValueError('support must lie in [0,1]')
        lab = rgb_to_lab(x01).double()
        # Source statistics come from the support only, weighted by it. Pixels
        # that are copied from the original cannot be moved by T, so letting
        # them define the distribution T is fitted to calibrates the map on
        # colour it can never touch.
        flat, wf = lab[0].flatten(1), w.flatten()
        total = wf.sum()
        if total < 2:
            raise ValueError('support holds less than two pixels of weight; nothing to recolour')
        self.source_mean = (flat * wf).sum(1) / total
        centered = flat - self.source_mean[:, None]
        # 原始的來源共變異數留著：目標可學的子類每一步都要用它重算 T0，
        # 而正則化後的版本已經套過特徵值地板，不是同一個東西。
        self._source_cov = (centered * wf) @ centered.T / total
        source, self.source_audit = regularize_covariance(self._source_cov, self.cov_floor)
        target, self.target_audit = regularize_covariance(self.target_cov.to(source), self.cov_floor)
        self.source_sigma = source.diagonal().clamp_min(self.cov_floor).sqrt()
        self.T0 = mk_matrix(source, target).to(lab.device)
        # reset clears the increment, never replaces the explicitly selected
        # target distribution.
        zero = torch.zeros_like(self.T0)
        if self.whiten:
            self.u = zero.clone().requires_grad_(True)
            self.delta = None
        else:
            self.delta = zero.clone().requires_grad_(True)
            self.u = None

    @property
    def stages(self):
        """`immunise.fit_caps` 逐段取幅度用的；這個載體只有一段。"""
        return [self]

    def set_amplitude(self, a):
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        self.amplitude = float(a)

    def params(self):
        return [self.u if self.whiten else self.delta]

    def increment(self):
        """`dT`, differentiable in whichever coordinate is being optimised."""
        if not self.whiten:
            return self.delta
        return self.u / self.source_sigma.to(self.u)[None, :]

    def raw_rgb(self, x):
        lab = rgb_to_lab(x).double()
        out = torch.einsum('ij,bjhw->bihw', self.T0+self.increment(),
                           lab-self.source_mean[None, :, None, None])
        return lab_to_rgb(out+self.target_mean.to(out)[None, :, None, None]).to(x.dtype)

    def render(self, x):
        w = self.support.to(device=x.device, dtype=x.dtype)
        out = w * self.raw_rgb(x).clamp(0, 1) + (1 - w) * x
        # Outside the support the result must be the original bit for bit, not
        # `1-w` times it: a float blend at w = 0 still round-trips through Lab.
        return torch.where(w > 0, out, x)

    @torch.no_grad()
    def project(self):
        sig = self.source_sigma
        if self.whiten:
            # In whitened coordinates the effect bound is exactly a row-wise
            # ball, so this is one projection instead of a box followed by a
            # rescale. The raw box is applied afterwards, in dT coordinates,
            # so the paper's constraint still holds.
            norm = self.u.norm(dim=1)
            self.u.mul_((self.epsilon_lab / norm.clamp_min(1e-12)).clamp(max=1.0)[:, None])
            over = (self.u / sig.to(self.u)[None, :]).abs().amax()
            if float(over) > self.radius:
                self.u.mul_(self.radius / over)
            return
        self.delta.clamp_(-self.radius, self.radius)
        if self.epsilon_lab is None:
            return
        # Row-wise: bound the standard deviation this perturbation adds to each
        # output channel, measured in Lab units through the source spread.
        effect = (self.delta * sig.to(self.delta)[None, :]).norm(dim=1)
        scale = (self.epsilon_lab / effect.clamp_min(1e-12)).clamp(max=1.0)
        self.delta.mul_(scale[:, None])

    def set_radius(self, r):
        if not np.isfinite(r) or r < 0:
            raise ValueError('radius must be finite and nonnegative')
        self.radius = float(r)

    def state_dict(self):
        return deepcopy({k: (v.detach().clone() if torch.is_tensor(v) else v)
                         for k, v in vars(self).items()})

    def load_state_dict(self, state):
        for k, v in deepcopy(state).items():
            setattr(self, k, v)
        name = 'u' if self.whiten else 'delta'
        setattr(self, name, getattr(self, name).detach().requires_grad_(True))

    @torch.no_grad()
    def diagnostics(self, x):
        from skimage.color import deltaE_ciede2000
        raw = self.raw_rgb(x)
        a = rgb_to_lab(x)[0].permute(1, 2, 0).cpu().numpy()
        b = rgb_to_lab(self.render(x))[0].permute(1, 2, 0).cpu().numpy()
        d = self.increment().detach()
        sig = self.source_sigma.to(d)[None, :]
        # Effect of the natural mapping vs. the learnt one, per output channel.
        # `luma_response` collapsing toward zero is the signature of the
        # degenerate solution: output lightness stops depending on the input.
        base = (self.T0 * sig).norm(dim=1)
        learnt = ((self.T0 + d) * sig).norm(dim=1)
        return {'T_distance': float(d.norm()), 'whiten': int(self.whiten),
                'box_boundary_fraction': float((d.abs() >= self.radius).double().mean()),
                'clipping_fraction': float(((raw < 0) | (raw > 1)).float().mean()),
                'deltaE00': float(deltaE_ciede2000(a, b).mean()),
                'support_area': float(self.support.to(torch.float64).mean()),
                'delta_effect_L': float((d * sig).norm(dim=1)[0]),
                'luma_response': float(learnt[0] / base[0].clamp_min(1e-12)),
                'luma_self_coefficient': float((self.T0 + d)[0, 0]),
                **{'source_'+k: v for k, v in self.source_audit.items()},
                **{'target_'+k: v for k, v in self.target_audit.items()}}


def verify_saved_protection(path, original, support):
    """Outside the support the delivered PNG must equal the original uint8.

    Checked on the file rather than the tensor: Lab round-trip, clamping and a
    soft support edge are all places a pixel can drift by one level without any
    intermediate assertion noticing.
    """
    actual = np.asarray(Image.open(path).convert('RGB'))
    expected = (original.detach().cpu()[0].permute(1, 2, 0)*255).round().clamp(0, 255).to(torch.uint8).numpy()
    outside = (support.cpu()[0, 0].numpy() <= 0)
    if actual.shape != expected.shape or not np.array_equal(actual[outside], expected[outside]):
        raise ValueError('saved pixels outside the support differ from original uint8')


def save_protected_png(path, image, original, support):
    keep = support.to(device=original.device, dtype=original.dtype) > 0
    image = torch.where(keep, image, original)
    arr = (image.detach().cpu()[0].permute(1, 2, 0)*255).round().clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(arr).save(path)
    verify_saved_protection(path, original, support)
