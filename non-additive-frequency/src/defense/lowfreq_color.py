"""不往高頻加東西的兩個色彩載體，以及驗證這一點的量測。

1. **分區的邊界。** 兩塊區域給不同色調時，邊界本身就是一條高頻的邊。
   既有的 `feather_inward(8)` 是為了硬遮罩設計的，尺度太細。
2. **放大局部對比。** 色彩映射矩陣的奇異值大於 1 時，輸入的色度梯度被整體
   放大，等於間接注入高頻。
3. **讓亮度參與色度的映射。** 亮度通道承載幾乎全部的紋理；用 L 去驅動 a/b，
   等於把亮度的高頻複製到色度上。

這一支的兩個參數化各自從構造上擋掉這三件事，並提供 `highfreq_report` 把
「有沒有加高頻」量出來，而不是宣稱。

兩個參數化
────────────────────────────────────────────────────────────────────
`ChromaAffineParam`   映射只在 (a,b) 兩維，L 不參與，且矩陣奇異值有上界。
                      色域處理走 `soft_gamut`，它在 RGB 域動作，導數恆 ≤ 1，
                      所以**任何梯度（含亮度）都只會被衰減、不會被放大**。
                      代價是 L 不再逐像素恆等——只有 `gamut='clip'` 或 `'scale'`
                      才保有那個更強的性質，但兩者各有自己的問題（見 soft_gamut）。
`RegionPaletteParam`  多塊區域各自一組目標配色，權重圖重度低通後混合。
                      空間變化的頻寬因此由高斯的 sigma 決定，不由遮罩的邊決定。

兩者都沿用 `NCFColorParam` 的來源統計、共變異數地板與 Monge–Kantorovitch
轉移（`mk_matrix`），只換掉「映射作用在哪些維度、以及空間上怎麼混合」。
"""
import math

import torch

from src.purify.ops import gaussian_blur

from .ncf_param import (NCFColorParam, lab_to_rgb, mk_matrix,  # noqa: F401
                        regularize_covariance, rgb_to_lab)


def _to_like(value, like):
    """把常數轉成與 `like` 同裝置、float64 的張量。

    這個小函式是為了一個實際發生過的錯誤而存在：`RegionPaletteParam.reset`
    直接用 `torch.as_tensor(cov)` 建出 CPU 張量，再與 GPU 上的來源統計一起送進
    `mk_matrix`，於是在遠端跑的時候整批崩在
    `Expected all tensors to be on the same device`。母類 `NCFColorParam` 有
    `.to(source)`，這裡漏了。全 CPU 的測試看不出這條，所以把裝置處理集中到一個
    可以單獨測的地方，並用 meta 裝置驗證它真的有搬。
    """
    return torch.as_tensor(value, dtype=torch.float64).to(like.device)


def soft_gamut(rgb, knee=.06):
    """平滑的軟裁，取代 `clamp(0,1)`。導數恆落在 [0,1]，所以不可能放大梯度。

    這個式子

        f(v) = v - k*softplus((v-1)/k) + k*softplus(-v/k)

    處處可微，且

        f'(v) = 1 - sigmoid((v-1)/k) - sigmoid(-v/k) 落在 [0, 1]，

    所以它只會衰減梯度、不會放大——高通能量因此只可能持平或下降，而且這是
    逐點的保證，不依賴任何影像統計。值域收斂在 (0,1) 內，最後的 clamp 只是
    數值保險，不再是造邊的那一步。`knee` 是轉折的寬度：越小越接近硬裁。

    **它在色域內就不是恆等。** `knee = 0.06` 下 `f(0) = 0.041589`、
    `f(1) = 0.958411`，`[0,1]` 上 `|f(v) - v|` 的平均是 0.005919，最大 0.041589
    出現在兩端；中段 `[0.2, 0.8]` 的最大偏離只有 0.002103。也就是說每張防禦圖
    都額外付一份集中在黑白兩端、與顏色無關的色調壓縮，而高通殘差比有一部分
    是這個壓縮買來的。

    在影像上量得到：整圖濾鏡把 `amplitude` 設成 0（色度逐位元不動）之後，防禦圖
    相對原圖的 dE00 仍有 0.0776 與 0.5149（兩張登記影像）。恆等設定不是恆等
    輸出。改 `knee` 會讓既有批次不可並列，要改就整批重跑。
    """
    if not (knee > 0):
        raise ValueError('knee 必須為正，否則退化成硬裁')
    k = float(knee)
    out = rgb - k * torch.nn.functional.softplus((rgb - 1.) / k)               + k * torch.nn.functional.softplus(-rgb / k)
    return out.clamp(0., 1.)


def gamut_scale(lab_l, ab_source, ab_mapped, tol=1e-4, iters=24):
    """求最大的整體係數 s，使 `ab_source + s*(ab_mapped-ab_source)` 全圖仍在 sRGB 內。

    為什麼是**整體**一個係數而不是逐像素
    ────────────────────────────────────────────────────────────────
    逐像素的係數自己就是一張有邊的圖，把它乘進色度等於換一種方式加高頻。
    整體係數是空間上的常數，不可能引入任何空間頻率。代價是飽和的少數像素
    會決定整張圖的幅度，這是為了保證而付的價，記錄在 `gamut_scale` 欄位裡
    讓人看得到。

    s 對「有沒有越界」不是嚴格單調（Lab 到 sRGB 的邊界不是凸的），但沿著這條
    線段越靠近原圖越安全，二分法取到的是一個保守的可行解；回傳前再驗一次。
    計算不參與梯度（與 `project()` 同性質，是投影不是目標）。
    """
    offset = ab_mapped - ab_source
    lo, hi = 0.0, 1.0
    def inside(s):
        lab = torch.cat([lab_l, ab_source + s * offset], dim=1)
        rgb = lab_to_rgb(lab)
        return bool(((rgb >= -tol) & (rgb <= 1 + tol)).all())
    if inside(hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if inside(mid):
            lo = mid
        else:
            hi = mid
    if not inside(lo):
        raise ValueError('色域縮放找不到可行解；原圖本身就超出 sRGB 範圍')
    return lo


def _clamp_singular_values(m, max_gain):
    """把奇異值夾進上界。**只在 `reset` 與 `project` 裡用，不在前向裡用。**

    理由是這個運算會製造出會炸掉自己梯度的重根：兩個奇異值都超過上界時，
    `clamp(max=g)` 把它們夾成**完全相等的 g**，而 SVD 的反向傳播含
    `1/(s_i^2 - s_j^2)`，重根即除以零。實測 `RegionPaletteParam` 的一塊區域
    T0 奇異值被夾成 1.000000/1.000000 之後，第一次 backward 就是 nan，訓練
    目標由 87.66 直接變 nan。

    約束因此改由 `project()` 在每步之後套用（那本來就是 PGD 的形狀），前向
    只用 `T0 + delta`。未訓練的臂逐位元不變——它們的 `delta` 是零，而 `T0`
    在 `reset` 裡已經投影過。
    """
    u, s, vh = torch.linalg.svd(m)
    return (u * s.clamp(max=max_gain)) @ vh


def _project_orthogonal(m):
    """極分解取正交因子：奇異值恰為 1。

    `_clamp_singular_values` 只設上界，而起點的 `T0_ab` 常常是收縮映射——
    色度被壓扁（去飽和），付了顏色位移卻換到比較小的色度變化。奇異值恰為 1
    時色度梯度既不放大也不縮小，而色相仍可轉到任意角度，所以顏色位移不受
    「目標配色離來源有多遠」限制。

    **這個等距只在 Lab 的 (a,b) 平面上成立，不保證 RGB 的高通殘差不上升。**
    Lab→RGB 是非線性的，同一個色度梯度轉到不同色相之後映進 RGB 的梯度可以變大，
    而且變多少隨影像內容而異（`tests/test_chroma_rotation.py` 有反例與兩組
    量到的形狀）。`hf_ratio_rgb_total` 每一批都要照量，不可推定。
    """
    u, _, vh = torch.linalg.svd(m)
    return u @ vh


def _rotation_2x2(degrees, *, dtype, device):
    """(a,b) 平面上的旋轉。奇異值恰為 1，且色度的長度逐點保持。

    色相旋轉是這一族裡顏色位移最大的那個方向：轉 180 度把色度向量整個反向，
    位移是色度長度的兩倍，而 Lab 色度平面上的梯度長度完全沒有被放大。

    **RGB 的高通殘差比仍然要量。** Lab 的等距不轉譯成 RGB 的等距，實測在
    180 度附近會超過 1（見 `_project_orthogonal` 與
    `tests/test_chroma_rotation.py`）。角度是自變數，不是安全保證。
    """
    t = torch.as_tensor(float(degrees) * math.pi / 180., dtype=dtype, device=device)
    c, s = torch.cos(t), torch.sin(t)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


class ChromaAffineParam(NCFColorParam):
    """鎖住亮度，只變換色度；映射的增益有上界。

    與母類的差別只有兩處：可學參數是 2x2（作用在 a,b），以及 `raw_rgb` 把
    輸出的 L 直接換回輸入的 L。`T0` 仍由母類算出 3x3，這裡取它的 (a,b) 子區塊
    當起點——這樣起點仍然是論文的分布轉移，只是限制在色度平面上。

    `gamut` 決定映射後怎麼回到 sRGB。預設 `'soft'`；`'clip'` 與 `'scale'` 保留
    供對照，兩者各自的問題寫在 `soft_gamut` 的說明裡。注意只有後兩者會讓輸出的
    L 逐像素等於輸入——`'soft'` 為了保住幅度放棄了那個更強的性質，換來的是
    逐點的「不放大梯度」保證，那才是使用者要的那一條。
    """

    name = 'chroma_affine'

    def __init__(self, target_mean, target_cov, *, support, radius=.2,
                 cov_floor=1e-4, epsilon_lab=None, max_gain=1.0, gamut='soft',
                 amplitude=1.0, isometric=False, rotation_deg=None):
        if not (max_gain > 0):
            raise ValueError('max_gain must be positive')
        self.set_amplitude(amplitude)
        self.isometric = bool(isometric)
        if rotation_deg is not None and not self.isometric:
            raise ValueError('rotation_deg 只在 isometric 臂上有定義：非等距臂會'
                             '再過一次奇異值上界，指定的角度不會逐字生效')
        self.rotation_deg = None if rotation_deg is None else float(rotation_deg)
        if gamut not in ('soft', 'scale', 'clip'):
            raise ValueError("gamut 必須是 'soft'（平滑軟裁，預設）、"
                             "'scale'（整體縮放）或 'clip'（硬裁，供對照）")
        self.max_gain = float(max_gain)
        self.gamut = gamut
        self.last_gamut_scale = 1.0
        # 白化是為了讓三個通道的步長一致；這裡只剩兩個同尺度的色度通道，
        # 沒有那個問題，所以不繼承那條路徑，避免兩套投影規則互相干擾。
        super().__init__(target_mean, target_cov, support=support, radius=radius,
                         cov_floor=cov_floor, epsilon_lab=epsilon_lab, whiten=False)

    def reset(self, x01, seed=0):
        super().reset(x01, seed)
        # 母類建立的是 3x3 的增量；這裡換成 2x2，並把 T0 的色度子區塊
        # 先壓進增益上界內，讓起點本身就滿足「不放大色度對比」。
        if self.rotation_deg is not None:
            # 明確的色相旋轉。MK 的 T0 投影到 O(2) 之後通常只是很小的旋轉，
            # 量不到「顏色推到極限」那個問題；旋轉角是那個極限的自變數。
            self.T0_ab = _rotation_2x2(self.rotation_deg,
                                       dtype=self.T0.dtype, device=self.T0.device)
        elif self.isometric:
            self.T0_ab = _project_orthogonal(self.T0[1:, 1:])
        else:
            self.T0_ab = _clamp_singular_values(self.T0[1:, 1:], self.max_gain)
        self.delta = torch.zeros_like(self.T0_ab).requires_grad_(True)
        self.u = None

    def params(self):
        return [self.delta]

    def increment(self):
        return self.delta

    def set_amplitude(self, a):
        """幅度是投影的一部分，不參與梯度，與 `gamut_scale` 同性質。

        幅度是**空間常數**，所以不可能引入任何空間頻率；這是它與「逐像素
        縮放回色域」的關鍵差別，後者本身就是一張有邊的圖。
        """
        if not (0. <= float(a) <= 1.):
            raise ValueError('amplitude 必須落在 [0,1]：超過 1 的外插會讓有效'
                             '映射的奇異值突破 max_gain，就不再保證不放大對比')
        self.amplitude = float(a)

    def chroma_matrix(self):
        """實際生效的 2x2 映射，已套用增益上界。**不含幅度插值。**

        幅度作用在色度的位移上（`raw_rgb`），不作用在矩陣上，因為插值的另一端
        是恆等而不是零。含幅度的那個矩陣是 `effective_chroma_matrix`。

        **前向不投影。** 約束由 `project()` 在每步之後套用，理由見
        `_clamp_singular_values`：在前向做 SVD 會讓重根的梯度變成 nan。
        `reset` 已經把 `T0_ab` 投影過，`delta` 是零時因此逐位元等同投影後的值。
        """
        return self.T0_ab + self.delta

    def effective_chroma_matrix(self):
        """含幅度插值之後真正作用在色度上的 2x2。

        `(1-a)·I + a·M` 的奇異值不超過 `max(1, sigma_max(M))`，所以
        `max_gain = 1` 時幅度插值不可能破壞「不放大色度對比」那條保證。
        """
        m = self.chroma_matrix()
        eye = torch.eye(2, dtype=m.dtype, device=m.device)
        return (1. - self.amplitude) * eye + self.amplitude * m

    def raw_rgb(self, x):
        lab = rgb_to_lab(x).double()
        ab = lab[:, 1:] - self.source_mean[1:].to(lab)[None, :, None, None]
        out_ab = torch.einsum('ij,bjhw->bihw', self.chroma_matrix(), ab)
        out_ab = out_ab + self.target_mean[1:].to(out_ab)[None, :, None, None]
        # 幅度：往原圖的色度插值。空間常數，故不引入任何空間頻率。
        out_ab = lab[:, 1:] + self.amplitude * (out_ab - lab[:, 1:])
        if self.gamut == 'scale':
            # 硬裁會造邊，造邊就是高頻（見 `gamut_scale` 的說明）。改成整體縮放：
            # 係數是空間常數，所以不可能引入任何空間頻率。係數本身不參與梯度，
            # 與 `project()` 同性質，是投影不是目標。
            src = lab[:, 1:]
            with torch.no_grad():
                s = gamut_scale(lab[:, :1], src, out_ab.detach())
            self.last_gamut_scale = s
            out_ab = src + s * (out_ab - src)
        # L 逐像素照抄輸入：亮度的高頻因此完全不動。
        out = torch.cat([lab[:, :1], out_ab], dim=1)
        return lab_to_rgb(out).to(x.dtype)

    def render(self, x):
        # 母類用硬裁；這裡改走軟裁，理由見 `soft_gamut`。
        w = self.support.to(device=x.device, dtype=x.dtype)
        raw = self.raw_rgb(x)
        out = w * (soft_gamut(raw) if self.gamut == 'soft' else raw.clamp(0, 1)) + (1 - w) * x
        return torch.where(w > 0, out, x)

    @torch.no_grad()
    def project(self):
        """**不是幂等的，而且最後的 delta 可以超出 `radius`。**

        順序是 L∞ 盒 → 效果球 → 奇異值上界，而最後一步作用在 `T0_ab + delta`
        上，減回去之後不再保證落在盒內，`radius` 因此不是 `delta` 的硬上界；
        `RegionPaletteParam.project` 有同構造下量到的數字。順序與原文
        `project_params` 相同，改動要整批重跑。
        """
        self.delta.clamp_(-self.radius, self.radius)
        if self.epsilon_lab is not None:
            sig = self.source_sigma[1:].to(self.delta)
            effect = (self.delta * sig[None, :]).norm(dim=1)
            self.delta.mul_(
                (self.epsilon_lab / effect.clamp_min(1e-12)).clamp(max=1.)[:, None])
        m = self.T0_ab + self.delta
        m = (_project_orthogonal(m) if self.isometric
             else _clamp_singular_values(m, self.max_gain))
        self.delta.copy_(m - self.T0_ab)

    @torch.no_grad()
    def diagnostics(self, x):
        m = self.chroma_matrix()
        s = torch.linalg.svdvals(m)
        base = torch.linalg.svdvals(self.T0_ab)
        raw = self.raw_rgb(x)
        return {'name': self.name, 'max_gain': self.max_gain, 'gamut': self.gamut,
                'amplitude': self.amplitude, 'isometric': int(self.isometric),
                'rotation_deg': ('' if self.rotation_deg is None
                                 else self.rotation_deg),
                'gamut_scale': float(self.last_gamut_scale),
                'clipping_fraction': float(((raw < 0) | (raw > 1)).double().mean()),
                'clipping_max': float((raw - raw.clamp(0, 1)).abs().max()),
                'chroma_gain_max': float(s.max()), 'chroma_gain_min': float(s.min()),
                'chroma_gain_at_start': float(base.max()),
                'gain_boundary_hit': int(bool(s.max() >= self.max_gain - 1e-9)),
                'T_distance': float(self.delta.detach().norm()),
                'support_area': float(self.support.to(torch.float64).mean()),
                **highfreq_report(x, self.render(x))}


class RegionPaletteParam:
    """每塊區域一組目標配色，權重圖重度低通後混合。

    `regions` 是 [(weight, target_mean, target_cov), ...]。`weight` 是
    (1,1,H,W) 的非負圖，會先被 `blur_sigma` 的高斯模糊，再逐像素正規化成
    和為 1。**模糊是這個類別存在的理由**：空間變化的頻寬由 sigma 決定，
    不由原始遮罩的邊決定，所以「衣物一個色調、背景另一個色調」不會在交界
    留下一條高頻的邊。sigma 太小就失去這個保證，因此它沒有預設值。

    來源統計取自整張支撐（與 `NCFColorParam` 一致），但每塊區域各自算一個
    Monge–Kantorovitch 矩陣。亮度預設鎖住，理由同 `ChromaAffineParam`。
    """

    name = 'region_palette'

    def __init__(self, regions, *, support, blur_sigma, radius=.2, cov_floor=1e-4,
                 epsilon_lab=None, max_gain=1.0, lock_luminance=True, gamut='soft',
                 amplitude=1.0):
        if len(regions) < 2:
            raise ValueError('至少要兩塊區域，否則就是單一全域配色，用 ChromaAffineParam')
        if not (blur_sigma > 0):
            raise ValueError('blur_sigma 必須為正：權重圖的平滑度就是不加高頻的保證')
        if not lock_luminance:
            raise NotImplementedError('目前只實作鎖亮度的版本；放開它要另外量亮度高頻')
        self.regions = list(regions)
        self.support = support
        self.blur_sigma = float(blur_sigma)
        self.cov_floor = cov_floor
        self.epsilon_lab = None if epsilon_lab is None else float(epsilon_lab)
        self.max_gain = float(max_gain)
        if gamut not in ('soft', 'scale', 'clip'):
            raise ValueError("gamut 必須是 'soft'、'scale' 或 'clip'")
        self.gamut = gamut
        self.last_gamut_scale = 1.0
        self.lock_luminance = True
        self.set_amplitude(amplitude)
        self.set_radius(radius)

    def set_radius(self, r):
        if not (r >= 0) or r != r:
            raise ValueError('radius must be finite and nonnegative')
        self.radius = float(r)

    def set_amplitude(self, a):
        """與 `ChromaAffineParam.set_amplitude` 同一個約定與同一個理由。"""
        if not (0. <= float(a) <= 1.):
            raise ValueError('amplitude 必須落在 [0,1]')
        self.amplitude = float(a)

    def reset(self, x01, seed=0):
        if x01.ndim != 4 or x01.shape[:2] != (1, 3):
            raise ValueError('RegionPalette requires one RGB image')
        w = self.support.to(device=x01.device, dtype=torch.float64)
        lab = rgb_to_lab(x01).double()
        flat, wf = lab[0].flatten(1), w[0, 0].flatten()
        total = wf.sum()
        if total < 2:
            raise ValueError('support holds less than two pixels of weight')
        self.source_mean = (flat * wf).sum(1) / total
        centred = flat - self.source_mean[:, None]
        source, self.source_audit = regularize_covariance(
            (centred * wf) @ centred.T / total, self.cov_floor)
        self.source_sigma = source.diagonal().clamp_min(self.cov_floor).sqrt()

        # 權重：先模糊再正規化。模糊後仍可能有像素全部為零（例如支撐外），
        # 那裡的權重退化成第一塊區域，但支撐外本來就會被 render 換回原圖。
        maps = torch.cat([r[0].to(device=x01.device, dtype=torch.float64)
                          for r in self.regions], dim=1)
        if maps.shape[-2:] != x01.shape[-2:]:
            raise ValueError('每塊區域的權重圖必須與影像同尺寸')
        if float(maps.min()) < 0:
            raise ValueError('權重圖不可為負')
        smooth = gaussian_blur(maps, self.blur_sigma)
        self.weights = smooth / smooth.sum(dim=1, keepdim=True).clamp_min(1e-12)

        self.T0 = []
        self.target_means = []
        for _, mean, cov in self.regions:
            target, _ = regularize_covariance(_to_like(cov, source), self.cov_floor)
            m = mk_matrix(source, target).to(lab.device)
            self.T0.append(_clamp_singular_values(m[1:, 1:], self.max_gain))
            self.target_means.append(_to_like(mean, source)[1:])
        self.delta = torch.zeros(len(self.regions), 2, 2, dtype=torch.float64,
                                 device=lab.device).requires_grad_(True)

    def params(self):
        return [self.delta]

    def increment(self):
        return self.delta

    def raw_rgb(self, x):
        lab = rgb_to_lab(x).double()
        ab = lab[:, 1:] - self.source_mean[1:].to(lab)[None, :, None, None]
        out = torch.zeros_like(ab)
        for k in range(len(self.regions)):
            # 前向不投影，理由見 `_clamp_singular_values`；約束由 `project()`
            # 在每步之後套用。
            m = self.T0[k] + self.delta[k]
            mapped = torch.einsum('ij,bjhw->bihw', m, ab)
            mapped = mapped + self.target_means[k].to(mapped)[None, :, None, None]
            out = out + self.weights[:, k:k+1] * mapped
        # 幅度：與 `ChromaAffineParam` 同一行插值，空間常數。
        out = lab[:, 1:] + self.amplitude * (out - lab[:, 1:])
        if self.gamut == 'scale':
            src = lab[:, 1:]
            with torch.no_grad():
                s = gamut_scale(lab[:, :1], src, out.detach())
            self.last_gamut_scale = s
            out = src + s * (out - src)
        return lab_to_rgb(torch.cat([lab[:, :1], out], dim=1)).to(x.dtype)

    def render(self, x):
        w = self.support.to(device=x.device, dtype=x.dtype)
        raw = self.raw_rgb(x)
        out = w * (soft_gamut(raw) if self.gamut == 'soft' else raw.clamp(0, 1)) + (1 - w) * x
        return torch.where(w > 0, out, x)

    @torch.no_grad()
    def project(self):
        """**不是幂等的，而且最後的 delta 可以超出 `radius`。**

        與 `ChromaAffineParam.project` 同一個順序與同一個後果。實測：
        `collision_region` 臂在真實影像上跑完無導數搜尋，收尾的 `|delta|`
        最大是 0.214925，而 `radius` 是 0.2（CPU 與 CUDA 兩邊相同）。可行集合
        比設定寫的盒子稍大。
        """
        self.delta.clamp_(-self.radius, self.radius)
        if self.epsilon_lab is not None:
            sig = self.source_sigma[1:].to(self.delta)
            effect = (self.delta * sig[None, None, :]).norm(dim=2)
            self.delta.mul_(
                (self.epsilon_lab / effect.clamp_min(1e-12)).clamp(max=1.)[..., None])
        for k in range(len(self.T0)):
            m = _clamp_singular_values(self.T0[k] + self.delta[k], self.max_gain)
            self.delta[k].copy_(m - self.T0[k])

    def state_dict(self):
        return {'delta': self.delta.detach().clone()}

    def load_state_dict(self, state):
        self.delta = state['delta'].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def diagnostics(self, x):
        gains = [float(torch.linalg.svdvals(
            _clamp_singular_values(self.T0[k]+self.delta[k], self.max_gain)).max())
            for k in range(len(self.regions))]
        raw = self.raw_rgb(x)
        return {'name': self.name, 'regions': len(self.regions), 'gamut': self.gamut,
                'amplitude': self.amplitude,
                'gamut_scale': float(self.last_gamut_scale),
                'clipping_fraction': float(((raw < 0) | (raw > 1)).double().mean()),
                'clipping_max': float((raw - raw.clamp(0, 1)).abs().max()),
                'blur_sigma': self.blur_sigma, 'max_gain': self.max_gain,
                'chroma_gain_max': max(gains),
                'weight_entropy': float(-(self.weights.clamp_min(1e-12).log()
                                          * self.weights).sum(1).mean()),
                'support_area': float(self.support.to(torch.float64).mean()),
                **highfreq_report(x, self.render(x))}


@torch.no_grad()
def highfreq_report(x, x_def, sigma=2.0):
    """防禦圖相對原圖的高通能量比，逐通道報。大於 1 代表高頻被加上去了。

    高通殘差取「影像減去高斯模糊」，與 `src/purify/ops.py` 的模糊同一個實作，
    所以這個量測與淨化端看到的是同一組頻帶。Lab 的三個通道分開報：亮度那一項
    是最要緊的，因為紋理幾乎都在 L；色度兩項用來檢查色彩映射有沒有放大對比。

    **RGB 那組才是操作性的量。** 淨化作用在像素域，JPEG、模糊與擴散重建吃的都是
    RGB 的高頻。Lab 的色度比值在接近無彩的影像上會失真放大：`task_env_weather_126577`
    的 a/b 高通能量基準只有 0.70，任何色偏都讓比值衝到 5 以上，但同一張圖同一個
    載體的 `hf_ratio_rgb_total` 是 0.927——像素域的總能量其實下降了。Lab 那三項
    保留供診斷（看得出改動落在亮度還是色度），不要拿來判定有沒有加高頻。
    """
    out = {}
    for space, a, b in (('rgb', x, x_def),
                        ('lab', rgb_to_lab(x), rgb_to_lab(x_def))):
        ha = a - gaussian_blur(a, sigma)
        hb = b - gaussian_blur(b, sigma)
        ea = ha.pow(2).mean(dim=(0, 2, 3))
        eb = hb.pow(2).mean(dim=(0, 2, 3))
        ratio = (eb / ea.clamp_min(1e-12)).tolist()
        names = ('r', 'g', 'b') if space == 'rgb' else ('L', 'a', 'b')
        for name, v in zip(names, ratio):
            out[f'hf_ratio_{space}_{name}'] = float(v)
        out[f'hf_ratio_{space}_total'] = float(eb.sum() / ea.sum().clamp_min(1e-12))
    out['hf_sigma'] = float(sigma)
    return out
