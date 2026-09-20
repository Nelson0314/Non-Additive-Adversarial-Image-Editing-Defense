"""Validated physical CIELab moments, with explicit class/source provenance.

NCF §3.2 uses 150 ADE20K classes and 20 clustered palettes per class. The
official release `150_20_hist.zip` **is** used here: `scripts/build_ncf_library.py`
reads its `(150, 20, 32, 32, 32)` RGB histograms and takes the Lab moments
analytically from the histogram weights. (An earlier note in this file claimed
the archive was unavailable; it is a GitHub release asset and downloads fine.)

What is still modified_from_paper is the *matching*, not the library: the paper
picks a palette per segmented ADE20K class and mixes them by area, which needs
their Swin-T UperNet. Pass `ade20k_classes` to name the class of a region whose
identity is already known and recover the matching for that region.
"""
import hashlib
import json
from pathlib import Path

import torch

from .ncf_param import regularize_covariance

LAB_UNITS = 'CIELab_D65_L_0_100_ab_signed'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NCFLibrary:
    def __init__(self, path, *, expected_sha256, required_classes, cov_floor=1e-4,
                 ade20k_classes=None):
        """`ade20k_classes` restores the paper's semantic matching.

        NCF §3.2–3.3 never applies an arbitrary palette to an arbitrary region:
        it draws from the **segmented class**. Collapsing all 3000 records onto
        one pseudo-class removed that, and the search then selected near-black
        (target L = 1.14) and near-white (L = 94.68) palettes for portraits.
        Naming the classes here restores the matching without running a
        segmenter, for a region whose class is already known — a clothing
        carrier is ADE20K `apparel`. `None` keeps the whole library.
        """
        self.sha256 = sha256(path)
        if self.sha256 != expected_sha256:
            raise ValueError(f'library hash mismatch: {path}')
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data['lab_units'] != LAB_UNITS:
            raise ValueError('unsupported Lab units (OpenCV uint8 Lab is not physical Lab)')
        self.metadata = dict(data['provenance'])
        self.records = data['distributions']
        self.ade20k_classes = None if ade20k_classes is None else sorted(ade20k_classes)
        if self.ade20k_classes is not None:
            available = {r.get('ade20k_class') for r in self.records}
            unknown = sorted(set(self.ade20k_classes) - available)
            if unknown:
                raise ValueError(f'unknown ADE20K classes {unknown}; '
                                 f'the library carries {len(available)} names')
            self.records = [r for r in self.records
                            if r.get('ade20k_class') in set(self.ade20k_classes)]
            if not self.records:
                raise ValueError(f'no records left after restricting to {self.ade20k_classes}')
        self.metadata['ade20k_classes'] = '|'.join(self.ade20k_classes or [])
        self.metadata['records_in_pool'] = len(self.records)
        self.audit = []
        names = [r['id'] for r in self.records]
        if len(names) != len(set(names)):
            raise ValueError('duplicate distribution ids')
        missing = sorted(set(required_classes)-{r['class'] for r in self.records})
        if missing:
            raise ValueError(f'missing library classes: {missing}; available: {sorted({r["class"] for r in self.records})}')
        for r in self.records:
            mean = torch.tensor(r['mean'], dtype=torch.float64)
            if mean.shape != (3,) or not torch.isfinite(mean).all() or not 0 <= mean[0] <= 100 or (mean[1:].abs() > 128).any():
                raise ValueError(f'invalid physical Lab mean: {r["id"]}')
            digest = r['source_sha256']
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'invalid source hash: {r["id"]}')
            _, audit = regularize_covariance(torch.tensor(r['covariance'], dtype=torch.float64), cov_floor)
            self.audit.append({'distribution_id': r['id'], 'class': r['class'], **audit})

    def sample(self, weights, generator):
        """Area-weighted mixture includes between-class covariance (§3.3)."""
        if not weights or any(w <= 0 for w in weights.values()) or abs(sum(weights.values())-1) > 1e-6:
            raise ValueError('class area weights must be positive and sum to one')
        selected = []
        for cls, w in sorted(weights.items()):
            pool = [r for r in self.records if r['class'] == cls]
            if not pool:
                raise ValueError(f'missing library class: {cls}')
            r = pool[int(torch.randint(len(pool), (), generator=generator))]
            selected.append((w, r, torch.tensor(r['mean'], dtype=torch.float64),
                             torch.tensor(r['covariance'], dtype=torch.float64)))
        mean = sum(w*m for w, _, m, _ in selected)
        cov = sum(w*(c+torch.outer(m-mean, m-mean)) for w, _, m, c in selected)
        return mean, cov, '|'.join(r['id'] for _, r, _, _ in selected)
