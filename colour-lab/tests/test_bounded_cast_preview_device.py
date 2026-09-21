"""只用 CPU 與無運算的 meta 裝置檢查 preview 的裝置傳遞及原有精度。"""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bounded_cast_preview as preview


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'cuda:1'])
def test_device_option_is_parsed_without_allocating(device):
    args = preview.build_parser().parse_args(['--device', device])
    assert args.device == torch.device(device)
    assert preview.build_parser().parse_args([]).device == torch.device('cpu')


def test_load_image_preserves_pixels_and_moves_only_device(tmp_path):
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    path = tmp_path / 'input.png'
    Image.fromarray(pixels).save(path)
    legacy = torch.from_numpy(pixels).permute(2, 0, 1)[None].float() / 255
    assert torch.equal(preview.load_image(path, 4), legacy)
    assert torch.equal(preview.load_image(path, 4, torch.device('cpu')), legacy)
    moved = preview.load_image(path, 4, 'meta')
    assert moved.device.type == 'meta' and moved.dtype == torch.float32
    assert moved.shape == legacy.shape


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_device_plumbing_keeps_certificate_in_float64(dtype):
    args = preview.build_parser().parse_args(['--device', 'cpu'])
    x = torch.full((1, 3, 2, 3), 0.4, dtype=dtype, device=args.device)
    carrier = preview.make_carrier(args, (32, 24), 0)
    carrier.reset(x, 0)
    prepared = carrier.prepare(x)
    assert all(t.dtype == torch.float64 for t in
               (prepared.base_lab, prepared.radius, prepared.fraction))
    assert all(p.dtype == dtype and p.device == x.device for p in carrier.params())
    rendered = carrier.render_prepared(prepared)
    assert rendered.dtype == dtype and rendered.device == x.device
    assert all(a.dtype == np.float64 for a in preview.measured_maps(x, rendered))


def test_main_default_and_explicit_cpu_outputs_match_without_optimisation(tmp_path, monkeypatch):
    """以固定 render 取代搜尋，檢查 CLI 到 PNG／CSV 的整條裝置路徑。"""
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3) + 30
    path = tmp_path / 'input.png'
    Image.fromarray(pixels).save(path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'images': [{'id': 'tiny', 'path': str(path)}]}),
                        encoding='utf-8')
    devices = []

    def fixed_candidate(x, args, pair, image_seed):
        devices.append((args.device, x.device))
        carrier = preview.make_carrier(args, pair, 0)
        carrier.reset(x, image_seed)
        with torch.no_grad():
            y = carrier.render(x)
        return dict(carrier=carrier, image=y, restart=0, step=0, seed=image_seed)

    monkeypatch.setattr(preview, 'ascend', fixed_candidate)
    outputs = []
    previous_threads = torch.get_num_threads()
    try:
        for name, extra in [('default', []), ('cpu', ['--device', 'cpu'])]:
            out = tmp_path / name
            preview.main(['--manifest', str(manifest), '--out', str(out),
                          '--size', '4', '--bounds', '32:24', '--threads', '1', *extra])
            with (out / 'results.csv').open(encoding='utf-8', newline='') as handle:
                row = next(csv.DictReader(handle))
            row.pop('seconds')
            outputs.append((row, (out / 'C32_d24/tiny.png').read_bytes()))
    finally:
        torch.set_num_threads(previous_threads)
    assert outputs[0] == outputs[1]
    assert devices == [(torch.device('cpu'), torch.device('cpu'))] * 2
