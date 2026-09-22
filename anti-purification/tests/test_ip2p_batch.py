import torch

from src.models.ip2p import IP2PWrapper


class _FakePipe:
    """記下 pipeline 收到什麼，並依 generator 逐張產生可預測的輸出。

    真的管線要 4 GB 權重，這裡只驗**接線**：批次有沒有把每張圖配到自己的
    generator、prompt 有沒有逐張對上。輸出取自各自的 generator，所以
    「批次 == 逐張」這條性質在假管線上就可以被否證。
    """

    def __init__(self):
        self.calls = []

    def __call__(self, *, prompt, image, generator, output_type, **kw):
        self.calls.append({'prompt': prompt, 'image': image,
                           'generator': generator, **kw})
        gens = generator if isinstance(generator, (list, tuple)) else [generator]
        n = image.shape[0]
        if len(gens) not in (1, n):
            raise AssertionError('generator 的數量必須是 1 或批次大小')
        out = torch.stack([
            torch.rand(3, *image.shape[-2:],
                       generator=gens[i if len(gens) == n else 0])
            for i in range(n)])
        return type('R', (), {'images': out})()

    def to(self, *a, **k):
        return self

    def set_progress_bar_config(self, **k):
        pass


def _wrapper():
    pipe = _FakePipe()
    w = IP2PWrapper.__new__(IP2PWrapper)
    w.pipe = pipe
    w.device = torch.device('cpu')
    w.compute_dtype = torch.float32
    w.backbone_dtype = w.vae_dtype = torch.float32

    class _V:
        dtype = torch.float32
    w._vae = _V()
    return w, pipe


def _patch_vae(w):
    type(w).vae = property(lambda self: self._vae)


def test_batch_gives_each_image_its_own_generator():
    """整批共用一個 generator 會讓輸出取決於批次怎麼分組，那是靜默的失效。"""
    w, pipe = _wrapper()
    _patch_vae(w)
    x = torch.rand(1, 3, 16, 16)
    w.edit_batch([x, x, x], ['a', 'b', 'c'], [1, 2, 3], steps=2)
    gens = pipe.calls[0]['generator']
    assert isinstance(gens, list)
    assert len(gens) == 3


def test_batch_passes_the_prompts_through_in_order():
    w, pipe = _wrapper()
    _patch_vae(w)
    x = torch.rand(1, 3, 16, 16)
    w.edit_batch([x, x], ['first', 'second'], [7, 8], steps=2)
    assert pipe.calls[0]['prompt'] == ['first', 'second']
    assert pipe.calls[0]['image'].shape[0] == 2


def test_batch_rejects_mismatched_list_lengths():
    w, _ = _wrapper()
    _patch_vae(w)
    x = torch.rand(1, 3, 16, 16)
    try:
        w.edit_batch([x, x], ['only one'], [1, 2], steps=2)
    except ValueError as e:
        assert '長度' in str(e)
    else:
        raise AssertionError('長度不一致必須拋錯，不得靜默截斷')


def test_batch_rejects_an_empty_batch():
    w, _ = _wrapper()
    _patch_vae(w)
    try:
        w.edit_batch([], [], [], steps=2)
    except ValueError as e:
        assert '空' in str(e)
    else:
        raise AssertionError('空批次必須拋錯')


def test_image_range_is_mapped_to_minus_one_to_one():
    """管線吃的是 [-1,1]，`edit` 與 `edit_batch` 必須用同一個映射。"""
    w, pipe = _wrapper()
    _patch_vae(w)
    x = torch.zeros(1, 3, 16, 16)
    w.edit_batch([x], ['p'], [1], steps=2)
    assert torch.allclose(pipe.calls[0]['image'], torch.full((1, 3, 16, 16), -1.0))
