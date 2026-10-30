import torch

from dataset.datamodule import get_cifar10_dataloaders
from models.backbone import build_backbone
from models.exit_heads import HEAD_BUILDERS
from utils.seed import set_seed


def first_batch(seed, workers):
    tr, _, _ = get_cifar10_dataloaders(batch_size=32, seed=seed, num_workers=workers)
    return next(iter(tr))


def test_same_seed_identical_first_batch_with_workers():
    x1, y1 = first_batch(42, 2)
    x2, y2 = first_batch(42, 2)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_different_seed_different_order():
    _, y1 = first_batch(42, 0)
    _, y2 = first_batch(43, 0)
    assert not torch.equal(y1, y2)


def test_same_seed_identical_init_and_outputs():
    x, _ = first_batch(42, 0)
    outs = []
    for _ in range(2):
        set_seed(42)
        model = build_backbone().eval()
        head = HEAD_BUILDERS["A4"](in_channels=256).eval()
        with torch.no_grad():
            feats = model.features[:17](x)
            outs.append((model(x), head(feats)))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


def test_second_epoch_order_differs_but_is_reproducible():
    def two_epochs():
        tr, _, _ = get_cifar10_dataloaders(batch_size=32, seed=42, num_workers=0)
        return [next(iter(tr))[1] for _ in range(2)]
    a, b = two_epochs(), two_epochs()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], a[1])
