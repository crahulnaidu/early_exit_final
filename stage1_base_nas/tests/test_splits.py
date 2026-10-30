import re
from pathlib import Path

from torchvision import transforms

from dataset.datamodule import get_cifar10_dataloaders
from utils.splits import compute_split, load_splits

ROOT = Path(__file__).resolve().parent.parent


def test_splits_file_matches_source_split():
    data = load_splits()
    tr, va = compute_split(42)
    assert data["train_idx"] == tr and data["val_idx"] == va
    assert len(set(tr) | set(va)) == 50000


def test_loaders_use_split_indices_and_eval_transforms():
    data = load_splits()
    tr, va, te = get_cifar10_dataloaders(batch_size=64, seed=42, num_workers=0)
    assert list(tr.dataset.indices) == data["train_idx"]
    assert list(va.dataset.indices) == data["val_idx"]
    assert len(te.dataset) == 10000
    random_ops = (transforms.RandomCrop, transforms.RandomHorizontalFlip)
    assert not any(isinstance(t, random_ops) for t in va.dataset.dataset.transform.transforms)
    assert not any(isinstance(t, random_ops) for t in te.dataset.dataset.transform.transforms)
    assert any(isinstance(t, random_ops) for t in tr.dataset.dataset.transform.transforms)
    assert va.sampler.__class__.__name__ == "SequentialSampler"
    assert te.sampler.__class__.__name__ == "SequentialSampler"


def test_quick_loaders_are_prefixes():
    data = load_splits()
    tr, va, te = get_cifar10_dataloaders(batch_size=32, seed=42, num_workers=0, quick=True)
    assert list(tr.dataset.indices) == data["train_idx"][:64]
    assert list(va.dataset.indices) == data["val_idx"][:64]


def test_every_script_gets_data_only_through_the_datamodule():
    for path in (ROOT / "scripts").rglob("*.py"):
        src = path.read_text()
        assert not re.search(r"\b(Subset|random_split|RandomState|CIFAR10)\(", src), path.name
