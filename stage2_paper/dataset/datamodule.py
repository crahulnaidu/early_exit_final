from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import DATA_DIR
from utils.seed import make_generator, seed_worker
from utils.splits import load_splits

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_cifar10_dataloaders(batch_size=128, seed=42, num_workers=4, quick=False, data_dir=DATA_DIR):
    """45k train (augmented) / 5k val (eval transforms) / 10k test (eval transforms).

    Split indices come from outputs/splits.json (verified on load, fixed SPLIT_SEED=42); `seed` only drives
    shuffling, augmentation and workers. The data folder is read-only
    (download=False). Every loader gets a seeded Generator and worker_init_fn.
    quick=True keeps only the first 2 batches of each split (smoke tests).
    """
    splits = load_splits()  # fixed SPLIT_SEED, independent of the run seed
    train_idx, val_idx = splits["train_idx"], splits["val_idx"]

    train_full = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=TRAIN_TRANSFORM)
    val_full = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=EVAL_TRANSFORM)
    test_full = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=EVAL_TRANSFORM)
    test_idx = list(range(len(test_full)))

    if quick:
        n = 2 * batch_size
        train_idx, val_idx, test_idx = train_idx[:n], val_idx[:n], test_idx[:n]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)
    test_set = Subset(test_full, test_idx)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True, worker_init_fn=seed_worker, generator=make_generator(seed))

    return loader(train_set, True), loader(val_set, False), loader(test_set, False)


def get_eval_loader(split, indices, batch_size=500, num_workers=4, seed=42, data_dir=DATA_DIR):
    """Eval-transform loader (no augmentation, no shuffle) over CIFAR-10 `split` ("train" or "test") indices."""
    ds = datasets.CIFAR10(root=str(data_dir), train=(split == "train"), download=False, transform=EVAL_TRANSFORM)
    return DataLoader(Subset(ds, list(indices)), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, worker_init_fn=seed_worker, generator=make_generator(seed))
