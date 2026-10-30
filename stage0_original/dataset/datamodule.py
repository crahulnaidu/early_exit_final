import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar10_dataloaders(data_dir="./data", batch_size=128, seed=42):
    set_seed(seed)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset_raw = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    # Fixed 45k train / 5k val split
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_idx, val_idx = indices[:45000], indices[45000:]

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(val_dataset_raw, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
