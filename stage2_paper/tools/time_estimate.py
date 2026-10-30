"""Measure throughput on the current GPU to estimate full-job runtimes (no outputs written).

Random-weight models; timing only. Reports seconds for: one backbone training epoch (01), one head-training
epoch (05), and one subset forward per fault location (08), using real CIFAR loaders for the training numbers.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset.datamodule import get_cifar10_dataloaders  # noqa: E402
from faults.evaluator import SubsetEvaluator  # noqa: E402
from faults.inject import LocationSampler, fault_locations  # noqa: E402
from models.backbone import build_backbone  # noqa: E402
from models.searched_exit_model import SearchedMultiExitVGG  # noqa: E402
from utils.freeze import head_parameters  # noqa: E402
from utils.seed import set_seed  # noqa: E402

N_BATCHES = 60


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    set_seed(42)
    dev = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    train_loader, _, _ = get_cifar10_dataloaders(batch_size=128, seed=42, num_workers=4)
    n_batches_epoch = len(train_loader)

    model = build_backbone().to(dev)
    opt = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    it = iter(train_loader)
    for i in range(N_BATCHES + 5):
        if i == 5:
            sync()
            t0 = time.time()
        x, y = next(it)
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sync()
    per_batch = (time.time() - t0) / N_BATCHES
    print(f"[01] {per_batch*1000:.1f} ms/batch -> train epoch {per_batch*n_batches_epoch:.1f} s "
          f"(+ val ~{per_batch*n_batches_epoch*0.04:.1f} s)", flush=True)

    ee = SearchedMultiExitVGG(model.features, model.avgpool, model.classifier, [16, 23], ["A4", "A3"]).to(dev)
    opt = optim.Adam(head_parameters(ee), lr=1e-3)
    it = iter(train_loader)
    for i in range(N_BATCHES + 5):
        if i == 5:
            sync()
            t0 = time.time()
        x, y = next(it)
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = ee(x)
        loss = sum(crit(z, y) for z in out[:-1])
        loss.backward()
        opt.step()
    sync()
    per_batch = (time.time() - t0) / N_BATCHES
    print(f"[05] {per_batch*1000:.1f} ms/batch -> head epoch {per_batch*n_batches_epoch:.1f} s", flush=True)

    ee.eval()
    imgs = torch.randn(9000, 3, 32, 32, device=dev)
    ev = SubsetEvaluator(ee, imgs, 1000)
    locs = fault_locations(ee)
    for loc in locs:
        s = LocationSampler(loc, locs[loc])
        sync()
        t0 = time.time()
        for _ in range(5):
            f = s.flip(s.draw(np.random.default_rng(0)))
            ev.logits(loc)
            s.unflip(f)
        sync()
        print(f"[08] {loc}: {(time.time()-t0)/5:.3f} s per forward on 9000 images", flush=True)


if __name__ == "__main__":
    main()
