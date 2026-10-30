"""Proxy head training on cached frozen-backbone features (Step 4/5).

FeatureCache keeps the cached train-subset / val features on the GPU (fp16) and computes non-cached positions on the
fly from the cached images. train_proxy_head trains ONE head for `epochs` epochs with the 05 KD loss (teacher = cached
final-classifier logits), Adam 1e-3 / wd 1e-4 / cosine, batch 128, no augmentation, seeded by the head's config so the
result does not depend on trial order. proxy_val_logits memoises the VAL logits (never head weights) on disk.
"""
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

from config import CACHE_DIR
from models.backbone import build_backbone
from models.head_builder import cfg_key, make_head, normalize_cfg
from utils.ckpt import load_checkpoint
from utils.paths import safe_path
from utils.train_heads import kd_loss

PROXY_EPOCHS = 8
PROXY_BATCH = 128


class FeatureCache:
    def __init__(self, device, quick=False, base_ckpt=None):
        self.dir = (CACHE_DIR / "quick") if quick else CACHE_DIR
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)
        self.device = device
        self.base_ckpt = base_ckpt
        self._feats, self._backbone = {}, None
        self.labels = {s: torch.from_numpy(np.load(self.dir / f"{s}_labels.npy")).to(device) for s in ("train", "val")}
        self.final = {s: torch.from_numpy(np.load(self.dir / f"{s}_final_logits.npy")).to(device)
                      for s in ("train", "val")}

    def feats(self, split, pos):
        if (split, pos) not in self._feats:
            path = self.dir / f"{split}_p{pos}.npy"
            if path.is_file():
                arr = torch.from_numpy(np.ascontiguousarray(np.load(path, mmap_mode="r")))
            else:
                arr = self._on_the_fly(split, pos)
            self._feats[(split, pos)] = arr.to(self.device)
        return self._feats[(split, pos)]

    def _on_the_fly(self, split, pos):
        if self._backbone is None:
            assert self.base_ckpt is not None, f"position {pos} is not cached and no backbone checkpoint was given"
            b = build_backbone()
            load_checkpoint(self.base_ckpt, b)
            self._backbone = b.features.to(self.device).eval()
        imgs = np.load(self.dir / f"{split}_images.npy", mmap_mode="r")
        out = []
        with torch.no_grad():
            for a in range(0, imgs.shape[0], 500):
                x = torch.from_numpy(np.ascontiguousarray(imgs[a:a + 500])).to(self.device).float()
                out.append(self._backbone[: pos + 1](x).half().cpu())
        return torch.cat(out)


def head_seed(position, cfg, base_seed):
    return int(hashlib.sha256(f"{cfg_key(position, cfg)}|{base_seed}".encode()).hexdigest()[:8], 16)


def train_proxy_head(cache, position, cfg, epochs=PROXY_EPOCHS, base_seed=42):
    """Train one head on cached features; return its VAL logits [N_val, 10] (float32, CPU)."""
    dev = cache.device
    seed = head_seed(position, cfg, base_seed)
    torch.manual_seed(seed)
    head = make_head(position, cfg).to(dev)
    xtr, ytr, teacher = cache.feats("train", position), cache.labels["train"], cache.final["train"]
    n = xtr.shape[0]
    g = torch.Generator().manual_seed(seed)
    opt = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for a in range(0, n, PROXY_BATCH):
            idx = perm[a:a + PROXY_BATCH]
            opt.zero_grad()
            loss = kd_loss(head(xtr[idx].float()), teacher[idx], ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    head.eval()
    xv = cache.feats("val", position)
    with torch.no_grad():
        logits = torch.cat([head(xv[a:a + 1000].float()) for a in range(0, xv.shape[0], 1000)])
    return logits.float().cpu()


def proxy_val_logits(cache, position, cfg, epochs=PROXY_EPOCHS, base_seed=42):
    """Memoised VAL logits of the proxy-trained head (shared on disk across processes; weights are not stored)."""
    cfg = normalize_cfg(cfg) if isinstance(cfg, dict) else cfg
    key = f"{cfg_key(position, cfg) if isinstance(cfg, dict) else f'p{position}_{cfg}'}|e{epochs}|s{base_seed}"
    path = cache.dir / "proxy_logits" / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".pt")
    if path.is_file():
        return torch.load(path)["logits"], True
    t0 = time.time()
    logits = train_proxy_head(cache, position, cfg, epochs, base_seed)
    tmp = safe_path(path.with_suffix(f".tmp{os.getpid()}"))
    torch.save({"key": key, "logits": logits, "seconds": time.time() - t0}, tmp)
    os.replace(tmp, path)
    return logits, False
