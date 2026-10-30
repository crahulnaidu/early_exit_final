"""Time proxy-head training at full size (20k x 8 epochs) with synthetic fp16 features (no outputs written)."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.backbone import FEATURE_SHAPES  # noqa: E402
from utils.proxy import train_proxy_head  # noqa: E402


class FakeCache:
    def __init__(self, device, pos, n_train=20000, n_val=5000):
        self.device = device
        C, H, W = FEATURE_SHAPES[pos]
        self._f = {"train": torch.randn(n_train, C, H, W, device=device, dtype=torch.float16),
                   "val": torch.randn(n_val, C, H, W, device=device, dtype=torch.float16)}
        self.labels = {"train": torch.randint(0, 10, (n_train,), device=device),
                       "val": torch.randint(0, 10, (n_val,), device=device)}
        self.final = {"train": torch.randn(n_train, 10, device=device)}

    def feats(self, split, pos):
        return self._f[split]


CASES = [
    ("cheap   p23 GAP h0", 23, {"n_conv": 0, "pooling": 1, "hidden_fc": 0}),
    ("typical p16 dw 1/4 pool2 h128", 16, {"n_conv": 1, "conv_type": "depthwise_separable", "channel_reduction": 0.25,
                                            "pooling": 2, "hidden_fc": 128}),
    ("heavy   p9 std x2 1/2 pool4 h256", 9, {"n_conv": 2, "conv_type": "standard", "channel_reduction": 0.5,
                                              "pooling": 4, "hidden_fc": 256}),
    ("worst   p2 std x2 1/2 pool4 h256", 2, {"n_conv": 2, "conv_type": "standard", "channel_reduction": 0.5,
                                              "pooling": 4, "hidden_fc": 256}),
]

if __name__ == "__main__":
    dev = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    for name, pos, cfg in CASES:
        cache = FakeCache(dev, pos)
        torch.cuda.synchronize()
        t0 = time.time()
        train_proxy_head(cache, pos, cfg, epochs=1)
        torch.cuda.synchronize()
        t1 = time.time() - t0
        print(f"{name:34s} 1 epoch {t1:6.2f}s -> 8 epochs ~{8 * t1:6.1f}s", flush=True)
        del cache
        torch.cuda.empty_cache()
