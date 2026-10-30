"""Step 4.3: does the 8-epoch cached-feature proxy rank heads like full 60-epoch training?

5 fixed configs spanning the space (two share position 16, so within-position ranking is tested too).
Proxy: utils/proxy.py (20k cached train subset, no augmentation, 8 epochs).
Full: 05 recipe (utils/train_heads.py, 60 epochs, augmented 45k train) with all 5 heads trained in one pass of the
frozen backbone (MultiProbeVGG; heads are independent, so this equals 5 separate runs up to data order).
Score = val accuracy of the head alone (argmax, full 5k val); full = best epoch per head.
Spearman rho < 0.6 -> exit code 1 (the chain stops before NAS).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from scipy import stats

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dataset.datamodule import get_cifar10_dataloaders
from models.backbone import build_backbone
from models.head_builder import cfg_key, fc_input_size
from models.searched_exit_model import MultiProbeVGG
from utils.ckpt import load_checkpoint
from utils.flops import head_cost
from utils.proxy import FeatureCache, proxy_val_logits
from utils.runmeta import add_common_args, ckpt_path, output_path, run_meta
from utils.seed import set_seed
from utils.train_heads import train_heads

CONFIGS = [
    (9, {"n_conv": 0, "pooling": 1, "hidden_fc": 0}),
    (16, {"n_conv": 1, "conv_type": "depthwise_separable", "channel_reduction": 0.25, "pooling": 2, "hidden_fc": 64}),
    (16, {"n_conv": 0, "pooling": 4, "hidden_fc": 256}),
    (23, {"n_conv": 2, "conv_type": "standard", "channel_reduction": 0.5, "pooling": 1, "hidden_fc": 128}),
    (33, {"n_conv": 1, "conv_type": "standard", "channel_reduction": 0.125, "pooling": 2, "hidden_fc": 0}),
]
RHO_MIN = 0.6


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = ckpt_path("vgg16_bn_cifar10_base.pth", args.quick)

    cache = FeatureCache(device, args.quick, base_ckpt)
    yv = cache.labels["val"].cpu()
    rows = []
    for pos, cfg in CONFIGS:
        t0 = time.time()
        logits, _ = proxy_val_logits(cache, pos, cfg)
        acc = 100.0 * (logits.argmax(1) == yv).float().mean().item()
        flops, params = head_cost(pos, cfg)
        rows.append({"position": pos, "cfg": json.dumps(cfg), "key": cfg_key(pos, cfg), "head_flops": flops,
                     "head_params": params, "fc_input": fc_input_size(pos, cfg), "proxy_val_acc": acc,
                     "proxy_seconds": round(time.time() - t0, 2)})
        print(f"[proxy] p{pos} {cfg} -> {acc:.2f}% ({rows[-1]['proxy_seconds']}s)", flush=True)

    backbone = build_backbone()
    load_checkpoint(base_ckpt, backbone)
    model = MultiProbeVGG(backbone.features, backbone.avgpool, backbone.classifier,
                          [p for p, _ in CONFIGS], [c for _, c in CONFIGS]).to(device)
    train_loader, val_loader, _ = get_cifar10_dataloaders(batch_size=args.batch_size, seed=args.seed,
                                                          num_workers=args.num_workers, quick=args.quick)
    t0 = time.time()
    history, best = train_heads(model, train_loader, val_loader, device, args.epochs,
                                log=lambda s: print(s, flush=True))
    full_minutes = (time.time() - t0) / 60
    for r, b, last in zip(rows, best, history[-1][:-1]):
        r["full_val_acc_best"], r["full_val_acc_last"] = b, last

    proxy = [r["proxy_val_acc"] for r in rows]
    full = [r["full_val_acc_best"] for r in rows]
    rho, p = stats.spearmanr(proxy, full)
    tau, _ = stats.kendalltau(proxy, full)
    ok = bool(rho >= RHO_MIN)
    df = pd.DataFrame(rows)
    df.insert(0, "seed", args.seed)
    df.to_csv(output_path("proxy_check.csv", args.quick, "nas"), index=False)
    res = run_meta(args, spearman_rho=float(rho), spearman_p=float(p), kendall_tau=float(tau), rho_min=RHO_MIN,
                   ok=ok, full_training_minutes=round(full_minutes, 2),
                   mean_proxy_seconds=float(df.proxy_seconds.mean()), rows=rows)
    with open(output_path("proxy_check.json", args.quick, "nas"), "w") as f:
        json.dump(res, f, indent=2)
    print(df[["position", "cfg", "proxy_val_acc", "full_val_acc_best", "proxy_seconds"]].to_string(index=False))
    print(f"[proxy check] Spearman rho = {rho:.3f} (p={p:.3g}), Kendall tau = {tau:.3f} -> "
          f"{'OK' if ok else 'FAILED (rho < %.1f): NAS must not run on this proxy' % RHO_MIN}", flush=True)
    if not ok and not args.quick:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
