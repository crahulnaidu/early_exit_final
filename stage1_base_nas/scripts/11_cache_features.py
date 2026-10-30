"""Step 4: cache frozen-backbone features for the NAS proxy (fp16, eval transforms, no augmentation).

Splits: a fixed seeded 20k subset of the 45k TRAIN indices (outputs/cache_train_subset.json) and the full 5k VAL
(val order). Also writes the VAL_CAL / VAL_SEL halves (outputs/val_cal_sel.json).
Positions: all 17 candidates if more than --min-free-gb is free on the work-folder disk, else positions 2 and 5 are
skipped (utils/proxy.py then computes them on the fly from the cached images).
Per split: images (fp16), final-classifier logits (fp32, KD teacher), labels, and one [N, C, H, W] fp16 array per
cached position, as .npy memmaps under cache/. Test images are never cached.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import CACHE_DIR, WORK_DIR
from dataset.datamodule import get_eval_loader
from models.backbone import FEATURE_SHAPES, build_backbone
from models.multi_exit import CANDIDATE_POSITIONS
from utils.ckpt import load_checkpoint
from utils.paths import safe_path
from utils.runmeta import add_common_args, ckpt_path, output_path, run_meta
from utils.seed import set_seed
from utils.splits import create_subsplits, load_splits

SKIP_IF_LOW_DISK = (2, 5)


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    cache_sub, val_cal_sel = create_subsplits()
    free_gb = shutil.disk_usage(WORK_DIR).free / 1e9
    positions = list(CANDIDATE_POSITIONS) if free_gb > args.min_free_gb else \
        [p for p in CANDIDATE_POSITIONS if p not in SKIP_IF_LOW_DISK]
    skipped = [p for p in CANDIDATE_POSITIONS if p not in positions]
    print(f"[disk] {free_gb:.1f} GB free (threshold {args.min_free_gb} GB) -> caching {len(positions)} positions, "
          f"on-the-fly: {skipped or 'none'}", flush=True)

    model = build_backbone()
    base_ckpt = ckpt_path("vgg16_bn_cifar10_base.pth", args.quick)
    bmeta = load_checkpoint(base_ckpt, model)
    model = model.to(device).eval()

    out_dir = (CACHE_DIR / "quick") if args.quick else CACHE_DIR
    splits = {"train": cache_sub["train_idx"], "val": load_splits()["val_idx"]}
    if args.quick:
        splits = {k: v[:256] for k, v in splits.items()}
    info = {}
    for split, idx in splits.items():
        n = len(idx)
        arrays = {
            "images": np.lib.format.open_memmap(str(safe_path(out_dir / f"{split}_images.npy")), "w+",
                                                np.float16, (n, 3, 32, 32)),
            "final_logits": np.lib.format.open_memmap(str(safe_path(out_dir / f"{split}_final_logits.npy")), "w+",
                                                      np.float32, (n, 10)),
            "labels": np.lib.format.open_memmap(str(safe_path(out_dir / f"{split}_labels.npy")), "w+", np.int64, (n,)),
        }
        for p in positions:
            arrays[p] = np.lib.format.open_memmap(str(safe_path(out_dir / f"{split}_p{p}.npy")), "w+", np.float16,
                                                  (n, *FEATURE_SHAPES[p]))
        a = 0
        loader = get_eval_loader("train", idx, batch_size=500, num_workers=args.num_workers, seed=args.seed)
        with torch.no_grad():
            for x, y in loader:
                b = a + x.shape[0]
                x = x.to(device)
                arrays["images"][a:b] = x.half().cpu().numpy()
                arrays["labels"][a:b] = y.numpy()
                h = x
                for i, layer in enumerate(model.features):
                    h = layer(h)
                    if i in arrays:
                        f16 = h.half()
                        assert torch.isfinite(f16).all(), f"fp16 overflow at position {i}"
                        arrays[i][a:b] = f16.cpu().numpy()
                arrays["final_logits"][a:b] = model.classifier(torch.flatten(model.avgpool(h), 1)).cpu().numpy()
                a = b
        for arr in arrays.values():
            arr.flush()
        n_correct = int((arrays["final_logits"].argmax(1) == arrays["labels"]).sum())
        acc = 100.0 * n_correct / n
        size = sum(Path(arr.filename).stat().st_size for arr in arrays.values()) / 1e9
        info[split] = {"n": n, "n_correct": n_correct, "backbone_acc": acc, "size_gb": size}
        print(f"[{split}] n={n} | backbone accuracy on cached logits {acc:.2f}% | {size:.2f} GB", flush=True)

    with open(output_path("backbone_results.json", args.quick)) as f:
        ref_val = json.load(f)["val_acc"]
    if not args.quick:   # integer counts, exact (backbone_results.json val_acc = 100 * correct / 5000)
        assert info["val"]["n_correct"] == round(ref_val * info["val"]["n"] / 100), (info["val"]["n_correct"], ref_val)
    meta = run_meta(args, positions=positions, on_the_fly=skipped, free_gb_before=free_gb, splits=info,
                    cache_subset_sha256=cache_sub["sha256"], val_cal_sel_sha256=val_cal_sel["sha256"],
                    backbone_epoch=bmeta.get("epoch"), backbone_val_acc=ref_val,
                    total_gb=sum(v["size_gb"] for v in info.values()), minutes=round((time.time() - t0) / 60, 2))
    with open(safe_path(out_dir / "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] cache total {meta['total_gb']:.2f} GB in {meta['minutes']} min | "
          f"free now {shutil.disk_usage(WORK_DIR).free / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
