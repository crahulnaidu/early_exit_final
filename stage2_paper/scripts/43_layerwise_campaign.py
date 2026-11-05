"""Layer-wise single-bit fault campaign for VGG16-BN/CIFAR-10.

Reuses the repository's checkpoint loader, data split, SubsetEvaluator, bit-exact checks and output guard.
Each accepted realization flips exactly one FP32 weight bit in exactly one Conv2d/Linear module.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BACKBONE_CKPT
from dataset.datamodule import get_cifar10_dataloaders
from faults.evaluator import SubsetEvaluator
from models.searched_exit_model import load_searched_model
from utils.paths import safe_path
from utils.seed import set_seed

OUT_ROOT = Path("/home1/jtt/early_exit_final/stage2_paper/results/layerwise")
EXPONENT_BITS = tuple(range(23, 31))
SIGN_BIT = 31
MAX_ATTEMPTS = 1000


def load_split(split, quick):
    _, val, test = get_cifar10_dataloaders(batch_size=500, seed=42, num_workers=4, quick=quick)
    loader = val if split == "val" else test
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)


def stack(logits):
    return torch.stack(logits, dim=1).float().cpu()


def checksum(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.ndim and (name.endswith(".weight")):
            h.update(name.encode())
            h.update(p.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def discover_layers(model):
    """Return stable qualified names and their modules for all Conv2d/Linear weights."""
    result = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight.dtype != torch.float32:
                raise TypeError(f"{name}.weight must be FP32, got {module.weight.dtype}")
            result.append((name, module))
    if not result:
        raise RuntimeError("No Conv2d/Linear modules discovered")
    return result


def _capture_inputs(model, images, module_name, batch_size, device):
    """Capture one target module's inputs on CPU; avoid storing every layer's activations simultaneously."""
    target = dict(model.named_modules())[module_name]
    captured = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach().cpu())

    handle = target.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                model(images[start:start + batch_size].to(device))
    finally:
        handle.remove()
    if sum(x.shape[0] for x in captured) != len(images):
        raise RuntimeError(f"Input hook captured wrong number of samples for {module_name}")
    return captured


def _run_feature_suffix(model, start_idx, x, batch_size):
    """Run features[start_idx:] on a cached input, returning logits for all exits and final classifier."""
    positions = list(model.positions)
    first_head = next((j for j, p in enumerate(positions) if p >= start_idx), len(positions))
    outputs = [[] for _ in range(len(positions) - first_head + 1)]
    with torch.no_grad():
        for begin in range(0, len(x), batch_size):
            z = x[begin:begin + batch_size].to(next(model.parameters()).device)
            head_j = first_head
            slot = 0
            for idx in range(start_idx, len(model.features)):
                z = model.features[idx](z)
                if head_j < len(positions) and idx == positions[head_j]:
                    outputs[slot].append(model.exit_heads[head_j](z).detach())
                    slot += 1
                    head_j += 1
            outputs[slot].append(model.classifier(torch.flatten(model.avgpool(z), 1)).detach())
    return [torch.cat(parts, dim=0) for parts in outputs]


def logits_for_layer(model, evaluator, layer_name, captured_inputs, images, batch_size):
    """Recompute only the path affected by the target weight fault."""
    clean = evaluator.clean_logits
    if layer_name.startswith("features."):
        parts = layer_name.split(".")
        feature_idx = int(parts[1])
        prefix_count = sum(p < feature_idx for p in model.positions)
        suffix = _run_feature_suffix(model, feature_idx, torch.cat(captured_inputs, dim=0), batch_size)
        return list(clean[:prefix_count]) + suffix

    if layer_name.startswith("exit_heads."):
        head_idx = int(layer_name.split(".")[1])
        changed = model.exit_heads[head_idx]
        position = model.positions[head_idx]
        logits = list(clean)
        chunks = []
        feats = evaluator.clean_feats[position]
        with torch.no_grad():
            for start in range(0, feats.shape[0], batch_size):
                chunks.append(changed(feats[start:start + batch_size]).detach())
        logits[head_idx] = torch.cat(chunks, dim=0)
        return logits

    if layer_name == "classifier" or layer_name.startswith("classifier."):
        # The parent classifier's pre-hook caches its input for this module name.
        # The caller stores that tensor under the synthetic key "classifier".
        flat = torch.cat(captured_inputs, dim=0).to(next(model.parameters()).device)
        chunks = []
        with torch.no_grad():
            for start in range(0, flat.shape[0], batch_size):
                chunks.append(model.classifier(flat[start:start + batch_size]).detach())
        logits = list(clean)
        logits[-1] = torch.cat(chunks, dim=0)
        return logits

    raise ValueError(f"Unsupported module path: {layer_name}")


def flip_one(module, index, bit):
    """Flip one FP32 bit and return the old/new float values."""
    flat = module.weight.data.view(-1)
    ints = flat.view(torch.int32)
    old = float(flat[index].item())
    ints[index] ^= (1 << int(bit))
    new = float(flat[index].item())
    return old, new


def unflip_one(module, index, bit):
    module.weight.data.view(-1).view(torch.int32)[index] ^= (1 << int(bit))


def all_finite(logits):
    return all(bool(torch.isfinite(z).all()) for z in logits)


def write_rows(path, rows):
    p = safe_path(path)
    if not rows:
        return
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ee-ckpt", required=True, help="Trained multi-exit checkpoint, e.g. checkpoints/ee_searched.pth")
    ap.add_argument("--tag", required=True, help="Unique output tag, e.g. nas_s11_t229")
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--layers", nargs="*", default=None, help="Optional exact qualified module names; default=all")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs must be >= 1")
    if args.quick:
        args.runs = min(args.runs, 2)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = BACKBONE_CKPT
    if args.quick:
        base_ckpt = Path(str(BACKBONE_CKPT).replace("/checkpoints/", "/checkpoints/quick/"))
    model, meta = load_searched_model(base_ckpt, args.ee_ckpt)
    model = model.to(device).eval()

    images, labels = load_split(args.split, args.quick)
    images = images.to(device)
    labels = labels.cpu()
    evaluator = SubsetEvaluator(model, images, batch_size=args.batch_size)
    layers = discover_layers(model)
    if args.layers:
        wanted = set(args.layers)
        layers = [(n, m) for n, m in layers if n in wanted]
        missing = wanted - {n for n, _ in layers}
        if missing:
            raise ValueError(f"Unknown/non-faultable layer names: {sorted(missing)}")

    out = OUT_ROOT / ("quick" if args.quick else "") / args.tag / args.split
    clean_path = out / "clean_logits.pt"
    torch.save({"logits": stack(evaluator.clean_logits), "labels": labels,
                "positions": meta["positions"], "archs": meta["archs"],
                "ee_ckpt": str(args.ee_ckpt), "split": args.split,
                "n_samples": len(labels)}, safe_path(clean_path))

    start_hash = checksum(model)
    layer_summary = []
    t0 = time.time()
    for layer_no, (layer_name, module) in enumerate(layers):
        # Capture the target module input once. Classifier linear layers are evaluated by rerunning the
        # parent classifier from its input, so cache the parent classifier input in that case.
        capture_name = "classifier" if layer_name.startswith("classifier.") else layer_name
        captured_inputs = _capture_inputs(model, images, capture_name, args.batch_size, device)
        if layer_name.startswith("classifier."):
            # Hooking the classifier parent captures its flattened input; one capture per batch.
            pass
        for mechanism in ("exponent", "sign"):
            stem = out / (mechanism + "__" + layer_name.replace(".", "__"))
            npy_path = Path(str(stem) + ".npy")
            csv_path = Path(str(stem) + ".csv")
            if npy_path.is_file() and csv_path.is_file():
                continue
            runs_logits, rows = [], []
            rng = np.random.default_rng(args.seed + 100_003 * (layer_no + 1) +
                                        (0 if mechanism == "exponent" else 50_000_000) +
                                        (50_000_000 if args.split == "val" else 0))
            for run in range(1, args.runs + 1):
                used, rejected = set(), []
                attempt = 0
                accepted = None
                while accepted is None:
                    attempt += 1
                    if attempt > MAX_ATTEMPTS:
                        raise RuntimeError(f"{layer_name} {mechanism} run {run}: exceeded {MAX_ATTEMPTS} attempts")
                    index = int(rng.integers(0, module.weight.numel()))
                    if index in used:
                        continue
                    bit = int(rng.choice(EXPONENT_BITS)) if mechanism == "exponent" else SIGN_BIT
                    old, new = flip_one(module, index, bit)
                    reason = "weight_nonfinite" if not np.isfinite(new) else None
                    logits = None
                    if reason is None:
                        logits = logits_for_layer(model, evaluator, layer_name, captured_inputs,
                                                  images, args.batch_size)
                        if not all_finite(logits):
                            reason = "logits_nonfinite"
                    if reason:
                        unflip_one(module, index, bit)
                        rejected.append({"index": index, "bit": bit, "reason": reason})
                        continue
                    accepted = {"index": index, "bit": bit, "old": old, "new": new}
                    used.add(index)
                runs_logits.append(stack(logits).numpy())
                unflip_one(module, accepted["index"], accepted["bit"])
                row = {"layer": layer_name, "mechanism": mechanism, "run": run,
                       "seed": args.seed, "index": accepted["index"], "bit": accepted["bit"],
                       "old": accepted["old"], "new": accepted["new"],
                       "n_attempts": attempt, "n_rejected": len(rejected),
                       "rejected": json.dumps(rejected), "undo_ok": True}
                rows.append(row)
                if run % 10 == 0:
                    print(f"[{args.split} {layer_name} {mechanism}] {run}/{args.runs}", flush=True)
            # Save only after the whole layer/mechanism finishes; partial files are never mistaken for complete.
            safe_path(npy_path).write_bytes(b"")
            np.save(npy_path, np.stack(runs_logits).astype(np.float32))
            write_rows(csv_path, rows)
            layer_summary.append({"layer": layer_name, "mechanism": mechanism, "runs": len(rows),
                                  "npy": str(npy_path), "csv": str(csv_path)})
            print(f"[done {args.split} {layer_name} {mechanism}] {len(rows)} realizations "
                  f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

    end_hash = checksum(model)
    if start_hash != end_hash:
        raise RuntimeError(f"Model weights changed after campaign: {start_hash} != {end_hash}")
    status = {"tag": args.tag, "split": args.split, "n_samples": len(labels),
              "runs_per_layer_mechanism": args.runs, "mechanisms": ["exponent", "sign"],
              "bit_policy": {"exponent": "uniform bit in [23,30]", "sign": 31},
              "layer_names": [n for n, _ in layers], "start_checksum": start_hash,
              "end_checksum": end_hash, "checksum_ok": True, "outputs": layer_summary,
              "positions": meta["positions"], "archs": meta["archs"]}
    with open(safe_path(out / "status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print(json.dumps({"status": "complete", "layers": len(layers), "outputs": len(layer_summary),
                      "checksum_ok": True}, indent=2))


if __name__ == "__main__":
    main()
