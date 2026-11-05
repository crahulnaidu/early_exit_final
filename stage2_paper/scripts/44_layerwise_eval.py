"""Offline layer-wise sensitivity evaluation using the repository's existing exit rules.

Each test fault is evaluated with the same stored logits for all decision policies. Thresholds use clean VAL only.
MENDNet faulty-VAL calibration is optional because its grid search is expensive and uses additional fault knowledge.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.rules2 import RULES, calibrate, calibrate_mendnet_clean, calibrate_mendnet_faulty, route
from utils.paths import safe_path


def as_logits(array, device):
    """[N, K+1, C] -> list of [N,C] tensors."""
    return [torch.as_tensor(array[:, j, :], device=device, dtype=torch.float32)
            for j in range(array.shape[1])]


def read_clean(path, device):
    d = torch.load(path, map_location="cpu")
    logits = as_logits(d["logits"].numpy(), device)
    return logits, d["labels"].to(device), d


def route_metrics(cal, alpha, logits, labels, clean_correct, clean_acc, device):
    exit_id, pred, mitig = route(cal, logits, alpha)
    flagged = exit_id == -1
    sdc = (~flagged) & (pred != labels) & clean_correct
    flags_on_clean_correct = flagged & clean_correct
    denom = int(sdc.sum() + flags_on_clean_correct.sum())
    n_clean = max(int(clean_correct.sum()), 1)
    correct = (pred == labels) & (~flagged)
    return {
        "clean_accuracy_pct": clean_acc,
        "fault_accuracy_pct": 100.0 * float(correct.float().mean()),
        "mitigated_accuracy_pct": 100.0 * float(mitig.eq(labels).float().mean()),
        "flag_rate_pct": 100.0 * float(flagged.float().mean()),
        "coverage_pct": 100.0 * float((~flagged).float().mean()),
        "sdc_rate_on_clean_correct_pct": 100.0 * float(sdc.sum()) / n_clean,
        "detection_ratio_on_clean_correct_pct": 100.0 * float(flags_on_clean_correct.sum()) / denom if denom else float("nan"),
        "n_clean_correct": int(clean_correct.sum()),
        "n_sdc": int(sdc.sum()),
        "n_flags_on_clean_correct": int(flags_on_clean_correct.sum()),
        **{f"exit_{j}_pct": 100.0 * float((exit_id == j).float().mean()) for j in range(len(logits))}
    }


def calibrate_methods(val_logits, yv):
    methods = {}
    methods["plain_dnn"] = calibrate("plain_dnn", val_logits, yv)
    methods["sdn_msp"] = calibrate("sdn_msp", val_logits, yv, alphas=(1.0,))
    methods["margin_band"] = calibrate("margin_band", val_logits, yv, alphas=(1.0,))
    methods["mendnet_clean"] = calibrate_mendnet_clean(val_logits, yv, alpha=1.0)
    return methods


def fault_metadata(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No run rows in {csv_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--include-faulty-val", action="store_true",
                    help="Also tune MENDNet gamma/tau on the matching layer/type validation fault set (expensive; extra fault knowledge).")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path("/home1/jtt/early_exit_final/stage2_paper/results/layerwise")
    if args.quick:
        base = base / "quick"
    root = base / args.tag
    val_logits, yv, val_meta = read_clean(root / "val" / "clean_logits.pt", device)
    test_logits_clean, yt, test_meta = read_clean(root / "test" / "clean_logits.pt", device)
    if len(yv) == 0 or len(yt) == 0:
        raise RuntimeError("Empty validation/test split")
    methods = calibrate_methods(val_logits, yv)
    clean_correct_by_method = {}
    clean_accuracy_by_method = {}
    for name, cal in methods.items():
        # SDC denominator must use the SAME test images under the clean model, not the validation split.
        e_test, p_test, _ = route(cal, test_logits_clean, args.alpha)
        clean_correct_by_method[name] = (e_test >= 0) & p_test.eq(yt)
        clean_accuracy_by_method[name] = 100.0 * float(clean_correct_by_method[name].float().mean())

    val_files = {p.name: p for p in (root / "val").glob("*.npy")}
    test_files = sorted((root / "test").glob("*.npy"))
    rows = []
    for test_path in test_files:
        val_path = val_files.get(test_path.name)
        if val_path is None:
            raise FileNotFoundError(f"Matching validation fault file missing for {test_path.name}")
        test_runs = np.load(test_path, mmap_mode="r")
        val_runs = np.load(val_path, mmap_mode="r")
        meta_rows = fault_metadata(test_path.with_suffix(".csv"))
        if len(meta_rows) != test_runs.shape[0] or len(val_runs) != test_runs.shape[0]:
            raise RuntimeError(f"Run count mismatch for {test_path.name}")
        first = meta_rows[0]
        layer, mechanism = first["layer"], first["mechanism"]
        local_methods = dict(methods)
        if args.include_faulty_val:
            val_fault_logits = []
            for r in range(val_runs.shape[0]):
                val_fault_logits.append(as_logits(np.asarray(val_runs[r]), device))
            local_methods["mendnet_faulty_val"] = calibrate_mendnet_faulty(val_fault_logits, yv)

        print(f"[eval] {layer} | {mechanism} | {test_runs.shape[0]} realizations", flush=True)
        for r in range(test_runs.shape[0]):
            fault_logits = as_logits(np.asarray(test_runs[r]), device)
            for name, cal in local_methods.items():
                e, p, _ = route(cal, fault_logits, args.alpha)
                S = clean_correct_by_method.get(name)
                if S is None:
                    e0, p0, _ = route(cal, test_logits_clean, args.alpha)
                    S = (e0 >= 0) & p0.eq(yt)
                metric = route_metrics(cal, args.alpha, fault_logits, yt, S,
                                       clean_accuracy_by_method.get(name, float("nan")), device)
                rows.append({"layer": layer, "mechanism": mechanism, "run": r + 1,
                             "method": name, "comparison_mode": "same_architecture_policy_comparison",
                             "bit": int(first["bit"]) if r == 0 else int(meta_rows[r]["bit"]),
                             **metric})
    result = pd.DataFrame(rows)
    out_dir = root / "evaluation"
    result.to_csv(safe_path(out_dir / "layerwise_runs.csv.gz"), index=False, compression="gzip")
    summary = (result.groupby(["layer", "mechanism", "method"], as_index=False)
               .agg(n_runs=("run", "size"),
                    fault_accuracy_pct=("fault_accuracy_pct", "mean"),
                    mitigated_accuracy_pct=("mitigated_accuracy_pct", "mean"),
                    flag_rate_pct=("flag_rate_pct", "mean"),
                    coverage_pct=("coverage_pct", "mean"),
                    sdc_rate_on_clean_correct_pct=("sdc_rate_on_clean_correct_pct", "mean"),
                    detection_ratio_on_clean_correct_pct=("detection_ratio_on_clean_correct_pct", "mean")))
    summary.to_csv(safe_path(out_dir / "layerwise_summary.csv"), index=False)
    status = {"tag": args.tag, "alpha": args.alpha, "n_layer_files": len(test_files),
              "methods": list(methods), "include_faulty_val": args.include_faulty_val,
              "n_rows": len(result),
              "note": "All methods consume the same faulted logits and share the same checkpoint/exit heads. This isolates routing-policy differences; it is not a native-architecture SDN/MENDNet reproduction."}
    with open(safe_path(out_dir / "status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
