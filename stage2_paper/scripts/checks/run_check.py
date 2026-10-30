"""Post-job checks for the overnight chain. Exit code 0 = all checks passed, 1 = a check failed (chain stops).

  run_check.py j1             test acc >= 90 % and the backbone checkpoint loads strict
  run_check.py j2 --ee NAME   both heads val acc > 60 % and the multi-exit checkpoint loads strict (backbone identical)
  run_check.py j3             clean test flag rate within +-1 pt of alpha (margin_band, alpha=1 %); vectorised == slow loop
  run_check.py j4 --rule R    k=0 sanity OK, all locations x k x runs present, every undo OK, final checksum OK
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.backbone import build_backbone
from models.searched_exit_model import load_searched_model
from utils.ckpt import load_checkpoint
from utils.runmeta import ckpt_path, output_path, results_path

RESULTS = []


def to_count(pct, n):
    """Integer number of correct samples from a stored accuracy percentage over n samples. The percentage must sit
    within 1e-3 samples of an integer count (true for any 100*correct/n, also after float32 rounding), so this
    recovers the exact count and never rounds away a real difference."""
    x = pct * n / 100.0
    c = int(round(x))
    assert abs(x - c) < 1e-3, f"{pct}% of {n} is not an integer count ({x})"
    return c


def check(name, ok, detail):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def j1(args):
    with open(output_path("backbone_results.json", args.quick)) as f:
        r = json.load(f)
    check("backbone test accuracy >= 90%", args.quick or r["test_acc"] >= 90.0,
          f"test {r['test_acc']:.2f}% (val {r['val_acc']:.2f}%, best epoch {r['best_epoch']})")
    meta = load_checkpoint(ckpt_path("vgg16_bn_cifar10_base.pth", args.quick), build_backbone())
    check("backbone checkpoint loads strict", True, f"seed {meta.get('seed')}, epoch {meta.get('epoch')}")


def j2(args):
    model, meta = load_searched_model(ckpt_path("vgg16_bn_cifar10_base.pth", args.quick), ckpt_path(args.ee, args.quick))
    check("multi-exit checkpoint loads strict, backbone identical", True,
          f"positions {meta['positions']} archs {meta['archs']} epoch {meta['epoch']}")
    accs = meta["head_val_accs"]
    check("every head val acc > 60%", args.quick or all(a > 60.0 for a in accs), f"head val accs {accs}")


def j3(args):
    c = pd.read_csv(output_path("clean_eval.csv", args.quick, args.tag))
    row = c[(c.rule == "margin_band") & (c.alpha == 1.0)].iloc[0]
    check("clean test flag rate within +-1 pt of alpha=1% (margin_band)", args.quick or abs(row.flag_rate - 1.0) <= 1.0,
          f"flag rate {row.flag_rate:.2f}%, accuracy {row.accuracy:.2f}%")
    with open(output_path("reference_check.json", args.quick, args.tag)) as f:
        ref = json.load(f)
    check("vectorised routing == slow per-sample loop (all rules, all alphas)", ref["all_identical"],
          json.dumps(ref["checks"]))


def j4(args):
    with open(results_path(f"k0_sanity_{args.rule}.json", args.quick, args.tag)) as f:
        k0 = json.load(f)
    check("k=0 sanity (SDC=0, no flags, deterministic)", k0["ok"],
          f"SDC {k0['n_sdc']}, flagged {k0['n_flagged']}, deterministic {k0['deterministic']}")
    with open(results_path(f"campaign_{args.rule}_status.json", args.quick, args.tag)) as f:
        st = json.load(f)
    check("weights checksum identical at start and end", st["checksum_ok"], st["checksum_end"][:16])
    expected = st["kmax"] * st["runs"]
    for loc in st["locations"]:
        df = pd.read_csv(results_path(f"faults_{args.rule}_{loc}.csv", args.quick, args.tag))
        n_unique = df[["k", "run"]].drop_duplicates().shape[0]
        check(f"{loc}: all runs present", n_unique == expected and len(df) == expected, f"{len(df)}/{expected} rows")
        check(f"{loc}: every undo bit-identical", bool(df.undo_ok.all()), f"{int((~df.undo_ok).sum())} failures")


def j0(args):
    with open(results_path("analysis_status.json", args.quick, args.tag)) as f:
        st = json.load(f)
    check("re-applied faults reproduce the campaign's SDC/flag counts", st["margin_reproduction_mismatches"] == 0,
          f"{st['margin_reproduction_mismatches']} mismatches over {st['runs']} runs")
    check("comparison-rule counts reproduced on runs with identical faults",
          st["energy_reproduction_mismatches_on_identical_fault_runs"] == 0,
          f"{st['runs_with_identical_faults_in_both_campaigns']} identical-fault runs")
    check("no non-finite replays, undo OK", st["nonfinite_replays"] == 0 and st["undo_ok"], str(st["undo_ok"]))


def cache(args):
    from config import CACHE_DIR
    d = (CACHE_DIR / "quick") if args.quick else CACHE_DIR
    with open(d / "meta.json") as f:
        m = json.load(f)
    n_val = m["splits"]["val"]["n"]
    n_cache = m["splits"]["val"].get("n_correct", None)
    n_cache = to_count(m["splits"]["val"]["backbone_acc"], n_val) if n_cache is None else n_cache
    n_ref = to_count(m["backbone_val_acc"], n_val)
    check("cached val logits reproduce the backbone val correct count", args.quick or n_cache == n_ref,
          f"{n_cache} vs {n_ref} of {n_val}")
    missing = [f"{s}_p{p}" for s in ("train", "val") for p in m["positions"] if not (d / f"{s}_p{p}.npy").is_file()]
    check("all cached position files present", not missing, f"{len(m['positions'])} positions, on-the-fly "
          f"{m['on_the_fly']}, {m['total_gb']:.2f} GB")


def proxy(args):
    with open(output_path("proxy_check.json", args.quick, "nas")) as f:
        r = json.load(f)
    check(f"proxy Spearman rho >= {r['rho_min']}", args.quick or r["ok"], f"rho = {r['spearman_rho']:.3f}")


def nas(args):
    name = args.run_name or args.sampler
    df = pd.read_csv(output_path(f"trials_{name}.csv", args.quick, "nas"))
    with open(output_path(f"meta_{name}.json", args.quick, "nas")) as f:
        m = json.load(f)
    expected = 2 if args.sampler == "handplaced" else m["trials"]
    check(f"{args.sampler}: all trials logged", len(df) == expected, f"{len(df)}/{expected}")
    check(f"{args.sampler}: objectives finite", bool(df[["acc_sel", "avg_flops", "head_params"]].notna().all().all()),
          f"feasible {int(df.feasible.sum())}")


def nas_model(args):
    with open(output_path("chosen_config.json", args.quick, "nas")) as f:
        cfg = json.load(f)
    model, meta = load_searched_model(ckpt_path("vgg16_bn_cifar10_base.pth", args.quick), ckpt_path(args.ee, args.quick))
    check("NAS checkpoint loads strict, backbone identical, config matches",
          meta["positions"] == cfg["positions"] and meta["archs"] == cfg["archs"], f"{meta['positions']} {cfg['rule']}")
    with open(output_path("clean_eval.json", args.quick, "nas")) as f:
        ce = json.load(f)
    with open(output_path("backbone_results.json", args.quick)) as f:
        bb = json.load(f)
    n_test = ce["rows"][0]["n"]
    n_nas = ce["meta"].get("backbone_test_correct", None)
    n_nas = to_count(ce["meta"]["backbone_test_acc"], n_test) if n_nas is None else n_nas
    n_bb = bb.get("test_correct", None)
    n_bb = to_count(bb["test_acc"], n_test) if n_bb is None else n_bb
    check("final classifier correct count == backbone test correct count (frozen backbone)", n_nas == n_bb,
          f"{n_nas} vs {n_bb} of {n_test}")
    c = pd.read_csv(output_path("clean_eval.csv", args.quick, "nas"))
    row = c[(c.rule == cfg["rule"]) & (c.alpha == 1.0)].iloc[0]
    check(f"clean test flag rate within +-1 pt of alpha=1% ({cfg['rule']})", args.quick or abs(row.flag_rate - 1.0) <= 1.0,
          f"flag {row.flag_rate:.2f}%, accuracy {row.accuracy:.2f}%, saved {row.saved_pct:.2f}%")
    with open(output_path("reference_check.json", args.quick, "nas")) as f:
        ref = json.load(f)
    check("vectorised routing == slow loop (NAS model)", ref["all_identical"], "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["j0", "j1", "j2", "j3", "j4", "cache", "proxy", "nas", "nas_model"])
    parser.add_argument("--sampler", default="tpe")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ee", default="ee_old16_23.pth")
    parser.add_argument("--rule", default="margin_band")
    parser.add_argument("--quick", action="store_true", help="check the --quick outputs (accuracy thresholds skipped)")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    try:
        {"j0": j0, "j1": j1, "j2": j2, "j3": j3, "j4": j4, "cache": cache, "proxy": proxy, "nas": nas,
         "nas_model": nas_model}[args.job](args)
    except Exception as e:  # missing file, bad checkpoint, ... -> failed check
        check(f"{args.job} checks ran", False, f"{type(e).__name__}: {e}")
    failed = [n for n, ok in RESULTS if not ok]
    print(f"[{args.job}] {'ALL CHECKS PASSED' if not failed else 'FAILED: ' + '; '.join(failed)}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
