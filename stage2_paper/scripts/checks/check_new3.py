"""Post-job checks for the NAS variant study (exit 1 on failure so the afterok chain stops).

  check_new3.py phase0
  check_new3.py variant --variant B1_menu --seeds 1 11 21 [--trials 250]
  check_new3.py probe            (probe validation finished, finite correlation)
  check_new3.py fidelity         (proxy fidelity finished)
  check_new3.py summary          (T1 table and picks written)
  check_new3.py campaign --tag nas --fm exponent --split test --runs 100 [--kmax 8]
  check_new3.py reproduce --tag nas   (source-campaign reproduction within the pre-registered tolerance, P5)
  check_new3.py ckpt --ckpt <path>    (trained heads checkpoint exists and loads)
  check_new3.py final                 (final_choice.json written, its checkpoint exists)
  check_new3.py offline --tags ...    (41 outputs for every tag)
  check_new3.py latency               (43 outputs; early-stop predictions agree with offline routing)
  check_new3.py paper                 (42 outputs)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.runmeta import output_path, results_path

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def phase0(a):
    d = Path("/home1/jtt/early_exit_final/stage2_paper/results/phase0")
    for f in ("0a_hv_seeds.csv", "0b_paired_tests.csv", "0c_heads_per_head.csv", "0c_heads_per_bit.csv",
              "0d_bit_summary.csv", "phase0_tables.md"):
        check(f"{f} written", (d / f).is_file())
    runs = pd.read_csv(d / "0d_bit_runs.csv")
    bad = int(((runs.n_sdc != runs.recorded_n_sdc) | (runs.n_flagged != runs.recorded_n_flagged)).sum())
    check("0d replay reproduces the campaign's k=1 G1 counts", bad == 0, f"{bad} mismatches over {len(runs)} runs")


def variant(a):
    for s in a.seeds:
        p = output_path(f"trials_{a.variant}_s{s}.csv", a.quick, "nas_variants")
        m = output_path(f"meta_{a.variant}_s{s}.json", a.quick, "nas_variants")
        n = len(pd.read_csv(p)) if p.is_file() else 0
        expected = 4 if a.quick else a.trials
        check(f"{a.variant} seed {s}: {expected} trials + meta", n == expected and m.is_file(), f"{n} rows")


def probe(a):
    with open(output_path("probe_validation.json", a.quick, "nas_variants")) as f:
        r = json.load(f)
    check("probe validation finished", a.quick or (r["n_configs"] >= 3 and math.isfinite(r["spearman_rho"])),
          json.dumps(r))


def fidelity(a):
    with open(output_path("proxy_fidelity.json", a.quick, "nas_variants")) as f:
        r = json.load(f)
    check("proxy fidelity finished", r["n_configs"] >= (4 if a.quick else 20), json.dumps(r["correlation_vs_full"]))


def summary(a):
    check("T1 written", results_path("T1_search_variants.csv", a.quick, "nas_variants").is_file())
    p = output_path("picks.json", a.quick, "nas_variants")
    check("picks written", p.is_file(), ", ".join(json.load(open(p)).keys()) if p.is_file() else "")


CAMP = Path("/home1/jtt/early_exit_final/stage2_paper/results/campaigns")
OFF = Path("/home1/jtt/early_exit_final/stage2_paper/results/offline")


def campaign(a):
    import numpy as np
    d = CAMP / ("quick" if a.quick else "") / a.tag
    st = json.load(open(d / f"status_{a.fm}_{a.split}.json"))
    check(f"{a.tag} {a.fm} {a.split}: checksum restored", st["checksum_ok"], json.dumps(st))
    files = sorted(d.glob(f"{a.fm}_{a.split}_*.npy"))
    expected = 12 if a.fm == "mendnet" else 4 * (2 if a.quick else a.kmax)
    check(f"{a.tag} {a.fm} {a.split}: {expected} fault files", len(files) == expected, f"{len(files)} files")
    bad_shape, bad_undo, nonfin = 0, 0, 0
    for f in files:
        arr = np.load(f, mmap_mode="r")
        bad_shape += int(arr.shape[0] != (2 if a.quick else a.runs))
        c = pd.read_csv(str(f)[:-4] + ".csv")
        bad_undo += int((c.undo_ok.astype(str) != "True").sum())
        if a.fm != "mendnet":
            nonfin += int(not np.isfinite(arr[-1]).all())
    check(f"{a.tag} {a.fm} {a.split}: run count per file", bad_shape == 0, f"{bad_shape} files with wrong run count")
    check(f"{a.tag} {a.fm} {a.split}: every run undone bit-exactly", bad_undo == 0, f"{bad_undo} failed undos")
    if a.fm != "mendnet":
        check(f"{a.tag} {a.fm} {a.split}: stored logits finite (rejection worked)", nonfin == 0, f"{nonfin} files")
    for s in ("val", "test"):
        f = d / f"clean_{s}{'_int8' if a.fm == 'int8' else ''}.pt"
        check(f"{f.name} written", f.is_file())


def reproduce(a):
    r = json.load(open(OFF / a.tag / "reproduction.json"))
    ident = r["identical"] / max(r["runs"], 1)
    exact = r["exact"] / max(r["identical"], 1)
    diff = (r["abs_diff_sdc"] + r["abs_diff_flag"]) / max(r["images"], 1)
    check(f"{a.tag}: clean-correct subset equals the source", r["n_subset_ours"] == r["n_subset_source"],
          f"{r['n_subset_ours']} vs {r['n_subset_source']}")
    check(f"{a.tag}: >= 95 % of runs have the source's faults", ident >= 0.95, f"{100 * ident:.2f}%")
    check(f"{a.tag}: >= 99 % of identical-fault runs reproduce SDC and flag counts exactly", exact >= 0.99,
          f"{100 * exact:.2f}%")
    check(f"{a.tag}: total count difference <= 0.01 % of evaluated images", diff <= 1e-4, f"{100 * diff:.5f}%")


def ckpt(a):
    import torch
    m = torch.load(a.ckpt, map_location="cpu")
    check(f"{a.ckpt} loads with meta", "meta" in m and "positions" in m["meta"], str(m.get("meta", {}).get("positions")))


def final(a):
    c = json.load(open(Path("/home1/jtt/early_exit_final/stage2_paper/outputs/final") /
                       ("final_choice_quick.json" if a.quick else "final_choice.json")))
    check("final choice written and checkpoint exists", Path(c["ckpt"]).is_file(), f"{c['label']} -> {c['ckpt']}")


def offline(a):
    for t in a.tags:
        d = OFF / ("quick" if a.quick else "") / t
        st = json.load(open(d / "status.json"))
        check(f"{t}: exponent evaluated", "exponent" in st and st["exponent"]["rows"] > 0, json.dumps(st)[:300])
        for fm in [k for k in st if k in ("exponent", "int8", "mendnet")]:
            s = pd.read_csv(d / f"summary_{fm}.csv")
            ok = s[s.location == "ALL"].sdc_rate.between(0, 100).all()
            check(f"{t} {fm}: summary rates within [0, 100]", ok)


def latency(a):
    p = Path("/home1/jtt/early_exit_final/stage2_paper/results/latency") / ("quick" if a.quick else "") / "latency.csv"
    d = pd.read_csv(p)
    agree = d[d["mode"] == "ee_stop"].pred_agreement_with_offline_pct
    check("latency rows written", len(d) > 0, f"{len(d)} rows")
    check("batch-1 early-stop predictions agree with offline routing (>= 99.5 %)", (agree >= 99.5).all(),
          ", ".join(f"{x:.2f}" for x in agree))


def paper(a):
    p = Path("/home1/jtt/early_exit_final/stage2_paper/results/paper") / ("quick" if a.quick else "") / "paper_tables.md"
    check("paper tables written", p.is_file() and "## T2." in p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["phase0", "variant", "probe", "fidelity", "summary", "campaign", "reproduce", "ckpt",
                                    "final", "offline", "latency", "paper"])
    ap.add_argument("--tag")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--fm", default="exponent")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--ckpt")
    ap.add_argument("--variant")
    ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    try:
        {"phase0": phase0, "variant": variant, "probe": probe, "fidelity": fidelity, "summary": summary,
         "campaign": campaign, "reproduce": reproduce, "ckpt": ckpt, "final": final, "offline": offline,
         "latency": latency, "paper": paper}[a.job](a)
    except Exception as e:
        check(f"{a.job} checks ran", False, f"{type(e).__name__}: {e}")
    print(f"[{a.job}] {'ALL CHECKS PASSED' if all(RESULTS) else 'FAILED'}", flush=True)
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
