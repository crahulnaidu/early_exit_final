"""Slack sweep for margin_band on the final model: does a looser exit threshold (more inputs leave early)
help under faults? Offline on stored logits. SDC on each method's clean-correct, unflagged subset S_m."""
import glob, sys, numpy as np, torch
sys.path.insert(0, "/home1/jtt/early_exit_final/stage2_paper")
from utils import rules2

D = "/home1/jtt/early_exit_final/stage2_paper/results/campaigns/final/"
SLACKS = [0.5, 1.0, 1.5, 2.0, 3.0]
va = torch.load(D + "clean_val.pt", map_location="cpu")
te = torch.load(D + "clean_test.pt", map_location="cpu")
split = lambda L: [torch.as_tensor(np.array(L[:, i])) for i in range(L.shape[1])]
yv, y = va["labels"], te["labels"]
cals = {s: rules2.calibrate("margin_band", split(va["logits"].numpy()), yv, slack=s) for s in SLACKS}
clean = {}
for s, c in cals.items():
    e, p, m = rules2.route(c, split(te["logits"].numpy()))
    ok = p.eq(y)
    clean[s] = ok
    print(f"slack {s}: t={[round(x['t'], 2) for x in c['exits']]} | clean acc {100*ok.float().mean():.2f} "
          f"FAR {100*(p < 0).float().mean():.2f} | exit share 1/2/3/final "
          + "/".join(f"{100*(e == k).float().mean():.1f}" for k in range(4)))

def run_eval(files):
    acc = {s: dict(sdc=[], fb=[], acc=[]) for s in SLACKS}
    for f in files:
        a = np.load(f, mmap_mode="r")
        for r in range(a.shape[0]):
            L = split(a[r])
            for s, c in cals.items():
                e, p, m = rules2.route(c, L)
                S = clean[s]
                acc[s]["sdc"].append(100 * ((p[S] >= 0) & (p[S] != y[S])).float().mean().item())
                acc[s]["acc"].append(100 * p.eq(y).float().mean().item())
                acc[s]["fb"].append(100 * m.eq(y).float().mean().item())
    return {s: {k: np.mean(v) for k, v in d.items()} for s, d in acc.items()}

groups = [("exp " + g, sorted(glob.glob(f"{D}exponent_test_{g}_k*.npy"))) for g in ("G1", "G2", "G3", "HEADS")]
groups += [("sign " + g, sorted(glob.glob(f"{D}mendnet_test_{g}_fr*.npy"))) for g in ("FP1", "FP2", "FP3", "FP4")]
print("group        | " + " | ".join(f"slack {s}: SDC / acc / acc+fb" for s in SLACKS))
for name, files in groups:
    r = run_eval(files)
    print(f"{name:12s} | " + " | ".join(f"{r[s]['sdc']:6.3f} / {r[s]['acc']:5.1f} / {r[s]['fb']:5.1f}" for s in SLACKS),
          flush=True)
