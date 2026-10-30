"""Direct test of 'weight faults make the network over-confident' on the stored final-model logits.
For every image that the plain network answers correctly when clean and wrongly under the fault (plain SDC),
compare its faulty final-classifier max logit / MSP / -energy with the clean validation distribution."""
import json, sys
import numpy as np, pandas as pd, torch
from scipy.special import logsumexp, softmax

D = "/home1/jtt/early_exit_final/stage2_paper/results/campaigns/final/"
val = torch.load(D + "clean_val.pt", map_location="cpu")["logits"].numpy()[:, 3].astype(np.float64)
te = torch.load(D + "clean_test.pt", map_location="cpu")
cl = te["logits"].numpy()[:, 3].astype(np.float64)
y = te["labels"].numpy()
ok = cl.argmax(1) == y

def scores(z):
    return z.max(1), softmax(z, 1).max(1), logsumexp(z, 1)  # max logit, MSP, -energy

vM, vP, vE = scores(val)
q = lambda a, p: np.percentile(a, p)
ref = dict(M_lo=q(vM, .5), M_hi=q(vM, 99.5), M_med=np.median(vM),
           P_lo=q(vP, 1), P_med=np.median(vP), E_lo=q(vE, 1), E_med=np.median(vE))

rows = []
for loc in ["G1", "G2", "G3"]:
    for k in range(1, 9):
        a = np.load(f"{D}exponent_test_{loc}_k{k}.npy", mmap_mode="r")
        meta = pd.read_csv(f"{D}exponent_test_{loc}_k{k}.csv")
        for r in range(a.shape[0]):
            z = np.asarray(a[r, :, 3], dtype=np.float64)
            sdc = ok & (z.argmax(1) != y)
            if not sdc.any():
                continue
            M, P, E = scores(z[sdc])
            bits = [f["bit"] for f in json.loads(meta.faults[r])]
            rows.append(dict(loc=loc, k=k, n=int(sdc.sum()), bit30=30 in bits,
                             above=(M > ref["M_hi"]).sum(), below=(M < ref["M_lo"]).sum(),
                             P_ge_med=(P >= ref["P_med"]).sum(), P_lo=(P < ref["P_lo"]).sum(),
                             E_ge_med=(E >= ref["E_med"]).sum(), E_lo=(E < ref["E_lo"]).sum(),
                             M_med=np.median(M)))
df = pd.DataFrame(rows)

def summ(d, name):
    n = d.n.sum()
    pct = lambda c: 100 * d[c].sum() / n
    print(f"{name:28s} SDC images {n:>9,d} | maxlogit above clean 99.5th {pct('above'):5.1f}%  below 0.5th {pct('below'):4.1f}%"
          f" | MSP >= clean median {pct('P_ge_med'):5.1f}%  MSP < clean 1st pct {pct('P_lo'):4.1f}%"
          f" | -E >= clean median {pct('E_ge_med'):5.1f}%  -E < clean 1st pct {pct('E_lo'):4.1f}%"
          f" | median maxlogit {np.average(d.M_med, weights=d.n):9.3g}")

print({k: round(v, 3) for k, v in ref.items()}, "clean test median maxlogit", round(np.median(cl.max(1)), 3))
summ(df, "all G1-G3")
summ(df[df.bit30], "runs with a bit-30 flip")
summ(df[~df.bit30], "runs without bit 30")
for loc in ["G1", "G2", "G3"]:
    summ(df[df["loc"] == loc], loc)
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", index=False)
